"""Stage vocabularies, collapse tables, and the Hypnogram container.

Every engine scores the same canonical grid: 30 s epochs starting at the
ECG recording's first sample (epoch *k* covers ``[k*30, (k+1)*30)`` seconds
on the recording clock). Engines that cannot score an epoch mark it
:data:`UNSCORED` rather than guessing.

Different engines speak different vocabularies (2-, 3-, 4- and 5-class).
Conversion is only ever *coarsening* — mapping a finer vocabulary onto a
coarser one via the explicit tables below. Inventing detail (e.g. splitting
NREM into light/deep) is a :class:`ValueError`, never an interpolation.

The 4-class Wake/Light/Deep/REM vocabulary is the canonical display
convention, adopted from Sleep² (Topalidis et al. 2023, Sensors 23(5):2390,
doi:10.3390/s23052390) — the H10-validated reference this feature measures
itself against. Code 0 is WAKE in every vocabulary, so ``stages >= 1`` is
always "asleep"; this invariant is asserted by the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

#: Epoch length shared by every engine (the AASM scoring epoch).
EPOCH_LEN_S = 30.0

#: Stage code for an epoch an engine could not score. Passes through every
#: collapse unchanged.
UNSCORED = -1


class StageVocab(StrEnum):
    """A stage vocabulary: which integer codes mean which stage."""

    #: 0=WAKE 1=SLEEP (sleepecg ``ws-gru-mesa``).
    WAKE_SLEEP = "wake_sleep_2"
    #: 0=WAKE 1=NREM 2=REM (sleepecg ``wrn-gru-mesa``).
    WAKE_REM_NREM = "wake_rem_nrem_3"
    #: 0=WAKE 1=LIGHT(N1+N2) 2=DEEP(N3) 3=REM — canonical display vocabulary.
    WAKE_LIGHT_DEEP_REM = "wldr_4"
    #: 0=WAKE 1=N1 2=N2 3=N3 4=REM (AASM five-class).
    AASM_5 = "aasm_5"


#: Stage labels per vocabulary, indexed by stage code.
_LABELS: dict[StageVocab, tuple[str, ...]] = {
    StageVocab.WAKE_SLEEP: ("Wake", "Sleep"),
    StageVocab.WAKE_REM_NREM: ("Wake", "NREM", "REM"),
    StageVocab.WAKE_LIGHT_DEEP_REM: ("Wake", "Light", "Deep", "REM"),
    StageVocab.AASM_5: ("Wake", "N1", "N2", "N3", "REM"),
}

#: Granularity rank — collapse may only move to a strictly-lower-or-equal rank.
_RANK: dict[StageVocab, int] = {
    StageVocab.WAKE_SLEEP: 0,
    StageVocab.WAKE_REM_NREM: 1,
    StageVocab.WAKE_LIGHT_DEEP_REM: 2,
    StageVocab.AASM_5: 3,
}

#: Explicit collapse tables, ``(src, dst) -> {src_code: dst_code}``.
#: Compositions are spelled out rather than chained so each table is
#: independently testable against the literature definitions.
_COLLAPSE: dict[tuple[StageVocab, StageVocab], dict[int, int]] = {
    (StageVocab.AASM_5, StageVocab.WAKE_LIGHT_DEEP_REM): {0: 0, 1: 1, 2: 1, 3: 2, 4: 3},
    (StageVocab.AASM_5, StageVocab.WAKE_REM_NREM): {0: 0, 1: 1, 2: 1, 3: 1, 4: 2},
    (StageVocab.AASM_5, StageVocab.WAKE_SLEEP): {0: 0, 1: 1, 2: 1, 3: 1, 4: 1},
    (StageVocab.WAKE_LIGHT_DEEP_REM, StageVocab.WAKE_REM_NREM): {0: 0, 1: 1, 2: 1, 3: 2},
    (StageVocab.WAKE_LIGHT_DEEP_REM, StageVocab.WAKE_SLEEP): {0: 0, 1: 1, 2: 1, 3: 1},
    (StageVocab.WAKE_REM_NREM, StageVocab.WAKE_SLEEP): {0: 0, 1: 1, 2: 1},
}

#: The code that means REM in each vocabulary (None where REM is not a class).
REM_CODE: dict[StageVocab, int | None] = {
    StageVocab.WAKE_SLEEP: None,
    StageVocab.WAKE_REM_NREM: 2,
    StageVocab.WAKE_LIGHT_DEEP_REM: 3,
    StageVocab.AASM_5: 4,
}

#: WAKE is code 0 in every vocabulary (invariant relied on throughout).
WAKE_CODE = 0


def stage_labels(vocab: StageVocab) -> list[str]:
    """Human-readable labels for a vocabulary, indexed by stage code."""
    return list(_LABELS[vocab])


def n_classes(vocab: StageVocab) -> int:
    return len(_LABELS[vocab])


def collapse(stages: np.ndarray, src: StageVocab, dst: StageVocab) -> np.ndarray:
    """Map stage codes from a finer vocabulary onto a coarser one.

    ``UNSCORED`` passes through. ``src == dst`` returns a copy. Mapping
    toward a *finer* vocabulary raises — detail is never invented.
    """
    if src == dst:
        return np.array(stages, dtype=np.int8, copy=True)
    if _RANK[dst] > _RANK[src]:
        raise ValueError(
            f"Cannot collapse {src.value} to the finer vocabulary {dst.value}: "
            "stage detail can be merged, never invented."
        )
    table = _COLLAPSE[(src, dst)]
    out = np.full(len(stages), UNSCORED, dtype=np.int8)
    for src_code, dst_code in table.items():
        out[stages == src_code] = dst_code
    return out


def common_vocab(a: StageVocab, b: StageVocab) -> StageVocab:
    """The coarser of two vocabularies (where a comparison must happen)."""
    return a if _RANK[a] <= _RANK[b] else b


def make_epoch_grid(duration_s: float, epoch_len_s: float = EPOCH_LEN_S) -> np.ndarray:
    """Canonical epoch start times covering the recording (last epoch may be
    partial; engines mark epochs they cannot score as UNSCORED)."""
    if duration_s <= 0:
        return np.array([])
    n_epochs = int(np.ceil(duration_s / epoch_len_s))
    return np.arange(n_epochs) * epoch_len_s


def epoch_indices(
    t_s: np.ndarray, n_epochs: int, epoch_len_s: float = EPOCH_LEN_S
) -> np.ndarray:
    """Epoch index for each time point; -1 where outside the grid.

    Vectorised (`floor divide`) — an overnight RR series is ~35k points and
    per-epoch boolean scans over it are the known performance trap.
    """
    idx = np.floor(np.asarray(t_s) / epoch_len_s).astype(np.int64)
    idx[(idx < 0) | (idx >= n_epochs)] = -1
    return idx


@dataclass
class Hypnogram:
    """One engine's staging of one night on the canonical epoch grid."""

    #: Stable engine key ("heuristic" | "sleepecg" | "external-5class").
    engine: str
    #: Human-readable engine name for the report.
    engine_label: str
    vocab: StageVocab
    epoch_len_s: float
    #: Epoch start times (s, recording clock) — the canonical grid.
    epoch_start_s: np.ndarray
    #: Stage code per epoch (int8, codes per ``vocab``, UNSCORED = -1).
    stages: np.ndarray
    #: Per-epoch class probabilities ``(n_epochs, n_classes)`` when the
    #: engine provides them; None otherwise.
    probabilities: np.ndarray | None
    #: Honest accuracy statement with citation — rendered next to the
    #: hypnogram in the report, never separable from it.
    accuracy_note: str
    notes: list[str] = field(default_factory=list)

    @property
    def n_epochs(self) -> int:
        return len(self.stages)

    def scored_mask(self) -> np.ndarray:
        return self.stages != UNSCORED

    def sleep_mask(self) -> np.ndarray:
        """True where the epoch is scored as any sleep stage."""
        return self.stages >= 1

    def collapsed(self, dst: StageVocab) -> Hypnogram:
        """A copy of this hypnogram in a coarser vocabulary (no probabilities:
        class probabilities do not survive a class merge unambiguously)."""
        return Hypnogram(
            engine=self.engine,
            engine_label=self.engine_label,
            vocab=dst,
            epoch_len_s=self.epoch_len_s,
            epoch_start_s=self.epoch_start_s,
            stages=collapse(self.stages, self.vocab, dst),
            probabilities=None,
            accuracy_note=self.accuracy_note,
            notes=list(self.notes),
        )
