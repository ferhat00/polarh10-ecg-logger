"""Epoch-by-epoch agreement between staging engines.

When two engines score the same night, their agreement is a quality signal
the user should see: high agreement adds confidence, low agreement means the
night's numbers should be read as a range, not a value. Comparison happens
in the coarser of the two vocabularies (detail is merged, never invented),
over epochs both engines actually scored.

Cohen's kappa is implemented here directly (a dozen lines) rather than
adding scikit-learn as a dependency for one formula. Cohen 1960, Educ
Psychol Meas 20(1):37-46, doi:10.1177/001316446002000104.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from app.sleep.stages import UNSCORED, Hypnogram, StageVocab, common_vocab

#: Below this many mutually scored epochs, kappa is noise — report nothing.
MIN_COMPARABLE_EPOCHS = 20


@dataclass
class AgreementRow:
    """Agreement between one pair of engines."""

    engine_a: str
    engine_b: str
    #: The vocabulary the comparison was made in (the coarser of the two).
    vocab: StageVocab
    kappa: float | None
    percent_agree: float | None
    n_epochs: int


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float | None:
    """Cohen's kappa for two equal-length integer label arrays.

    Returns None for degenerate marginals (chance agreement = 1, i.e. both
    raters constant), where kappa is undefined.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) != len(b) or len(a) == 0:
        return None
    po = float(np.mean(a == b))
    labels = np.union1d(a, b)
    pe = float(
        sum(np.mean(a == lab) * np.mean(b == lab) for lab in labels)
    )
    if pe >= 1.0:
        return None
    return (po - pe) / (1.0 - pe)


def pairwise_agreement(hyps: list[Hypnogram]) -> list[AgreementRow]:
    """Agreement rows for every pair of hypnograms on the same grid."""
    rows: list[AgreementRow] = []
    for ha, hb in combinations(hyps, 2):
        if ha.n_epochs != hb.n_epochs:
            continue  # different grids are not comparable
        vocab = common_vocab(ha.vocab, hb.vocab)
        sa = ha.collapsed(vocab).stages
        sb = hb.collapsed(vocab).stages
        both = (sa != UNSCORED) & (sb != UNSCORED)
        n = int(np.sum(both))
        if n < MIN_COMPARABLE_EPOCHS:
            rows.append(
                AgreementRow(ha.engine, hb.engine, vocab, None, None, n)
            )
            continue
        rows.append(
            AgreementRow(
                engine_a=ha.engine,
                engine_b=hb.engine,
                vocab=vocab,
                kappa=cohen_kappa(sa[both], sb[both]),
                percent_agree=float(np.mean(sa[both] == sb[both]) * 100.0),
                n_epochs=n,
            )
        )
    return rows
