"""Cohen's kappa and cross-engine agreement, pinned to hand-computed values."""

from __future__ import annotations

import numpy as np

from app.sleep.agreement import cohen_kappa, pairwise_agreement
from app.sleep.stages import EPOCH_LEN_S, UNSCORED, Hypnogram, StageVocab


def _hyp(stages, engine="a", vocab=StageVocab.WAKE_LIGHT_DEEP_REM) -> Hypnogram:
    stages = np.asarray(stages, dtype=np.int8)
    return Hypnogram(
        engine=engine,
        engine_label=engine,
        vocab=vocab,
        epoch_len_s=EPOCH_LEN_S,
        epoch_start_s=np.arange(len(stages)) * EPOCH_LEN_S,
        stages=stages,
        probabilities=None,
        accuracy_note="test",
    )


class TestCohenKappa:
    def test_perfect_agreement(self) -> None:
        a = np.array([0, 1, 2, 0, 1, 2])
        assert cohen_kappa(a, a.copy()) == 1.0

    def test_hand_computed_confusion(self) -> None:
        # 2x2 example: po = 0.7, marginals a: 0.5/0.5, b: 0.6/0.4
        # pe = 0.5*0.6 + 0.5*0.4 = 0.5 ; kappa = (0.7-0.5)/0.5 = 0.4
        a = np.array([0] * 5 + [1] * 5)
        b = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])
        assert abs(cohen_kappa(a, b) - 0.4) < 1e-12

    def test_degenerate_marginals_return_none(self) -> None:
        a = np.zeros(10, dtype=int)
        assert cohen_kappa(a, a.copy()) is None

    def test_independent_labels_near_zero(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.integers(0, 4, 5000)
        b = rng.integers(0, 4, 5000)
        assert abs(cohen_kappa(a, b)) < 0.05

    def test_length_mismatch_none(self) -> None:
        assert cohen_kappa(np.array([0, 1]), np.array([0])) is None


class TestPairwiseAgreement:
    def test_cross_vocab_pair_collapses_to_coarser(self) -> None:
        n = 30
        # 4-class engine says Light everywhere except two wake epochs;
        # 3-class engine agrees once collapsed (Light -> NREM).
        four = _hyp([1] * (n - 2) + [0, 0], engine="four")
        three = _hyp(
            [1] * (n - 2) + [0, 0], engine="three", vocab=StageVocab.WAKE_REM_NREM
        )
        rows = pairwise_agreement([four, three])
        assert len(rows) == 1
        row = rows[0]
        assert row.vocab == StageVocab.WAKE_REM_NREM
        assert row.percent_agree == 100.0
        assert row.kappa == 1.0
        assert row.n_epochs == n

    def test_unscored_epochs_dropped_from_comparison(self) -> None:
        n = 40
        a = [1] * n
        b = [1] * n
        a[0] = UNSCORED
        b[1] = UNSCORED
        b[2] = 0  # one real disagreement
        rows = pairwise_agreement([_hyp(a, "a"), _hyp(b, "b")])
        assert rows[0].n_epochs == n - 2
        assert abs(rows[0].percent_agree - (n - 3) / (n - 2) * 100.0) < 1e-9

    def test_too_few_comparable_epochs_reports_no_kappa(self) -> None:
        rows = pairwise_agreement([_hyp([1] * 5, "a"), _hyp([1] * 5, "b")])
        assert rows[0].kappa is None
        assert rows[0].percent_agree is None
        assert rows[0].n_epochs == 5

    def test_mismatched_grids_skipped(self) -> None:
        assert pairwise_agreement([_hyp([1] * 30, "a"), _hyp([1] * 29, "b")]) == []

    def test_three_engines_give_three_pairs(self) -> None:
        hyps = [_hyp([1] * 30, e) for e in ("a", "b", "c")]
        rows = pairwise_agreement(hyps)
        assert {(r.engine_a, r.engine_b) for r in rows} == {
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
        }
