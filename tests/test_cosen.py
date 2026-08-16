"""COSEn sign-convention test — this bug is silent if it ships.

Higher (less negative) COSEn = MORE irregular. Three synthetic 30-beat
series pin both the ordering and the approximate values from Lake & Moorman
(2011): a metronome around −4.8, healthy sinus around −1.8, and AF-like
chaos around +0.2.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.screening.cosen import cosen


def _metronome(rng: np.random.Generator) -> np.ndarray:
    return 800.0 + rng.normal(0.0, 2.0, 30)


def _healthy_sinus(rng: np.random.Generator) -> np.ndarray:
    return 800.0 + rng.normal(0.0, 30.0, 30)


def _af_like(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(450.0, 1250.0, 30)


class TestSignConvention:
    """The one that keeps the sign honest."""

    def test_ordering_metronome_sinus_af(self) -> None:
        rng = np.random.default_rng(2011)
        c_metro = cosen(_metronome(rng))
        c_sinus = cosen(_healthy_sinus(rng))
        c_af = cosen(_af_like(rng))
        assert c_metro is not None and c_sinus is not None and c_af is not None
        # Strict ordering: more irregular ⇒ higher (less negative).
        assert c_metro < c_sinus < c_af

    def test_approximate_values(self) -> None:
        # Averaged over seeds to damp 30-beat sampling variance; the values
        # themselves come from the Lake & Moorman ranges.
        metros, sinuses, afs = [], [], []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            metros.append(cosen(_metronome(rng)))
            sinuses.append(cosen(_healthy_sinus(rng)))
            afs.append(cosen(_af_like(rng)))
        assert np.mean([c for c in metros if c is not None]) == pytest.approx(-4.8, abs=0.5)
        assert np.mean([c for c in sinuses if c is not None]) == pytest.approx(-1.8, abs=0.5)
        assert np.mean([c for c in afs if c is not None]) == pytest.approx(0.2, abs=0.5)

    def test_more_variability_means_higher_cosen(self) -> None:
        rng = np.random.default_rng(7)
        base = rng.normal(0.0, 1.0, 30)
        series = [cosen(800.0 + sigma * base) for sigma in (5.0, 20.0, 80.0)]
        assert all(c is not None for c in series)
        assert series[0] < series[1] < series[2]


class TestDegenerateInputs:
    def test_perfect_metronome_is_none(self) -> None:
        # Zero spread: entropy unestimable; treated as maximally regular,
        # never as irregular.
        assert cosen(np.full(30, 800.0)) is None

    def test_too_short_is_none(self) -> None:
        assert cosen(np.array([800.0, 810.0])) is None

    def test_normalised_for_heart_rate(self) -> None:
        # The −ln(mean RR) term: the same *relative* variability at a faster
        # rate should not read as dramatically different irregularity.
        rng = np.random.default_rng(3)
        base = rng.normal(0.0, 1.0, 30)
        slow = cosen(1000.0 + 40.0 * base)  # CV = 4 %
        fast = cosen(500.0 + 20.0 * base)  # CV = 4 %
        assert slow is not None and fast is not None
        assert abs(slow - fast) < 0.35
