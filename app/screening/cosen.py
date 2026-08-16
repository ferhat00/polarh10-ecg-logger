"""Coefficient of Sample Entropy (COSEn) for RR irregularity screening.

Lake DE, Moorman JR. "Accurate estimation of entropy in very short
physiological time series: the problem of atrial fibrillation detection in
implanted ventricular devices." Am J Physiol Heart Circ Physiol
2011;300:H319-H325.

    COSEn = SampEn(m=1, r) + ln(2r) − ln(mean RR)

The ``+ ln(2r)`` term converts the tolerance-dependent SampEn into a density
estimate (making the value robust to the choice of r); the ``− ln(mean RR)``
term normalises for heart rate.

**Sign convention — pinned by test:** HIGHER (less negative) means MORE
irregular. A metronomic series sits near −4.8, healthy sinus near −1.8, and
AF-like chaos near +0.2. This is easy to invert silently; the unit test in
``tests/test_cosen.py`` exists precisely to keep the sign honest.
"""

from __future__ import annotations

import numpy as np

#: Matching-template length (Lake & Moorman use m=1 for very short windows).
COSEN_M = 1

#: Tolerance as a fraction of the window's RR standard deviation. Lake &
#: Moorman choose r to keep match counts adequate in short windows; 0.2·σ is
#: the conventional choice and reproduces their reported value ranges.
COSEN_R_FRACTION = 0.2


def cosen(rr_ms: np.ndarray) -> float | None:
    """COSEn of an RR window (ms). Returns None where undefined.

    Undefined when: the window is too short, the RR spread is degenerate
    (a perfect metronome has no meaningful entropy estimate — treat as
    maximally regular), or no template matches exist at all.
    """
    x = np.asarray(rr_ms, dtype=np.float64)
    if len(x) < COSEN_M + 2:
        return None
    sd = float(np.std(x, ddof=1))
    mean_rr = float(np.mean(x))
    if sd <= 0.0 or mean_rr <= 0.0:
        return None
    r = COSEN_R_FRACTION * sd

    sampen = _sample_entropy(x, m=COSEN_M, r=r)
    if sampen is None:
        return None
    return sampen + np.log(2.0 * r) - np.log(mean_rr)


def _sample_entropy(x: np.ndarray, m: int, r: float) -> float | None:
    """SampEn(m, r) with the Chebyshev norm, excluding self-matches.

    Returns None when there are no template matches of length m (entropy
    unestimable) and ``inf``-avoidance when there are matches at m but none
    at m+1 (Lake & Moorman handle this with the minimum-count convention; for
    screening we return None and let the caller treat the window as
    inconclusive rather than fabricating a value).
    """
    def match_count(length: int) -> int:
        count = 0
        templates = np.lib.stride_tricks.sliding_window_view(x, length)
        for i in range(len(templates) - 1):
            d = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
            count += int(np.sum(d < r))
        return count

    b = match_count(m)
    a = match_count(m + 1)
    if b == 0 or a == 0:
        return None
    return float(-np.log(a / b))
