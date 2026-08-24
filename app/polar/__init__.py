"""Polar Flow context: opt-in AccessLink sync and shared presentation.

The client (``app.polar.client``) talks to Polar **only** when the wearer has
explicitly opted in (``ECGLOG_POLAR_ENABLED=1`` plus AccessLink client
credentials) and only for an account they have linked themselves — with the
flag unset the routes are not registered and no request is ever made.

What this data is, and what it is not
-------------------------------------
Flow returns figures Polar has already derived on the wrist: sleep stages,
a sleep score, Nightly Recharge, and an overnight RMSSD. They are useful as
*context* — they cover every night, including the great majority with no ECG
session behind them — and they are the objective counterpart to the
subjective ``sleep_quality_1_5`` field and the ``poor-sleep`` tag.

They are not interchangeable with anything this app measures. Flow's RMSSD is
PPG-derived, from a wrist band, averaged over four hours of sleep; a session's
RMSSD is ECG-derived from R-peaks at ~130 Hz over minutes of controlled
posture. Same unit, different measurement. They are displayed side by side and
never merged into one series, and nothing here reaches ``app.screening``:
Polar's scores are proprietary composites, and a screening flag carries a
medical disclaimer that only auditable thresholds have any business behind.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol


class _HasNightColumns(Protocol):  # pragma: no cover - typing aid only
    date: dt.date
    sleep_score: int | None
    hrv_rmssd_ms: int | None
    hr_avg_bpm: int | None
    nightly_recharge_status: int | None
    ans_charge: float | None
    breathing_rate_avg: float | None
    total_sleep_s: int | None


#: Polar's 1-6 Nightly Recharge scale, spelled out.
RECHARGE_STATUS: dict[int, str] = {
    1: "very poor",
    2: "poor",
    3: "compromised",
    4: "OK",
    5: "good",
    6: "very good",
}

#: Polar's 1-5 ANS charge scale, relative to the wearer's own 28-day norm.
ANS_STATUS: dict[int, str] = {
    1: "much below usual",
    2: "below usual",
    3: "usual",
    4: "above usual",
    5: "much above usual",
}


def format_duration(seconds: int | None) -> str | None:
    """``27000`` -> ``"7 h 30 m"``. None stays None."""
    if seconds is None or seconds < 0:
        return None
    hours, remainder = divmod(int(seconds), 3600)
    return f"{hours} h {remainder // 60:02d} m"


def format_night_line(night: _HasNightColumns | None) -> str | None:
    """One human-readable line from a night's columns; None when there is none.

    Partial data is normal — Polar computes sleep and Nightly Recharge from
    different windows and either can be missing — so only present values are
    printed.
    """
    if night is None:
        return None
    bits: list[str] = []
    duration = format_duration(night.total_sleep_s)
    if duration:
        bits.append(f"slept {duration}")
    if night.sleep_score is not None:
        bits.append(f"sleep score {night.sleep_score}")
    if night.nightly_recharge_status is not None:
        label = RECHARGE_STATUS.get(night.nightly_recharge_status, "unknown")
        bits.append(f"recharge {label}")
    if night.hrv_rmssd_ms is not None:
        bits.append(f"overnight RMSSD {night.hrv_rmssd_ms} ms")
    if night.hr_avg_bpm is not None:
        bits.append(f"sleeping HR {night.hr_avg_bpm} bpm")
    if not bits:
        return None
    return " · ".join(bits) + " (Polar Flow, wrist PPG)"


def night_before(person, moment: dt.datetime | None):
    """The Flow night whose result date matches the day of ``moment``.

    Polar dates a night by the morning it ends, so a session recorded on the
    12th is preceded by the night dated the 12th. Returns None when the person
    has no linked account, no data for that day, or no recording timestamp —
    all normal states.
    """
    if moment is None:
        return None
    target = moment.date()
    for night in person.flow_nights:
        if night.date == target:
            return night
    return None
