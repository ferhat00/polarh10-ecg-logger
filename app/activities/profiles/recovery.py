"""Recovery / post-exercise — falling HR, HRV still normalising."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.activities import metrics as m
from app.activities.base import (
    ActivityInputs,
    ActivityProfile,
    MetricFamily,
    PersonContext,
    Suppression,
)
from app.activities.registry import register
from app.screening.thresholds import HRLimits


@register
class RecoveryProfile(ActivityProfile):
    key = "recovery"
    display_name = "Recovery (post-exercise)"
    description = (
        "Falling HR after effort. Emphasises HRR60/HRR120, the recovery time "
        "constant, and RMSSD reactivation."
    )
    expected_hr_range = (60, 150)
    expected_motion = "low"
    comparison_key = "recovery"
    static_suppressions = (
        Suppression(
            family=MetricFamily.TIME,
            mode="caution",
            reason=(
                "HRV is still normalising after exercise — parasympathetic "
                "reactivation takes minutes to hours, so absolute values here are "
                "not comparable to a rested baseline."
            ),
        ),
        Suppression(
            family=MetricFamily.FREQUENCY,
            mode="caution",
            reason=(
                "Post-exercise spectra are non-stationary by definition; band powers "
                "over a falling HR mix recovery dynamics into every band."
            ),
        ),
    )
    interpretation_notes = (
        "Compare recovery sessions with other recovery sessions taken at a similar "
        "delay after similar effort — never with rested baselines.",
    )

    def hr_limits(self, ctx: PersonContext) -> HRLimits:
        return HRLimits(
            tachy_bpm=130.0,
            brady_bpm=45.0,
            context="post-exercise recovery",
            brady_note=(
                "Late-recovery HR can legitimately undershoot resting values in "
                "trained people; this is a low-specificity observation."
            ),
        )

    def derived_metrics(self, inputs: ActivityInputs, ctx: PersonContext) -> dict[str, Any]:
        starts, rmssd = m.per_minute_rmssd(inputs.rr)
        out: dict[str, Any] = {
            "hrr60_bpm": m.hr_recovery(inputs.rr, 60.0),
            "hrr120_bpm": m.hr_recovery(inputs.rr, 120.0),
            "recovery_tau_s": m.recovery_time_constant(inputs.rr),
            "per_minute_rmssd_ms": rmssd,
            "per_minute_rmssd_t_s": starts,
        }
        if len(rmssd) >= 3:
            slope = float(np.polyfit(np.array(starts) / 60.0, rmssd, 1)[0])
            out["rmssd_reactivation_ms_per_min"] = slope
        return out
