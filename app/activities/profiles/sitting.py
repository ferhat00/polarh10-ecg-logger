"""Sitting — a good secondary baseline."""

from __future__ import annotations

from typing import Any

from app.activities import metrics as m
from app.activities.base import ActivityInputs, ActivityProfile, PersonContext
from app.activities.registry import register


@register
class SittingProfile(ActivityProfile):
    key = "sitting"
    display_name = "Sitting"
    description = "Low HR, moderate HRV. Full HRV suite is valid; a good secondary baseline."
    expected_hr_range = (50, 90)
    expected_motion = "minimal"
    comparison_key = "sitting"
    interpretation_notes = (
        "Sitting is a usable secondary baseline, but sitting values are not "
        "interchangeable with supine ones — posture alone shifts RMSSD and HF.",
    )

    def derived_metrics(self, inputs: ActivityInputs, ctx: PersonContext) -> dict[str, Any]:
        starts, values = m.per_minute_rmssd(inputs.rr)
        return {
            "per_minute_rmssd_ms": values,
            "per_minute_rmssd_t_s": starts,
            "hr_trend_bpm_per_min": m.hr_trend_bpm_per_min(inputs.rr),
        }
