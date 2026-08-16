"""Lying down / supine — the gold-standard baseline."""

from __future__ import annotations

from typing import Any

from app.activities import metrics as m
from app.activities.base import ActivityInputs, ActivityProfile, PersonContext
from app.activities.registry import register


@register
class SupineProfile(ActivityProfile):
    key = "supine"
    display_name = "Lying down (supine)"
    description = (
        "Lowest HR, highest RMSSD and HF power, least motion. The reference "
        "session type for all longitudinal comparison, and the best rhythm-"
        "screening quality this strap can produce."
    )
    expected_hr_range = (40, 80)
    expected_motion = "minimal"
    is_reference_baseline = True
    comparison_key = "supine"
    interpretation_notes = (
        "Supine recordings are the reference baseline: compare supine to supine. "
        "Posture dominates HRV, so these values are not comparable to any upright "
        "or moving session.",
    )

    def derived_metrics(self, inputs: ActivityInputs, ctx: PersonContext) -> dict[str, Any]:
        starts, values = m.per_minute_rmssd(inputs.rr)
        return {
            "per_minute_rmssd_ms": values,
            "per_minute_rmssd_t_s": starts,
            "hr_trend_bpm_per_min": m.hr_trend_bpm_per_min(inputs.rr),
        }
