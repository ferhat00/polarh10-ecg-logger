"""Walking — HR trends and settling; frequency metrics get a caution."""

from __future__ import annotations

from typing import Any

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
class WalkingProfile(ActivityProfile):
    key = "walking"
    display_name = "Walking"
    description = (
        "Mild HR elevation, moderate motion artifact, settling trends. Emphasises "
        "HR trend and settling rate, per-minute stability, and motion-vs-HR coupling."
    )
    expected_hr_range = (70, 125)
    expected_motion = "moderate"
    comparison_key = "walking"
    static_suppressions = (
        Suppression(
            family=MetricFamily.FREQUENCY,
            mode="caution",
            reason=(
                "Movement rhythms (step cadence and its harmonics) alias into the "
                "LF/HF bands during walking — read frequency-domain values as "
                "contaminated estimates, not autonomic measurements."
            ),
        ),
    )
    interpretation_notes = (
        "Walking HRV reflects motion and effort as much as autonomic state; the "
        "informative signals here are the HR trend and how quickly it settles.",
    )

    def hr_limits(self, ctx: PersonContext) -> HRLimits:
        return HRLimits(
            tachy_bpm=135.0,
            brady_bpm=50.0,
            context="walking",
            brady_note=(
                "Sustained low HR during an activity labelled walking may simply mean "
                "the session was mislabelled or mostly standing still."
            ),
        )

    def derived_metrics(self, inputs: ActivityInputs, ctx: PersonContext) -> dict[str, Any]:
        starts, means = m.per_minute_hr(inputs.rr)
        settling = m.hr_settling(inputs.rr)
        out: dict[str, Any] = {
            "hr_trend_bpm_per_min": m.hr_trend_bpm_per_min(inputs.rr),
            "per_minute_hr_bpm": means,
            "per_minute_hr_t_s": starts,
            "motion_hr_coupling_r": m.motion_hr_coupling(inputs.rr, inputs.quality),
        }
        if settling is not None:
            out["settled_at_min"] = settling[0]
            out["settled_hr_bpm"] = settling[1]
        return out
