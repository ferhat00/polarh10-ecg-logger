"""Swimming — frequent electrode dropout; honesty over degraded numbers."""

from __future__ import annotations

from typing import Any

from app.activities import metrics as m
from app.activities.base import ActivityInputs, ActivityProfile, PersonContext
from app.activities.registry import register
from app.screening.thresholds import HRLimits


@register
class SwimmingProfile(ActivityProfile):
    key = "swimming"
    display_name = "Swimming"
    description = (
        "Water disrupts electrode contact and the H10 buffers internally, so "
        "large excluded fractions are normal. Whatever survives quality gating "
        "is reported; above 30% excluded the session is declared not analysable."
    )
    expected_hr_range = (90, 175)
    expected_motion = "severe"
    comparison_key = "swimming"
    #: Above this excluded fraction, report plainly that the session is not
    #: analysable rather than presenting degraded numbers.
    max_excluded_fraction = 0.30
    interpretation_notes = (
        "Treat every swimming metric as best-effort: electrode dropout is the "
        "rule in water, and the excluded-time figure is the first number to read.",
    )

    def hr_limits(self, ctx: PersonContext) -> HRLimits:
        max_hr, _ = ctx.effective_max_hr()
        return HRLimits(
            tachy_bpm=max_hr,
            brady_bpm=50.0,
            context="swimming",
            brady_note=(
                "Apparent sustained low HR while swimming is more often dropout "
                "than physiology."
            ),
        )

    def derived_metrics(self, inputs: ActivityInputs, ctx: PersonContext) -> dict[str, Any]:
        max_hr, max_src = ctx.effective_max_hr()
        return {
            "hr_zones_fraction": m.hr_zones(inputs.rr, max_hr),
            "max_hr_basis": max_src,
            "excluded_fraction": round(inputs.excluded_fraction, 3),
        }
