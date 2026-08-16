"""Standing — LF/Mayer dominance is the expected physiology, not a finding."""

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
from app.screening.thresholds import HRLimits, default_limits


@register
class StandingProfile(ActivityProfile):
    key = "standing"
    display_name = "Standing"
    description = (
        "Upright posture: baroreflex (Mayer-wave) LF dominance, suppressed HF, "
        "RMSSD well below supine. Emphasises the LF band and orthostatic character."
    )
    expected_hr_range = (55, 105)
    expected_motion = "low"
    comparison_key = "standing"
    static_suppressions = (
        Suppression(
            family=MetricFamily.TIME,
            mode="caution",
            reason=(
                "Low RMSSD is *expected* when standing — vagal withdrawal is normal "
                "upright physiology, not a pattern worth noting. Compare standing "
                "sessions only with other standing sessions."
            ),
        ),
        Suppression(
            family=MetricFamily.FREQUENCY,
            mode="caution",
            reason=(
                "Suppressed HF and LF dominance are the normal upright pattern. The "
                "~0.1 Hz peak is the baroreflex Mayer wave, not respiration."
            ),
        ),
    )
    interpretation_notes = (
        "A dominant spectral peak near 0.1 Hz here is the Mayer wave — textbook "
        "upright posture, not a respiratory rate.",
    )

    def hr_limits(self, ctx: PersonContext) -> HRLimits:
        base = default_limits(ctx.athlete_baseline)
        # Standing HR runs ~10 bpm above supine; sustained >115 while simply
        # standing is what's worth reviewing (postural tachycardia territory).
        return HRLimits(
            tachy_bpm=115.0,
            brady_bpm=base.brady_bpm,
            context="standing",
            brady_note=base.brady_note,
        )

    def derived_metrics(self, inputs: ActivityInputs, ctx: PersonContext) -> dict[str, Any]:
        out: dict[str, Any] = {
            "mayer_peak_hz": inputs.hrv.lf_peak_hz,
            "hr_trend_bpm_per_min": m.hr_trend_bpm_per_min(inputs.rr),
        }
        ortho = m.orthostatic_response(inputs.rr, inputs.markers)
        if ortho is not None:
            out["orthostatic_response"] = ortho
        return out
