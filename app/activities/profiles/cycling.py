"""Cycling — as running, plus pedalling-cadence aliasing."""

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
from app.activities.profiles.running import IntensityHRVSuppressionMixin
from app.activities.registry import register
from app.screening.thresholds import HRLimits


@register
class CyclingProfile(IntensityHRVSuppressionMixin, ActivityProfile):
    key = "cycling"
    display_name = "Cycling"
    description = (
        "High HR and motion with the added twist of rhythmic pedalling cadence. "
        "Emphasises HR zones, drift, and HRR60."
    )
    expected_hr_range = (100, 190)
    expected_motion = "high"
    comparison_key = "cycling"
    static_suppressions = (
        Suppression(
            family=MetricFamily.FREQUENCY,
            mode="caution",
            reason=(
                "Pedalling cadence (~1-1.7 Hz and subharmonics) can alias directly "
                "into the HRV bands — a spectral peak here may be the cadence, not "
                "autonomic rhythm."
            ),
        ),
    )
    interpretation_notes = (
        "Torso stability makes cycling signal cleaner than running, but cadence "
        "aliasing means frequency-domain values still deserve suspicion.",
    )

    def hr_limits(self, ctx: PersonContext) -> HRLimits:
        max_hr, _ = ctx.effective_max_hr()
        return HRLimits(
            tachy_bpm=max_hr,
            brady_bpm=50.0,
            context="cycling",
            brady_note=(
                "Sustained low HR during an activity labelled cycling usually means "
                "coasting or a mislabelled session."
            ),
        )

    def derived_metrics(self, inputs: ActivityInputs, ctx: PersonContext) -> dict[str, Any]:
        max_hr, max_src = ctx.effective_max_hr()
        peak = m.rolling_peak_hr(inputs.rr)
        out: dict[str, Any] = {
            "hr_zones_fraction": m.hr_zones(inputs.rr, max_hr),
            "max_hr_basis": max_src,
            "cardiac_drift_proxy_bpm_per_min": m.cardiac_drift_bpm_per_min(inputs.rr),
            "hrr60_bpm": m.hr_recovery(inputs.rr, 60.0),
        }
        if peak is not None:
            out["peak_hr_bpm"] = peak[1]
            out["peak_hr_t_s"] = peak[0]
        return out
