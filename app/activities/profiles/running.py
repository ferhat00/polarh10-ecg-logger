"""Running — HR zones, drift, recovery; HRV suppressed at intensity."""

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

#: HRV is suppressed entirely above this heart-rate-reserve fraction: at high
#: intensity, vagal tone is withdrawn and RR variability is dominated by
#: mechanics and noise, not autonomic state (Task Force 1996; Michael et al.,
#: Front Physiol 2017;8:301 on HRV during exercise).
HRV_SUPPRESSION_HRR_FRACTION = 0.70


class IntensityHRVSuppressionMixin:
    """Shared by running/cycling: suppress the HRV suite at high intensity."""

    def dynamic_suppressions(
        self, inputs: ActivityInputs, ctx: PersonContext
    ) -> list[Suppression]:
        mean_hr = inputs.hrv.mean_hr_bpm
        if mean_hr is None:
            return []
        frac = ctx.hr_reserve_fraction(mean_hr)
        if frac <= HRV_SUPPRESSION_HRR_FRACTION:
            return []
        max_hr, max_src = ctx.effective_max_hr()
        reason = (
            f"Session mean HR {mean_hr:.0f} bpm is {frac * 100:.0f}% of heart-rate "
            f"reserve (max HR {max_hr:.0f}, {max_src}) — above the "
            f"{HRV_SUPPRESSION_HRR_FRACTION * 100:.0f}% level where HRV stops being "
            "meaningful. The HRV suite is suppressed for this session; HR-based "
            "metrics (zones, drift, recovery) remain."
        )
        return [
            Suppression(family=f, mode="suppress", reason=reason)
            for f in MetricFamily.HRV_ALL
        ]


@register
class RunningProfile(IntensityHRVSuppressionMixin, ActivityProfile):
    key = "running"
    display_name = "Running"
    description = (
        "High HR, heavy motion artifact, cardiac drift. Emphasises HR zones, "
        "drift, and post-exercise recovery (HRR60)."
    )
    expected_hr_range = (110, 195)
    expected_motion = "high"
    comparison_key = "running"
    static_suppressions = (
        Suppression(
            family=MetricFamily.FREQUENCY,
            mode="caution",
            reason=(
                "Stride cadence and its harmonics alias into the LF/HF bands while "
                "running; any surviving frequency values are motion-contaminated."
            ),
        ),
    )
    interpretation_notes = (
        "Expect and take seriously a significant excluded fraction — heavy motion "
        "is normal for running and the excluded time is reported, not interpolated.",
    )

    def hr_limits(self, ctx: PersonContext) -> HRLimits:
        max_hr, src = ctx.effective_max_hr()
        return HRLimits(
            tachy_bpm=max_hr,
            brady_bpm=50.0,
            context="running",
            brady_note=(
                "Sustained low HR during an activity labelled running usually means "
                "the session was mislabelled."
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
