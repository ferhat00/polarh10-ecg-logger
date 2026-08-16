"""Activity-profile base interface.

Each profile declares its expected HR range, expected motion level, which
metric families are valid or suppressed (with the reason), activity-specific
derived metrics, and the comparison key it belongs to. Suppression is
enforced by :func:`apply_suppressions`, which blanks suppressed values before
storage so nothing downstream can accidentally use them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from app.pipeline.hrv import HRVResult
from app.pipeline.process import PipelineResult
from app.pipeline.quality import QualityResult
from app.pipeline.rr import RRSeries
from app.screening.thresholds import HRLimits, default_limits


class MetricFamily:
    """Families a profile can suppress or caution."""

    #: Beat-to-beat variability statistics (SDNN, RMSSD, pNN50, per-window).
    TIME = "time_domain"
    FREQUENCY = "frequency_domain"
    NONLINEAR = "nonlinear"
    #: All three HRV families ("the HRV suite").
    HRV_ALL = (TIME, FREQUENCY, NONLINEAR)


@dataclass(frozen=True)
class Suppression:
    """A family that is suppressed (values blanked) or cautioned (badged)."""

    family: str
    mode: str  # "suppress" | "caution"
    reason: str


@dataclass
class PersonContext:
    """Person-level parameters the profiles need, with honest fallbacks."""

    athlete_baseline: bool = False
    age_years: float | None = None
    max_hr_bpm: int | None = None
    resting_hr_bpm: int | None = None

    #: Fallback when neither a measured max HR nor an age is available.
    DEFAULT_MAX_HR = 185.0
    DEFAULT_RESTING_HR = 60.0

    def effective_max_hr(self) -> tuple[float, str]:
        """(max HR, provenance note). Measured → Tanaka estimate → default.

        Tanaka, Monahan & Seals, "Age-predicted maximal heart rate
        revisited," JACC 2001;37:153-156: HRmax ≈ 208 − 0.7·age.
        """
        if self.max_hr_bpm:
            return float(self.max_hr_bpm), "measured max HR"
        if self.age_years:
            return 208.0 - 0.7 * self.age_years, "estimated from age (Tanaka 208−0.7·age)"
        return self.DEFAULT_MAX_HR, (
            f"default {self.DEFAULT_MAX_HR:.0f} bpm — no measured max HR or date of "
            "birth on file; heart-rate-reserve figures are approximate"
        )

    def effective_resting_hr(self) -> tuple[float, str]:
        if self.resting_hr_bpm:
            return float(self.resting_hr_bpm), "measured resting HR"
        return self.DEFAULT_RESTING_HR, (
            f"default {self.DEFAULT_RESTING_HR:.0f} bpm — no measured resting HR on file"
        )

    def hr_reserve_fraction(self, hr_bpm: float) -> float:
        """Karvonen fraction: (HR − rest) / (max − rest), clamped to ≥ 0."""
        max_hr, _ = self.effective_max_hr()
        rest_hr, _ = self.effective_resting_hr()
        denom = max(1.0, max_hr - rest_hr)
        return max(0.0, (hr_bpm - rest_hr) / denom)

    @classmethod
    def from_person(cls, person: Any) -> PersonContext:
        """Build from a Person model row (duck-typed; no model import)."""
        age = None
        if getattr(person, "date_of_birth", None) is not None:
            import datetime as dt

            age = (dt.date.today() - person.date_of_birth).days / 365.25
        return cls(
            athlete_baseline=bool(getattr(person, "athlete_baseline", False)),
            age_years=age,
            max_hr_bpm=getattr(person, "max_hr_bpm", None),
            resting_hr_bpm=getattr(person, "resting_hr_bpm", None),
        )


@dataclass
class ActivityInputs:
    """The slice of pipeline output the profiles consume (testable directly)."""

    rr: RRSeries
    hrv: HRVResult
    quality: QualityResult
    markers: list[tuple[float, str]] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def excluded_fraction(self) -> float:
        total = self.quality.excluded_total_s + self.quality.analysed_total_s
        return self.quality.excluded_total_s / total if total > 0 else 0.0

    @classmethod
    def from_pipeline(
        cls, result: PipelineResult, markers: list[tuple[float, str]] | None = None
    ) -> ActivityInputs:
        return cls(
            rr=result.rr,
            hrv=result.hrv,
            quality=result.quality,
            markers=markers or [],
            duration_s=result.quality.excluded_total_s + result.quality.analysed_total_s,
        )


@dataclass
class ActivityAnalysis:
    """What a profile concluded about a session."""

    profile_key: str
    hr_limits: HRLimits
    #: mode="suppress" entries actually applied (values blanked).
    suppressions: list[Suppression]
    #: mode="caution" entries — values shown with the warning attached.
    cautions: list[Suppression]
    #: Activity-specific derived metrics (stored in metrics.extras).
    extras: dict[str, Any]
    notes: list[str]
    not_analysable: bool = False
    not_analysable_reason: str | None = None


class ActivityProfile:
    """Base class. Subclasses set the class attributes and override hooks."""

    key: str = ""
    display_name: str = ""
    description: str = ""
    expected_hr_range: tuple[int, int] = (40, 100)
    #: "minimal" | "low" | "moderate" | "high" | "severe"
    expected_motion: str = "minimal"
    #: Supine is the gold-standard reference; nothing else may claim it.
    is_reference_baseline: bool = False
    #: Sessions compare within the same comparison key by default.
    comparison_key: str = ""
    #: Above this excluded fraction the session is declared not analysable.
    max_excluded_fraction: float | None = None
    static_suppressions: tuple[Suppression, ...] = ()
    interpretation_notes: tuple[str, ...] = ()

    def hr_limits(self, ctx: PersonContext) -> HRLimits:
        """Sustained-HR screening limits for this activity."""
        return default_limits(ctx.athlete_baseline)

    def dynamic_suppressions(
        self, inputs: ActivityInputs, ctx: PersonContext
    ) -> list[Suppression]:
        """Suppressions that depend on the session (e.g. intensity)."""
        return []

    def derived_metrics(self, inputs: ActivityInputs, ctx: PersonContext) -> dict[str, Any]:
        """Activity-specific extras for metrics.extras."""
        return {}

    # -- template method -----------------------------------------------------

    def analyze(self, inputs: ActivityInputs, ctx: PersonContext) -> ActivityAnalysis:
        notes = list(self.interpretation_notes)

        if (
            self.max_excluded_fraction is not None
            and inputs.excluded_fraction > self.max_excluded_fraction
        ):
            reason = (
                f"{inputs.excluded_fraction * 100:.0f}% of the recording was excluded for "
                f"signal quality — above the {self.max_excluded_fraction * 100:.0f}% limit "
                f"for {self.display_name.lower()}. This session isn't analysable; the "
                "numbers that could be computed from the remainder would be misleading "
                "rather than merely imprecise."
            )
            return ActivityAnalysis(
                profile_key=self.key,
                hr_limits=self.hr_limits(ctx),
                suppressions=[
                    Suppression(family=f, mode="suppress", reason=reason)
                    for f in MetricFamily.HRV_ALL
                ],
                cautions=[],
                extras={},
                notes=notes,
                not_analysable=True,
                not_analysable_reason=reason,
            )

        all_supp = list(self.static_suppressions) + self.dynamic_suppressions(inputs, ctx)
        return ActivityAnalysis(
            profile_key=self.key,
            hr_limits=self.hr_limits(ctx),
            suppressions=[s for s in all_supp if s.mode == "suppress"],
            cautions=[s for s in all_supp if s.mode == "caution"],
            extras=self.derived_metrics(inputs, ctx),
            notes=notes,
        )


#: HRVResult attributes belonging to each suppressible family. Mean RR / mean
#: HR are rate statistics, not variability — they survive HRV suppression
#: (HR zones and drift still need them).
_FAMILY_FIELDS: dict[str, tuple[str, ...]] = {
    MetricFamily.TIME: (
        "sdnn_ms",
        "rmssd_ms",
        "pnn50_pct",
        "sdnn_per_window_ms",
        "sdnn_window_t_s",
    ),
    MetricFamily.FREQUENCY: (
        "vlf_power_ms2",
        "lf_power_ms2",
        "hf_power_ms2",
        "lf_hf_ratio",
        "lf_peak_hz",
        "psd_method",
    ),
    MetricFamily.NONLINEAR: (
        "sd1_ms",
        "sd2_ms",
        "sd1_sd2_ratio",
        "sample_entropy",
        "dfa_alpha1",
    ),
}


def apply_suppressions(
    hrv: HRVResult, suppressions: list[Suppression]
) -> tuple[HRVResult, list[str]]:
    """Blank suppressed families on a copy of the HRV result.

    Returns the censored copy and a note per suppressed family. Suppressed
    values are removed *before* storage so no comparison, report, or export
    can accidentally resurrect them.
    """
    censored = copy.deepcopy(hrv)
    notes: list[str] = []
    for s in suppressions:
        if s.mode != "suppress":
            continue
        for attr in _FAMILY_FIELDS.get(s.family, ()):
            current = getattr(censored, attr, None)
            if isinstance(current, list):
                setattr(censored, attr, [])
            else:
                setattr(censored, attr, None)
        notes.append(f"{s.family} metrics suppressed: {s.reason}")
    censored.notes.extend(notes)
    return censored, notes
