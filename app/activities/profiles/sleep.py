"""Sleep (overnight) — the sleep-staging profile.

Sessions of this profile get sleep-stage analysis (:mod:`app.sleep`) on top
of the standard pipeline: per-30 s-epoch staging by every available engine,
sleep-architecture metrics, and the report's sleep section.

Screening: the bradycardia threshold drops overnight (nocturnal dipping is
physiology, not a finding — see ``sleep_limits`` in
:mod:`app.screening.thresholds`), so a normal night does not raise a low-HR
flag.

HRV over a whole night violates the stationarity the frequency-domain and
nonlinear metrics assume — autonomic state swings systematically between
stages — so those families are *cautioned* (shown with a warning), not
suppressed: the per-stage table in the sleep section is the readable
version of the same information.
"""

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
from app.screening.thresholds import HRLimits, sleep_limits


@register
class SleepProfile(ActivityProfile):
    key = "sleep"
    display_name = "Sleep (overnight)"
    description = (
        "Overnight recording for sleep-stage analysis: time in each stage, "
        "sleep efficiency, awakenings, and per-stage HR/HRV. Start the "
        "recording at lights-off and stop it on getting up — the analysis "
        "treats the recording span as time in bed."
    )
    expected_hr_range = (35, 90)
    expected_motion = "low"
    is_reference_baseline = False
    comparison_key = "sleep"
    requests_sleep_staging = True
    #: A night with more than half the signal excluded cannot support
    #: stage-fraction claims — the missing time would dominate every number.
    max_excluded_fraction = 0.5
    static_suppressions = (
        Suppression(
            family=MetricFamily.FREQUENCY,
            mode="caution",
            reason=(
                "Whole-night spectra average across sleep stages whose autonomic "
                "states differ systematically (the Task Force 1996 stationarity "
                "requirement is violated by design overnight); read the per-stage "
                "table in the sleep section instead."
            ),
        ),
        Suppression(
            family=MetricFamily.NONLINEAR,
            mode="caution",
            reason=(
                "Sample entropy and DFA α1 over a whole night mix stage-dependent "
                "dynamics; the values are descriptive only."
            ),
        ),
    )
    interpretation_notes = (
        "Sleep sessions compare with other sleep sessions only.",
        "Stage estimates come from heart-rate patterns (plus movement when an "
        "accelerometer file is attached), not brain activity: they are "
        "estimates with known error, not a sleep study.",
    )

    def hr_limits(self, ctx: PersonContext) -> HRLimits:
        return sleep_limits(ctx.athlete_baseline)

    def derived_metrics(self, inputs: ActivityInputs, ctx: PersonContext) -> dict[str, Any]:
        # Light per-RR extras only — staging itself runs in the processing
        # step (see requests_sleep_staging) with access to the full inputs.
        return {
            "lowest_sustained_hr_bpm": m.lowest_sustained_hr(inputs.rr),
            "per_minute_hr_bpm": m.per_minute_hr(inputs.rr)[1],
            "per_minute_hr_t_s": m.per_minute_hr(inputs.rr)[0],
        }
