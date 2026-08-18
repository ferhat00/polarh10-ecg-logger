"""Every screening threshold in one place, with its source.

This module is the single point of truth for rule-based screening. Nothing
elsewhere in the app may hardcode a screening number. All rules are screening
heuristics for a single-lead chest strap, not diagnostic criteria.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Sustained heart-rate screening
# ---------------------------------------------------------------------------

#: Sustained HR above this at rest crosses the tachycardia screening
#: threshold. Source: conventional sinus-tachycardia definition, >100 bpm at
#: rest (Olshansky & Sullivan, JACC 2013;61:793-801).
TACHY_SUSTAINED_BPM = 100.0

#: Sustained HR below this at rest crosses the bradycardia screening
#: threshold. Source: conventional <60 bpm definition, while noting its low
#: specificity (Kusumoto et al., 2018 ACC/AHA/HRS bradycardia guideline,
#: Circulation 2019;140:e382-e482).
BRADY_SUSTAINED_BPM = 60.0

#: Bradycardia threshold when the person is marked as having an athlete
#: baseline. Endurance training commonly produces resting rates of 40-50 bpm
#: through elevated vagal tone and intrinsic adaptation (sinus bradycardia in
#: athletes: Sharma et al., international recommendations for ECG
#: interpretation in athletes, JACC 2017;69:1057-1075).
ATHLETE_BRADY_BPM = 45.0

#: Bradycardia threshold while asleep. Nocturnal HR normally dips 10-30 %
#: below waking rest ("dipping"); rates in the 40s are unremarkable in
#: healthy sleepers and sinus rates in the 30s with pauses are common in
#: trained athletes overnight (Kusumoto et al., 2018 ACC/AHA/HRS bradycardia
#: guideline, Circulation 2019;140:e382-e482 — nocturnal bradycardia in
#: normals; Sharma et al., JACC 2017;69:1057-1075 — athletes). Keeping the
#: resting threshold overnight would flag nearly every healthy night.
SLEEP_BRADY_BPM = 40.0
SLEEP_ATHLETE_BRADY_BPM = 35.0

#: Tachycardia threshold while asleep: unchanged from rest — a *sustained*
#: rate above 100 bpm during sleep remains worth surfacing.
SLEEP_TACHY_BPM = TACHY_SUSTAINED_BPM

#: "Sustained" means a rolling window of this length stays across the
#: threshold — single-beat excursions never flag. ~45 s per the project spec;
#: comfortably longer than transient sinus arrhythmia or a startle response.
SUSTAINED_WINDOW_S = 45.0

#: Step between successive sustained-HR windows.
SUSTAINED_STEP_S = 5.0

#: Minimum RR intervals a window must contain to be evaluated.
SUSTAINED_MIN_BEATS = 10

# ---------------------------------------------------------------------------
# Ectopy screening
# ---------------------------------------------------------------------------

#: Ectopic-beat percentage above which a flag is raised. An ectopic burden
#: over ~1 % is the point where population studies begin to associate burden
#: with remodelling risk (Dukes et al., JACC 2015;66:101-109 — reported for
#: ventricular ectopy; used here as a conservative screening cut for
#: single-lead-confirmed ectopic beats of any origin).
ECTOPY_PCT_THRESHOLD = 1.0

#: A repeating alternating pattern (bigeminy: ectopic every other beat) flags
#: at any burden if at least this many ectopics form the pattern.
BIGEMINY_MIN_RUN = 3

# ---------------------------------------------------------------------------
# Ectopy event grouping
# ---------------------------------------------------------------------------

#: Two consecutive confirmed ectopic beats form a couplet; this vocabulary
#: follows the classical ventricular-ectopy grading tradition (Lown & Wolf,
#: Circulation 1971;44:130-142), used here for single-lead-confirmed ectopic
#: beats of any origin.
COUPLET_BEATS = 2

#: Three or more consecutive confirmed ectopic beats form a run (salvo), per
#: the same grading tradition.
RUN_MIN_BEATS = 3

#: Beat spacing that defines the alternating patterns: bigeminy is an ectopic
#: every other beat (spacing 2), trigeminy every third beat (spacing 3).
#: Episode length still requires BIGEMINY_MIN_RUN pattern beats.
BIGEMINY_SPACING = 2
TRIGEMINY_SPACING = 3

#: A post-ectopic pause counts as "full compensatory" when
#: (RR_pre + RR_post) reaches this fraction of twice the local reference RR —
#: the classical full-pause criterion (the sinus beat after the ectopic lands
#: on schedule; see e.g. Marriott's Practical Electrocardiography). The 5 %
#: tolerance absorbs the ±7.7 ms sample quantisation at 130 Hz (~2 % of an
#: 800 ms interval) plus normal sinus-rate drift. A full pause is *typical of*
#: ventricular origin in the literature, but at 130 Hz on a single lead this
#: ratio is reported as a timing descriptor only, never an origin call.
PAUSE_COMPLETE_RATIO = 0.95

# ---------------------------------------------------------------------------
# RR irregularity (possible AF) screening
# ---------------------------------------------------------------------------

#: COSEn threshold. Lake & Moorman (Am J Physiol Heart Circ Physiol
#: 2011;300:H319-25) report AF detection thresholds near -1.2 for very short
#: RR windows; -1.0 is used as the conservative screening cut. Sign
#: convention: HIGHER (less negative) = MORE irregular.
COSEN_THRESHOLD = -1.0

#: RR coefficient of variation (σ/μ) threshold over the same window. AF
#: typically shows CV well above 0.10, while healthy resting sinus rhythm
#: with high vagal tone stays below it (Tateno & Glass, Med Biol Eng Comput
#: 2001;39:664-671 CV-based AF detection). Both gates must trip — requiring
#: agreement is what keeps healthy high-HRV sinus rhythm from flagging.
RR_CV_THRESHOLD = 0.10

#: Rolling window length in beats for the irregularity indices (Lake &
#: Moorman validated COSEn down to 12-beat windows; 30 gives stabler CV).
IRREGULARITY_WINDOW_BEATS = 30

#: Step between successive irregularity windows, in beats.
IRREGULARITY_STEP_BEATS = 5

#: Consecutive dual-positive windows required before flagging (~45+ beats of
#: sustained irregularity; transient runs never flag).
IRREGULARITY_MIN_CONSECUTIVE = 3


@dataclass(frozen=True)
class HRLimits:
    """Activity-adjusted sustained-HR screening limits.

    Activity profiles supply these; the defaults describe rest. ``context``
    appears in flag descriptions so a reader knows which expectation was
    applied.
    """

    tachy_bpm: float = TACHY_SUSTAINED_BPM
    brady_bpm: float = BRADY_SUSTAINED_BPM
    context: str = "resting"
    #: Extra sentence appended to any bradycardia flag (e.g. athlete note).
    brady_note: str | None = None


def default_limits(athlete_baseline: bool) -> HRLimits:
    """Resting limits, adjusted for a per-user athlete baseline."""
    if athlete_baseline:
        return HRLimits(
            brady_bpm=ATHLETE_BRADY_BPM,
            brady_note=(
                "The athlete-baseline setting is enabled for this person, so the low-HR "
                "screening threshold was lowered to "
                f"{ATHLETE_BRADY_BPM:.0f} bpm — endurance training commonly produces "
                "resting rates in the 40s without any pattern worth noting."
            ),
        )
    return HRLimits(
        brady_note=(
            "A low resting rate is a low-specificity observation, especially in "
            "physically trained people; consider enabling the athlete-baseline setting "
            "if that applies."
        ),
    )


def sleep_limits(athlete_baseline: bool) -> HRLimits:
    """Overnight limits: the bradycardia threshold drops because nocturnal
    HR dipping is normal physiology, not a finding."""
    if athlete_baseline:
        return HRLimits(
            tachy_bpm=SLEEP_TACHY_BPM,
            brady_bpm=SLEEP_ATHLETE_BRADY_BPM,
            context="sleeping",
            brady_note=(
                "Sleep plus the athlete-baseline setting lowers the low-HR screening "
                f"threshold to {SLEEP_ATHLETE_BRADY_BPM:.0f} bpm — nocturnal sinus "
                "rates in the 30s are a recognised finding in endurance-trained "
                "sleepers, not a pattern worth noting by itself."
            ),
        )
    return HRLimits(
        tachy_bpm=SLEEP_TACHY_BPM,
        brady_bpm=SLEEP_BRADY_BPM,
        context="sleeping",
        brady_note=(
            f"The low-HR screening threshold during sleep is {SLEEP_BRADY_BPM:.0f} bpm "
            "rather than the resting threshold: nocturnal heart rate normally dips "
            "10-30% below waking rest, and rates in the 40s are unremarkable in "
            "healthy sleepers."
        ),
    )
