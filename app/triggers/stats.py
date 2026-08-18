"""Per-tag ectopy rate statistics: negative-binomial rate ratios with guards.

The design follows the trigger-trial literature (docs/RESEARCH.md §4):

* Unit of analysis is the session; the outcome is the confirmed-ectopic count
  with ``log(analysed hours)`` as the exposure offset, so motion-excluded
  time never inflates a rate.
* Counts are modelled negative-binomially — hourly ectopic burden shows a
  coefficient of variation near 60 % (Hamon et al., Heart Rhythm 2015) and
  6-hour windows swing 12-fold (Ahn et al., J Med Internet Res 2024), so a
  Poisson likelihood would produce absurdly narrow intervals. When the NB
  fit does not converge, a Poisson GLM with robust (HC1) errors stands in
  and the row says so.
* A sin/cos hour-of-day pair guards the circadian confound: ectopy has a
  diurnal rhythm and so do most exposures — without this, a caffeine
  comparison rediscovers the morning (docs/RESEARCH.md §4.1).
* One model per tag: with N-of-1 sample sizes a joint model over all tags is
  routinely rank-deficient. The dashboard states the multiple-comparison
  caveat instead of hiding it.

Everything reported is an *association* between what the wearer tagged and
what the detector counted — never a diagnosis, never causation.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np

from app.models import Person, ProcessingStatus, Session

# --- Reporting guards (each cited; below these the variability literature
# says an estimate is noise, and the UI must say "insufficient data") -------

#: Minimum sessions on each side of a tag comparison. With two 24-h periods
#: only a >83 % change exceeds spontaneous ectopy variation (Morganroth et
#: al., Circulation 1978;58:408-414); five per arm is already a bare minimum.
MIN_TAGGED_SESSIONS = 5
MIN_UNTAGGED_SESSIONS = 5

#: Minimum confirmed ectopics across all sessions before any model is fit —
#: count models on near-empty outcomes are degenerate.
MIN_TOTAL_ECTOPIC = 10

#: Sessions shorter than this carry too little exposure to contribute a
#: meaningful rate (10 min; same order as the settling period after donning
#: a dry-electrode strap, Joutsen et al., Sci Rep 2024).
MIN_ANALYSED_S = 600.0

#: The activity covariate joins the model only when at least two
#: comparison-key groups each have this many sessions; otherwise the dummy
#: coefficients are noise and the covariate is dropped (with a note).
MIN_SESSIONS_PER_ACTIVITY_LEVEL = 3


@dataclass(frozen=True)
class SessionObservation:
    session_id: int
    recorded_at: dt.datetime
    analysed_hours: float
    duration_s: float
    ectopy_count: int
    ectopy_per_hour: float
    comparison_key: str | None
    tags: frozenset[str]
    reduced_confidence: bool


@dataclass
class AssemblyNotes:
    """Honest accounting of which sessions the statistics could not use."""

    n_not_done: int = 0
    n_awaiting_reprocess: int = 0
    n_too_short: int = 0
    awaiting_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class TriggerEffect:
    tag_slug: str
    tag_name: str
    n_tagged: int
    n_untagged: int
    tagged_rate_per_hour: float | None
    untagged_rate_per_hour: float | None
    rate_ratio: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    #: "ok" | "poisson_fallback" | "insufficient_data" | "no_events" |
    #: "model_failed"
    status: str
    #: One honest sentence for the UI when there is no (reliable) estimate.
    detail: str


@dataclass
class TriggerAnalysis:
    observations: list[SessionObservation]
    effects: list[TriggerEffect]
    notes: AssemblyNotes
    model_notes: list[str]

    @property
    def total_ectopics(self) -> int:
        return sum(o.ectopy_count for o in self.observations)

    @property
    def total_analysed_hours(self) -> float:
        return sum(o.analysed_hours for o in self.observations)

    @property
    def overall_rate_per_hour(self) -> float | None:
        hours = self.total_analysed_hours
        return (self.total_ectopics / hours) if hours > 0 else None


def assemble_observations(
    person: Person,
) -> tuple[list[SessionObservation], AssemblyNotes]:
    """Sessions of one person as model observations, with usage accounting."""
    observations: list[SessionObservation] = []
    notes = AssemblyNotes()
    for session in person.sessions:
        if session.processing_status != ProcessingStatus.DONE:
            notes.n_not_done += 1
            continue
        metrics = session.metrics
        if metrics is None or metrics.ectopy_beats_n is None:
            notes.n_awaiting_reprocess += 1
            notes.awaiting_ids.append(session.id)
            continue
        analysed_s = session.analysed_s or 0.0
        if analysed_s < MIN_ANALYSED_S:
            notes.n_too_short += 1
            continue
        observations.append(
            SessionObservation(
                session_id=session.id,
                recorded_at=session.recorded_at or session.created_at,
                analysed_hours=analysed_s / 3600.0,
                duration_s=session.duration_s or analysed_s,
                ectopy_count=int(metrics.ectopy_beats_n),
                ectopy_per_hour=float(metrics.ectopy_per_hour or 0.0),
                comparison_key=_comparison_key(session),
                tags=frozenset(t.slug for t in session.trigger_tags),
                reduced_confidence=bool(session.reduced_confidence),
            )
        )
    observations.sort(key=lambda o: o.recorded_at)
    return observations, notes


def _comparison_key(session: Session) -> str | None:
    activity = session.activity_type
    if activity is None:
        return None
    if activity.comparison_key:
        return activity.comparison_key
    return activity.profile_key


def analyse_triggers(person: Person) -> TriggerAnalysis:
    """The full per-tag analysis for one person."""
    observations, notes = assemble_observations(person)
    model_notes: list[str] = []
    seen_slugs = sorted({slug for o in observations for slug in o.tags})
    tag_names = _tag_names(seen_slugs)

    effects = [
        fit_tag_effect(observations, slug, tag_names.get(slug, slug), model_notes)
        for slug in seen_slugs
    ]
    if observations:
        model_notes.append(
            "Hour-of-day uses the recording timestamps as stored (UTC-derived); "
            "sessions recorded across time zones shift the circadian covariate "
            "accordingly."
        )
    return TriggerAnalysis(
        observations=observations, effects=effects, notes=notes, model_notes=model_notes
    )


def _tag_names(slugs: list[str]) -> dict[str, str]:
    from app.extensions import db
    from app.models import TriggerTag

    if not slugs:
        return {}
    rows = db.session.query(TriggerTag).filter(TriggerTag.slug.in_(slugs)).all()
    return {t.slug: t.name for t in rows}


def fit_tag_effect(
    observations: list[SessionObservation],
    tag_slug: str,
    tag_name: str,
    model_notes: list[str] | None = None,
) -> TriggerEffect:
    """One tag's rate-ratio estimate, or an honest refusal."""
    tagged = [o for o in observations if tag_slug in o.tags]
    untagged = [o for o in observations if tag_slug not in o.tags]

    def rate(group: list[SessionObservation]) -> float | None:
        hours = sum(o.analysed_hours for o in group)
        return (sum(o.ectopy_count for o in group) / hours) if hours > 0 else None

    base = dict(
        tag_slug=tag_slug,
        tag_name=tag_name,
        n_tagged=len(tagged),
        n_untagged=len(untagged),
        tagged_rate_per_hour=rate(tagged),
        untagged_rate_per_hour=rate(untagged),
        rate_ratio=None,
        ci_low=None,
        ci_high=None,
        p_value=None,
    )

    if len(tagged) < MIN_TAGGED_SESSIONS or len(untagged) < MIN_UNTAGGED_SESSIONS:
        return TriggerEffect(
            **base,
            status="insufficient_data",
            detail=(
                f"Needs at least {MIN_TAGGED_SESSIONS} tagged and "
                f"{MIN_UNTAGGED_SESSIONS} untagged sessions (have {len(tagged)} "
                f"and {len(untagged)}): below that, day-to-day ectopy "
                "variability dominates any estimate."
            ),
        )
    total_events = sum(o.ectopy_count for o in observations)
    if total_events < MIN_TOTAL_ECTOPIC:
        return TriggerEffect(
            **base,
            status="insufficient_data",
            detail=(
                f"Only {total_events} confirmed ectopic beat(s) across all "
                f"sessions; a count model needs at least {MIN_TOTAL_ECTOPIC}."
            ),
        )
    if sum(o.ectopy_count for o in tagged) == 0 or sum(o.ectopy_count for o in untagged) == 0:
        return TriggerEffect(
            **base,
            status="no_events",
            detail=(
                "One side of the comparison has zero confirmed ectopic beats; "
                "the rates are shown but a ratio would be unbounded."
            ),
        )

    y, x, exposure, dropped = _design_matrix(observations, tag_slug)
    if model_notes is not None:
        for note in dropped:
            model_notes.append(f"{tag_name}: {note}")

    try:
        return _fit(base, y, x, exposure)
    except Exception as exc:  # noqa: BLE001 - any optimizer failure → honest row
        return TriggerEffect(
            **base,
            status="model_failed",
            detail=f"Model fitting failed ({type(exc).__name__}); rates shown only.",
        )


def _design_matrix(
    observations: list[SessionObservation], tag_slug: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """(counts, design, exposure_hours, dropped-covariate notes).

    Column 1 is always the tag indicator. Hour-of-day enters as a sin/cos
    pair; the activity covariate only when its groups are populated enough.
    Degrees-of-freedom guard: activity drops first, then the hour pair — the
    minimum model is intercept + tag.
    """
    n = len(observations)
    y = np.array([o.ectopy_count for o in observations], dtype=float)
    exposure = np.array([o.analysed_hours for o in observations])
    tag_col = np.array([1.0 if tag_slug in o.tags else 0.0 for o in observations])

    hours = np.array(
        [o.recorded_at.hour + o.recorded_at.minute / 60.0 for o in observations]
    )
    hour_cols = [np.sin(2 * np.pi * hours / 24.0), np.cos(2 * np.pi * hours / 24.0)]

    dropped: list[str] = []
    activity_cols: list[np.ndarray] = []
    keys = [o.comparison_key or "unspecified" for o in observations]
    counts = {k: keys.count(k) for k in set(keys)}
    populated = sorted(k for k, c in counts.items() if c >= MIN_SESSIONS_PER_ACTIVITY_LEVEL)
    if len(counts) > 1:
        if len(populated) >= 2 and populated == sorted(counts):
            for level in populated[1:]:  # first level is the reference
                activity_cols.append(
                    np.array([1.0 if k == level else 0.0 for k in keys])
                )
        else:
            dropped.append(
                "activity covariate dropped — not every activity group has "
                f"{MIN_SESSIONS_PER_ACTIVITY_LEVEL}+ sessions."
            )

    cols = [np.ones(n), tag_col, *hour_cols, *activity_cols]
    if len(cols) + 2 > n and activity_cols:
        cols = [np.ones(n), tag_col, *hour_cols]
        dropped.append("activity covariate dropped — too few sessions for it.")
    if len(cols) + 2 > n:
        cols = [np.ones(n), tag_col]
        dropped.append("hour-of-day covariate dropped — too few sessions for it.")
    return y, np.column_stack(cols), exposure, dropped


def _fit(base: dict, y: np.ndarray, x: np.ndarray, exposure: np.ndarray) -> TriggerEffect:
    import statsmodels.api as sm

    tag_idx = 1  # column order fixed by _design_matrix

    try:
        nb = sm.NegativeBinomial(y, x, exposure=exposure).fit(
            method="bfgs", maxiter=200, disp=0
        )
        converged = bool(nb.mle_retvals.get("converged", False))
        bse_ok = np.all(np.isfinite(nb.bse[: x.shape[1]]))
        if converged and bse_ok:
            return _effect_from(base, nb, tag_idx, status="ok")
    except Exception:  # noqa: BLE001 - fall through to the Poisson fallback
        pass

    glm = sm.GLM(y, x, family=sm.families.Poisson(), exposure=exposure).fit(
        cov_type="HC1"
    )
    return _effect_from(base, glm, tag_idx, status="poisson_fallback")


def _effect_from(base: dict, res, tag_idx: int, status: str) -> TriggerEffect:
    coef = float(res.params[tag_idx])
    se = float(res.bse[tag_idx])
    detail = ""
    if status == "poisson_fallback":
        detail = (
            "The negative-binomial fit did not converge; a Poisson model with "
            "robust errors stands in, so this interval is approximate."
        )
    estimated = {
        **base,
        "rate_ratio": math.exp(coef),
        "ci_low": math.exp(coef - 1.96 * se),
        "ci_high": math.exp(coef + 1.96 * se),
        "p_value": float(res.pvalues[tag_idx]),
    }
    return TriggerEffect(**estimated, status=status, detail=detail)


# --- Descriptive helpers ---------------------------------------------------


def rates_by_tag(
    observations: list[SessionObservation],
) -> dict[str, tuple[list[float], list[float]]]:
    """tag slug → (tagged per-session rates, untagged per-session rates)."""
    slugs = sorted({slug for o in observations for slug in o.tags})
    out: dict[str, tuple[list[float], list[float]]] = {}
    for slug in slugs:
        tagged = [o.ectopy_per_hour for o in observations if slug in o.tags]
        untagged = [o.ectopy_per_hour for o in observations if slug not in o.tags]
        out[slug] = (tagged, untagged)
    return out


@dataclass
class HourProfile:
    """Exposure-normalised ectopy rate by clock hour."""

    hours: list[int]
    events_per_hour: list[float | None]
    n_events: list[int]
    exposure_hours: list[float]
    n_sessions_used: int
    n_sessions_skipped: int


def hour_of_day_profile(
    observations: list[SessionObservation],
    event_offsets_by_session: dict[int, np.ndarray],
) -> HourProfile:
    """Events and exposure allocated to wall-clock hours.

    A session's analysed time is spread across the clock hours its recording
    span covers, proportionally to overlap — a stated simplification (the
    positions of intra-session exclusions are ignored). Sessions without
    cached event arrays are skipped and counted.
    """
    exposure = np.zeros(24)
    events = np.zeros(24, dtype=int)
    used = skipped = 0
    for o in observations:
        offsets = event_offsets_by_session.get(o.session_id)
        if offsets is None:
            skipped += 1
            continue
        used += 1
        start = o.recorded_at
        span_s = max(o.duration_s, 1.0)
        analysed_fraction = (o.analysed_hours * 3600.0) / span_s
        # Walk the wall-clock hours the recording overlaps.
        t = 0.0
        while t < span_s:
            moment = start + dt.timedelta(seconds=t)
            step = min(3600.0 - (moment.minute * 60 + moment.second), span_s - t)
            exposure[moment.hour] += step * analysed_fraction / 3600.0
            t += step
        for offset in offsets:
            events[(start + dt.timedelta(seconds=float(offset))).hour] += 1
    return HourProfile(
        hours=list(range(24)),
        events_per_hour=[
            (events[h] / exposure[h]) if exposure[h] > 0 else None for h in range(24)
        ],
        n_events=list(events),
        exposure_hours=list(exposure),
        n_sessions_used=used,
        n_sessions_skipped=skipped,
    )
