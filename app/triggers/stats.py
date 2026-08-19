"""Per-tag statistics: multi-outcome tag effects with guards.

The design follows the trigger-trial literature (docs/RESEARCH.md §4):

* Unit of analysis is the session. Three outcomes are supported (the
  ``OUTCOMES`` registry): **ectopy** — the original confirmed-ectopic count
  with ``log(analysed hours)`` as the exposure offset, so motion-excluded
  time never inflates a rate; **ln_rmssd** — the trend-tracking log of
  RMSSD, fit by OLS with robust (HC1) errors and reported as a % change;
  **resting_hr** — the lowest-sustained-60 s heart rate, fit the same way
  and reported as a Δbpm. All outcomes share the same design matrix
  (tag + circadian pair + activity dummies) and the same minimum-n guards.
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

#: Environment context becomes a (descriptive-only) dashboard section once
#: this many usable sessions carry a value — below that even a rank
#: correlation is noise. No regression covariate is fit at personal-dataset
#: sizes: the published effect sizes are percent-level (Gold 2000; Wang
#: 2020), far below what dozens of sessions can resolve honestly.
MIN_ENV_OBSERVATIONS = 12


@dataclass(frozen=True)
class OutcomeSpec:
    """One supported outcome for the per-tag models."""

    key: str
    label: str
    #: "rate_ratio" (count model) | "pct_change" | "delta"
    kind: str
    #: Unit of the per-group summary columns shown in the table.
    group_unit: str
    #: The no-effect reference value for the forest plot (1.0 or 0.0).
    null_value: float
    #: Short model description for the table's notes column.
    model_label: str


OUTCOMES: dict[str, OutcomeSpec] = {
    "ectopy": OutcomeSpec(
        key="ectopy",
        label="Ectopy burden",
        kind="rate_ratio",
        group_unit="/h",
        null_value=1.0,
        model_label="negative-binomial model",
    ),
    "ln_rmssd": OutcomeSpec(
        key="ln_rmssd",
        label="RMSSD (vagal HRV)",
        kind="pct_change",
        group_unit="ms",
        null_value=0.0,
        model_label="linear model on ln(RMSSD), robust errors",
    ),
    "resting_hr": OutcomeSpec(
        key="resting_hr",
        label="Resting HR",
        kind="delta",
        group_unit="bpm",
        null_value=0.0,
        model_label="linear model, robust errors",
    ),
}


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
    #: Gaussian outcomes (None when unavailable or suppressed).
    ln_rmssd: float | None = None
    rmssd_ms: float | None = None
    resting_hr_bpm: float | None = None
    #: Environment context (None when never fetched).
    env_temp_c: float | None = None
    env_pm25_ugm3: float | None = None

    def outcome_value(self, outcome: OutcomeSpec) -> float | None:
        """The modelling-scale value of this observation for one outcome."""
        if outcome.key == "ectopy":
            return float(self.ectopy_count)
        if outcome.key == "ln_rmssd":
            return self.ln_rmssd
        if outcome.key == "resting_hr":
            return self.resting_hr_bpm
        raise KeyError(outcome.key)


@dataclass
class AssemblyNotes:
    """Honest accounting of which sessions the statistics could not use."""

    n_not_done: int = 0
    n_awaiting_reprocess: int = 0
    n_too_short: int = 0
    awaiting_ids: list[int] = field(default_factory=list)
    #: Sessions usable in general but lacking the selected outcome.
    n_missing_outcome: int = 0


@dataclass(frozen=True)
class TriggerEffect:
    tag_slug: str
    tag_name: str
    n_tagged: int
    n_untagged: int
    #: Per-group summaries in the outcome's display unit (/h, ms, bpm).
    tagged_rate_per_hour: float | None
    untagged_rate_per_hour: float | None
    #: The effect estimate; its meaning follows ``effect_kind``:
    #: rate ratio (ectopy), % change (ln RMSSD), or Δ (resting HR).
    rate_ratio: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    #: "ok" | "poisson_fallback" | "insufficient_data" | "no_events" |
    #: "model_failed"
    status: str
    #: One honest sentence for the UI when there is no (reliable) estimate.
    detail: str
    #: "rate_ratio" | "pct_change" | "delta" — how to read the fields above.
    effect_kind: str = "rate_ratio"


@dataclass
class TriggerAnalysis:
    observations: list[SessionObservation]
    effects: list[TriggerEffect]
    notes: AssemblyNotes
    model_notes: list[str]
    outcome: OutcomeSpec = OUTCOMES["ectopy"]

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
        extras = metrics.extras or {}
        rmssd = metrics.rmssd_ms
        resting = extras.get("resting_hr_bpm")
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
                ln_rmssd=math.log(rmssd) if rmssd and rmssd > 0 else None,
                rmssd_ms=float(rmssd) if rmssd is not None else None,
                resting_hr_bpm=float(resting) if resting is not None else None,
                env_temp_c=session.env_temp_c,
                env_pm25_ugm3=session.env_pm25_ugm3,
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


def analyse_triggers(person: Person, outcome: str = "ectopy") -> TriggerAnalysis:
    """The full per-tag analysis for one person, for one outcome."""
    spec = OUTCOMES[outcome]
    observations, notes = assemble_observations(person)
    model_notes: list[str] = []

    usable = observations
    if spec.key != "ectopy":
        usable = [o for o in observations if o.outcome_value(spec) is not None]
        notes.n_missing_outcome = len(observations) - len(usable)
        if notes.n_missing_outcome:
            model_notes.append(
                f"{notes.n_missing_outcome} session(s) lack a usable "
                f"{spec.label} value (suppressed or unavailable) and are not "
                "in these models."
            )

    seen_slugs = sorted({slug for o in usable for slug in o.tags})
    tag_names = _tag_names(seen_slugs)
    effects = [
        fit_tag_effect(usable, slug, tag_names.get(slug, slug), model_notes, spec)
        for slug in seen_slugs
    ]
    if usable:
        model_notes.append(
            "Hour-of-day uses the recording timestamps as stored (UTC-derived); "
            "sessions recorded across time zones shift the circadian covariate "
            "accordingly."
        )
    return TriggerAnalysis(
        observations=usable,
        effects=effects,
        notes=notes,
        model_notes=model_notes,
        outcome=spec,
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
    outcome: OutcomeSpec = OUTCOMES["ectopy"],
) -> TriggerEffect:
    """One tag's effect estimate for the given outcome, or an honest refusal."""
    tagged = [o for o in observations if tag_slug in o.tags]
    untagged = [o for o in observations if tag_slug not in o.tags]

    base = dict(
        tag_slug=tag_slug,
        tag_name=tag_name,
        n_tagged=len(tagged),
        n_untagged=len(untagged),
        tagged_rate_per_hour=_group_summary(tagged, outcome),
        untagged_rate_per_hour=_group_summary(untagged, outcome),
        rate_ratio=None,
        ci_low=None,
        ci_high=None,
        p_value=None,
        effect_kind=outcome.kind,
    )

    if len(tagged) < MIN_TAGGED_SESSIONS or len(untagged) < MIN_UNTAGGED_SESSIONS:
        return TriggerEffect(
            **base,
            status="insufficient_data",
            detail=(
                f"Needs at least {MIN_TAGGED_SESSIONS} tagged and "
                f"{MIN_UNTAGGED_SESSIONS} untagged sessions (have {len(tagged)} "
                f"and {len(untagged)}): below that, day-to-day "
                "variability dominates any estimate."
            ),
        )

    if outcome.key != "ectopy":
        return _fit_gaussian_effect(base, observations, tag_slug, outcome, model_notes)

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


def _group_summary(
    group: list[SessionObservation], outcome: OutcomeSpec
) -> float | None:
    """Per-group summary in the outcome's display unit.

    Ectopy: events per analysed hour. RMSSD: geometric mean in ms (the
    display-scale counterpart of the ln model). Resting HR: plain mean bpm.
    """
    if not group:
        return None
    if outcome.key == "ectopy":
        hours = sum(o.analysed_hours for o in group)
        return (sum(o.ectopy_count for o in group) / hours) if hours > 0 else None
    values = [o.outcome_value(outcome) for o in group]
    values = [v for v in values if v is not None]
    if not values:
        return None
    mean = float(np.mean(values))
    return math.exp(mean) if outcome.key == "ln_rmssd" else mean


def _fit_gaussian_effect(
    base: dict,
    observations: list[SessionObservation],
    tag_slug: str,
    outcome: OutcomeSpec,
    model_notes: list[str] | None,
) -> TriggerEffect:
    """OLS with robust errors on a Gaussian outcome; refusals stay honest."""
    values = np.array([o.outcome_value(outcome) for o in observations], dtype=float)
    if float(np.std(values, ddof=1)) <= 1e-9:
        return TriggerEffect(
            **base,
            status="insufficient_data",
            detail=f"{outcome.label} shows no variation across these sessions.",
        )

    _, x, _, dropped = _design_matrix(observations, tag_slug)
    if model_notes is not None:
        for note in dropped:
            model_notes.append(f"{base['tag_name']}: {note}")

    try:
        import statsmodels.api as sm

        res = sm.OLS(values, x).fit(cov_type="HC1")
        coef = float(res.params[1])  # column order fixed by _design_matrix
        se = float(res.bse[1])
        if not (math.isfinite(coef) and math.isfinite(se)):
            raise ValueError("non-finite estimate")
    except Exception as exc:  # noqa: BLE001 - any failure → honest row
        return TriggerEffect(
            **base,
            status="model_failed",
            detail=f"Model fitting failed ({type(exc).__name__}); means shown only.",
        )

    lo, hi = coef - 1.96 * se, coef + 1.96 * se
    if outcome.kind == "pct_change":
        effect, ci_low, ci_high = (
            (math.exp(coef) - 1.0) * 100.0,
            (math.exp(lo) - 1.0) * 100.0,
            (math.exp(hi) - 1.0) * 100.0,
        )
    else:  # delta
        effect, ci_low, ci_high = coef, lo, hi
    estimated = {
        **base,
        "rate_ratio": effect,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(res.pvalues[1]),
    }
    return TriggerEffect(**estimated, status="ok", detail=outcome.model_label)


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


# --- Environment context (descriptive only) --------------------------------

#: (observation attribute, label, unit) of the offered environment variables.
ENV_VARIABLES: tuple[tuple[str, str, str], ...] = (
    ("env_temp_c", "Temperature", "°C"),
    ("env_pm25_ugm3", "PM2.5", "µg/m³"),
)


@dataclass(frozen=True)
class EnvAssociation:
    """One environment variable vs the selected outcome — descriptive only."""

    var_key: str
    var_label: str
    unit: str
    n: int
    #: Spearman rank correlation with the outcome (display scale), and its p.
    rho: float | None
    rho_p: float | None
    #: (range label, n sessions, mean outcome in display unit) per tertile.
    tertiles: list[tuple[str, int, float]]
    #: Paired (env value, outcome display value) for the scatter figure.
    points: list[tuple[float, float]]


def env_associations(
    observations: list[SessionObservation], outcome: OutcomeSpec
) -> list[EnvAssociation]:
    """Descriptive env-vs-outcome summaries, one per variable with data.

    Deliberately NOT a regression covariate: the published environmental
    effect sizes are percent-level, far below what a personal dataset can
    resolve — a rank correlation plus tertile means is the honest ceiling.
    Variables with fewer than MIN_ENV_OBSERVATIONS valued sessions are
    omitted entirely.
    """
    from scipy import stats as sp_stats

    out: list[EnvAssociation] = []
    for attr, label, unit in ENV_VARIABLES:
        pairs = []
        for o in observations:
            env_value = getattr(o, attr)
            display = _display_outcome(o, outcome)
            if env_value is not None and display is not None:
                pairs.append((float(env_value), float(display)))
        if len(pairs) < MIN_ENV_OBSERVATIONS:
            continue
        env_vals = np.array([p[0] for p in pairs])
        out_vals = np.array([p[1] for p in pairs])
        if float(np.std(env_vals)) <= 1e-9 or float(np.std(out_vals)) <= 1e-9:
            continue
        rho, rho_p = sp_stats.spearmanr(env_vals, out_vals)

        order = np.argsort(env_vals)
        thirds = np.array_split(order, 3)
        tertiles: list[tuple[str, int, float]] = []
        for idx in thirds:
            if len(idx) == 0:
                continue
            lo, hi = float(env_vals[idx].min()), float(env_vals[idx].max())
            tertiles.append(
                (f"{lo:.0f}–{hi:.0f} {unit}", len(idx), float(np.mean(out_vals[idx])))
            )
        out.append(
            EnvAssociation(
                var_key=attr,
                var_label=label,
                unit=unit,
                n=len(pairs),
                rho=float(rho) if math.isfinite(rho) else None,
                rho_p=float(rho_p) if math.isfinite(rho_p) else None,
                tertiles=tertiles,
                points=pairs,
            )
        )
    return out


def _display_outcome(o: SessionObservation, outcome: OutcomeSpec) -> float | None:
    """The outcome in its display unit (per-hour rate, RMSSD ms, bpm)."""
    if outcome.key == "ectopy":
        return o.ectopy_per_hour
    if outcome.key == "ln_rmssd":
        return o.rmssd_ms
    return o.resting_hr_bpm


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
