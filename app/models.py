"""SQLAlchemy models.

Schema notes
------------
* ``Person`` is profile separation only, not authentication — several people
  may share one chest strap.
* Provenance and quality figures (unit/epoch detection, correction rate,
  excluded time) are first-class ``Session`` columns because they drive badges
  and comparison guards throughout the UI and must be queryable.
* ``Flag.disclaimer`` is NOT NULL with a CHECK constraint: the screening
  disclaimer is part of the flag object itself so no template change can
  separate a flag from its disclaimer.
"""

from __future__ import annotations

import datetime as dt
import re

from sqlalchemy import CheckConstraint, Column, ForeignKey, String, Table, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.extensions import db


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def slugify(name: str) -> str:
    """Lowercase, ASCII-ish, hyphen-separated slug for filesystem paths."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "person"


class Person(db.Model):
    __tablename__ = "person"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    #: Filesystem-safe identifier used for upload and log paths.
    slug: Mapped[str] = mapped_column(String(140), unique=True, index=True)
    date_of_birth: Mapped[dt.date | None] = mapped_column(default=None)
    #: Trained endurance athletes commonly show resting HR well below 60 bpm;
    #: the bradycardia screen lowers its threshold and says so when this is set.
    athlete_baseline: Mapped[bool] = mapped_column(default=False)
    #: Optional measured maximum HR. Used for HR-reserve computations
    #: (running/cycling HRV suppression). Fallback: 220 - age, then a default.
    max_hr_bpm: Mapped[int | None] = mapped_column(default=None)
    #: Optional known resting HR, used for HR-reserve (Karvonen) computations.
    resting_hr_bpm: Mapped[int | None] = mapped_column(default=None)
    #: Optional, "male"/"female". Used ONLY as a covariate by the SleepECG
    #: sleep-stage classifiers (trained with a binary sex feature); never for
    #: screening thresholds. Unknown is a fully supported value.
    sex: Mapped[str | None] = mapped_column(String(10), default=None)
    created_at: Mapped[dt.datetime] = mapped_column(default=utcnow)

    sessions: Mapped[list[Session]] = relationship(
        back_populates="person", cascade="all, delete-orphan"
    )
    annotations: Mapped[list[Annotation]] = relationship(
        back_populates="person", cascade="all, delete-orphan"
    )
    #: Opt-in Polar Flow link. Deleting the person deletes the stored token.
    polar_account: Mapped[PolarAccount | None] = relationship(
        back_populates="person", cascade="all, delete-orphan", uselist=False
    )
    flow_nights: Mapped[list[FlowNight]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
        order_by="FlowNight.date",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Person {self.id} {self.slug!r}>"


class ActivityType(db.Model):
    """Maps a user-facing activity name to a coded activity profile.

    The eight built-in activities are seeded rows with ``is_builtin=True``.
    Activities added from the UI reference an existing profile via
    ``profile_key`` (they inherit its behaviour) and may override the expected
    HR range and the comparison grouping key.
    """

    __tablename__ = "activity_type"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    #: Key into the coded activity-profile registry (``app.activities``).
    profile_key: Mapped[str] = mapped_column(String(40), index=True)
    is_builtin: Mapped[bool] = mapped_column(default=False)
    #: Optional overrides for UI-added activities; None = inherit profile.
    expected_hr_min: Mapped[int | None] = mapped_column(default=None)
    expected_hr_max: Mapped[int | None] = mapped_column(default=None)
    comparison_key: Mapped[str | None] = mapped_column(String(40), default=None)
    created_at: Mapped[dt.datetime] = mapped_column(default=utcnow)

    sessions: Mapped[list[Session]] = relationship(back_populates="activity_type")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ActivityType {self.name!r} -> {self.profile_key!r}>"


#: Sessions ↔ trigger tags. Plain association table: the pairing carries no
#: data of its own, and SQLAlchemy cleans its rows up on either side's delete.
session_trigger_tag = Table(
    "session_trigger_tag",
    db.metadata,
    Column("session_id", ForeignKey("session.id"), primary_key=True),
    Column("trigger_tag_id", ForeignKey("trigger_tag.id"), primary_key=True),
)


class TriggerTag(db.Model):
    """A candidate trigger/exposure a session can be tagged with.

    Tags describe exposures in the hours before or during a recording
    (caffeine, alcohol, poor sleep, ...) so ectopy statistics can group
    sessions by exposure. The built-in vocabulary is seeded from the
    self-selected trigger menus of the I-STOP-AFib and CRAVE trials
    (``app.triggers.vocabulary``); custom tags are one row with
    ``is_builtin=False``, mirroring the ActivityType pattern. The slug is the
    stable statistics key — renaming a tag never orphans its history.
    """

    __tablename__ = "trigger_tag"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    slug: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    is_builtin: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[dt.datetime] = mapped_column(default=utcnow)

    sessions: Mapped[list[Session]] = relationship(
        secondary=session_trigger_tag, back_populates="trigger_tags"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<TriggerTag {self.slug!r}>"


class ProcessingStatus:
    """String constants for ``Session.processing_status``."""

    PENDING = "pending"
    RUNNING = "running"
    NEEDS_MAPPING = "needs_mapping"  # loader raised AmbiguousFormatError
    DONE = "done"
    ERROR = "error"

    ALL = (PENDING, RUNNING, NEEDS_MAPPING, DONE, ERROR)


#: Valid values for ``Session.body_position``. Posture is the largest
#: within-subject HRV modifier (supine vagal indices far exceed sitting or
#: standing — docs/CONTEXT_METRICS.md §3.1), so it is a first-class,
#: CHECK-constrained column. A chest accelerometer cannot distinguish sitting
#: from standing, so ACC autofill only ever writes the four lying values;
#: "sitting"/"standing" come from the user.
BODY_POSITIONS = (
    "supine",
    "prone",
    "left",
    "right",
    "sitting",
    "standing",
    "moving",
    "unknown",
)

#: Valid values for ``Session.body_position_source``.
BODY_POSITION_SOURCES = ("user", "acc")


class Session(db.Model):
    __tablename__ = "session"
    __table_args__ = (
        UniqueConstraint("file_sha256", name="uq_session_file_sha256"),
        CheckConstraint(
            "processing_status IN ('pending', 'running', 'needs_mapping', 'done', 'error')",
            name="ck_session_status",
        ),
        CheckConstraint(
            "body_position IN ('supine', 'prone', 'left', 'right', "
            "'sitting', 'standing', 'moving', 'unknown')",
            name="ck_session_body_position",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("person.id"), index=True)
    activity_type_id: Mapped[int | None] = mapped_column(
        ForeignKey("activity_type.id"), index=True
    )

    #: Start of the recording, taken from the CSV timestamps (not upload time).
    recorded_at: Mapped[dt.datetime | None] = mapped_column(default=None)
    duration_s: Mapped[float | None] = mapped_column(default=None)
    #: Derived from the timestamps, never hardcoded (real files run ~130.03 Hz).
    sampling_rate_hz: Mapped[float | None] = mapped_column(default=None)
    context_note: Mapped[str | None] = mapped_column(String(2000), default=None)

    # --- Optional structured context (docs/CONTEXT_METRICS.md) ------------
    # Real columns, not extras keys: trends and trigger statistics group and
    # filter on them (the ectopy-column precedent). All nullable — absence of
    # context is the normal case, never an error.
    #: One of BODY_POSITIONS; user-selected at upload/edit, or auto-filled
    #: from the accelerometer when a lying posture clearly dominates.
    body_position: Mapped[str | None] = mapped_column(String(12), default=None)
    #: "user" or "acc" — who set body_position. ACC autofill never overwrites
    #: a user-entered value.
    body_position_source: Mapped[str | None] = mapped_column(String(10), default=None)
    #: Alcoholic drinks in the 24 h before the recording. Dose matters
    #: (≈ +2.5 bpm sleeping HR and −3.5 ms overnight RMSSD per drink at
    #: population scale), hence a count, not a tag.
    alcohol_drinks_24h: Mapped[int | None] = mapped_column(default=None)
    #: Subjective sleep quality of the prior night, 1 (very poor) – 5 (very
    #: good). Kept separate from the objective sleep metrics deliberately —
    #: subjective–objective divergence is itself informative.
    sleep_quality_1_5: Mapped[int | None] = mapped_column(default=None)

    # --- Environment at recording time (opt-in Open-Meteo lookup) ---------
    # Window-averaged over the recording span at the configured home
    # location. NULL means not fetched (feature off, no location, fetch
    # failed, or regional data gap) — always a supported state.
    env_temp_c: Mapped[float | None] = mapped_column(default=None)
    env_apparent_temp_c: Mapped[float | None] = mapped_column(default=None)
    env_humidity_pct: Mapped[float | None] = mapped_column(default=None)
    env_pressure_hpa: Mapped[float | None] = mapped_column(default=None)
    env_pm25_ugm3: Mapped[float | None] = mapped_column(default=None)
    env_pm10_ugm3: Mapped[float | None] = mapped_column(default=None)
    env_ozone_ugm3: Mapped[float | None] = mapped_column(default=None)
    env_no2_ugm3: Mapped[float | None] = mapped_column(default=None)
    env_aqi: Mapped[float | None] = mapped_column(default=None)
    env_daylight_h: Mapped[float | None] = mapped_column(default=None)
    #: Which endpoint produced the values ("open-meteo-forecast" /
    #: "open-meteo-archive").
    env_source: Mapped[str | None] = mapped_column(String(24), default=None)
    #: When the lookup ran; reprocessing does not refetch once set.
    env_fetched_at: Mapped[dt.datetime | None] = mapped_column(default=None)

    original_filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(500))
    file_sha256: Mapped[str] = mapped_column(String(64), index=True)

    # --- Optional accelerometer companion file (sleep sessions) -----------
    #: The ACC export is a second, optional upload: movement feeds sleep
    #: staging (wake detection) and the report's movement trace. Indexed but
    #: NOT unique — re-uploading the same ACC with a different ECG is legal.
    acc_original_filename: Mapped[str | None] = mapped_column(String(255), default=None)
    acc_stored_path: Mapped[str | None] = mapped_column(String(500), default=None)
    acc_file_sha256: Mapped[str | None] = mapped_column(
        String(64), index=True, default=None
    )

    processing_status: Mapped[str] = mapped_column(
        String(20), default=ProcessingStatus.PENDING
    )
    error_message: Mapped[str | None] = mapped_column(String(2000), default=None)
    #: User answers from the format-mapping form (loader FormatOverrides),
    #: kept so reprocessing applies them again.
    format_overrides: Mapped[dict | None] = mapped_column(JSON, default=None)
    #: Which engine actually produced the analysis ("neurokit2" / "biosppy").
    engine_used: Mapped[str | None] = mapped_column(String(40), default=None)

    # --- Provenance of the loader's auto-detection (surfaced in the report) ---
    #: "mV" or "uV" — which unit branch the amplitude auto-detect took.
    ecg_unit_detected: Mapped[str | None] = mapped_column(String(8), default=None)
    #: "unix" or "polar2000" — which epoch interpretation was accepted.
    epoch_detected: Mapped[str | None] = mapped_column(String(16), default=None)

    # --- Quality headline numbers (badges + comparison guards) ---
    beats_corrected_pct: Mapped[float | None] = mapped_column(default=None)
    #: True when beats_corrected_pct > 5 %, per Kubios guidance. The badge
    #: follows the session everywhere it appears.
    reduced_confidence: Mapped[bool] = mapped_column(default=False)
    #: Excluded time is a headline number, not a footnote.
    excluded_s: Mapped[float | None] = mapped_column(default=None)
    analysed_s: Mapped[float | None] = mapped_column(default=None)

    created_at: Mapped[dt.datetime] = mapped_column(default=utcnow)

    person: Mapped[Person] = relationship(back_populates="sessions")
    activity_type: Mapped[ActivityType | None] = relationship(back_populates="sessions")
    metrics: Mapped[Metrics | None] = relationship(
        back_populates="session", cascade="all, delete-orphan", uselist=False
    )
    flags: Mapped[list[Flag]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    excluded_segments: Mapped[list[ExcludedSegment]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    trigger_tags: Mapped[list[TriggerTag]] = relationship(
        secondary=session_trigger_tag,
        back_populates="sessions",
        order_by="TriggerTag.name",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Session {self.id} person={self.person_id} status={self.processing_status}>"


class Metrics(db.Model):
    """One row of HRV metrics per session (1996 Task Force definitions).

    Deliberately absent: QRS width, QT, QTc, PR interval, ECG axis. At ~130 Hz
    one sample is 7.7 ms and interval delineation returns quantisation
    artifacts, not physiology. These are never computed or stored.
    """

    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("session.id"), unique=True, index=True
    )

    n_beats: Mapped[int | None] = mapped_column(default=None)

    # Time domain
    mean_rr_ms: Mapped[float | None] = mapped_column(default=None)
    mean_hr_bpm: Mapped[float | None] = mapped_column(default=None)
    #: Whole-record SDNN is trend-inclusive and length-dependent; per-window
    #: SDNN values live in ``extras["sdnn_per_window_ms"]`` and are always
    #: reported alongside it.
    sdnn_ms: Mapped[float | None] = mapped_column(default=None)
    rmssd_ms: Mapped[float | None] = mapped_column(default=None)
    pnn50_pct: Mapped[float | None] = mapped_column(default=None)

    # Frequency domain (VLF 0.0033-0.04, LF 0.04-0.15, HF 0.15-0.4 Hz)
    vlf_power_ms2: Mapped[float | None] = mapped_column(default=None)
    lf_power_ms2: Mapped[float | None] = mapped_column(default=None)
    hf_power_ms2: Mapped[float | None] = mapped_column(default=None)
    #: Engine-dependent (44 % spread observed between engines on identical
    #: input); descriptive only, never used for scoring or flagging.
    lf_hf_ratio: Mapped[float | None] = mapped_column(default=None)

    # Nonlinear
    sd1_ms: Mapped[float | None] = mapped_column(default=None)
    sd2_ms: Mapped[float | None] = mapped_column(default=None)
    sd1_sd2_ratio: Mapped[float | None] = mapped_column(default=None)
    sample_entropy: Mapped[float | None] = mapped_column(default=None)
    dfa_alpha1: Mapped[float | None] = mapped_column(default=None)

    # Ectopy burden (confirmed ectopic beats; app.pipeline.events).
    # NULL on all of these means the session was processed before the events
    # pipeline existed and needs a reprocess to populate them. Real columns,
    # not extras keys, because trigger statistics group and filter on them.
    ectopy_beats_n: Mapped[int | None] = mapped_column(default=None)
    #: Confirmed ectopics per hour of *analysed* time.
    ectopy_per_hour: Mapped[float | None] = mapped_column(default=None)
    ectopy_pct_beats: Mapped[float | None] = mapped_column(default=None)
    single_n: Mapped[int | None] = mapped_column(default=None)
    couplet_n: Mapped[int | None] = mapped_column(default=None)
    run_n: Mapped[int | None] = mapped_column(default=None)
    longest_run_beats: Mapped[int | None] = mapped_column(default=None)
    bigeminy_episode_n: Mapped[int | None] = mapped_column(default=None)
    trigeminy_episode_n: Mapped[int | None] = mapped_column(default=None)

    # Sleep architecture (sleep sessions only; app.sleep). NULL everywhere
    # else. Real columns, not extras keys, because cross-night trends group
    # and filter on them (the ectopy-column precedent). Values come from the
    # primary staging engine recorded in ``sleep_engine``; per-engine detail
    # lives in extras["sleep"].
    tst_min: Mapped[float | None] = mapped_column(default=None)
    sleep_efficiency_pct: Mapped[float | None] = mapped_column(default=None)
    sol_min: Mapped[float | None] = mapped_column(default=None)
    waso_min: Mapped[float | None] = mapped_column(default=None)
    #: Stage minutes are vocabulary-dependent: a 3-class engine fills none of
    #: light/deep (it cannot tell them apart), a 4/5-class engine fills all.
    light_min: Mapped[float | None] = mapped_column(default=None)
    deep_min: Mapped[float | None] = mapped_column(default=None)
    rem_min: Mapped[float | None] = mapped_column(default=None)
    awakenings_n: Mapped[int | None] = mapped_column(default=None)
    #: Which staging engine produced the columns above.
    sleep_engine: Mapped[str | None] = mapped_column(String(40), default=None)

    # ECG-derived respiration (app.pipeline.respiration). An ESTIMATE from
    # heartbeat and R-wave-amplitude modulation, never measured airflow
    # (validated on this hardware: Schaffarczyk 2022, Sensors 22:7156 —
    # r = 0.85 vs gas exchange, degrading at high intensity). Real columns
    # because nightly respiratory-rate baselines trend across sessions.
    resp_rate_median_brpm: Mapped[float | None] = mapped_column(default=None)
    resp_rate_p5_brpm: Mapped[float | None] = mapped_column(default=None)
    resp_rate_p95_brpm: Mapped[float | None] = mapped_column(default=None)

    #: Activity-specific derived metrics and per-window series.
    extras: Mapped[dict | None] = mapped_column(JSON, default=None)

    session: Mapped[Session] = relationship(back_populates="metrics")


class Flag(db.Model):
    """A rule-based screening flag.

    The disclaimer is part of the flag row (NOT NULL, non-empty CHECK) so it
    cannot be separated from the flag by a template change. "No flags raised"
    must never be rendered as clinical clearance.
    """

    __tablename__ = "flag"
    __table_args__ = (
        CheckConstraint("length(trim(disclaimer)) > 0", name="ck_flag_disclaimer_nonempty"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("session.id"), index=True)
    kind: Mapped[str] = mapped_column(String(40))
    severity: Mapped[str] = mapped_column(String(20))
    description: Mapped[str] = mapped_column(String(2000))
    #: The measurements that produced the flag.
    evidence: Mapped[dict] = mapped_column(JSON)
    #: The threshold definition that was crossed.
    threshold: Mapped[dict] = mapped_column(JSON)
    disclaimer: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[dt.datetime] = mapped_column(default=utcnow)

    session: Mapped[Session] = relationship(back_populates="flags")


class ExcludedSegment(db.Model):
    """A stretch of recording excluded from analysis — always reported.

    Bad signal is excluded and surfaced, never silently interpolated.
    """

    __tablename__ = "excluded_segment"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("session.id"), index=True)
    start_s: Mapped[float]
    end_s: Mapped[float]
    reason: Mapped[str] = mapped_column(String(200))

    session: Mapped[Session] = relationship(back_populates="excluded_segments")


class Annotation(db.Model):
    """A dated free-text note by the user, shown in the decision-log timeline."""

    __tablename__ = "annotation"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("person.id"), index=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("session.id"), default=None)
    text: Mapped[str] = mapped_column(String(4000))
    created_at: Mapped[dt.datetime] = mapped_column(default=utcnow)

    person: Mapped[Person] = relationship(back_populates="annotations")


class PolarAccount(db.Model):
    """A linked Polar Flow account, one per person (opt-in AccessLink sync).

    The access token is a long-lived bearer credential for the wearer's Polar
    account: AccessLink tokens do not expire unless revoked, and there is no
    refresh token. It is stored here in plaintext because the app's threat
    model is a single local machine and the token must survive restarts —
    which is exactly why the person page offers an Unlink button and the docs
    name https://account.polar.com as the way to revoke it at the source.

    ``polar_user_id`` is Polar's own identifier (``x_user_id`` from the token
    response); ``member_id`` is the identifier we chose when registering.
    """

    __tablename__ = "polar_account"

    id: Mapped[int] = mapped_column(primary_key=True)
    #: Unique: one Flow account per person. Re-linking replaces the row.
    person_id: Mapped[int] = mapped_column(
        ForeignKey("person.id"), unique=True, index=True
    )
    polar_user_id: Mapped[str] = mapped_column(String(64), index=True)
    access_token: Mapped[str] = mapped_column(String(255))
    member_id: Mapped[str] = mapped_column(String(120))
    linked_at: Mapped[dt.datetime] = mapped_column(default=utcnow)
    last_sync_at: Mapped[dt.datetime | None] = mapped_column(default=None)
    #: Human-readable outcome of the last sync, including honest failures
    #: ("rate limited", "re-authorisation needed"). Shown on the person page.
    last_sync_note: Mapped[str | None] = mapped_column(String(500), default=None)

    person: Mapped[Person] = relationship(back_populates="polar_account")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PolarAccount person={self.person_id} polar_user={self.polar_user_id!r}>"


class FlowNight(db.Model):
    """One night of Polar Flow data for one person, keyed by Polar's result date.

    Scalars are real columns rather than an extras blob because trends and the
    trigger statistics group and filter on them — the same reasoning as the
    ectopy and environment columns on ``Session``. The three sample maps are
    JSON: they are at most a few hundred points, are never analysed
    sample-by-sample, and so do not warrant the ``.npz`` treatment that ECG
    arrays get.

    Every field is nullable. A night with sleep but no Nightly Recharge (or
    the reverse) is a normal outcome, not an error: Polar computes them from
    different windows and either can be missing.
    """

    __tablename__ = "flow_night"
    __table_args__ = (
        UniqueConstraint("person_id", "date", name="uq_flow_night_person_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("person.id"), index=True)
    #: Polar's "result date" for the night — the key both endpoints agree on.
    date: Mapped[dt.date] = mapped_column(index=True)
    #: Which device produced the night. The only way to tell a Loop night from
    #: a watch night once several devices share one Flow account.
    source_device_id: Mapped[str | None] = mapped_column(String(64), default=None)

    # --- Sleep Plus Stages (GET /v3/users/sleep/{date}) -------------------
    sleep_start: Mapped[dt.datetime | None] = mapped_column(default=None)
    sleep_end: Mapped[dt.datetime | None] = mapped_column(default=None)
    #: Stage durations in seconds, as Polar reports them.
    light_sleep_s: Mapped[int | None] = mapped_column(default=None)
    deep_sleep_s: Mapped[int | None] = mapped_column(default=None)
    rem_sleep_s: Mapped[int | None] = mapped_column(default=None)
    unrecognized_sleep_s: Mapped[int | None] = mapped_column(default=None)
    total_interruption_s: Mapped[int | None] = mapped_column(default=None)
    #: Polar's own 1–100 composite. A proprietary score, displayed as-is and
    #: never fed into screening.
    sleep_score: Mapped[int | None] = mapped_column(default=None)
    sleep_charge: Mapped[int | None] = mapped_column(default=None)
    continuity: Mapped[float | None] = mapped_column(default=None)
    continuity_class: Mapped[int | None] = mapped_column(default=None)
    sleep_cycles: Mapped[int | None] = mapped_column(default=None)

    # --- Nightly Recharge (GET /v3/users/nightly-recharge/{date}) ---------
    # All averaged over a 4-hour window starting 30 min after sleep onset.
    hr_avg_bpm: Mapped[int | None] = mapped_column(default=None)
    beat_to_beat_avg_ms: Mapped[int | None] = mapped_column(default=None)
    #: Polar documents this explicitly as RMSSD in milliseconds. It is NOT
    #: comparable to a session's RMSSD: PPG-derived, wrist-worn, four hours of
    #: sleep, versus ECG R-peaks over minutes of controlled posture.
    hrv_rmssd_ms: Mapped[int | None] = mapped_column(default=None)
    breathing_rate_avg: Mapped[float | None] = mapped_column(default=None)
    #: 1 (very poor) – 6 (very good).
    nightly_recharge_status: Mapped[int | None] = mapped_column(default=None)
    #: −10.0 … +10.0, relative to the wearer's own trailing 28 days.
    ans_charge: Mapped[float | None] = mapped_column(default=None)
    #: 1 (much below usual) – 5 (much above usual).
    ans_charge_status: Mapped[int | None] = mapped_column(default=None)

    # --- Sample series (small, display-only) ------------------------------
    #: {"00:41": 14, ...} — 5-minute HRV averages in ms.
    hrv_samples: Mapped[dict | None] = mapped_column(JSON, default=None)
    #: {"00:39": 13.4, ...} — 5-minute breathing-rate averages.
    breathing_samples: Mapped[dict | None] = mapped_column(JSON, default=None)
    #: [{"sample_time": "00:02:08", "heart_rate": 63}, ...] for the day.
    hr_samples: Mapped[list | None] = mapped_column(JSON, default=None)

    fetched_at: Mapped[dt.datetime] = mapped_column(default=utcnow)

    person: Mapped[Person] = relationship(back_populates="flow_nights")

    @property
    def total_sleep_s(self) -> int | None:
        """Summed stage durations, or None when no stage data arrived."""
        parts = [
            self.light_sleep_s,
            self.deep_sleep_s,
            self.rem_sleep_s,
            self.unrecognized_sleep_s,
        ]
        present = [p for p in parts if p is not None]
        return sum(present) if present else None

    def has_sleep(self) -> bool:
        return self.sleep_start is not None or self.sleep_score is not None

    def has_recharge(self) -> bool:
        return self.hrv_rmssd_ms is not None or self.nightly_recharge_status is not None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<FlowNight person={self.person_id} {self.date}>"
