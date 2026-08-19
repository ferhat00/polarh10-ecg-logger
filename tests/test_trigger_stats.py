"""Trigger statistics: guards, NB fit, fallback, assembly, hour profile."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from flask import Flask

from app.extensions import db
from app.models import Metrics, Person, ProcessingStatus, Session, TriggerTag
from app.triggers.stats import (
    MIN_ANALYSED_S,
    MIN_ENV_OBSERVATIONS,
    OUTCOMES,
    SessionObservation,
    assemble_observations,
    env_associations,
    fit_tag_effect,
    hour_of_day_profile,
    rates_by_tag,
)

BASE_T = dt.datetime(2026, 6, 1, 8, 0, tzinfo=dt.UTC)


def _obs(
    i: int,
    count: int,
    tags: frozenset[str] = frozenset(),
    hours: float = 1.0,
    hour_of_day: int | None = None,
    comparison_key: str | None = "sitting",
    ln_rmssd: float | None = None,
    resting_hr_bpm: float | None = None,
    env_temp_c: float | None = None,
    env_pm25_ugm3: float | None = None,
) -> SessionObservation:
    recorded = BASE_T + dt.timedelta(days=i)
    if hour_of_day is not None:
        recorded = recorded.replace(hour=hour_of_day)
    return SessionObservation(
        session_id=i,
        recorded_at=recorded,
        analysed_hours=hours,
        duration_s=hours * 3600.0,
        ectopy_count=count,
        ectopy_per_hour=count / hours,
        comparison_key=comparison_key,
        tags=tags,
        reduced_confidence=False,
        ln_rmssd=ln_rmssd,
        rmssd_ms=float(np.exp(ln_rmssd)) if ln_rmssd is not None else None,
        resting_hr_bpm=resting_hr_bpm,
        env_temp_c=env_temp_c,
        env_pm25_ugm3=env_pm25_ugm3,
    )


def _nb_counts(rng: np.random.Generator, mu: float, alpha: float, n: int) -> np.ndarray:
    """Overdispersed (NB2) draws with mean mu and Var = mu + alpha mu^2."""
    r = 1.0 / alpha
    p = r / (r + mu)
    return rng.negative_binomial(r, p, n)


def _simulated(rate_ratio: float = 3.0, n_per_arm: int = 20) -> list[SessionObservation]:
    rng = np.random.default_rng(42)
    base = 10.0
    obs = []
    for i in range(n_per_arm):
        obs.append(
            _obs(
                i,
                int(_nb_counts(rng, base * rate_ratio, 0.3, 1)[0]),
                tags=frozenset({"caffeine"}),
                hour_of_day=8 + (i % 12),
            )
        )
    for i in range(n_per_arm):
        obs.append(
            _obs(
                n_per_arm + i,
                int(_nb_counts(rng, base, 0.3, 1)[0]),
                hour_of_day=8 + (i % 12),
            )
        )
    return obs


class TestGuards:
    def test_too_few_tagged_sessions(self) -> None:
        obs = [_obs(i, 5, frozenset({"caffeine"}) if i < 2 else frozenset()) for i in range(12)]
        effect = fit_tag_effect(obs, "caffeine", "Caffeine")
        assert effect.status == "insufficient_data"
        assert effect.rate_ratio is None
        assert "2" in effect.detail  # says how many it has
        # Descriptive rates still reported.
        assert effect.tagged_rate_per_hour == 5.0

    def test_too_few_events(self) -> None:
        obs = [
            _obs(i, 1 if i == 0 else 0, frozenset({"caffeine"}) if i < 6 else frozenset())
            for i in range(12)
        ]
        effect = fit_tag_effect(obs, "caffeine", "Caffeine")
        assert effect.status == "insufficient_data"
        assert "count model" in effect.detail

    def test_zero_arm_yields_no_ratio(self) -> None:
        obs = [
            _obs(i, 12 if i < 6 else 0, frozenset({"caffeine"}) if i < 6 else frozenset())
            for i in range(12)
        ]
        effect = fit_tag_effect(obs, "caffeine", "Caffeine")
        assert effect.status == "no_events"
        assert effect.rate_ratio is None
        assert effect.untagged_rate_per_hour == 0.0


class TestModel:
    def test_nb_recovers_simulated_rate_ratio(self) -> None:
        effect = fit_tag_effect(_simulated(), "caffeine", "Caffeine")
        assert effect.status == "ok"
        assert effect.rate_ratio is not None
        assert 1.5 < effect.rate_ratio < 6.0
        assert effect.ci_low < 3.0 < effect.ci_high

    def test_null_effect_interval_covers_one(self) -> None:
        effect = fit_tag_effect(_simulated(rate_ratio=1.0), "caffeine", "Caffeine")
        assert effect.status in ("ok", "poisson_fallback")
        assert effect.ci_low < 1.0 < effect.ci_high

    def test_poisson_fallback_when_nb_fit_raises(self, monkeypatch) -> None:
        import statsmodels.api as sm

        def boom(*args, **kwargs):
            raise ValueError("no convergence")

        monkeypatch.setattr(sm.NegativeBinomial, "fit", boom)
        effect = fit_tag_effect(_simulated(), "caffeine", "Caffeine")
        assert effect.status == "poisson_fallback"
        assert effect.rate_ratio is not None
        assert np.isfinite(effect.ci_low) and np.isfinite(effect.ci_high)
        assert "Poisson" in effect.detail

    def test_model_failed_when_everything_raises(self, monkeypatch) -> None:
        import statsmodels.api as sm

        def boom(*args, **kwargs):
            raise ValueError("nope")

        monkeypatch.setattr(sm.NegativeBinomial, "fit", boom)
        monkeypatch.setattr(sm.GLM, "fit", boom)
        effect = fit_tag_effect(_simulated(), "caffeine", "Caffeine")
        assert effect.status == "model_failed"
        assert effect.rate_ratio is None
        assert effect.tagged_rate_per_hour is not None

    def test_mixed_activity_covariate_dropped_when_sparse(self) -> None:
        obs = _simulated()
        # One lone session in a second activity group: covariate must drop.
        obs[0] = _obs(
            0, 25, frozenset({"caffeine"}), hour_of_day=9, comparison_key="running"
        )
        notes: list[str] = []
        effect = fit_tag_effect(obs, "caffeine", "Caffeine", notes)
        assert effect.status in ("ok", "poisson_fallback")
        assert any("activity covariate dropped" in n for n in notes)


def _gaussian_obs(
    n_per_arm: int = 15,
    effect_pct: float = -20.0,
    seed: int = 7,
    outcome: str = "ln_rmssd",
) -> list[SessionObservation]:
    """Tagged sessions carry a known shift on the modelling scale."""
    rng = np.random.default_rng(seed)
    obs: list[SessionObservation] = []
    for i in range(n_per_arm * 2):
        tagged = i < n_per_arm
        if outcome == "ln_rmssd":
            value = np.log(45.0) + rng.normal(0, 0.10)
            if tagged:
                value += np.log(1 + effect_pct / 100.0)
            obs.append(
                _obs(
                    i,
                    0,
                    tags=frozenset({"alcohol"}) if tagged else frozenset(),
                    hour_of_day=8 + (i % 12),
                    ln_rmssd=float(value),
                )
            )
        else:
            value = 55.0 + rng.normal(0, 2.0) + (effect_pct if tagged else 0.0)
            obs.append(
                _obs(
                    i,
                    0,
                    tags=frozenset({"alcohol"}) if tagged else frozenset(),
                    hour_of_day=8 + (i % 12),
                    resting_hr_bpm=float(value),
                )
            )
    return obs


class TestGaussianOutcomes:
    def test_ols_recovers_rmssd_drop(self) -> None:
        effect = fit_tag_effect(
            _gaussian_obs(effect_pct=-20.0),
            "alcohol",
            "Alcohol",
            outcome=OUTCOMES["ln_rmssd"],
        )
        assert effect.status == "ok"
        assert effect.effect_kind == "pct_change"
        assert effect.rate_ratio == pytest.approx(-20.0, abs=8.0)
        assert effect.ci_low < -20.0 < effect.ci_high or effect.ci_low < effect.rate_ratio
        # Group summaries are geometric means in ms.
        assert effect.tagged_rate_per_hour == pytest.approx(36.0, abs=4.0)
        assert effect.untagged_rate_per_hour == pytest.approx(45.0, abs=4.0)
        assert "linear model" in effect.detail

    def test_ols_recovers_resting_hr_delta(self) -> None:
        effect = fit_tag_effect(
            _gaussian_obs(effect_pct=3.0, outcome="resting_hr"),
            "alcohol",
            "Alcohol",
            outcome=OUTCOMES["resting_hr"],
        )
        assert effect.status == "ok"
        assert effect.effect_kind == "delta"
        assert effect.rate_ratio == pytest.approx(3.0, abs=2.0)

    def test_min_sessions_guard_applies(self) -> None:
        obs = _gaussian_obs(n_per_arm=3)
        effect = fit_tag_effect(
            obs, "alcohol", "Alcohol", outcome=OUTCOMES["ln_rmssd"]
        )
        assert effect.status == "insufficient_data"

    def test_variance_guard(self) -> None:
        obs = [
            _obs(
                i,
                0,
                tags=frozenset({"alcohol"}) if i < 6 else frozenset(),
                ln_rmssd=np.log(40.0),
            )
            for i in range(12)
        ]
        effect = fit_tag_effect(
            obs, "alcohol", "Alcohol", outcome=OUTCOMES["ln_rmssd"]
        )
        assert effect.status == "insufficient_data"
        assert "no variation" in effect.detail


class TestEnvAssociations:
    def test_below_threshold_is_omitted(self) -> None:
        obs = [
            _obs(i, i, env_temp_c=15.0 + i)
            for i in range(MIN_ENV_OBSERVATIONS - 1)
        ]
        assert env_associations(obs, OUTCOMES["ectopy"]) == []

    def test_monotonic_relation_yields_rho_and_tertiles(self) -> None:
        n = MIN_ENV_OBSERVATIONS + 3
        obs = [
            _obs(i, i, hours=1.0, env_temp_c=10.0 + i, ln_rmssd=np.log(60.0 - i))
            for i in range(n)
        ]
        assocs = env_associations(obs, OUTCOMES["ectopy"])
        assert len(assocs) == 1  # PM2.5 absent everywhere → only temperature
        a = assocs[0]
        assert a.var_key == "env_temp_c"
        assert a.n == n
        assert a.rho == pytest.approx(1.0)  # count rises with temperature
        assert len(a.tertiles) == 3
        assert sum(t[1] for t in a.tertiles) == n

        # RMSSD outcome uses the display scale (ms), falling with temp.
        assocs = env_associations(obs, OUTCOMES["ln_rmssd"])
        assert assocs[0].rho == pytest.approx(-1.0)


class TestAssembly:
    @pytest.fixture()
    def person(self, app: Flask) -> Person:
        p = Person(name="Stats", slug="stats")
        db.session.add(p)
        db.session.commit()
        return p

    def _session(
        self,
        person: Person,
        sha: str,
        status: str = ProcessingStatus.DONE,
        analysed_s: float = 1800.0,
        ectopy_n: int | None = 3,
        with_metrics: bool = True,
    ) -> Session:
        s = Session(
            person_id=person.id,
            original_filename="x.csv",
            stored_path=f"/tmp/{sha}.csv",
            file_sha256=sha,
            processing_status=status,
            analysed_s=analysed_s,
            duration_s=analysed_s,
            recorded_at=BASE_T,
        )
        db.session.add(s)
        db.session.flush()
        if with_metrics:
            db.session.add(
                Metrics(
                    session_id=s.id,
                    ectopy_beats_n=ectopy_n,
                    ectopy_per_hour=(
                        ectopy_n / (analysed_s / 3600.0) if ectopy_n is not None else None
                    ),
                )
            )
        db.session.commit()
        return s

    def test_accounting(self, app: Flask, person: Person) -> None:
        usable = self._session(person, "a" * 64)
        awaiting = self._session(person, "b" * 64, ectopy_n=None)
        self._session(person, "c" * 64, analysed_s=MIN_ANALYSED_S - 1)
        self._session(person, "d" * 64, status=ProcessingStatus.PENDING, with_metrics=False)

        tag = TriggerTag(name="Caffeine", slug="caffeine", is_builtin=True)
        db.session.add(tag)
        usable.trigger_tags = [tag]
        db.session.commit()

        observations, notes = assemble_observations(person)
        assert [o.session_id for o in observations] == [usable.id]
        assert observations[0].tags == frozenset({"caffeine"})
        assert observations[0].ectopy_count == 3
        assert notes.n_awaiting_reprocess == 1
        assert notes.awaiting_ids == [awaiting.id]
        assert notes.n_too_short == 1
        assert notes.n_not_done == 1

    def test_gaussian_and_env_fields_assembled(self, app: Flask, person: Person) -> None:
        s = self._session(person, "e" * 64)
        s.env_temp_c = 21.5
        s.env_pm25_ugm3 = 9.0
        s.metrics.rmssd_ms = 42.0
        s.metrics.extras = {"resting_hr_bpm": 55.0}
        db.session.commit()

        observations, _ = assemble_observations(person)
        o = observations[0]
        assert o.rmssd_ms == pytest.approx(42.0)
        assert o.ln_rmssd == pytest.approx(np.log(42.0))
        assert o.resting_hr_bpm == pytest.approx(55.0)
        assert o.env_temp_c == pytest.approx(21.5)
        assert o.env_pm25_ugm3 == pytest.approx(9.0)


class TestDescriptive:
    def test_rates_by_tag(self) -> None:
        obs = [
            _obs(0, 4, frozenset({"caffeine"})),
            _obs(1, 0),
            _obs(2, 2, frozenset({"caffeine", "stress"})),
        ]
        rates = rates_by_tag(obs)
        assert rates["caffeine"] == ([4.0, 2.0], [0.0])
        assert rates["stress"] == ([2.0], [4.0, 0.0])

    def test_hour_profile_allocates_exposure_and_events(self) -> None:
        # Two hours of recording starting 08:00; events at +10 min and +90 min.
        obs = [_obs(0, 2, hours=2.0, hour_of_day=8)]
        profile = hour_of_day_profile(
            obs, {0: np.array([600.0, 5400.0])}
        )
        assert profile.n_sessions_used == 1
        assert profile.exposure_hours[8] == pytest.approx(1.0)
        assert profile.exposure_hours[9] == pytest.approx(1.0)
        assert profile.n_events[8] == 1
        assert profile.n_events[9] == 1
        assert profile.events_per_hour[8] == pytest.approx(1.0)
        assert profile.events_per_hour[3] is None  # no exposure that hour

    def test_hour_profile_skips_sessions_without_cache(self) -> None:
        obs = [_obs(0, 2), _obs(1, 1)]
        profile = hour_of_day_profile(obs, {0: np.array([])})
        assert profile.n_sessions_used == 1
        assert profile.n_sessions_skipped == 1
