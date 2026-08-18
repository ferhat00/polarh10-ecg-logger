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
    SessionObservation,
    assemble_observations,
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
