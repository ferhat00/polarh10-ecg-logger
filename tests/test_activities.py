"""Activity-profile tests: registry, suppression rules, derived metrics."""

from __future__ import annotations

import numpy as np
import pytest
from flask import Flask

from app.activities import all_profiles, get_profile, resolve_profile
from app.activities import metrics as m
from app.activities.base import (
    ActivityInputs,
    MetricFamily,
    PersonContext,
    apply_suppressions,
)
from app.activities.profiles.running import HRV_SUPPRESSION_HRR_FRACTION
from app.activities.seed import ensure_builtin_activity_types
from app.extensions import db
from app.models import ActivityType
from app.pipeline.hrv import HRVResult, compute_hrv
from app.pipeline.quality import QualityResult, QualityWindow
from app.pipeline.rr import RRSeries

BUILTIN_KEYS = {
    "supine",
    "sitting",
    "standing",
    "walking",
    "running",
    "cycling",
    "swimming",
    "recovery",
}


def _series(rr_ms: np.ndarray) -> RRSeries:
    t = np.cumsum(rr_ms) / 1000.0
    return RRSeries(
        rr_ms=rr_ms,
        t_s=t,
        discontinuity=np.zeros(len(rr_ms), dtype=bool),
        n_dropped_excluded=0,
        n_dropped_ceiling=0,
        n_dropped_floor=0,
    )


def _steady(bpm: float, duration_s: float = 400.0) -> RRSeries:
    rr = 60000.0 / bpm
    n = int(duration_s * 1000 / rr)
    rng = np.random.default_rng(4)
    t = np.arange(n) * rr / 1000.0
    return _series(rr + 0.02 * rr * np.sin(2 * np.pi * 0.1 * t) + rng.normal(0, 5.0, n))


def _quality(duration_s: float = 400.0, excluded_s: float = 0.0) -> QualityResult:
    windows = []
    t = 0.0
    while t < duration_s:
        excluded = t < excluded_s
        windows.append(
            QualityWindow(
                start_s=t,
                end_s=min(t + 5.0, duration_s),
                sqi_mean=0.95,
                wander_rms_mv=0.01,
                excluded=excluded,
                reasons=("flat/dead signal",) if excluded else (),
            )
        )
        t += 5.0
    segments = [(0.0, excluded_s, "flat/dead signal")] if excluded_s else []
    return QualityResult(
        windows=windows,
        excluded_segments=segments,
        sqi=np.array([]),
        wander_mv=np.array([]),
        excluded_total_s=excluded_s,
        analysed_total_s=duration_s - excluded_s,
    )


def _inputs(rr: RRSeries, excluded_s: float = 0.0, markers=None) -> ActivityInputs:
    duration = float(rr.t_s[-1]) + excluded_s
    return ActivityInputs(
        rr=rr,
        hrv=compute_hrv(rr),
        quality=_quality(duration, excluded_s),
        markers=markers or [],
        duration_s=duration,
    )


class TestRegistry:
    def test_all_eight_builtins_registered(self) -> None:
        assert set(all_profiles()) == BUILTIN_KEYS

    def test_only_supine_is_reference_baseline(self) -> None:
        refs = [k for k, p in all_profiles().items() if p.is_reference_baseline]
        assert refs == ["supine"]

    def test_every_profile_fully_declared(self) -> None:
        for key, p in all_profiles().items():
            assert p.key == key
            assert p.display_name and p.description
            assert p.expected_hr_range[0] < p.expected_hr_range[1]
            assert p.expected_motion in ("minimal", "low", "moderate", "high", "severe")
            assert p.comparison_key
            assert p.interpretation_notes

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(KeyError, match="hovercraft"):
            get_profile("hovercraft")


class TestIntensitySuppression:
    """Running/cycling suppress the whole HRV suite above ~70 % HR reserve."""

    CTX = PersonContext(max_hr_bpm=190, resting_hr_bpm=50)

    def test_high_intensity_suppresses_hrv(self) -> None:
        # 165 bpm with max 190/rest 50 → 82 % HRR.
        analysis = get_profile("running").analyze(_inputs(_steady(165.0)), self.CTX)
        suppressed = {s.family for s in analysis.suppressions}
        assert suppressed == set(MetricFamily.HRV_ALL)
        assert any("heart-rate" in s.reason and "reserve" in s.reason
                   for s in analysis.suppressions)

    def test_easy_run_keeps_hrv(self) -> None:
        # 120 bpm → 50 % HRR: below the suppression level.
        ctx_frac = self.CTX.hr_reserve_fraction(120.0)
        assert ctx_frac < HRV_SUPPRESSION_HRR_FRACTION
        analysis = get_profile("running").analyze(_inputs(_steady(120.0)), self.CTX)
        assert analysis.suppressions == []
        # …but the frequency caution still applies (stride aliasing).
        assert any(c.family == MetricFamily.FREQUENCY for c in analysis.cautions)

    def test_cycling_shares_the_rule_and_warns_about_cadence(self) -> None:
        analysis = get_profile("cycling").analyze(_inputs(_steady(170.0)), self.CTX)
        assert {s.family for s in analysis.suppressions} == set(MetricFamily.HRV_ALL)
        cautions = " ".join(c.reason for c in analysis.cautions)
        assert "cadence" in cautions.lower()

    def test_suppression_actually_blanks_values(self) -> None:
        inputs = _inputs(_steady(165.0))
        analysis = get_profile("running").analyze(inputs, self.CTX)
        censored, notes = apply_suppressions(inputs.hrv, analysis.suppressions)
        assert censored.rmssd_ms is None
        assert censored.sdnn_ms is None
        assert censored.lf_power_ms2 is None
        assert censored.sample_entropy is None
        assert censored.sdnn_per_window_ms == []
        # Rate statistics survive — zones and drift still need them.
        assert censored.mean_hr_bpm is not None
        assert notes and all("suppressed" in n for n in notes)


class TestSwimmingExclusionGate:
    def test_over_30pct_excluded_not_analysable(self) -> None:
        rr = _steady(120.0, duration_s=280.0)
        inputs = _inputs(rr, excluded_s=140.0)  # 33 % of 420 s total
        analysis = get_profile("swimming").analyze(inputs, PersonContext())
        assert analysis.not_analysable is True
        assert "isn't analysable" in (analysis.not_analysable_reason or "")
        assert {s.family for s in analysis.suppressions} == set(MetricFamily.HRV_ALL)
        assert analysis.extras == {}

    def test_under_30pct_analysable_with_extras(self) -> None:
        rr = _steady(120.0, duration_s=380.0)
        inputs = _inputs(rr, excluded_s=40.0)  # ~10 %
        analysis = get_profile("swimming").analyze(inputs, PersonContext())
        assert analysis.not_analysable is False
        assert "hr_zones_fraction" in analysis.extras


class TestStandingAndRecoveryCautions:
    def test_standing_warns_low_rmssd_is_expected(self) -> None:
        analysis = get_profile("standing").analyze(_inputs(_steady(75.0)), PersonContext())
        reasons = " ".join(c.reason for c in analysis.cautions)
        assert "expected" in reasons
        assert "Mayer" in reasons

    def test_recovery_warns_not_comparable_to_baseline(self) -> None:
        analysis = get_profile("recovery").analyze(_inputs(_steady(90.0)), PersonContext())
        reasons = " ".join(c.reason for c in analysis.cautions)
        assert "not comparable" in reasons

    def test_walking_frequency_caution(self) -> None:
        analysis = get_profile("walking").analyze(_inputs(_steady(90.0)), PersonContext())
        assert any(
            c.family == MetricFamily.FREQUENCY and "alias" in c.reason
            for c in analysis.cautions
        )


class TestActivityHRLimits:
    def test_running_limits_scale_with_max_hr(self) -> None:
        ctx = PersonContext(max_hr_bpm=190)
        limits = get_profile("running").hr_limits(ctx)
        assert limits.tachy_bpm == 190.0
        assert limits.context == "running"

    def test_supine_uses_rest_limits_with_athlete(self) -> None:
        limits = get_profile("supine").hr_limits(PersonContext(athlete_baseline=True))
        assert limits.brady_bpm == 45.0


class TestPersonContext:
    def test_max_hr_fallback_chain(self) -> None:
        assert PersonContext(max_hr_bpm=188).effective_max_hr()[0] == 188.0
        est, src = PersonContext(age_years=40).effective_max_hr()
        assert est == pytest.approx(208 - 0.7 * 40)
        assert "Tanaka" in src
        default, src = PersonContext().effective_max_hr()
        assert default == 185.0
        assert "approximate" in src

    def test_hr_reserve_fraction(self) -> None:
        ctx = PersonContext(max_hr_bpm=190, resting_hr_bpm=50)
        assert ctx.hr_reserve_fraction(120.0) == pytest.approx(0.5)
        assert ctx.hr_reserve_fraction(40.0) == 0.0


class TestDerivedMetrics:
    def test_hr_trend_recovers_known_slope(self) -> None:
        # HR declining 3 bpm over 5 min = −0.6 bpm/min.
        n = 400
        t_approx = np.arange(n) * 0.75
        hr = 80.0 - 0.6 * t_approx / 60.0
        rr = _series(60000.0 / hr)
        assert m.hr_trend_bpm_per_min(rr) == pytest.approx(-0.6, abs=0.05)

    def test_hr_zones_fractions(self) -> None:
        # Half the time at 120 bpm (63 % of 190), half at 160 (84 %).
        rr = np.concatenate([np.full(240, 500.0), np.full(320, 375.0)])
        zones = m.hr_zones(_series(rr), max_hr_bpm=190.0)
        assert zones is not None
        assert zones["z2"] == pytest.approx(0.5, abs=0.02)  # 120 bpm → 63 %
        assert zones["z4"] == pytest.approx(0.5, abs=0.02)  # 160 bpm → 84 %

    def test_hrr60_from_constructed_recovery(self) -> None:
        # 2 min at 160 bpm, then exponential fall toward 80.
        seg1 = np.full(320, 375.0)  # 160 bpm for 120 s
        rr_fall = []
        t_cursor = 0.0
        while t_cursor < 300.0:
            hr_now = 80.0 + 80.0 * np.exp(-t_cursor / 60.0)
            rr_now = 60000.0 / hr_now
            rr_fall.append(rr_now)
            t_cursor += rr_now / 1000.0
        rr = _series(np.concatenate([seg1, rr_fall]))
        hrr60 = m.hr_recovery(rr, 60.0)
        assert hrr60 is not None
        # After 60 s of τ=60 s decay from 160 → expect ≈ 160 − (80+80/e) ≈ 50.6,
        # blurred by the ±10 s averaging window.
        assert hrr60 == pytest.approx(50.0, abs=8.0)

    def test_recovery_tau_recovers_time_constant(self) -> None:
        seg1 = np.full(160, 375.0)  # 60 s at 160
        rr_fall = []
        t_cursor = 0.0
        while t_cursor < 400.0:
            hr_now = 80.0 + 80.0 * np.exp(-t_cursor / 75.0)
            rr_now = 60000.0 / hr_now
            rr_fall.append(rr_now)
            t_cursor += rr_now / 1000.0
        rr = _series(np.concatenate([seg1, rr_fall]))
        tau = m.recovery_time_constant(rr)
        assert tau is not None
        assert tau == pytest.approx(75.0, rel=0.2)

    def test_orthostatic_response_from_marker(self) -> None:
        # 70 bpm for 2 min, then 85 bpm — marker at the transition.
        rr = np.concatenate([np.full(140, 857.0), np.full(200, 706.0)])
        series = _series(rr)
        t_transition = float(np.cumsum(rr)[139] / 1000.0)
        result = m.orthostatic_response(series, [(t_transition, "standing up")])
        assert result is not None
        assert result["source"] == "marker"
        assert result["delta_hr_bpm"] == pytest.approx(15.0, abs=2.0)

    def test_settling_detects_plateau(self) -> None:
        # Falls for 4 min then flat at 90.
        rr_fall = 60000.0 / np.linspace(110.0, 90.0, 300)
        rr_flat = np.full(400, 60000.0 / 90.0)
        settled = m.hr_settling(_series(np.concatenate([rr_fall, rr_flat])))
        assert settled is not None
        settled_at_min, settled_hr = settled
        assert settled_hr == pytest.approx(90.0, abs=1.0)
        assert settled_at_min >= 2.0


class TestActivityTypeResolution:
    def test_seed_creates_eight_builtins(self, app: Flask) -> None:
        added = ensure_builtin_activity_types()
        assert added == 8
        assert ensure_builtin_activity_types() == 0  # idempotent

    def test_custom_activity_inherits_with_overrides(self, app: Flask) -> None:
        ensure_builtin_activity_types()
        yoga = ActivityType(
            name="Yoga",
            profile_key="supine",
            is_builtin=False,
            expected_hr_min=45,
            expected_hr_max=95,
            comparison_key="yoga",
        )
        db.session.add(yoga)
        db.session.commit()
        resolved = resolve_profile(yoga)
        assert resolved.profile.key == "supine"
        assert resolved.expected_hr_range == (45, 95)
        assert resolved.comparison_key == "yoga"
        # Builtin keeps profile values.
        builtin = db.session.query(ActivityType).filter_by(profile_key="walking").one()
        assert resolve_profile(builtin).comparison_key == "walking"


class TestActivityViews:
    def test_list_seeds_and_shows_builtins(self, client, app: Flask) -> None:
        resp = client.get("/activities/")
        assert resp.status_code == 200
        assert b"Lying down (supine)" in resp.data
        assert b"reference baseline" in resp.data

    def test_add_custom_activity(self, client, app: Flask) -> None:
        resp = client.post(
            "/activities/new",
            data={"name": "Rowing", "profile_key": "cycling", "expected_hr_min": "90",
                  "expected_hr_max": "180"},
            follow_redirects=True,
        )
        assert resp.status_code == 200
        row = db.session.query(ActivityType).filter_by(name="Rowing").one()
        assert row.profile_key == "cycling"
        assert row.is_builtin is False

    def test_duplicate_name_rejected(self, client, app: Flask) -> None:
        client.post("/activities/new", data={"name": "Rowing", "profile_key": "cycling"})
        resp = client.post(
            "/activities/new",
            data={"name": "Rowing", "profile_key": "running"},
            follow_redirects=True,
        )
        assert b"already exists" in resp.data
        assert db.session.query(ActivityType).filter_by(name="Rowing").count() == 1


class TestSuppressionSafety:
    def test_apply_suppressions_does_not_mutate_original(self) -> None:
        rr = _steady(75.0)
        hrv = compute_hrv(rr)
        original_rmssd = hrv.rmssd_ms
        from app.activities.base import Suppression

        censored, _ = apply_suppressions(
            hrv,
            [Suppression(family=MetricFamily.TIME, mode="suppress", reason="test")],
        )
        assert censored.rmssd_ms is None
        assert hrv.rmssd_ms == original_rmssd

    def test_cautions_do_not_blank_values(self) -> None:
        rr = _steady(75.0)
        hrv = compute_hrv(rr)
        from app.activities.base import Suppression

        censored, notes = apply_suppressions(
            hrv,
            [Suppression(family=MetricFamily.TIME, mode="caution", reason="test")],
        )
        assert censored.rmssd_ms == hrv.rmssd_ms
        assert notes == []


def test_hrv_result_type_alias() -> None:
    """Guard: apply_suppressions field lists stay in sync with HRVResult."""
    from app.activities.base import _FAMILY_FIELDS

    hrv = HRVResult()
    for fields in _FAMILY_FIELDS.values():
        for name in fields:
            assert hasattr(hrv, name), f"HRVResult no longer has {name}"
