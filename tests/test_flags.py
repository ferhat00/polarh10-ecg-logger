"""Flag-object tests: the disclaimer and language rules are unbreakable."""

from __future__ import annotations

import pytest

from app.screening.flags import (
    DISCLAIMER,
    NO_FLAGS_STATEMENT,
    ScreeningFlag,
)


def _flag(**overrides: object) -> ScreeningFlag:
    kwargs: dict = {
        "kind": "test",
        "severity": "review",
        "description": "Screening threshold exceeded — a pattern worth noting.",
        "evidence": {"value": 1},
        "threshold": {"limit": 0},
    }
    kwargs.update(overrides)
    return ScreeningFlag(**kwargs)


class TestDisclaimerWeldedOn:
    def test_default_disclaimer_present(self) -> None:
        assert _flag().disclaimer == DISCLAIMER
        assert "not a diagnosis" in DISCLAIMER

    def test_empty_disclaimer_rejected(self) -> None:
        with pytest.raises(ValueError, match="disclaimer"):
            _flag(disclaimer="")

    def test_whitespace_disclaimer_rejected(self) -> None:
        with pytest.raises(ValueError, match="disclaimer"):
            _flag(disclaimer="   ")

    def test_disclaimer_survives_serialisation(self) -> None:
        payload = _flag().as_dict()
        assert payload["disclaimer"] == DISCLAIMER


class TestLanguageRules:
    @pytest.mark.parametrize(
        "bad",
        [
            "You have tachycardia.",
            "This indicates a problem.",
            "Abnormal rhythm detected.",
            "12 PVCs were found.",
            "Premature ventricular contractions present.",
            "A diagnosis of AF is likely.",
            "Findings indicative of ectopy.",
        ],
    )
    def test_diagnostic_phrasing_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="phrasing"):
            _flag(description=bad)

    @pytest.mark.parametrize(
        "good",
        [
            "Pattern consistent with possible atrial fibrillation; single-lead data "
            "cannot resolve P-waves.",
            "Screening threshold exceeded for sustained heart rate.",
            "Ectopic beats confirmed by three independent checks — a pattern worth noting.",
        ],
    )
    def test_screening_phrasing_accepted(self, good: str) -> None:
        assert _flag(description=good).description == good

    def test_unknown_severity_rejected(self) -> None:
        with pytest.raises(ValueError, match="severity"):
            _flag(severity="critical")


class TestEvidenceAndThresholdRequired:
    def test_empty_evidence_rejected(self) -> None:
        with pytest.raises(ValueError, match="evidence"):
            _flag(evidence={})

    def test_empty_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            _flag(threshold={})


class TestNoFlagsStatement:
    def test_never_reads_as_clearance(self) -> None:
        assert "not clinical clearance" in NO_FLAGS_STATEMENT
        lowered = NO_FLAGS_STATEMENT.lower()
        assert "you're fine" not in lowered
        assert "healthy" not in lowered
        assert "normal" not in lowered
