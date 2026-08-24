"""The staging-engine registry: the list, the ranking, and live availability."""

from __future__ import annotations

import pytest

from app.sleep.engines import EngineStatus, registry
from app.sleep.stages import StageVocab


class TestEngineList:
    def test_every_engine_appears_in_order(self) -> None:
        statuses = registry.engine_statuses({})
        assert [s.key for s in statuses] == list(registry.ENGINE_ORDER)
        assert set(registry.ENGINE_ORDER) == {
            "external-5class",
            "sleepecg",
            "heuristic",
        }

    def test_heuristic_is_always_available(self) -> None:
        heuristic = registry.engine_status("heuristic", {})
        assert heuristic is not None and heuristic.available is True

    def test_external_reports_its_missing_configuration(self) -> None:
        external = registry.engine_status("external-5class", {})
        assert external is not None
        assert external.available is False
        # An unavailable engine is a fact with a reason, never an error.
        assert "ECGLOG_SLEEP_EXTERNAL_DIR" in external.unavailable_reason

    def test_unknown_key_is_none_not_an_error(self) -> None:
        assert registry.engine_status("no-such-engine", {}) is None

    def test_precedence_is_the_registry_order(self) -> None:
        from app.sleep import orchestrator

        assert orchestrator.ENGINE_PRECEDENCE is registry.ENGINE_ORDER


class TestMonkeypatchableStatus:
    """status() must be resolved on the module at call time.

    Availability is monkeypatched by attribute across the sleep tests; a
    reference captured at import would silently ignore that and the suite
    would start depending on whether the optional extras happen to be
    installed on the machine running it.
    """

    def test_registry_reflects_a_patched_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.sleep.engines import sleepecg_engine

        monkeypatch.setattr(
            sleepecg_engine,
            "status",
            lambda: EngineStatus(
                sleepecg_engine.ENGINE_KEY,
                sleepecg_engine.ENGINE_LABEL,
                False,
                "patched off",
            ),
        )
        status = registry.engine_status("sleepecg", {})
        assert status is not None
        assert status.available is False
        assert status.unavailable_reason == "patched off"


class TestVocabularies:
    def test_each_engine_declares_the_vocab_it_emits(self) -> None:
        assert registry.engine_vocab("heuristic") is StageVocab.WAKE_LIGHT_DEEP_REM
        assert registry.engine_vocab("sleepecg") is StageVocab.WAKE_REM_NREM
        assert registry.engine_vocab("external-5class") is StageVocab.AASM_5
        assert registry.engine_vocab("no-such-engine") is None

    def test_vocab_summary_names_the_stages(self) -> None:
        assert registry.vocab_summary("heuristic") == "Wake/Light/Deep/REM"
        # The dropdown must warn that a 3-class engine has no light/deep split.
        assert registry.vocab_summary("sleepecg") == "Wake/NREM/REM"
        assert registry.vocab_summary("no-such-engine") == ""
