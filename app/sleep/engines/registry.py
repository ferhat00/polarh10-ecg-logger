"""Which staging engines exist, and whether each can run here.

Single source of truth for the engine list, the best-available precedence,
and the availability probe — the parts the orchestrator *and* the UI both
need. Deliberately status-only: the three engines take genuinely different
inputs (per-epoch features / R-peak times / raw ECG over a subprocess), so a
uniform ``run()`` abstraction would be a fiction that either passes every
input to every engine or hides a dispatch table. Each call site still wires
its own engine call, which is where it reads clearly.

``status`` is looked up on its module *at call time*, never captured at
import: availability is monkeypatched by attribute in the tests, and a
captured reference would silently defeat that.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.sleep.engines import EngineStatus
from app.sleep.stages import StageVocab, stage_labels

#: Best-first. Also the primary-engine precedence: the first engine in this
#: order that produced a hypnogram fills the queryable ``Metrics`` columns,
#: unless the session asked for a specific engine.
ENGINE_ORDER: tuple[str, ...] = ("external-5class", "sleepecg", "heuristic")


def engine_statuses(config: Mapping) -> list[EngineStatus]:
    """Live status for every staging engine, in :data:`ENGINE_ORDER`."""
    from app.sleep.engines import external_ecg_staging, heuristic, sleepecg_engine

    return [
        external_ecg_staging.status(config),
        sleepecg_engine.status(),
        EngineStatus(
            key=heuristic.ENGINE_KEY,
            label=heuristic.ENGINE_LABEL,
            available=True,
        ),
    ]


def engine_status(key: str, config: Mapping) -> EngineStatus | None:
    """One engine's live status, or None for a key this build doesn't know."""
    return next((s for s in engine_statuses(config) if s.key == key), None)


def engine_vocab(key: str) -> StageVocab | None:
    """The stage vocabulary an engine emits — known before it runs."""
    from app.sleep.engines import external_ecg_staging, heuristic, sleepecg_engine

    return {
        heuristic.ENGINE_KEY: heuristic.ENGINE_VOCAB,
        sleepecg_engine.ENGINE_KEY: sleepecg_engine.ENGINE_VOCAB,
        external_ecg_staging.ENGINE_KEY: external_ecg_staging.ENGINE_VOCAB,
    }.get(key)


def vocab_summary(key: str) -> str:
    """e.g. ``"Wake/Light/Deep/REM"`` — what this engine can tell apart.

    Shown beside each option in the engine dropdown: a 3-class engine
    reports no light/deep split rather than inventing one, and the user
    should see that before choosing, not afterwards.
    """
    vocab = engine_vocab(key)
    return "/".join(stage_labels(vocab)) if vocab is not None else ""
