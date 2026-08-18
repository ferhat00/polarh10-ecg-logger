"""Sleep staging engines.

Each engine stages the canonical 30 s epoch grid and returns a
:class:`~app.sleep.stages.Hypnogram`. Engines differ in inputs and
availability:

* :mod:`heuristic <app.sleep.engines.heuristic>` — cited rules over per-epoch
  cardiac features; no optional dependencies, always available.
* :mod:`sleepecg <app.sleep.engines.sleepecg_engine>` — pre-trained GRU
  classifiers from the ``sleepecg`` package; needs the optional
  ``requirements-sleep.txt`` extras (TensorFlow).
* :mod:`external <app.sleep.engines.external_ecg_staging>` — an optional,
  separately-installed AGPL tool driven over a subprocess boundary.

An unavailable engine is a fact to report (an :class:`EngineStatus` with a
reason), never an error: session processing must not fail because a staging
engine is missing or broken.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineStatus:
    """Whether an engine can run here, and if not, exactly why."""

    key: str
    label: str
    available: bool
    unavailable_reason: str | None = None
