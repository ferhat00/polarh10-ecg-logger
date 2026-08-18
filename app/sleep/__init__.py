"""Sleep-stage analysis from overnight recordings.

Package layout:

* :mod:`app.sleep.stages` — stage vocabularies, collapse tables, the
  :class:`~app.sleep.stages.Hypnogram` container, and the canonical 30 s
  epoch grid every engine must score against.
* :mod:`app.sleep.summary` — sleep-architecture arithmetic (TST, efficiency,
  WASO, latencies) and per-stage HRV statistics.
* :mod:`app.sleep.agreement` — epoch-by-epoch agreement (Cohen's kappa)
  between engines.
* :mod:`app.sleep.actigraphy` — accelerometer activity counts and the
  movement-based wake override.
* :mod:`app.sleep.epochs` — per-epoch cardiac features for the built-in
  heuristic engine.
* :mod:`app.sleep.engines` — the staging engines themselves.
* :mod:`app.sleep.orchestrator` — runs every available engine over one
  session and assembles the result for persistence and the report.

Nothing in this package diagnoses anything. Heart-rate-based sleep staging
is an estimate whose ceiling — even for the best published model validated
on this exact strap (Sleep²/Topalidis et al. 2023, Sensors 23(5):2390,
doi:10.3390/s23052390) — is ~80% epoch agreement with polysomnography.
Every hypnogram carries its accuracy note into the report.
"""
