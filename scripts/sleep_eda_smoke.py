"""Headless run of the sleep EDA notebook's analysis path.

Exercises exactly what ``notebooks/sleep_eda.ipynb`` runs — load, per-epoch
features, every engine, summaries, agreement — without needing Jupyter
installed, and cross-checks the cache reconstruction against the hypnogram
the app stored at processing time.

    .venv/Scripts/python scripts/sleep_eda_smoke.py <sleep-log.csv> [acc.csv]

Exits non-zero if the analysis produces no hypnogram, or if the cached and
recomputed heuristic stagings disagree beyond the tolerance below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

from notebooks.sleep_eda_core import (  # noqa: E402
    engine_statuses,
    epoch_table,
    load_session,
    stage_night,
)

#: The cached hypnogram was produced by the same deterministic engine, so the
#: recomputation should match it exactly. A small tolerance is allowed only
#: because the cache may predate a threshold change in the engine; anything
#: larger means the cache reconstruction is wrong, not merely stale.
MAX_CACHE_MISMATCH_PCT = 1.0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    csv_path = Path(argv[1])
    acc_path = Path(argv[2]) if len(argv) > 2 else None

    print("=== engines ===")
    for status in engine_statuses():
        mark = "available" if status.available else f"UNAVAILABLE ({status.unavailable_reason})"
        print(f"  {status.key:<16} {mark}")

    print("\n=== load ===")
    data = load_session(csv_path)
    print(f"  source          {data.source}")
    print(f"  start (local)   {data.start_time.astimezone():%Y-%m-%d %H:%M:%S %Z}")
    print(f"  duration        {data.duration_s / 3600.0:.2f} h")
    print(f"  RR intervals    {len(data.rr)}")
    print(f"  R-peaks         {len(data.peak_times_s)}")
    print(f"  epochs          {data.n_epochs}")

    print("\n=== staging ===")
    analysis, features = stage_night(data, acc_path=acc_path)
    if not analysis.hypnograms:
        # Not necessarily a bug: a sub-30-minute recording is refused by
        # design. The notes say which it was.
        print("  No hypnogram was produced — reasons below.")
        for note in analysis.notes:
            print(f"  note: {note}")
        return 1

    print(f"  movement source {features.movement_source}")
    print(f"  primary engine  {analysis.primary_engine}")
    for hyp in analysis.hypnograms:
        summary = analysis.summaries[hyp.engine]
        unscored = int(np.sum(~hyp.scored_mask()))
        print(
            f"  {hyp.engine:<16} vocab={hyp.vocab.value:<16} "
            f"TST={summary.tst_min:.1f} min  SE={summary.sleep_efficiency_pct:.1f}%  "
            f"unscored={unscored} epochs"
        )
        print(f"      stage % of TST: {_pct(summary.stage_pct_of_tst)}")

    print("\n=== agreement ===")
    if not analysis.agreement:
        print("  (only one engine produced a hypnogram — nothing to compare)")
    for row in analysis.agreement:
        kappa = "n/a" if row.kappa is None else f"{row.kappa:.3f}"
        agree = "n/a" if row.percent_agree is None else f"{row.percent_agree:.1f}%"
        print(
            f"  {row.engine_a} vs {row.engine_b} in {row.vocab.value}: "
            f"kappa={kappa} agreement={agree} over {row.n_epochs} epochs"
        )

    print("\n=== per-epoch table ===")
    table = epoch_table(data, analysis, features)
    print(f"  {len(table)} rows x {len(table.columns)} columns")
    print(f"  columns: {', '.join(table.columns)}")

    print("\n=== cache cross-check ===")
    status = _check_cache(data, analysis)

    print("\n=== notes ===")
    for note in analysis.notes + features.notes:
        print(f"  - {note}")

    return status


def _pct(mapping: dict) -> str:
    return ", ".join(f"{label} {value:.1f}%" for label, value in mapping.items()) or "n/a"


def _check_cache(data, analysis) -> int:
    """Compare recomputed stages against the ones stored in the .npz."""
    if data.source != "cache" or not data.cached_stages:
        print("  (no stored hypnogram to compare against)")
        return 0
    worst = 0.0
    for hyp in analysis.hypnograms:
        stored = data.cached_stages.get(hyp.engine)
        if stored is None:
            print(f"  {hyp.engine}: not in the cache (engine was unavailable at processing time)")
            continue
        if len(stored) != hyp.n_epochs:
            print(
                f"  FAIL {hyp.engine}: cache has {len(stored)} epochs, "
                f"recomputed {hyp.n_epochs}"
            )
            return 1
        mismatch = float(np.mean(stored != hyp.stages) * 100.0)
        worst = max(worst, mismatch)
        verdict = "ok" if mismatch <= MAX_CACHE_MISMATCH_PCT else "FAIL"
        print(f"  {verdict} {hyp.engine}: {mismatch:.2f}% of epochs differ from the cache")
    return 0 if worst <= MAX_CACHE_MISMATCH_PCT else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
