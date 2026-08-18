"""Sleep-section figures: hypnograms, stage distribution, movement trace.

Same conventions as :mod:`app.report.figures` — matplotlib only, base64
PNGs, ECG-paper palette. The hypnogram y-axis uses the clinical convention
(Wake on top, then REM, then progressively deeper NREM), and every panel
carries its engine's accuracy note so the estimate and its error travel
together.
"""

from __future__ import annotations

import numpy as np

from app.report.figures import (
    ACCENT,
    GRID_SMALL,
    INK,
    PAPER,
    WARN,
    Figure,
    _finish,
    _style_axes,
    plt,
)
from app.sleep.actigraphy import AccEpochs
from app.sleep.stages import UNSCORED, Hypnogram, stage_labels
from app.sleep.summary import SleepSummary

#: Display order top→bottom per vocabulary size (indices are stage codes;
#: the y position of code c is DISPLAY_ORDER[n_classes].index-like maps).
_CLINICAL_ORDER: dict[int, list[int]] = {
    2: [0, 1],  # Wake, Sleep
    3: [0, 2, 1],  # Wake, REM, NREM
    4: [0, 3, 1, 2],  # Wake, REM, Light, Deep
    5: [0, 4, 1, 2, 3],  # Wake, REM, N1, N2, N3
}


def hypnogram_figure(
    hyps: list[Hypnogram], override_epochs_n: dict[str, int] | None = None
) -> Figure | None:
    """One step-plot panel per engine on a shared time axis."""
    hyps = [h for h in hyps if h.n_epochs > 0]
    if not hyps:
        return None
    override_epochs_n = override_epochs_n or {}

    fig, axes = plt.subplots(
        len(hyps),
        1,
        figsize=(10.0, 1.9 * len(hyps) + 0.6),
        sharex=True,
        squeeze=False,
    )
    fig.patch.set_facecolor(PAPER)

    for ax, hyp in zip(axes[:, 0], hyps, strict=True):
        _style_axes(ax)
        labels = stage_labels(hyp.vocab)
        order = _CLINICAL_ORDER[len(labels)]
        # y position per stage code: top row = 0 → invert axis later.
        y_of_code = {code: row for row, code in enumerate(order)}

        t_h = hyp.epoch_start_s / 3600.0
        y = np.array(
            [y_of_code.get(int(s), np.nan) for s in hyp.stages], dtype=float
        )
        scored = hyp.stages != UNSCORED

        # Step plot; unscored epochs break the line.
        y_plot = np.where(scored, y, np.nan)
        ax.step(t_h, y_plot, where="post", color=INK, linewidth=1.2)
        # Tint sleep depth: fill below the line lightly.
        ax.fill_between(
            t_h, y_plot, len(labels) - 0.5, step="post",
            color=ACCENT, alpha=0.10, linewidth=0,
        )
        if np.any(~scored):
            for t in t_h[~scored]:
                ax.axvspan(
                    t, t + hyp.epoch_len_s / 3600.0, color=GRID_SMALL, alpha=0.8,
                    linewidth=0,
                )

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([labels[code] for code in order])
        ax.set_ylim(len(labels) - 0.5, -0.5)  # Wake on top
        ax.grid(axis="x", color=GRID_SMALL, linewidth=0.5)

        n_over = override_epochs_n.get(hyp.engine, 0)
        title = hyp.engine_label
        if n_over:
            title += f" — {n_over} epoch(s) re-scored to Wake by movement"
        ax.set_title(title, fontsize=9, loc="left")
        ax.text(
            0.0,
            -0.32,
            hyp.accuracy_note,
            transform=ax.transAxes,
            fontsize=6.5,
            color=INK,
            alpha=0.75,
            wrap=True,
            va="top",
        )

    axes[-1, 0].set_xlabel("Hours since recording start")
    fig.subplots_adjust(hspace=0.9, left=0.09, right=0.98, top=0.93, bottom=0.22)
    return _finish(
        fig,
        "Hypnogram per staging engine (Wake at the top, deeper sleep lower; "
        "grey spans could not be scored). Engines are estimates from heart-"
        "beat patterns — disagreement between panels is honest uncertainty.",
    )


def stage_distribution(summaries: dict[str, SleepSummary]) -> Figure | None:
    """Grouped horizontal bars: minutes per stage per engine."""
    if not summaries:
        return None
    stage_attrs = [
        ("Wake", "wake_min"),
        ("Light", "light_min"),
        ("Deep", "deep_min"),
        ("REM", "rem_min"),
        ("NREM", "nrem_min"),
        ("Sleep", "sleep_min"),
    ]
    rows: list[tuple[str, str, float]] = []  # (stage, engine, minutes)
    for engine, summary in summaries.items():
        for label, attr in stage_attrs:
            value = getattr(summary, attr)
            if value is not None:
                rows.append((label, engine, value))
    if not rows:
        return None

    stages = [s for s, _a in stage_attrs if any(r[0] == s for r in rows)]
    stages = list(dict.fromkeys(stages))
    engines = list(summaries.keys())
    fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(stages) * max(1, len(engines)) + 1.2))
    fig.patch.set_facecolor(PAPER)
    _style_axes(ax)

    bar_h = 0.8 / max(1, len(engines))
    colors = [ACCENT, INK, WARN]
    for j, engine in enumerate(engines):
        ys, widths = [], []
        for i, stage in enumerate(stages):
            match = [r for r in rows if r[0] == stage and r[1] == engine]
            if match:
                ys.append(i + (j - (len(engines) - 1) / 2.0) * bar_h)
                widths.append(match[0][2])
        ax.barh(
            ys, widths, height=bar_h * 0.9,
            color=colors[j % len(colors)], alpha=0.85, label=engine,
        )
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)
    ax.invert_yaxis()
    ax.set_xlabel("Minutes")
    ax.grid(axis="x", color=GRID_SMALL, linewidth=0.5)
    if len(engines) > 1:
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return _finish(
        fig,
        "Minutes per stage, by engine. A 3-class engine reports NREM as one "
        "block — light/deep detail is never invented.",
    )


def movement_trace(acc: AccEpochs) -> Figure | None:
    """Per-epoch activity counts with the wake-override threshold."""
    if len(acc.counts) == 0:
        return None
    fig, ax = plt.subplots(figsize=(10.0, 2.0))
    fig.patch.set_facecolor(PAPER)
    _style_axes(ax)
    t_h = acc.epoch_start_s / 3600.0
    ax.fill_between(
        t_h, 0, np.nan_to_num(acc.counts, nan=0.0), step="post",
        color=ACCENT, alpha=0.55, linewidth=0,
    )
    if np.isfinite(acc.threshold):
        ax.axhline(acc.threshold, color=WARN, linewidth=1.0, linestyle="--")
        ax.text(
            0.995, acc.threshold, " wake-override threshold", color=WARN,
            fontsize=7, ha="right", va="bottom", transform=ax.get_yaxis_transform(),
        )
    no_cov = ~np.isfinite(acc.counts)
    for t in t_h[no_cov]:
        ax.axvspan(t, t + 30.0 / 3600.0, color=GRID_SMALL, alpha=0.8, linewidth=0)
    ax.set_xlabel("Hours since recording start")
    ax.set_ylabel("Activity (mg·s)")
    ax.grid(axis="x", color=GRID_SMALL, linewidth=0.5)
    fig.tight_layout()
    return _finish(
        fig,
        "Accelerometer activity per 30 s epoch (band-passed magnitude, "
        "te Lindert & Van Someren 2013 convention). Sustained movement above "
        "the dashed threshold re-scores those epochs to Wake; grey spans had "
        "no accelerometer coverage.",
    )
