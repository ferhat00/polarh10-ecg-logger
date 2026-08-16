"""Figures for the comparison and longitudinal views."""

from __future__ import annotations

import matplotlib.pyplot as plt
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
)

_SERIES_COLORS = (ACCENT, INK, WARN, "#4C6E91", "#8E5D8E", "#7A6C5D")


def overlay_series(
    series: list[tuple[str, list[float], list[float]]],
    ylabel: str,
    caption: str,
) -> Figure | None:
    """Overlaid per-minute traces, one line per session."""
    series = [(label, x, y) for label, x, y in series if len(x) >= 2]
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 3.0), dpi=150)
    fig.patch.set_facecolor(PAPER)
    _style_axes(ax)
    for i, (label, x, y) in enumerate(series):
        ax.plot(
            np.asarray(x) / 60.0,
            y,
            color=_SERIES_COLORS[i % len(_SERIES_COLORS)],
            lw=1.3,
            marker="o",
            markersize=2.5,
            label=label,
        )
    ax.set_xlabel("minutes into session")
    ax.set_ylabel(ylabel)
    ax.grid(color=GRID_SMALL, lw=0.4)
    ax.legend(fontsize=7, framealpha=0.9)
    fig.tight_layout()
    return _finish(fig, caption)


def trend_figure(
    dates: list,
    values: list[float | None],
    band: tuple[list, list, list] | None,
    ylabel: str,
    caption: str,
) -> Figure | None:
    """One quantity over calendar time with an optional rolling baseline band."""
    xy = [(d, v) for d, v in zip(dates, values, strict=True) if v is not None]
    if len(xy) < 2:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 3.0), dpi=150)
    fig.patch.set_facecolor(PAPER)
    _style_axes(ax)

    if band is not None:
        mean, lo, hi = band
        seg = [
            (d, m, a, b)
            for d, m, a, b in zip(dates, mean, lo, hi, strict=True)
            if m is not None
        ]
        if seg:
            ax.fill_between(
                [s[0] for s in seg],
                [s[2] for s in seg],
                [s[3] for s in seg],
                color=ACCENT,
                alpha=0.12,
                label="rolling baseline ±1.96 σ",
            )
            ax.plot(
                [s[0] for s in seg], [s[1] for s in seg],
                color=ACCENT, lw=1.0, linestyle="--",
            )

    ax.plot(
        [d for d, _ in xy], [v for _, v in xy],
        color=INK, lw=1.2, marker="o", markersize=4,
    )
    ax.set_ylabel(ylabel)
    ax.grid(color=GRID_SMALL, lw=0.4)
    if band is not None:
        ax.legend(fontsize=7, framealpha=0.9)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return _finish(fig, caption)
