"""Figures for the triggers dashboard, in the report's visual language."""

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
from app.triggers.stats import HourProfile, SessionObservation, TriggerEffect


def burden_timeline(observations: list[SessionObservation]) -> Figure | None:
    """Ectopic beats per analysed hour, per session, over calendar time."""
    if len(observations) < 2:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 3.0), dpi=150)
    fig.patch.set_facecolor(PAPER)
    _style_axes(ax)
    dates = [o.recorded_at for o in observations]
    rates = [o.ectopy_per_hour for o in observations]
    ax.plot(dates, rates, color=INK, lw=1.0, alpha=0.6)
    normal = [not o.reduced_confidence for o in observations]
    ax.scatter(
        [d for d, n in zip(dates, normal, strict=True) if n],
        [r for r, n in zip(rates, normal, strict=True) if n],
        s=22, color=ACCENT, zorder=3, label="session",
    )
    if not all(normal):
        ax.scatter(
            [d for d, n in zip(dates, normal, strict=True) if not n],
            [r for r, n in zip(rates, normal, strict=True) if not n],
            s=26, facecolors="none", edgecolors=WARN, zorder=3,
            label="reduced confidence",
        )
        ax.legend(fontsize=7, framealpha=0.9)
    ax.set_ylabel("confirmed ectopic beats / h")
    ax.set_ylim(bottom=0)
    ax.grid(color=GRID_SMALL, lw=0.4)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return _finish(
        fig,
        "Per-session ectopic-beat rate over analysed time. Large session-to-"
        "session swings are expected — day-to-day burden varies severalfold "
        "in the monitoring literature.",
    )


def burden_by_tag(
    rates: dict[str, tuple[list[float], list[float]]],
    names: dict[str, str],
) -> Figure | None:
    """Jittered per-session rates, tagged vs untagged, one row per tag."""
    rates = {k: v for k, v in rates.items() if v[0]}
    if not rates:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 0.8 + 0.55 * len(rates)), dpi=150)
    fig.patch.set_facecolor(PAPER)
    _style_axes(ax)
    rng = np.random.default_rng(3)
    labels = []
    for row, (slug, (tagged, untagged)) in enumerate(sorted(rates.items())):
        labels.append(names.get(slug, slug))
        for values, color, offset in ((untagged, INK, -0.16), (tagged, ACCENT, 0.16)):
            if values:
                jitter = rng.uniform(-0.08, 0.08, len(values))
                ax.scatter(
                    values, np.full(len(values), row + offset) + jitter,
                    s=18, color=color, alpha=0.75,
                )
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("confirmed ectopic beats / h (per session)")
    ax.set_xlim(left=0)
    ax.grid(color=GRID_SMALL, lw=0.4, axis="x")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=ACCENT, label="tagged"),
        plt.Line2D([], [], marker="o", ls="", color=INK, label="without tag"),
    ]
    ax.legend(handles=handles, fontsize=7, framealpha=0.9)
    fig.tight_layout()
    return _finish(
        fig,
        "Each dot is one session. Overlapping clouds mean the tag explains "
        "little; the inference table quantifies this with intervals.",
    )


def hour_profile_figure(profile: HourProfile) -> Figure | None:
    """Exposure-normalised ectopy rate by clock hour."""
    if profile.n_sessions_used == 0 or sum(profile.n_events) == 0:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 2.6), dpi=150)
    fig.patch.set_facecolor(PAPER)
    _style_axes(ax)
    values = [v if v is not None else 0.0 for v in profile.events_per_hour]
    covered = [v is not None for v in profile.events_per_hour]
    ax.bar(
        profile.hours, values, width=0.8,
        color=[ACCENT if c else GRID_SMALL for c in covered],
    )
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("clock hour (recording timestamps)")
    ax.set_ylabel("ectopic beats / h")
    ax.grid(color=GRID_SMALL, lw=0.4, axis="y")
    fig.tight_layout()
    return _finish(
        fig,
        "Rate per clock hour, normalised by how much analysed time falls in "
        "each hour. Hours with no recorded exposure show empty. Ectopy has a "
        "circadian rhythm of its own — this is the pattern the models adjust "
        "for.",
    )


def forest(effects: list[TriggerEffect], outcome=None) -> Figure | None:
    """Per-tag effects with 95 % CIs; estimated rows only.

    Axis scale and reference line follow the outcome: rate ratios draw on a
    log axis around 1.0, %-change and Δ effects on a linear axis around 0.
    """
    rows = [
        e
        for e in effects
        if e.rate_ratio is not None and e.ci_low is not None and e.ci_high is not None
    ]
    if not rows:
        return None
    kind = rows[0].effect_kind
    fig, ax = plt.subplots(figsize=(8.4, 0.9 + 0.5 * len(rows)), dpi=150)
    fig.patch.set_facecolor(PAPER)
    _style_axes(ax)
    for i, e in enumerate(rows):
        color = ACCENT if e.status == "ok" else WARN
        ax.plot([e.ci_low, e.ci_high], [i, i], color=color, lw=1.6)
        ax.plot([e.rate_ratio], [i], marker="s", color=color, markersize=6)
    ax.set_yticks(
        range(len(rows)),
        [f"{e.tag_name} (n={e.n_tagged}/{e.n_untagged})" for e in rows],
        fontsize=8,
    )
    ax.grid(color=GRID_SMALL, lw=0.4, axis="x")

    if kind == "rate_ratio":
        ax.axvline(1.0, color=INK, lw=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_xlabel("rate ratio (tagged vs untagged), 95% CI — log scale")
        caption = (
            "Squares right of the dashed line mean more ectopic beats per hour "
            "in tagged sessions; intervals crossing 1.0 are compatible with no "
            "association. Orange rows used the Poisson fallback."
        )
    elif kind == "pct_change":
        ax.axvline(0.0, color=INK, lw=0.8, linestyle="--")
        ax.set_xlabel("% change in RMSSD (tagged vs untagged), 95% CI")
        caption = (
            "Squares right of the dashed line mean higher RMSSD in tagged "
            "sessions; intervals crossing 0 are compatible with no association."
        )
    else:  # delta
        ax.axvline(0.0, color=INK, lw=0.8, linestyle="--")
        ax.set_xlabel("Δ resting HR (bpm, tagged vs untagged), 95% CI")
        caption = (
            "Squares right of the dashed line mean a higher resting HR in "
            "tagged sessions; intervals crossing 0 are compatible with no "
            "association."
        )
    fig.tight_layout()
    return _finish(fig, caption)


def env_scatter(assoc, outcome) -> Figure | None:
    """Outcome vs one environment variable — descriptive scatter only."""
    if not assoc.points:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=150)
    fig.patch.set_facecolor(PAPER)
    _style_axes(ax)
    x = [p[0] for p in assoc.points]
    y = [p[1] for p in assoc.points]
    ax.scatter(x, y, s=18, color=ACCENT, alpha=0.75, edgecolors="none")
    ax.set_xlabel(f"{assoc.var_label} ({assoc.unit})")
    ax.set_ylabel(f"{outcome.label} ({outcome.group_unit})")
    ax.grid(color=GRID_SMALL, lw=0.4)
    rho_text = (
        f"Spearman ρ = {assoc.rho:+.2f} (n = {assoc.n})"
        if assoc.rho is not None
        else f"n = {assoc.n}"
    )
    ax.set_title(rho_text, fontsize=9, color=INK)
    fig.tight_layout()
    return _finish(
        fig,
        f"{outcome.label} per session against {assoc.var_label.lower()} at the "
        f"recording's time and home location ({rho_text}) — descriptive only; "
        "no model is fitted at this sample size, and published environmental "
        "effects are percent-level (see docs/CONTEXT_METRICS.md).",
    )
