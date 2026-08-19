"""Append-only decision log: one Markdown file per person.

An entry is appended on every completed analysis and every user annotation.
The file is **never rewritten or reordered** — the only file operation this
module performs is open-for-append. It is the durable, human-readable record
of what was analysed and what was decided, independent of the database.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from flask import current_app

from app.environment import format_environment_line
from app.models import Person, Session


def log_path_for(person: Person) -> Path:
    log_dir = Path(current_app.config["LOG_DIR"])
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{person.slug}_history.md"


def _append(person: Person, entry: str) -> None:
    """The single write primitive: append-only, trailing blank line."""
    path = log_path_for(person)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(entry.rstrip() + "\n\n")


def read_entries(person: Person) -> list[str]:
    """Split the log into entries for the timeline view (read-only)."""
    path = log_path_for(person)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    entries = ["## " + chunk for chunk in text.split("## ") if chunk.strip()]
    return entries


def append_session_entry(session: Session) -> None:
    """Log a completed analysis in the spec's entry format."""
    person = session.person
    metrics = session.metrics
    extras = (metrics.extras if metrics and metrics.extras else {}) or {}
    when = session.recorded_at or session.created_at
    activity = session.activity_type.name if session.activity_type else "Unspecified"

    lines = [f"## {when:%Y-%m-%d %H:%M} — {activity} — session {session.id}"]

    duration_min = (session.duration_s or 0) / 60.0
    n_beats = metrics.n_beats if metrics and metrics.n_beats else 0
    lines.append(
        f"- **Duration:** {duration_min:.1f} min · {n_beats:,} beats · "
        f"fs {session.sampling_rate_hz:.2f} Hz"
        if session.sampling_rate_hz
        else f"- **Duration:** {duration_min:.1f} min"
    )

    quality_bits = [f"{session.beats_corrected_pct or 0:.2f}% beats corrected"]
    quality_bits.append(f"{session.excluded_s or 0:.0f} s excluded")
    if extras.get("sqi_mean") is not None:
        quality_bits.append(f"SQI mean {extras['sqi_mean']:.2f}")
    if session.reduced_confidence:
        quality_bits.append("REDUCED CONFIDENCE (>5% corrected)")
    lines.append("- **Quality:** " + " · ".join(quality_bits))

    metric_bits: list[str] = []
    if metrics:
        if metrics.mean_hr_bpm is not None:
            metric_bits.append(f"mean HR {metrics.mean_hr_bpm:.0f}")
        if extras.get("resting_hr_bpm") is not None:
            metric_bits.append(f"resting HR {extras['resting_hr_bpm']:.0f}")
        if metrics.rmssd_ms is not None:
            metric_bits.append(f"RMSSD {metrics.rmssd_ms:.1f} ms")
        if metrics.sdnn_ms is not None:
            sdnn = f"SDNN {metrics.sdnn_ms:.1f} ms (trend-inclusive)"
            per_window = extras.get("sdnn_per_window_ms") or []
            if per_window:
                sdnn += f" · per-min SDNN {min(per_window):.0f}–{max(per_window):.0f} ms"
            metric_bits.append(sdnn)
    lines.append("- **Key metrics:** " + (" · ".join(metric_bits) if metric_bits else "—"))

    if metrics and metrics.resp_rate_median_brpm is not None:
        lines.append(
            f"- **Respiration (EDR estimate):** median "
            f"{metrics.resp_rate_median_brpm:.1f} brpm "
            f"({metrics.resp_rate_p5_brpm:.1f}–{metrics.resp_rate_p95_brpm:.1f})"
        )

    activity_bits = _activity_line(extras.get("activity") or {})
    if activity_bits:
        lines.append("- **Activity-specific:** " + " · ".join(activity_bits))

    if metrics and metrics.ectopy_beats_n is not None:
        ectopy_bits = [f"{metrics.ectopy_beats_n} confirmed"]
        if metrics.ectopy_per_hour is not None:
            ectopy_bits[0] += f" ({metrics.ectopy_per_hour:.2f}/h)"
        if metrics.couplet_n:
            ectopy_bits.append(f"{metrics.couplet_n} couplet(s)")
        if metrics.run_n:
            ectopy_bits.append(f"{metrics.run_n} run(s), longest {metrics.longest_run_beats}")
        if metrics.bigeminy_episode_n:
            ectopy_bits.append(f"{metrics.bigeminy_episode_n} bigeminy-pattern episode(s)")
        lines.append("- **Ectopic beats:** " + " · ".join(ectopy_bits))

    if extras.get("not_analysable"):
        lines.append("- **Flags:** session not analysable — screening skipped")
    elif session.flags:
        lines.append(
            "- **Flags:** "
            + " · ".join(f"{f.kind} ({f.severity})" for f in session.flags)
        )
    else:
        lines.append("- **Flags:** none raised")

    note_lines = [
        n for n in extras.get("notes", []) if "morphology outlier" in n or "suppressed" in n
    ]
    if note_lines:
        lines.append("- **Notes:** " + " ".join(note_lines))
    if session.trigger_tags:
        lines.append("- **Triggers:** " + " · ".join(t.name for t in session.trigger_tags))
    context_bits: list[str] = []
    if session.body_position:
        context_bits.append(
            f"position {session.body_position} ({session.body_position_source or 'user'})"
        )
    if session.alcohol_drinks_24h is not None:
        context_bits.append(f"alcohol {session.alcohol_drinks_24h} drink(s)/24 h")
    if session.sleep_quality_1_5:
        context_bits.append(f"sleep quality {session.sleep_quality_1_5}/5")
    if context_bits:
        lines.append("- **Context fields:** " + " · ".join(context_bits))
    posture = extras.get("posture") or {}
    if posture.get("dominant"):
        pct = posture.get("pct_by_posture") or {}
        posture_bits = [
            f"{name} {value:.0f}%"
            for name, value in sorted(pct.items(), key=lambda kv: -kv[1])
        ]
        posture_bits.append(f"{posture.get('n_transitions', 0)} change(s)")
        lines.append("- **Posture (ACC):** " + " · ".join(posture_bits))
    if session.env_fetched_at:
        env_line = format_environment_line(session)
        if env_line:
            lines.append(f"- **Environment:** {env_line}")
    if session.context_note:
        lines.append(f"- **Context:** {session.context_note}")
    lines.append(
        f"- **Engine:** {session.engine_used or 'unknown'} · Lipponen-Tarvainen correction"
    )

    _append(person, "\n".join(lines))


def append_annotation_entry(
    person: Person, text: str, session_id: int | None = None
) -> None:
    """Log a dated free-text annotation by the user."""
    now = dt.datetime.now(dt.UTC)
    header = f"## {now:%Y-%m-%d %H:%M} — Annotation"
    if session_id is not None:
        header += f" — session {session_id}"
    _append(person, f"{header}\n{text.strip()}")


def _activity_line(activity_extras: dict) -> list[str]:
    """Compact human-readable highlights from the activity extras."""
    formats = {
        "hr_trend_bpm_per_min": ("HR trend {:+.2f} bpm/min", float),
        "settled_hr_bpm": ("settled to {:.0f} bpm", float),
        "settled_at_min": ("at min {:.1f}", float),
        "hrr60_bpm": ("HRR60 {:.0f} bpm", float),
        "hrr120_bpm": ("HRR120 {:.0f} bpm", float),
        "recovery_tau_s": ("recovery τ {:.0f} s", float),
        "cardiac_drift_proxy_bpm_per_min": ("drift proxy {:+.2f} bpm/min", float),
        "mayer_peak_hz": ("Mayer peak {:.3f} Hz", float),
        "peak_hr_bpm": ("peak HR {:.0f} bpm", float),
        "motion_hr_coupling_r": ("motion–HR coupling r={:.2f}", float),
        "rmssd_reactivation_ms_per_min": ("RMSSD reactivation {:+.2f} ms/min", float),
    }
    bits: list[str] = []
    for key, (fmt, cast) in formats.items():
        value = activity_extras.get(key)
        if value is not None:
            try:
                bits.append(fmt.format(cast(value)))
            except (TypeError, ValueError):
                continue
    ortho = activity_extras.get("orthostatic_response")
    if isinstance(ortho, dict) and ortho.get("delta_hr_bpm") is not None:
        bits.append(
            f"orthostatic ΔHR {ortho['delta_hr_bpm']:+.0f} bpm ({ortho.get('source', '')})"
        )
    return bits
