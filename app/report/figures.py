"""Matplotlib figures for the session report. Server-side only, no JS.

The rhythm strips are drawn at true clinical scale — 25 mm/s and 10 mm/mV,
so one small (1 mm) square is 40 ms × 0.1 mV and the grid means something.
Figure geometry is computed in millimetres and the HTML sets the image's CSS
width in mm to preserve the ratio.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from app.pipeline.hrv import HF_BAND, LF_BAND, VLF_BAND, HRVResult  # noqa: E402
from app.pipeline.quality import QualityResult  # noqa: E402
from app.pipeline.rr import RRSeries  # noqa: E402
from app.pipeline.template import MORPHOLOGY_R_THRESHOLD, BeatMorphology  # noqa: E402

# ECG-paper palette (project spec).
PAPER = "#FFFCFA"
GRID_SMALL = "#F6CFC4"
GRID_LARGE = "#E89A85"
INK = "#1B1E28"
ACCENT = "#2E5E4E"
WARN = "#B4472E"

#: Clinical scale.
MM_PER_S = 25.0
MM_PER_MV = 10.0

_DPI = 150
_MM_PER_INCH = 25.4


@dataclass
class Figure:
    """A rendered figure: base64 PNG data URI plus display hints."""

    data_uri: str
    caption: str
    #: When set, the <img> is given this CSS width in mm (true-scale strips).
    css_width_mm: float | None = None


def _finish(fig: plt.Figure, caption: str, css_width_mm: float | None = None) -> Figure:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return Figure(f"data:image/png;base64,{data}", caption, css_width_mm)


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PAPER)
    for spine in ax.spines.values():
        spine.set_color(GRID_LARGE)
    ax.tick_params(colors=INK, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


# ---------------------------------------------------------------------------
# Rhythm strips at true scale
# ---------------------------------------------------------------------------


def rhythm_strip(
    time_s: np.ndarray,
    ecg_mv: np.ndarray,
    start_s: float,
    duration_s: float,
    peak_times_s: np.ndarray | None = None,
    outlier_times_s: np.ndarray | None = None,
    label: str = "",
    with_calibration: bool = True,
) -> Figure | None:
    """One rhythm strip at 25 mm/s and 10 mm/mV on a real 1 mm/5 mm grid."""
    mask = (time_s >= start_s) & (time_s < start_s + duration_s)
    if int(np.sum(mask)) < 10:
        return None
    t = time_s[mask] - start_s
    x = ecg_mv[mask]

    # Vertical range: ±1.5 mV → 30 mm tall; horizontal: duration × 25 mm/s
    # plus a 6 mm calibration block.
    y_min_mv, y_max_mv = -1.5, 1.5
    cal_mm = 6.0 if with_calibration else 0.0
    width_mm = duration_s * MM_PER_S + cal_mm + 2.0
    height_mm = (y_max_mv - y_min_mv) * MM_PER_MV + 2.0

    fig, ax = plt.subplots(
        figsize=(width_mm / _MM_PER_INCH, height_mm / _MM_PER_INCH), dpi=_DPI
    )
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Work in millimetre coordinates so the grid is exact.
    ax.set_xlim(-cal_mm, duration_s * MM_PER_S + 2.0)
    ax.set_ylim(y_min_mv * MM_PER_MV - 1.0, y_max_mv * MM_PER_MV + 1.0)

    # 1 mm / 5 mm grid.
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    for gx in np.arange(np.floor(x_lo), x_hi, 1.0):
        ax.axvline(gx, color=GRID_SMALL, lw=0.35, zorder=0)
    for gy in np.arange(np.floor(y_lo), y_hi, 1.0):
        ax.axhline(gy, color=GRID_SMALL, lw=0.35, zorder=0)
    for gx in np.arange(np.floor(x_lo / 5.0) * 5.0, x_hi, 5.0):
        ax.axvline(gx, color=GRID_LARGE, lw=0.6, zorder=1)
    for gy in np.arange(np.floor(y_lo / 5.0) * 5.0, y_hi, 5.0):
        ax.axhline(gy, color=GRID_LARGE, lw=0.6, zorder=1)

    # Calibration pulse: 1 mV tall, 200 ms wide (5 mm), like a real printout.
    if with_calibration:
        pulse_x = [-cal_mm + 0.5, -cal_mm + 0.5, -cal_mm + 0.5 + 5.0, -cal_mm + 0.5 + 5.0]
        pulse_y = [0.0, 1.0 * MM_PER_MV, 1.0 * MM_PER_MV, 0.0]
        ax.plot(pulse_x, pulse_y, color=INK, lw=1.1, zorder=3)

    ax.plot(t * MM_PER_S, x * MM_PER_MV, color=INK, lw=0.8, zorder=4)

    if peak_times_s is not None:
        peaks_in = peak_times_s[(peak_times_s >= start_s) & (peak_times_s < start_s + duration_s)]
        for pt in peaks_in:
            ax.plot(
                (pt - start_s) * MM_PER_S,
                y_max_mv * MM_PER_MV - 1.0,
                marker="v",
                color=ACCENT,
                markersize=3,
                zorder=5,
            )
    if outlier_times_s is not None:
        outs = outlier_times_s[
            (outlier_times_s >= start_s) & (outlier_times_s < start_s + duration_s)
        ]
        for ot in outs:
            ax.axvspan(
                (ot - start_s) * MM_PER_S - 2.5,
                (ot - start_s) * MM_PER_S + 2.5,
                color=WARN,
                alpha=0.12,
                zorder=2,
            )

    ax.set_xticks([])
    ax.set_yticks([])
    caption = (
        f"{label} — 25 mm/s, 10 mm/mV; one small square = 40 ms × 0.1 mV. "
        f"Strip starts at {_fmt_clock(start_s)}."
    )
    return _finish(fig, caption, css_width_mm=width_mm)


def _fmt_clock(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Session-level figures
# ---------------------------------------------------------------------------


def hr_timeseries(
    rr: RRSeries,
    excluded_segments: list[tuple[float, float, str]],
    duration_s: float,
) -> Figure | None:
    if len(rr) < 10:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 2.8), dpi=_DPI)
    fig.patch.set_facecolor(PAPER)
    _style_axes(ax)

    t_min = rr.t_s / 60.0
    hr = 60000.0 / rr.rr_ms
    ax.plot(t_min, hr, color=INK, lw=0.7, alpha=0.85)

    if len(rr) >= 30:
        coeffs = np.polyfit(t_min, hr, 1)
        ax.plot(
            t_min,
            np.polyval(coeffs, t_min),
            color=ACCENT,
            lw=1.4,
            linestyle="--",
            label=f"trend {coeffs[0]:+.2f} bpm/min",
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    for seg_start, seg_end, _reason in excluded_segments:
        ax.axvspan(seg_start / 60.0, seg_end / 60.0, color=WARN, alpha=0.15)

    ax.set_xlim(0, max(duration_s / 60.0, float(t_min[-1])))
    ax.set_xlabel("minutes")
    ax.set_ylabel("bpm")
    ax.grid(color=GRID_SMALL, lw=0.4)
    fig.tight_layout()
    return _finish(
        fig,
        "Heart rate across the session with least-squares trend. "
        "Shaded spans are excluded time (reported, never interpolated).",
    )


def poincare(rr: RRSeries, hrv: HRVResult) -> Figure | None:
    if len(rr) < 20:
        return None
    # Only pairs within contiguous runs.
    keep = ~rr.discontinuity[1:]
    x = rr.rr_ms[:-1][keep]
    y = rr.rr_ms[1:][keep]
    if len(x) < 10:
        return None

    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=_DPI)
    fig.patch.set_facecolor(PAPER)
    _style_axes(ax)
    ax.scatter(x, y, s=4, color=INK, alpha=0.35, linewidths=0)

    if hrv.sd1_ms and hrv.sd2_ms:
        mean = float(np.mean(rr.rr_ms))
        ellipse = Ellipse(
            (mean, mean),
            width=2 * hrv.sd2_ms,
            height=2 * hrv.sd1_ms,
            angle=45.0,
            fill=False,
            color=ACCENT,
            lw=1.5,
        )
        ax.add_patch(ellipse)
        lim_pad = max(3 * hrv.sd2_ms, 60.0)
        ax.set_xlim(mean - lim_pad, mean + lim_pad)
        ax.set_ylim(mean - lim_pad, mean + lim_pad)
        ax.plot(ax.get_xlim(), ax.get_xlim(), color=GRID_LARGE, lw=0.6, zorder=0)

    ax.set_xlabel("RRₙ (ms)")
    ax.set_ylabel("RRₙ₊₁ (ms)")
    sd_txt = (
        f"SD1 {hrv.sd1_ms:.1f} ms · SD2 {hrv.sd2_ms:.1f} ms"
        if hrv.sd1_ms and hrv.sd2_ms
        else ""
    )
    fig.tight_layout()
    return _finish(fig, f"Poincaré plot with SD1/SD2 ellipse. {sd_txt}")


def rr_histogram(rr: RRSeries) -> Figure | None:
    if len(rr) < 20:
        return None
    fig, ax = plt.subplots(figsize=(3.6, 2.6), dpi=_DPI)
    fig.patch.set_facecolor(PAPER)
    _style_axes(ax)
    ax.hist(rr.rr_ms, bins=40, color=ACCENT, edgecolor=PAPER, lw=0.3)
    ax.set_xlabel("RR (ms)")
    ax.set_ylabel("beats")
    fig.tight_layout()
    return _finish(fig, "RR-interval distribution (analysed beats only).")


def sdnn_per_window(hrv: HRVResult) -> Figure | None:
    if not hrv.sdnn_per_window_ms:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 2.6), dpi=_DPI)
    fig.patch.set_facecolor(PAPER)
    _style_axes(ax)
    t = np.array(hrv.sdnn_window_t_s) / 60.0
    ax.bar(t, hrv.sdnn_per_window_ms, width=0.85, color=ACCENT, edgecolor=PAPER)
    if hrv.sdnn_ms:
        ax.axhline(
            hrv.sdnn_ms,
            color=WARN,
            lw=1.2,
            linestyle="--",
            label=f"whole-record (trend-inclusive) {hrv.sdnn_ms:.1f} ms",
        )
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.set_xlabel("minute")
    ax.set_ylabel("SDNN (ms)")
    fig.tight_layout()
    return _finish(
        fig,
        "Per-minute SDNN against the whole-record figure. A whole-record value far "
        "above the per-minute bars is trend, not beat-to-beat variability.",
    )


def spectrum(hrv: HRVResult) -> Figure | None:
    if not hrv.psd_freq_hz:
        return None
    f = np.array(hrv.psd_freq_hz)
    p = np.array(hrv.psd_ms2_per_hz)
    fig, ax = plt.subplots(figsize=(5.4, 2.8), dpi=_DPI)
    fig.patch.set_facecolor(PAPER)
    _style_axes(ax)

    for (lo, hi), color, name in (
        (VLF_BAND, GRID_LARGE, "VLF"),
        (LF_BAND, "#C7DAD2", "LF"),
        (HF_BAND, "#E4D6C4", "HF"),
    ):
        ax.axvspan(lo, hi, color=color, alpha=0.35)
        ax.text(
            (lo + hi) / 2, 0.95, name, transform=ax.get_xaxis_transform(),
            ha="center", fontsize=7, color=INK,
        )

    ax.plot(f, p, color=INK, lw=1.0)
    peak_txt = ""
    if hrv.lf_peak_hz:
        ax.axvline(hrv.lf_peak_hz, color=ACCENT, lw=1.0, linestyle=":")
        peak_txt = (
            f" Dominant LF-band peak at {hrv.lf_peak_hz:.3f} Hz — in the baroreflex "
            "Mayer-wave band; not a respiratory rate."
        )
    ax.set_xlim(0.0, 0.5)
    ax.set_xlabel("Hz")
    ax.set_ylabel("ms²/Hz")
    fig.tight_layout()
    return _finish(
        fig,
        f"RR spectrum ({hrv.psd_method}) with Task Force bands shaded. "
        f"Band powers are descriptive only.{peak_txt}",
    )


def beat_template(
    morph: BeatMorphology,
    ecg_clean: np.ndarray,
    fs_hz: float,
    peak_indices: np.ndarray,
    max_normal_beats: int = 40,
) -> Figure | None:
    """Template with a sample of normal beats faint and every outlier bold."""
    if len(morph.template) == 0:
        return None
    pre = int(round(-morph.template_t_ms[0] / 1000.0 * fs_hz))
    post = len(morph.template) - pre

    def beat_window(i: int) -> np.ndarray | None:
        p = int(peak_indices[i])
        lo, hi = p - pre, p + post
        if lo < 0 or hi > len(ecg_clean):
            return None
        return ecg_clean[lo:hi]

    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=_DPI)
    fig.patch.set_facecolor(PAPER)
    _style_axes(ax)

    normal = np.flatnonzero(~morph.outlier_mask)
    step = max(1, len(normal) // max_normal_beats)
    for i in normal[::step]:
        w = beat_window(int(i))
        if w is not None:
            ax.plot(morph.template_t_ms, w, color=INK, lw=0.4, alpha=0.12, zorder=1)

    outliers = np.flatnonzero(morph.outlier_mask)
    for i in outliers:
        w = beat_window(int(i))
        if w is not None:
            ax.plot(morph.template_t_ms, w, color=WARN, lw=0.9, alpha=0.7, zorder=3)

    ax.plot(
        morph.template_t_ms, morph.template, color=ACCENT, lw=2.0,
        label="median template", zorder=4,
    )
    ax.set_xlabel("ms from R")
    ax.set_ylabel("mV")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.tight_layout()

    caption = (
        f"Beat overlay: sampled normal beats (faint), morphology outliers "
        f"(r < {MORPHOLOGY_R_THRESHOLD:.2f}, red), median template (green). "
        f"{len(outliers)} outlier(s)"
    )
    caption += (
        f": {int(np.sum(morph.motion_explained))} coincide with motion/degraded "
        f"quality, {int(np.sum(morph.ectopy_candidate))} premature ectopy "
        "candidate(s)." if len(outliers) else "."
    )
    return _finish(fig, caption)


def quality_traces(quality: QualityResult) -> Figure | None:
    if not quality.windows:
        return None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 3.4), dpi=_DPI, sharex=True)
    fig.patch.set_facecolor(PAPER)
    t = [w.start_s / 60.0 for w in quality.windows]
    sqi = [w.sqi_mean for w in quality.windows]
    wander = [w.wander_rms_mv for w in quality.windows]

    for ax in (ax1, ax2):
        _style_axes(ax)
        for seg_start, seg_end, _r in quality.excluded_segments:
            ax.axvspan(seg_start / 60.0, seg_end / 60.0, color=WARN, alpha=0.15)

    ax1.plot(t, sqi, color=ACCENT, lw=1.0)
    ax1.set_ylabel("SQI")
    ax1.set_ylim(0, 1.05)
    ax2.plot(t, wander, color=INK, lw=1.0)
    ax2.set_ylabel("wander RMS (mV)")
    ax2.set_xlabel("minutes")
    fig.tight_layout()
    return _finish(
        fig,
        "Per-window signal quality index and baseline-wander RMS (0.7 Hz low-pass "
        "of the raw signal — the motion proxy). Shaded spans are excluded.",
    )
