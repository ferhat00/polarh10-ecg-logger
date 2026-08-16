"""Loader tests: ragged rows, epoch detection, unit auto-detect, sampling rate.

A wrong unit or epoch assumption corrupts everything downstream, so every
detection branch — including every refuse-to-guess path — is pinned here.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pytest

from app.ingest.exceptions import AmbiguousFormatError, LoaderError
from app.ingest.loader import (
    POLAR_EPOCH_2000_UNIX_S,
    FormatOverrides,
    load_polar_csv,
)

#: A fixed "now" so epoch plausibility windows are deterministic in tests.
NOW = dt.datetime(2026, 8, 16, 12, 0, tzinfo=dt.UTC)
START = dt.datetime(2026, 8, 14, 10, 0, tzinfo=dt.UTC)


def make_csv(
    tmp_path: Path,
    name: str = "ecg_20260814.csv",
    n: int = 400,
    fs: float = 130.03,
    start: dt.datetime = START,
    epoch: str = "unix",
    amplitude: float = 0.5,
    header: str = "time,ecg,hr,rr,marker",
    delim: str = ",",
    decimal: str = ".",
    hr_every: int = 130,
    rr_every: int = 115,
    marker_at: dict[int, str] | None = None,
    extra_lines: list[str] | None = None,
) -> Path:
    """Write a synthetic export mimicking the verified Polar shape."""
    start_unix_ns = int(start.timestamp() * 1e9)
    if epoch == "polar2000":
        start_ns = start_unix_ns - POLAR_EPOCH_2000_UNIX_S * 10**9
    else:
        start_ns = start_unix_ns

    lines = [header]
    for i in range(n):
        t = start_ns + round(i * 1e9 / fs)
        ecg = amplitude * math.sin(2 * math.pi * i / 100)
        val = f"{ecg:.3f}"
        if decimal == ",":
            val = val.replace(".", ",")
        fields = [str(t), val]
        # Ragged rows exactly like the real export: hr appended on some rows
        # (3 fields), rr in the 4th position on others (4 fields).
        if rr_every and i % rr_every == 0 and i > 0:
            fields += ["", "912"]
        elif hr_every and i % hr_every == 0 and i > 0:
            fields.append("69")
        if marker_at and i in marker_at:
            while len(fields) < 5:
                fields.append("")
            fields[4] = marker_at[i]
        lines.append(delim.join(fields))
    if extra_lines:
        lines.extend(extra_lines)
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class TestRaggedParsing:
    def test_ragged_rows_parse_and_streams_extracted(self, tmp_path: Path) -> None:
        path = make_csv(tmp_path, marker_at={200: "lap"})
        rec = load_polar_csv(path, now=NOW)
        assert len(rec.ecg_mv) == 400
        # hr rows exist but rr rows take precedence at overlaps; both sparse.
        assert len(rec.device_hr_bpm) >= 2
        assert np.all(rec.device_hr_bpm == 69)
        assert len(rec.device_rr_ms) >= 2
        assert np.all(rec.device_rr_ms == 912)
        assert rec.markers == [(pytest.approx(200 / 130.03, abs=0.01), "lap")]

    def test_five_field_row_accepted(self, tmp_path: Path) -> None:
        path = make_csv(tmp_path, marker_at={100: "transition:standing"})
        rec = load_polar_csv(path, now=NOW)
        assert rec.markers[0][1] == "transition:standing"

    def test_row_wider_than_header_raises(self, tmp_path: Path) -> None:
        start_ns = int(START.timestamp() * 1e9)
        path = make_csv(
            tmp_path, extra_lines=[f"{start_ns + 10**9},0.1,69,912,lap,EXTRA"]
        )
        with pytest.raises(AmbiguousFormatError) as exc:
            load_polar_csv(path, now=NOW)
        assert exc.value.questions[0].key == "columns"

    def test_minimal_two_column_header(self, tmp_path: Path) -> None:
        path = make_csv(tmp_path, header="time,ecg", hr_every=0, rr_every=0)
        rec = load_polar_csv(path, now=NOW)
        assert len(rec.device_hr_bpm) == 0
        assert len(rec.device_rr_ms) == 0


class TestEpochDetection:
    def test_unix_epoch_detected(self, tmp_path: Path) -> None:
        rec = load_polar_csv(make_csv(tmp_path, epoch="unix"), now=NOW)
        assert rec.epoch_detected == "unix"
        assert abs((rec.start_time - START).total_seconds()) < 1

    def test_polar2000_epoch_detected(self, tmp_path: Path) -> None:
        rec = load_polar_csv(make_csv(tmp_path, epoch="polar2000"), now=NOW)
        assert rec.epoch_detected == "polar2000"
        assert abs((rec.start_time - START).total_seconds()) < 1

    def test_filename_date_agreement_noted(self, tmp_path: Path) -> None:
        rec = load_polar_csv(make_csv(tmp_path, name="ecg_2026-08-14.csv"), now=NOW)
        assert any("agrees" in n for n in rec.notes)

    def test_filename_date_mismatch_warned_not_fatal(self, tmp_path: Path) -> None:
        rec = load_polar_csv(make_csv(tmp_path, name="ecg_2024-01-01.csv"), now=NOW)
        assert rec.epoch_detected == "unix"
        assert any("differs" in n for n in rec.notes)

    def test_implausible_timestamps_raise(self, tmp_path: Path) -> None:
        # Second-resolution timestamps: implausible under both ns epochs.
        lines = ["time,ecg"] + [
            f"{1_700_000_000 + i},{0.4 * math.sin(i / 10):.3f}" for i in range(400)
        ]
        path = tmp_path / "weird.csv"
        path.write_text("\n".join(lines), encoding="utf-8")
        with pytest.raises(AmbiguousFormatError) as exc:
            load_polar_csv(path, now=NOW)
        assert exc.value.questions[0].key == "epoch"

    def test_epoch_override_respected(self, tmp_path: Path) -> None:
        path = make_csv(tmp_path, epoch="polar2000")
        rec = load_polar_csv(path, overrides=FormatOverrides(epoch="polar2000"), now=NOW)
        assert rec.epoch_detected == "polar2000"

    def test_future_recording_implausible(self, tmp_path: Path) -> None:
        future = dt.datetime(2056, 1, 1, tzinfo=dt.UTC)
        with pytest.raises(AmbiguousFormatError):
            load_polar_csv(make_csv(tmp_path, start=future, name="f.csv"), now=NOW)


class TestUnitDetection:
    def test_mv_branch(self, tmp_path: Path) -> None:
        rec = load_polar_csv(make_csv(tmp_path, amplitude=0.5), now=NOW)
        assert rec.ecg_unit_detected == "mV"
        assert rec.p99_abs_amplitude < 5
        assert float(np.max(np.abs(rec.ecg_mv))) == pytest.approx(0.5, abs=0.01)
        assert any("millivolts" in n for n in rec.notes)

    def test_uv_branch_divides_by_1000(self, tmp_path: Path) -> None:
        rec = load_polar_csv(make_csv(tmp_path, amplitude=500.0), now=NOW)
        assert rec.ecg_unit_detected == "uV"
        # 500 µV amplitude → 0.5 mV after conversion.
        assert float(np.max(np.abs(rec.ecg_mv))) == pytest.approx(0.5, abs=0.01)
        assert any("microvolts" in n for n in rec.notes)

    def test_ambiguous_amplitude_raises(self, tmp_path: Path) -> None:
        # p99 ≈ 10: too big for chest-strap mV, too small for µV R-waves.
        with pytest.raises(AmbiguousFormatError) as exc:
            load_polar_csv(make_csv(tmp_path, amplitude=10.0), now=NOW)
        assert exc.value.questions[0].key == "ecg_unit"

    def test_unit_override_respected(self, tmp_path: Path) -> None:
        path = make_csv(tmp_path, amplitude=10.0)
        rec = load_polar_csv(path, overrides=FormatOverrides(ecg_unit="mV"), now=NOW)
        assert rec.ecg_unit_detected == "mV"
        assert float(np.max(np.abs(rec.ecg_mv))) == pytest.approx(10.0, abs=0.1)


class TestSamplingRate:
    def test_derived_not_hardcoded(self, tmp_path: Path) -> None:
        rec = load_polar_csv(make_csv(tmp_path, fs=130.03, n=2000), now=NOW)
        assert rec.sampling_rate_hz == pytest.approx(130.03, abs=0.005)
        assert rec.sampling_rate_hz != 130.0
        assert rec.sample_interval_std_ms < 0.01

    def test_different_rate_detected(self, tmp_path: Path) -> None:
        rec = load_polar_csv(make_csv(tmp_path, fs=129.7, n=2000), now=NOW)
        assert rec.sampling_rate_hz == pytest.approx(129.7, abs=0.005)

    def test_implausible_rate_raises(self, tmp_path: Path) -> None:
        with pytest.raises(LoaderError, match="sampling rate"):
            load_polar_csv(make_csv(tmp_path, fs=10.0, n=400), now=NOW)

    def test_duration(self, tmp_path: Path) -> None:
        rec = load_polar_csv(make_csv(tmp_path, n=1301), now=NOW)
        assert rec.duration_s == pytest.approx(1300 / 130.03, abs=0.01)


class TestDelimiterAndDecimal:
    def test_semicolon_raises_then_loads_with_overrides(self, tmp_path: Path) -> None:
        path = make_csv(
            tmp_path,
            header="time;ecg;hr;rr;marker",
            delim=";",
            decimal=",",
            name="semi.csv",
        )
        with pytest.raises(AmbiguousFormatError) as exc:
            load_polar_csv(path, now=NOW)
        assert exc.value.questions[0].key == "delimiter"

        # Delimiter answered but decimal commas remain → next specific question.
        with pytest.raises(AmbiguousFormatError) as exc2:
            load_polar_csv(path, overrides=FormatOverrides(delimiter=";"), now=NOW)
        assert exc2.value.questions[0].key == "decimal"

        rec = load_polar_csv(
            path, overrides=FormatOverrides(delimiter=";", decimal=","), now=NOW
        )
        assert rec.ecg_unit_detected == "mV"
        assert len(rec.ecg_mv) == 400

    def test_garbage_rows_raise(self, tmp_path: Path) -> None:
        path = make_csv(tmp_path, extra_lines=["not,numbers"] * 50)
        with pytest.raises(AmbiguousFormatError):
            load_polar_csv(path, now=NOW)


class TestHeaderMapping:
    def test_unrecognised_columns_raise_with_mapping_question(self, tmp_path: Path) -> None:
        path = make_csv(tmp_path, header="Timestamp,Voltage,hr,rr,marker")
        with pytest.raises(AmbiguousFormatError) as exc:
            load_polar_csv(path, now=NOW)
        q = exc.value.questions[0]
        assert q.key == "columns"
        assert "Timestamp" in q.observed

    def test_column_map_override_resolves(self, tmp_path: Path) -> None:
        path = make_csv(tmp_path, header="Timestamp,Voltage,hr,rr,marker")
        rec = load_polar_csv(
            path,
            overrides=FormatOverrides(column_map={"Timestamp": "time", "Voltage": "ecg"}),
            now=NOW,
        )
        assert len(rec.ecg_mv) == 400

    def test_headerless_file_raises(self, tmp_path: Path) -> None:
        start_ns = int(START.timestamp() * 1e9)
        lines = [f"{start_ns + round(i * 1e9 / 130.03)},0.1" for i in range(400)]
        path = tmp_path / "nohdr.csv"
        path.write_text("\n".join(lines), encoding="utf-8")
        with pytest.raises(AmbiguousFormatError) as exc:
            load_polar_csv(path, now=NOW)
        assert "header" in exc.value.questions[0].question


class TestUnusableFiles:
    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        with pytest.raises(LoaderError, match="empty"):
            load_polar_csv(path, now=NOW)

    def test_header_only(self, tmp_path: Path) -> None:
        path = tmp_path / "hdr.csv"
        path.write_text("time,ecg,hr,rr,marker\n", encoding="utf-8")
        with pytest.raises(LoaderError, match="no data rows"):
            load_polar_csv(path, now=NOW)

    def test_too_short(self, tmp_path: Path) -> None:
        path = make_csv(tmp_path, n=20)
        with pytest.raises(LoaderError, match="too short"):
            load_polar_csv(path, now=NOW)


class TestErrorSerialisation:
    def test_ambiguous_error_as_dict(self, tmp_path: Path) -> None:
        path = make_csv(tmp_path, amplitude=10.0)
        with pytest.raises(AmbiguousFormatError) as exc:
            load_polar_csv(path, now=NOW)
        payload = exc.value.as_dict()
        assert payload["questions"][0]["key"] == "ecg_unit"
        assert payload["questions"][0]["options"] == ["mV", "uV"]
        assert "message" in payload
