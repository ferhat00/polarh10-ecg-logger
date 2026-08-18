"""ACC loader tests: detection, honest errors, no guessed units."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from app.ingest.acc_loader import load_polar_acc_csv
from app.ingest.exceptions import LoaderError
from tests.synth_util import make_acc_csv_bytes

START = dt.datetime(2026, 8, 14, 10, 0, tzinfo=dt.UTC)
NOW = dt.datetime(2026, 8, 15, 12, 0, tzinfo=dt.UTC)


def _write(tmp_path: Path, payload: bytes, name: str = "acc.csv") -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


class TestHappyPath:
    def test_polar_shape_parses(self, tmp_path: Path) -> None:
        path = _write(tmp_path, make_acc_csv_bytes(duration_s=10.0, fs_hz=50.0))
        acc = load_polar_acc_csv(path, now=NOW)
        assert abs(acc.sampling_rate_hz - 50.0) < 0.5
        assert acc.epoch_detected == "unix"
        assert abs((acc.start_time - START).total_seconds()) < 1
        assert len(acc.time_s) == 500
        # Gravity magnitude ~ sqrt(120^2 + 990^2) ≈ 997 mg on average.
        mag = np.sqrt(acc.x_mg**2 + acc.y_mg**2 + acc.z_mg**2)
        assert 900 < float(np.mean(mag)) < 1100

    def test_decimal_comma_variant(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path, make_acc_csv_bytes(duration_s=10.0, fs_hz=50.0, decimal=",")
        )
        acc = load_polar_acc_csv(path, now=NOW)
        assert any("Decimal commas" in n for n in acc.notes)
        assert 900 < float(np.mean(np.abs(acc.z_mg))) < 1100

    def test_generic_time_xyz_shape(self, tmp_path: Path) -> None:
        t_ns = (int(START.timestamp() * 1e9) + np.arange(200) * 20_000_000).astype(
            np.int64
        )
        lines = ["time,x,y,z"] + [
            f"{t},10.0,20.0,990.0" for t in t_ns
        ]
        path = _write(tmp_path, ("\n".join(lines)).encode())
        acc = load_polar_acc_csv(path, now=NOW)
        assert abs(acc.sampling_rate_hz - 50.0) < 0.5
        assert np.allclose(acc.z_mg, 990.0)


class TestErrors:
    def test_wrong_header_lists_observed_and_expected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, b"a;b;c\n1;2;3\n")
        with pytest.raises(LoaderError, match="Found columns: a, b, c"):
            load_polar_acc_csv(path, now=NOW)

    def test_too_short_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path, make_acc_csv_bytes(duration_s=1.0, fs_hz=50.0))
        with pytest.raises(LoaderError, match="too short"):
            load_polar_acc_csv(path, now=NOW)

    def test_empty_file(self, tmp_path: Path) -> None:
        path = _write(tmp_path, b"")
        with pytest.raises(LoaderError, match="empty"):
            load_polar_acc_csv(path, now=NOW)

    def test_implausible_timestamps_raise(self, tmp_path: Path) -> None:
        # Millisecond timestamps: not plausible under either ns epoch.
        t_ms = (int(START.timestamp() * 1e3) + np.arange(200) * 20).astype(np.int64)
        lines = ["time,x,y,z"] + [f"{t},0,0,990" for t in t_ms]
        path = _write(tmp_path, ("\n".join(lines)).encode())
        with pytest.raises(LoaderError, match="epoch"):
            load_polar_acc_csv(path, now=NOW)

    def test_nonnumeric_axis_raises(self, tmp_path: Path) -> None:
        start_ns = int(START.timestamp() * 1e9)
        lines = ["time,x,y,z"] + [
            f"{start_ns + i * 20_000_000},oops,0,990" for i in range(200)
        ]
        path = _write(tmp_path, ("\n".join(lines)).encode())
        with pytest.raises(LoaderError, match="not numeric"):
            load_polar_acc_csv(path, now=NOW)

    def test_implausible_rate_raises(self, tmp_path: Path) -> None:
        # 1 Hz "ACC" — below the plausible window.
        t_ns = (int(START.timestamp() * 1e9) + np.arange(200) * 10**9).astype(np.int64)
        lines = ["time,x,y,z"] + [f"{t},0,0,990" for t in t_ns]
        path = _write(tmp_path, ("\n".join(lines)).encode())
        with pytest.raises(LoaderError, match="outside the plausible"):
            load_polar_acc_csv(path, now=NOW)
