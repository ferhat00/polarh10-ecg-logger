"""External 5-class engine adapter: status reasons, h5 round trip via a
stub scorer script, preprocessing math, and failure handling.

No code from the AGPL tool is used anywhere — the stub scorer below stands
in for it, asserting the adapter's side of the documented interface.
"""

from __future__ import annotations

import datetime as dt
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from app.sleep.engines import external_ecg_staging as ext
from app.sleep.stages import UNSCORED, StageVocab

START = dt.datetime(2026, 8, 10, 22, 30, tzinfo=dt.UTC)

h5py = pytest.importorskip("h5py", reason="h5 exchange needs the optional extras")


#: A stand-in "clone": reads input.h5, checks the contract, writes results.h5
#: with a deterministic stage pattern (epoch index mod 5).
STUB_SCORER = textwrap.dedent(
    """
    import sys
    import h5py
    import numpy as np

    path = sys.argv[1]
    with h5py.File(path, "r") as f:
        ecgs = f["ecgs"][:]
        demographics = f["demographics"][:]
        midnight_offset = float(np.asarray(f["midnight_offset"]))
    assert ecgs.ndim == 2 and ecgs.shape[1] == 7680, ecgs.shape
    assert ecgs.dtype == np.float32
    assert float(np.max(np.abs(ecgs))) <= 1.0
    assert demographics.shape == (2, 1)
    assert -1.0 <= midnight_offset <= 1.0
    stages = (np.arange(len(ecgs)) % 5).astype(np.int64)
    out = path.rsplit("input.h5", 1)[0] + "results.h5"
    with h5py.File(out, "w") as f:
        f.create_dataset("predicted_stages", data=stages)
    """
)


def _make_clone(tmp_path: Path) -> Path:
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "train.py").write_text(STUB_SCORER, encoding="utf-8")
    return clone


def _config(clone: Path) -> dict:
    return {
        "SLEEP_EXTERNAL_DIR": str(clone),
        "SLEEP_EXTERNAL_PYTHON": sys.executable,
        "SLEEP_EXTERNAL_TIMEOUT_S": 60,
    }


def _synthetic_ecg(duration_s: float, fs_hz: float = 130.03):
    from tests.synth_util import build_ecg_from_rr

    n_beats = int(duration_s / 0.9) + 2
    rr_ms = np.full(n_beats, 900.0)
    t, ecg, beats = build_ecg_from_rr(rr_ms, fs_hz=fs_hz)
    keep = t <= duration_s
    return ecg[keep], beats[beats <= duration_s]


class TestStatus:
    def test_unconfigured(self) -> None:
        st = ext.status({})
        assert st.available is False
        assert "ECGLOG_SLEEP_EXTERNAL_DIR" in st.unavailable_reason

    def test_missing_dir_and_python(self, tmp_path: Path) -> None:
        st = ext.status(
            {
                "SLEEP_EXTERNAL_DIR": str(tmp_path / "nope"),
                "SLEEP_EXTERNAL_PYTHON": str(tmp_path / "nopython"),
            }
        )
        assert st.available is False
        assert "not a directory" in st.unavailable_reason
        assert "does not exist" in st.unavailable_reason

    def test_dir_without_train_py(self, tmp_path: Path) -> None:
        st = ext.status(
            {
                "SLEEP_EXTERNAL_DIR": str(tmp_path),
                "SLEEP_EXTERNAL_PYTHON": sys.executable,
            }
        )
        assert st.available is False
        assert "train.py" in st.unavailable_reason

    def test_configured_ok(self, tmp_path: Path) -> None:
        assert ext.status(_config(_make_clone(tmp_path))).available is True


class TestPreprocessing:
    def test_resample_length_and_bounds(self) -> None:
        ecg, beats = _synthetic_ecg(90.0)
        x = ext.preprocess_ecg(ecg, 130.03, beats)
        expected = len(ecg) * 256.0 / 130.03
        assert abs(len(x) - expected) <= 2
        assert float(np.max(np.abs(x))) <= 1.0
        # Median-centred.
        assert abs(float(np.median(x))) < 0.01

    def test_r_peak_amplitude_near_half(self) -> None:
        ecg, beats = _synthetic_ecg(90.0)
        x = ext.preprocess_ecg(ecg, 130.03, beats)
        idx = np.round(beats * 256.0).astype(int)
        idx = idx[idx < len(x)]
        peak_amps = np.array(
            [np.max(np.abs(x[max(0, i - 100) : i + 100])) for i in idx]
        )
        # 90th percentile of per-beat peaks scaled to ~0.5.
        assert 0.4 < float(np.percentile(peak_amps, 90)) < 0.6


class TestRoundTrip:
    def test_stub_scorer_round_trip(self, tmp_path: Path) -> None:
        clone = _make_clone(tmp_path)
        duration = 8 * 30.0  # 8 full epochs
        ecg, beats = _synthetic_ecg(duration)
        hyp = ext.stage_external(
            ecg,
            130.03,
            beats,
            START,
            duration,
            age_years=40.0,
            sex="male",
            stored_path=str(tmp_path / "session.csv"),
            config=_config(clone),
        )
        assert hyp.vocab == StageVocab.AASM_5
        scored = hyp.stages[hyp.stages != UNSCORED]
        # The stub writes epoch_index % 5.
        assert scored.tolist() == [(i % 5) for i in range(len(scored))]
        assert len(scored) >= 7
        # Exchange dir removed on success.
        assert not (tmp_path / "session.csv.sleep_external").exists()

    def test_collapses_into_orchestrator_vocab(self, tmp_path: Path) -> None:
        clone = _make_clone(tmp_path)
        duration = 5 * 30.0
        ecg, beats = _synthetic_ecg(duration)
        hyp = ext.stage_external(
            ecg, 130.03, beats, START, duration, None, None,
            str(tmp_path / "s.csv"), _config(clone),
        )
        four = hyp.collapsed(StageVocab.WAKE_LIGHT_DEEP_REM)
        # 0,1,2,3,4 -> W, Light, Light, Deep, REM
        scored = four.stages[four.stages != UNSCORED]
        assert scored.tolist()[:5] == [0, 1, 1, 2, 3][: len(scored)]
        assert any("demographics" in n for n in hyp.notes)  # defaulted

    def test_nonzero_exit_keeps_exchange_dir(self, tmp_path: Path) -> None:
        clone = tmp_path / "clone"
        clone.mkdir()
        (clone / "train.py").write_text(
            "import sys; sys.stderr.write('boom'); sys.exit(3)", encoding="utf-8"
        )
        ecg, beats = _synthetic_ecg(60.0)
        with pytest.raises(RuntimeError, match="exited with 3"):
            ext.stage_external(
                ecg, 130.03, beats, START, 60.0, None, None,
                str(tmp_path / "s.csv"), _config(clone),
            )
        assert (tmp_path / "s.csv.sleep_external" / "input.h5").exists()

    def test_missing_results_file_reported(self, tmp_path: Path) -> None:
        clone = tmp_path / "clone"
        clone.mkdir()
        (clone / "train.py").write_text("pass", encoding="utf-8")
        ecg, beats = _synthetic_ecg(60.0)
        with pytest.raises(RuntimeError, match="no results.h5"):
            ext.stage_external(
                ecg, 130.03, beats, START, 60.0, None, None,
                str(tmp_path / "s.csv"), _config(clone),
            )

    def test_unrecognised_results_dataset(self, tmp_path: Path) -> None:
        path = tmp_path / "results.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("mystery", data=np.zeros(4))
        with pytest.raises(ValueError, match="mystery"):
            ext.parse_results_h5(path, 4)

    def test_epoch_count_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "results.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("stages", data=np.zeros(3, dtype=np.int64))
        with pytest.raises(ValueError, match="expected 4"):
            ext.parse_results_h5(path, 4)


class TestMidnightOffset:
    def test_offset_encoding(self, tmp_path: Path) -> None:
        ecg, beats = _synthetic_ecg(60.0)
        out = tmp_path / "input.h5"
        # 22:30 UTC == -1.5 h from midnight in UTC; the adapter uses the
        # local clock, so compute the expectation the same way.
        local = START.astimezone()
        seconds = local.hour * 3600 + local.minute * 60 + local.second
        if seconds > 43200:
            seconds -= 86400
        expected = seconds / 43200.0
        ext.export_input_h5(ecg, 130.03, beats, START, 40.0, "female", out)
        with h5py.File(out, "r") as f:
            assert float(np.asarray(f["midnight_offset"])) == pytest.approx(
                expected, abs=1e-6
            )
            demo = f["demographics"][:]
        assert demo[0, 0] == 0.0  # female
        assert demo[1, 0] == pytest.approx(0.40)
