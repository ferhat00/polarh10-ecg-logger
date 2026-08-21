"""The processing queue: one worker, serialised, deduplicated.

A night is a large CSV and, with the external staging engine configured, a
subprocess that can run for half an hour. Bulk re-analysis of a sleep
history must therefore queue, not fan out — otherwise N nights means N
concurrent pipelines contending for the same cores and the same SQLite file.
"""

from __future__ import annotations

import threading

import pytest
from flask import Flask

from app import processing


@pytest.fixture()
def drained() -> None:
    """Leave the module-level queue as clean as it was found."""
    yield
    with processing._LOCK:
        processing._QUEUED.clear()
    while not processing._QUEUE.empty():
        processing._QUEUE.get_nowait()
        processing._QUEUE.task_done()


class TestSyncMode:
    def test_process_sync_bypasses_the_queue(
        self, app: Flask, monkeypatch: pytest.MonkeyPatch, drained: None
    ) -> None:
        """Tests must stay inline and deterministic — no threads, no queue."""
        seen: list[int] = []
        monkeypatch.setattr(processing, "_process", lambda a, sid: seen.append(sid))
        app.config["PROCESS_SYNC"] = True

        processing.submit_processing(app, 7)
        assert seen == [7]
        assert processing.queue_depth() == 0


class TestQueueing:
    def test_sessions_queue_and_drain_in_order(
        self, app: Flask, monkeypatch: pytest.MonkeyPatch, drained: None
    ) -> None:
        app.config["PROCESS_SYNC"] = False
        order: list[int] = []
        overlapped = threading.Event()
        busy = threading.Lock()

        def slow(_app: Flask, session_id: int) -> None:
            if not busy.acquire(blocking=False):
                overlapped.set()  # two sessions processing at once
                return
            try:
                order.append(session_id)
            finally:
                busy.release()

        monkeypatch.setattr(processing, "_process", slow)

        for session_id in (1, 2, 3):
            processing.submit_processing(app, session_id)
        processing._QUEUE.join()

        assert order == [1, 2, 3]
        assert not overlapped.is_set(), "the worker must process one at a time"
        assert processing.queue_depth() == 0

    def test_a_session_queued_twice_processes_once(
        self, app: Flask, monkeypatch: pytest.MonkeyPatch, drained: None
    ) -> None:
        """A double-click, or a night ticked in a batch it is already in."""
        app.config["PROCESS_SYNC"] = False
        gate = threading.Event()
        seen: list[int] = []

        def blocking(_app: Flask, session_id: int) -> None:
            seen.append(session_id)
            gate.wait(timeout=5.0)

        monkeypatch.setattr(processing, "_process", blocking)

        processing.submit_processing(app, 11)
        processing.submit_processing(app, 11)  # still queued — ignored
        assert processing.queue_depth() == 1
        gate.set()
        processing._QUEUE.join()
        assert seen == [11]

    def test_a_failing_session_does_not_kill_the_worker(
        self, app: Flask, monkeypatch: pytest.MonkeyPatch, drained: None
    ) -> None:
        app.config["PROCESS_SYNC"] = False
        seen: list[int] = []

        def flaky(_app: Flask, session_id: int) -> None:
            seen.append(session_id)
            if session_id == 1:
                raise RuntimeError("pipeline exploded")

        monkeypatch.setattr(processing, "_process", flaky)

        processing.submit_processing(app, 1)
        processing.submit_processing(app, 2)
        processing._QUEUE.join()
        assert seen == [1, 2]
        assert processing.queue_depth() == 0
