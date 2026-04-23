"""
Phase 2 tests: FlatBuffers serialisation round-trip + hook drop behaviour.

Run with:  pytest test_phase2.py -v
"""
from __future__ import annotations

import queue
import time
import threading

import flatbuffers
import pytest

# Adjust path so generated/ and serialise.py are importable.
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from serialise import encode
from generated.omnistream.TrainingEvent import TrainingEvent, CreateTrainingEvent
from hook import OmnistreamHook


# ── Serialise / deserialise round-trip ───────────────────────────────────

class TestFlatBuffersRoundTrip:

    def _decode(self, buf: bytes) -> TrainingEvent:
        return TrainingEvent.GetRootAs(bytearray(buf), 0)

    def test_step_round_trips(self):
        buf = encode(42, 1.0, 0.9, 0.5)
        ev  = self._decode(buf)
        assert ev.Step() == 42

    def test_loss_round_trips(self):
        buf = encode(0, 1.2345, 0.0, 0.0)
        ev  = self._decode(buf)
        assert abs(ev.Loss() - 1.2345) < 1e-4

    def test_accuracy_round_trips(self):
        buf = encode(0, 0.0, 0.876, 0.0)
        ev  = self._decode(buf)
        assert abs(ev.Accuracy() - 0.876) < 1e-4

    def test_grad_norm_round_trips(self):
        buf = encode(0, 0.0, 0.0, 3.14)
        ev  = self._decode(buf)
        assert abs(ev.GradNorm() - 3.14) < 1e-4

    def test_timestamp_round_trips(self):
        ts  = time.time_ns()
        buf = encode(0, 0.0, 0.0, 0.0, timestamp_ns=ts)
        ev  = self._decode(buf)
        assert ev.TimestampNs() == ts

    def test_timestamp_auto_filled(self):
        t0  = time.time_ns()
        buf = encode(0, 0.0, 0.0, 0.0)   # no explicit timestamp
        t1  = time.time_ns()
        ev  = self._decode(buf)
        assert t0 <= ev.TimestampNs() <= t1

    def test_output_is_bytes(self):
        assert isinstance(encode(0, 0.0, 0.0, 0.0), bytes)

    def test_output_is_valid_flatbuffer(self):
        buf = encode(99, 0.5, 0.8, 1.2)
        # Should not raise — if the buffer is malformed, GetRootAs will crash.
        ev  = self._decode(buf)
        assert ev.Step() == 99

    def test_zero_fields_encode_cleanly(self):
        buf = encode(0, 0.0, 0.0, 0.0, 0)
        ev  = self._decode(buf)
        assert ev.Step()      == 0
        assert ev.Loss()      == 0.0
        assert ev.Accuracy()  == 0.0
        assert ev.GradNorm()  == 0.0
        assert ev.TimestampNs() == 0

    def test_large_step_number(self):
        large = 2**31 - 1    # max int32 — well within uint32 range
        buf   = encode(large, 0.1, 0.9, 0.5)
        ev    = self._decode(buf)
        assert ev.Step() == large

    def test_successive_encodes_are_independent(self):
        buf1 = encode(1, 1.0, 0.5, 0.2)
        buf2 = encode(2, 2.0, 0.6, 0.3)
        ev1  = self._decode(buf1)
        ev2  = self._decode(buf2)
        assert ev1.Step() == 1
        assert ev2.Step() == 2
        assert abs(ev1.Loss() - 1.0) < 1e-4
        assert abs(ev2.Loss() - 2.0) < 1e-4


# ── Hook drop / non-blocking behaviour ───────────────────────────────────

class _NullHook(OmnistreamHook):
    """OmnistreamHook subclass that never sends to ZMQ — safe for unit tests."""

    def _send_loop(self) -> None:
        """Drain queue silently — no ZMQ context opened."""
        while True:
            try:
                item = self._queue.get(timeout=0.05)
                if item is None:
                    break
            except queue.Empty:
                if not self._running.is_set():
                    break


class TestHookBehaviour:

    def test_on_step_is_non_blocking(self):
        hook = _NullHook()
        hook.start()
        t0 = time.perf_counter()
        for i in range(100):
            hook.on_step(i, 0.5, 0.9, 1.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        hook.stop()
        # 100 on_step() calls should complete in well under 50ms.
        assert elapsed_ms < 50, f"on_step() took {elapsed_ms:.1f}ms for 100 calls"

    def test_drops_when_queue_full(self):
        hook = _NullHook(queue_maxsize=5)
        # Don't start the drain thread — queue fills up immediately.
        for i in range(20):
            hook.on_step(i, 0.5, 0.9, 1.0)
        assert hook.steps_dropped > 0

    def test_emitted_plus_dropped_equals_total(self):
        hook = _NullHook(queue_maxsize=10)
        total = 30
        for i in range(total):
            hook.on_step(i, 0.5, 0.9, 1.0)
        assert hook.steps_emitted + hook.steps_dropped == total

    def test_stop_flushes_queue(self):
        received: list[bytes] = []

        class _CollectHook(OmnistreamHook):
            def _send_loop(self_inner):
                while True:
                    try:
                        item = self_inner._queue.get(timeout=0.05)
                        if item is None:
                            break
                        received.append(item)
                    except queue.Empty:
                        if not self_inner._running.is_set():
                            break

        hook = _CollectHook(queue_maxsize=100)
        hook.start()
        N = 50
        for i in range(N):
            hook.on_step(i, float(i), 0.5, 1.0)
        hook.stop()
        assert len(received) == N

    def test_compute_grad_norm_returns_float(self):
        import torch
        import torch.nn as nn
        model = nn.Linear(4, 2)
        x     = torch.randn(8, 4)
        y     = torch.randint(0, 2, (8,))
        loss  = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        gn = OmnistreamHook.compute_grad_norm(model)
        assert isinstance(gn, float)
        assert gn > 0.0

    def test_compute_grad_norm_zero_without_backward(self):
        import torch.nn as nn
        model = nn.Linear(4, 2)
        # No backward called — gradients are None.
        gn = OmnistreamHook.compute_grad_norm(model)
        assert gn == 0.0
