"""
Unit tests for emitter.py serialisation.

Run with:  pytest test_emitter.py -v
"""

import struct
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trainer.emitter import serialise, fake_loss, fake_accuracy, fake_grad_norm, WIRE_FORMAT


class TestSerialise:
    def test_output_is_24_bytes(self):
        payload = serialise(0, 0.0, 0.0, 0.0, 0)
        assert len(payload) == 24

    def test_round_trip_step(self):
        payload = serialise(42, 1.5, 0.9, 0.3, 1234567890)
        step, loss, acc, grad, ts = struct.unpack(WIRE_FORMAT, payload)
        assert step == 42

    def test_round_trip_floats(self):
        payload = serialise(0, 1.234567, 0.876543, 2.345678, 0)
        _, loss, acc, grad, _ = struct.unpack(WIRE_FORMAT, payload)
        assert abs(loss - 1.234567) < 1e-5
        assert abs(acc  - 0.876543) < 1e-5
        assert abs(grad - 2.345678) < 1e-5

    def test_round_trip_timestamp(self):
        ts = 1_700_000_000_000_000_000
        payload = serialise(0, 0.0, 0.0, 0.0, ts)
        _, _, _, _, recv_ts = struct.unpack(WIRE_FORMAT, payload)
        assert recv_ts == ts

    def test_big_endian_byte_order(self):
        # Step=1 in big-endian should have bytes 0x00 0x00 0x00 0x01 at offset 0.
        payload = serialise(1, 0.0, 0.0, 0.0, 0)
        assert payload[0:4] == b'\x00\x00\x00\x01'

    def test_step_zero(self):
        payload = serialise(0, 0.0, 0.0, 0.0, 0)
        step, _, _, _, _ = struct.unpack(WIRE_FORMAT, payload)
        assert step == 0

    def test_max_step(self):
        max_step = 2**32 - 1
        payload  = serialise(max_step, 0.0, 0.0, 0.0, 0)
        step, _, _, _, _ = struct.unpack(WIRE_FORMAT, payload)
        assert step == max_step


class TestFakeCurves:
    def test_loss_is_positive(self):
        for step in [0, 100, 500, 999]:
            assert fake_loss(step, 1000) > 0

    def test_loss_generally_decreases(self):
        # Average early loss should exceed average late loss.
        early = sum(fake_loss(s, 1000) for s in range(0,  100)) / 100
        late  = sum(fake_loss(s, 1000) for s in range(900, 1000)) / 100
        assert early > late

    def test_accuracy_bounded(self):
        for loss in [0.01, 0.5, 1.0, 2.0]:
            acc = fake_accuracy(loss)
            assert 0.0 <= acc <= 1.0

    def test_grad_norm_non_negative(self):
        for loss in [0.01, 0.5, 1.5]:
            assert fake_grad_norm(loss) >= 0.0

    def test_no_nan_or_inf(self):
        for step in range(0, 1000, 50):
            loss = fake_loss(step, 1000)
            acc  = fake_accuracy(loss)
            grad = fake_grad_norm(loss)
            assert math.isfinite(loss)
            assert math.isfinite(acc)
            assert math.isfinite(grad)
