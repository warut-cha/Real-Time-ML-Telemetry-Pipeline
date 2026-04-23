"""
OmniStream ML — Phase 1 dummy emitter.

Simulates a PyTorch training loop by generating realistic-looking loss/accuracy
curves and streaming them to the C++ engine via ZMQ PUSH socket.

Wire format (big-endian, 24 bytes):
    struct.pack(">IfffQ", step, loss, accuracy, grad_norm, timestamp_ns)

Phase 2 will replace this script with a real PyTorch training hook.
The ZMQ socket setup and wire format stay identical.

Usage:
    pip install pyzmq
    python emitter.py [--endpoint tcp://127.0.0.1:5555] [--hz 10] [--steps 1000]
"""

import argparse
import math
import random
import struct
import time
import zmq



# Fake training curve generators

def fake_loss(step: int, total_steps: int) -> float:
    """Decaying exponential with noise — looks like real training loss."""
    base      = 2.5 * math.exp(-4.0 * step / total_steps)
    noise     = random.gauss(0, 0.02)
    # Occasionally inject a gradient spike to give Phase 4 something to detect.
    spike     = 0.4 if random.random() < 0.005 else 0.0
    return max(0.01, base + noise + spike)


def fake_accuracy(loss: float) -> float:
    """Accuracy rises as loss falls — loosely correlated."""
    base  = 1.0 - loss / 3.0
    noise = random.gauss(0, 0.01)
    return min(0.999, max(0.0, base + noise))


def fake_grad_norm(loss: float) -> float:
    """Gradient norm loosely follows loss with occasional explosions."""
    base      = loss * 2.5
    noise     = random.gauss(0, 0.1)
    explosion = random.uniform(5.0, 15.0) if random.random() < 0.003 else 0.0
    return max(0.0, base + noise + explosion)


# Wire serialisation — MUST stay in sync with engine/src/zmq_receiver.cpp

WIRE_FORMAT = ">IfffQ"   # big-endian: uint32, float, float, float, uint64


def serialise(step: int, loss: float, accuracy: float,
              grad_norm: float, timestamp_ns: int) -> bytes:
    return struct.pack(WIRE_FORMAT, step, loss, accuracy, grad_norm, timestamp_ns)

# Latency probe helpers

def now_ns() -> int:
    return time.time_ns()


# Main

def main() -> None:
    parser = argparse.ArgumentParser(description="OmniStream ML dummy emitter")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555",
                        help="ZMQ PUSH endpoint (default: tcp://127.0.0.1:5555)")
    parser.add_argument("--hz",    type=float, default=10.0,
                        help="Emission rate in Hz (default: 10)")
    parser.add_argument("--steps", type=int,   default=2000,
                        help="Total training steps to simulate (default: 2000)")
    args = parser.parse_args()

    ctx  = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(args.endpoint)

    # Allow ZMQ to buffer up to 1000 messages if the engine is slow to connect.
    sock.setsockopt(zmq.SNDHWM, 1000)
    # Don't block on send — drop if HWM reached.
    sock.setsockopt(zmq.SNDTIMEO, 0)

    interval = 1.0 / args.hz
    print(f"[emitter] Connecting to {args.endpoint}")
    print(f"[emitter] Rate: {args.hz} Hz | Steps: {args.steps}")
    print("[emitter] Starting in 1 s — connect the engine now...")
    time.sleep(1.0)

    dropped = 0
    for step in range(args.steps):
        loss      = fake_loss(step, args.steps)
        accuracy  = fake_accuracy(loss)
        grad_norm = fake_grad_norm(loss)
        ts_ns     = now_ns()

        payload = serialise(step, loss, accuracy, grad_norm, ts_ns)

        try:
            sock.send(payload, zmq.NOBLOCK)
        except zmq.Again:
            dropped += 1

        if step % 50 == 0:
            print(f"  step={step:5d}  loss={loss:.4f}  acc={accuracy:.4f}"
                  f"  grad_norm={grad_norm:.4f}  dropped={dropped}")

        time.sleep(interval)

    print(f"\n[emitter] Done. {args.steps} steps emitted, {dropped} dropped.")
    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
