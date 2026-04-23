#!/usr/bin/env python3
"""
Measures the cost of OmnistreamHook.on_step() as a percentage of a
realistic training step, without requiring a GPU or a running engine.

Methodology
-----------
1. Run N steps of a realistic dummy training loop (forward + backward
   on a small model) WITHOUT the hook → baseline_ms per step.
2. Run the same loop WITH the hook (ZMQ send disabled via --no-zmq) →
   hooked_ms per step.
3. Report overhead = (hooked_ms - baseline_ms) / baseline_ms * 100.

Target: overhead < 1% of step time.

Usage (Example)
-----
    python benchmark_hook.py [--steps 500] [--batch-size 64] [--no-zmq]
"""

from __future__ import annotations

import argparse
import statistics
import time
import queue
import threading

import torch
import torch.nn as nn
import torch.optim as optim

from serialise import encode


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


#Stub hook that serialises but does NOT send over ZMQ

class _StubHook:
    """
    Mirrors OmnistreamHook.on_step() exactly, but drops the buffer instead
    of pushing to ZMQ.  Isolates serialisation overhead from network I/O.
    """
    def __init__(self) -> None:
        self.steps_emitted = 0
        self.steps_dropped = 0
        self._q: queue.Queue[bytes | None] = queue.Queue(maxsize=1000)
        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()

    def on_step(self, step: int, loss: float, acc: float, gn: float) -> None:
        buf = encode(step, loss, acc, gn)
        try:
            self._q.put_nowait(buf)
            self.steps_emitted += 1
        except queue.Full:
            self.steps_dropped += 1

    def stop(self) -> None:
        self._q.put(None)
        self._t.join(timeout=2.0)

    def _drain(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            # Discard, measuring serialise() + queue overhead only.


def run_loop(
    steps:      int,
    batch_size: int,
    hook:       _StubHook | None,
) -> list[float]:
    """
    Returns a list of per-step wall-clock times in microseconds.
    """
    device    = torch.device("cpu")
    model     = TinyModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    times: list[float] = []

    for step in range(steps):
        x = torch.randn(batch_size, 784)
        y = torch.randint(0, 10, (batch_size,))

        t0 = time.perf_counter()

        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()

        grad_norm = 0.0
        if hook:
            grads     = [p.grad for p in model.parameters() if p.grad is not None]
            grad_norm = torch.norm(
                torch.stack([torch.norm(g.detach(), 2.0) for g in grads]), 2.0
            ).item()

        optimizer.step()

        if hook:
            preds    = logits.argmax(dim=1)
            accuracy = (preds == y).float().mean().item()
            hook.on_step(step, loss.item(), accuracy, grad_norm)

        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000000)   # ms

    return times


def print_stats(label: str, times: list[float]) -> None:
    s = sorted(times)
    n = len(s)
    print(f"\n  {label}")
    print(f"    n        : {n}")
    print(f"    mean     : {statistics.mean(s):.1f} µs")
    print(f"    median   : {statistics.median(s):.1f} µs")
    print(f"    p95      : {s[int(n * 0.95)]:.1f} µs")
    print(f"    p99      : {s[int(n * 0.99)]:.1f} µs")
    print(f"    max      : {max(s):.1f} µs")


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniStream hook overhead benchmark")
    parser.add_argument("--steps",      type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup",     type=int, default=20,
                        help="Steps to discard before measuring (default: 20)")
    args = parser.parse_args()

    print(f"[bench] steps={args.steps} batch_size={args.batch_size} warmup={args.warmup}")
    print("[bench] Running baseline (no hook)...")

    baseline_times_raw = run_loop(args.steps, args.batch_size, hook=None)
    baseline_times     = baseline_times_raw[args.warmup:]

    print("[bench] Running hooked loop (serialise + queue, no ZMQ)...")
    hook               = _StubHook()
    hooked_times_raw   = run_loop(args.steps, args.batch_size, hook=hook)
    hooked_times       = hooked_times_raw[args.warmup:]
    hook.stop()

    print_stats("Baseline (no hook)", baseline_times)
    print_stats("Hooked   (serialise + queue)", hooked_times)

    baseline_mean = statistics.mean(baseline_times)
    hooked_mean   = statistics.mean(hooked_times)
    overhead_pct  = (hooked_mean - baseline_mean) / baseline_mean * 100

    print(f"\n  Overhead : {overhead_pct:+.2f}%  "
          f"({' under 1%' if overhead_pct < 1.0 else 'exceeds 1% target'})")
    print(f"  Emitted  : {hook.steps_emitted}  Dropped: {hook.steps_dropped}")


if __name__ == "__main__":
    main()