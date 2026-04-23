#!/usr/bin/env python3
"""
OmniStream ML — Phase 1 latency probe.

Sends a single timestamped message to the engine and measures round-trip
time via a separate ZMQ REQ/REP socket (requires the engine to be running
with --probe-port enabled in Phase 3; for now, measure one-way emit latency).

Usage:
    python latency_probe.py [--n 200]
"""

import argparse
import statistics
import struct
import time
import zmq

WIRE_FORMAT = ">IfffQ"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of probe messages (default: 200)")
    args = parser.parse_args()

    ctx  = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(args.endpoint)
    sock.setsockopt(zmq.SNDTIMEO, 100)

    latencies_us: list[float] = []

    print(f"[probe] Sending {args.n} probe messages to {args.endpoint}")
    time.sleep(0.5)

    for i in range(args.n):
        t0  = time.perf_counter()
        ts  = time.time_ns()
        msg = struct.pack(WIRE_FORMAT, i, 0.5, 0.5, 1.0, ts)
        try:
            sock.send(msg, zmq.NOBLOCK)
        except zmq.Again:
            print(f"  dropped message {i}")
            continue
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1_000_000)
        time.sleep(0.005)

    sock.close()
    ctx.term()

    if not latencies_us:
        print("[probe] No messages sent successfully.")
        return

    print(f"\n[probe] Python-side send latency over {len(latencies_us)} messages:")
    print(f"  min    : {min(latencies_us):.1f} µs")
    print(f"  median : {statistics.median(latencies_us):.1f} µs")
    print(f"  p99    : {sorted(latencies_us)[int(len(latencies_us)*0.99)]:.1f} µs")
    print(f"  max    : {max(latencies_us):.1f} µs")
    print()
    print("Note: this measures Python ZMQ send latency only.")
    print("Full pipeline latency (emit → browser render) is measured in the")
    print("React dashboard via the ts_ns field embedded in each message.")


if __name__ == "__main__":
    main()
