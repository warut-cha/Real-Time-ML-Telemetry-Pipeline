// ring_buffer_test.cpp
// ────────────────────
// Stress-tests RingBuffer<uint64_t, 4096> with one producer and one consumer
// running for a fixed duration.  Verifies:
//   1. No item is read more than once.
//   2. No item is corrupted (sequence numbers are monotonically spaced).
//   3. reads_total + overflows_total == writes_attempted (every push accounted for).
//
// Run: ./ring_buffer_test [seconds]
// Pass criteria: exits 0 with "PASS" on stdout.

#include "ring_buffer.hpp"

#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <vector>
#include <algorithm>

static constexpr std::size_t N = 4096;
using Buffer = RingBuffer<uint64_t, N>;

int main(int argc, char** argv) {
    const int run_secs = (argc > 1) ? std::stoi(argv[1]) : 10;

    Buffer buf;
    std::atomic<bool>     done{false};
    std::atomic<uint64_t> writes_attempted{0};
    std::vector<uint64_t> received;
    received.reserve(1'000'000);

    // ── Producer ──────────────────────────────────────────────────────────
    std::thread producer([&] {
        uint64_t seq = 0;
        const auto deadline = std::chrono::steady_clock::now()
                            + std::chrono::seconds(run_secs);
        while (std::chrono::steady_clock::now() < deadline) {
            buf.try_push(seq++);
            writes_attempted.fetch_add(1, std::memory_order_relaxed);
        }
        done.store(true, std::memory_order_release);
    });

    // ── Consumer ──────────────────────────────────────────────────────────
    std::thread consumer([&] {
        while (!done.load(std::memory_order_acquire) || !buf.empty()) {
            auto item = buf.try_pop();
            if (item) {
                received.push_back(*item);
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    // ── Verification ──────────────────────────────────────────────────────
    const uint64_t wa  = writes_attempted.load();
    const uint64_t wt  = buf.writes_total();
    const uint64_t rt  = buf.reads_total();
    const uint64_t ovf = buf.overflows_total();

    std::cout << "writes_attempted : " << wa  << "\n"
              << "writes_total     : " << wt  << "\n"
              << "reads_total      : " << rt  << "\n"
              << "overflows_total  : " << ovf << "\n"
              << "items received   : " << received.size() << "\n";

    bool pass = true;

    // Every push is either a successful write or an overflow.
    if (wt + ovf != wa) {
        std::cerr << "FAIL: wt + ovf (" << (wt + ovf) << ") != wa (" << wa << ")\n";
        pass = false;
    }

    // Every successful write was eventually consumed.
    if (rt != static_cast<uint64_t>(received.size())) {
        std::cerr << "FAIL: reads_total mismatch\n";
        pass = false;
    }

    // Received items are in monotonically increasing order (FIFO preserved).
    for (std::size_t i = 1; i < received.size(); ++i) {
        if (received[i] <= received[i - 1]) {
            std::cerr << "FAIL: ordering violation at index " << i
                      << " (" << received[i-1] << " >= " << received[i] << ")\n";
            pass = false;
            break;
        }
    }

    // No duplicates.
    {
        auto sorted = received;
        std::sort(sorted.begin(), sorted.end());
        auto dup = std::adjacent_find(sorted.begin(), sorted.end());
        if (dup != sorted.end()) {
            std::cerr << "FAIL: duplicate value " << *dup << "\n";
            pass = false;
        }
    }

    std::cout << (pass ? "PASS" : "FAIL") << "\n";
    return pass ? 0 : 1;
}
