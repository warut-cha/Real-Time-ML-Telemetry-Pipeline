#pragma once
#include <atomic>
#include <array>
#include <optional>
#include <cstdint>
#include <cstddef>

// RingBuffer<T, N>
// ---------------
// Lock-free single-producer / single-consumer (SPSC) ring buffer.
//
// Correctness model
// -----------------
// The producer owns write_pos_ and the consumer owns read_pos_.
// Each side reads the other's index with acquire semantics and writes its
// own with release semantics.  This establishes the happens-before edge:
//
//   producer: slots_[pos] = value           (plain store to slot)
//             write_pos_.store(release)      (publishes the write)
//   consumer: write_pos_.load(acquire)       (sees the publication)
//             value = slots_[pos]            (safe to read now)
//
// No mutexes, no CAS loops.  One producer thread and one consumer thread
// only — SPSC is the contract.  Violating it (two producers, two consumers)
// is undefined behaviour.
//
// Capacity
// --------
// N must be a power of two.  The implementation uses bitwise masking
// (pos & (N-1)) instead of modulo for O(1) index wrapping.
// Effective capacity is N-1 slots (one slot is always kept empty to
// distinguish full from empty without a separate counter).
//
// Usage
// -------------
//   ZmqReceiver (producer thread) → RingBuffer<TrainingEvent, 4096>
//   → consumer thread: MmapLog writer + WsBroadcaster
//
// Diagnostic counters (reads_total, writes_total, overflows_total) are
// all std::atomic so the metrics HTTP endpoint can read them lock-free
// from the main thread at any time.

template<typename T, std::size_t N>
class RingBuffer {
    static_assert(N >= 2 && (N & (N - 1)) == 0,
        "RingBuffer capacity N must be a power of two >= 2");

public:
    RingBuffer() : write_pos_(0), read_pos_(0),
                   writes_total_(0), reads_total_(0), overflows_total_(0) {}

    //Producer API

    // Try to push one item.  Returns true on success, false if full (overflow).
    // Called from the producer thread only.
    bool try_push(const T& item) noexcept {
        const std::size_t w = write_pos_.load(std::memory_order_relaxed);
        const std::size_t next_w = (w + 1) & MASK;

        if (next_w == read_pos_.load(std::memory_order_acquire)) {
            // Buffer full — drop and count.
            overflows_total_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        slots_[w] = item;
        write_pos_.store(next_w, std::memory_order_release);
        writes_total_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool try_push(T&& item) noexcept {
        const std::size_t w = write_pos_.load(std::memory_order_relaxed);
        const std::size_t next_w = (w + 1) & MASK;

        if (next_w == read_pos_.load(std::memory_order_acquire)) {
            overflows_total_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        slots_[w] = std::move(item);
        write_pos_.store(next_w, std::memory_order_release);
        writes_total_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Consumer API

    // Try to pop one item.  Returns the item if available, std::nullopt if empty.
    // Called from the consumer thread only.
    std::optional<T> try_pop() noexcept {
        const std::size_t r = read_pos_.load(std::memory_order_relaxed);

        if (r == write_pos_.load(std::memory_order_acquire)) {
            return std::nullopt;   // empty
        }

        T item = std::move(slots_[r]);
        read_pos_.store((r + 1) & MASK, std::memory_order_release);
        reads_total_.fetch_add(1, std::memory_order_relaxed);
        return item;
    }

    // Diagnostic counters (readable from any thread)

    std::size_t size() const noexcept {
        const std::size_t w = write_pos_.load(std::memory_order_acquire);
        const std::size_t r = read_pos_.load(std::memory_order_acquire);
        return (w - r) & MASK;
    }

    bool empty() const noexcept {
        return read_pos_.load(std::memory_order_acquire)
            == write_pos_.load(std::memory_order_acquire);
    }

    bool full() const noexcept {
        const std::size_t w = write_pos_.load(std::memory_order_acquire);
        return ((w + 1) & MASK) == read_pos_.load(std::memory_order_acquire);
    }

    // Depth as a fraction of capacity (0.0 – 1.0).
    float fill_ratio() const noexcept {
        return static_cast<float>(size()) / static_cast<float>(capacity());
    }

    static constexpr std::size_t capacity() noexcept { return N - 1; }

    uint64_t writes_total()    const noexcept { return writes_total_.load(std::memory_order_relaxed); }
    uint64_t reads_total()     const noexcept { return reads_total_.load(std::memory_order_relaxed); }
    uint64_t overflows_total() const noexcept { return overflows_total_.load(std::memory_order_relaxed); }

private:
    static constexpr std::size_t MASK = N - 1;

    // Cache-line padding prevents false sharing between producer and consumer
    // cache lines.  The producer reads/writes write_pos_ and slots_[write_pos_];
    // the consumer reads/writes read_pos_ and slots_[read_pos_].
    // Without padding, both cache lines touch the same 64-byte block and
    // bouncing between cores costs ~100ns per operation on modern CPUs.
    alignas(64) std::atomic<std::size_t> write_pos_;
    alignas(64) std::atomic<std::size_t> read_pos_;

    alignas(64) std::array<T, N> slots_;

    // Counters: separate cache line so they don't interfere with hot path.
    alignas(64) std::atomic<uint64_t> writes_total_;
    alignas(64) std::atomic<uint64_t> reads_total_;
    alignas(64) std::atomic<uint64_t> overflows_total_;
};
