#pragma once
#include "types.hpp"
#include <cmath>
#include <cstdint>
#include <atomic>

// AdaptiveSampler
// ---------------
// Decides whether a TrainingEvent should be forwarded to the WebSocket
// broadcaster and written to the mmap log, based on ring buffer pressure.
//
// Two sampling modes, selected automatically:
//
//   PASSTHROUGH  (fill_ratio < low_threshold)
//     Every event passes.  Default during normal operation.
//
//   CHANGE_POINT (fill_ratio >= high_threshold)
//     An event only passes if at least one metric has changed significantly
//     from its recent rolling mean.  This preserves the "interesting moments"
//     (loss spikes, gradient explosions, accuracy jumps) while discarding
//     the steady-state run that floods the buffer.
//
// Hysteresis between low_ and high_threshold prevents rapid mode flipping
// when the buffer hovers near the boundary.
//
// All state is POD — no heap allocation, safe to embed in the engine
// without dynamic memory.

class AdaptiveSampler {
public:
    struct Config {
        float low_threshold  = 0.50f;  // switch to change-point above this fill
        float high_threshold = 0.80f;  // switch back to passthrough below this
        float loss_delta     = 0.02f;  // min |delta_loss| to count as a change
        float grad_delta     = 0.50f;  // min |delta_grad_norm| to count as a change
        float acc_delta      = 0.01f;  // min |delta_accuracy| to count as a change
        uint32_t ema_window  = 32;     // EMA smoothing factor (alpha = 2/(window+1))
    };

    explicit AdaptiveSampler(const Config& cfg = {}) : cfg_(cfg) {}

    // Returns true if the event should be forwarded.
    // fill_ratio: current ring buffer depth as a fraction of capacity (0–1).
    bool should_forward(const TrainingEvent& ev, float fill_ratio) noexcept {
        update_ema(ev);
        update_mode(fill_ratio);
        total_seen_.fetch_add(1, std::memory_order_relaxed);

        if (mode_ == Mode::PASSTHROUGH) {
            total_forwarded_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }

        // CHANGE_POINT: forward only if a metric moved significantly.
        bool changed =
            std::fabs(ev.loss      - ema_loss_)      > cfg_.loss_delta ||
            std::fabs(ev.grad_norm - ema_grad_norm_) > cfg_.grad_delta ||
            std::fabs(ev.accuracy  - ema_accuracy_)  > cfg_.acc_delta;

        if (changed) {
            total_forwarded_.fetch_add(1, std::memory_order_relaxed);
        } else {
            total_dropped_.fetch_add(1, std::memory_order_relaxed);
        }
        return changed;
    }

    // Diagnostics

    enum class Mode : uint8_t { PASSTHROUGH, CHANGE_POINT };

    Mode     current_mode()    const noexcept { return mode_; }
    uint64_t total_seen()      const noexcept { return total_seen_.load(std::memory_order_relaxed); }
    uint64_t total_forwarded() const noexcept { return total_forwarded_.load(std::memory_order_relaxed); }
    uint64_t total_dropped()   const noexcept { return total_dropped_.load(std::memory_order_relaxed); }

    float ema_loss()      const noexcept { return ema_loss_; }
    float ema_grad_norm() const noexcept { return ema_grad_norm_; }
    float ema_accuracy()  const noexcept { return ema_accuracy_; }

private:
    void update_ema(const TrainingEvent& ev) noexcept {
        // Exponential moving average: alpha = 2 / (window + 1)
        const float alpha = 2.0f / (static_cast<float>(cfg_.ema_window) + 1.0f);
        if (first_event_) {
            ema_loss_      = ev.loss;
            ema_grad_norm_ = ev.grad_norm;
            ema_accuracy_  = ev.accuracy;
            first_event_   = false;
        } else {
            ema_loss_      += alpha * (ev.loss      - ema_loss_);
            ema_grad_norm_ += alpha * (ev.grad_norm - ema_grad_norm_);
            ema_accuracy_  += alpha * (ev.accuracy  - ema_accuracy_);
        }
    }

    void update_mode(float fill_ratio) noexcept {
        if (mode_ == Mode::PASSTHROUGH && fill_ratio >= cfg_.high_threshold) {
            mode_ = Mode::CHANGE_POINT;
        } else if (mode_ == Mode::CHANGE_POINT && fill_ratio < cfg_.low_threshold) {
            mode_ = Mode::PASSTHROUGH;
        }
    }

    Config cfg_;
    Mode   mode_       = Mode::PASSTHROUGH;
    bool   first_event_= true;

    float  ema_loss_      = 0.0f;
    float  ema_grad_norm_ = 0.0f;
    float  ema_accuracy_  = 0.0f;

    std::atomic<uint64_t> total_seen_{0};
    std::atomic<uint64_t> total_forwarded_{0};
    std::atomic<uint64_t> total_dropped_{0};
};
