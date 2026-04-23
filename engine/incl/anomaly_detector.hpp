#pragma once
#include "types.hpp"
#include "training_event_generated.h"

#include <string>
#include <vector>
#include <cstdint>
#include <atomic>
#include <functional>

// AnomalyDetector
// ---------------
// Called from the Pipeline consumer thread on every TrainingEvent.
// Maintains rolling statistics and detects three anomaly classes:
//
//   GRAD_EXPLOSION  grad_norm > explosion_factor x ema_grad_norm
//   LOSS_SPIKE      loss      > spike_factor      x ema_loss
//   LOSS_PLATEAU    no improvement in loss over plateau_steps steps
//
// When an anomaly is detected:
//   1. An AnomalyMarker FlatBuffer is written to a sidecar file
//      (<base>.anomalies) alongside the main binary log.
//   2. The in-memory chapter_index_ is updated.
//   3. An optional callback fires (for the metrics JSON snapshot).
//
// Chapter index
// -------------
// On replay-open the ReplayEngine scans the sidecar file and builds
// the same vector<ChapterEntry> in one linear pass.  The React scrubber
// reads this via GET /chapters and renders tick marks on the timeline.

class AnomalyDetector {
public:
    struct Config {
        float    explosion_factor = 10.0f;  // grad spike threshold multiplier
        float    spike_factor     = 3.0f;   // loss spike threshold multiplier
        uint32_t plateau_steps    = 50;     // steps without improvement -> plateau
        uint32_t ema_window       = 32;
    };

    struct ChapterEntry {
        uint32_t                     step;
        omnistream::AnomalyType      type;
        float                        severity;   // 0–1, for scrubber colour
        float                        metric_value;
        std::string                  description;
    };

    explicit AnomalyDetector(const std::string& sidecar_path, const Config& cfg = {});
    ~AnomalyDetector();

    AnomalyDetector(const AnomalyDetector&)            = delete;
    AnomalyDetector& operator=(const AnomalyDetector&) = delete;

    // Feed one event.  May write to the sidecar file.
    void process(const TrainingEvent& ev) noexcept;

    // Chapter index — safe to read from any thread after start.
    const std::vector<ChapterEntry>& chapters() const noexcept { return chapter_index_; }
    uint64_t anomaly_count() const noexcept {
        return anomaly_count_.load(std::memory_order_relaxed);
    }

    // Close the sidecar file.  Called from Pipeline::stop().
    void close() noexcept;

private:
    void update_ema(const TrainingEvent& ev) noexcept;
    void write_marker(uint32_t step, omnistream::AnomalyType type,
                      float severity, float metric_value,
                      float threshold, const std::string& desc,
                      uint64_t ts_ns) noexcept;

    Config      cfg_;
    std::string sidecar_path_;
    int         fd_    = -1;

    // Rolling statistics.
    bool    first_     = true;
    float   ema_loss_  = 0.0f;
    float   ema_grad_  = 0.0f;
    float   best_loss_ = 1e9f;
    uint32_t steps_since_improvement_ = 0;

    std::vector<ChapterEntry>      chapter_index_;
    std::atomic<uint64_t>          anomaly_count_{0};
};
