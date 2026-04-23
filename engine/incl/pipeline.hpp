#pragma once
#include "types.hpp"
#include "ring_buffer.hpp"
#include "adaptive_sampler.hpp"
#include "mmap_log.hpp"
#include "anomaly_detector.hpp"
#include "ws_broadcaster.hpp"
#include "metrics_server.hpp"

#include <thread>
#include <atomic>
#include <string>

// Pipeline owns the ring buffer and consumer thread.
// also accepts TensorSnapshot via push_snapshot() and
// forwards them directly to the WsBroadcaster (no ring buffer needed —
// snapshots are infrequent and large, not high-frequency scalars).

class Pipeline {
public:
    static constexpr std::size_t RING_CAPACITY = 4096;

    struct Config {
        std::string  log_path      = "training.omlog";
        std::string  sidecar_path  = "training.omlog.anomalies";
        uint16_t     metrics_port  = 9090;
        AdaptiveSampler::Config  sampler{};
        AnomalyDetector::Config  anomaly{};
    };

    explicit Pipeline(WsBroadcaster& broadcaster, const Config& cfg = {});
    ~Pipeline();

    Pipeline(const Pipeline&)            = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    void start();
    void stop();

    // Called from ZmqReceiver thread — non-blocking.
    void push(const TrainingEvent& ev)               noexcept;
    void push_snapshot(const TensorSnapshot& snap)   noexcept;
    void push_topology(const std::string& json)      noexcept;

    std::string metrics_json() const;
    const AnomalyDetector& anomaly_detector() const noexcept { return anomaly_; }

private:
    void consumer_loop();

    WsBroadcaster&    broadcaster_;
    Config            cfg_;

    RingBuffer<TrainingEvent, RING_CAPACITY> ring_;
    AdaptiveSampler   sampler_;
    MmapLog           log_;
    AnomalyDetector   anomaly_;
    MetricsServer     metrics_;

    std::thread       consumer_thread_;
    std::atomic<bool> running_{false};

    // Snapshot counters (diagnostics only)
    std::atomic<uint64_t> snapshots_forwarded_{0};
};
