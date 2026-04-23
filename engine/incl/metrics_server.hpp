#pragma once
#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <cstdint>
#include <chrono>

// MetricsServer
// Serves a single JSON endpoint at GET /metrics on a given port.
// Uses cpp-httplib (single-header, fetched via CMake FetchContent).
//
// The caller supplies a snapshot callback that returns a JSON string.
// This keeps MetricsServer decoupled from the engine internals — it
// doesn't need to know about RingBuffer or MmapLog types directly.
//
// Example response (pretty-printed here, compact in practice):
// {
//   "ring_buffer": {
//     "size": 12,
//     "capacity": 4095,
//     "fill_pct": 0.29,
//     "writes_total": 14823,
//     "reads_total": 14811,
//     "overflows_total": 0
//   },
//   "sampler": {
//     "mode": "passthrough",
//     "seen": 14823,
//     "forwarded": 14823,
//     "dropped": 0
//   },
//   "log": {
//     "records": 14811,
//     "path": "training.omlog"
//   },
//   "ws": {
//     "clients": 1,
//     "frames_sent": 14811
//   }
// }
//
// The React dashboard polls this endpoint every 500ms and renders
// a live ring-buffer depth gauge alongside the training charts.

class MetricsServer {
public:
    using SnapshotFn = std::function<std::string()>;
    using ChaptersFn = std::function<std::string()>;

    explicit MetricsServer(uint16_t port);
    ~MetricsServer();

    MetricsServer(const MetricsServer&) = delete;
    MetricsServer& operator=(const MetricsServer&) = delete;

    void start(SnapshotFn snapshot_fn, ChaptersFn chapters_fn = nullptr);
    
    void stop();

private:
    uint16_t   port_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    void*      server_ptr_{nullptr};  // opaque httplib::Server*
};
