#pragma once
#include "mmap_log.hpp"
#include "ws_broadcaster.hpp"
#include "anomaly_detector.hpp"

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <functional>
#include <cstdint>
#include <map>
// ReplayEngine
// ---------------
// Opens a baseline.omlog file and a matching .anomalies
// sidecar, then replays the log to the WsBroadcaster at a requested speed.
//
// Media-player API
//   open(path)         memory-map the log + load chapter index
//   play(speed)        start broadcasting; speed = 1.0 / 5.0 / 10.0
//   pause()            freeze at current step
//   seek(step)         jump to a specific step (O(1) via fixed-record layout)
//   stop()             close and clean up
//
// WebSocket control protocol (received on /replay path)
//   {"cmd":"play",  "speed": 5.0}
//   {"cmd":"pause"}
//   {"cmd":"seek",  "step": 420}
//   {"cmd":"stop"}
//
// Replay accuracy
// ---------------
// Events are broadcast preserving the original inter-event timing scaled
// by the speed factor.  std::this_thread::sleep_until() is used so the
// replay stays aligned to wall clock even if individual broadcasts are
// slightly late.
//
// Chapter index
// -------------
// Loaded from the .anomalies sidecar on open().  Exposed via chapters()
// so main.cpp can serve it from the /chapters HTTP endpoint.

class ReplayEngine {
public:
    enum class State { IDLE, PLAYING, PAUSED };

    struct ReplayStatus {
        State    state        = State::IDLE;
        uint64_t current_step = 0;
        uint64_t total_steps  = 0;
        float    speed        = 1.0f;
        uint64_t frames_sent  = 0;
    };

    explicit ReplayEngine(WsBroadcaster& broadcaster);
    ~ReplayEngine();

    ReplayEngine(const ReplayEngine&)            = delete;
    ReplayEngine& operator=(const ReplayEngine&) = delete;

    // Open a log file and load the chapter index.
    // Returns false if the file is missing or has a bad header.
    bool open(const std::string& log_path) noexcept;

    void play(float speed = 1.0f) noexcept;
    void pause()                  noexcept;
    void seek(uint64_t step)      noexcept;
    void stop()                   noexcept;

    // Handle a JSON control command from the WebSocket client.
    void handle_command(std::string_view json) noexcept;

    ReplayStatus status() const noexcept;

    const std::vector<AnomalyDetector::ChapterEntry>& chapters() const noexcept {
        return chapters_;
    }

    std::string chapters_json() const noexcept;

private:
    void replay_loop();

    WsBroadcaster& broadcaster_;

    // mmap'd log (read-only, opened separately from the writer).
    int             fd_      = -1;
    const uint8_t*  map_     = nullptr;
    std::size_t     map_size_= 0;
    uint64_t        total_records_ = 0;

    std::vector<AnomalyDetector::ChapterEntry> chapters_;
    std::map<uint64_t, std::string> snapshots_;
    std::thread           thread_;
    std::atomic<State>    state_{State::IDLE};
    std::atomic<uint64_t> seek_target_{UINT64_MAX};  // UINT64_MAX = no pending seek
    std::atomic<float>    speed_{1.0f};
    std::atomic<uint64_t> current_step_{0};
    std::atomic<uint64_t> frames_sent_{0};
};
