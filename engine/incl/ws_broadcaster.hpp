#pragma once
#include "types.hpp"
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <functional>

// WsBroadcaster — uWebSockets server broadcasting JSON to all connected clients.
//
// Two broadcast methods:
//   broadcast()          — TrainingEvent scalars  → {"type":"event", ...}
//   broadcast_snapshot() — TensorSnapshot tensors → {"type":"snapshot", ...}
//
// Both are thread-safe and use uWS::Loop::defer() to post work onto the
// uWS event loop thread, so callers never block.
class WsBroadcaster {
public:
    using MessageCallback = std::function<void(std::string_view)>;

    explicit WsBroadcaster(uint16_t port = 8080);
    ~WsBroadcaster();

    WsBroadcaster(const WsBroadcaster&)            = delete;
    WsBroadcaster& operator=(const WsBroadcaster&) = delete;

    // Set before start() — called from the uWS event loop thread.
    void set_message_handler(MessageCallback cb) { message_cb_ = std::move(cb); }

    void start();
    void stop();

    void broadcast(const TrainingEvent& event);
    void broadcast_snapshot(const TensorSnapshot& snap);
    void broadcast_topology(const std::string& json_payload);

    // Send a raw JSON string to all connected clients (used for status updates).
    void broadcast_raw(const std::string& json);

    uint32_t connected_clients() const { return connected_clients_.load(std::memory_order_relaxed); }
    uint64_t frames_sent()       const { return frames_sent_.load(std::memory_order_relaxed); }

private:
    uint16_t        port_;
    MessageCallback message_cb_;
    std::thread     thread_;
    std::atomic<bool>     running_{false};
    std::atomic<uint32_t> connected_clients_{0};
    std::atomic<uint64_t> frames_sent_{0};
    void* loop_ptr_{nullptr};
    void* app_ptr_{nullptr};
};