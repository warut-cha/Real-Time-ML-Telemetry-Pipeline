#pragma once
#include "types.hpp"
#include <functional>
#include <string>
#include <atomic>
#include <thread>

// ZmqReceiver listens on a ZMQ PULL socket.
//
// Wire format detection (Phase 4)
// ────────────────────────────────
// Every ZMQ frame starts with a 1-byte type discriminator:
//   0x01 → TrainingEvent  FlatBuffer
//   0x02 → TensorSnapshot FlatBuffer
//
// Frames without a valid discriminator are rejected (parse_error++).
// This is backward-compatible: Phase 2 emitters that don't send the
// discriminator byte will fail the size check and be counted as errors.
// The Python hook.py sends the discriminator byte before the FlatBuffer.
//
// Threading: all callbacks are invoked from the receive thread.
// Callers must ensure callbacks are thread-safe.

class ZmqReceiver {
public:
    using EventCallback    = std::function<void(const TrainingEvent&)>;
    using SnapshotCallback = std::function<void(const TensorSnapshot&)>;

    static constexpr uint8_t TAG_EVENT    = 0x01;
    static constexpr uint8_t TAG_SNAPSHOT = 0x02;
    static constexpr uint8_t TAG_TOPOLOGY = 0x03;

    // Called with the raw JSON payload of a topology message.
    // The engine broadcasts it verbatim — it's already valid JSON.
    using TopologyCallback = std::function<void(const std::string&)>;

    explicit ZmqReceiver(const std::string& endpoint = "tcp://127.0.0.1:5555");
    ~ZmqReceiver();

    ZmqReceiver(const ZmqReceiver&)            = delete;
    ZmqReceiver& operator=(const ZmqReceiver&) = delete;

    // event_cb     fires for every TrainingEvent received.
    // snapshot_cb  fires for every TensorSnapshot received (may be null).
    // topology_cb  fires for every GraphTopology JSON message (may be null).
    void start(EventCallback event_cb,
               SnapshotCallback snapshot_cb  = nullptr,
               TopologyCallback topology_cb  = nullptr);
    void stop();

    uint64_t messages_received() const { return messages_received_.load(std::memory_order_relaxed); }
    uint64_t parse_errors()      const { return parse_errors_.load(std::memory_order_relaxed); }
    uint64_t snapshots_received()const { return snapshots_received_.load(std::memory_order_relaxed); }

private:
    void run_loop();

    static TrainingEvent  deserialise_event   (const uint8_t* data, size_t size);
    static TensorSnapshot deserialise_snapshot(const uint8_t* data, size_t size);

    std::string      endpoint_;
    EventCallback    event_cb_;
    SnapshotCallback snapshot_cb_;
    TopologyCallback topology_cb_;
    std::thread      thread_;
    std::atomic<bool> running_{false};

    std::atomic<uint64_t> messages_received_{0};
    std::atomic<uint64_t> parse_errors_{0};
    std::atomic<uint64_t> snapshots_received_{0};
};
