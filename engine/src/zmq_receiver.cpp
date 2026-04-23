#include "../incl/zmq_receiver.hpp"

#include "flatbuffers/flatbuffers.h"
#include "../generated/training_event_generated.h"

#include <zmq.hpp>
#include <chrono>
#include <cstring>
#include <iostream>
#include <cstdint>

ZmqReceiver::ZmqReceiver(const std::string& endpoint) : endpoint_(endpoint) {}
ZmqReceiver::~ZmqReceiver() { stop(); }

void ZmqReceiver::start(EventCallback event_cb, SnapshotCallback snapshot_cb, TopologyCallback topology_cb) {
    event_cb_    = std::move(event_cb);
    snapshot_cb_ = std::move(snapshot_cb);
    topology_cb_ = std::move(topology_cb);
    running_.store(true, std::memory_order_release);
    thread_ = std::thread(&ZmqReceiver::run_loop, this);
}

void ZmqReceiver::stop() {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
}

void ZmqReceiver::run_loop() {
    zmq::context_t ctx{1};
    zmq::socket_t  sock{ctx, zmq::socket_type::pull};
    sock.set(zmq::sockopt::rcvtimeo, 100);
    sock.bind(endpoint_);
    std::cout << "[zmq] Listening on " << endpoint_ << " (multi-type FlatBuffers)\n";

    while (running_.load(std::memory_order_acquire)) {
        zmq::message_t msg;
        if (!sock.recv(msg, zmq::recv_flags::none)) continue;

        // Minimum: 1-byte discriminator + 8-byte FlatBuffer header
        if (msg.size() < 9) {
            parse_errors_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        const auto* raw  = static_cast<const uint8_t*>(msg.data());
        const uint8_t tag = raw[0];
        const uint8_t* buf = raw + 1;
        const size_t   sz  = msg.size() - 1;

        try {
            if (tag == TAG_EVENT) {
                auto ev = deserialise_event(buf, sz);
                messages_received_.fetch_add(1, std::memory_order_relaxed);
                if (event_cb_) event_cb_(ev);

            } else if (tag == TAG_SNAPSHOT) {
                auto snap = deserialise_snapshot(buf, sz);
                snapshots_received_.fetch_add(1, std::memory_order_relaxed);
                if (snapshot_cb_) snapshot_cb_(snap);

            } else if (tag == TAG_TOPOLOGY) {
                // Topology is plain JSON after the tag byte 
                std::string json(reinterpret_cast<const char*>(buf), sz);
                if (topology_cb_) topology_cb_(json);

            } else {
                flatbuffers::Verifier v(raw, msg.size());
                if (omnistream::VerifyTrainingEventBuffer(v)) {
                    auto ev = deserialise_event(raw, msg.size());
                    messages_received_.fetch_add(1, std::memory_order_relaxed);
                    if (event_cb_) event_cb_(ev);
                } else {
                    parse_errors_.fetch_add(1, std::memory_order_relaxed);
                    std::cerr << "[zmq] Unknown tag 0x" << std::hex
                              << static_cast<int>(tag) << std::dec << "\n";
                }
            }
        } catch (const std::exception& e) {
            parse_errors_.fetch_add(1, std::memory_order_relaxed);
            std::cerr << "[zmq] Deserialise error: " << e.what() << "\n";
        }
    }

    std::cout << "[zmq] Stopped. events=" << messages_received_
              << " snapshots=" << snapshots_received_
              << " errors=" << parse_errors_ << "\n";
}

TrainingEvent ZmqReceiver::deserialise_event(const uint8_t* data, size_t size) {
    flatbuffers::Verifier v(data, size);
    if (!omnistream::VerifyTrainingEventBuffer(v))
        throw std::runtime_error("TrainingEvent FlatBuffer verification failed");

    const auto* wire = omnistream::GetTrainingEvent(data);
    TrainingEvent ev;
    ev.step         = wire->step();
    ev.loss         = wire->loss();
    ev.accuracy     = wire->accuracy();
    ev.grad_norm    = wire->grad_norm();
    ev.timestamp_ns = wire->timestamp_ns();
    if (ev.timestamp_ns == 0) {
        using namespace std::chrono;
        ev.timestamp_ns = static_cast<uint64_t>(
            duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count());
    }
    return ev;
}

TensorSnapshot ZmqReceiver::deserialise_snapshot(const uint8_t* data, size_t size) {
    // TensorSnapshot FlatBuffer; verify before reading any vectors.
    const auto* wire = flatbuffers::GetRoot<omnistream::TensorSnapshot>(data);

    TensorSnapshot snap;
    snap.step        = wire->step();
    snap.layer_name  = wire->layer_name() ? wire->layer_name()->str() : "";
    snap.tensor_type = omnistream::EnumNameTensorType(wire->tensor_type());
    snap.sample_rate = wire->sample_rate();
    snap.timestamp_ns= wire->timestamp_ns();

    if (wire->shape()) {
        snap.shape.reserve(wire->shape()->size());
        for (auto v : *wire->shape()) snap.shape.push_back(v);
    }
    if (wire->values()) {
        snap.values.reserve(wire->values()->size());
        for (auto v : *wire->values()) snap.values.push_back(v);
    }
    return snap;
}
