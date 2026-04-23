#include "../incl/pipeline.hpp"
#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>

Pipeline::Pipeline(WsBroadcaster& broadcaster, const Config& cfg)
    : broadcaster_(broadcaster), cfg_(cfg)
    , sampler_(cfg.sampler), log_(cfg.log_path)
    , anomaly_(cfg.sidecar_path, cfg.anomaly), metrics_(cfg.metrics_port)
{}

Pipeline::~Pipeline() { stop(); }

void Pipeline::start() {
    running_.store(true, std::memory_order_release);
    metrics_.start([this]{ return metrics_json(); });
    consumer_thread_ = std::thread(&Pipeline::consumer_loop, this);
    std::cout << "[pipeline] Started. log=" << cfg_.log_path
              << " sidecar=" << cfg_.sidecar_path
              << " metrics_port=" << cfg_.metrics_port << "\n";
}

void Pipeline::stop() {
    running_.store(false, std::memory_order_release);
    if (consumer_thread_.joinable()) consumer_thread_.join();
    metrics_.stop();
    anomaly_.close();
    log_.close();
}

void Pipeline::push(const TrainingEvent& ev) noexcept {
    ring_.try_push(ev);
}

// Broadcast directly to the WebSocket from the ZmqReceiver thread.
// WsBroadcaster::broadcast_snapshot() is thread-safe.
void Pipeline::push_snapshot(const TensorSnapshot& snap) noexcept {
    broadcaster_.broadcast_snapshot(snap);
    snapshots_forwarded_.fetch_add(1, std::memory_order_relaxed);
}

void Pipeline::push_topology(const std::string& json) noexcept {
    broadcaster_.broadcast_topology(json);
}

void Pipeline::consumer_loop() {
    constexpr auto IDLE_SLEEP = std::chrono::microseconds(100);
    while (running_.load(std::memory_order_acquire)) {
        auto item = ring_.try_pop();
        if (!item) { std::this_thread::sleep_for(IDLE_SLEEP); continue; }
        const TrainingEvent& ev = *item;
        const float fill = ring_.fill_ratio();
        log_.append(ev);
        anomaly_.process(ev);
        if (sampler_.should_forward(ev, fill)) broadcaster_.broadcast(ev);
    }
    while (auto item = ring_.try_pop()) {
        log_.append(*item);
        anomaly_.process(*item);
    }
}

std::string Pipeline::metrics_json() const {
    std::ostringstream o;
    o << std::fixed << std::setprecision(2);
    const float fill_pct = ring_.fill_ratio() * 100.0f;
    const auto  mode_str = (sampler_.current_mode() == AdaptiveSampler::Mode::PASSTHROUGH)
                           ? "passthrough" : "change_point";
    o << "{"
      << "\"ring_buffer\":{"
          << "\"size\":"            << ring_.size()              << ","
          << "\"capacity\":"        << ring_.capacity()          << ","
          << "\"fill_pct\":"        << fill_pct                  << ","
          << "\"writes_total\":"    << ring_.writes_total()      << ","
          << "\"reads_total\":"     << ring_.reads_total()       << ","
          << "\"overflows_total\":" << ring_.overflows_total()
      << "},"
      << "\"sampler\":{"
          << "\"mode\":\""          << mode_str                     << "\","
          << "\"seen\":"            << sampler_.total_seen()        << ","
          << "\"forwarded\":"       << sampler_.total_forwarded()   << ","
          << "\"dropped\":"         << sampler_.total_dropped()     << ","
          << "\"ema_loss\":"        << sampler_.ema_loss()          << ","
          << "\"ema_grad_norm\":"   << sampler_.ema_grad_norm()
      << "},"
      << "\"anomaly\":{"
          << "\"count\":"           << anomaly_.anomaly_count()
      << "},"
      << "\"snapshots\":{"
          << "\"forwarded\":"       << snapshots_forwarded_.load(std::memory_order_relaxed)
      << "},"
      << "\"log\":{"
          << "\"records\":"         << log_.record_count()  << ","
          << "\"path\":\""          << cfg_.log_path        << "\""
      << "},"
      << "\"ws\":{"
          << "\"clients\":"         << broadcaster_.connected_clients() << ","
          << "\"frames_sent\":"     << broadcaster_.frames_sent()
      << "}"
      << "}";
    return o.str();
}
