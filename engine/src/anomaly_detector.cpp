#include "../incl/anomaly_detector.hpp"

#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include <cstring>
#include <cmath>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>
#ifdef _WIN32
#include <basetsd.h>
typedef SSIZE_T ssize_t;
#endif
#include "flatbuffers/flatbuffers.h"

static uint64_t mono_ns() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count()
    );
}

// Sidecar file format: length-prefixed FlatBuffers records.
// [uint32_t size][bytes × size][uint32_t size][bytes × size]...
// The replay engine reads this with the same framing to rebuild the chapter index.
static bool write_framed(int fd, const uint8_t* data, uint32_t size) noexcept {
    if (::write(fd, &size, 4) != 4) return false;
    return ::write(fd, data, size) == static_cast<ssize_t>(size);
}

AnomalyDetector::AnomalyDetector(const std::string& path, const Config& cfg)
    : cfg_(cfg), sidecar_path_(path)
{
    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd_ < 0) {
        std::cerr << "[anomaly] Could not open sidecar: " << path
                  << " (" << std::strerror(errno) << ")\n";
    } else {
        std::cout << "[anomaly] Sidecar: " << path << "\n";
    }
}

AnomalyDetector::~AnomalyDetector() {
    close();
}

void AnomalyDetector::close() noexcept {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
        std::cout << "[anomaly] Closed. anomalies="
                  << anomaly_count_.load(std::memory_order_relaxed) << "\n";
    }
}

void AnomalyDetector::process(const TrainingEvent& ev) noexcept {
    if (first_) {
        ema_loss_ = ev.loss;
        ema_grad_ = ev.grad_norm;
        best_loss_ = ev.loss;
        first_ = false;
        return;
    }

    update_ema(ev);

    const float alpha = 2.0f / (static_cast<float>(cfg_.ema_window) + 1.0f);

    //Gradient explosion
    const float grad_threshold = cfg_.explosion_factor * ema_grad_;
    if (ema_grad_ > 0.01f && ev.grad_norm > grad_threshold) {
        const float severity = std::min(1.0f,
            (ev.grad_norm - grad_threshold) / grad_threshold);

        std::ostringstream desc;
        desc << std::fixed << std::setprecision(3)
             << "grad_norm " << ev.grad_norm
             << " > " << cfg_.explosion_factor << "x ema (" << ema_grad_ << ")";

        write_marker(ev.step, omnistream::AnomalyType_GRAD_EXPLOSION,
                     severity, ev.grad_norm, grad_threshold,
                     desc.str(), ev.timestamp_ns);
    }

    //Loss spike
    const float loss_threshold = cfg_.spike_factor * ema_loss_;
    if (ema_loss_ > 0.01f && ev.loss > loss_threshold) {
        const float severity = std::min(1.0f,
            (ev.loss - loss_threshold) / loss_threshold);

        std::ostringstream desc;
        desc << std::fixed << std::setprecision(3)
             << "loss " << ev.loss
             << " > " << cfg_.spike_factor << "x ema (" << ema_loss_ << ")";

        write_marker(ev.step, omnistream::AnomalyType_LOSS_SPIKE,
                     severity, ev.loss, loss_threshold,
                     desc.str(), ev.timestamp_ns);
    }

    //Loss plateau
    if (ev.loss < best_loss_ - 1e-4f) {
        best_loss_ = ev.loss;
        steps_since_improvement_ = 0;
    } else {
        steps_since_improvement_++;
    }

    if (steps_since_improvement_ == cfg_.plateau_steps) {
        std::ostringstream desc;
        desc << "no loss improvement for " << cfg_.plateau_steps
             << " steps (best=" << std::fixed << std::setprecision(4)
             << best_loss_ << ")";

        write_marker(ev.step, omnistream::AnomalyType_LOSS_PLATEAU,
                     0.5f, ev.loss, best_loss_,
                     desc.str(), ev.timestamp_ns);
    }
}

void AnomalyDetector::update_ema(const TrainingEvent& ev) noexcept {
    const float alpha = 2.0f / (static_cast<float>(cfg_.ema_window) + 1.0f);
    ema_loss_ += alpha * (ev.loss      - ema_loss_);
    ema_grad_ += alpha * (ev.grad_norm - ema_grad_);
}

void AnomalyDetector::write_marker(
    uint32_t step, omnistream::AnomalyType type,
    float severity, float metric_value, float threshold,
    const std::string& desc, uint64_t ts_ns) noexcept
{
    flatbuffers::FlatBufferBuilder builder(128);
    auto desc_offset = builder.CreateString(desc);
    auto offset = omnistream::CreateAnomalyMarker(
        builder, step, type, severity, metric_value, threshold,
        desc_offset, ts_ns);
    builder.Finish(offset);

    if (fd_ >= 0) {
        write_framed(fd_,
            builder.GetBufferPointer(),
            builder.GetSize());
    }

    chapter_index_.push_back({
        step, type, severity, metric_value, desc
    });

    anomaly_count_.fetch_add(1, std::memory_order_relaxed);

    std::cout << "[anomaly] " << omnistream::EnumNameAnomalyType(type)
              << " step=" << step
              << " severity=" << std::fixed << std::setprecision(2) << severity
              << " val=" << metric_value << "\n";
}
