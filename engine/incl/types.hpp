#pragma once
#include <cstdint>
#include <string>
#include <vector>

// TrainingEvent
// Scalar metrics emitted every training step.
// JSON wire format for WebSocket: {"type":"event","step":N,...}
struct TrainingEvent {
    uint32_t step         = 0;
    float    loss         = 0.0f;
    float    accuracy     = 0.0f;
    float    grad_norm    = 0.0f;
    uint64_t timestamp_ns = 0;

    [[nodiscard]] std::string to_json() const {
        return "{\"type\":\"event\""
             ",\"step\":"       + std::to_string(step)
             + ",\"loss\":"       + std::to_string(loss)
             + ",\"accuracy\":"   + std::to_string(accuracy)
             + ",\"grad_norm\":"  + std::to_string(grad_norm)
             + ",\"ts_ns\":"      + std::to_string(timestamp_ns)
             + "}";
    }
};

// TensorSnapshot
// Real weights / activations / gradients from a named model layer.
// JSON wire format: {"type":"snapshot","step":N,"layer_name":"...","tensor_type":"WEIGHT",...}
//
// values[] is a flat array; shape[] describes dimensions.
// e.g. conv1.weight: shape=[32,1,3,3], values.size()==288
//
// sample_rate < 1.0 means only a random subset of values was sent —
// the visualiser scales the heatmap accordingly.
struct TensorSnapshot {
    uint32_t             step         = 0;
    std::string          layer_name;
    std::string          tensor_type;   // "WEIGHT" | "GRADIENT" | "ACTIVATION" | "NODE_EMBEDDING"
    std::vector<uint32_t> shape;
    std::vector<float>   values;
    float                sample_rate  = 1.0f;
    uint64_t             timestamp_ns = 0;

    [[nodiscard]] std::string to_json() const {
        std::string s = "{\"type\":\"snapshot\""
                       ",\"step\":"         + std::to_string(step)
                     + ",\"layer_name\":\"" + layer_name + "\""
                     + ",\"tensor_type\":\"" + tensor_type + "\""
                     + ",\"sample_rate\":" + std::to_string(sample_rate)
                     + ",\"ts_ns\":"       + std::to_string(timestamp_ns)
                     + ",\"shape\":[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) s += ',';
            s += std::to_string(shape[i]);
        }
        s += "],\"values\":[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) s += ',';
            // 4 decimal places
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%.4f", values[i]);
            s += buf;
        }
        s += "]}";
        return s;
    }
};
