// anomaly_detector_test.cpp
// Tests: explosion, spike, plateau detection, severity scaling, chapter index.

#include "../incl/anomaly_detector.hpp"
#include "../incl/types.hpp"
#include <iostream>
#include <cstring>

static int failures = 0;
#define CHECK(cond, msg) \
    do { if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } } while(0)

static TrainingEvent ev(uint32_t step, float loss, float acc, float grad) {
    TrainingEvent e;
    e.step = step; e.loss = loss; e.accuracy = acc;
    e.grad_norm = grad; e.timestamp_ns = 0;
    return e;
}

int main() {
    // ── 1. No anomaly during normal warm-up ───────────────────────────────
    {
        AnomalyDetector d("/dev/null");
        for (int i = 0; i < 10; ++i)
            d.process(ev(i, 1.0f, 0.9f, 1.0f));
        CHECK(d.anomaly_count() == 0, "no anomaly during stable warmup");
        CHECK(d.chapters().empty(),   "chapters empty during stable warmup");
    }

    // ── 2. Gradient explosion detected ────────────────────────────────────
    {
        AnomalyDetector d("/dev/null");
        for (int i = 0; i < 20; ++i) d.process(ev(i, 1.0f, 0.9f, 1.0f));
        d.process(ev(20, 1.0f, 0.9f, 50.0f));  // 50x ema ≈ 1 → explosion
        CHECK(d.anomaly_count() >= 1, "gradient explosion detected");
        bool found = false;
        for (const auto& c : d.chapters())
            if (c.type == omnistream::AnomalyType_GRAD_EXPLOSION) found = true;
        CHECK(found, "GRAD_EXPLOSION in chapter index");
    }

    // ── 3. Loss spike detected ────────────────────────────────────────────
    {
        AnomalyDetector d("/dev/null");
        for (int i = 0; i < 20; ++i) d.process(ev(i, 0.5f, 0.9f, 1.0f));
        d.process(ev(20, 5.0f, 0.9f, 1.0f));   // 10x ema ≈ 0.5 → spike
        bool found = false;
        for (const auto& c : d.chapters())
            if (c.type == omnistream::AnomalyType_LOSS_SPIKE) found = true;
        CHECK(found, "LOSS_SPIKE in chapter index");
    }

    // ── 4. Plateau detected after N steps without improvement ─────────────
    {
        AnomalyDetector::Config cfg;
        cfg.plateau_steps = 10;
        AnomalyDetector d("/dev/null", cfg);
        d.process(ev(0, 1.0f, 0.9f, 1.0f));  // seed best_loss
        for (int i = 1; i <= 10; ++i)
            d.process(ev(i, 1.05f, 0.9f, 1.0f));  // no improvement
        bool found = false;
        for (const auto& c : d.chapters())
            if (c.type == omnistream::AnomalyType_LOSS_PLATEAU) found = true;
        CHECK(found, "LOSS_PLATEAU detected after plateau_steps");
    }

    // ── 5. Improvement resets plateau counter ─────────────────────────────
    {
        AnomalyDetector::Config cfg;
        cfg.plateau_steps = 5;
        AnomalyDetector d("/dev/null", cfg);
        d.process(ev(0, 1.0f, 0.9f, 1.0f));
        for (int i = 1; i <= 4; ++i) d.process(ev(i, 1.05f, 0.9f, 1.0f));
        d.process(ev(5, 0.8f, 0.9f, 1.0f));   // improvement — resets counter
        for (int i = 6; i <= 9; ++i) d.process(ev(i, 0.85f, 0.9f, 1.0f));
        bool plateau = false;
        for (const auto& c : d.chapters())
            if (c.type == omnistream::AnomalyType_LOSS_PLATEAU) plateau = true;
        CHECK(!plateau, "improvement should reset plateau counter");
    }

    // ── 6. Chapter step matches event step ────────────────────────────────
    {
        AnomalyDetector d("/dev/null");
        for (int i = 0; i < 20; ++i) d.process(ev(i, 1.0f, 0.9f, 1.0f));
        d.process(ev(99, 1.0f, 0.9f, 100.0f));
        if (!d.chapters().empty()) {
            CHECK(d.chapters().back().step == 99, "chapter step matches event step");
        }
    }

    // ── 7. Severity bounded 0–1 ───────────────────────────────────────────
    {
        AnomalyDetector d("/dev/null");
        for (int i = 0; i < 20; ++i) d.process(ev(i, 1.0f, 0.9f, 1.0f));
        d.process(ev(20, 1.0f, 0.9f, 9999.0f));  // extreme spike
        for (const auto& c : d.chapters())
            CHECK(c.severity >= 0.0f && c.severity <= 1.0f,
                  "severity bounded 0-1");
    }

    std::cout << (failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
