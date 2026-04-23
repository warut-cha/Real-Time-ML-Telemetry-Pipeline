// adaptive_sampler_test.cpp
// ─────────────────────────
// Unit tests for AdaptiveSampler.
// Run: ./adaptive_sampler_test

#include "adaptive_sampler.hpp"
#include "types.hpp"

#include <iostream>
#include <cstdlib>
#include <cassert>

static TrainingEvent make_ev(uint32_t step, float loss, float acc, float grad) {
    TrainingEvent ev;
    ev.step      = step;
    ev.loss      = loss;
    ev.accuracy  = acc;
    ev.grad_norm = grad;
    ev.timestamp_ns = 0;
    return ev;
}

static int failures = 0;
#define CHECK(cond, msg) \
    do { if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } } while(0)

int main() {
    // ── 1. Passthrough mode below threshold ───────────────────────────────
    {
        AdaptiveSampler s;
        auto ev = make_ev(0, 1.0f, 0.5f, 1.0f);
        // fill_ratio = 0.3 < 0.5 threshold
        bool fwd = s.should_forward(ev, 0.3f);
        CHECK(fwd, "should forward below threshold");
        CHECK(s.current_mode() == AdaptiveSampler::Mode::PASSTHROUGH,
              "mode should be PASSTHROUGH below threshold");
    }

    // ── 2. Switches to CHANGE_POINT above high threshold ──────────────────
    {
        AdaptiveSampler s;
        auto ev = make_ev(0, 1.0f, 0.5f, 1.0f);
        s.should_forward(ev, 0.9f);  // first call seeds EMA
        auto ev2 = make_ev(1, 1.0f, 0.5f, 1.0f);  // no change
        s.should_forward(ev2, 0.9f);
        CHECK(s.current_mode() == AdaptiveSampler::Mode::CHANGE_POINT,
              "mode should switch to CHANGE_POINT above 0.8 fill");
    }

    // ── 3. Drops steady-state events in CHANGE_POINT mode ─────────────────
    {
        AdaptiveSampler s;
        // Seed EMA
        for (int i = 0; i < 10; ++i)
            s.should_forward(make_ev(i, 1.0f, 0.5f, 1.0f), 0.9f);

        // Same values — no change
        auto ev = make_ev(10, 1.0f, 0.5f, 1.0f);
        bool fwd = s.should_forward(ev, 0.9f);
        CHECK(!fwd, "should DROP steady-state in CHANGE_POINT mode");
        CHECK(s.total_dropped() > 0, "dropped counter should increment");
    }

    // ── 4. Forwards significant loss change in CHANGE_POINT mode ──────────
    {
        AdaptiveSampler s;
        for (int i = 0; i < 10; ++i)
            s.should_forward(make_ev(i, 1.0f, 0.5f, 1.0f), 0.9f);

        // Big loss spike
        auto ev = make_ev(10, 2.5f, 0.5f, 1.0f);
        bool fwd = s.should_forward(ev, 0.9f);
        CHECK(fwd, "should FORWARD significant loss change");
    }

    // ── 5. Forwards gradient explosion in CHANGE_POINT mode ───────────────
    {
        AdaptiveSampler s;
        for (int i = 0; i < 10; ++i)
            s.should_forward(make_ev(i, 1.0f, 0.5f, 1.0f), 0.9f);

        auto ev = make_ev(10, 1.0f, 0.5f, 8.0f);  // grad_norm spike
        bool fwd = s.should_forward(ev, 0.9f);
        CHECK(fwd, "should FORWARD gradient explosion");
    }

    // ── 6. Hysteresis: stays in CHANGE_POINT until fill drops below 0.5 ───
    {
        AdaptiveSampler s;
        auto ev = make_ev(0, 1.0f, 0.5f, 1.0f);
        s.should_forward(ev, 0.9f);  // enters CHANGE_POINT
        s.should_forward(ev, 0.6f);  // between 0.5 and 0.8 — stays
        CHECK(s.current_mode() == AdaptiveSampler::Mode::CHANGE_POINT,
              "should NOT leave CHANGE_POINT at fill=0.6 (hysteresis)");

        s.should_forward(ev, 0.4f);  // below 0.5 — reverts
        CHECK(s.current_mode() == AdaptiveSampler::Mode::PASSTHROUGH,
              "should revert to PASSTHROUGH below 0.5 fill");
    }

    // ── 7. Total accounting: seen == forwarded + dropped ──────────────────
    {
        AdaptiveSampler s;
        for (int i = 0; i < 50; ++i) {
            float loss = (i % 10 == 0) ? 3.0f : 1.0f;
            s.should_forward(make_ev(i, loss, 0.5f, 1.0f), 0.9f);
        }
        CHECK(s.total_seen() == s.total_forwarded() + s.total_dropped(),
              "seen == forwarded + dropped");
    }

    if (failures == 0)
        std::cout << "PASS (all checks)\n";
    else
        std::cout << "FAIL (" << failures << " check(s) failed)\n";

    return failures == 0 ? 0 : 1;
}
