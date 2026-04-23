#include "../incl/zmq_receiver.hpp"
#include "../incl/ws_broadcaster.hpp"
#include "../incl/pipeline.hpp"
#include "../incl/replay_engine.hpp"
#include "../incl/metrics_server.hpp"

#include <iostream>
#include <csignal>
#include <atomic>
#include <chrono>
#include <thread>
#include <string>

static std::atomic<bool> g_shutdown{false};
static void handle_signal(int) { g_shutdown.store(true, std::memory_order_release); }

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --zmq    <endpoint>  ZMQ PULL endpoint    (default: tcp://127.0.0.1:5555)\n"
              << "  --port   <n>         Live WebSocket port   (default: 8080)\n"
              << "  --rport  <n>         Replay WebSocket port (default: 8081)\n"
              << "  --mport  <n>         Metrics HTTP port     (default: 9090)\n"
              << "  --log    <path>      Binary log path       (default: training.omlog)\n"
              << "  --replay <path>      Open log for replay   (skips live mode)\n"
              << "  --help               Show this message\n";
}

// Build a replay status JSON message to push to connected clients.
static std::string replay_status_json(const ReplayEngine& r) {
    const auto s = r.status();
    std::ostringstream o;
    o << "{\"type\":\"replay_status\""
      << ",\"state\":\"" << (s.state == ReplayEngine::State::PLAYING ? "playing"
                           : s.state == ReplayEngine::State::PAUSED  ? "paused" : "idle")
      << "\",\"step\":"        << s.current_step
      << ",\"total_steps\":"   << s.total_steps
      << ",\"speed\":"         << s.speed
      << ",\"frames_sent\":"   << s.frames_sent << "}";
    return o.str();
}

int main(int argc, char** argv) {
    std::string zmq_endpoint = "tcp://127.0.0.1:5555";
    uint16_t    ws_port      = 8080;
    uint16_t    replay_port  = 8081;
    uint16_t    metrics_port = 9090;
    std::string log_path     = "training.omlog";
    std::string replay_path;

    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if (flag == "--help") { print_usage(argv[0]); return 0; }
        if (i + 1 < argc) {
            if (flag == "--zmq")    zmq_endpoint = argv[++i];
            if (flag == "--port")   ws_port      = static_cast<uint16_t>(std::stoi(argv[++i]));
            if (flag == "--rport")  replay_port  = static_cast<uint16_t>(std::stoi(argv[++i]));
            if (flag == "--mport")  metrics_port = static_cast<uint16_t>(std::stoi(argv[++i]));
            if (flag == "--log")    log_path     = argv[++i];
            if (flag == "--replay") replay_path  = argv[++i];
        }
    }

    std::signal(SIGINT,  handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::cout << "=== OmniStream ML Engine — Phase 4 ===\n";

    WsBroadcaster live_broadcaster{ws_port};
    live_broadcaster.start();

    WsBroadcaster replay_broadcaster{replay_port};

    ReplayEngine replay{replay_broadcaster};

    replay_broadcaster.set_message_handler([&replay, &replay_broadcaster, replay_port](std::string_view msg) {
        std::cout << "[ws:" << replay_port << "] Received command from UI: " << msg << "\n";
        
        if (msg.find("\"reload\"") != std::string_view::npos) {
            std::string target_file = "baseline.omlog";
            
            // JSON Parsing ignore space
            auto file_pos = msg.find("\"file\"");
            if (file_pos != std::string_view::npos) {
                auto colon_pos = msg.find(":", file_pos);
                auto quote1 = msg.find("\"", colon_pos);
                if (quote1 != std::string_view::npos) {
                    auto quote2 = msg.find("\"", quote1 + 1);
                    if (quote2 != std::string_view::npos) {
                        target_file = std::string(msg.substr(quote1 + 1, quote2 - quote1 - 1));
                    }
                }
            }
            
            std::cout << "[replay] Hot-swapping tape to: " << target_file << "\n";
            replay.open(target_file);
            
            // Rewind the tape (Currently not working)
            replay.handle_command("{\"cmd\":\"pause\"}");
            replay.handle_command("{\"cmd\":\"seek\",\"step\":0}");
            
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            replay_broadcaster.broadcast_raw(replay_status_json(replay));
            return;
        }
        
        replay.handle_command(msg);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        replay_broadcaster.broadcast_raw(replay_status_json(replay));
    });

    if (!replay_path.empty()) {
        // Replay Mode
        std::cout << "Mode: replay  file=" << replay_path << "\n"
                  << "Replay WS port  : " << replay_port   << "\n"
                  << "Metrics port    : " << metrics_port  << "\n\n";

        if (!replay.open(replay_path)) {
            std::cerr << "[main] Cannot open replay file: " << replay_path << "\n";
            return 1;
        }

        replay_broadcaster.start();

        // HTTP API for /metrics (replay status) and /chapters.
        MetricsServer replay_api{metrics_port};
            replay_api.start(
                //The Snapshot Function (Metrics)
                [&replay] { return replay_status_json(replay); },
                
                // 2. The Chapters Function
                [&replay] { 
                    // Convert the Replay engine's chapters into a JSON string. Currently dont have any JSON file so it will return [] for now
                    return "[]";
                }
        );

        std::cout << "[replay] records=" << replay.status().total_steps
                  << "  chapters=" << replay.chapters().size() << "\n";
        std::cout << "[replay] Connect dashboard, switch to replay tab, then\n"
                  << "         send: {\"cmd\":\"play\",\"speed\":1.0}\n\n";

        // Periodically push status to keep the scrubber position updated
        // while playing (every 200ms is smooth enough).
        auto last_status = std::chrono::steady_clock::now();
        while (!g_shutdown.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            auto now = std::chrono::steady_clock::now();
            if (now - last_status >= std::chrono::milliseconds(200)) {
                // Only push if someone is connected to avoids pointless allocs.
                if (replay_broadcaster.connected_clients() > 0) {
                    replay_broadcaster.broadcast_raw(replay_status_json(replay));
                }
                last_status = now;
            }
        }

        replay.stop();
        replay_api.stop();

    } else {
        // Live mode
        std::cout << "Mode: live\n"
                  << "ZMQ endpoint  : " << zmq_endpoint << "\n"
                  << "Live WS port  : " << ws_port       << "\n"
                  << "Replay WS port: " << replay_port   << "\n"
                  << "Metrics port  : " << metrics_port  << "\n"
                  << "Log path      : " << log_path      << "\n\n";

        // In live mode the replay broadcaster is also started so users can
        // switch to replay mode in the dashboard without restarting the engine.
        replay_broadcaster.start();

        Pipeline::Config pcfg;
        pcfg.log_path     = log_path;
        pcfg.sidecar_path = log_path + ".anomalies";
        pcfg.metrics_port = metrics_port;

        Pipeline pipeline{live_broadcaster, pcfg};
        pipeline.start();

        ZmqReceiver receiver{zmq_endpoint};
        receiver.start(
            [&pipeline](const TrainingEvent& ev)    { pipeline.push(ev); },
            [&pipeline](const TensorSnapshot& snap) { pipeline.push_snapshot(snap); },
            [&pipeline](const std::string& json)    { pipeline.push_topology(json); }
        );

        auto last_print = std::chrono::steady_clock::now();
        while (!g_shutdown.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            auto now = std::chrono::steady_clock::now();
            if (now - last_print >= std::chrono::seconds(5)) {
                std::cout << "[stats] " << pipeline.metrics_json() << "\n";
                last_print = now;
            }
        }

        std::cout << "\n[main] Shutting down...\n";
        receiver.stop();
        pipeline.stop();
    }

    live_broadcaster.stop();
    replay_broadcaster.stop();
    std::cout << "[main] Clean exit.\n";
    return 0;
}
