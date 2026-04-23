#include "../incl/metrics_server.hpp"

#include "httplib.h"
#include <iostream>
#include <filesystem>
#include <chrono> 
#include <string>

MetricsServer::MetricsServer(uint16_t port) : port_(port) {}

MetricsServer::~MetricsServer() {
    stop();
}

void MetricsServer::start(SnapshotFn snapshot_fn, ChaptersFn chapters_fn) {
    running_.store(true, std::memory_order_release);

    thread_ = std::thread([this, sfn = std::move(snapshot_fn), cfn = std::move(chapters_fn)] {
        auto* svr = new httplib::Server();
        server_ptr_ = svr;

        svr->Get("/metrics", [&sfn](const httplib::Request&, httplib::Response& res) {
            res.set_content(sfn(), "application/json");
            res.set_header("Access-Control-Allow-Origin", "*");
        });

        svr->Get("/chapters", [&cfn](const httplib::Request&, httplib::Response& res) {
            // If the brain provided a chapters function, call it! Otherwise return empty array.
            std::string payload = cfn ? cfn() : "[]"; 
            res.set_content(payload, "application/json");
            res.set_header("Access-Control-Allow-Origin", "*");
        });

        svr->Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
            res.set_header("Access-Control-Allow-Origin", "*");
        });

        svr->Post("/api/save-baseline", [](const httplib::Request&, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            try {
                std::string base_name = "baseline";
                std::string new_file = base_name + ".omlog";
                std::string new_topology = base_name + ".topology.json";
                std::string new_snapshots = base_name + ".snapshots.bin";
                
                if (!std::filesystem::exists("training.omlog")) {
                    throw std::runtime_error("training.omlog not found. Is the Python script running?");
                }
                
                // 1. We can finally use the clean, blazing-fast OS copy now that the lock is fixed!
                std::filesystem::copy_file("training.omlog", new_file, std::filesystem::copy_options::overwrite_existing);
                
                // 2. Copy the static topology sidecar if it exists
                if (std::filesystem::exists("latest_topology.json")) {
                    std::filesystem::copy_file("latest_topology.json", new_topology, std::filesystem::copy_options::overwrite_existing);
                }

                // 3. Copy the binary snapshot sidecar if it exists
                if (std::filesystem::exists("latest_snapshots.bin")) {
                    std::filesystem::copy_file("latest_snapshots.bin", new_snapshots, std::filesystem::copy_options::overwrite_existing);
                }
                
                std::string json = "{\"status\":\"success\",\"file\":\"" + new_file + "\"}";
                res.set_content(json, "application/json");
                std::cout << "[metrics] Baseline successfully overwritten!\n";
                
            } catch(const std::exception& e) {
                res.status = 500;
                res.set_content(std::string("{\"error\":\"") + e.what() + "\"}", "application/json");
                std::cerr << "[metrics] Save failed: " << e.what() << "\n";
            }
        });

        svr->Options("/api/save-baseline", [](const httplib::Request&, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type");
        });

        std::cout << "[metrics] Listening on port " << port_ << "\n";
        svr->listen("0.0.0.0", static_cast<int>(port_));

        delete svr;
        server_ptr_ = nullptr;
    });
}

void MetricsServer::stop() {
    if (server_ptr_) {
        static_cast<httplib::Server*>(server_ptr_)->stop();
    }
    if (thread_.joinable()) {
        thread_.join();
    }
    running_.store(false, std::memory_order_release);
}