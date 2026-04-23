#include "../incl/ws_broadcaster.hpp"

// uWebSockets must be included before any other headers that pull in sys/socket.h
// to avoid macro conflicts on Linux.

#define UWS_NO_ZLIB
#include <App.h>  // uWebSockets

#include <iostream>
#include <string>
#include <mutex>
#include <vector>
struct PerSocketData {};

struct WsBroadcasterImpl {
    uWS::App*   app  = nullptr;
    uWS::Loop*  loop = nullptr;
    std::vector<uWS::WebSocket<false, true, PerSocketData>*> clients;
    std::mutex  clients_mutex;
};

WsBroadcaster::WsBroadcaster(uint16_t port) : port_(port) {}
WsBroadcaster::~WsBroadcaster() { stop(); }

void WsBroadcaster::start() {
    running_.store(true, std::memory_order_release);

    thread_ = std::thread([this] {
        auto* impl   = new WsBroadcasterImpl{};
        app_ptr_     = impl;

        uWS::App app;
        impl->app  = &app;
        impl->loop = uWS::Loop::get();

        app.ws<PerSocketData>("/stream", {
            .compression      = uWS::DISABLED,
            .maxPayloadLength = 4 * 1024 * 1024,
            .idleTimeout      = 0,

            .open = [impl, this](auto* ws) {
                connected_clients_.fetch_add(1, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lk(impl->clients_mutex);
                impl->clients.push_back(ws);
                std::cout << "[ws:" << port_ << "] client connected. total="
                          << connected_clients_.load() << "\n";
            },

            // Forward incoming messages to the registered callback.
            // Used by replay mode to receive play/pause/seek commands.
            .message = [this](auto* ws, std::string_view msg, uWS::OpCode op) {
                std::cout << "\n[ws:" << port_ << "] RAW PACKET RECEIVED!\n"
                          << "   -> OpCode: " << (int)op << "\n"
                          << "   -> Payload: " << msg << "\n";

                if (!message_cb_) {
                    std::cout << "   -> ERROR: message_cb_ is NULL! The brain isn't connected!\n";
                } else {
                    std::cout << "   -> Passing to handler...\n";
                    message_cb_(msg);
                }
                std::cout << "--------------------------------\n";
            },

            .close = [impl, this](auto* ws, int code, std::string_view reason) {
                connected_clients_.fetch_sub(1, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lk(impl->clients_mutex);
                impl->clients.erase(
                    std::remove(impl->clients.begin(), impl->clients.end(), ws),
                    impl->clients.end()
                );
                std::cout << "[ws:" << port_ << "] client disconnected (code=" << code
                          << "). total=" << connected_clients_.load() << "\n";
            }
        })
        .listen(port_, [this](auto* listen_socket) {
            if (listen_socket) {
                std::cout << "[ws:" << port_ << "] listening\n";
            } else {
                std::cerr << "[ws:" << port_ << "] failed to bind\n";
                running_.store(false, std::memory_order_release);
            }
        })
        .run();

        delete impl;
        app_ptr_ = nullptr;
    });
}

void WsBroadcaster::stop() {
    running_.store(false, std::memory_order_release);
    if (app_ptr_) {
        auto* impl = static_cast<WsBroadcasterImpl*>(app_ptr_);
        if (impl->loop) {
            impl->loop->defer([impl] { impl->app->close(); });
        }
    }
    if (thread_.joinable()) thread_.join();
}

//Shared send helper 

static void send_to_all(WsBroadcasterImpl* impl,
                        std::atomic<uint64_t>& frames_sent,
                        std::string payload)
{
    impl->loop->defer([impl, &frames_sent, p = std::move(payload)] {
        std::lock_guard<std::mutex> lk(impl->clients_mutex);
        for (auto* ws : impl->clients)
            ws->send(p, uWS::OpCode::TEXT, false);
        frames_sent.fetch_add(
            static_cast<uint64_t>(impl->clients.size()),
            std::memory_order_relaxed);
    });
}

void WsBroadcaster::broadcast(const TrainingEvent& event) {
    if (!app_ptr_) return;
    send_to_all(static_cast<WsBroadcasterImpl*>(app_ptr_),
                frames_sent_, event.to_json());
}

void WsBroadcaster::broadcast_snapshot(const TensorSnapshot& snap) {
    if (!app_ptr_) return;
    send_to_all(static_cast<WsBroadcasterImpl*>(app_ptr_),
                frames_sent_, snap.to_json());
}

void WsBroadcaster::broadcast_topology(const std::string& json) {
    if (!app_ptr_) return;
    send_to_all(static_cast<WsBroadcasterImpl*>(app_ptr_),
                frames_sent_, json);
}

void WsBroadcaster::broadcast_raw(const std::string& json) {
    if (!app_ptr_) return;
    send_to_all(static_cast<WsBroadcasterImpl*>(app_ptr_),
                frames_sent_, json);
}
