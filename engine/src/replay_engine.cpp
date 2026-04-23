#define _CRT_SECURE_NO_WARNINGS
#define _CRT_NONSTDC_NO_DEPRECATE

#include "../incl/replay_engine.hpp"

#include <fcntl.h>
#ifdef _WIN32
    #define NOINMAX
    #include <windows.h>
    #include <io.h>
    #define ssize_t intptr_t
    #define PROT_READ     1
    #define PROT_WRITE    2
    #define MAP_SHARED    1
    #define MAP_FAILED    ((void *)-1)
    #define MS_SYNC       1

    static void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
        HANDLE hFile = (HANDLE)_get_osfhandle(fd);
        if (hFile == INVALID_HANDLE_VALUE) return MAP_FAILED;
        DWORD flProtect = PAGE_READONLY;
        if (prot & PROT_WRITE) flProtect = PAGE_READWRITE;
        HANDLE hMap = CreateFileMappingA(hFile, NULL, flProtect, 0, 0, NULL);
        if (!hMap) return MAP_FAILED;
        DWORD dwDesiredAccess = FILE_MAP_READ;
        if (prot & PROT_WRITE) dwDesiredAccess = FILE_MAP_WRITE;
        void* map = MapViewOfFile(hMap, dwDesiredAccess, 0, offset, length);
        CloseHandle(hMap); 
        return map ? map : MAP_FAILED;
    }
    static int munmap(void* addr, size_t) { return UnmapViewOfFile(addr) ? 0 : -1; }
#else
    #include <sys/mman.h>
    #include <unistd.h>
#endif
#include <sys/stat.h>
#include <cstring>
#include <cerrno>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <filesystem>
// Simple JSON field extractor
static bool json_string(std::string_view json, std::string_view key, std::string& out) {
    auto pos = json.find(key);
    if (pos == std::string_view::npos) return false;
    pos = json.find(':', pos);
    if (pos == std::string_view::npos) return false;
    pos = json.find('"', pos);
    if (pos == std::string_view::npos) return false;
    auto end = json.find('"', pos + 1);
    if (end == std::string_view::npos) return false;
    out = std::string(json.substr(pos + 1, end - pos - 1));
    return true;
}

static bool json_double(std::string_view json, std::string_view key, double& out) {
    auto pos = json.find(key);
    if (pos == std::string_view::npos) return false;
    pos = json.find(':', pos);
    if (pos == std::string_view::npos) return false;
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    try {
        std::size_t consumed;
        out = std::stod(std::string(json.substr(pos)), &consumed);
        return consumed > 0;
    } catch (...) { return false; }
}

//ReplayEngine

ReplayEngine::ReplayEngine(WsBroadcaster& bc) : broadcaster_(bc) {}

ReplayEngine::~ReplayEngine() {
    stop();
}bool ReplayEngine::open(const std::string& log_path) noexcept {
    if (map_ != nullptr && map_ != MAP_FAILED) {
        ::munmap(const_cast<uint8_t*>(map_), map_size_);
        map_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }

    std::string target_path = log_path;

    if (log_path == "baseline.omlog") {
        
        for (const auto& entry : std::filesystem::directory_iterator(".")) {
            std::string fname = entry.path().filename().string();
            if (fname.find("baseline_playing_") == 0) {
                try {
                    std::filesystem::remove(entry);
                } catch (...) { /* Ignore if Windows is currently holding it */ }
            }
        }

        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        target_path = "baseline_playing_" + std::to_string(now) + ".omlog";

        try {
            std::filesystem::copy_file(log_path, target_path, std::filesystem::copy_options::overwrite_existing);
        } catch (const std::exception& e) {
            std::cerr << "[replay] Failed to create shadow copy: " << e.what() << "\n";
            return false;
        }
    }

    fd_ = ::open(target_path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        std::cerr << "[replay] Cannot open " << target_path << ": " << std::strerror(errno) << "\n";
        return false;
    }

    struct stat st{};
    ::fstat(fd_, &st);
    map_size_ = static_cast<std::size_t>(st.st_size);

    map_ = static_cast<const uint8_t*>(
        ::mmap(nullptr, map_size_, PROT_READ, MAP_SHARED, fd_, 0)
    );
    if (map_ == MAP_FAILED) {
        std::cerr << "[replay] mmap failed: " << std::strerror(errno) << "\n";
        ::close(fd_); fd_ = -1; map_ = nullptr;
        return false;
    }

    if (map_size_ < sizeof(MmapLog::FileHeader)) {
        std::cerr << "[replay] File too small\n";
        return false;
    }
    const auto* hdr = reinterpret_cast<const MmapLog::FileHeader*>(map_);
    if (hdr->magic != MmapLog::FILE_MAGIC) {
        std::cerr << "[replay] Bad magic 0x" << std::hex << hdr->magic << "\n";
        return false;
    }
    total_records_ = hdr->record_count;

    std::cout << "[replay] Opened " << target_path << " records=" << total_records_ << "\n";

    chapters_.clear();
    const std::string sidecar = log_path + ".anomalies";
    int sfd = ::open(sidecar.c_str(), O_RDONLY);
    if (sfd >= 0) {
        uint32_t sz = 0;
        while (::read(sfd, &sz, 4) == 4) {
            std::vector<uint8_t> buf(sz);
            if (::read(sfd, buf.data(), sz) != static_cast<ssize_t>(sz)) break;

            flatbuffers::Verifier v(buf.data(), sz);
            const auto* marker = flatbuffers::GetRoot<omnistream::AnomalyMarker>(buf.data());
            AnomalyDetector::ChapterEntry entry;
            entry.step         = marker->step();
            entry.type         = marker->anomaly_type();
            entry.severity     = marker->severity();
            entry.metric_value = marker->metric_value();
            entry.description  = marker->description() ? marker->description()->str() : "";
            chapters_.push_back(entry);
        }
        ::close(sfd);
        std::cout << "[replay] Loaded " << chapters_.size() << " chapter(s)\n";
    }

    std::string topo_file = log_path;
    size_t ext_pos = topo_file.rfind(".omlog");
    if (ext_pos != std::string::npos) {
        topo_file.replace(ext_pos, 6, ".topology.json"); 
    }

    if (std::filesystem::exists(topo_file)) {
        std::ifstream ifs(topo_file);
        std::string topo_json((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        broadcaster_.broadcast_raw(topo_json);
        std::cout << "[replay] Broadcasted GNN Topology\n";
    }

    snapshots_.clear();
    std::string snap_file = log_path;
    size_t ext_pos2 = snap_file.rfind(".omlog");
    if (ext_pos2 != std::string::npos) {
        snap_file.replace(ext_pos2, 6, ".snapshots.bin");
    }

    int snap_fd = ::open(snap_file.c_str(), O_RDONLY);
    if (snap_fd >= 0) {
        uint64_t step = 0;
        uint32_t sz = 0;
        while (::read(snap_fd, &step, 8) == 8 && ::read(snap_fd, &sz, 4) == 4) {
            std::string buf(sz, '\0');
            if (::read(snap_fd, buf.data(), sz) == static_cast<ssize_t>(sz)) {
                snapshots_[step] = std::move(buf);
            } else break;
        }
        ::close(snap_fd);
        std::cout << "[replay] Loaded " << snapshots_.size() << " tensor snapshots\n";
    }

    current_step_.store(0, std::memory_order_relaxed);
    state_.store(State::PAUSED, std::memory_order_release); 
    frames_sent_.store(0, std::memory_order_relaxed);
    seek_target_.store(0, std::memory_order_release);

    return true;
}

void ReplayEngine::play(float speed) noexcept {
    speed_.store(speed, std::memory_order_release);
    state_.store(State::PLAYING, std::memory_order_release);
    
    // Check if the background worker actually exists!
    if (!thread_.joinable()) {
        thread_ = std::thread(&ReplayEngine::replay_loop, this);
    }
}

void ReplayEngine::pause() noexcept {
    state_.store(State::PAUSED, std::memory_order_release);
}

void ReplayEngine::seek(uint64_t step) noexcept {
    const uint64_t clamped = (std::min)(step, total_records_ > 0 ? total_records_ - 1 : 0);
    seek_target_.store(clamped, std::memory_order_release);
}

void ReplayEngine::stop() noexcept {
    state_.store(State::IDLE, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
    if (map_ && map_ != MAP_FAILED) {
        ::munmap(const_cast<uint8_t*>(map_), map_size_);
        map_ = nullptr;
    }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

void ReplayEngine::handle_command(std::string_view json) noexcept {
    std::string cmd;
    if (!json_string(json, "\"cmd\"", cmd)) return;

    if (cmd == "play") {
        double spd = 1.0;
        json_double(json, "\"speed\"", spd);
        play(static_cast<float>(spd));
    } else if (cmd == "pause") {
        pause();
    } else if (cmd == "seek") {
        double step = 0;
        json_double(json, "\"step\"", step);
        seek(static_cast<uint64_t>(step));
    } else if (cmd == "stop") {
        stop();
    }
}

ReplayEngine::ReplayStatus ReplayEngine::status() const noexcept {
    return {
        state_.load(std::memory_order_acquire),
        current_step_.load(std::memory_order_relaxed),
        total_records_,
        speed_.load(std::memory_order_relaxed),
        frames_sent_.load(std::memory_order_relaxed),
    };
}

void ReplayEngine::replay_loop() {
    using Clock    = std::chrono::steady_clock;
    using Duration = std::chrono::nanoseconds;

    constexpr std::size_t HDR_SIZE = sizeof(MmapLog::FileHeader);
    constexpr std::size_t REC_SIZE = sizeof(MmapLog::LogRecord);

    uint64_t pos     = current_step_.load(std::memory_order_relaxed);
    auto     t_epoch = Clock::now(); 

    uint64_t log_t0  = 0;
    if (total_records_ > 0 && map_) {
        const auto* first = reinterpret_cast<const MmapLog::LogRecord*>(map_ + HDR_SIZE);
        log_t0 = first->timestamp_ns;
    }

    while (state_.load(std::memory_order_acquire) != State::IDLE) {
        if (state_.load(std::memory_order_acquire) == State::PAUSED) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            t_epoch = Clock::now();  
            continue;
        }

        const uint64_t target = seek_target_.exchange(UINT64_MAX, std::memory_order_acq_rel);
        if (target != UINT64_MAX) {
            pos = (std::min)(target, total_records_ > 0 ? total_records_ - 1 : 0ULL);
            t_epoch = Clock::now();
            if (pos < total_records_ && map_) {
                const auto* rec = reinterpret_cast<const MmapLog::LogRecord*>(map_ + HDR_SIZE + pos * REC_SIZE);
                log_t0 = rec->timestamp_ns;
            }
        }

        if (!map_ || pos >= total_records_) {
            state_.store(State::PAUSED, std::memory_order_release);
            std::cout << "[replay] End of log at step " << pos << "\n";
            continue;
        }

        const auto* rec = reinterpret_cast<const MmapLog::LogRecord*>(map_ + HDR_SIZE + pos * REC_SIZE);

        if (rec->magic != MmapLog::RECORD_MAGIC) {
            std::cerr << "[replay] Bad record magic at step " << pos << "\n";
            ++pos;
            continue;
        }

        if (log_t0 > 0 && pos > 0) {
            const uint64_t log_elapsed_ns = rec->timestamp_ns - log_t0;
            const auto wall_target = t_epoch + std::chrono::nanoseconds(
                static_cast<int64_t>(static_cast<double>(log_elapsed_ns) / static_cast<double>(speed_.load(std::memory_order_relaxed)))
            );
            std::this_thread::sleep_until(wall_target);
        }

        TrainingEvent ev;
        ev.step         = rec->step;
        ev.loss         = rec->loss;
        ev.accuracy     = rec->accuracy;
        ev.grad_norm    = rec->grad_norm;
        ev.timestamp_ns = rec->timestamp_ns;

        broadcaster_.broadcast(ev);
        frames_sent_.fetch_add(1, std::memory_order_relaxed);
        
        current_step_.store(pos, std::memory_order_relaxed);
        ++pos;
    }
}

std::string ReplayEngine::chapters_json() const noexcept {
    std::ostringstream o;
    o << std::fixed << std::setprecision(3);
    o << "[";
    for (std::size_t i = 0; i < chapters_.size(); ++i) {
        const auto& c = chapters_[i];
        if (i) o << ",";
        o << "{"
          << "\"step\":"         << c.step         << ","
          << "\"type\":\""       << omnistream::EnumNameAnomalyType(c.type) << "\","
          << "\"severity\":"     << c.severity      << ","
          << "\"metric_value\":" << c.metric_value  << ","
          << "\"description\":\"";
        for (char ch : c.description) {
            if (ch == '"')       o << "\\\"";
            else if (ch == '\\') o << "\\\\";
            else                 o << ch;
        }
        o << "\"}";
    }
    o << "]";
    return o.str();
}