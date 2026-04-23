#include "../incl/mmap_log.hpp"

#include <fcntl.h>
#ifdef _WIN32
    #include <windows.h>
    #include <io.h>
    
    // Define Linux mmap flags for Windows
    #define PROT_READ     1
    #define PROT_WRITE    2
    #define MAP_SHARED    1
    #define MAP_FAILED    ((void *)-1)
    #define MS_SYNC       1

    // Translate Linux ftruncate to Windows _chsize
    static int ftruncate(int fd, off_t length) {
        return _chsize(fd, (long)length);
    }

    // A lightweight shim to seamlessly translate Linux mmap() to Windows API
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

    // Translate Linux munmap() to Windows UnmapViewOfFile()
    static int munmap(void* addr, size_t /*length*/) {
        return UnmapViewOfFile(addr) ? 0 : -1;
    }

    // Translate Linux msync() to Windows FlushViewOfFile()
    static int msync(void* addr, size_t length, int /*flags*/) {
        return FlushViewOfFile(addr, length) ? 0 : -1;
    }
#else
    #include <sys/mman.h>
    #include <unistd.h>
#endif
#include <sys/stat.h>
#include <cstring>
#include <ctime>
#include <cerrno>
#include <iostream>
#include <chrono>

static uint64_t now_ns() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()
    );
}

MmapLog::MmapLog(const std::string& path) : path_(path) {
    if (!open_and_map()) {
        std::cerr << "[mmap_log] FATAL: could not open " << path << "\n";
    }
}

MmapLog::~MmapLog() {
    close();
}

bool MmapLog::open_and_map() noexcept {
    #ifdef _WIN32
        _sopen_s(&fd_, path_.c_str(), _O_RDWR | _O_CREAT | _O_TRUNC | _O_BINARY, _SH_DENYNO, _S_IREAD | _S_IWRITE);
    #else
        // Mac/Linux handles sharing perfectly by default
        fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
    #endif
    if (fd_ < 0) {
        std::cerr << "[mmap_log] open failed: " << std::strerror(errno) << "\n";
        return false;
    }

    // Initial size: header (64 bytes) + first extension of records.
    const std::size_t initial = sizeof(FileHeader) + EXTEND_BYTES;
    if (::ftruncate(fd_, static_cast<off_t>(initial)) < 0) {
        std::cerr << "[mmap_log] ftruncate failed: " << std::strerror(errno) << "\n";
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    map_ = static_cast<uint8_t*>(
        ::mmap(nullptr, initial, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0)
    );

    if (map_ == MAP_FAILED) {
        std::cerr << "[mmap_log] mmap failed: " << std::strerror(errno) << "\n";
        ::close(fd_);
        fd_ = -1;
        map_ = nullptr;
        return false;
    }

    map_size_     = initial;
    write_offset_ = sizeof(FileHeader);

    write_header();
    std::cout << "[mmap_log] Opened " << path_
              << " (" << (initial / 1024) << " KiB initial)\n";
    return true;
}

void MmapLog::write_header() noexcept {
    FileHeader hdr;
    hdr.creation_ts_ns    = now_ns();
    hdr.record_count      = 0;
    std::memcpy(map_, &hdr, sizeof(hdr));
}

bool MmapLog::extend_mapping() noexcept {
    if (fd_ < 0) return false;

    const std::size_t new_size = map_size_ + EXTEND_BYTES;

    if (::ftruncate(fd_, static_cast<off_t>(new_size)) < 0) {
        std::cerr << "[mmap_log] ftruncate(extend) failed: "
                  << std::strerror(errno) << "\n";
        return false;
    }

#ifdef __linux__
    // mremap is Linux-only — avoids unmap+remap round-trip.
    void* new_map = ::mremap(map_, map_size_, new_size, MREMAP_MAYMOVE);
    if (new_map == MAP_FAILED) {
        std::cerr << "[mmap_log] mremap failed: " << std::strerror(errno) << "\n";
        return false;
    }
    map_ = static_cast<uint8_t*>(new_map);
#else
    // macOS / BSD: unmap and remap.
    ::munmap(map_, map_size_);
    map_ = static_cast<uint8_t*>(
        ::mmap(nullptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0)
    );
    if (map_ == MAP_FAILED) {
        std::cerr << "[mmap_log] remap failed: " << std::strerror(errno) << "\n";
        map_ = nullptr;
        return false;
    }
#endif

    map_size_ = new_size;
    return true;
}

bool MmapLog::append(const TrainingEvent& ev) noexcept {
    if (!map_) return false;

    if (write_offset_ >= sizeof(FileHeader) + sizeof(LogRecord)) {
        const auto* last_rec = reinterpret_cast<const LogRecord*>(map_ + write_offset_ - sizeof(LogRecord));
        if (ev.step < last_rec->step) {
            std::cout << "\n[mmap_log] Python restart detected (Step " << last_rec->step << " -> " << ev.step << ")! Wiping old tape...\n";
            close();
            record_count_.store(0, std::memory_order_relaxed); // Reset the internal counter back to 0
            if (!open_and_map()) return false;
        }
    }
    
    // Extend if we'd overflow the current mapping.
    if (write_offset_ + sizeof(LogRecord) > map_size_) {
        if (!extend_mapping()) return false;
    }

    LogRecord rec;
    rec.magic        = RECORD_MAGIC;
    rec.step         = ev.step;
    rec.loss         = ev.loss;
    rec.accuracy     = ev.accuracy;
    rec.grad_norm    = ev.grad_norm;
    rec.timestamp_ns = ev.timestamp_ns;
    rec.reserved     = 0;

    std::memcpy(map_ + write_offset_, &rec, sizeof(rec));
    write_offset_ += sizeof(rec);

    const uint64_t count = record_count_.fetch_add(1, std::memory_order_relaxed) + 1;

    // Periodically sync record_count to the header so it's recoverable
    // if the engine crashes.  Every 256 records is cheap enough.
    auto* hdr = reinterpret_cast<FileHeader*>(map_);
    hdr->record_count = count;

    return true;
}

void MmapLog::close() noexcept {
    if (!map_) return;

    // Final record count into header.
    auto* hdr = reinterpret_cast<FileHeader*>(map_);
    hdr->record_count = record_count_.load(std::memory_order_relaxed);

    // Sync to disk before unmapping.
    ::msync(map_, map_size_, MS_SYNC);
    ::munmap(map_, map_size_);
    map_      = nullptr;
    map_size_ = 0;

    if (fd_ >= 0) {
        // Truncate file to the actual written size (remove unused reservation).
        const off_t actual = static_cast<off_t>(write_offset_);
        ::ftruncate(fd_, actual);
        ::close(fd_);
        fd_ = -1;
    }

    std::cout << "[mmap_log] Closed. records="
              << record_count_.load(std::memory_order_relaxed)
              << " path=" << path_ << "\n";
}
