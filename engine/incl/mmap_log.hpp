#pragma once
#include "types.hpp"
#include <string>
#include <cstdint>
#include <atomic>

// MmapLog
// Appends TrainingEvent records to a memory-mapped binary log file.
//
// Design
// ------
// The file grows in fixed-size chunks (EXTEND_BYTES).  When the write
// position reaches the mapped region's end, the file is extended with
// ftruncate() and remapped with mremap() (Linux) / a remap on macOS.
//
// Each record is a fixed-size LogRecord struct:
//   [magic: uint32][step: uint32][loss: float][accuracy: float]
//   [grad_norm: float][timestamp_ns: uint64][reserved: uint32]
// Total: 32 bytes per record — power of two, cache-aligned.
//
// Fixed-size records make replay O(1): seek(step * sizeof(LogRecord)).
// Phase 4's replay engine relies on this — do not change the struct layout.
//
// Thread safety
// -------------
// append() is called from the consumer thread only (same thread that drains
// the ring buffer).  record_count() is safe to read from any thread.
//
// File header
// -----------
// The first 64 bytes of the file are reserved for a FileHeader struct
// (magic, version, record_count, creation_timestamp_ns).  Records start
// at offset 64.  The replay engine reads the header first to validate the
// file and know how many records to expect.

class MmapLog {
public:
    struct alignas(32) LogRecord {
        uint32_t magic        = RECORD_MAGIC;
        uint32_t step         = 0;
        float    loss         = 0.0f;
        float    accuracy     = 0.0f;
        float    grad_norm    = 0.0f;
        uint32_t reserved     = 0;
        uint64_t timestamp_ns = 0;
        // Total: 4+4+4+4+4+4+8 = 32 bytes
    };

    struct FileHeader {
        uint32_t magic              = FILE_MAGIC;
        uint32_t version            = FILE_VERSION;
        uint64_t creation_ts_ns     = 0;
        uint64_t record_count       = 0;    // updated on close
        uint32_t record_size_bytes  = sizeof(LogRecord);
        uint32_t header_size_bytes  = 64;
        uint8_t  reserved[32]       = {};
        // Total: 64 bytes
    };

    static constexpr uint32_t RECORD_MAGIC  = 0x4F4D4C52u;  // 'OMLR'
    static constexpr uint32_t FILE_MAGIC    = 0x4F4D4C46u;  // 'OMLF'
    static constexpr uint32_t FILE_VERSION  = 0x00030000u;  // major=3 minor=0

    // EXTEND_BYTES: how many bytes to add per ftruncate call.
    // 4096 records × 32 bytes = 128 KiB per extension.
    static constexpr std::size_t EXTEND_RECORDS = 4096;
    static constexpr std::size_t EXTEND_BYTES   = EXTEND_RECORDS * sizeof(LogRecord);

    explicit MmapLog(const std::string& path);
    ~MmapLog();

    MmapLog(const MmapLog&)            = delete;
    MmapLog& operator=(const MmapLog&) = delete;

    // Append one record.  Called from the consumer thread only.
    // Returns false if the file cannot be extended (disk full, etc.).
    bool append(const TrainingEvent& ev) noexcept;

    // Flush and close.  Called once on engine shutdown.
    // Updates the record_count field in the file header.
    void close() noexcept;

    // Safe to read from any thread.
    uint64_t record_count() const noexcept {
        return record_count_.load(std::memory_order_relaxed);
    }

    const std::string& path() const noexcept { return path_; }

private:
    bool open_and_map() noexcept;
    bool extend_mapping() noexcept;
    void write_header() noexcept;
    std::size_t data_offset_for(uint64_t index) const noexcept;

    std::string path_;
    int         fd_          = -1;
    uint8_t*    map_         = nullptr;   // base of mmap region
    std::size_t map_size_    = 0;         // current mmap size in bytes
    std::size_t write_offset_= 0;         // byte offset of next write

    std::atomic<uint64_t> record_count_{0};
};
