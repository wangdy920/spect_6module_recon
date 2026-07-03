#include "io/CSRTypes.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
  fs::path matrixDir;
  fs::path metadataPath;
  fs::path outPath;
  uint32_t colsOverride = 0;
  bool quiet = false;
};

struct MatrixMeta {
  bool available = false;
  uint32_t totalRows = 0;
  uint32_t totalVoxels = 0;
  uint32_t compactAngles = 0;
  uint32_t detNx = 0;
  uint32_t compactDetZ = 0;
  uint32_t raysPerAngle = 0;
  uint32_t fullAngles = 0;
  uint32_t nx = 0;
  uint32_t ny = 0;
  uint32_t nz = 0;
  int32_t use90 = 0;
  int32_t useZ = 0;
  int32_t use45 = 0;
  int32_t symCode = 0;
  int32_t symmetryOrder = 1;
};

struct ChunkInfo {
  fs::path path;
  uint32_t rows = 0;
  int32_t angleOffset = 0;
  int32_t angleCount = 0;
  uint64_t fileSize = 0;
};

[[noreturn]] void failUsage(const std::string &message) {
  throw std::runtime_error(message +
                           "\n\nUsage:\n"
                           "  spect_convert_6module_sysmat_to_csrb "
                           "--matrix-dir <dir> --out <matrix.csrb> "
                           "[--metadata <metadata.bin>] [--cols <N>] [--quiet]\n");
}

void printUsage() {
  std::cout
      << "Convert 6module metadata.bin + chunk_*.bin to SPECT-recon CSRB.\n\n"
      << "Usage:\n"
      << "  spect_convert_6module_sysmat_to_csrb --matrix-dir <dir> --out <matrix.csrb>\n\n"
      << "Options:\n"
      << "  --metadata <path>  Override metadata.bin path.\n"
      << "  --cols <N>         Column count when metadata.bin is unavailable.\n"
      << "  --quiet            Suppress progress output.\n\n"
      << "Notes:\n"
      << "  The converter writes direct compact rows. If metadata reports 45/90/Z\n"
      << "  symmetry, the CSRB is compact, not full-angle symmetry-expanded.\n";
}

Options parseArgs(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto requireValue = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc)
        failUsage("Missing value for " + name);
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      printUsage();
      std::exit(0);
    } else if (arg == "--matrix-dir") {
      opt.matrixDir = requireValue(arg);
    } else if (arg == "--metadata") {
      opt.metadataPath = requireValue(arg);
    } else if (arg == "--out") {
      opt.outPath = requireValue(arg);
    } else if (arg == "--cols") {
      const std::string value = requireValue(arg);
      const unsigned long long parsed = std::stoull(value);
      if (parsed == 0 || parsed > std::numeric_limits<uint32_t>::max())
        failUsage("--cols must be in 1..uint32_max");
      opt.colsOverride = static_cast<uint32_t>(parsed);
    } else if (arg == "--quiet") {
      opt.quiet = true;
    } else {
      failUsage("Unknown option: " + arg);
    }
  }

  if (opt.matrixDir.empty())
    failUsage("--matrix-dir is required");
  if (opt.outPath.empty())
    failUsage("--out is required");
  if (opt.metadataPath.empty())
    opt.metadataPath = opt.matrixDir / "metadata.bin";
  return opt;
}

uint64_t fileSize(const fs::path &path) {
  std::error_code ec;
  const auto size = fs::file_size(path, ec);
  if (ec)
    throw std::runtime_error("Cannot stat file: " + path.string());
  return static_cast<uint64_t>(size);
}

void readExact(std::istream &in, void *dst, size_t bytes,
               const std::string &context) {
  if (bytes == 0)
    return;
  in.read(static_cast<char *>(dst), static_cast<std::streamsize>(bytes));
  if (!in)
    throw std::runtime_error("Failed to read " + context);
}

void writeExact(std::ostream &out, const void *src, size_t bytes,
                const std::string &context) {
  if (bytes == 0)
    return;
  out.write(static_cast<const char *>(src), static_cast<std::streamsize>(bytes));
  if (!out)
    throw std::runtime_error("Failed to write " + context);
}

void skipBytes(std::istream &in, uint64_t bytes, const std::string &context) {
  if (bytes == 0)
    return;
  if (bytes > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()))
    throw std::runtime_error("Seek distance too large while skipping " + context);
  in.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
  if (!in)
    throw std::runtime_error("Failed to skip " + context);
}

bool startsWith(const std::string &s, const std::string &prefix) {
  return s.size() >= prefix.size() &&
         std::equal(prefix.begin(), prefix.end(), s.begin());
}

bool endsWith(const std::string &s, const std::string &suffix) {
  return s.size() >= suffix.size() &&
         std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

std::string lowerString(std::string s) {
  for (char &ch : s)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

bool isChunkFile(const fs::path &path) {
  const std::string name = lowerString(path.filename().string());
  return startsWith(name, "chunk_") && endsWith(name, ".bin");
}

int32_t readI32Word(const std::vector<uint8_t> &bytes, size_t index,
                    int32_t fallback = 0) {
  const size_t offset = index * sizeof(int32_t);
  if (offset + sizeof(int32_t) > bytes.size())
    return fallback;
  int32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

uint32_t checkedU32(int32_t value, const std::string &name) {
  if (value < 0)
    throw std::runtime_error(name + " is negative in metadata");
  return static_cast<uint32_t>(value);
}

MatrixMeta readMetadata(const fs::path &path) {
  MatrixMeta meta;
  if (!fs::exists(path))
    return meta;

  const uint64_t size = fileSize(path);
  if (size < 8 * sizeof(int32_t))
    throw std::runtime_error("metadata.bin is too small: " + path.string());
  const size_t readSize = static_cast<size_t>(
      std::min<uint64_t>(size, 24ull * sizeof(int32_t)));
  std::vector<uint8_t> bytes(readSize);
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open metadata: " + path.string());
  readExact(in, bytes.data(), bytes.size(), "metadata header");

  meta.available = true;
  meta.totalRows = checkedU32(readI32Word(bytes, 0), "total_rows");
  meta.totalVoxels = checkedU32(readI32Word(bytes, 1), "total_voxels");
  meta.compactAngles = checkedU32(readI32Word(bytes, 3), "compact_angles");
  meta.detNx = checkedU32(readI32Word(bytes, 4), "det_nx");
  meta.compactDetZ = checkedU32(readI32Word(bytes, 5), "compact_det_z");
  meta.raysPerAngle = checkedU32(readI32Word(bytes, 6), "rays_per_angle");
  meta.fullAngles = checkedU32(readI32Word(bytes, 7), "full_angles");
  meta.nx = checkedU32(readI32Word(bytes, 8), "Nx");
  meta.ny = checkedU32(readI32Word(bytes, 9), "Ny");
  meta.nz = checkedU32(readI32Word(bytes, 10), "Nz");
  meta.use90 = readI32Word(bytes, 17);
  meta.useZ = readI32Word(bytes, 18);
  meta.use45 = readI32Word(bytes, 19);
  meta.symCode = readI32Word(bytes, 20);
  meta.symmetryOrder = readI32Word(bytes, 21, meta.use45 ? 8 : (meta.use90 ? 4 : 1));
  if (meta.symmetryOrder <= 0)
    meta.symmetryOrder = 1;

  if (meta.totalRows == 0 || meta.totalVoxels == 0)
    throw std::runtime_error("metadata.bin has invalid rows/cols");
  return meta;
}

ChunkInfo readChunkHeader(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open chunk: " + path.string());
  int32_t rows = 0;
  int32_t angleOffset = 0;
  int32_t angleCount = 0;
  readExact(in, &rows, sizeof(rows), "chunk rows");
  readExact(in, &angleOffset, sizeof(angleOffset), "chunk angle_offset");
  readExact(in, &angleCount, sizeof(angleCount), "chunk angle_count");
  if (rows < 0 || angleCount < 0)
    throw std::runtime_error("Chunk header has negative dimensions: " +
                             path.string());
  return ChunkInfo{path, static_cast<uint32_t>(rows), angleOffset, angleCount,
                   fileSize(path)};
}

std::vector<ChunkInfo> discoverChunks(const fs::path &matrixDir) {
  if (!fs::is_directory(matrixDir))
    throw std::runtime_error("--matrix-dir is not a directory: " +
                             matrixDir.string());
  std::vector<ChunkInfo> chunks;
  for (const auto &entry : fs::directory_iterator(matrixDir)) {
    if (!entry.is_regular_file())
      continue;
    if (isChunkFile(entry.path()))
      chunks.push_back(readChunkHeader(entry.path()));
  }
  std::sort(chunks.begin(), chunks.end(), [](const ChunkInfo &a,
                                             const ChunkInfo &b) {
    if (a.angleOffset != b.angleOffset)
      return a.angleOffset < b.angleOffset;
    return a.path.filename().string() < b.path.filename().string();
  });
  if (chunks.empty())
    throw std::runtime_error("No chunk_*.bin files found in " +
                             matrixDir.string());
  return chunks;
}

uint64_t sumRows(const std::vector<ChunkInfo> &chunks) {
  uint64_t rows = 0;
  for (const auto &chunk : chunks)
    rows += chunk.rows;
  return rows;
}

std::ifstream openChunkPayload(const ChunkInfo &chunk) {
  std::ifstream in(chunk.path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open chunk: " + chunk.path.string());
  skipBytes(in, 3 * sizeof(int32_t), "chunk header");
  return in;
}

void scanRowPtr(const std::vector<ChunkInfo> &chunks, uint64_t totalRows,
                std::vector<uint32_t> &rowPtr, uint64_t &nnz) {
  rowPtr.clear();
  if (totalRows > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
    throw std::runtime_error("CSRB rows exceed uint32 range");
  rowPtr.reserve(static_cast<size_t>(totalRows) + 1);
  rowPtr.push_back(0);
  nnz = 0;

  for (const auto &chunk : chunks) {
    auto in = openChunkPayload(chunk);
    for (uint32_t row = 0; row < chunk.rows; ++row) {
      int32_t n = 0;
      readExact(in, &n, sizeof(n), "row nnz");
      if (n < 0)
        throw std::runtime_error("Negative row nnz in " + chunk.path.string());
      nnz += static_cast<uint32_t>(n);
      if (nnz > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
        throw std::runtime_error(
            "Total nnz exceeds current CSRB row_ptr uint32 range");
      rowPtr.push_back(static_cast<uint32_t>(nnz));
      skipBytes(in, static_cast<uint64_t>(n) *
                        (sizeof(int32_t) + sizeof(float)),
                "row payload");
    }
    const std::streamoff pos = in.tellg();
    if (pos < 0 || static_cast<uint64_t>(pos) != chunk.fileSize)
      throw std::runtime_error("Chunk size mismatch while scanning " +
                               chunk.path.string());
  }

  if (rowPtr.size() != static_cast<size_t>(totalRows) + 1)
    throw std::runtime_error("Scanned row count does not match metadata");
}

void writeColumns(const std::vector<ChunkInfo> &chunks, uint32_t cols,
                  std::ostream &out) {
  std::vector<int32_t> colBuf;
  for (const auto &chunk : chunks) {
    auto in = openChunkPayload(chunk);
    for (uint32_t row = 0; row < chunk.rows; ++row) {
      int32_t n = 0;
      readExact(in, &n, sizeof(n), "row nnz");
      if (n < 0)
        throw std::runtime_error("Negative row nnz in " + chunk.path.string());
      colBuf.resize(static_cast<size_t>(n));
      readExact(in, colBuf.data(), colBuf.size() * sizeof(int32_t),
                "row columns");
      for (int32_t col : colBuf) {
        if (col < 0 || static_cast<uint32_t>(col) >= cols)
          throw std::runtime_error("Column index out of range in " +
                                   chunk.path.string());
      }
      writeExact(out, colBuf.data(), colBuf.size() * sizeof(uint32_t),
                 "CSRB col_idx");
      skipBytes(in, static_cast<uint64_t>(n) * sizeof(float), "row values");
    }
  }
}

void writeValues(const std::vector<ChunkInfo> &chunks, std::ostream &out) {
  std::vector<float> valBuf;
  for (const auto &chunk : chunks) {
    auto in = openChunkPayload(chunk);
    for (uint32_t row = 0; row < chunk.rows; ++row) {
      int32_t n = 0;
      readExact(in, &n, sizeof(n), "row nnz");
      if (n < 0)
        throw std::runtime_error("Negative row nnz in " + chunk.path.string());
      skipBytes(in, static_cast<uint64_t>(n) * sizeof(int32_t), "row columns");
      valBuf.resize(static_cast<size_t>(n));
      readExact(in, valBuf.data(), valBuf.size() * sizeof(float), "row values");
      writeExact(out, valBuf.data(), valBuf.size() * sizeof(float),
                 "CSRB val");
    }
  }
}

void logMetaWarnings(const MatrixMeta &meta) {
  if (!meta.available)
    return;
  if (meta.symmetryOrder > 1 || meta.useZ != 0) {
    std::cerr
        << "[WARN] metadata reports compact symmetry: sym_code="
        << meta.symCode << ", symmetry_order=" << meta.symmetryOrder
        << ", use_z_symmetry=" << meta.useZ << "\n"
        << "[WARN] This tool writes the compact rows directly to CSRB. It does "
           "not expand full-angle/full-Z symmetry.\n"
        << "[WARN] Use this CSRB only with matching compact projection rows, "
           "or regenerate the system matrix without symmetry for standard "
           "SPECT-recon projection input.\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options opt = parseArgs(argc, argv);
    const MatrixMeta meta = readMetadata(opt.metadataPath);
    const auto chunks = discoverChunks(opt.matrixDir);
    const uint64_t chunkRows = sumRows(chunks);
    const uint64_t totalRows = meta.available ? meta.totalRows : chunkRows;
    const uint32_t cols =
        opt.colsOverride != 0 ? opt.colsOverride : meta.totalVoxels;

    if (cols == 0)
      throw std::runtime_error(
          "Column count is unknown. Provide metadata.bin or --cols <N>.");
    if (chunkRows != totalRows) {
      std::ostringstream oss;
      oss << "Chunk row sum (" << chunkRows << ") does not match total rows ("
          << totalRows << ")";
      throw std::runtime_error(oss.str());
    }
    if (totalRows > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
      throw std::runtime_error("CSRB rows exceed uint32 range");

    if (meta.available && meta.raysPerAngle > 0) {
      for (const auto &chunk : chunks) {
        const uint64_t expectedRows =
            static_cast<uint64_t>(chunk.angleCount) * meta.raysPerAngle;
        if (expectedRows != chunk.rows) {
          std::ostringstream oss;
          oss << "Chunk " << chunk.path.filename().string()
              << " rows=" << chunk.rows << " but angle_count*rays_per_angle="
              << expectedRows;
          throw std::runtime_error(oss.str());
        }
      }
    }

    if (!opt.quiet) {
      std::cout << "[convert] chunks=" << chunks.size()
                << ", rows=" << totalRows << ", cols=" << cols << "\n";
      if (meta.available) {
        std::cout << "[convert] metadata compact_angles=" << meta.compactAngles
                  << ", full_angles=" << meta.fullAngles
                  << ", rays_per_angle=" << meta.raysPerAngle << "\n";
      }
    }
    logMetaWarnings(meta);

    std::vector<uint32_t> rowPtr;
    uint64_t nnz = 0;
    if (!opt.quiet)
      std::cout << "[convert] pass 1/3: scanning row counts\n";
    scanRowPtr(chunks, totalRows, rowPtr, nnz);

    if (!opt.outPath.parent_path().empty())
      fs::create_directories(opt.outPath.parent_path());
    std::ofstream out(opt.outPath, std::ios::binary);
    if (!out)
      throw std::runtime_error("Cannot open output: " + opt.outPath.string());

    spect::io::CSRHeader header;
    header.rows = static_cast<uint32_t>(totalRows);
    header.cols = cols;
    header.nnz = nnz;
    writeExact(out, &header, sizeof(header), "CSRB header");
    writeExact(out, rowPtr.data(), rowPtr.size() * sizeof(uint32_t),
               "CSRB row_ptr");

    if (!opt.quiet)
      std::cout << "[convert] pass 2/3: writing columns\n";
    writeColumns(chunks, cols, out);

    if (!opt.quiet)
      std::cout << "[convert] pass 3/3: writing values\n";
    writeValues(chunks, out);
    out.close();
    if (!out)
      throw std::runtime_error("Failed while closing output: " +
                               opt.outPath.string());

    const uint64_t expectedSize =
        sizeof(spect::io::CSRHeader) +
        (static_cast<uint64_t>(header.rows) + 1) * sizeof(uint32_t) +
        nnz * sizeof(uint32_t) + nnz * sizeof(float);
    const uint64_t actualSize = fileSize(opt.outPath);
    if (actualSize != expectedSize)
      throw std::runtime_error("Output CSRB size mismatch after write");

    if (!opt.quiet) {
      const double rowPtrMiB =
          static_cast<double>(rowPtr.size() * sizeof(uint32_t)) /
          (1024.0 * 1024.0);
      std::cout << "[convert] done: nnz=" << nnz
                << ", output=" << opt.outPath.string()
                << ", row_ptr_memory=" << rowPtrMiB << " MiB\n";
    }
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "[ERROR] " << ex.what() << "\n";
    return 1;
  }
}
