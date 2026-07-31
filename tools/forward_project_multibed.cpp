// spect_forward_project - matched-model synthetic multi-bed forward projector.
//
// Forward-projects a known continuous global phantom through the exact same
// five-bed (or N-bed) compact-symmetric operator chain used by the production
// joint reconstruction:
//
//   global PSF forward (Operator mode, same kernels)  [optional]
//   -> per-bed local window extraction with virtual Z flip (joint semantics)
//   -> compact chunk CUDA forward projection to the native full-angle layout
//   -> per-bed BIN projection files (2 x int32 header + double payload)
//
// The resulting Bed*.bin files can be fed back into SPECT_recon -dfList for a
// noise-free matched-model closed loop, to separate implementation/contract
// defects from unmodelled bed-dependent response in real data.

#include "app/BedOffsets.h"
#include "app/ProjectionOps.h"
#include "app/ReconConfig.h"
#include "cuda/CompactSymmetricProjector.h"
#include "filters/Convolution.h"
#include "filters/PSFConfig.h"
#include "io/CompactChunkCSR.h"
#include "io/CompactSymmetryMetadata.h"
#include "io/VolumeLoader.h"
#include "cuda/SparseOpsBackend.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ToolArgs {
  std::string chunkDir;
  std::string phantomPath;
  std::string outDir;
  int nx = 108;
  int ny = 108;
  int baseNz = 80;
  int bedCount = 5;
  std::vector<int> bedOverlaps;  // per-joint; single value is replicated
  bool zBedFlip = true;
  double psfFwhmMm = 0.0;  // <=0 disables PSF
  double psfMinSigmas = 4.0;
  double spacingMm = 0.3;
  int threads = 8;
};

[[noreturn]] void usage(const char *reason) {
  std::cerr << "Error: " << reason << "\n"
            << "Usage: spect_forward_project -chunkCSRDir <dir> -phantom <mhd>\n"
            << "         -outDir <dir> -dim Nx,Ny,baseNz -beds N\n"
            << "         -bedOverlap o1[,o2,...] [-zbedflip 0|1]\n"
            << "         [-psf fwhm_mm[,min_sigmas]] [-vox spacing_mm] [-th N]\n";
  throw std::runtime_error(reason);
}

std::vector<double> parseDoubles(const std::string &text) {
  std::vector<double> out;
  size_t start = 0;
  while (start <= text.size()) {
    const size_t comma = text.find(',', start);
    const std::string token =
        text.substr(start, comma == std::string::npos ? std::string::npos
                                                      : comma - start);
    if (token.empty())
      usage("empty numeric list element");
    out.push_back(std::stod(token));
    if (comma == std::string::npos)
      break;
    start = comma + 1;
  }
  return out;
}

ToolArgs parseArgs(int argc, char **argv) {
  ToolArgs args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (++i >= argc)
        usage((std::string("missing value for ") + name).c_str());
      return argv[i];
    };
    if (flag == "-chunkCSRDir") {
      args.chunkDir = next("-chunkCSRDir");
    } else if (flag == "-phantom") {
      args.phantomPath = next("-phantom");
    } else if (flag == "-outDir") {
      args.outDir = next("-outDir");
    } else if (flag == "-dim") {
      const auto dims = parseDoubles(next("-dim"));
      if (dims.size() != 3)
        usage("-dim expects Nx,Ny,baseNz");
      args.nx = static_cast<int>(dims[0]);
      args.ny = static_cast<int>(dims[1]);
      args.baseNz = static_cast<int>(dims[2]);
    } else if (flag == "-beds") {
      args.bedCount = std::stoi(next("-beds"));
    } else if (flag == "-bedOverlap") {
      const auto values = parseDoubles(next("-bedOverlap"));
      for (double v : values)
        args.bedOverlaps.push_back(static_cast<int>(v));
    } else if (flag == "-zbedflip") {
      args.zBedFlip = std::stoi(next("-zbedflip")) != 0;
    } else if (flag == "-psf") {
      const auto values = parseDoubles(next("-psf"));
      if (values.empty() || values.size() > 2)
        usage("-psf expects fwhm_mm[,min_sigmas]");
      args.psfFwhmMm = values[0];
      if (values.size() == 2)
        args.psfMinSigmas = values[1];
    } else if (flag == "-vox") {
      args.spacingMm = std::stod(next("-vox"));
    } else if (flag == "-th") {
      args.threads = std::stoi(next("-th"));
    } else {
      usage(("unknown argument: " + flag).c_str());
    }
  }
  if (args.chunkDir.empty() || args.phantomPath.empty() || args.outDir.empty())
    usage("-chunkCSRDir, -phantom and -outDir are required");
  if (args.bedCount < 1)
    usage("-beds must be >= 1");
  if (args.bedOverlaps.empty())
    args.bedOverlaps.push_back(0);
  if (args.bedOverlaps.size() == 1 && args.bedCount > 2)
    args.bedOverlaps.assign(static_cast<size_t>(args.bedCount - 1),
                            args.bedOverlaps.front());
  if (args.bedOverlaps.size() < static_cast<size_t>(args.bedCount - 1))
    usage("not enough -bedOverlap joints for the requested bed count");
  return args;
}

void writeProjectionBin(const std::filesystem::path &path, int32_t numAngles,
                        int32_t raysPerAngle,
                        const std::vector<double> &payload) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out)
    throw std::runtime_error("cannot open output projection BIN: " +
                             path.string());
  out.write(reinterpret_cast<const char *>(&numAngles), sizeof(numAngles));
  out.write(reinterpret_cast<const char *>(&raysPerAngle), sizeof(raysPerAngle));
  out.write(reinterpret_cast<const char *>(payload.data()),
            static_cast<std::streamsize>(payload.size() * sizeof(double)));
  out.flush();
  if (!out)
    throw std::runtime_error("failed writing projection BIN: " +
                             path.string());
}

} // namespace

int main(int argc, char **argv) {
  try {
    const ToolArgs args = parseArgs(argc, argv);

    const std::filesystem::path metaPath =
        std::filesystem::path(args.chunkDir) / "metadata.bin";
    const spect::io::CompactSymmetryMetadata meta =
        spect::io::loadCompactSymmetryMetadata(metaPath.string());
    if (meta.Nx != args.nx || meta.Ny != args.ny || meta.Nz != args.baseNz) {
      throw std::runtime_error(
          "compact metadata geometry does not match -dim: metadata " +
          std::to_string(meta.Nx) + "x" + std::to_string(meta.Ny) + "x" +
          std::to_string(meta.Nz) + " vs " + std::to_string(args.nx) + "x" +
          std::to_string(args.ny) + "x" + std::to_string(args.baseNz));
    }
    spect::io::CompactChunkCSRDirectory chunks =
        spect::io::openCompactChunkCSRDirectory(args.chunkDir, meta);

    const std::vector<int> bedZStarts = spect::app::computeBedZOffsets(
        args.bedCount, args.baseNz, args.bedOverlaps);
    const int globalNz = bedZStarts.back() + args.baseNz;

    int phantomNx = 0, phantomNy = 0, phantomNz = 0;
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
    std::vector<double> phantom = spect::io::loadVolumeWithSpacing(
        args.phantomPath, phantomNx, phantomNy, phantomNz, dx, dy, dz);
    if (phantomNx != args.nx || phantomNy != args.ny ||
        phantomNz != globalNz) {
      throw std::runtime_error(
          "phantom geometry mismatch: got " + std::to_string(phantomNx) + "x" +
          std::to_string(phantomNy) + "x" + std::to_string(phantomNz) +
          ", expected " + std::to_string(args.nx) + "x" +
          std::to_string(args.ny) + "x" + std::to_string(globalNz));
    }

    spect::ReconOpts opts;
    opts.threads = std::max(1, args.threads);
    opts.sparseOps = &spect::getSparseOpsCUDA();

    spect::filters::PSFKernels kernels;
    const spect::filters::PSFKernels *psfPtr = nullptr;
    if (args.psfFwhmMm > 0.0) {
      spect::filters::PSFConf psfConf;
      psfConf.mode = spect::filters::PSFMode::Operator;
      psfConf.fwhm_mm = args.psfFwhmMm;
      psfConf.min_sigmas = args.psfMinSigmas;
      psfConf.z_blur = true;
      psfConf.vox_x = args.spacingMm;
      psfConf.vox_y = args.spacingMm;
      psfConf.vox_z = args.spacingMm;
      opts.psf = psfConf;
      kernels = spect::filters::make_psf_kernels(psfConf);
      psfPtr = &kernels;
    }

    // Match the joint reconstruction forward chain: global Operator PSF first,
    // then per-bed local window extraction (with virtual Z flip), then the
    // compact chunk CUDA forward projection per bed.
    const std::vector<double> *bedSource = &phantom;
    std::vector<double> blurred;
    if (psfPtr != nullptr) {
      spect::apply_psf_forward(phantom, blurred, args.nx, args.ny, globalNz,
                               opts, psfPtr, opts.threads);
      bedSource = &blurred;
    }

    const size_t fullRows = static_cast<size_t>(meta.fullRows());
    std::filesystem::create_directories(args.outDir);
    std::vector<double> sums(static_cast<size_t>(args.bedCount), 0.0);
    for (int bed = 0; bed < args.bedCount; ++bed) {
      const int zStart = bedZStarts[static_cast<size_t>(bed)];
      const std::vector<double> local =
          spect::app::extractBedVolumeFromGlobal(
              *bedSource, args.nx, args.ny, args.baseNz, zStart,
              args.zBedFlip);
      std::vector<double> expected(fullRows, 0.0);
      const bool ok = spect::cuda::computeCompactChunkForwardProjectionCUDA(
          chunks, meta, local.data(), opts.threads, expected.data(), nullptr);
      if (!ok) {
        throw std::runtime_error(
            "compact chunk forward projection requires the CUDA backend.");
      }
      double sum = 0.0;
      for (double v : expected) {
        if (!std::isfinite(v) || v < 0.0)
          throw std::runtime_error("non-finite or negative expected bin.");
        sum += v;
      }
      sums[static_cast<size_t>(bed)] = sum;
      const std::filesystem::path outPath =
          std::filesystem::path(args.outDir) /
          ("Bed" + std::to_string(bed + 1) + ".bin");
      writeProjectionBin(outPath, meta.fullNumAngles,
                         meta.fullRaysPerAngle(), expected);
      std::cout << "[bed " << (bed + 1) << "/" << args.bedCount
                << "] zStart=" << zStart << " sum=" << sum << " -> "
                << outPath.string() << "\n";
    }

    std::ofstream manifest(
        std::filesystem::path(args.outDir) / "forward_manifest.txt",
        std::ios::binary | std::ios::trunc);
    manifest << "phantom=" << args.phantomPath << "\n"
             << "chunk_dir=" << args.chunkDir << "\n"
             << "beds=" << args.bedCount << "\n"
             << "base_nz=" << args.baseNz << "\n"
             << "global_nz=" << globalNz << "\n"
             << "bed_z_starts=";
    for (size_t i = 0; i < bedZStarts.size(); ++i)
      manifest << (i == 0 ? "" : ",") << bedZStarts[i];
    manifest << "\n"
             << "z_bed_flip=" << (args.zBedFlip ? 1 : 0) << "\n"
             << "psf_fwhm_mm=" << args.psfFwhmMm << "\n"
             << "psf_min_sigmas=" << args.psfMinSigmas << "\n"
             << "psf_kernel_x=" << (psfPtr ? psfPtr->kx.size() : 0) << "\n"
             << "full_rows_per_bed=" << fullRows << "\n";
    for (size_t bed = 0; bed < sums.size(); ++bed)
      manifest << "bed" << (bed + 1) << "_sum=" << sums[bed] << "\n";
    std::cout << "done. globalNz=" << globalNz << "\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "spect_forward_project failed: " << ex.what() << "\n";
    return 1;
  }
}
