#include "pipeline/RunManifest.h"

#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <iomanip>
#include <sstream>

#include "filters/PSFConfig.h"
#include "core/BedPathUtils.h"
#include "pipeline/RunLogger.h"
#include "regularizer/RegConfig.h"

namespace spect {
namespace main {

namespace {

std::string jsonEscape(const std::string &text) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string out;
  out.reserve(text.size() + 8);
  for (char ch : text) {
    const unsigned char code = static_cast<unsigned char>(ch);
    switch (ch) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (code < 0x20u) {
        out += "\\u00";
        out += hex[code >> 4u];
        out += hex[code & 0x0fu];
      } else {
        out += ch;
      }
      break;
    }
  }
  return out;
}

class JsonWriter {
public:
  explicit JsonWriter(std::ostringstream &out) : out_(out) {}

  void beginObject() {
    if (!firstField_.empty())
      throw std::logic_error("JsonWriter: root object already started");
    out_ << "{";
    firstField_.push_back(true);
  }

  void beginObject(const std::string &key) {
    fieldPrefix(key);
    out_ << "{";
    firstField_.push_back(true);
  }

  void endObject() {
    if (firstField_.empty())
      throw std::logic_error("JsonWriter: no object to end");
    const bool empty = firstField_.back();
    firstField_.pop_back();
    if (!empty)
      out_ << '\n' << std::string(firstField_.size() * 2, ' ');
    out_ << "}";
  }

  void string(const std::string &key, const std::string &value) {
    fieldPrefix(key);
    out_ << "\"" << jsonEscape(value) << "\"";
  }

  void integer(const std::string &key, long long value) {
    fieldPrefix(key);
    out_ << value;
  }

  void number(const std::string &key, double value) {
    fieldPrefix(key);
    writeNumber(value);
  }

  void boolean(const std::string &key, bool value) {
    fieldPrefix(key);
    out_ << (value ? "true" : "false");
  }

  template <typename T>
  void integerVector(const std::string &key, const std::vector<T> &values) {
    fieldPrefix(key);
    out_ << "[";
    for (size_t i = 0; i < values.size(); ++i) {
      if (i > 0)
        out_ << ", ";
      out_ << values[i];
    }
    out_ << "]";
  }

  void numberVector(const std::string &key,
                    const std::vector<double> &values) {
    fieldPrefix(key);
    out_ << "[";
    for (size_t i = 0; i < values.size(); ++i) {
      if (i > 0)
        out_ << ", ";
      writeNumber(values[i]);
    }
    out_ << "]";
  }

  void stringVector(const std::string &key,
                    const std::vector<std::string> &values) {
    fieldPrefix(key);
    out_ << "[";
    for (size_t i = 0; i < values.size(); ++i) {
      if (i > 0)
        out_ << ", ";
      out_ << "\"" << jsonEscape(values[i]) << "\"";
    }
    out_ << "]";
  }

private:
  void fieldPrefix(const std::string &key) {
    if (firstField_.empty())
      throw std::logic_error("JsonWriter: field outside object");
    if (firstField_.back()) {
      firstField_.back() = false;
    } else {
      out_ << ",";
    }
    out_ << '\n' << std::string(firstField_.size() * 2, ' ') << "\""
         << jsonEscape(key) << "\": ";
  }

  void writeNumber(double value) {
    if (std::isfinite(value)) {
      out_ << std::fixed << std::setprecision(17) << value;
    } else {
      out_ << "null";
    }
  }

  std::ostringstream &out_;
  std::vector<bool> firstField_;
};

std::string timestampUtcNow() {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &now);
#else
  gmtime_r(&now, &tm);
#endif
  char buffer[32] = {};
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buffer;
}

std::string psfModeName(filters::PSFMode mode) {
  switch (mode) {
  case filters::PSFMode::None:
    return "none";
  case filters::PSFMode::Operator:
    return "operator";
  case filters::PSFMode::Sieve:
    return "sieve";
  }
  return "unknown";
}

std::string psfKernelName(filters::PSFKernel kernel) {
  switch (kernel) {
  case filters::PSFKernel::Adjoint:
    return "adjoint";
  case filters::PSFKernel::Castor:
    return "castor";
  }
  return "unknown";
}


OutputGeometry deriveSavedGeometry(const OutputGeometry &geometry,
                                   const OutputTransform &transform) {
  if (geometry.nx <= 0 || geometry.ny <= 0 || geometry.nz <= 0)
    throw std::runtime_error("writeRunManifest: invalid output geometry");

  OutputGeometry saved = geometry;
  const size_t bedCount = transform.bedZStarts.size();
  if (transform.baseNz > 0 && bedCount > 1) {
    if (transform.outputBedOverlaps.size() + 1 != bedCount) {
      throw std::runtime_error(
          "writeRunManifest: output overlap count does not match beds");
    }

    if (transform.baseNz < transform.canonicalGeom.cropLayers) {
      throw std::runtime_error(
          "writeRunManifest: canonical output has an invalid bed depth");
    }
    for (int overlap : transform.outputBedOverlaps) {
      if (transform.baseNz - overlap != transform.canonicalGeom.stepLayers) {
        throw std::runtime_error(
            "writeRunManifest: canonical output has an invalid bed step");
      }
    }
    const long long savedNz =
        static_cast<long long>(transform.canonicalGeom.cropLayers) +
        static_cast<long long>(bedCount - 1) * transform.canonicalGeom.stepLayers;
    if (savedNz > static_cast<long long>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(
          "writeRunManifest: saved output depth is invalid");
    }
    saved.nz = static_cast<int>(savedNz);
  }

  auto applyCenterCrop = [](int requested, int &actual,
                            const char *axisName) {
    if (requested <= 0)
      return;
    if (requested > actual) {
      throw std::runtime_error(std::string("writeRunManifest: saved output ") +
                               axisName + " crop exceeds geometry");
    }
    actual = requested;
  };
  applyCenterCrop(transform.centerCropNx, saved.nx, "X");
  applyCenterCrop(transform.centerCropNy, saved.ny, "Y");
  applyCenterCrop(transform.centerCropNz, saved.nz, "Z");
  return saved;
}

} // namespace

void writeRunManifest(const RunManifestInput &input) {
  if (input.args == nullptr || input.compactMeta == nullptr ||
      input.bedZStarts == nullptr || input.outputTransform == nullptr ||
      input.geometry == nullptr) {
    throw std::runtime_error(
        "writeRunManifest: required input pointers are null");
  }

  const cli::ReconArgs &args = *input.args;
  const io::CompactSymmetryMetadata &meta = *input.compactMeta;
  const std::vector<int> &bedZStarts = *input.bedZStarts;
  const OutputTransform &transform = *input.outputTransform;
  const int bedCount = static_cast<int>(bedZStarts.size());
  const OutputGeometry &geometry = *input.geometry;
  const OutputGeometry savedGeometry =
      deriveSavedGeometry(geometry, transform);

  // Re-derive the applied per-bed gains from the exact source passed to the
  // final save. This also covers a transient sieve save copy without retaining
  // that full-volume copy after the manifest has been written.
  std::vector<double> fittedGains;
  if (bedCount > 1 && input.finalSaveVolume != nullptr &&
      !input.finalSaveVolume->empty() &&
      transform.baseNz >= transform.canonicalGeom.cropLayers && geometry.nx > 0 &&
      geometry.ny > 0) {
    const int cropStart = (transform.baseNz - transform.canonicalGeom.cropLayers) / 2;
    fittedGains = fitCenter66BedGains(*input.finalSaveVolume, geometry.nx,
                                      geometry.ny, transform, cropStart,
                                      geometry.vx, geometry.vy);
  }

  const bool bsremActive =
      args.algorithm == cli::ReconArgs::Algorithm::BSREM;
  const bool regularizationActive =
      bsremActive && args.useReg && args.reg.beta > 0.0 &&
      args.reg.type != RegType::None;
  std::vector<std::string> producedFiles;
  const std::string iterStem =
      input.outputPrefix + "_it" + std::to_string(input.savedIteration);
  producedFiles.push_back(iterStem + ".mhd");
  producedFiles.push_back(iterStem + ".raw");
  if (args.smoothApplyStage == "output") {
    producedFiles.push_back(iterStem + "_smooth.mhd");
    producedFiles.push_back(iterStem + "_smooth.raw");
  }
  if (args.butterApplyStage == "output") {
    producedFiles.push_back(iterStem + "_butter.mhd");
    producedFiles.push_back(iterStem + "_butter.raw");
  }

  std::vector<double> physicalZOffsetsMm(bedZStarts.size(), 0.0);
  for (size_t bed = 1; bed < bedZStarts.size(); ++bed) {
    const int stepLayers = transform.canonicalGeom.stepLayers;
    physicalZOffsetsMm[bed] =
        physicalZOffsetsMm[bed - 1] + stepLayers * geometry.vz;
  }
  std::ostringstream out;
  JsonWriter json(out);
  json.beginObject();
  json.string("schema", "spect_recon_run_manifest_v1");
  json.string("timestamp_utc", timestampUtcNow());
  json.string(
      "reference_prototype",
      "6module_bak_src/spect_recon_PSC_6module_24pin_YSO_bedmerge.cu");

  json.beginObject("matrix");
  json.string("chunk_csr_dir", args.chunkCSRDir);
  json.string("metadata_path",
              input.compactMetaPath != nullptr ? *input.compactMetaPath : "");
  json.string("metadata_layout",
              meta.legacyDoubleVoxelSizeLayout
                  ? "legacy_float64_voxel_92"
                  : "canonical_float32_voxel_88");
  json.integerVector("dims", std::vector<int>{meta.Nx, meta.Ny, meta.Nz});
  json.number("voxel_size_mm", meta.voxelSize);
  json.string("symmetry_mode", "sym90z");
  json.integer("symmetry_order", meta.symmetryOrder);
  json.integer("full_angles", meta.fullNumAngles);
  json.integer("det_nx", meta.detNx);
  json.integer("det_nz", meta.detNz);
  if (args.dualWindow()) {
    // Joint dual-window MLEM (MB-JR): second matrix + per-window weights.
    json.string("chunk_csr_dir2", args.chunkCSRDir2);
    json.numberVector("energy_window_weights", args.energyWindowWeights);
    json.string("multi_window_model", "joint_likelihood_mb_jr");
  }
  json.string("csrq_residency", args.csrqResidency);
  json.endObject();

  json.beginObject("execution");
  json.string("backend", "single_gpu_cuda");
  json.string("scheduler", "iteration_major_compact_chunk");
  json.string("matrix_layout", "compact_symmetric");
  json.boolean("multi_gpu", false);
  json.endObject();

  json.beginObject("projection");
  const std::vector<std::string> &bedFiles = args.projPaths;
  if (!bedFiles.empty()) {
    for (size_t bed = 0; bed < bedFiles.size(); ++bed) {
      const int bedNumber = spect::core::extractBedNumberFromPath(bedFiles[bed]);
      json.string(std::to_string(bedNumber > 0 ? bedNumber
                                               : static_cast<int>(bed) + 1),
                  bedFiles[bed]);
    }
  } else {
    json.string("1", args.projPath);
  }
  if (args.dualWindow()) {
    const std::vector<std::string> window2 =
        args.projPaths2.empty() ? std::vector<std::string>{args.projPath2}
                                : args.projPaths2;
    json.stringVector("window2_files", window2);
    std::vector<std::string> background2 = args.bProjPaths2;
    if (background2.empty() && !args.bProjPath2.empty())
      background2 = {args.bProjPath2};
    if (!background2.empty())
      json.stringVector("window2_background_files", background2);
  }
  json.boolean("merge_time_segments", args.mergeTimeSegments);
  if (args.mergeTimeSegments && !args.projectionSourceGroups.empty()) {
    json.integer("time_segment_count",
                 static_cast<long long>(
                     args.projectionSourceGroups.front().size()));
    json.string("time_segment_merge_operation",
                "elementwise_sum_angle_major_in_memory");
    for (size_t bed = 0; bed < args.projectionSourceGroups.size(); ++bed)
      json.stringVector("bed" + std::to_string(bed + 1) + "_sources",
                        args.projectionSourceGroups[bed]);
  }
  json.string("physical_z_convention", "Bed1=min_physical_z");
  json.string("bed_number_direction", "increasing_bed_number=+Z");
  json.boolean("z_bed_flip", args.zBedFlip);
  json.numberVector("physical_z_offsets_mm", physicalZOffsetsMm);
  json.string("local_z_direction",
              args.zBedFlip ? "increasing_local_z=-storage_z"
                            : "increasing_local_z=+storage_z");
  json.string("bad_detector_file", args.badDetectorFile);
  json.string("norm_factor_file", args.normFactorFile);
  json.endObject();

  json.beginObject("algorithm");
  json.string("opti", bsremActive ? "bsrem" : "mlem");
  json.integer("iterations", args.numIters);
  json.integer("save_interval", args.saveInterval);
  if (bsremActive) {
    json.beginObject("bsrem");
    json.number("alpha0", args.bsrem_alpha0);
    json.number("gamma", args.bsrem_gamma);
    json.string("alpha_schedule",
                "alpha(k)=alpha0/(1+gamma*k), k=iteration-1");
    json.boolean("curv_denom", args.bsremUseCurvDenom);
    json.boolean("curv_freeze", args.bsremFreezeCurv);
    json.integer("curv_warmup_iters", args.bsremCurvWarmupIters);
    json.number("curv_eps", args.bsremCurvEps);
    json.boolean("kappa_map", args.bsremUseKappaMap);
    json.number("kappa_eps", args.bsremKappaEps);
    json.numberVector(
        "kappa_clip",
        std::vector<double>{args.bsremKappaClipMin, args.bsremKappaClipMax});
    json.boolean("soft_mask", args.bsremSoftMask);
    json.number("soft_mask_lambda", args.bsremSoftMaskLambdaOut);
    json.number("sens_floor_rel", args.bsremSensFloorRel);
    json.boolean("mlem_equivalence", args.bsremMlemEquivalenceMode);
    json.boolean("diagnostics", args.bsremDiagnostics);
    json.boolean("regularization_enabled", regularizationActive);
    json.string("regularizer_type", regTypeToString(args.reg.type));
    json.number("regularizer_beta", args.reg.beta);
    json.number("regularizer_tv_eps", args.reg.tv_eps);
    json.number("regularizer_rdp_delta", args.reg.delta);
    json.number("regularizer_rdp_gamma", args.reg.gamma);
    json.boolean("regularizer_skip_outside_mask",
                 args.reg.skip_outside_mask);
    json.boolean("regularizer_ct_guidance", args.reg.ct_image != nullptr);
    json.number("regularizer_ct_sigma", args.reg.ct_sigma);
    json.number("regularizer_ct_weight_floor", args.reg.ct_weight_floor);
    json.endObject();
  }
  json.beginObject("psf");
  json.string("mode", psfModeName(args.psf.mode));
  json.string("kernel", psfKernelName(args.psf.kernel));
  json.number("fwhm_mm", args.psf.fwhm_mm);
  json.number("min_sigmas", args.psf.min_sigmas);
  json.boolean("z_blur", args.psf.z_blur);
  json.endObject();
  json.string("reg", regTypeToString(args.reg.type));
  json.number("reg_beta", args.reg.beta);
  json.boolean("ctac", args.doCTAC);
  json.string("attenuation_map_path", args.attMapPath);
  json.string("attenuation_map_z_layout",
              args.attMapPath.empty() ? "" : "local_matrix_z");
  json.string("regularizer_ct_path", args.regCtPath);
  json.string("regularizer_ct_z_layout",
              args.regCtPath.empty() ? "" : "reconstruction_storage_z");
  json.endObject();

  json.beginObject("multibed");
  json.string("stitch_method", "center66_cosine10");
  json.integer("bed_count", static_cast<long long>(bedZStarts.size()));
  json.integerVector("bed_z_starts", bedZStarts);
  json.integerVector("output_bed_overlaps",
                     transform.outputBedOverlaps);
  json.integer("bed_step_layers", transform.canonicalGeom.stepLayers);
  json.number("bed_step_mm",
              args.bedStepMm > 0.0 ? args.bedStepMm
                                   : spect::core::sixmodule::kMultiBedStepMm);
  json.boolean("single_bed_step_aligned", args.singleBedStepAligned);
  json.integer("overlap_layers", transform.canonicalGeom.blendOverlap);
  json.numberVector("fitted_bed_gains", fittedGains);
  json.string("gain_fit_method", "adjacent_overlap_log_median_mad");
  json.string("gain_anchor", "Bed1=1");
  json.string("gain_fit_reliability_policy",
              "prototype_log_chain_no_gate");
  json.endObject();

  json.beginObject("output");
  json.string("dir", args.outputDir);
  json.string("prefix", input.outputPrefix);
  json.integerVector(
      "dimensions",
      std::vector<int>{savedGeometry.nx, savedGeometry.ny, savedGeometry.nz});
  json.numberVector(
      "voxel_mm",
      std::vector<double>{savedGeometry.vx, savedGeometry.vy,
                          savedGeometry.vz});
  json.boolean("z_out_flip", transform.flipZWholeOutput);
  json.boolean("x_out_flip", transform.flipXWholeOutput);
  json.boolean("seam_xy_align", transform.seamXyAlign);
  json.string("increasing_z_index_direction",
              transform.flipZWholeOutput ? "-physical_z" : "+physical_z");
  json.integer("crop_end_start", args.cropEndStart);
  json.integer("crop_end_end", args.cropEndEnd);
  json.integer("saved_iteration", input.savedIteration);
  json.beginObject("filters");
  json.string("smooth_apply_stage", args.smoothApplyStage);
  json.number("smooth_fwhm_mm", args.outputSmoothFwhmMm);
  json.number("smooth_sigmas", args.outputSmoothSigmas);
  json.string("butter_apply_stage", args.butterApplyStage);
  json.number("butter_cutoff_cpcm", args.outputButterCutoffCpcm);
  json.integer("butter_order", args.outputButterOrder);
  json.boolean("nonnegative_clamp", args.outputNonnegativeClamp);
  json.boolean("overwrite_primary", args.outputFiltersOverwritePrimary);
  std::string primaryDerivation = "prepared_primary";
  if (args.outputFiltersOverwritePrimary) {
    const bool smooth = args.smoothApplyStage == "output";
    const bool butter = args.butterApplyStage == "output";
    if (smooth && butter)
      primaryDerivation = "smooth_then_butter";
    else if (smooth)
      primaryDerivation = "smooth";
    else if (butter)
      primaryDerivation = "butter";
  }
  json.string("primary_derivation", primaryDerivation);
  json.string("derivation", "independent_from_prepared_primary");
  json.endObject();
  json.stringVector("files", producedFiles);
  json.endObject();
  json.endObject();
  out << '\n';

  const std::filesystem::path manifestPath =
      std::filesystem::path(args.outputDir) /
      (input.outputPrefix + "_manifest.json");
  std::ofstream manifestOut(manifestPath,
                            std::ios::binary | std::ios::trunc);
  if (!manifestOut) {
    throw std::runtime_error("Failed opening run manifest: " +
                             manifestPath.string());
  }
  const std::string contents = out.str();
  manifestOut.write(contents.data(),
                    static_cast<std::streamsize>(contents.size()));
  manifestOut.flush();
  bool ioFailed = !manifestOut;
  manifestOut.close();
  ioFailed = ioFailed || manifestOut.fail();
  if (ioFailed) {
    throw std::runtime_error("Failed writing or closing run manifest: " +
                             manifestPath.string());
  }
  logInfo("Run manifest: " + manifestPath.string());
}

} // namespace main
} // namespace spect
