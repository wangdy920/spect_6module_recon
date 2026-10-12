// ReconArgs.h - Reconstruction argument structure
// Extracted from SPECT-recon.h for modular CLI architecture
#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

// Use modular headers instead of original_source_backup
#include "filters/PSFConfig.h"       // PSFConf, PSFMode
#include "regularizer/RegConfig.h"   // RegConf, RegType

// Bring types into current namespace for convenience
using spect::filters::PSFConf;
using spect::filters::PSFKernel;
using spect::filters::PSFMode;
using spect::RegConf;
using spect::RegType;

namespace spect {
namespace cli {

/**
 * @brief Complete reconstruction arguments structure.
 * 
 * This structure holds all command-line and configuration file parameters
 * for SPECT reconstruction.
 */
struct ReconArgs {
    // ========== Basic Reconstruction Parameters ==========
    int Nx = 0, Ny = 0, Nz = 0;  ///< Image dimensions (voxels)
    bool dimExplicit = false;    ///< True if dimensions explicitly set
    float Vx = 1, Vy = 1, Vz = 1; ///< Voxel sizes (mm)
    double dx = 1.0, dy = 1.0, dz = 1.0; ///< Double precision spacing
    int detectorChannels = 40;    ///< Detector channels (fallback)
    int detNz = 0, detNx = 0;     ///< 2D detector dimensions
    int numAngles = 72;           ///< Number of projection angles
    std::string badDetectorFile;   ///< Global numbers or hardware IDs; empty list disables masking
    bool badDetectorFileExplicit = false; ///< CLI path is resolved from cwd, not config dir
    int numIters = 50;            ///< Total iterations
    /// Ordered-subset schedule over compact angles: "N" or "N1:i1,N2:i2,..."
    /// (iterations summing to numIters, each N dividing the compact angle
    /// count and the previous N). "1" is the full-data update (MLEM / BSREM).
    std::string subsetSchedule = "1";
    int saveInterval = 0;         ///< Save every N full iterations; 0 = final only
    int numThreads = 8;           ///< Thread count
    std::string cudaDeviceValue;  ///< Requested CUDA device id from env/CLI
    bool cudaDeviceRequested = false; ///< True if CUDA device selection was requested
    bool cudaDeviceFromCli = false;   ///< True if -cudaDevice overrides env

    // ========== Input/Output Paths ==========
    std::string projPath;                ///< Single-bed projection file
    std::vector<std::string> projPaths;  ///< Multi-bed projection list
    int bedFirst = 0;                    ///< -beds: first BedK.bin to search (0 = unset)
    int bedLast = 0;                     ///< -beds: last BedK.bin to search, inclusive
    bool mergeTimeSegments = false;      ///< Sum matching BedN data across sibling timestamp/SPECT directories
    bool outputFiltersOverwritePrimary = false; ///< Write output-filter chain to the unsuffixed volume
    bool matrixGridOutput = false; ///< Single-bed output without PSC center cropping
    std::vector<std::vector<std::string>> projectionSourceGroups; ///< Actual per-bed sources recorded in the manifest
    std::string outputDir = "output";    ///< Output directory
    std::string outputName;              ///< Output filename base

    // ========== Config Provenance ==========
    std::string configPath = "spect_config.ini";
    bool configExplicit = false;  ///< -config was used explicitly
    bool configFound = false;     ///< Config file exists
    bool configLoaded = false;    ///< Config file loaded successfully
    bool reconSectionsOverrideCli = false; ///< Selected recipe/global reconstruction keys override CLI when true
    std::string isotope;          ///< Optional acquisition isotope; checked against the selected recipe
    std::string projectionModel;  ///< single, merged or joint; projection contract supplied by wrapper
    std::string matrixRoot = "matrix"; ///< Executable-relative deployment root
    std::string scannerConfig; ///< SN-specific matrix settings, relative to the global config.
    std::string scannerName;
    std::string projectionInputMode;
    std::string matrixId, matrixId2; ///< Stable directory IDs, never guessed from isotope
    bool chunkCSRDir2Explicit = false;
    std::string normFactorFile2;   ///< Second channel correction c (never inherit channel 1 norm)
    std::string algoConfPath;     ///< Algorithm config file path (empty = auto-detect)
    bool helpRequested = false;    ///< Print usage and exit successfully
    bool versionRequested = false; ///< Print version and exit successfully
    bool dumpConf = false;        ///< Print merged config and exit

    // ========== Filtering Options ==========
    bool useSmooth = false;
    float smoothSigma = 1.0f;
    int smoothKernel = 3;
    bool useButter = false;
    float butterCutoff = 0.35f;
    int butterOrder = 2;
    // Filter apply stages: "none" | "in_recon" | "output".
    // CLI -smooth/-butter imply "in_recon"; INI may set the stage explicitly.
    std::string smoothApplyStage = "none"; ///< Gaussian smoothing stage
    std::string butterApplyStage = "none"; ///< Butterworth stage
    float outputSmoothFwhmMm = 0.30f;       ///< Output Gaussian FWHM (mm)
    float outputSmoothSigmas = 3.0f;        ///< Output kernel radius (N*sigma)
    float outputButterCutoffCpcm = 5.0f;    ///< Output cutoff (cycles/cm)
    int outputButterOrder = 8;              ///< Output Butterworth order
    bool outputNonnegativeClamp = true;     ///< Clamp saved copies after filtering
    bool usePSF = false;
    PSFConf psf;                  ///< PSF configuration
    bool psfModeExplicit = false; ///< -psfMode was used explicitly

    // ========== Regularization ==========
    bool useReg = false;
    RegConf reg;  ///< Regularization configuration

    // ========== Cylindrical Mask ==========
    bool useCylindricalMask = true;
    float cylinderDiameterMM = 30.0f;
    float cylinderHeightMM = 21.6f;
    bool cylinderDimsExplicit = false;
    double cylCenterX = std::numeric_limits<double>::quiet_NaN();
    double cylCenterY = std::numeric_limits<double>::quiet_NaN();
    double cylCenterZ = std::numeric_limits<double>::quiet_NaN();

    // ========== System Matrix Options ==========
    bool doCTAC = false;          ///< Iterative CTAC
    std::string ctacModel = "line-integral"; ///< fixed production model
    int ctacRotSign = 1;          ///< Voxel rotation sign matching sysmat generator (+1/-1)
    std::string ctacAutoMuFrom;   ///< Early-iteration recon volume for auto water mu-map
    double ctacAutoMuThresh = 0.05;  ///< Relative threshold vs P99.9 activity
    double ctacAutoMuValue = 0.0152; ///< Water mu (mm^-1) used to fill the body
    std::string ctacAutoMuNuclide = "Tc99m"; ///< Water mu lookup key
    bool ctacAutoMuValueExplicit = false; ///< true when mu was manually set
    std::string ctacAutoMuValueSource = "Tc99m";
    /// Photon energy (keV) the supplied mu map / auto-mu fill value refers to.
    /// 0 = derive from the auto_mu_nuclide preset. With window energies the
    /// CTAC factor table is scaled per window as att^(mu_w(E_k)/mu_w(E_map)).
    double ctacMapEnergyKeV = 0.0;
    /// Photopeak energy (keV) of each energy window (one per matrix). Empty =
    /// no scaling (the table is used at the map energy for every window).
    std::vector<double> ctacWindowEnergiesKeV;

    // ========== Attenuation & CT guidance ==========
    std::string attMapPath;               ///< Attenuation map file
    std::string regCtPath;                ///< Registered HU CT for TV/RDP weights
    std::string bProjPath;                ///< Background projection
    std::vector<std::string> bProjPaths;  ///< Multi-bed background
    std::string ctacFactorCacheDir;       ///< Directory for cached CTAC attenuation factor tables

    // ========== CSR Cache ==========
    std::string compactDenomCachePath;     ///< Optional cached compact denominator MHD
    bool compactDenomCachePathExplicit = false; ///< True if -compactDenomCache was used
    std::string compactDenomCacheDir;      ///< Directory for auto_<manifest-hash> denominator caches
    std::string chunkCSRDir;               ///< Directory with metadata.bin + chunk_*.csrb
    bool chunkCSRDirExplicit = false;      ///< True if -chunkCSRDir was used explicitly

    // ========== Joint dual-window reconstruction (MB-JR) ==========
    // A second energy window with its own system matrix and projections shares
    // the activity estimate; both windows enter one Poisson likelihood
    // (Lu-177: 113 keV + 208 keV). CSRQ matrices only.
    std::string chunkCSRDir2;              ///< Second window matrix directory
    std::string projPath2;                 ///< Second window single-bed projection
    std::vector<std::string> projPaths2;   ///< Second window multi-bed projections
    std::string bProjPath2;                ///< Second window background projection
    std::vector<std::string> bProjPaths2;  ///< Second window multi-bed background
    /// Window weights w_k (window mean = w_k * A_k x + b_k): emission
    /// intensity x window efficiency unless the matrices already carry it.
    /// Required whenever chunkCSRDir2 is set; empty means single window.
    std::vector<double> energyWindowWeights;
    /// CSRQ device residency: "auto" keeps a matrix resident when it fits next
    /// to the reserve and streams it otherwise; "streamed" always streams
    /// (two device chunk slots per matrix), e.g. when the GPU is shared.
    std::string csrqResidency = "auto";

    bool dualWindow() const { return !chunkCSRDir2.empty(); }

    // ========== Miscellaneous ==========
    std::vector<float> voxelSizeMM = {1.0f, 1.0f, 1.0f};
    int verbosity = 1;
    bool projectionResidual = false; ///< Write final compact projection residual CSV

    // ========== Optimization Algorithm ==========
    enum class Algorithm { MLEM, BSREM };
    Algorithm algorithm = Algorithm::MLEM;

    float bsrem_alpha0 = 1.0f;
    float bsrem_gamma = 0.05f;
    bool bsremUseCurvDenom = false;          ///< Enable BSREM denominator D = S + beta*Dreg
    bool bsremFreezeCurv = true;             ///< Freeze curvature map after warmup iterations
    int bsremCurvWarmupIters = 2;            ///< Iterations used to refresh curvature before freeze
    double bsremCurvEps = 1e-6;              ///< Stability epsilon for Dreg surrogate |grad|/max(f,eps)
    bool bsremUseKappaMap = false;           ///< Enable BSREM kappa map: beta_j = beta * kappa_j
    double bsremKappaEps = 1e-6;             ///< Stability epsilon for kappa map (1/sqrt(S+eps))
    double bsremKappaClipMin = 0.1;          ///< Lower clipping bound after kappa normalization
    double bsremKappaClipMax = 10.0;         ///< Upper clipping bound after kappa normalization
    bool bsremSoftMask = false;              ///< Enable soft-mask penalty boosting via kappa map
    double bsremSoftMaskLambdaOut = 100.0;   ///< Penalty boost outside support: 1+lambda*(1-M)^2
    bool bsremMlemEquivalenceMode = false; ///< beta=0, alpha=1, no clip; host-path MLEM match
    bool bsremDiagnostics = false;         ///< Write iteration_metrics.csv
    int bsremDiagInterval = 5;             ///< After the first 10 iters, record every N
    /// Freeze (zero) voxels with s_j < rel * max(s). 0 disables (default,
    /// preserves frozen baselines). Contains the multi-bed edge-slice blowup
    /// seen on unpenalized runs (C.3) and is a PNCG precondition.
    float bsremSensFloorRel = 0.0f;

    // ========== Body Mask ==========
    float maskFromAttThreshold = std::numeric_limits<float>::quiet_NaN();

    // ========== Multi-Bed ==========
    int bedCount = 1;
    std::vector<int> bedOverlaps;  ///< Fixed Bed1..Bed5 overlap table before setup; actual joints after setup
    std::vector<int> outputBedOverlaps; ///< Physical overlaps retained for output hard-cut
    int detZEdgeCensor = 0;    ///< Censor k detZ edge rows per side in compact numerator+denominator (0 = off)
    std::string detZRowWeightsFile; ///< Detector efficiency weights applied to compact sensitivity only (empty = off)
    bool normFactorFileExplicit = false;
    std::string normFactorFile; ///< Projection correction c; reciprocal 1/c is applied to compact sensitivity
    bool zBedFlip = false;         ///< Reverse local bed Z; compact multi-bed applies it in joint mapping
    double bedStepMm = 0.0;        ///< Bed step in mm; 0 = PSC default, or unknown for matrix-grid output.
    bool singleBedStepAligned = true; ///< Single-bed crop follows step+blend;
                                      ///< false keeps legacy 72 layers
    bool zOutFlip = false;         ///< Flip the final output volume in Z
    bool xOutFlip = false;         ///< Flip the final output volume in X
    double xyOutRotationDeg = 0.0; ///< Output-only rotation: positive maps +X toward +Y
    bool seamXyAlign = false;      ///< Align adjacent beds in XY from overlap NCC; default off

    // ========== Output Z-axis Cropping ==========
    // For zeroing final output edge slices after output flips (Z size unchanged)
    int cropEndStart = 0; // Number of top slices to zero (z=0)
    int cropEndEnd = 0;   // Number of end slices to zero (z=Nz-1)

    // ========== Helper Methods ==========
    
    /**
     * @brief Get voxel spacing as a 3-element array.
     */
    std::array<double, 3> getVoxelSpacing() const {
        return {static_cast<double>(Vx), static_cast<double>(Vy), static_cast<double>(Vz)};
    }

    /**
     * @brief Check if multi-bed reconstruction is configured.
     */
    bool isMultiBed() const {
        return !projPaths.empty() && projPaths.size() > 1;
    }

    /**
     * @brief Get effective bed count.
     */
    int getEffectiveBedCount() const {
        if (!projPaths.empty())
            return static_cast<int>(projPaths.size());
        return bedCount;
    }

    /**
     * @brief Get algorithm name as string.
     */
    const char* getAlgorithmName() const {
        switch (algorithm) {
            case Algorithm::MLEM: return "MLEM";
            case Algorithm::BSREM: return "BSREM";
            default: return "Unknown";
        }
    }
};

} // namespace cli
} // namespace spect

namespace spect::main {
using ReconArgs = cli::ReconArgs;
}

