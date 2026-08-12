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
    int saveInterval = 5;         ///< Save every N iterations
    int numThreads = 8;           ///< Thread count
    std::string cudaDeviceValue;  ///< Requested CUDA device id from env/CLI
    bool cudaDeviceRequested = false; ///< True if CUDA device selection was requested
    bool cudaDeviceFromCli = false;   ///< True if -cudaDevice overrides env

    // ========== Input/Output Paths ==========
    std::string projPath;                ///< Single-bed projection file
    std::vector<std::string> projPaths;  ///< Multi-bed projection list
    bool mergeTimeSegments = false;      ///< Sum matching BedN data across sibling timestamp/SPECT directories
    bool outputFiltersOverwritePrimary = false; ///< Write output-filter chain to the unsuffixed volume
    std::vector<std::vector<std::string>> projectionSourceGroups; ///< Actual per-bed sources recorded in the manifest
    std::string outputDir = "output";    ///< Output directory
    std::string outputName;              ///< Output filename base

    // ========== Config Provenance ==========
    std::string configPath = "spect_config.ini";
    bool configExplicit = false;  ///< -config was used explicitly
    bool configFound = false;     ///< Config file exists
    bool configLoaded = false;    ///< Config file loaded successfully
    bool reconSectionsOverrideCli = false; ///< Global INI owns Reg/PSF/filter/postprocess fields
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

    // ========== Attenuation & CT guidance ==========
    std::string attMapPath;               ///< Attenuation map file
    std::string regCtPath;                ///< Registered CT/mu-map for TV/RDP weights
    std::string bProjPath;                ///< Background projection
    std::vector<std::string> bProjPaths;  ///< Multi-bed background
    std::string ctacFactorCacheDir;       ///< Directory for cached CTAC attenuation factor tables

    // ========== CSR Cache ==========
    std::string compactDenomCachePath;     ///< Optional cached compact denominator MHD
    std::string chunkCSRDir;               ///< Directory with metadata.bin + chunk_*.csrb
    bool chunkCSRDirExplicit = false;      ///< True if -chunkCSRDir was used explicitly

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

    // ========== Body Mask ==========
    float maskFromAttThreshold = std::numeric_limits<float>::quiet_NaN();

    // ========== Multi-Bed ==========
    int bedCount = 1;
    std::vector<int> bedOverlaps;  ///< Fixed Bed1..Bed5 overlap table before setup; actual joints after setup
    std::vector<int> outputBedOverlaps; ///< Physical overlaps retained for output hard-cut
    int detZEdgeCensor = 0;    ///< Censor k detZ edge rows per side in compact numerator+denominator (0 = off)
    std::string detZRowWeightsFile; ///< Detector efficiency weights applied to compact sensitivity only (empty = off)
    std::string normFactorFile; ///< Projection correction c; reciprocal 1/c is applied to compact sensitivity
    bool zBedFlip = false;         ///< Reverse local bed Z; compact multi-bed applies it in joint mapping
    double bedStepMm = 0.0;        ///< Scanner bed step in mm (0 = use kMultiBedStepMm default 16.8)
    bool zOutFlip = false;         ///< Flip the final output volume in Z
    bool xOutFlip = false;         ///< Flip the final output volume in X

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


