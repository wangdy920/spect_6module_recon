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
    int numIters = 50;            ///< Total iterations
    int saveInterval = 5;         ///< Save every N iterations
    int numSubsets = 1;           ///< Subsets (1 = MLEM)
    int numThreads = 8;           ///< Thread count
    std::string sparseBackend = "auto"; ///< Sparse backend preference: auto/cpu/ref/cuda

    // ========== Input/Output Paths ==========
    std::string projPath;                ///< Single-bed projection file
    std::vector<std::string> projPaths;  ///< Multi-bed projection list
    std::string sysIndexPath;            ///< System matrix index file
    std::string sysWeightPath;           ///< System matrix weight file
    std::string outputDir = "output";    ///< Output directory
    std::string outputName;              ///< Output filename base

    // ========== Config Provenance ==========
    std::string configPath = "spect_config.ini";
    bool configExplicit = false;  ///< -config was used explicitly
    bool configFound = false;     ///< Config file exists
    bool configLoaded = false;    ///< Config file loaded successfully
    std::string algoConfPath;     ///< Algorithm config file path (empty = auto-detect)
    bool dumpConf = false;        ///< Print merged config and exit

    // ========== Filtering Options ==========
    bool useSmooth = false;
    float smoothSigma = 1.0f;
    int smoothKernel = 3;
    bool useButter = false;
    std::string butterBackend = "auto";
    float butterCutoff = 0.35f;
    int butterOrder = 2;
    bool usePSF = false;
    PSFConf psf;                  ///< PSF configuration
    bool psfProj = false;         ///< Per-angle 2D PSF in projection domain
    bool psfModeExplicit = false; ///< -psfMode was used explicitly

    // ========== Regularization ==========
    bool useReg = false;
    RegConf reg;  ///< Regularization configuration

    // ========== Cylindrical Mask ==========
    bool useCylindricalMask = false;
    float cylinderDiameterMM = 30.0f;
    float cylinderHeightMM = 30.0f;
    double cylCenterX = std::numeric_limits<double>::quiet_NaN();
    double cylCenterY = std::numeric_limits<double>::quiet_NaN();
    double cylCenterZ = std::numeric_limits<double>::quiet_NaN();

    // ========== System Matrix Options ==========
    double weightPruneEps = 0.0;  ///< Prune threshold (0 = off)
    bool Astream = false;         ///< Stream build from disk
    bool ApruneMask = false;      ///< Prune columns outside mask
    bool doCTAC = false;          ///< Iterative CTAC
    std::string ctacModel = "line-integral"; ///< line-integral | per-voxel (legacy approx)
    int ctacRotSign = 1;          ///< Voxel rotation sign matching sysmat generator (+1/-1)
    std::string ctacAutoMuFrom;   ///< Early-iteration recon volume for auto water mu-map
    double ctacAutoMuThresh = 0.05;  ///< Relative threshold vs P99.9 activity
    double ctacAutoMuValue = 0.0152; ///< Water mu (mm^-1) used to fill the body

    // ========== Attenuation & Prior ==========
    std::string attMapPath;               ///< Attenuation map file
    std::string priorPath;                ///< Anatomical prior image
    std::string bProjPath;                ///< Background projection
    std::vector<std::string> bProjPaths;  ///< Multi-bed background

    // ========== CSR Cache ==========
    std::string saveCSRPath;  ///< Save CSR to binary
    std::string loadCSRPath;  ///< Load CSR from binary
    bool loadCSRExplicit = false; ///< True if -loadCSR was used explicitly
    bool mapCSR = false;      ///< Memory-map CSR
    std::string matrixLayout = "standard"; ///< standard or compact-symmetric
    bool matrixLayoutExplicit = false;     ///< True if -matrixLayout was used explicitly
    std::string compactMetaPath;           ///< Optional compact metadata sidecar
    std::string compactDenomCachePath;     ///< Optional cached compact denominator MHD
    std::string chunkCSRDir;               ///< Directory with metadata.bin + chunk_*.csrb
    bool chunkCSRDirExplicit = false;      ///< True if -chunkCSRDir was used explicitly

    // ========== Miscellaneous ==========
    std::vector<float> voxelSizeMM = {1.0f, 1.0f, 1.0f};
    int josephPasses = 0;
    int verbosity = 1;

    // ========== Optimization Algorithm ==========
    enum class Algorithm { MLEM, BSREM, ADMM };
    Algorithm algorithm = Algorithm::MLEM;

    // ========== ADMM Parameters ==========
    double admmRho = 1.0;  ///< ADMM augmented Lagrangian penalty ρ
    int admmTvInner = 20;  ///< Chambolle-Pock inner iterations for TV proximal
    bool admmAdaptive = true; ///< Adaptive ρ adjustment (Boyd 2011 / CASToR)
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
    int bsremSoftMaskBandVox = 2;            ///< Soft-mask transition width in voxels
    double bsremSoftMaskLambdaOut = 100.0;   ///< Penalty boost outside support: 1+lambda*(1-M)^2

    // ========== Body Mask ==========
    float maskFromAttThreshold = std::numeric_limits<float>::quiet_NaN();

    // ========== Multi-Bed ==========
    int bedCount = 1;
    std::vector<int> bedOverlaps;  ///< Fixed Bed1..Bed5 overlap table before setup; actual joints after setup
    bool zBedFlip = true;          ///< Flip each dfList bed segment in Z before output
    bool zOutFlip = false;         ///< Flip the final output volume in Z

    // ========== Symmetry expansion ==========
    // Reverses the 4-fold XY 90-degree rotation sense in compact-symmetric
    // (Sym90Z/Sym45Z) image-side voxel mapping. Off = use matrix's baked
    // handedness (default). On = swap CCW<->CW branches to compensate for a
    // generator whose voxel_rotation_sign disagrees with gantry direction.
    bool symReverseXY = false;     ///< Reverse Sym90Z/Sym45Z XY rotation handedness

    // ========== Output Z-axis Cropping ==========
    // For final output edge trimming (crop from top/end of Z after flips)
    int cropEndStart = 0; // Number of slices to crop from top (z=0)
    int cropEndEnd = 0;   // Number of slices to crop from end (z=Nz-1)

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
            case Algorithm::ADMM: return "ADMM";
            default: return "Unknown";
        }
    }
};

} // namespace cli
} // namespace spect

// Backward compatibility alias for existing code
using ReconArgs = spect::cli::ReconArgs;


