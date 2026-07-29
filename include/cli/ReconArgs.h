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
    std::string badDetectorFile;   ///< Global numbers or hardware IDs; empty list disables masking
    bool badDetectorFileExplicit = false; ///< CLI path is resolved from cwd, not config dir
    int numIters = 50;            ///< Total iterations
    int saveInterval = 5;         ///< Save every N iterations
    int numSubsets = 1;           ///< Subsets (1 = MLEM)
    int numThreads = 8;           ///< Thread count
    std::string cudaDeviceValue;  ///< Requested CUDA device id from env/CLI
    bool cudaDeviceRequested = false; ///< True if CUDA device selection was requested
    bool cudaDeviceFromCli = false;   ///< True if -cudaDevice overrides env

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
    bool projectionResidual = false; ///< Write final compact projection residual CSV

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
    std::vector<int> outputBedOverlaps; ///< Physical overlaps retained for output hard-cut
    bool bedOverlapInRecon = false; ///< Use bed overlap in joint reconstruction geometry
    std::string bedCropMethod = "sigmoid"; ///< direct | sigmoid (output-only)
    bool bedSeamPostprocess = true; ///< Request seam postprocess; active only with multi-bed output overlap stitch
    int detZEdgeCensor = 0;    ///< Censor k detZ edge rows per side in compact numerator+denominator (0 = off)
    std::string detZRowWeightsFile; ///< Detector efficiency weights applied to compact sensitivity only (empty = off)
    std::string normFactorFile; ///< Projection correction c; reciprocal 1/c is applied to compact sensitivity
    std::string bedObjectProfile = "auto"; ///< auto | mouse | phantom | epguan | zimu | derenzo
    bool bedDerenzoPostprocess = true; ///< Enable explicit Derenzo axial continuity processing
    double bedDerenzoAxialStrength = 0.30; ///< Positive-only high-confidence rod fill, [0,1]
    double bedPhantomCloseStrength = 1.0; ///< Phantom contour-interpolation blend, [0,1]
    double bedEpguanCloseStrength = 1.0; ///< EP tube contour-interpolation blend, [0,1]
    bool bedEpguanWallMaxPreserve = false; ///< Keep brighter original EP wall at seams
    double bedEpguanSeamEdgeStrength = 0.60; ///< C4 blend at seam-window edges, [0,1]
    int bedEpguanSeamFeatherSlices = 4; ///< Slices from edge to full C4 blend
    bool bedEpguanBackgroundFilter = true; ///< Attenuate signal outside EP tube support
    double bedEpguanBackgroundRetain = 0.08; ///< Far-background retained fraction, [0,1]
    double bedEpguanSupportThreshold = 0.06; ///< Smoothed projection threshold fraction, (0,1]
    double bedEpguanSupportMarginMm = 1.5; ///< Fully protected margin and feather width in mm
    double bedZimuCloseStrength = 0.5; ///< Zimu topology-locked smoothing blend, [0,1]
    double bedMouseSeamStrength = 0.0; ///< Bilateral confidence fusion blend, [0,1]
    int bedSeamHalfWidth = 8; ///< Phantom seam half-width; zimu is capped at 4
    double fovRadiusMm = 0.0;       ///< Final-output circular XY FOV radius; 0 disables
    double analysisFovRadiusMm = 0.0; ///< Classifier/support FOV; 0 follows fovRadiusMm
    bool zBedFlip = false;         ///< Reverse local bed Z; compact multi-bed applies it in joint mapping
    bool zOutFlip = false;         ///< Flip the final output volume in Z
    bool xOutFlip = false;         ///< Flip the final output volume in X

    // ========== Symmetry expansion ==========
    // Reverses the 4-fold XY 90-degree rotation sense in compact-symmetric
    // (Sym90Z/Sym45Z) image-side voxel mapping. Off = use matrix's baked
    // handedness (default). On = swap CCW<->CW branches to compensate for a
    // generator whose voxel_rotation_sign disagrees with gantry direction.
    bool symReverseXY = false;     ///< Reverse Sym90Z/Sym45Z XY rotation handedness

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
            case Algorithm::ADMM: return "ADMM";
            default: return "Unknown";
        }
    }
};

} // namespace cli
} // namespace spect

// Backward compatibility alias for existing code
using ReconArgs = spect::cli::ReconArgs;


