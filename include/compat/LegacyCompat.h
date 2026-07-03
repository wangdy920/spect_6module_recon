/**
 * @file LegacyCompat.h
 * @brief Compatibility layer for migrating from original_source_backup
 * 
 * This header provides backward-compatible access to all modular functionality,
 * allowing the original SPECT-recon.cpp to compile against the new modular structure.
 * 
 * Usage: Replace includes in SPECT-recon.cpp with just:
 *   #include "compat/LegacyCompat.h"
 */

#pragma once

// ================= Standard Library =================
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// ================= Third-Party =================
#include <matio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ================= Modular Headers =================
// Core types and utilities
#include "core/Types.h"

// IO functionality
#include "io/CSRBuilder.h"
#include "io/MatFileReader.h"
#include "io/MHDWriter.h"
#include "io/VolumeLoader.h"
#include "io/ProjectionLoader.h"

// Filters
#include "filters/PSFConfig.h"
#include "filters/Convolution.h"
#include "filters/ButterworthFilter.h"

// Regularizers
#include "regularizer/RegConfig.h"
#include "regularizer/RegularizerCompat.h"

// Core utilities
#include "core/Masks.h"

// App - ReconOpts and helpers
#include "app/ReconConfig.h"
#include "app/ProjectionOps.h"
#include "app/ReconCompat.h"

// ================= Backward Compatibility Using Statements =================

// Import types to global namespace for backward compatibility
using spect::io::MatArray;
using spect::io::readMat;
using spect::io::printMatInfo;
using spect::io::CSR;
using spect::io::CSRView;
using spect::io::CSRHeader;
using spect::io::CSRMap;
using spect::io::CSRValidationLevel;
using spect::io::validateCSR;
using spect::io::validateCSRView;
using spect::io::loadCSRBinary;
using spect::io::saveCSRBinary;
using spect::io::saveMHDRAW;
using spect::io::loadVolume;
using spect::io::buildCSR_from_mat;
using spect::io::buildCSR_from_mat_stream;

using spect::filters::PSFConf;
using spect::filters::PSFMode;
using spect::filters::PSFKernels;
using spect::filters::ButterworthConf;
using spect::filters::BWType;
using spect::filters::Spacing3;
using spect::filters::make_psf_kernels;
using spect::filters::makeGaussian1D;
using spect::filters::conv1d_x_mirror;
using spect::filters::conv1d_y_mirror;
using spect::filters::conv1d_z_mirror;
using spect::filters::conv3_separable_norm;
using spect::filters::idx3;

using spect::RegType;
using spect::RegConf;
using spect::stringToRegType;
using spect::regTypeToString;
using spect::regularizer::compute_reg_gradient;
using spect::regularizer::compute_reg_energy;
using spect::regularizer::osl_update_inplace;

using spect::core::makeCylMask_mm;
using spect::core::makeSphereMask_mm;
using spect::core::makeMaskFromThreshold;
using spect::core::applyMaskInplace;

// App types
using spect::ReconOpts;
using spect::should_use_threadlocal_accum;
using spect::make_simple_subsets;
using spect::needs_psf_blur;
using spect::computeNormAT;
using spect::computeNormAT_Atten;

// Filter functions for backward compatibility
using spect::filters::conv3d_separable_inplace;
using spect::filters::conv2d_separable_inplace;
using spect::filters::apply_butterworth_inplace_fftw;

// ================= ConfigParser Class =================
// (Copied from original_source_backup/SPECT-recon.h for compatibility)

class ConfigParser {
public:
    std::map<std::string, std::map<std::string, std::string>> sections;

    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open())
            return false;

        std::string line, currentSection;
        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            if (!line.empty()) {
                auto lastNonWs = line.find_last_not_of(" \t\r\n");
                if (lastNonWs != std::string::npos)
                    line.erase(lastNonWs + 1);
            }

            if (line.empty() || line[0] == ';' || line[0] == '#')
                continue;

            if (line[0] == '[' && line.back() == ']') {
                currentSection = line.substr(1, line.length() - 2);
            } else {
                size_t eq = line.find('=');
                if (eq != std::string::npos) {
                    std::string key = line.substr(0, eq);
                    std::string val = line.substr(eq + 1);
                    // Trim key/val
                    key.erase(0, key.find_first_not_of(" \t"));
                    auto lastK = key.find_last_not_of(" \t");
                    if (lastK != std::string::npos) key.erase(lastK + 1);
                    val.erase(0, val.find_first_not_of(" \t"));
                    auto lastV = val.find_last_not_of(" \t");
                    if (lastV != std::string::npos) val.erase(lastV + 1);

                    if (!currentSection.empty()) {
                        sections[currentSection][key] = val;
                    }
                }
            }
        }
        return true;
    }

    std::string getString(const std::string& sec, const std::string& key,
                          const std::string& def = "") const {
        auto sit = sections.find(sec);
        if (sit != sections.end()) {
            auto kit = sit->second.find(key);
            if (kit != sit->second.end())
                return kit->second;
        }
        return def;
    }

    int getInt(const std::string& sec, const std::string& key, int def) const {
        std::string val = getString(sec, key);
        if (val.empty())
            return def;
        try {
            return std::stoi(val);
        } catch (...) {
            return def;
        }
    }

    double getDouble(const std::string& sec, const std::string& key,
                     double def) const {
        std::string val = getString(sec, key);
        if (val.empty())
            return def;
        try {
            return std::stod(val);
        } catch (...) {
            return def;
        }
    }

    bool getBool(const std::string& sec, const std::string& key, bool def) const {
        std::string val = getString(sec, key);
        if (val.empty())
            return def;
        if (val == "true" || val == "True" || val == "1" || val == "on")
            return true;
        if (val == "false" || val == "False" || val == "0" || val == "off")
            return false;
        return def;
    }
};

// ================= ReconArgs Structure =================
// (Copied from original_source_backup/SPECT-recon.h for compatibility)

#ifndef SPECT_LEGACYCOMPAT_NO_RECONARGS
struct ReconArgs {
    // 基本参数
    int Nx = 0, Ny = 0, Nz = 0;
    bool dimExplicit = false;
    float Vx = 1, Vy = 1, Vz = 1;
    double dx = 1.0, dy = 1.0, dz = 1.0;
    int detectorChannels = 40;
    int detNz = 0, detNx = 0;
    int numAngles = 72;
    int numIters = 50, saveInterval = 5;
    int numSubsets = 1;
    int numThreads = 8;

    // 输入输出
    std::string projPath;
    std::vector<std::string> projPaths;
    std::string sysIndexPath;
    std::string sysWeightPath;
    std::string outputDir = "output";
    std::string outputName;
    std::string configPath = "spect_config.ini";
    bool configExplicit = false;
    bool configFound = false;
    bool configLoaded = false;

    // 滤波选项
    bool useSmooth = false;
    float smoothSigma = 1.0f;
    int smoothKernel = 3;
    bool useButter = false;
    float butterCutoff = 0.35;
    int butterOrder = 2;
    bool usePSF = false;
    PSFConf psf;
    bool psfProj = false;
    bool psfModeExplicit = false;

    // 正则选项
    bool useReg = false;
    RegConf reg;

    // 圆柱形掩模
    bool useCylindricalMask = false;
    float cylinderDiameterMM = 30.0f;
    float cylinderHeightMM = 30.0f;
    double cylCenterX = std::numeric_limits<double>::quiet_NaN();
    double cylCenterY = std::numeric_limits<double>::quiet_NaN();
    double cylCenterZ = std::numeric_limits<double>::quiet_NaN();

    // 系统矩阵构建优化
    double weightPruneEps = 0.0;
    bool Astream = false;
    bool ApruneMask = false;
    bool doCTAC = false;

    std::string attMapPath;
    std::string priorPath;
    std::string bProjPath;
    std::vector<std::string> bProjPaths;
    std::string saveCSRPath;
    std::string loadCSRPath;
    bool mapCSR = false;
    std::vector<float> voxelSizeMM = {1.0f, 1.0f, 1.0f};
    int josephPasses = 0;
    int verbosity = 1;

    // Optimization Algorithm
    enum class Algorithm { MLEM, BSREM };
    Algorithm algorithm = Algorithm::MLEM;
    float bsrem_alpha0 = 1.0f;
    float bsrem_gamma = 0.05f;

    // Body Mask from Attenuation Map
    float maskFromAttThreshold = std::numeric_limits<float>::quiet_NaN();

    // Multi-bed reconstruction
    int bedCount = 1;
    std::vector<int> bedOverlaps;

    // Output Z-axis cropping (for multi-bed edge trimming)
    int cropEndStart = 0; // Number of slices to crop from start (z=0)
    int cropEndEnd = 0;   // Number of slices to crop from end (z=Nz-1)
};
#endif // SPECT_LEGACYCOMPAT_NO_RECONARGS
