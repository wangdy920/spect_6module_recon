/**
 * @file RegConfig.h
 * @brief Regularization configuration types
 * 
 * Migrated from original_source_backup/Regularizers.h
 */

#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace spect {

/**
 * @brief Regularization type enumeration
 */
enum class RegType {
    None,
    TVIsotropic,
    TVHuber,
    Tikhonov,
    MRP,
    NLM,
    Bowsher,
    PLS,
    RDP,
    LogCosh,
    Unknown
};

/**
 * @brief Convert string to RegType
 */
inline RegType stringToRegType(const std::string& str) {
    if (str == "None" || str == "none" || str.empty()) return RegType::None;
    if (str == "TVIsotropic" || str == "tv" || str == "TV") return RegType::TVIsotropic;
    if (str == "TVHuber" || str == "tvhuber" || str == "Huber") return RegType::TVHuber;
    if (str == "Tikhonov" || str == "tikhonov" || str == "L2") return RegType::Tikhonov;
    if (str == "MRP" || str == "mrp") return RegType::MRP;
    if (str == "NLM" || str == "nlm") return RegType::NLM;
    if (str == "Bowsher" || str == "bowsher") return RegType::Bowsher;
    if (str == "PLS" || str == "pls") return RegType::PLS;
    if (str == "RDP" || str == "rdp") return RegType::RDP;
    if (str == "LogCosh" || str == "logcosh") return RegType::LogCosh;
    return RegType::Unknown;
}

/**
 * @brief Convert RegType to string
 */
inline std::string regTypeToString(RegType type) {
    switch (type) {
        case RegType::None: return "None";
        case RegType::TVIsotropic: return "TVIsotropic";
        case RegType::TVHuber: return "TVHuber";
        case RegType::Tikhonov: return "Tikhonov";
        case RegType::MRP: return "MRP";
        case RegType::NLM: return "NLM";
        case RegType::Bowsher: return "Bowsher";
        case RegType::PLS: return "PLS";
        case RegType::RDP: return "RDP";
        case RegType::LogCosh: return "LogCosh";
        default: return "Unknown";
    }
}

/**
 * @brief Regularization configuration structure
 */
struct RegConf {
    RegType type = RegType::None;
    double beta = 0.0;                ///< Regularization weight β
    double tv_eps = 1e-6;             ///< TV |∇f| stability term
    double mrp_eps = 1e-8;            ///< MRP sqrt stability term
    bool skip_outside_mask = true;    ///< Skip voxels outside mask
    
    // NLM parameters
    double nlm_h = 0.1;
    int nlm_search = 3;               ///< Search radius (3 => 7x7x7)
    int nlm_patch = 1;                ///< Patch radius (1 => 3x3x3)

    // Anatomical Priors
    const std::vector<double>* anatomical_image = nullptr;
    
    // Bowsher
    int bowsher_k = 13;               ///< Neighbors to select (out of 26)
    
    // PLS
    double pls_eta = 1.0;             ///< Edge preservation parameter
    
    // TV / Huber / RDP / LogCosh
    double delta = 0.01;              ///< Huber delta / RDP epsilon / LogCosh delta
    double gamma = 0.0;               ///< RDP gamma (Nuyts 2002)

    // QGGMRF / Other
    double sigma = 1.0;
    double power = 2.0;

    // Bowsher Pre-computed neighbors
    mutable std::vector<int> bowsher_indices;
    mutable bool bowsher_precomputed = false;

    // PLS Pre-computed normals
    mutable std::vector<float> pls_nx, pls_ny, pls_nz;
    mutable bool pls_precomputed = false;

    // Kernel-based regularization
    int kernel = 3;

    // Update Schedule
    int numIter = 1;
    int numSubsets = 1;
    int updateFreq = 1;
    int updateStart = 0;
    int updateEnd = 9999;
    int updateInterval = 1;
    enum class UpdateType { Subset, Iteration };
    UpdateType updateType = UpdateType::Subset;
};

} // namespace spect
