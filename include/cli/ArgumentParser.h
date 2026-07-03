// ArgumentParser.h - Command-line argument parser for SPECT reconstruction
// Extracted from SPECT-recon.cpp for modular CLI architecture
#pragma once

#include "ConfigParser.h"
#include "ReconArgs.h"
#include "StringUtils.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace spect {
namespace cli {

/**
 * @brief Command-line argument parser for SPECT reconstruction.
 * 
 * Handles:
 * - INI config file loading (exe-local config first, cwd fallback)
 * - Command-line argument parsing with validation
 * - Type-safe handler dispatch
 */
class ArgumentParser {
public:
    using Handler = std::function<bool(int&, int, char**, ReconArgs&)>;

    ArgumentParser() { registerHandlers(); }

    /**
     * @brief Parse command-line arguments.
     * @param argc Argument count.
     * @param argv Argument vector.
     * @param args Output ReconArgs structure.
     * @return true if parsing succeeded, false on error.
     */
    bool parse(int argc, char** argv, ReconArgs& args);

    /**
     * @brief Print usage information to stdout.
     */
    static void printUsage();

private:
    std::map<std::string, Handler> handlers_;

    void registerHandlers();

    /**
     * @brief Get a required argument value, handling Windows paths.
     */
    static std::string getRequiredArg(int& i, int argc, char** argv,
                                       const std::string& argName);

    /**
     * @brief Apply configuration file values to ReconArgs.
     */
    static void applyConfig(ReconArgs& args, const ConfigParser& cfg);

    /**
     * @brief Apply algorithm-specific config file values to ReconArgs.
     * Only affects algorithm-related sections ([Algorithm], [Regularization],
     * [BSREM], [ADMM], [BackendHint]). Does NOT override geometry/IO/detector.
     */
    static void applyAlgoConf(ReconArgs& args, const std::string& path);

    /**
     * @brief Print merged configuration to stdout (for -dumpConf).
     */
    static void dumpMergedConfig(const ReconArgs& args);

    /**
     * @brief Parse boolean from optional next argument.
     */
    static bool parseOnOffOptional(int& i, int argc, char** argv, bool& out);

    /**
     * @brief Parse reconstruction dimensions (x,y,z or single value).
     */
    static bool parseDimensions(const std::string& value, ReconArgs& args);

    /**
     * @brief Parse iteration specification (iter[:saveInterval]).
     */
    static bool parseIterations(const std::string& value, ReconArgs& args);

    /**
     * @brief Parse PSF parameters from legacy format.
     */
    static bool parsePSFParams(const std::string& value, ReconArgs& args);

    /**
     * @brief Parse output path into directory and filename.
     */
    static bool parseOutputPath(const std::string& path, ReconArgs& args);

    /**
     * @brief Validate all arguments after parsing.
     */
    static bool validateArgs(const ReconArgs& args);

    /**
     * @brief Resolve the directory that contains the running executable.
     */
    static std::filesystem::path executableDir(char** argv);

    /**
     * @brief Resolve default global config path, preferring exe-local config.
     */
    static std::string resolveDefaultConfigPath(char** argv);

    /**
     * @brief Resolve algorithm config path relative to global/exe config dirs.
     */
    static std::string resolveAlgoConfigPath(const std::string& algName,
                                             const std::string& configPath,
                                             char** argv);

    /**
     * @brief Apply exe-local matrix discovery and resolve default chunk paths.
     */
    static bool applyMatrixAutoDiscovery(ReconArgs& args, char** argv,
                                         std::string& error);

    /**
     * @brief Resolve compact chunk directory paths relative to the executable.
     */
    static std::filesystem::path resolveChunkCSRDirPath(
        const std::string& value, const std::filesystem::path& exeDir,
        bool explicitCliPath);

    /**
     * @brief Find compact matrix directories under exe-local matrix/.
     */
    static std::vector<std::filesystem::path>
    findExeLocalMatrixDirs(const std::filesystem::path& exeDir);

    /**
     * @brief Sort BedN-style paths by numeric bed order and normalize slashes.
     */
    static std::vector<std::string>
    sortPathsByBedOrder(std::vector<std::string> paths);
};

// ============================================================================
// Inline Implementation
// ============================================================================

inline std::filesystem::path ArgumentParser::executableDir(char** argv) {
#ifdef _WIN32
    std::array<char, 32768> buffer{};
    const DWORD copied = GetModuleFileNameA(
        nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (copied > 0 && copied < buffer.size()) {
        return std::filesystem::path(std::string(buffer.data(), copied))
            .parent_path();
    }
#endif
    if (argv && argv[0] && argv[0][0] != '\0') {
        try {
            return std::filesystem::absolute(std::filesystem::path(argv[0]))
                .parent_path();
        } catch (...) {
        }
    }
    return std::filesystem::current_path();
}

inline std::string ArgumentParser::resolveDefaultConfigPath(char** argv) {
    const std::filesystem::path exeDir = executableDir(argv);
    const std::vector<std::filesystem::path> candidates = {
        exeDir / "config" / "spect_config.ini",
        exeDir / "spect_config.ini",
        exeDir.parent_path() / "config" / "spect_config.ini",
        std::filesystem::path("config") / "spect_config.ini",
        std::filesystem::path("spect_config.ini"),
    };
    for (const auto& path : candidates) {
        if (!path.empty() && std::filesystem::exists(path))
            return path.string();
    }
    return candidates.front().string();
}

inline std::string
ArgumentParser::resolveAlgoConfigPath(const std::string& algName,
                                      const std::string& configPath,
                                      char** argv) {
    if (algName.empty())
        return {};
    const std::string filename = "algo_" + algName + ".conf";
    std::vector<std::filesystem::path> candidates;
    const std::filesystem::path configDir =
        std::filesystem::path(configPath).parent_path();
    if (!configDir.empty())
        candidates.push_back(configDir / filename);

    const std::filesystem::path exeDir = executableDir(argv);
    candidates.push_back(exeDir / "config" / filename);
    candidates.push_back(exeDir.parent_path() / "config" / filename);
    candidates.push_back(std::filesystem::path("config") / filename);

    for (const auto& path : candidates) {
        if (!path.empty() && std::filesystem::exists(path))
            return path.string();
    }
    return {};
}

inline std::filesystem::path ArgumentParser::resolveChunkCSRDirPath(
    const std::string& value, const std::filesystem::path& exeDir,
    bool explicitCliPath) {
    const std::filesystem::path raw(value);
    if (raw.empty())
        return raw;
    if (raw.is_absolute())
        return raw.lexically_normal();

    const std::filesystem::path exeCandidate =
        (exeDir / raw).lexically_normal();
    const std::filesystem::path cwdCandidate =
        std::filesystem::absolute(raw).lexically_normal();

    std::error_code ec;
    if (explicitCliPath) {
        if (std::filesystem::exists(cwdCandidate, ec))
            return cwdCandidate;
        ec.clear();
        if (std::filesystem::exists(exeCandidate, ec))
            return exeCandidate;
        return raw.lexically_normal();
    }

    if (std::filesystem::exists(exeCandidate, ec))
        return exeCandidate;
    ec.clear();
    if (std::filesystem::exists(cwdCandidate, ec))
        return cwdCandidate;
    return exeCandidate;
}

inline std::vector<std::filesystem::path>
ArgumentParser::findExeLocalMatrixDirs(const std::filesystem::path& exeDir) {
    std::vector<std::filesystem::path> out;
    const std::filesystem::path matrixRoot = exeDir / "matrix";
    std::error_code ec;
    if (!std::filesystem::is_directory(matrixRoot, ec))
        return out;

    for (const auto& entry :
         std::filesystem::directory_iterator(matrixRoot, ec)) {
        if (ec)
            break;
        if (!entry.is_directory(ec))
            continue;
        const std::filesystem::path candidate = entry.path();
        if (std::filesystem::exists(candidate / "metadata.bin", ec))
            out.push_back(candidate.lexically_normal());
        ec.clear();
    }
    std::sort(out.begin(), out.end());
    return out;
}

inline bool ArgumentParser::applyMatrixAutoDiscovery(ReconArgs& args,
                                                     char** argv,
                                                     std::string& error) {
    const std::filesystem::path exeDir = executableDir(argv);

    if (args.loadCSRExplicit && !args.chunkCSRDirExplicit)
        args.chunkCSRDir.clear();

    if (!args.chunkCSRDir.empty()) {
        args.chunkCSRDir = resolveChunkCSRDirPath(
            args.chunkCSRDir, exeDir, args.chunkCSRDirExplicit).string();
        if (args.matrixLayout == "standard" && !args.matrixLayoutExplicit)
            args.matrixLayout = "compact-symmetric";
        return true;
    }

    const bool hasStandardMatrixInput =
        !args.loadCSRPath.empty() || !args.sysIndexPath.empty() ||
        !args.sysWeightPath.empty();
    if (hasStandardMatrixInput ||
        (args.matrixLayoutExplicit && args.matrixLayout == "standard")) {
        return true;
    }

    const auto candidates = findExeLocalMatrixDirs(exeDir);
    if (candidates.empty())
        return true;
    if (candidates.size() == 1) {
        args.chunkCSRDir = candidates.front().string();
        if (!args.matrixLayoutExplicit)
            args.matrixLayout = "compact-symmetric";
        return true;
    }

    std::ostringstream oss;
    oss << "Multiple compact matrix directories found under "
        << (exeDir / "matrix").string()
        << ". Pass -chunkCSRDir or set [SystemMatrix] default_chunk_csr_dir.\n";
    oss << "Candidates:";
    for (const auto& candidate : candidates)
        oss << "\n  " << candidate.string();
    error = oss.str();
    return false;
}

inline std::vector<std::string>
ArgumentParser::sortPathsByBedOrder(std::vector<std::string> paths) {
    auto bedOrder = [](const std::string& path) {
        const std::string s = toLower(path);
        size_t pos = 0;
        while ((pos = s.find("bed", pos)) != std::string::npos) {
            size_t p = pos + 3;
            while (p < s.size() &&
                   (s[p] == '_' || s[p] == '-' || std::isspace(
                                                   static_cast<unsigned char>(s[p])))) {
                ++p;
            }
            if (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) {
                int value = 0;
                while (p < s.size() &&
                       std::isdigit(static_cast<unsigned char>(s[p]))) {
                    value = value * 10 + (s[p] - '0');
                    ++p;
                }
                return value;
            }
            pos += 3;
        }
        return std::numeric_limits<int>::max();
    };

    for (std::string& path : paths)
        path = normalizePath(path);
    std::stable_sort(paths.begin(), paths.end(),
                     [&](const std::string& a, const std::string& b) {
                         return bedOrder(a) < bedOrder(b);
                     });
    return paths;
}

inline bool ArgumentParser::parse(int argc, char** argv, ReconArgs& args) {
    // 0. Pre-scan for -config, -alg, -algoConf
    std::string configPath = resolveDefaultConfigPath(argv);
    std::string preScanAlg;
    std::string preScanAlgoConf;
    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];
        if (key == "-config" && i + 1 < argc) {
            configPath = argv[++i];
            args.configExplicit = true;
        } else if ((key == "-alg" || key == "-opti") && i + 1 < argc) {
            preScanAlg = toLower(std::string(argv[++i]));
        } else if (key == "-algoConf" && i + 1 < argc) {
            preScanAlgoConf = argv[++i];
        }
    }
    args.configPath = configPath;

    // 1. Load global config if exists
    ConfigParser cfg;
    if (std::filesystem::exists(configPath)) {
        args.configFound = true;
        if (cfg.load(configPath)) {
            args.configLoaded = true;
            applyConfig(args, cfg);
        }
    } else {
        args.configFound = false;
    }

    // 2. Load algorithm config (algo_conf layer)
    //    Priority: explicit -algoConf > auto-detect from -alg > auto-detect from global INI algorithm
    std::string algoConfPath;
    if (!preScanAlgoConf.empty()) {
        algoConfPath = preScanAlgoConf;
    } else {
        // Determine algorithm name for auto-detect
        std::string algName = preScanAlg;
        if (algName.empty()) {
            // Use algorithm set by global config
            switch (args.algorithm) {
                case ReconArgs::Algorithm::MLEM:     algName = "mlem"; break;
                case ReconArgs::Algorithm::BSREM:    algName = "bsrem"; break;
                case ReconArgs::Algorithm::ADMM:     algName = "admm"; break;
            }
        }
        if (!algName.empty()) {
            algoConfPath = resolveAlgoConfigPath(algName, configPath, argv);
        }
    }
    if (!algoConfPath.empty()) {
        args.algoConfPath = algoConfPath;
        applyAlgoConf(args, algoConfPath);
    }

    // 3. Process command-line arguments (highest priority override)
    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];
        auto it = handlers_.find(key);
        if (it != handlers_.end()) {
            if (!it->second(i, argc, argv, args))
                return false;
        } else {
            std::cerr << "Unknown argument: " << key << "\n";
            return false;
        }
    }

    // 4. Resolve matrix defaults after all config/CLI layers are known.
    std::string matrixDiscoveryError;
    if (!applyMatrixAutoDiscovery(args, argv, matrixDiscoveryError)) {
        std::cerr << "Error: " << matrixDiscoveryError << "\n";
        return false;
    }

    // 5. Handle -dumpConf: print merged config and exit
    if (args.dumpConf) {
        dumpMergedConfig(args);
        std::exit(0);
    }

    // 6. Validate
    if (!validateArgs(args))
        return false;
    if (!args.projPaths.empty()) {
        args.bedCount = std::max(1, static_cast<int>(args.projPaths.size()));
    }
    return true;
}

inline void ArgumentParser::registerHandlers() {
    handlers_.clear();

    auto dataHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        // Treat -i/-input as aliases for -df (path normalization/drive-letter
        // handling is keyed on the arg name inside getRequiredArg).
        a.projPath = ArgumentParser::getRequiredArg(i, argc, argv, "-df");
        return true;
    };
    handlers_["-df"] = dataHandler;
    handlers_["-i"] = dataHandler;
    handlers_["-input"] = dataHandler;

    handlers_["-dfList"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.projPaths = ArgumentParser::sortPathsByBedOrder(
            splitPathList(ArgumentParser::getRequiredArg(i, argc, argv, "-dfList")));
        return true;
    };
    handlers_["-indexMat"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.sysIndexPath = ArgumentParser::getRequiredArg(i, argc, argv, "-indexMat");
        return true;
    };
    handlers_["-weightMat"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.sysWeightPath = ArgumentParser::getRequiredArg(i, argc, argv, "-weightMat");
        return true;
    };
    handlers_["-saveCSR"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.saveCSRPath = ArgumentParser::getRequiredArg(i, argc, argv, "-saveCSR");
        return true;
    };
    handlers_["-loadCSR"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.loadCSRPath = ArgumentParser::getRequiredArg(i, argc, argv, "-loadCSR");
        a.loadCSRExplicit = true;
        return true;
    };
    handlers_["-chunkCSRDir"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.chunkCSRDir = ArgumentParser::getRequiredArg(i, argc, argv, "-chunkCSRDir");
        a.chunkCSRDirExplicit = true;
        return true;
    };
    handlers_["-mapCSR"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.mapCSR);
    };
    handlers_["-matrixLayout"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        std::string val = toLower(
            ArgumentParser::getRequiredArg(i, argc, argv, "-matrixLayout"));
        if (val == "compact" || val == "compact_symmetric")
            val = "compact-symmetric";
        a.matrixLayout = val;
        a.matrixLayoutExplicit = true;
        return true;
    };
    handlers_["-compactMeta"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.compactMetaPath =
            ArgumentParser::getRequiredArg(i, argc, argv, "-compactMeta");
        return true;
    };
    handlers_["-compactDenomCache"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.compactDenomCachePath =
            ArgumentParser::getRequiredArg(i, argc, argv, "-compactDenomCache");
        return true;
    };

    // --- Dimensions & Spacing ---
    handlers_["-dim"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseDimensions(
            ArgumentParser::getRequiredArg(i, argc, argv, "-dim"), a);
    };
    auto voxelHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        auto parts = split(ArgumentParser::getRequiredArg(i, argc, argv, k), ',');
        if (parts.size() == 1) {
            a.Vx = a.Vy = a.Vz = (float)std::stod(parts[0]);
            a.voxelSizeMM = {a.Vx, a.Vy, a.Vz};
            a.dx = a.Vx;
            a.dy = a.Vy;
            a.dz = a.Vz;
            return true;
        }
        if (parts.size() != 3) {
            std::cerr << "Error: Invalid " << k << " format (expected v or x,y,z).\n";
            return false;
        }
        a.Vx = (float)std::stod(parts[0]);
        a.Vy = (float)std::stod(parts[1]);
        a.Vz = (float)std::stod(parts[2]);
        a.voxelSizeMM = {a.Vx, a.Vy, a.Vz};
        a.dx = a.Vx;
        a.dy = a.Vy;
        a.dz = a.Vz;
        return true;
    };
    handlers_["-vox"] = voxelHandler;
    handlers_["-spacing"] = voxelHandler;

    handlers_["-detN"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        auto parts = split(ArgumentParser::getRequiredArg(i, argc, argv, "-detN"), ',');
        if (parts.size() == 1) {
            a.detNz = a.detNx = std::stoi(parts[0]);
            a.detectorChannels = a.detNz;
        } else if (parts.size() == 2) {
            a.detNz = std::stoi(parts[0]);
            a.detNx = std::stoi(parts[1]);
            a.detectorChannels = std::max(a.detNz, a.detNx);
        } else {
            std::cerr << "Error: Invalid -detN format (expected nz or nz,nx).\n";
            return false;
        }
        return true;
    };

    auto anglesHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        a.numAngles = std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, k));
        return true;
    };
    handlers_["-ang"] = anglesHandler;
    handlers_["-angles"] = anglesHandler;

    // --- Algorithm & Iterations ---
    handlers_["-it"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseIterations(
            ArgumentParser::getRequiredArg(i, argc, argv, "-it"), a);
    };
    auto threadsHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        a.numThreads = std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, k));
        return true;
    };
    handlers_["-th"] = threadsHandler;
    handlers_["-threads"] = threadsHandler;
    handlers_["-t"] = threadsHandler;
    handlers_["-sparseBackend"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        std::string val = toLower(ArgumentParser::getRequiredArg(i, argc, argv, "-sparseBackend"));
        if (val == "reference") val = "ref";
        a.sparseBackend = val;
        return true;
    };
    handlers_["-spBackend"] = handlers_["-sparseBackend"];

    handlers_["-subsets"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.numSubsets =
            std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, "-subsets"));
        return true;
    };

    auto algHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        const std::string val =
            toLower(ArgumentParser::getRequiredArg(i, argc, argv, k));
        if (val == "mlem") {
            a.algorithm = ReconArgs::Algorithm::MLEM;
        } else if (val == "bsrem") {
            a.algorithm = ReconArgs::Algorithm::BSREM;
        } else if (val == "admm") {
            a.algorithm = ReconArgs::Algorithm::ADMM;
        } else {
            std::cerr << "Error: Unknown optimisation algorithm '" << val << "'.\n";
            return false;
        }
        return true;
    };
    handlers_["-opti"] = algHandler;
    handlers_["-alg"] = algHandler;

    handlers_["-admmRho"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.admmRho = std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv, "-admmRho"));
        return true;
    };

    handlers_["-admmTvInner"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.admmTvInner = std::stoi(
            ArgumentParser::getRequiredArg(i, argc, argv, "-admmTvInner"));
        return true;
    };

    handlers_["-admmAdaptive"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        auto val = ArgumentParser::getRequiredArg(i, argc, argv, "-admmAdaptive");
        a.admmAdaptive = (val == "on" || val == "true" || val == "1");
        return true;
    };

    handlers_["-bsrem"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        auto parts = split(ArgumentParser::getRequiredArg(i, argc, argv, "-bsrem"), ',');
        if (parts.empty())
            return false;
        a.bsrem_alpha0 = (float)std::stod(parts[0]);
        if (parts.size() > 1)
            a.bsrem_gamma = (float)std::stod(parts[1]);
        return true;
    };
    handlers_["-bsremCurv"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.bsremUseCurvDenom);
    };
    handlers_["-bsremCurvFreeze"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.bsremFreezeCurv);
    };
    handlers_["-bsremCurvWarmup"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.bsremCurvWarmupIters = std::stoi(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bsremCurvWarmup"));
        return true;
    };
    handlers_["-bsremCurvEps"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.bsremCurvEps = std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bsremCurvEps"));
        return true;
    };
    handlers_["-bsremKappa"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.bsremUseKappaMap);
    };
    handlers_["-bsremKappaEps"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.bsremKappaEps = std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bsremKappaEps"));
        return true;
    };
    handlers_["-bsremKappaClip"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        auto parts = split(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bsremKappaClip"), ',');
        if (parts.size() != 2) {
            std::cerr << "Error: -bsremKappaClip requires min,max.\n";
            return false;
        }
        a.bsremKappaClipMin = std::stod(parts[0]);
        a.bsremKappaClipMax = std::stod(parts[1]);
        return true;
    };
    handlers_["-bsremSoftMask"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.bsremSoftMask);
    };
    handlers_["-bsremSoftBand"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.bsremSoftMaskBandVox = std::stoi(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bsremSoftBand"));
        return true;
    };
    handlers_["-bsremSoftLambda"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.bsremSoftMaskLambdaOut = std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bsremSoftLambda"));
        return true;
    };

    // --- IO & Verbosity ---
    handlers_["-fout"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOutputPath(
            ArgumentParser::getRequiredArg(i, argc, argv, "-fout"), a);
    };
    auto outDirHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        a.outputDir = ArgumentParser::getRequiredArg(i, argc, argv, k);
        return true;
    };
    handlers_["-o"] = outDirHandler;
    handlers_["-output"] = outDirHandler;
    auto outNameHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        a.outputName = ArgumentParser::getRequiredArg(i, argc, argv, k);
        return true;
    };
    handlers_["-n"] = outNameHandler;
    handlers_["-name"] = outNameHandler;
    auto verbosityHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        a.verbosity = std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, k));
        return true;
    };
    handlers_["-vb"] = verbosityHandler;
    handlers_["-v"] = verbosityHandler;

    // --- Filtering & PSF ---
    handlers_["-conv"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parsePSFParams(
            ArgumentParser::getRequiredArg(i, argc, argv, "-conv"), a);
    };
    handlers_["-psf"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        auto parts = split(ArgumentParser::getRequiredArg(i, argc, argv, "-psf"), ',');
        if (parts.empty())
            return false;
        const double sigma_mm = std::stod(parts[0]);
        if (sigma_mm <= 0.0) {
            std::cerr << "Error: -psf sigma must be positive.\n";
            return false;
        }
        a.usePSF = true;
        a.psf.fwhm_mm = sigma_mm * 2.354820045;
        if (!a.psfModeExplicit)
            a.psf.mode = PSFMode::Operator;
        return true;
    };
    handlers_["-psfProj"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.psfProj);
    };
    handlers_["-psfMode"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string val =
            toLower(ArgumentParser::getRequiredArg(i, argc, argv, "-psfMode"));
        a.psfModeExplicit = true;
        if (val == "operator")
            a.psf.mode = PSFMode::Operator;
        else if (val == "postsmooth")
            a.psf.mode = PSFMode::PostSmooth;
        else if (val == "none")
            a.psf.mode = PSFMode::None;
        else if (val == "projdomain")
            a.psf.mode = PSFMode::ProjDomain;
        else {
            std::cerr << "Error: Unknown -psfMode '" << val << "'.\n";
            return false;
        }
        return true;
    };
    handlers_["-smooth"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.useSmooth = true;
        a.smoothSigma =
            (float)std::stod(ArgumentParser::getRequiredArg(i, argc, argv, "-smooth"));
        return true;
    };
    handlers_["-smoothKernel"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.smoothKernel =
            std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, "-smoothKernel"));
        return true;
    };
    handlers_["-butter"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        auto parts =
            split(ArgumentParser::getRequiredArg(i, argc, argv, "-butter"), ',');
        if (parts.empty())
            return false;
        a.useButter = true;
        a.butterCutoff = std::stod(parts[0]);
        if (parts.size() > 1)
            a.butterOrder = std::stoi(parts[1]);
        return true;
    };
    handlers_["-butterOrder"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.butterOrder =
            std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, "-butterOrder"));
        return true;
    };
    handlers_["-butterBackend"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.butterBackend = toLower(ArgumentParser::getRequiredArg(i, argc, argv, "-butterBackend"));
        return true;
    };
    handlers_["-bwBackend"] = handlers_["-butterBackend"];

    // --- Regularization ---
    handlers_["-reg"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        auto parts = split(ArgumentParser::getRequiredArg(i, argc, argv, "-reg"), ',');
        if (parts.empty())
            return false;
        std::string typeStr = parts[0];
        const std::string typeLower = toLower(typeStr);
        if (typeLower == "none")
            typeStr = "None";
        else if (typeLower == "tv")
            typeStr = "TVIsotropic";
        else if (typeLower == "nlm")
            typeStr = "NLM";
        else if (typeLower == "bowsher")
            typeStr = "Bowsher";
        else if (typeLower == "pls")
            typeStr = "PLS";
        else if (typeLower == "rdp")
            typeStr = "RDP";
        else if (typeLower == "logcosh" || typeLower == "lc")
            typeStr = "LogCosh";

        const RegType t = stringToRegType(typeStr);
        if (t == RegType::Unknown) {
            std::cerr << "Error: Unknown regularization type '" << parts[0] << "'.\n";
            return false;
        }
        if (t == RegType::None) {
            a.useReg = false;
            a.reg.type = RegType::None;
        } else {
            a.useReg = true;
            a.reg.type = t;
            if (parts.size() >= 2)
                a.reg.beta = std::stod(parts[1]);
            if (t == RegType::NLM) {
                if (parts.size() >= 3)
                    a.reg.nlm_h = std::stod(parts[2]);
                if (parts.size() >= 4)
                    a.reg.nlm_search = std::stoi(parts[3]);
                if (parts.size() >= 5)
                    a.reg.nlm_patch = std::stoi(parts[4]);
            } else if (t == RegType::Bowsher) {
                if (parts.size() >= 3)
                    a.reg.bowsher_k = std::stoi(parts[2]);
            } else if (t == RegType::PLS) {
                if (parts.size() >= 3)
                    a.reg.pls_eta = std::stod(parts[2]);
            } else if (t == RegType::RDP) {
                if (parts.size() >= 3)
                    a.reg.gamma = std::stod(parts[2]);
                if (parts.size() >= 4)
                    a.reg.delta = std::stod(parts[3]);
            } else if (t == RegType::LogCosh) {
                if (parts.size() >= 3)
                    a.reg.delta = std::stod(parts[2]);
            }
        }
        return true;
    };

    auto regValHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.useReg = true;
        const std::string key = argv[i];
        double val = std::stod(ArgumentParser::getRequiredArg(i, argc, argv, key));
        if (key == "-regBeta")
            a.reg.beta = val;
        else if (key == "-regDelta")
            a.reg.delta = val;
        else if (key == "-regPower")
            a.reg.power = val;
        else if (key == "-regSigma")
            a.reg.sigma = val;
        return true;
    };
    handlers_["-regBeta"] = regValHandler;
    handlers_["-regDelta"] = regValHandler;
    handlers_["-regPower"] = regValHandler;
    handlers_["-regSigma"] = regValHandler;

    handlers_["-regKernel"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.useReg = true;
        a.reg.kernel =
            std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, "-regKernel"));
        return true;
    };
    handlers_["-regIter"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.useReg = true;
        a.reg.numIter =
            std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, "-regIter"));
        return true;
    };
    handlers_["-regSubsets"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.useReg = true;
        a.reg.numSubsets = std::stoi(
            ArgumentParser::getRequiredArg(i, argc, argv, "-regSubsets"));
        return true;
    };
    handlers_["-regUpdateFreq"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.useReg = true;
        a.reg.updateFreq = std::stoi(
            ArgumentParser::getRequiredArg(i, argc, argv, "-regUpdateFreq"));
        return true;
    };
    handlers_["-regUpdateStart"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.useReg = true;
        a.reg.updateStart = std::stoi(
            ArgumentParser::getRequiredArg(i, argc, argv, "-regUpdateStart"));
        return true;
    };
    handlers_["-regUpdateEnd"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.useReg = true;
        a.reg.updateEnd =
            std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, "-regUpdateEnd"));
        return true;
    };
    handlers_["-regUpdateInterval"] = [](int& i, int argc, char** argv,
                                        ReconArgs& a) {
        a.useReg = true;
        a.reg.updateInterval = std::stoi(
            ArgumentParser::getRequiredArg(i, argc, argv, "-regUpdateInterval"));
        return true;
    };
    handlers_["-regUpdateType"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.useReg = true;
        const std::string val =
            toLower(ArgumentParser::getRequiredArg(i, argc, argv, "-regUpdateType"));
        if (val == "subset")
            a.reg.updateType = RegConf::UpdateType::Subset;
        else if (val == "iteration")
            a.reg.updateType = RegConf::UpdateType::Iteration;
        else
            return false;
        return true;
    };

    // --- Mask & Geometry Extras ---
    auto cylHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        auto parts = split(ArgumentParser::getRequiredArg(i, argc, argv, k), ',');
        if (parts.size() < 2)
            return false;
        a.useCylindricalMask = true;
        a.cylinderDiameterMM = (float)std::stod(parts[0]);
        a.cylinderHeightMM = (float)std::stod(parts[1]);
        return true;
    };
    handlers_["-cyl"] = cylHandler;
    handlers_["-cylMask"] = cylHandler;

    auto cylCenterHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        auto parts = split(ArgumentParser::getRequiredArg(i, argc, argv, k), ',');
        if (parts.size() != 3)
            return false;
        a.cylCenterX = std::stod(parts[0]);
        a.cylCenterY = std::stod(parts[1]);
        a.cylCenterZ = std::stod(parts[2]);
        return true;
    };
    handlers_["-cylc"] = cylCenterHandler;
    handlers_["-cylCenter"] = cylCenterHandler;

    handlers_["-maskFromAtt"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.maskFromAttThreshold = (float)std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv, "-maskFromAtt"));
        return true;
    };
    handlers_["-bedOverlap"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        auto parts =
            split(ArgumentParser::getRequiredArg(i, argc, argv, "-bedOverlap"), ',');
        a.bedOverlaps.clear();
        for (auto& p : parts) {
            if (!p.empty())
                a.bedOverlaps.push_back(std::stoi(p));
        }
        return true;
    };
    handlers_["-cropEnd"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        auto parts =
            split(ArgumentParser::getRequiredArg(i, argc, argv, "-cropEnd"), ',');
        if (parts.size() == 1) {
            int v = std::stoi(parts[0]);
            a.cropEndStart = v;
            a.cropEndEnd = v;
        } else if (parts.size() >= 2) {
            a.cropEndStart = std::stoi(parts[0]);
            a.cropEndEnd = std::stoi(parts[1]);
        }
        return true;
    };
    handlers_["-zbedflip"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.zBedFlip);
    };
    handlers_["-zoutflip"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.zOutFlip);
    };

    // --- Physics & Correction ---
    handlers_["-ctac"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        (void)i;
        (void)argc;
        (void)argv;
        a.doCTAC = true;
        return true;
    };
    handlers_["-ctacModel"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        std::string value = ArgumentParser::getRequiredArg(i, argc, argv, "-ctacModel");
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (value == "line-integral" || value == "lineintegral" || value == "ray") {
            a.ctacModel = "line-integral";
        } else if (value == "per-voxel" || value == "pervoxel" || value == "voxel" ||
                   value == "legacy") {
            a.ctacModel = "per-voxel";
        } else {
            std::cerr << "Error: -ctacModel expects line-integral|per-voxel, got '"
                      << value << "'\n";
            return false;
        }
        return true;
    };
    handlers_["-ctacAutoMu"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.ctacAutoMuFrom = ArgumentParser::getRequiredArg(i, argc, argv, "-ctacAutoMu");
        return true;
    };
    handlers_["-ctacAutoMuThresh"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.ctacAutoMuThresh =
            std::stod(ArgumentParser::getRequiredArg(i, argc, argv, "-ctacAutoMuThresh"));
        if (!(a.ctacAutoMuThresh > 0.0) || a.ctacAutoMuThresh >= 1.0) {
            std::cerr << "Error: -ctacAutoMuThresh must be in (0, 1).\n";
            return false;
        }
        return true;
    };
    handlers_["-ctacAutoMuValue"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.ctacAutoMuValue =
            std::stod(ArgumentParser::getRequiredArg(i, argc, argv, "-ctacAutoMuValue"));
        if (!(a.ctacAutoMuValue > 0.0)) {
            std::cerr << "Error: -ctacAutoMuValue (mu per mm) must be positive.\n";
            return false;
        }
        return true;
    };
    handlers_["-ctacRotSign"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string value = ArgumentParser::getRequiredArg(i, argc, argv, "-ctacRotSign");
        if (value == "1" || value == "+1" || value == "cw") {
            a.ctacRotSign = 1;
        } else if (value == "-1" || value == "ccw") {
            a.ctacRotSign = -1;
        } else {
            std::cerr << "Error: -ctacRotSign expects 1|-1 (cw|ccw), got '" << value
                      << "'\n";
            return false;
        }
        return true;
    };
    handlers_["-attMap"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.attMapPath = ArgumentParser::getRequiredArg(i, argc, argv, "-attMap");
        return true;
    };
    handlers_["-prior"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.priorPath = ArgumentParser::getRequiredArg(i, argc, argv, "-prior");
        return true;
    };
    handlers_["-bProj"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.bProjPath = ArgumentParser::getRequiredArg(i, argc, argv, "-bProj");
        return true;
    };
    handlers_["-bProjList"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.bProjPaths = ArgumentParser::sortPathsByBedOrder(
            splitPathList(ArgumentParser::getRequiredArg(i, argc, argv, "-bProjList")));
        return true;
    };

    // --- Performance & Misc ---
    handlers_["-Aprune"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.weightPruneEps =
            std::stod(ArgumentParser::getRequiredArg(i, argc, argv, "-Aprune"));
        return true;
    };
    handlers_["-Astream"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.Astream);
    };
    handlers_["-ApruneMask"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.ApruneMask);
    };
    handlers_["-joe"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.josephPasses =
            std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, "-joe"));
        return true;
    };
    handlers_["-config"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        (void)argc;
        (void)argv;
        (void)a;
        ++i;
        return true; // Handled by pre-scan
    };
    handlers_["-algoConf"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.algoConfPath = ArgumentParser::getRequiredArg(i, argc, argv, "-algoConf");
        return true; // Loading handled in parse() after CLI pass
    };
    handlers_["-dumpConf"] = [](int& i, int argc, char** argv, ReconArgs& a) {
        (void)i;
        (void)argc;
        (void)argv;
        a.dumpConf = true;
        return true;
    };
    auto helpHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        (void)i;
        (void)argc;
        (void)argv;
        (void)a;
        ArgumentParser::printUsage();
        std::exit(0);
        return true;
    };
    handlers_["-h"] = helpHandler;
    handlers_["-help"] = helpHandler;
    handlers_["--help"] = helpHandler;
}

inline void ArgumentParser::applyConfig(ReconArgs& a, const ConfigParser& cfg) {
    // Basic
    std::string dims = cfg.getString("Basic", "dimensions");
    if (!dims.empty()) {
        auto d = split(dims, ',');
        if (d.size() == 3) {
            a.Nx = std::stoi(d[0]);
            a.Ny = std::stoi(d[1]);
            a.Nz = std::stoi(d[2]);
            a.dimExplicit = true;
        }
    }
    std::string vox = cfg.getString("Basic", "voxel_size");
    if (!vox.empty()) {
        auto v = split(vox, ',');
        if (v.size() == 3) {
            a.Vx = (float)std::stod(v[0]);
            a.Vy = (float)std::stod(v[1]);
            a.Vz = (float)std::stod(v[2]);
            a.voxelSizeMM = {a.Vx, a.Vy, a.Vz};
            a.dx = a.Vx;
            a.dy = a.Vy;
            a.dz = a.Vz; // Sync double spacing
        }
    }
    a.numThreads = cfg.getInt("Basic", "threads", a.numThreads);
    a.sparseBackend = toLower(cfg.getString("Reconstruction", "sparse_backend", a.sparseBackend));
    a.butterBackend = toLower(cfg.getString("Reconstruction", "butter_backend", a.butterBackend));
    a.verbosity = cfg.getInt("Basic", "verbosity", a.verbosity);

    // Detector
    std::string detCh = cfg.getString("Detector", "channels");
    if (!detCh.empty()) {
        auto ch = split(detCh, ',');
        if (ch.size() == 1) {
            a.detectorChannels = std::stoi(ch[0]);
            a.detNz = a.detNx = a.detectorChannels;
        } else if (ch.size() == 2) {
            a.detNz = std::stoi(ch[0]);
            a.detNx = std::stoi(ch[1]);
            a.detectorChannels = std::max(a.detNz, a.detNx);
        }
    }
    a.numAngles = cfg.getInt("Detector", "angles", a.numAngles);

    // Recon
    std::string alg = cfg.getString("Reconstruction", "algorithm");
    if (alg == "BSREM")
        a.algorithm = ReconArgs::Algorithm::BSREM;
    else if (alg == "MLEM")
        a.algorithm = ReconArgs::Algorithm::MLEM;
    else if (alg == "ADMM")
        a.algorithm = ReconArgs::Algorithm::ADMM;

    std::string iters = cfg.getString("Reconstruction", "iterations");
    if (!iters.empty()) {
        auto it = split(iters, ':');
        if (!it.empty())
            a.numIters = std::stoi(it[0]);
        if (it.size() > 1)
            a.saveInterval = std::stoi(it[1]);
    }
    a.numSubsets = cfg.getInt("Reconstruction", "subsets", a.numSubsets);
    a.bsrem_alpha0 =
        (float)cfg.getDouble("Reconstruction", "bsrem_alpha0", a.bsrem_alpha0);
    a.bsrem_gamma =
        (float)cfg.getDouble("Reconstruction", "bsrem_gamma", a.bsrem_gamma);
    a.bsremUseCurvDenom = cfg.getBool("Reconstruction", "bsrem_curv_denom", a.bsremUseCurvDenom);
    a.bsremFreezeCurv = cfg.getBool("Reconstruction", "bsrem_curv_freeze", a.bsremFreezeCurv);
    a.bsremCurvWarmupIters = cfg.getInt(
        "Reconstruction", "bsrem_curv_warmup_iters", a.bsremCurvWarmupIters);
    a.bsremCurvEps = cfg.getDouble("Reconstruction", "bsrem_curv_eps", a.bsremCurvEps);
    a.bsremUseKappaMap =
        cfg.getBool("Reconstruction", "bsrem_kappa_map", a.bsremUseKappaMap);
    a.bsremKappaEps =
        cfg.getDouble("Reconstruction", "bsrem_kappa_eps", a.bsremKappaEps);
    a.bsremKappaClipMin = cfg.getDouble(
        "Reconstruction", "bsrem_kappa_clip_min", a.bsremKappaClipMin);
    a.bsremKappaClipMax = cfg.getDouble(
        "Reconstruction", "bsrem_kappa_clip_max", a.bsremKappaClipMax);
    a.bsremSoftMask =
        cfg.getBool("Reconstruction", "bsrem_soft_mask", a.bsremSoftMask);
    a.bsremSoftMaskBandVox = cfg.getInt(
        "Reconstruction", "bsrem_soft_mask_band_vox", a.bsremSoftMaskBandVox);
    a.bsremSoftMaskLambdaOut = cfg.getDouble(
        "Reconstruction", "bsrem_soft_mask_lambda_out", a.bsremSoftMaskLambdaOut);
    a.josephPasses = cfg.getInt("Reconstruction", "joseph_passes", a.josephPasses);
    a.bProjPath = cfg.getString("Reconstruction", "b_proj", a.bProjPath);

    // IO
    a.outputDir = cfg.getString("InputOutput", "output_dir", a.outputDir);
    a.outputName = cfg.getString("InputOutput", "output_name", a.outputName);

    // System Matrix
    a.weightPruneEps =
        cfg.getDouble("SystemMatrix", "prune_epsilon", a.weightPruneEps);
    a.Astream = cfg.getBool("SystemMatrix", "stream_build", a.Astream);
    a.ApruneMask = cfg.getBool("SystemMatrix", "prune_mask", a.ApruneMask);
    a.mapCSR = cfg.getBool("SystemMatrix", "map_csr", a.mapCSR);
    std::string layout = cfg.getString("SystemMatrix", "matrix_layout");
    if (layout.empty())
        layout = cfg.getString("Matrix", "layout");
    if (!layout.empty()) {
        layout = toLower(layout);
        if (layout == "compact" || layout == "compact_symmetric")
            layout = "compact-symmetric";
        a.matrixLayout = layout;
    }
    std::string chunkDir =
        cfg.getString("SystemMatrix", "default_chunk_csr_dir");
    if (chunkDir.empty())
        chunkDir = cfg.getString("SystemMatrix", "chunk_csr_dir");
    if (chunkDir.empty())
        chunkDir = cfg.getString("Matrix", "chunk_csr_dir");
    if (!chunkDir.empty())
        a.chunkCSRDir = normalizePath(chunkDir);

    // Regularization
    std::string regT = cfg.getString("Regularization", "type");
    if (regT != "None" && !regT.empty()) {
        a.useReg = true;
        a.reg.type = stringToRegType(regT);
        if (a.reg.type == RegType::Unknown) {
            std::cerr << "Error: Unknown regularization type '" << regT
                      << "' in config file.\n";
            std::exit(1);
        }
        a.reg.beta = cfg.getDouble("Regularization", "beta", a.reg.beta);
        double eps = cfg.getDouble("Regularization", "epsilon", 1e-6);
        a.reg.tv_eps = eps;
        a.reg.mrp_eps = eps;
        // Delta for LogCosh/RDP
        a.reg.delta = cfg.getDouble("Regularization", "delta", a.reg.delta);

        // NLM
        a.reg.nlm_h = cfg.getDouble("NLM", "h", a.reg.nlm_h);
        a.reg.nlm_search = cfg.getInt("NLM", "search_radius", a.reg.nlm_search);
        a.reg.nlm_patch = cfg.getInt("NLM", "patch_radius", a.reg.nlm_patch);

        // Priors
        a.priorPath =
            cfg.getString("AnatomicalPriors", "prior_image", a.priorPath);
        a.reg.bowsher_k =
            cfg.getInt("AnatomicalPriors", "bowsher_k", a.reg.bowsher_k);
        a.reg.pls_eta = cfg.getDouble("AnatomicalPriors", "pls_eta", a.reg.pls_eta);
    }

    // PSF
    std::string psfModeStr = cfg.getString("PSF", "mode");
    if (!psfModeStr.empty()) {
        std::string psfModeLower = toLower(psfModeStr);
        if (psfModeLower == "none")
            a.psf.mode = PSFMode::None;
        else if (psfModeLower == "operator")
            a.psf.mode = PSFMode::Operator;
        else if (psfModeLower == "postsmooth")
            a.psf.mode = PSFMode::PostSmooth;
        else if (psfModeLower == "projdomain")
            a.psf.mode = PSFMode::ProjDomain;
        else {
            std::cerr << "Warning: Unknown PSF mode '" << psfModeStr
                      << "' in config, using None.\n";
            a.psf.mode = PSFMode::None;
        }
        // If mode is set to something other than None, enable PSF
        if (a.psf.mode != PSFMode::None)
            a.usePSF = true;
    }
    a.psf.fwhm_mm = cfg.getDouble("PSF", "fwhm", a.psf.fwhm_mm);

    std::string minSigmas = cfg.getString("PSF", "min_sigmas");
    if (!minSigmas.empty()) {
        auto ms = split(minSigmas, ',');
        if (!ms.empty())
            a.psf.min_sigmas = std::stod(ms[0]);
    }

    // Mask
    a.useCylindricalMask = cfg.getBool("Mask", "use_cylinder", a.useCylindricalMask);
    double attTh = cfg.getDouble("Mask", "att_threshold", -1.0);
    if (attTh >= 0)
        a.maskFromAttThreshold = (float)attTh;

    std::string cylD = cfg.getString("Mask", "cylinder_dims");
    if (!cylD.empty()) {
        auto cd = split(cylD, ',');
        if (cd.size() >= 2) {
            a.cylinderDiameterMM = (float)std::stod(cd[0]);
            a.cylinderHeightMM = (float)std::stod(cd[1]);
        }
    }
    std::string cylCenter = cfg.getString("Mask", "cylinder_center");
    if (!cylCenter.empty()) {
        auto cc = split(cylCenter, ',');
        if (cc.size() == 3) {
            a.cylCenterX = std::stod(cc[0]);
            a.cylCenterY = std::stod(cc[1]);
            a.cylCenterZ = std::stod(cc[2]);
        }
    }

    // Filtering
    if (cfg.hasSection("Filtering")) {
        double bwCut = cfg.getDouble("Filtering", "butterworth_cutoff", 0.0);
        if (bwCut > 0.0) {
            a.butterCutoff = (float)bwCut;
        }
        a.butterOrder = cfg.getInt("Filtering", "butterworth_order", a.butterOrder);
        std::string bwBackend = cfg.getString("Filtering", "butterworth_backend");
        if (!bwBackend.empty())
            a.butterBackend = toLower(bwBackend);
        double gSigma = cfg.getDouble("Filtering", "gaussian_sigma", 0.0);
        if (gSigma > 0.0) {
            a.useSmooth = true;
            a.smoothSigma = (float)gSigma;
        }
        a.smoothKernel = cfg.getInt("Filtering", "gaussian_kernel", a.smoothKernel);
    }

    // MultiBed
    if (cfg.hasSection("MultiBed")) {
        std::string overlap = cfg.getString("MultiBed", "bed_overlap");
        if (!overlap.empty()) {
            a.bedOverlaps.clear();
            auto op = split(overlap, ',');
            for (auto& p : op) {
                if (!p.empty())
                    a.bedOverlaps.push_back(std::stoi(p));
            }
        }
        std::string cropEnd = cfg.getString("MultiBed", "crop_end");
        if (!cropEnd.empty()) {
            auto ce = split(cropEnd, ',');
            if (ce.size() == 1) {
                int v = std::stoi(ce[0]);
                a.cropEndStart = v;
                a.cropEndEnd = v;
            } else if (ce.size() >= 2) {
                a.cropEndStart = std::stoi(ce[0]);
                a.cropEndEnd = std::stoi(ce[1]);
            }
        }
        a.zBedFlip = cfg.getBool("MultiBed", "z_bed_flip", a.zBedFlip);
        a.zOutFlip = cfg.getBool("MultiBed", "z_out_flip", a.zOutFlip);
    }
}

inline void ArgumentParser::applyAlgoConf(ReconArgs& a, const std::string& path) {
    ConfigParser cfg;
    if (!cfg.load(path)) {
        std::cerr << "Warning: Cannot load algorithm config '" << path << "'.\n";
        return;
    }

    // [Algorithm] section
    std::string alg = cfg.getString("Algorithm", "name");
    if (!alg.empty()) {
        std::string algLower = toLower(alg);
        if (algLower == "mlem")
            a.algorithm = ReconArgs::Algorithm::MLEM;
        else if (algLower == "bsrem")
            a.algorithm = ReconArgs::Algorithm::BSREM;
        else if (algLower == "admm")
            a.algorithm = ReconArgs::Algorithm::ADMM;
    }
    std::string iters = cfg.getString("Algorithm", "iterations");
    if (!iters.empty()) {
        auto it = split(iters, ':');
        if (!it.empty())
            a.numIters = std::stoi(it[0]);
        if (it.size() > 1)
            a.saveInterval = std::stoi(it[1]);
    }
    if (cfg.hasKey("Algorithm", "subsets"))
        a.numSubsets = cfg.getInt("Algorithm", "subsets", a.numSubsets);

    // [Regularization] section
    std::string regT = cfg.getString("Regularization", "type");
    if (!regT.empty()) {
        std::string regLower = toLower(regT);
        if (regLower == "none") {
            a.useReg = false;
            a.reg.type = RegType::None;
        } else if (regLower == "tv")
            regT = "TVIsotropic";

        RegType rt = stringToRegType(regT);
        if (rt != RegType::Unknown && rt != RegType::None) {
            a.useReg = true;
            a.reg.type = rt;
        }
    }
    if (cfg.hasKey("Regularization", "beta"))
        a.reg.beta = cfg.getDouble("Regularization", "beta", a.reg.beta);
    if (cfg.hasKey("Regularization", "delta"))
        a.reg.delta = cfg.getDouble("Regularization", "delta", a.reg.delta);
    if (cfg.hasKey("Regularization", "epsilon")) {
        double eps = cfg.getDouble("Regularization", "epsilon", 1e-6);
        a.reg.tv_eps = eps;
        a.reg.mrp_eps = eps;
    }

    // [BSREM] section
    if (cfg.hasSection("BSREM")) {
        a.bsrem_alpha0 = (float)cfg.getDouble("BSREM", "alpha0", a.bsrem_alpha0);
        a.bsrem_gamma = (float)cfg.getDouble("BSREM", "gamma", a.bsrem_gamma);
        a.bsremUseCurvDenom = cfg.getBool("BSREM", "curv_denom", a.bsremUseCurvDenom);
        a.bsremFreezeCurv = cfg.getBool("BSREM", "curv_freeze", a.bsremFreezeCurv);
        a.bsremCurvWarmupIters = cfg.getInt("BSREM", "curv_warmup_iters", a.bsremCurvWarmupIters);
        a.bsremCurvEps = cfg.getDouble("BSREM", "curv_eps", a.bsremCurvEps);
        a.bsremUseKappaMap = cfg.getBool("BSREM", "kappa_map", a.bsremUseKappaMap);
        a.bsremKappaEps = cfg.getDouble("BSREM", "kappa_eps", a.bsremKappaEps);
        std::string kappaClip = cfg.getString("BSREM", "kappa_clip");
        if (!kappaClip.empty()) {
            auto kc = split(kappaClip, ',');
            if (kc.size() == 2) {
                a.bsremKappaClipMin = std::stod(kc[0]);
                a.bsremKappaClipMax = std::stod(kc[1]);
            }
        }
        a.bsremSoftMask = cfg.getBool("BSREM", "soft_mask", a.bsremSoftMask);
        a.bsremSoftMaskBandVox = cfg.getInt("BSREM", "soft_mask_band", a.bsremSoftMaskBandVox);
        a.bsremSoftMaskLambdaOut = cfg.getDouble("BSREM", "soft_mask_lambda", a.bsremSoftMaskLambdaOut);
    }

    // [ADMM] section
    if (cfg.hasSection("ADMM")) {
        a.admmRho = cfg.getDouble("ADMM", "rho", a.admmRho);
        a.admmTvInner = cfg.getInt("ADMM", "tv_inner", a.admmTvInner);
        a.admmAdaptive = cfg.getBool("ADMM", "adaptive", a.admmAdaptive);
    }

    // [BackendHint] section
    if (cfg.hasSection("BackendHint")) {
        std::string sp = cfg.getString("BackendHint", "sparse");
        if (!sp.empty())
            a.sparseBackend = toLower(sp);
    }
}

inline std::string ArgumentParser::getRequiredArg(int& i, int argc, char** argv,
                                                   const std::string& argName) {
    if (i + 1 >= argc) {
        std::cerr << "Error: Missing value for argument '" << argName << "'\n";
        std::exit(1);
    }
    std::string value = argv[++i];
    
    // Check if this is a path argument
    bool isPathArg =
        (argName == "-df" || argName == "-i" || argName == "-input" ||
         argName == "-dfList" || argName == "-bProjList" ||
         argName == "-indexMat" || argName == "-weightMat" ||
         argName == "-fout" || argName == "-loadCSR" ||
         argName == "-chunkCSRDir" || argName == "-saveCSR" ||
         argName == "-attMap" || argName == "-prior" || argName == "-bProj" ||
         argName == "-config" || argName == "-algoConf");
    
    if (isPathArg) {
        // Handle Windows drive letter edge case (D: followed by path)
        if (value.length() == 2 && value[1] == ':' &&
            std::isalpha(static_cast<unsigned char>(value[0]))) {
            if (i + 1 < argc && argv[i + 1][0] == '\\')
                value += argv[++i];
            else if (i + 1 < argc && argv[i + 1][0] != '-')
                value += argv[++i];
        }
        value = normalizePath(value);
    }
    return value;
}

inline bool ArgumentParser::parseOnOffOptional(int& i, int argc, char** argv, bool& out) {
    if (i + 1 < argc) {
        const std::string next = argv[i + 1];
        if (!next.empty() && next[0] != '-') {
            bool v = false;
            if (parseOnOff(next, v)) {
                out = v;
                ++i;
                return true;
            }
        }
    }
    out = true;
    return true;
}

inline bool ArgumentParser::parseDimensions(const std::string& value, ReconArgs& args) {
    auto dimensions = split(value, ',');
    if (dimensions.size() == 1) {
        args.Nx = args.Ny = args.Nz = std::stoi(dimensions[0]);
    } else if (dimensions.size() == 3) {
        args.Nx = std::stoi(dimensions[0]);
        args.Ny = std::stoi(dimensions[1]);
        args.Nz = std::stoi(dimensions[2]);
    } else {
        std::cerr << "Error: Invalid -dim format (expected x,y,z or single value).\n";
        return false;
    }
    args.dimExplicit = true;
    return true;
}

inline bool ArgumentParser::parseIterations(const std::string& value, ReconArgs& args) {
    auto iterations = split(value, ':');
    if (iterations.empty())
        return false;
    args.numIters = std::stoi(iterations[0]);
    if (iterations.size() > 1)
        args.saveInterval = std::stoi(iterations[1]);
    return true;
}

inline bool ArgumentParser::parsePSFParams(const std::string& value, ReconArgs& args) {
    if (value.find("::psf") != std::string::npos) {
        args.usePSF = true;
        auto psfParams = value.substr(0, value.find("::psf"));
        auto params = split(psfParams, ',');
        if (params.size() >= 2)
            args.psf.fwhm_mm = std::stod(params[1]) * 2.355;
        args.psf.mode = PSFMode::Gaussian;
        return true;
    }
    return false;
}

inline bool ArgumentParser::parseOutputPath(const std::string& path, ReconArgs& args) {
    try {
        std::filesystem::path p(path);
        args.outputDir = p.parent_path().string();
        args.outputName = p.filename().string();
        if (args.outputDir.empty())
            args.outputDir = ".";
        return true;
    } catch (...) {
        return false;
    }
}

inline bool ArgumentParser::validateArgs(const ReconArgs& args) {
    // Validate projection paths
    if (!args.projPaths.empty() && !args.projPath.empty()) {
        std::cerr << "Error: Use either -df or -dfList, not both.\n";
        return false;
    }
    if (!args.bProjPaths.empty() && !args.bProjPath.empty()) {
        std::cerr << "Error: Use either -bProj or -bProjList, not both.\n";
        return false;
    }
    if (args.projPath.empty() && args.projPaths.empty()) {
        std::cerr << "Error: Projection data path (-df) is required.\n";
        return false;
    }
    if (args.projPaths.size() > 5) {
        std::cerr << "Error: -dfList supports at most 5 beds.\n";
        return false;
    }
    if (args.bProjPaths.size() > 5) {
        std::cerr << "Error: -bProjList supports at most 5 beds.\n";
        return false;
    }
    if (args.bedOverlaps.size() > 5) {
        std::cerr << "Error: -bedOverlap supports at most 5 fixed bed slots.\n";
        return false;
    }

    // Validate dimensions. In compact chunk mode metadata supplies geometry
    // before projection loading in MainProgram.
    const bool compactChunkGeometry =
        args.matrixLayout == "compact-symmetric" && !args.chunkCSRDir.empty();
    if (!compactChunkGeometry &&
        (args.Nx <= 0 || args.Ny <= 0 || args.Nz <= 0)) {
        std::cerr << "Error: Invalid reconstruction dimensions (-dim).\n";
        return false;
    }
    if (args.dx <= 0 || args.dy <= 0 || args.dz <= 0) {
        std::cerr << "Error: Invalid voxel spacing (-vox).\n";
        return false;
    }

    // Validate iteration parameters
    if (args.numIters <= 0) {
        std::cerr << "Error: Number of iterations (-it) must be positive.\n";
        return false;
    }
    if (args.numSubsets <= 0) {
        std::cerr << "Error: Number of subsets (-subsets) must be positive.\n";
        return false;
    }
    if (args.numAngles <= 0) {
        std::cerr << "Error: Number of angles (-ang) must be positive.\n";
        return false;
    }
    if (args.numThreads <= 0) {
        std::cerr << "Error: Number of threads (-th) must be positive.\n";
        return false;
    }
    if (!(args.sparseBackend == "auto" || args.sparseBackend == "cpu" ||
          args.sparseBackend == "ref" || args.sparseBackend == "reference" ||
          args.sparseBackend == "cuda" || args.sparseBackend == "gpu")) {
        std::cerr << "Error: Invalid -sparseBackend (expected auto/cpu/ref/cuda).\n";
        return false;
    }
    if (!(args.matrixLayout == "standard" ||
          args.matrixLayout == "compact-symmetric")) {
        std::cerr << "Error: Invalid -matrixLayout (expected standard or compact-symmetric).\n";
        return false;
    }
    if (!args.chunkCSRDir.empty() && !args.loadCSRPath.empty()) {
        std::cerr << "Error: Use either -loadCSR or -chunkCSRDir, not both.\n";
        return false;
    }
    if (!args.chunkCSRDir.empty() && args.matrixLayout != "compact-symmetric") {
        std::cerr << "Error: -chunkCSRDir requires -matrixLayout compact-symmetric.\n";
        return false;
    }
    if (!args.compactDenomCachePath.empty() &&
        args.matrixLayout != "compact-symmetric") {
        std::cerr << "Error: -compactDenomCache requires -matrixLayout compact-symmetric.\n";
        return false;
    }
    if (args.matrixLayout == "compact-symmetric") {
        if (args.loadCSRPath.empty() && args.chunkCSRDir.empty()) {
            std::cerr << "Error: compact-symmetric layout requires -loadCSR, -chunkCSRDir, [SystemMatrix] default_chunk_csr_dir, or exactly one exe-local matrix directory.\n";
            return false;
        }
        if (!(args.sparseBackend == "cuda" || args.sparseBackend == "gpu")) {
            std::cerr << "Error: compact-symmetric layout requires -sparseBackend cuda.\n";
            return false;
        }
        const bool compactAlgorithmOk =
            args.algorithm == ReconArgs::Algorithm::MLEM ||
            args.algorithm == ReconArgs::Algorithm::BSREM;
        if (!compactAlgorithmOk || args.numSubsets != 1) {
            std::cerr << "Error: compact-symmetric layout supports only -alg mlem|bsrem -subsets 1.\n";
            return false;
        }
        if ((args.doCTAC || !args.attMapPath.empty()) &&
            args.chunkCSRDir.empty()) {
            std::cerr << "Error: compact-symmetric CTAC requires -chunkCSRDir "
                         "(single-file compact matrix does not support it).\n";
            return false;
        }
        if (args.doCTAC && args.projPaths.size() > 1) {
            std::cerr << "Error: compact-symmetric CTAC currently supports a "
                         "single bed (device attenuation-table cache holds one "
                         "bed).\n";
            return false;
        }
        const bool compactHasReg =
            args.useReg && args.reg.type != RegType::None;
        if (compactHasReg &&
            (args.algorithm != ReconArgs::Algorithm::BSREM ||
             args.reg.type != RegType::RDP)) {
            std::cerr << "Error: compact-symmetric regularization currently supports only -alg bsrem -reg RDP,...\n";
            return false;
        }
        if (args.psfProj || args.psf.mode == PSFMode::ProjDomain ||
            args.psf.mode == PSFMode::PostSmooth) {
            std::cerr << "Error: compact-symmetric Phase 1 supports only image-domain -psfMode Operator or None.\n";
            return false;
        }
    }
        if (!(args.butterBackend == "auto" || args.butterBackend == "fftw" ||
                    args.butterBackend == "cpu" || args.butterBackend == "cufft" ||
                    args.butterBackend == "cuda" || args.butterBackend == "gpu")) {
                std::cerr << "Error: Invalid -butterBackend (expected auto/fftw/cufft).\n";
                return false;
        }
    if (args.bsremCurvWarmupIters < 0) {
        std::cerr << "Error: BSREM curvature warmup iterations must be >= 0.\n";
        return false;
    }
    if (!(args.bsremCurvEps > 0.0)) {
        std::cerr << "Error: BSREM curvature epsilon must be > 0.\n";
        return false;
    }
    if (!(args.bsremKappaEps > 0.0)) {
        std::cerr << "Error: BSREM kappa epsilon must be > 0.\n";
        return false;
    }
    if (!(args.bsremKappaClipMin > 0.0) ||
        !(args.bsremKappaClipMax >= args.bsremKappaClipMin)) {
        std::cerr << "Error: BSREM kappa clip range is invalid.\n";
        return false;
    }
    if (args.bsremSoftMask && args.bsremSoftMaskBandVox < 1) {
        std::cerr << "Error: BSREM soft-mask band must be >= 1 voxel.\n";
        return false;
    }
    if (args.bsremSoftMask && args.bsremSoftMaskLambdaOut < 0.0) {
        std::cerr << "Error: BSREM soft-mask lambda must be >= 0.\n";
        return false;
    }
    if (args.bsremSoftMask && std::isnan(args.maskFromAttThreshold)) {
        std::cerr << "Error: BSREM soft-mask requires -maskFromAtt.\n";
        return false;
    }

    // Validate CTAC auto mu-map options
    if (!args.ctacAutoMuFrom.empty() && !args.doCTAC) {
        std::cerr << "Error: -ctacAutoMu requires -ctac.\n";
        return false;
    }
    if (!args.ctacAutoMuFrom.empty() && !args.attMapPath.empty()) {
        std::cerr << "Error: -ctacAutoMu and -attMap are mutually exclusive; "
                     "pick the measured mu-map or the auto-generated one explicitly.\n";
        return false;
    }

    // Validate PSF parameters
    if (args.psfProj && !args.usePSF) {
        std::cerr << "Error: -psfProj requires -psf.\n";
        return false;
    }
    if (args.usePSF && args.psf.fwhm_mm <= 0.0) {
        std::cerr << "Error: PSF sigma (-psf) must be positive.\n";
        return false;
    }

    // Validate regularization
    if (args.useReg) {
        if (args.reg.beta < 0) {
            std::cerr
                << "Error: Regularization beta (-regBeta) must be non-negative.\n";
            return false;
        }
        if ((args.reg.type == RegType::TVIsotropic || 
             args.reg.type == RegType::TVHuber ||
             args.reg.type == RegType::RDP) && args.reg.delta < 0) {
            std::cerr << "Error: TV/Huber/RDP regularization delta (-regDelta) must "
                         "be non-negative.\n";
            return false;
        }
        if ((args.reg.type == RegType::Bowsher || args.reg.type == RegType::PLS) &&
            args.priorPath.empty() && args.attMapPath.empty()) {
            std::cerr << "Error: Anatomical regularization (Bowsher/PLS) requires an "
                         "anatomical prior (-prior or -attMap).\n";
            return false;
        }
    }

    // Validate smoothing
    if (args.useSmooth && args.smoothSigma <= 0) {
        std::cerr << "Error: Smoothing sigma (-smooth) must be positive.\n";
        return false;
    }

    // Validate cylindrical mask
    if (args.useCylindricalMask) {
        if (args.cylinderDiameterMM <= 0) {
            std::cerr << "Error: Cylinder diameter (-cylMask) must be positive.\n";
            return false;
        }
        if (args.cylinderHeightMM <= 0) {
            std::cerr << "Error: Cylinder height (-cylHeight) must be positive.\n";
            return false;
        }
    }

    return true;
}

inline void ArgumentParser::dumpMergedConfig(const ReconArgs& a) {
    std::cout << "# ── Merged Configuration (after all layers) ──\n\n";

    std::cout << "[Basic]\n";
    std::cout << "dimensions = " << a.Nx << "," << a.Ny << "," << a.Nz << "\n";
    std::cout << "voxel_size = " << a.Vx << "," << a.Vy << "," << a.Vz << "\n";
    std::cout << "threads    = " << a.numThreads << "\n";
    std::cout << "verbosity  = " << a.verbosity << "\n\n";

    std::cout << "[Detector]\n";
    std::cout << "channels = " << a.detNz;
    if (a.detNx != a.detNz) std::cout << "," << a.detNx;
    std::cout << "\nangles   = " << a.numAngles << "\n\n";

    std::cout << "[Algorithm]\n";
    std::cout << "name       = " << a.getAlgorithmName() << "\n";
    std::cout << "iterations = " << a.numIters << ":" << a.saveInterval << "\n";
    std::cout << "subsets    = " << a.numSubsets << "\n\n";

    std::cout << "[Regularization]\n";
    const char* regNames[] = {"None","TVIsotropic","TVHuber","Tikhonov","MRP",
                              "NLM","Bowsher","PLS","RDP","LogCosh"};
    int regIdx = static_cast<int>(a.reg.type);
    std::cout << "type  = " << (regIdx >= 0 && regIdx <= 9 ? regNames[regIdx] : "Unknown") << "\n";
    std::cout << "beta  = " << a.reg.beta << "\n";
    std::cout << "delta = " << a.reg.delta << "\n\n";

    if (a.algorithm == ReconArgs::Algorithm::BSREM) {
        std::cout << "[BSREM]\n";
        std::cout << "alpha0            = " << a.bsrem_alpha0 << "\n";
        std::cout << "gamma             = " << a.bsrem_gamma << "\n";
        std::cout << "curv_denom        = " << (a.bsremUseCurvDenom ? "true" : "false") << "\n";
        std::cout << "curv_freeze       = " << (a.bsremFreezeCurv ? "true" : "false") << "\n";
        std::cout << "curv_warmup_iters = " << a.bsremCurvWarmupIters << "\n";
        std::cout << "curv_eps          = " << a.bsremCurvEps << "\n";
        std::cout << "kappa_map         = " << (a.bsremUseKappaMap ? "true" : "false") << "\n";
        std::cout << "kappa_eps         = " << a.bsremKappaEps << "\n";
        std::cout << "kappa_clip        = " << a.bsremKappaClipMin << "," << a.bsremKappaClipMax << "\n";
        std::cout << "soft_mask         = " << (a.bsremSoftMask ? "true" : "false") << "\n";
        std::cout << "soft_mask_band    = " << a.bsremSoftMaskBandVox << "\n";
        std::cout << "soft_mask_lambda  = " << a.bsremSoftMaskLambdaOut << "\n\n";
    }
    if (a.algorithm == ReconArgs::Algorithm::ADMM) {
        std::cout << "[ADMM]\n";
        std::cout << "rho      = " << a.admmRho << "\n";
        std::cout << "tv_inner = " << a.admmTvInner << "\n";
        std::cout << "adaptive = " << (a.admmAdaptive ? "true" : "false") << "\n\n";
    }

    std::cout << "[Filtering]\n";
    std::cout << "butterworth_cutoff  = " << a.butterCutoff << "\n";
    std::cout << "butterworth_order   = " << a.butterOrder << "\n";
    std::cout << "butterworth_backend = " << a.butterBackend << "\n";
    std::cout << "gaussian_sigma      = " << a.smoothSigma << "\n";
    std::cout << "gaussian_kernel     = " << a.smoothKernel << "\n\n";

    std::cout << "[PSF]\n";
    const char* psfModes[] = {"None","Operator","PostSmooth","ProjDomain","Gaussian","File"};
    int psfIdx = static_cast<int>(a.psf.mode);
    std::cout << "mode = " << (psfIdx >= 0 && psfIdx <= 5 ? psfModes[psfIdx] : "Unknown") << "\n";
    std::cout << "fwhm = " << a.psf.fwhm_mm << "\n\n";

    std::cout << "[Mask]\n";
    std::cout << "use_cylinder   = " << (a.useCylindricalMask ? "true" : "false") << "\n";
    std::cout << "cylinder_dims  = " << a.cylinderDiameterMM << "," << a.cylinderHeightMM << "\n\n";

    std::cout << "[BackendHint]\n";
    std::cout << "sparse = " << a.sparseBackend << "\n";
    std::cout << "butter = " << a.butterBackend << "\n\n";

    std::cout << "[Matrix]\n";
    std::cout << "layout       = " << a.matrixLayout << "\n";
    std::cout << "compact_meta = " << (a.compactMetaPath.empty() ? "(default)" : a.compactMetaPath) << "\n";
    std::cout << "compact_denom_cache = " << (a.compactDenomCachePath.empty() ? "(none)" : a.compactDenomCachePath) << "\n";
    std::cout << "chunk_csr_dir = " << (a.chunkCSRDir.empty() ? "(none)" : a.chunkCSRDir) << "\n\n";

    std::cout << "[MultiBed]\n";
    std::cout << "bed_count    = " << a.getEffectiveBedCount() << "\n";
    std::cout << "bed_overlap  = ";
    if (a.bedOverlaps.empty()) {
        std::cout << "0";
    } else {
        for (size_t idx = 0; idx < a.bedOverlaps.size(); ++idx) {
            if (idx > 0) std::cout << ",";
            std::cout << a.bedOverlaps[idx];
        }
    }
    std::cout << "\n";
    std::cout << "crop_end     = " << a.cropEndStart << "," << a.cropEndEnd << "  # top,end\n";
    std::cout << "z_bed_flip   = " << (a.zBedFlip ? 1 : 0) << "\n";
    std::cout << "z_out_flip   = " << (a.zOutFlip ? 1 : 0) << "\n\n";

    std::cout << "[InputOutput]\n";
    std::cout << "output_dir  = " << a.outputDir << "\n";
    std::cout << "output_name = " << a.outputName << "\n\n";

    std::cout << "# ── Config sources ──\n";
    std::cout << "# global_ini  = " << a.configPath << (a.configLoaded ? " (loaded)" : " (not found)") << "\n";
    std::cout << "# algo_conf   = " << (a.algoConfPath.empty() ? "(none)" : a.algoConfPath) << "\n";
}

inline void ArgumentParser::printUsage() {
    std::cout << "SPECT Reconstruction Tool\n";
    std::cout << "Usage: SPECT-recon [options]\n\n";

    std::cout << "=== Quick Start ===\n";
    std::cout << "  SPECT-recon -df proj.bin -o recon_output   # uses single exe-local bin/matrix/* if present\n";
    std::cout << "  SPECT-recon -df proj.mat -loadCSR matrix.csrb\n";
    std::cout << "  SPECT-recon -df proj.bin -loadCSR matrix.csrb -detN 40,40\n";
    std::cout << "  SPECT-recon -df proj.mat -loadCSR matrix.csrb -alg bsrem\n";
    std::cout << "  SPECT-recon -df proj.mat -loadCSR matrix.csrb -alg admm -it 100\n\n";

    std::cout << "  Windows paths: use quotes or forward slashes\n";
    std::cout << "    -df \"D:\\path\\file.mat\"  or  -df D:/path/file.bin\n\n";

    std::cout << "=== Data I/O ===\n";
    std::cout << "  -df, -i <path>          Projection MAT or BIN file\n";
    std::cout << "  -dfList <p1,p2,...>      Multi-bed projection MAT/BIN list, sorted by BedN when present (max 5 beds)\n";
    std::cout << "  -loadCSR <path>          Pre-computed CSR binary\n";
    std::cout << "  -chunkCSRDir <dir>       Directory with metadata.bin + chunk_*.csrb\n";
    std::cout << "                           Default: config default_chunk_csr_dir, or the only exe-local bin/matrix/* candidate\n";
    std::cout << "  -matrixLayout <layout>   standard | compact-symmetric\n";
    std::cout << "  -compactMeta <path>      metadata.bin for compact-symmetric layout\n";
    std::cout << "  -compactDenomCache <mhd> Reuse/save compact global denominator cache; use separate files for PSF/no-PSF/FWHM/bed variants\n";
    std::cout << "  -saveCSR <path>          Save CSR binary after build\n";
    std::cout << "  -indexMat <file>         System matrix voxel index file\n";
    std::cout << "  -weightMat <file>        System matrix weight file\n";
    std::cout << "  -attMap <path>           Attenuation map (mu, mm^-1) for attenuation correction / Chang's AC\n";
    std::cout << "  -bProj <path>            Background/scatter projection MAT/BIN\n";
    std::cout << "  -bProjList <list>        Multi-bed background MAT/BIN list, sorted by BedN when present\n";
    std::cout << "  -prior <path>            Anatomical prior for Bowsher/PLS\n";
    std::cout << "  -fout <dir/name>         Output path prefix\n";
    std::cout << "  -o <dir>                 Output directory\n";
    std::cout << "  -n <name>                Output filename base\n\n";

    std::cout << "=== Algorithm ===\n";
    std::cout << "  -alg <name>              MLEM | BSREM | ADMM\n";
    std::cout << "                           Auto-loads config/algo_<name>.conf\n";
    std::cout << "  -algoConf <path>         Custom algorithm config file\n";
    std::cout << "  -it <N[:save]>           Iterations (default: from algo conf)\n";
    std::cout << "  -subsets <N>             Subsets (1=ML-EM, >1=OS-EM)\n";
    std::cout << "  -sparseBackend <b>       auto | cpu | ref | cuda\n\n";

    std::cout << "=== Algorithm: BSREM ===\n";
    std::cout << "  -bsrem <a0[,g]>          BSREM alpha0 and gamma\n";
    std::cout << "  -bsremCurv on|off        Curvature denominator D=S+beta*Dreg\n";
    std::cout << "  -bsremCurvFreeze on|off  Freeze curvature after warmup\n";
    std::cout << "  -bsremCurvWarmup <n>     Warmup iterations (default 2)\n";
    std::cout << "  -bsremCurvEps <v>        Stability epsilon (default 1e-6)\n";
    std::cout << "  -bsremKappa on|off       Kappa map beta_j=beta*kappa_j\n";
    std::cout << "  -bsremKappaEps <v>       Kappa epsilon (default 1e-6)\n";
    std::cout << "  -bsremKappaClip <min,max>  Kappa clip range (default 0.1,10)\n";
    std::cout << "  -bsremSoftMask on|off    Soft-mask penalty boosting\n";
    std::cout << "  -bsremSoftBand <vox>     Transition width (default 2)\n";
    std::cout << "  -bsremSoftLambda <v>     Outside penalty boost (default 100)\n\n";

    std::cout << "=== Algorithm: ADMM ===\n";
    std::cout << "  -admmRho <value>         Penalty parameter rho (default 1.0)\n";
    std::cout << "  -admmTvInner <n>         Chambolle-Pock inner iterations (default 20)\n";
    std::cout << "  -admmAdaptive on|off     Adaptive rho (default on, Boyd 2011)\n\n";

    std::cout << "=== Regularization ===\n";
    std::cout << "  -reg <type[,beta,delta]> TV | RDP | LogCosh | Tikhonov | MRP | NLM | ...\n";
    std::cout << "  -regBeta <beta>          Regularization weight\n";
    std::cout << "  -regDelta <delta>        Transition parameter\n\n";

    std::cout << "=== Filtering ===\n";
    std::cout << "  -butter <fc[,order]>     Butterworth low-pass (default: 0.35,2)\n";
    std::cout << "  -butterBackend <b>       auto | fftw | cufft\n";
    std::cout << "  -smooth <sigma>          Gaussian smoothing sigma (mm)\n";
    std::cout << "  -smoothKernel <n>        Gaussian kernel half-width\n";
    std::cout << "  -conv <params>           Legacy convolution filter\n\n";

    std::cout << "=== PSF ===\n";
    std::cout << "  -psf <sigma>             PSF sigma (mm), enables PSF modeling\n";
    std::cout << "  -psfMode <mode>          None | Operator | PostSmooth | ProjDomain\n";
    std::cout << "  -psfProj on|off          Per-angle 2D PSF in projection domain\n";
    std::cout << "\n";

    std::cout << "=== Geometry ===\n";
    std::cout << "  -dim <Nx,Ny,Nz>          Reconstruction grid (voxels)\n";
    std::cout << "  -vox <Vx,Vy,Vz>          Voxel size (mm)\n";
    std::cout << "  -ang <N>                 Projection angles\n";
    std::cout << "  -detN <Nz[,Nx]>          Detector channels\n\n";

    std::cout << "=== Multi-Bed ===\n";
    std::cout << "  -bedOverlap <b1,b2,b3,b4,b5> Fixed Bed1..Bed5 overlap table; slot b1=Bed1-Bed2\n";
    std::cout << "  -cropEnd <top[,end]>      Final output Z cropping after output flips\n";
    std::cout << "  -zbedflip <0|1>           Flip each dfList bed segment on output (default 1)\n";
    std::cout << "  -zoutflip <0|1>           Flip whole output volume on Z after per-bed flip (default 0)\n";
    std::cout << "  -ctac                     Enable iterative attenuation correction (uses -attMap; mu in mm^-1)\n";
    std::cout << "  -ctacModel <m>            line-integral (default; angle-dependent parallel-ray Int mu ds)\n";
    std::cout << "                            or per-voxel (legacy exp(-mu) approximation, not quantitative)\n";
    std::cout << "  -ctacRotSign <1|-1>       Rotation sign matching sysmat generator --voxel-rotation-sign (default 1)\n";
    std::cout << "  -ctacAutoMu <vol>         Build water mu-map from an early-iteration recon volume\n";
    std::cout << "                            (threshold + closing + hole fill; for activity-filled phantoms;\n";
    std::cout << "                            mutually exclusive with -attMap; generated map saved as *_automu)\n";
    std::cout << "  -ctacAutoMuThresh <f>     Relative threshold vs P99.9 activity (default 0.05)\n";
    std::cout << "  -ctacAutoMuValue <mu>     Body fill mu in mm^-1 (default 0.0152 = water @140keV narrow-beam;\n";
    std::cout << "                            consider ~0.012 broad-beam when scatter is uncorrected)\n\n";

    std::cout << "=== Masking ===\n";
    std::cout << "  -cyl <diam,height>        Cylindrical mask (mm)\n";
    std::cout << "  -cylc <cx,cy,cz>          Cylinder center (1-based voxel)\n";
    std::cout << "  -maskFromAtt <thresh>     Body mask from mu-map\n\n";

    std::cout << "=== Advanced ===\n";
    std::cout << "  -config <path>            Global config (default search: exe/config, exe, cwd/config, cwd)\n";
    std::cout << "  -dumpConf                 Print merged config and exit\n";
    std::cout << "  -Aprune <eps>             Prune system-matrix weights (default 1e-10)\n";
    std::cout << "  -Astream on|off           Stream build from disk (default off)\n";
    std::cout << "  -ApruneMask on|off        Prune columns outside mask (default on)\n";
    std::cout << "  -mapCSR on|off            Memory-mapped CSR loading (default off)\n";
    std::cout << "  -joe <n>                  Joseph-like smoothing passes (default 0)\n";
    std::cout << "  -vb <0|1|2>               Verbosity level (default 1)\n";
    std::cout << "  -th <N>                   Thread count\n";
    std::cout << "  -h, -help, --help         Show this help\n\n";

    std::cout << "  Algorithm-specific defaults are loaded beside the global config or from config/algo_<name>.conf.\n";
    std::cout << "  Override any value with CLI flags. Use -dumpConf to inspect.\n";
    std::cout << "  See: doc/USAGE_CN.md for full parameter reference.\n\n";
}

} // namespace cli
} // namespace spect
