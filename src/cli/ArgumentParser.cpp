// ArgumentParser implementation (moved from header, PR15 move-only).

#include "cli/ArgumentParser.h"

#include "cli/Version.h"
#include "core/AttenuationCoefficients.h"
#include "io/BadDetectorMask.h"
#include "io/DetZRowWeights.h"

namespace spect {
namespace cli {

namespace {

bool isPublicRegularizer(RegType type) {
    return type == RegType::None || type == RegType::TVIsotropic ||
           type == RegType::RDP;
}

} // namespace

std::filesystem::path ArgumentParser::executableDir(char** argv) {
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

std::string ArgumentParser::resolveDefaultConfigPath(char** argv) {
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

std::string
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

std::filesystem::path ArgumentParser::resolveChunkCSRDirPath(
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

std::vector<std::filesystem::path>
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

bool ArgumentParser::applyMatrixAutoDiscovery(ReconArgs& args,
                                                     char** argv,
                                                     std::string& error) {
    const std::filesystem::path exeDir = executableDir(argv);

    if (!args.chunkCSRDir.empty()) {
        args.chunkCSRDir = resolveChunkCSRDirPath(
            args.chunkCSRDir, exeDir, args.chunkCSRDirExplicit).string();
        return true;
    }

    const auto candidates = findExeLocalMatrixDirs(exeDir);
    if (candidates.empty())
        return true;
    if (candidates.size() == 1) {
        args.chunkCSRDir = candidates.front().string();
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

ArgumentParser::ParseResult ArgumentParser::parse(int argc, char** argv,
                                                  ReconArgs& args) {
  try {
    // 0. Pre-scan for -config, -opti, -algoConf
    std::string configPath = resolveDefaultConfigPath(argv);
    std::string preScanAlg;
    std::string preScanAlgoConf;
    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];
        if (key == "-config" && i + 1 < argc) {
            configPath = argv[++i];
            args.configExplicit = true;
        } else if (key == "-opti" && i + 1 < argc) {
            preScanAlg = toLower(std::string(argv[++i]));
        } else if (key == "-algoConf" && i + 1 < argc) {
            preScanAlgoConf = argv[++i];
        }
    }
    args.configPath = configPath;

    if (const char* cudaDeviceEnv = std::getenv("SPECT_CUDA_DEVICE")) {
        if (cudaDeviceEnv[0] != '\0') {
            args.cudaDeviceValue = cudaDeviceEnv;
            args.cudaDeviceRequested = true;
            args.cudaDeviceFromCli = false;
        }
    }

    // 1. Load global config if exists
    ConfigParser cfg;
    if (std::filesystem::exists(configPath)) {
        args.configFound = true;
        if (cfg.load(configPath)) {
            args.configLoaded = true;
            applyConfig(args, cfg);
        } else if (args.configExplicit) {
            std::cerr << "Error: Cannot load explicit config file '"
                      << configPath << "'.\n";
            return ParseResult::ExitFailure;
        }
    } else {
        args.configFound = false;
        if (args.configExplicit) {
            std::cerr << "Error: Explicit config file not found: '"
                      << configPath << "'.\n";
            return ParseResult::ExitFailure;
        }
    }

    // 2. Load algorithm config (algo_conf layer)
    //    Priority: explicit -algoConf > auto-detect from -opti > auto-detect from global INI algorithm
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
            }
        }
        if (!algName.empty()) {
            algoConfPath = resolveAlgoConfigPath(algName, configPath, argv);
        }
    }
    if (!algoConfPath.empty()) {
        args.algoConfPath = algoConfPath;
        if (!applyAlgoConf(args, algoConfPath))
            return ParseResult::ExitFailure;
    }

    // 3. Process command-line arguments (highest priority override)
    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];
        auto it = handlers_.find(key);
        if (it != handlers_.end()) {
            if (!it->second(i, argc, argv, args))
                return ParseResult::ExitFailure;
            if (args.helpRequested || args.versionRequested)
                return ParseResult::ExitSuccess;
        } else {
            std::cerr << "Unknown argument: " << key << "\n";
            return ParseResult::ExitFailure;
        }
    }

    // 4. Resolve matrix defaults after all config/CLI layers are known.
    std::string matrixDiscoveryError;
    if (!applyMatrixAutoDiscovery(args, argv, matrixDiscoveryError)) {
        std::cerr << "Error: " << matrixDiscoveryError << "\n";
        return ParseResult::ExitFailure;
    }

    // 5. Normalize derived defaults, then handle -dumpConf and validation.
    normalizeArgs(args);

    // 6. Handle -dumpConf: print merged config and exit
    if (args.dumpConf) {
        dumpMergedConfig(args);
        return ParseResult::ExitSuccess;
    }

    // 7. Validate
    if (!validateArgs(args))
        return ParseResult::ExitFailure;
    if (!args.projPaths.empty()) {
        args.bedCount = std::max(1, static_cast<int>(args.projPaths.size()));
    }
    return ParseResult::Success;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return ParseResult::ExitFailure;
  }
}

void ArgumentParser::bindHandlerAliases(
    const Handler& handler, std::initializer_list<const char*> flags) {
    for (const char* flag : flags)
        handlers_[flag] = handler;
}

void ArgumentParser::registerHandlers() {
    handlers_.clear();
    auto rejectCliOption = [](const char* message) {
        return [message](int& i, int argc, char** argv, ReconArgs& a) {
            (void)i;
            (void)argc;
            (void)argv;
            (void)a;
            std::cerr << "Error: " << message << "\n";
            return false;
        };
    };

    auto dataHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        // Treat -i/-input as aliases for -df (path normalization/drive-letter
        // handling is keyed on the arg name inside getRequiredArg).
        a.projPath = ArgumentParser::getRequiredArg(i, argc, argv, "-df");
        return true;
    };
    bindHandlerAliases(dataHandler, {"-df", "-i", "-input"});

    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.projPaths = spect::core::sortPathsByBedOrder(
            splitPathList(ArgumentParser::getRequiredArg(i, argc, argv, "-dfList")));
        return true;
    }, {"-dfList"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.chunkCSRDir = ArgumentParser::getRequiredArg(i, argc, argv, "-chunkCSRDir");
        a.chunkCSRDirExplicit = true;
        return true;
    }, {"-chunkCSRDir"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.compactDenomCachePath =
            ArgumentParser::getRequiredArg(i, argc, argv, "-compactDenomCache");
        return true;
    }, {"-compactDenomCache"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.ctacFactorCacheDir =
            ArgumentParser::getRequiredArg(i, argc, argv, "-ctacFactorCacheDir");
        return true;
    }, {"-ctacFactorCacheDir"});

    // --- Dimensions & Spacing ---
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseDimensions(
            ArgumentParser::getRequiredArg(i, argc, argv, "-dim"), a);
    }, {"-dim"});
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
    bindHandlerAliases(voxelHandler, {"-vox", "-spacing"});

    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
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
    }, {"-detN"});

    auto anglesHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        a.numAngles = std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, k));
        return true;
    };
    bindHandlerAliases(anglesHandler, {"-ang", "-angles"});

    // --- Algorithm & Iterations ---
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseIterations(
            ArgumentParser::getRequiredArg(i, argc, argv, "-it"), a);
    }, {"-it"});
    auto threadsHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        a.numThreads = std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, k));
        return true;
    };
    bindHandlerAliases(threadsHandler, {"-th"});
    bindHandlerAliases(threadsHandler, {"-threads"});
    bindHandlerAliases(threadsHandler, {"-t"});
    bindHandlerAliases(
        rejectCliOption("-sparseBackend was removed; the sparse backend is fixed to CUDA."),
        {"-sparseBackend", "-spBackend"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.cudaDeviceValue = ArgumentParser::getRequiredArg(i, argc, argv, "-cudaDevice");
        a.cudaDeviceRequested = true;
        a.cudaDeviceFromCli = true;
        return true;
    }, {"-cudaDevice"});

    bindHandlerAliases(
        rejectCliOption("-subsets was removed; the compact-symmetric layout is always single-subset (MLEM-style full update)."),
        {"-subsets"});

    auto algHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        const std::string val =
            toLower(ArgumentParser::getRequiredArg(i, argc, argv, k));
        if (val == "mlem") {
            a.algorithm = ReconArgs::Algorithm::MLEM;
        } else if (val == "bsrem") {
            a.algorithm = ReconArgs::Algorithm::BSREM;
        } else {
            std::cerr << "Error: Unknown optimisation algorithm '" << val << "'.\n";
            return false;
        }
        return true;
    };
    bindHandlerAliases(algHandler, {"-opti"});
    bindHandlerAliases(
        rejectCliOption("-alg was removed; use -opti mlem|bsrem."),
        {"-alg"});

    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        auto parts = split(ArgumentParser::getRequiredArg(i, argc, argv, "-bsrem"), ',');
        if (parts.empty())
            return false;
        a.bsrem_alpha0 = (float)std::stod(parts[0]);
        if (parts.size() > 1)
            a.bsrem_gamma = (float)std::stod(parts[1]);
        return true;
    }, {"-bsrem"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.bsremUseCurvDenom);
    }, {"-bsremCurv"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.bsremFreezeCurv);
    }, {"-bsremCurvFreeze"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.bsremCurvWarmupIters = std::stoi(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bsremCurvWarmup"));
        return true;
    }, {"-bsremCurvWarmup"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.bsremCurvEps = std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bsremCurvEps"));
        return true;
    }, {"-bsremCurvEps"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.bsremUseKappaMap);
    }, {"-bsremKappa"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.bsremKappaEps = std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bsremKappaEps"));
        return true;
    }, {"-bsremKappaEps"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        auto parts = split(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bsremKappaClip"), ',');
        if (parts.size() != 2) {
            std::cerr << "Error: -bsremKappaClip requires min,max.\n";
            return false;
        }
        a.bsremKappaClipMin = std::stod(parts[0]);
        a.bsremKappaClipMax = std::stod(parts[1]);
        return true;
    }, {"-bsremKappaClip"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.bsremSoftMask);
    }, {"-bsremSoftMask"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.bsremSoftMaskBandVox = std::stoi(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bsremSoftBand"));
        return true;
    }, {"-bsremSoftBand"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.bsremSoftMaskLambdaOut = std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bsremSoftLambda"));
        return true;
    }, {"-bsremSoftLambda"});

    // --- IO & Verbosity ---
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOutputPath(
            ArgumentParser::getRequiredArg(i, argc, argv, "-fout"), a);
    }, {"-fout"});
    auto outDirHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        a.outputDir = ArgumentParser::getRequiredArg(i, argc, argv, k);
        return true;
    };
    bindHandlerAliases(outDirHandler, {"-o"});
    bindHandlerAliases(outDirHandler, {"-output"});
    auto outNameHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        a.outputName = ArgumentParser::getRequiredArg(i, argc, argv, k);
        return true;
    };
    bindHandlerAliases(outNameHandler, {"-n"});
    bindHandlerAliases(outNameHandler, {"-name"});
    auto verbosityHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        a.verbosity = std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, k));
        return true;
    };
    bindHandlerAliases(verbosityHandler, {"-vb"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(
            i, argc, argv, a.projectionResidual);
    }, {"-projectionResidual"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        (void)i;
        (void)argc;
        (void)argv;
        printVersion();
        a.versionRequested = true;
        return true;
    }, {"-v", "--version"});

    // --- Filtering & PSF ---
    bindHandlerAliases(
        rejectCliOption("-conv was removed; use image-domain -psf [fwhm_mm[,N]]."),
        {"-conv"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        std::string value;
        if (i + 1 < argc) {
            const std::string next = argv[i + 1];
            if (!next.empty() && next[0] != '-')
                value = argv[++i];
        }
        const auto parts = value.empty() ? std::vector<std::string>{}
                                         : split(value, ',');
        if (parts.size() > 2) {
            std::cerr << "Error: -psf expects fwhm_mm[,N].\n";
            return false;
        }
        const double fwhm_mm =
            parts.empty() || parts[0].empty() ? a.psf.fwhm_mm : std::stod(parts[0]);
        if (fwhm_mm <= 0.0) {
            std::cerr << "Error: -psf FWHM must be positive.\n";
            return false;
        }
        if (parts.size() > 1 && !parts[1].empty())
            a.psf.min_sigmas = std::stod(parts[1]);
        a.usePSF = true;
        a.psf.fwhm_mm = fwhm_mm;
        if (!a.psfModeExplicit)
            a.psf.mode = PSFMode::Operator;
        return true;
    }, {"-psf"});
    bindHandlerAliases(
        rejectCliOption("-psfProj was removed; use image-domain -psf Operator."),
        {"-psfProj"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string val =
            toLower(ArgumentParser::getRequiredArg(i, argc, argv, "-psfMode"));
        a.psfModeExplicit = true;
        if (val == "operator")
            a.psf.mode = PSFMode::Operator;
        else if (val == "none")
            a.psf.mode = PSFMode::None;
        else {
            std::cerr << "Error: -psfMode expects None|Operator; got '" << val << "'.\n";
            return false;
        }
        return true;
    }, {"-psfMode"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.useSmooth = true;
        a.smoothSigma =
            (float)std::stod(ArgumentParser::getRequiredArg(i, argc, argv, "-smooth"));
        return true;
    }, {"-smooth"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.smoothKernel =
            std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, "-smoothKernel"));
        return true;
    }, {"-smoothKernel"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        auto parts =
            split(ArgumentParser::getRequiredArg(i, argc, argv, "-butter"), ',');
        if (parts.empty())
            return false;
        a.useButter = true;
        a.butterCutoff = std::stod(parts[0]);
        if (parts.size() > 1)
            a.butterOrder = std::stoi(parts[1]);
        return true;
    }, {"-butter"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.butterOrder =
            std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, "-butterOrder"));
        return true;
    }, {"-butterOrder"});
    bindHandlerAliases(
        rejectCliOption("-butterBackend was removed; Butterworth uses cuFFT."),
        {"-butterBackend", "-bwBackend"});

    // --- Regularization ---
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        auto parts = split(ArgumentParser::getRequiredArg(i, argc, argv, "-reg"), ',');
        if (parts.empty())
            return false;
        std::string typeStr = parts[0];
        const std::string typeLower = toLower(typeStr);
        if (typeLower == "none")
            typeStr = "None";
        else if (typeLower == "tv")
            typeStr = "TVIsotropic";
        else if (typeLower == "rdp")
            typeStr = "RDP";

        const RegType t = stringToRegType(typeStr);
        if (t == RegType::Unknown) {
            std::cerr << "Error: Unknown regularization type '" << parts[0] << "'.\n";
            return false;
        }
        if (!isPublicRegularizer(t)) {
            std::cerr << "Error: Only TV and RDP regularization are supported.\n";
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
            if (t == RegType::RDP) {
                if (parts.size() >= 3)
                    a.reg.gamma = std::stod(parts[2]);
                if (parts.size() >= 4)
                    a.reg.delta = std::stod(parts[3]);
            } else if (t == RegType::TVIsotropic) {
                if (parts.size() >= 3)
                    a.reg.delta = std::stod(parts[2]);
            }
        }
        return true;
    }, {"-reg"});

    auto regValHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        a.useReg = true;
        const std::string key = argv[i];
        double val = std::stod(ArgumentParser::getRequiredArg(i, argc, argv, key));
        if (key == "-regBeta")
            a.reg.beta = val;
        else if (key == "-regDelta")
            a.reg.delta = val;
        return true;
    };
    bindHandlerAliases(regValHandler, {"-regBeta"});
    bindHandlerAliases(regValHandler, {"-regDelta"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.regCtPath = ArgumentParser::getRequiredArg(i, argc, argv, "-regCT");
        return true;
    }, {"-regCT"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string key = argv[i];
        const double value = std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv, key));
        if (key == "-regCtSigma")
            a.reg.ct_sigma = value;
        else
            a.reg.ct_weight_floor = value;
        return true;
    }, {"-regCtSigma", "-regCtFloor"});
    bindHandlerAliases(
        rejectCliOption("-regPower was removed; use -reg TV or -reg RDP."),
        {"-regPower"});
    bindHandlerAliases(
        rejectCliOption("-regSigma was removed; use -reg TV or -reg RDP."),
        {"-regSigma"});

    const Handler removedRegOption =
        rejectCliOption("this regularization option was removed; use -reg TV or -reg RDP.");
    bindHandlerAliases(removedRegOption,
                       {"-regKernel", "-regIter", "-regSubsets",
                        "-regUpdateFreq", "-regUpdateStart", "-regUpdateEnd",
                        "-regUpdateInterval", "-regUpdateType"});

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
    bindHandlerAliases(cylHandler, {"-cyl"});
    bindHandlerAliases(cylHandler, {"-cylMask"});

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
    bindHandlerAliases(cylCenterHandler, {"-cylc"});
    bindHandlerAliases(cylCenterHandler, {"-cylCenter"});

    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.maskFromAttThreshold = (float)std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv, "-maskFromAtt"));
        return true;
    }, {"-maskFromAtt"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        auto parts =
            split(ArgumentParser::getRequiredArg(i, argc, argv, "-bedOverlap"), ',');
        a.bedOverlaps.clear();
        for (auto& p : parts) {
            if (!p.empty())
                a.bedOverlaps.push_back(std::stoi(p));
        }
        return true;
    }, {"-bedOverlap"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.bedCropMethod = toLower(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bedCropMethod"));
        return true;
    }, {"-bedCropMethod"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv,
                                                  a.bedSeamPostprocess);
    }, {"-bedSeamPostprocess"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.detZEdgeCensor = std::stoi(
            ArgumentParser::getRequiredArg(i, argc, argv, "-detZEdgeCensor"));
        return true;
    }, {"-detZEdgeCensor"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.detZRowWeightsFile = ArgumentParser::getRequiredArg(
            i, argc, argv, "-detZRowWeightsFile");
        return true;
    }, {"-detZRowWeightsFile"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.normFactorFile = ArgumentParser::getRequiredArg(
            i, argc, argv, "-normFactorFile");
        return true;
    }, {"-normFactorFile"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.badDetectorFile = ArgumentParser::getRequiredArg(
            i, argc, argv, "-badDetectorFile");
        a.badDetectorFileExplicit = true;
        return true;
    }, {"-badDetectorFile"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.bedObjectProfile = toLower(ArgumentParser::getRequiredArg(
            i, argc, argv, "-bedObjectProfile"));
        return true;
    }, {"-bedObjectProfile"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(
            i, argc, argv, a.bedDerenzoPostprocess);
    }, {"-derenzoPostprocess"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.bedDerenzoAxialStrength = std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv,
                                           "-derenzoAxialStrength"));
        return true;
    }, {"-derenzoAxialStrength"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(
            i, argc, argv, a.bedDerenzoAdhesion);
    }, {"-derenzoAdhesion"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.bedDerenzoAdhesionInner = std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv,
                                           "-derenzoAdhesionInner"));
        return true;
    }, {"-derenzoAdhesionInner"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.bedDerenzoAdhesionOuter = std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv,
                                           "-derenzoAdhesionOuter"));
        return true;
    }, {"-derenzoAdhesionOuter"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.bedDerenzoCoreDamp = std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv, "-derenzoCoreDamp"));
        return true;
    }, {"-derenzoCoreDamp"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
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
    }, {"-cropEnd"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.zBedFlip);
    }, {"-zbedflip"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.zOutFlip);
    }, {"-zoutflip"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.xOutFlip);
    }, {"-xoutflip"});

    // --- Physics & Correction ---
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        (void)i;
        (void)argc;
        (void)argv;
        a.doCTAC = true;
        return true;
    }, {"-ctac"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        std::string value = ArgumentParser::getRequiredArg(i, argc, argv, "-ctacModel");
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (value == "line-integral" || value == "lineintegral" || value == "ray") {
            a.ctacModel = "line-integral";
        } else {
            std::cerr << "Error: -ctacModel expects line-integral, got '"
                      << value << "'\n";
            return false;
        }
        return true;
    }, {"-ctacModel"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.ctacAutoMuFrom = ArgumentParser::getRequiredArg(i, argc, argv, "-ctacAutoMu");
        return true;
    }, {"-ctacAutoMu"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.ctacAutoMuThresh =
            std::stod(ArgumentParser::getRequiredArg(i, argc, argv, "-ctacAutoMuThresh"));
        if (!(a.ctacAutoMuThresh > 0.0) || a.ctacAutoMuThresh >= 1.0) {
            std::cerr << "Error: -ctacAutoMuThresh must be in (0, 1).\n";
            return false;
        }
        return true;
    }, {"-ctacAutoMuThresh"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.ctacAutoMuValue =
            std::stod(ArgumentParser::getRequiredArg(i, argc, argv, "-ctacAutoMuValue"));
        a.ctacAutoMuValueExplicit = true;
        a.ctacAutoMuValueSource = "manual";
        if (!(a.ctacAutoMuValue > 0.0)) {
            std::cerr << "Error: -ctacAutoMuValue (mu per mm) must be positive.\n";
            return false;
        }
        return true;
    }, {"-ctacAutoMuValue"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
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
    }, {"-ctacRotSign"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.attMapPath = ArgumentParser::getRequiredArg(i, argc, argv, "-attMap");
        return true;
    }, {"-attMap"});
    bindHandlerAliases(
        rejectCliOption("-prior was removed with anatomical regularizers; use TV or RDP."),
        {"-prior"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.bProjPath = ArgumentParser::getRequiredArg(i, argc, argv, "-bProj");
        return true;
    }, {"-bProj"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.bProjPaths = spect::core::sortPathsByBedOrder(
            splitPathList(ArgumentParser::getRequiredArg(i, argc, argv, "-bProjList")));
        return true;
    }, {"-bProjList"});

    // --- Performance & Misc ---
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.weightPruneEps =
            std::stod(ArgumentParser::getRequiredArg(i, argc, argv, "-Aprune"));
        return true;
    }, {"-Aprune"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.Astream);
    }, {"-Astream"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.ApruneMask);
    }, {"-ApruneMask"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.josephPasses =
            std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, "-joe"));
        return true;
    }, {"-joe"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        (void)argc;
        (void)argv;
        (void)a;
        ++i;
        return true; // Handled by pre-scan
    }, {"-config"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.algoConfPath = ArgumentParser::getRequiredArg(i, argc, argv, "-algoConf");
        return true; // Loading handled in parse() after CLI pass
    }, {"-algoConf"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        (void)i;
        (void)argc;
        (void)argv;
        a.dumpConf = true;
        return true;
    }, {"-dumpConf"});
    auto helpHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        (void)i;
        (void)argc;
        (void)argv;
        ArgumentParser::printUsage();
        a.helpRequested = true;
        return true;
    };
    bindHandlerAliases(helpHandler, {"-h", "-help", "--help"});
}

void ArgumentParser::applyConfig(ReconArgs& a, const ConfigParser& cfg) {
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
    a.verbosity = cfg.getInt("Basic", "verbosity", a.verbosity);
    a.projectionResidual = cfg.getBool(
        "Diagnostics", "projection_residual", a.projectionResidual);

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
    a.badDetectorFile = cfg.getString(
        "Detector", "bad_detector_file", a.badDetectorFile);

    // Recon
    std::string alg = cfg.getString("Reconstruction", "algorithm");
    if (alg == "BSREM")
        a.algorithm = ReconArgs::Algorithm::BSREM;
    else if (alg == "MLEM")
        a.algorithm = ReconArgs::Algorithm::MLEM;

    std::string iters = cfg.getString("Reconstruction", "iterations");
    if (!iters.empty()) {
        auto it = split(iters, ':');
        if (!it.empty())
            a.numIters = std::stoi(it[0]);
        if (it.size() > 1)
            a.saveInterval = std::stoi(it[1]);
    }
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

    if (cfg.hasKey("CTAC", "auto_mu_nuclide"))
        a.ctacAutoMuNuclide =
            cfg.getString("CTAC", "auto_mu_nuclide", a.ctacAutoMuNuclide);
    if (cfg.hasKey("CTAC", "auto_mu_value")) {
        a.ctacAutoMuValue =
            cfg.getDouble("CTAC", "auto_mu_value", a.ctacAutoMuValue);
        a.ctacAutoMuValueExplicit = true;
        a.ctacAutoMuValueSource = "manual";
    }

    // IO
    a.outputDir = cfg.getString("InputOutput", "output_dir", a.outputDir);
    a.outputName = cfg.getString("InputOutput", "output_name", a.outputName);

    // System Matrix
    a.weightPruneEps =
        cfg.getDouble("SystemMatrix", "prune_epsilon", a.weightPruneEps);
    a.Astream = cfg.getBool("SystemMatrix", "stream_build", a.Astream);
    a.ApruneMask = cfg.getBool("SystemMatrix", "prune_mask", a.ApruneMask);
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
            throw std::runtime_error("Unknown regularization type '" + regT +
                                     "' in config file.");
        }
        if (!isPublicRegularizer(a.reg.type)) {
            throw std::runtime_error(
                "Only TV and RDP regularization are supported in config.");
        }
        a.reg.beta = cfg.getDouble("Regularization", "beta", a.reg.beta);
        double eps = cfg.getDouble("Regularization", "epsilon", 1e-6);
        a.reg.tv_eps = eps;
        a.reg.delta = cfg.getDouble("Regularization", "delta", a.reg.delta);
        a.reg.gamma = cfg.getDouble("Regularization", "gamma", a.reg.gamma);
        a.regCtPath = normalizePath(
            cfg.getString("Regularization", "ct_image", a.regCtPath));
        a.reg.ct_sigma =
            cfg.getDouble("Regularization", "ct_sigma", a.reg.ct_sigma);
        a.reg.ct_weight_floor = cfg.getDouble(
            "Regularization", "ct_weight_floor", a.reg.ct_weight_floor);
    }

    // PSF
    std::string psfModeStr = cfg.getString("PSF", "mode");
    if (!psfModeStr.empty()) {
        std::string psfModeLower = toLower(psfModeStr);
        if (psfModeLower == "none")
            a.psf.mode = PSFMode::None;
        else if (psfModeLower == "operator")
            a.psf.mode = PSFMode::Operator;
        else {
            throw std::runtime_error(
                "PSF mode in config must be None or Operator; got '" +
                psfModeStr + "'.");
        }
        // If mode is set to something other than None, enable PSF
        if (a.psf.mode != PSFMode::None)
            a.usePSF = true;
    }
    a.psf.fwhm_mm = cfg.getDouble("PSF", "fwhm", a.psf.fwhm_mm);
    a.psf.z_blur = cfg.getBool("PSF", "z_blur", a.psf.z_blur);

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
        a.bedCropMethod = toLower(cfg.getString(
            "MultiBed", "crop_method", a.bedCropMethod));
        a.bedSeamPostprocess = cfg.getBool(
            "MultiBed", "seam_postprocess", a.bedSeamPostprocess);
        a.detZEdgeCensor = cfg.getInt(
            "MultiBed", "detz_edge_censor", a.detZEdgeCensor);
        a.detZRowWeightsFile = cfg.getString(
            "MultiBed", "detz_row_weights_file", a.detZRowWeightsFile);
        a.normFactorFile = cfg.getString(
            "MultiBed", "normalization_factor_file", a.normFactorFile);
        a.bedObjectProfile = toLower(cfg.getString(
            "MultiBed", "object_profile", a.bedObjectProfile));
        a.bedDerenzoPostprocess = cfg.getBool(
            "MultiBed", "derenzo_postprocess", a.bedDerenzoPostprocess);
        a.bedDerenzoAxialStrength = cfg.getDouble(
            "MultiBed", "derenzo_axial_strength",
            a.bedDerenzoAxialStrength);
        a.bedDerenzoAdhesion = cfg.getBool(
            "MultiBed", "derenzo_adhesion", a.bedDerenzoAdhesion);
        a.bedDerenzoAdhesionInner = cfg.getDouble(
            "MultiBed", "derenzo_adhesion_inner", a.bedDerenzoAdhesionInner);
        a.bedDerenzoAdhesionOuter = cfg.getDouble(
            "MultiBed", "derenzo_adhesion_outer", a.bedDerenzoAdhesionOuter);
        a.bedDerenzoCoreDamp = cfg.getDouble(
            "MultiBed", "derenzo_core_damp", a.bedDerenzoCoreDamp);
        a.bedPhantomCloseStrength = cfg.getDouble(
            "MultiBed", "phantom_close_strength", a.bedPhantomCloseStrength);
        a.bedEpguanCloseStrength = cfg.getDouble(
            "MultiBed", "epguan_close_strength", a.bedEpguanCloseStrength);
        a.bedEpguanWallMaxPreserve = cfg.getBool(
            "MultiBed", "epguan_wall_max_preserve",
            a.bedEpguanWallMaxPreserve);
        a.bedEpguanSeamEdgeStrength = cfg.getDouble(
            "MultiBed", "epguan_seam_edge_strength",
            a.bedEpguanSeamEdgeStrength);
        a.bedEpguanSeamFeatherSlices = cfg.getInt(
            "MultiBed", "epguan_seam_feather_slices",
            a.bedEpguanSeamFeatherSlices);
        a.bedEpguanBackgroundFilter = cfg.getBool(
            "MultiBed", "epguan_background_filter",
            a.bedEpguanBackgroundFilter);
        a.bedEpguanBackgroundRetain = cfg.getDouble(
            "MultiBed", "epguan_background_retain",
            a.bedEpguanBackgroundRetain);
        a.bedEpguanSupportThreshold = cfg.getDouble(
            "MultiBed", "epguan_support_threshold",
            a.bedEpguanSupportThreshold);
        a.bedEpguanSupportMarginMm = cfg.getDouble(
            "MultiBed", "epguan_support_margin_mm",
            a.bedEpguanSupportMarginMm);
        a.bedZimuCloseStrength = cfg.getDouble(
            "MultiBed", "zimu_close_strength", a.bedZimuCloseStrength);
        a.bedMouseSeamStrength = cfg.getDouble(
            "MultiBed", "mouse_seam_strength", a.bedMouseSeamStrength);
        a.bedSeamHalfWidth = cfg.getInt(
            "MultiBed", "seam_half_width", a.bedSeamHalfWidth);
        a.fovRadiusMm = cfg.getDouble("MultiBed", "fov_radius_mm",
                                      a.fovRadiusMm);
        a.analysisFovRadiusMm = cfg.getDouble(
            "MultiBed", "analysis_fov_radius_mm", a.analysisFovRadiusMm);
        a.zBedFlip = cfg.getBool("MultiBed", "z_bed_flip", a.zBedFlip);
        a.zOutFlip = cfg.getBool("MultiBed", "z_out_flip", a.zOutFlip);
        a.xOutFlip = cfg.getBool("MultiBed", "x_out_flip", a.xOutFlip);
    }
}

bool ArgumentParser::applyAlgoConf(ReconArgs& a, const std::string& path) {
    ConfigParser cfg;
    if (!cfg.load(path)) {
        std::cerr << "Error: Cannot load algorithm config '" << path << "'.\n";
        return false;
    }

    // [Algorithm] section
    std::string alg = cfg.getString("Algorithm", "name");
    if (!alg.empty()) {
        std::string algLower = toLower(alg);
        if (algLower == "mlem")
            a.algorithm = ReconArgs::Algorithm::MLEM;
        else if (algLower == "bsrem")
            a.algorithm = ReconArgs::Algorithm::BSREM;
    }
    std::string iters = cfg.getString("Algorithm", "iterations");
    if (!iters.empty()) {
        auto it = split(iters, ':');
        if (!it.empty())
            a.numIters = std::stoi(it[0]);
        if (it.size() > 1)
            a.saveInterval = std::stoi(it[1]);
    }

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
        if (rt != RegType::Unknown && !isPublicRegularizer(rt)) {
            throw std::runtime_error(
                "Only TV and RDP regularization are supported in algo config.");
        }
        if (rt != RegType::Unknown && rt != RegType::None) {
            a.useReg = true;
            a.reg.type = rt;
        }
    }
    if (cfg.hasKey("Regularization", "beta"))
        a.reg.beta = cfg.getDouble("Regularization", "beta", a.reg.beta);
    if (cfg.hasKey("Regularization", "delta"))
        a.reg.delta = cfg.getDouble("Regularization", "delta", a.reg.delta);
    if (cfg.hasKey("Regularization", "gamma"))
        a.reg.gamma = cfg.getDouble("Regularization", "gamma", a.reg.gamma);
    if (cfg.hasKey("Regularization", "epsilon")) {
        double eps = cfg.getDouble("Regularization", "epsilon", 1e-6);
        a.reg.tv_eps = eps;
    }
    if (cfg.hasKey("Regularization", "ct_image"))
        a.regCtPath = normalizePath(
            cfg.getString("Regularization", "ct_image", a.regCtPath));
    if (cfg.hasKey("Regularization", "ct_sigma"))
        a.reg.ct_sigma =
            cfg.getDouble("Regularization", "ct_sigma", a.reg.ct_sigma);
    if (cfg.hasKey("Regularization", "ct_weight_floor"))
        a.reg.ct_weight_floor = cfg.getDouble(
            "Regularization", "ct_weight_floor", a.reg.ct_weight_floor);

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

    return true;
}

std::string ArgumentParser::getRequiredArg(int& i, int argc, char** argv,
                                                   const std::string& argName) {
    if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for argument '" + argName + "'");
    }
    std::string value = argv[++i];
    
    // Check if this is a path argument
    bool isPathArg =
        (argName == "-df" || argName == "-i" || argName == "-input" ||
         argName == "-dfList" || argName == "-bProjList" ||
         argName == "-fout" ||
         argName == "-chunkCSRDir" ||
         argName == "-attMap" || argName == "-regCT" || argName == "-bProj" ||
         argName == "-detZRowWeightsFile" || argName == "-normFactorFile" ||
         argName == "-badDetectorFile" ||
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

bool ArgumentParser::parseOnOffOptional(int& i, int argc, char** argv, bool& out) {
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

bool ArgumentParser::parseDimensions(const std::string& value, ReconArgs& args) {
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

bool ArgumentParser::parseIterations(const std::string& value, ReconArgs& args) {
    auto iterations = split(value, ':');
    if (iterations.empty())
        return false;
    args.numIters = std::stoi(iterations[0]);
    if (iterations.size() > 1)
        args.saveInterval = std::stoi(iterations[1]);
    return true;
}

void ArgumentParser::normalizeArgs(ReconArgs& args) {
    if (!args.ctacAutoMuValueExplicit) {
        double mu = 0.0;
        if (spect::core::waterMuPerMMForNuclide(args.ctacAutoMuNuclide, mu)) {
            args.ctacAutoMuValue = mu;
            if (const auto *preset =
                    spect::core::waterAttenuationPreset(args.ctacAutoMuNuclide)) {
                args.ctacAutoMuValueSource = preset->key;
            } else {
                args.ctacAutoMuValueSource = args.ctacAutoMuNuclide;
            }
        }
    }
    if (!args.badDetectorFile.empty() && !args.badDetectorFileExplicit) {
        const std::filesystem::path raw(args.badDetectorFile);
        if (!raw.is_absolute()) {
            const std::filesystem::path configDir =
                std::filesystem::path(args.configPath).parent_path();
            if (!configDir.empty())
                args.badDetectorFile =
                    (configDir / raw).lexically_normal().string();
        }
    }
}

bool ArgumentParser::parseOutputPath(const std::string& path, ReconArgs& args) {
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

bool ArgumentParser::validateIoArgs(const ReconArgs& args) {
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
    if (args.projPaths.size() > static_cast<size_t>(spect::core::kMaxBeds)) {
        std::cerr << "Error: -dfList supports at most "
                  << spect::core::kMaxBeds << " beds.\n";
        return false;
    }
    if (args.bProjPaths.size() > static_cast<size_t>(spect::core::kMaxBeds)) {
        std::cerr << "Error: -bProjList supports at most "
                  << spect::core::kMaxBeds << " beds.\n";
        return false;
    }
    if (args.bedOverlaps.size() > static_cast<size_t>(spect::core::kMaxBeds)) {
        std::cerr << "Error: -bedOverlap supports at most "
                  << spect::core::kMaxBeds << " fixed bed slots.\n";
        return false;
    }
    if (!std::isfinite(args.fovRadiusMm) || args.fovRadiusMm < 0.0) {
        std::cerr << "Error: [MultiBed] fov_radius_mm must be finite and >= 0.\n";
        return false;
    }
    if (!std::isfinite(args.analysisFovRadiusMm) ||
        args.analysisFovRadiusMm < 0.0) {
        std::cerr << "Error: [MultiBed] analysis_fov_radius_mm must be finite and >= 0.\n";
        return false;
    }
    const bool weightedOutputStitch =
        args.bedCropMethod == "sigmoid" ||
        args.bedCropMethod == "center66_cosine10";
    if (args.bedCropMethod != "direct" && !weightedOutputStitch) {
        std::cerr << "Error: [MultiBed] crop_method must be direct, sigmoid, "
                     "or center66_cosine10.\n";
        return false;
    }
    if (args.bedSeamPostprocess && !weightedOutputStitch) {
        std::cerr << "Error: [MultiBed] seam_postprocess requires weighted "
                     "output stitching.\n";
        return false;
    }
    if (args.bedObjectProfile != "auto" &&
        args.bedObjectProfile != "mouse" &&
        args.bedObjectProfile != "phantom" &&
        args.bedObjectProfile != "epguan" &&
        args.bedObjectProfile != "zimu" &&
        args.bedObjectProfile != "derenzo") {
        std::cerr << "Error: [MultiBed] object_profile must be auto, mouse, "
                     "phantom, epguan, zimu, or derenzo.\n";
        return false;
    }
    if (!std::isfinite(args.bedPhantomCloseStrength) ||
        args.bedPhantomCloseStrength < 0.0 ||
        args.bedPhantomCloseStrength > 1.0 ||
        !std::isfinite(args.bedEpguanCloseStrength) ||
        args.bedEpguanCloseStrength < 0.0 ||
        args.bedEpguanCloseStrength > 1.0 ||
        !std::isfinite(args.bedZimuCloseStrength) ||
        args.bedZimuCloseStrength < 0.0 ||
        args.bedZimuCloseStrength > 1.0 ||
        !std::isfinite(args.bedMouseSeamStrength) ||
        args.bedMouseSeamStrength < 0.0 ||
        args.bedMouseSeamStrength > 1.0) {
        std::cerr << "Error: [MultiBed] phantom_close_strength, "
                     "epguan_close_strength, zimu_close_strength, and "
                     "mouse_seam_strength must be "
                     "finite and in [0,1].\n";
        return false;
    }
    if (args.bedMouseSeamStrength > 0.0 &&
        args.bedCropMethod != "sigmoid") {
        std::cerr << "Error: [MultiBed] mouse_seam_strength > 0 requires "
                     "crop_method=sigmoid.\n";
        return false;
    }
    if (!std::isfinite(args.bedDerenzoAxialStrength) ||
        args.bedDerenzoAxialStrength < 0.0 ||
        args.bedDerenzoAxialStrength > 1.0) {
        std::cerr << "Error: [MultiBed] derenzo_axial_strength must be "
                     "finite and in [0,1].\n";
        return false;
    }
    const auto inUnitRange = [](double value) {
        return std::isfinite(value) && value >= 0.0 && value <= 1.0;
    };
    if (!inUnitRange(args.bedDerenzoAdhesionInner) ||
        !inUnitRange(args.bedDerenzoAdhesionOuter) ||
        args.bedDerenzoAdhesionInner > args.bedDerenzoAdhesionOuter) {
        std::cerr << "Error: [MultiBed] derenzo_adhesion_inner/_outer must be "
                     "finite, in [0,1] and inner <= outer.\n";
        return false;
    }
    if (!inUnitRange(args.bedDerenzoCoreDamp)) {
        std::cerr << "Error: [MultiBed] derenzo_core_damp must be finite and "
                     "in [0,1].\n";
        return false;
    }
    if (!std::isfinite(args.bedEpguanSeamEdgeStrength) ||
        args.bedEpguanSeamEdgeStrength < 0.0 ||
        args.bedEpguanSeamEdgeStrength > 1.0 ||
        args.bedEpguanSeamFeatherSlices < 0) {
        std::cerr << "Error: [MultiBed] epguan_seam_edge_strength must be "
                     "finite and in [0,1], and epguan_seam_feather_slices "
                     "must be >= 0.\n";
        return false;
    }
    if (!std::isfinite(args.bedEpguanBackgroundRetain) ||
        args.bedEpguanBackgroundRetain < 0.0 ||
        args.bedEpguanBackgroundRetain > 1.0 ||
        !std::isfinite(args.bedEpguanSupportThreshold) ||
        args.bedEpguanSupportThreshold <= 0.0 ||
        args.bedEpguanSupportThreshold > 1.0 ||
        !std::isfinite(args.bedEpguanSupportMarginMm) ||
        args.bedEpguanSupportMarginMm < 0.0) {
        std::cerr << "Error: [MultiBed] EPguan background parameters require "
                     "retain in [0,1], threshold in (0,1], and margin_mm >= 0.\n";
        return false;
    }
    if (args.bedSeamHalfWidth < 0) {
        std::cerr << "Error: [MultiBed] seam_half_width must be >= 0.\n";
        return false;
    }
    return true;
}

bool ArgumentParser::validateGeometryArgs(const ReconArgs& args) {
    const bool compactChunkGeometry = !args.chunkCSRDir.empty();
    if (!compactChunkGeometry &&
        (args.Nx <= 0 || args.Ny <= 0 || args.Nz <= 0)) {
        std::cerr << "Error: Invalid reconstruction dimensions (-dim).\n";
        return false;
    }
    if (args.dx <= 0 || args.dy <= 0 || args.dz <= 0) {
        std::cerr << "Error: Invalid voxel spacing (-vox).\n";
        return false;
    }
    return true;
}

bool ArgumentParser::validateAlgorithmArgs(const ReconArgs& args) {
    if (args.numIters <= 0) {
        std::cerr << "Error: Number of iterations (-it) must be positive.\n";
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
    if (!args.ctacAutoMuValueExplicit &&
        spect::core::waterAttenuationPreset(args.ctacAutoMuNuclide) == nullptr) {
        std::cerr << "Error: [CTAC] auto_mu_nuclide must be one of "
                  << spect::core::supportedWaterMuNuclides() << ".\n";
        return false;
    }
    if (!(args.ctacAutoMuValue > 0.0)) {
        std::cerr << "Error: CTAC auto-mu value must be positive.\n";
        return false;
    }
    if (args.detZEdgeCensor < 0) {
        std::cerr << "Error: -detZEdgeCensor must be >= 0.\n";
        return false;
    }
    if (args.detZEdgeCensor > 0) {
        if (args.chunkCSRDir.empty()) {
            std::cerr << "Error: -detZEdgeCensor requires the compact-symmetric "
                         "chunk layout (-chunkCSRDir).\n";
            return false;
        }
        if (args.detNz > 0 && 2 * args.detZEdgeCensor >= args.detNz) {
            std::cerr << "Error: -detZEdgeCensor must satisfy 2*k < detNz.\n";
            return false;
        }
    }
    if (!args.detZRowWeightsFile.empty()) {
        if (args.chunkCSRDir.empty()) {
            std::cerr << "Error: -detZRowWeightsFile requires the "
                         "compact-symmetric chunk layout (-chunkCSRDir).\n";
            return false;
        }
        try {
            (void)spect::io::loadDetZRowWeights(args.detZRowWeightsFile,
                                                args.detNz, args.detNx,
                                                args.numAngles);
        } catch (const std::exception& ex) {
            std::cerr << "Error: " << ex.what() << "\n";
            return false;
        }
    }
    if (!args.normFactorFile.empty()) {
        if (!args.detZRowWeightsFile.empty()) {
            std::cerr << "Error: -normFactorFile and -detZRowWeightsFile are "
                         "mutually exclusive.\n";
            return false;
        }
        if (args.chunkCSRDir.empty()) {
            std::cerr << "Error: -normFactorFile requires the "
                         "compact-symmetric chunk layout (-chunkCSRDir).\n";
            return false;
        }
        if (!args.bProjPath.empty() || !args.bProjPaths.empty()) {
            std::cerr << "Error: -normFactorFile currently supports only "
                         "background-free reconstruction (no -bProj/"
                         "-bProjList).\n";
            return false;
        }
        try {
            const auto values = spect::io::loadDetZRowWeights(
                args.normFactorFile, args.detNz, args.detNx, args.numAngles);
            if (args.detNz > 0 && args.detNx > 0 && args.numAngles > 0 &&
                values.size() != static_cast<size_t>(args.detNz) *
                                     static_cast<size_t>(args.detNx) *
                                     static_cast<size_t>(args.numAngles)) {
                std::cerr << "Error: -normFactorFile must contain one "
                             "correction factor per projection bin.\n";
                return false;
            }
        } catch (const std::exception& ex) {
            std::cerr << "Error: " << ex.what() << "\n";
            return false;
        }
    }
    if (!args.badDetectorFile.empty()) {
        try {
            const auto badDetectors =
                spect::io::loadBadDetectorList(args.badDetectorFile);
            if (!badDetectors.empty() && args.chunkCSRDir.empty()) {
                std::cerr << "Error: a non-empty -badDetectorFile requires "
                             "the canonical compact-symmetric chunk layout "
                             "with identity projections.\n";
                return false;
            }
        } catch (const std::exception& ex) {
            std::cerr << "Error: " << ex.what() << "\n";
            return false;
        }
    }
    if (args.chunkCSRDir.empty()) {
        std::cerr << "Error: -chunkCSRDir is required (or set [SystemMatrix] default_chunk_csr_dir, or provide exactly one exe-local matrix directory).\n";
        return false;
    }
    {
        const bool compactAlgorithmOk =
            args.algorithm == ReconArgs::Algorithm::MLEM ||
            args.algorithm == ReconArgs::Algorithm::BSREM;
        if (!compactAlgorithmOk) {
            std::cerr << "Error: compact-symmetric layout supports only -opti mlem|bsrem.\n";
            return false;
        }
        if ((args.doCTAC || !args.attMapPath.empty()) &&
            args.chunkCSRDir.empty()) {
            std::cerr << "Error: compact-symmetric CTAC requires -chunkCSRDir "
                         "(single-file compact matrix does not support it).\n";
            return false;
        }
        const bool compactHasReg =
            args.useReg && args.reg.type != RegType::None;
        if (compactHasReg &&
            args.algorithm != ReconArgs::Algorithm::BSREM) {
            std::cerr << "Error: compact-symmetric regularization requires -opti bsrem with TV or RDP.\n";
            return false;
        }
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

    if (!args.ctacAutoMuFrom.empty() && !args.doCTAC) {
        std::cerr << "Error: -ctacAutoMu requires -ctac.\n";
        return false;
    }
    if (!args.ctacAutoMuFrom.empty() && !args.attMapPath.empty()) {
        std::cerr << "Error: -ctacAutoMu and -attMap are mutually exclusive; "
                     "pick the measured mu-map or the auto-generated one explicitly.\n";
        return false;
    }

    if (args.usePSF && args.psf.fwhm_mm <= 0.0) {
        std::cerr << "Error: PSF FWHM (-psf) must be positive.\n";
        return false;
    }
    if (args.usePSF && args.psf.min_sigmas <= 0.0) {
        std::cerr << "Error: PSF kernel truncation N (-psf fwhm,N) must be positive.\n";
        return false;
    }

    if (args.useReg) {
        if (!isPublicRegularizer(args.reg.type)) {
            std::cerr << "Error: Only TV and RDP regularization are supported.\n";
            return false;
        }
        if (args.reg.beta < 0) {
            std::cerr
                << "Error: Regularization beta (-regBeta) must be non-negative.\n";
            return false;
        }
        if ((args.reg.type == RegType::TVIsotropic ||
             args.reg.type == RegType::RDP) && args.reg.delta < 0) {
            std::cerr << "Error: TV/RDP regularization delta (-regDelta) must "
                         "be non-negative.\n";
            return false;
        }
        if ((!args.regCtPath.empty() || !args.attMapPath.empty()) &&
            !(args.reg.ct_sigma > 0.0)) {
            std::cerr << "Error: -regCtSigma must be positive.\n";
            return false;
        }
        if (args.reg.ct_weight_floor < 0.0 ||
            args.reg.ct_weight_floor > 1.0) {
            std::cerr << "Error: -regCtFloor must be in [0,1].\n";
            return false;
        }
    }

    if (args.useSmooth && args.smoothSigma <= 0) {
        std::cerr << "Error: Smoothing sigma (-smooth) must be positive.\n";
        return false;
    }

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

bool ArgumentParser::validateArgs(const ReconArgs& args) {
    if (!validateIoArgs(args))
        return false;
    if (!validateGeometryArgs(args))
        return false;
    return validateAlgorithmArgs(args);
}

void ArgumentParser::dumpMergedConfig(const ReconArgs& a) {
    std::cout << "# ── Merged Configuration (after all layers) ──\n\n";

    std::cout << "[Basic]\n";
    std::cout << "dimensions = " << a.Nx << "," << a.Ny << "," << a.Nz << "\n";
    std::cout << "voxel_size = " << a.Vx << "," << a.Vy << "," << a.Vz << "\n";
    std::cout << "threads    = " << a.numThreads << "\n";
    std::cout << "verbosity  = " << a.verbosity << "\n\n";

    std::cout << "[Diagnostics]\n";
    std::cout << "projection_residual = "
              << (a.projectionResidual ? "true" : "false") << "\n\n";

    std::cout << "[Detector]\n";
    std::cout << "channels = " << a.detNz;
    if (a.detNx != a.detNz) std::cout << "," << a.detNx;
    std::cout << "\nangles   = " << a.numAngles << "\n";
    std::cout << "bad_detector_file = "
              << (a.badDetectorFile.empty() ? "(none)" : a.badDetectorFile)
              << "\n\n";

    std::cout << "[Algorithm]\n";
    std::cout << "name       = " << a.getAlgorithmName() << "\n";
    std::cout << "iterations = " << a.numIters << ":" << a.saveInterval << "\n\n";

    std::cout << "[Regularization]\n";
    std::cout << "type  = " << regTypeToString(a.reg.type) << "\n";
    std::cout << "beta  = " << a.reg.beta << "\n";
    std::cout << "delta = " << a.reg.delta << "\n";
    if (!a.regCtPath.empty()) {
        std::cout << "ct_image = " << a.regCtPath << "\n";
        std::cout << "ct_sigma = " << a.reg.ct_sigma << "\n";
        std::cout << "ct_weight_floor = " << a.reg.ct_weight_floor << "\n";
    }
    std::cout << "\n";

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
    std::cout << "[Filtering]\n";
    std::cout << "butterworth_cutoff  = " << a.butterCutoff << "\n";
    std::cout << "butterworth_order   = " << a.butterOrder << "\n";
    std::cout << "gaussian_sigma      = " << a.smoothSigma << "\n";
    std::cout << "gaussian_kernel     = " << a.smoothKernel << "\n\n";

    std::cout << "[PSF]\n";
    const char* psfModes[] = {"None","Operator"};
    int psfIdx = static_cast<int>(a.psf.mode);
    std::cout << "mode = " << (psfIdx >= 0 && psfIdx <= 1 ? psfModes[psfIdx] : "Unknown") << "\n";
    std::cout << "fwhm = " << a.psf.fwhm_mm << "\n";
    std::cout << "z_blur = " << (a.psf.z_blur ? "true" : "false") << "\n";
    std::cout << "min_sigmas = " << a.psf.min_sigmas << "\n\n";

    std::cout << "[CTAC]\n";
    std::cout << "auto_mu_nuclide = " << a.ctacAutoMuNuclide << "\n";
    std::cout << "auto_mu_value   = " << a.ctacAutoMuValue << "\n";
    std::cout << "auto_mu_source  = " << a.ctacAutoMuValueSource << "\n\n";

    std::cout << "[Mask]\n";
    std::cout << "use_cylinder   = " << (a.useCylindricalMask ? "true" : "false") << "\n";
    std::cout << "cylinder_dims  = " << a.cylinderDiameterMM << "," << a.cylinderHeightMM << "\n\n";

    std::cout << "[Backend]\n";
    std::cout << "sparse = cuda\n";
    std::cout << "butter = cufft\n\n";

    std::cout << "[Matrix]\n";
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
    std::cout << "crop_method = " << a.bedCropMethod
              << "  # center66_cosine10 (default)|sigmoid|direct; center crop uses step56/cosine10\n";
    std::cout << "seam_postprocess = " << (a.bedSeamPostprocess ? 1 : 0)
              << "  # request; active only for multi-bed output overlap stitch\n";
    std::cout << "detz_edge_censor = " << a.detZEdgeCensor
              << "  # detZ edge rows censored per side (0=off; compact chunk layout only)\n";
    std::cout << "detz_row_weights_file = "
              << (a.detZRowWeightsFile.empty() ? "(none)" : a.detZRowWeightsFile)
              << "  # detZ efficiency weights (detNz or detNz*detNx floats); sensitivity only\n";
    std::cout << "normalization_factor_file = "
              << (a.normFactorFile.empty() ? "(none)" : a.normFactorFile)
              << "  # full projection correction c; reciprocal 1/c enters sensitivity\n";
    std::cout << "object_profile = " << a.bedObjectProfile
              << "  # auto|mouse|phantom|epguan|zimu|derenzo\n";
    std::cout << "derenzo_postprocess = "
              << (a.bedDerenzoPostprocess ? 1 : 0)
              << "  # runs when the profile is derenzo (explicit or auto)\n";
    std::cout << "derenzo_axial_strength = "
              << a.bedDerenzoAxialStrength
              << "  # positive-only per-rod fill [0,1]\n";
    std::cout << "derenzo_adhesion = " << (a.bedDerenzoAdhesion ? 1 : 0)
              << "  # in-plane rod adhesion fill + core damping\n";
    std::cout << "derenzo_adhesion_inner = " << a.bedDerenzoAdhesionInner
              << "  # adhesion fraction at the inner sectors [0,1]\n";
    std::cout << "derenzo_adhesion_outer = " << a.bedDerenzoAdhesionOuter
              << "  # adhesion fraction at the rim sectors [0,1]\n";
    std::cout << "derenzo_core_damp = " << a.bedDerenzoCoreDamp
              << "  # rod-core high-frequency damping [0,1]\n";
    std::cout << "phantom_close_strength = " << a.bedPhantomCloseStrength
              << "  # contour interpolation blend [0,1]\n";
    std::cout << "epguan_close_strength = " << a.bedEpguanCloseStrength
              << "  # EP tube contour interpolation blend [0,1]\n";
    std::cout << "epguan_wall_max_preserve = "
              << (a.bedEpguanWallMaxPreserve ? 1 : 0) << "\n";
    std::cout << "epguan_seam_edge_strength = "
              << a.bedEpguanSeamEdgeStrength << "  # [0,1]\n";
    std::cout << "epguan_seam_feather_slices = "
              << a.bedEpguanSeamFeatherSlices << "\n";
    std::cout << "epguan_background_filter = "
              << (a.bedEpguanBackgroundFilter ? 1 : 0) << "\n";
    std::cout << "epguan_background_retain = "
              << a.bedEpguanBackgroundRetain << "  # [0,1]\n";
    std::cout << "epguan_support_threshold = "
              << a.bedEpguanSupportThreshold << "  # (0,1]\n";
    std::cout << "epguan_support_margin_mm = "
              << a.bedEpguanSupportMarginMm << "\n";
    std::cout << "zimu_close_strength = " << a.bedZimuCloseStrength
              << "  # topology-locked Gaussian sigma=0.3mm blend [0,1]\n";
    std::cout << "mouse_seam_strength = " << a.bedMouseSeamStrength
              << "  # bilateral confidence fusion blend [0,1]\n";
    std::cout << "seam_half_width = " << a.bedSeamHalfWidth
              << "  # capped at overlap/2; zimu additionally capped at 4\n";
    std::cout << "fov_radius_mm = " << a.fovRadiusMm << "  # 0=off\n";
    std::cout << "analysis_fov_radius_mm = " << a.analysisFovRadiusMm
              << "  # 0=follows fov_radius_mm\n";
    std::cout << "crop_end     = " << a.cropEndStart << "," << a.cropEndEnd << "  # top,end slices zeroed\n";
    std::cout << "z_bed_flip   = " << (a.zBedFlip ? 1 : 0) << "\n";
    std::cout << "z_out_flip   = " << (a.zOutFlip ? 1 : 0) << "\n";
    std::cout << "x_out_flip   = " << (a.xOutFlip ? 1 : 0) << "\n\n";

    std::cout << "[InputOutput]\n";
    std::cout << "output_dir  = " << a.outputDir << "\n";
    std::cout << "output_name = " << a.outputName << "\n\n";

    std::cout << "# ── Config sources ──\n";
    std::cout << "# global_ini  = " << a.configPath << (a.configLoaded ? " (loaded)" : " (not found)") << "\n";
    std::cout << "# algo_conf   = " << (a.algoConfPath.empty() ? "(none)" : a.algoConfPath) << "\n";
}

void ArgumentParser::printUsage() {
    std::cout << "SPECT Reconstruction Tool\n";
    std::cout << "Usage: SPECT-recon [options]\n\n";

    std::cout << "=== Quick Start ===\n";
    std::cout << "  SPECT-recon -df proj.bin -o recon_output   # uses single exe-local bin/matrix/* if present\n";
    std::cout << "  SPECT-recon -df proj.bin -chunkCSRDir matrix/default_psc\n\n";

    std::cout << "  Windows paths: use quotes or forward slashes\n";
    std::cout << "    -df \"D:\\path\\file.mat\"  or  -df D:/path/file.bin\n\n";

    std::cout << "=== Data I/O ===\n";
    std::cout << "  -df, -i <path>          Projection MAT or BIN file\n";
    std::cout << "  -dfList <p1,p2,...>      Multi-bed projection MAT/BIN list, sorted by BedN when present (max "
              << spect::core::kMaxBeds << " beds)\n";
    std::cout << "  -chunkCSRDir <dir>       Directory with metadata.bin + chunk_*.csrb\n";
    std::cout << "                           Default: config default_chunk_csr_dir, or the only exe-local bin/matrix/* candidate\n";
    std::cout << "  -compactDenomCache <mhd> Reuse/save compact global denominator cache; use separate files for PSF/no-PSF/FWHM/bed variants\n";
    std::cout << "  -ctacFactorCacheDir <dir> Directory for cached CTAC attenuation factor tables (default: recon_out/ctac_factor_cache)\n";
    std::cout << "  -attMap <path>           Attenuation map (mu, mm^-1) for attenuation correction / Chang's AC\n";
    std::cout << "  -bProj <path>            Background/scatter projection MAT/BIN\n";
    std::cout << "  -bProjList <list>        Multi-bed background MAT/BIN list, sorted by BedN when present\n";
    std::cout << "  -fout <dir/name>         Output path prefix\n";
    std::cout << "  -o <dir>                 Output directory\n";
    std::cout << "  -n <name>                Output filename base\n\n";

    std::cout << "=== Algorithm ===\n";
    std::cout << "  -opti <name>             MLEM | BSREM\n";
    std::cout << "                           Auto-loads config/algo_<name>.conf\n";
    std::cout << "  -algoConf <path>         Custom algorithm config file\n";
    std::cout << "  -it <N[:save]>           Iterations (default: from algo conf)\n";
    std::cout << "  -cudaDevice <id>         Select CUDA device (overrides SPECT_CUDA_DEVICE)\n\n";

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

    std::cout << "=== Regularization ===\n";
    std::cout << "  -reg <type[,beta,delta]> TV | RDP\n";
    std::cout << "  -regBeta <beta>          Regularization weight\n";
    std::cout << "  -regDelta <delta>        Transition parameter\n\n";
    std::cout << "  -regCT <path>            Registered CT/mu-map for TV/RDP edge weights\n";
    std::cout << "  -regCtSigma <value>      CT edge scale (default 0.01 for mu mm^-1)\n";
    std::cout << "  -regCtFloor <0..1>       Minimum CT edge weight (default 0.05)\n\n";

    std::cout << "=== Filtering ===\n";
    std::cout << "  -butter <fc[,order]>     Butterworth low-pass (default: 0.35,2)\n";
    std::cout << "  -smooth <sigma>          Gaussian smoothing sigma (mm)\n";
    std::cout << "  -smoothKernel <n>        Gaussian kernel half-width\n\n";

    std::cout << "=== PSF ===\n";
    std::cout << "  -psf [fwhm_mm[,N]]       PSF Gaussian FWHM (mm); N*sigma truncation (defaults: 1.5,4)\n";
    std::cout << "  -psfMode <mode>          None | Operator\n";
    std::cout << "\n";
    std::cout << "  SPECT_CUDA_PSF_DOUBLE=1  Force double-precision CUDA PSF (default float)\n";
    std::cout << "  SPECT_CUDA_VOLUME_OPS=0  Disable CUDA volume ops; PSF uses CPU fallback\n";

    std::cout << "=== Geometry ===\n";
    std::cout << "  -dim <Nx,Ny,Nz>          Reconstruction grid (voxels)\n";
    std::cout << "  -vox <Vx,Vy,Vz>          Voxel size (mm)\n";
    std::cout << "  -ang <N>                 Projection angles\n";
    std::cout << "  -detN <Nz[,Nx]>          Detector channels\n\n";
    std::cout << "  -badDetectorFile <file>  Skip global 0..95 or link/FPGA/dector blocks in compact identity reconstruction\n\n";

    std::cout << "=== Multi-Bed ===\n";
    std::cout << "  -bedOverlap <b1,b2,b3,b4,b5> Fixed Bed1..Bed5 overlap table; slot b1=Bed1-Bed2\n";
    std::cout << "  -bedCropMethod <center66_cosine10|sigmoid|direct> Output stitching method (default center66_cosine10)\n";
    std::cout << "  -bedSeamPostprocess <0|1> Request seam postprocess (default 1; only runs with multi-bed output overlap)\n";
    std::cout << "  -detZEdgeCensor <k>       Censor k detZ edge rows per side in compact numerator+denominator (default 0)\n";
    std::cout << "  -detZRowWeightsFile <f>   Efficiency weights: text detZ/per-bin/full projection; sensitivity only\n";
    std::cout << "  -normFactorFile <f>       Full projection correction factors c; applies reciprocal 1/c to sensitivity\n";
    std::cout << "  -bedObjectProfile <auto|mouse|phantom|epguan|zimu|derenzo> Override automatic object profile\n";
    std::cout << "  -derenzoPostprocess <0|1> Enable Derenzo axial continuity processing (auto-detected profile)\n";
    std::cout << "  -derenzoAxialStrength <s> Positive-only per-rod fill strength [0,1]\n";
    std::cout << "  -derenzoAdhesion <0|1>    Enable Derenzo in-plane adhesion fill and rod-core damping\n";
    std::cout << "  -derenzoAdhesionInner <s> Adhesion fraction at the inner sectors [0,1]\n";
    std::cout << "  -derenzoAdhesionOuter <s> Adhesion fraction at the rim sectors [0,1]\n";
    std::cout << "  -derenzoCoreDamp <s>      Rod-core high-frequency damping [0,1]\n";
    std::cout << "  -cropEnd <top[,end]>      Zero final Z top/end slices after output flips\n";
    std::cout << "  -zbedflip <0|1>           Reverse local dfList bed Z (inside compact joint mapping for multi-bed)\n";
    std::cout << "  -zoutflip <0|1>           Flip whole output volume on Z after per-bed flip (default 0)\n";
    std::cout << "  -xoutflip <0|1>           Flip whole output volume on X (default 0)\n";
    std::cout << "  -ctac                     Enable iterative attenuation correction (uses -attMap; mu in mm^-1)\n";
    std::cout << "  -ctacModel <m>            line-integral (default; angle-dependent parallel-ray Int mu ds)\n";
    std::cout << "  -ctacRotSign <1|-1>       Rotation sign matching sysmat generator --voxel-rotation-sign (default 1)\n";
    std::cout << "  -ctacAutoMu <vol>         Build water mu-map from an early-iteration recon volume\n";
    std::cout << "                            (threshold + closing + hole fill; for activity-filled phantoms;\n";
    std::cout << "                            mutually exclusive with -attMap; generated map saved as *_automu)\n";
    std::cout << "  -ctacAutoMuThresh <f>     Relative threshold vs P99.9 activity (default 0.05)\n";
    std::cout << "  -ctacAutoMuValue <mu>     Debug override for auto-mu body fill, mm^-1\n";
    std::cout << "                            Default comes from [CTAC] auto_mu_nuclide (Tc99m => 0.0152)\n\n";

    std::cout << "=== Masking ===\n";
    std::cout << "  -cyl <diam,height>        Cylindrical mask (mm)\n";
    std::cout << "  -cylc <cx,cy,cz>          Cylinder center (1-based voxel)\n";
    std::cout << "  -maskFromAtt <thresh>     Body mask from mu-map\n\n";

    std::cout << "=== Advanced ===\n";
    std::cout << "  -config <path>            Global config (default search: exe/config, exe, cwd/config, cwd)\n";
    std::cout << "  -dumpConf                 Print merged config and exit\n";
    std::cout << "  -projectionResidual [on|off]  Write final compact projection residual CSV\n";
    std::cout << "  -Aprune <eps>             Prune system-matrix weights (default 1e-10)\n";
    std::cout << "  -Astream on|off           Stream build from disk (default off)\n";
    std::cout << "  -ApruneMask on|off        Prune columns outside mask (default on)\n";
    std::cout << "  -joe <n>                  Joseph-like smoothing passes (default 0)\n";
    std::cout << "  -vb <0|1|2>               Verbosity level (default 1)\n";
    std::cout << "  -v, --version             Print version string and exit\n";
    std::cout << "  -th <N>                   Thread count\n";
    std::cout << "  -h, -help, --help         Show this help\n\n";
    std::cout << "  SPECT_FAIL_ON_ZERO_CHECKPOINT=1  Abort before writing an all-zero checkpoint\n";

    std::cout << "  Algorithm-specific defaults are loaded beside the global config or from config/algo_<name>.conf.\n";
    std::cout << "  Override any value with CLI flags. Use -dumpConf to inspect.\n";
    std::cout << "  See: doc/USAGE_CN.md for full parameter reference.\n\n";
}

} // namespace cli
} // namespace spect
