// ArgumentParser implementation (moved from header, PR15 move-only).

#include "cli/ArgumentParser.h"

#include "cli/Version.h"
#include "core/AttenuationCoefficients.h"
#include "core/SixModuleGeometry.h"
#include "io/BadDetectorMask.h"
#include "io/DetZRowWeights.h"

namespace spect {
namespace cli {

namespace {

int parseInteger(const std::string& value, int minimum, const char* option) {
    size_t consumed = 0;
    const int result = std::stoi(value, &consumed);
    if (consumed != value.size() || result < minimum)
        throw std::runtime_error(std::string(option) + " has an invalid integer: " + value);
    return result;
}

bool isPublicRegularizer(RegType type) {
    return type == RegType::None || type == RegType::TVIsotropic ||
           type == RegType::RDP;
}

bool rejectedCliAliasError(const std::string& key, std::string& error) {
    if (key == "-subset-mode" || key == "-subset-diagnostics") {
        error = "Error: this build supports angle subsets only; use -it A:B or -subsets.\n";
        return true;
    }
    if (key == "-iter" || key == "-iters" || key == "-iteration" ||
        key == "-iterations") {
        error = "Error: '" + key +
                "' is not supported; use -it <iterations[:subsets]>.\n";
        return true;
    }
    if (key == "-verbose" || key == "-verbosity") {
        error = "Error: '" + key +
                "' is not supported; use -vb <0|1|2>.\n";
        return true;
    }
    return false;
}


std::string canonicalIsotope(std::string value) {
    value = toLower(value);
    if (value == "tc99m" || value == "tc-99m" || value == "99mtc") return "Tc99m";
    if (value == "lu177" || value == "lu-177" || value == "177lu") return "Lu177";
    if (value.empty()) return value;
    throw std::runtime_error("Unsupported isotope: " + value);
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
    std::error_code ec;
    for (const auto& path : candidates) {
        if (path.empty())
            continue;
        if (std::filesystem::exists(path, ec) && !ec)
            return path.string();
        ec.clear();
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

    const auto root = (std::filesystem::path(args.matrixRoot).is_absolute()
        ? std::filesystem::path(args.matrixRoot) : exeDir / args.matrixRoot).lexically_normal();
    auto resolveId = [&](const std::string& id) {
        if (id.empty() || id == "." || id == ".." ||
            id.find_first_of("/\\:") != std::string::npos)
            throw std::runtime_error("Invalid matrix_id: " + id);
        std::vector<std::filesystem::path> found;
        for (const auto& candidate : {root / id, root / args.isotope / id}) {
            if (std::filesystem::is_regular_file(candidate / "metadata.bin") &&
                std::find(found.begin(), found.end(), candidate) == found.end())
                found.push_back(candidate);
        }
        if (found.size() != 1)
            throw std::runtime_error("matrix_id '" + id + "' requires exactly one metadata.bin directory under " + root.string());
        return found.front().string();
    };
    if (!args.matrixId.empty()) args.chunkCSRDir = resolveId(args.matrixId);
    if (!args.matrixId2.empty()) args.chunkCSRDir2 = resolveId(args.matrixId2);
    if (!args.chunkCSRDir2.empty())
        args.chunkCSRDir2 = resolveChunkCSRDirPath(args.chunkCSRDir2, exeDir, args.chunkCSRDir2Explicit).string();
    if (!args.projectionModel.empty() &&
        ((args.projectionModel == "joint") != args.dualWindow()))
        throw std::runtime_error("projection_model disagrees with the number of matrices.");
    for (auto* file : {&args.normFactorFile, &args.normFactorFile2})
        if (!file->empty() && !std::filesystem::path(*file).is_absolute())
            *file = ((file == &args.normFactorFile && args.normFactorFileExplicit)
                ? std::filesystem::absolute(*file) : exeDir / *file).lexically_normal().string();
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
    std::string configPath;
    std::string preScanAlg;
    std::string preScanAlgoConf;
    std::string requestedIsotope;
    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];
        if (key == "-config" && i + 1 < argc) {
            configPath = argv[++i];
            args.configExplicit = true;
        } else if (key == "-isotope" && i + 1 < argc) {
            requestedIsotope = canonicalIsotope(argv[++i]);
        } else if (key == "-opti" && i + 1 < argc) {
            preScanAlg = toLower(std::string(argv[++i]));
        } else if (key == "-algoConf" && i + 1 < argc) {
            preScanAlgoConf = argv[++i];
        }
    }
    if (args.configExplicit && preScanAlgoConf.empty()) {
        ConfigParser supplied;
        if (supplied.load(configPath) && supplied.hasSection("Algorithm") &&
            !supplied.hasKey("Reconstruction", "default_config")) {
            // Existing -config calls may name a complete recipe. Preserve the
            // deployed defaults underneath it instead of loading another algo file.
            preScanAlgoConf = configPath;
            args.configExplicit = false;
        }
    }
    if (!args.configExplicit)
        configPath = resolveDefaultConfigPath(argv);
    args.configPath = configPath;

    if (const char* cudaDeviceEnv = std::getenv("SPECT_CUDA_DEVICE")) {
        if (cudaDeviceEnv[0] != '\0') {
            args.cudaDeviceValue = cudaDeviceEnv;
            args.cudaDeviceRequested = true;
            args.cudaDeviceFromCli = false;
        }
    }

    // 1. Load global config if it exists. Error-code overloads keep an
    // inaccessible optional default from aborting before CLI parsing.
    ConfigParser cfg;
    std::error_code configExistsError;
    const bool configExists =
        std::filesystem::exists(configPath, configExistsError);
    if (configExistsError) {
        args.configFound = false;
        if (args.configExplicit) {
            std::cerr << "Error: Cannot access explicit config file '"
                      << configPath << "': "
                      << configExistsError.message() << ".\n";
            return ParseResult::ExitFailure;
        }
    } else if (configExists) {
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
    const std::string defaultRecipe = cfg.getString("Reconstruction",
        requestedIsotope.empty() ? "default_config" : "default_config_" + requestedIsotope,
        cfg.getString("Reconstruction", "default_config"));
    std::string algoConfPath;
    if (!preScanAlgoConf.empty()) {
        algoConfPath = preScanAlgoConf;
    } else if (!defaultRecipe.empty()) {
        const std::filesystem::path selected(defaultRecipe);
        algoConfPath = (selected.is_absolute() ? selected :
            std::filesystem::path(configPath).parent_path() / selected).lexically_normal().string();
    } else {
        // Legacy configs without default_config keep algorithm auto-detection.
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
        ConfigParser recipe;
        if (!recipe.load(algoConfPath))
            throw std::runtime_error("Cannot load reconstruction recipe: " + algoConfPath);
        if (recipe.hasKey("SystemMatrix", "matrix_id") ||
            recipe.hasKey("SystemMatrix", "chunk_csr_dir") ||
            recipe.hasKey("SystemMatrix", "default_chunk_csr_dir") || recipe.hasKey("Matrix", "chunk_csr_dir")) {
            for (const auto* key : {"matrix_id", "default_chunk_csr_dir", "chunk_csr_dir"})
                cfg.erase("SystemMatrix", key);
            cfg.erase("Matrix", "chunk_csr_dir");
        }
        if (recipe.hasKey("SystemMatrix", "matrix_id2") || recipe.hasKey("SystemMatrix", "chunk_csr_dir2")) {
            cfg.erase("SystemMatrix", "matrix_id2");
            cfg.erase("SystemMatrix", "chunk_csr_dir2");
        }
        cfg.overlay(recipe);
        applyConfig(args, cfg);
    }

    // 3. Process command-line arguments (highest priority override)
    bool cliIterationSubsets = false;
    bool cliSubsetSchedule = false;
    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];
        if (key == "-it" && i + 1 < argc)
            cliIterationSubsets = cliIterationSubsets ||
                std::string(argv[i + 1]).find(':') != std::string::npos;
        if (key == "-subsets")
            cliSubsetSchedule = true;
        auto it = handlers_.find(key);
        if (it != handlers_.end()) {
            if (!it->second(i, argc, argv, args))
                return ParseResult::ExitFailure;
            if (args.helpRequested || args.versionRequested)
                return ParseResult::ExitSuccess;
        } else {
            std::string aliasError;
            if (rejectedCliAliasError(key, aliasError)) {
                std::cerr << aliasError;
                return ParseResult::ExitFailure;
            }
            std::cerr << "[NOTE] Ignoring unsupported CLI argument: " << key;
            if (key.find('=') == std::string::npos && i + 1 < argc) {
                const std::string next = argv[i + 1];
                if (!next.empty() && next[0] != '-')
                    std::cerr << " " << argv[++i];
            }
            std::cerr << "\n";
        }
    }

    if (cliIterationSubsets && cliSubsetSchedule) {
        std::cerr << "Error: use -it A:B or -it A -subsets <schedule>, not both.\n";
        return ParseResult::ExitFailure;
    }
    if (args.reconSectionsOverrideCli) {
        // Reapply only configured reconstruction fields. Runtime destinations,
        // logging and device/CPU-thread selections always remain caller-owned.
        ConfigParser protectedConfig = cfg;
        for (const auto* key : {"output_dir", "output_name"})
            protectedConfig.erase("InputOutput", key);
        for (const auto* key : {"threads", "verbosity"})
            protectedConfig.erase("Basic", key);
        applyConfig(args, protectedConfig);
    }

    if (!requestedIsotope.empty()) {
        if (!args.isotope.empty() && args.isotope != requestedIsotope)
            throw std::runtime_error("Acquisition isotope disagrees with selected reconstruction recipe.");
        args.isotope = requestedIsotope;
    }
    // 4. Resolve matrix defaults after all config/CLI layers are known.
    std::string matrixDiscoveryError;
    if (!applyMatrixAutoDiscovery(args, argv, matrixDiscoveryError)) {
        std::cerr << "Error: " << matrixDiscoveryError << "\n";
        return ParseResult::ExitFailure;
    }
    const std::filesystem::path exeDir = executableDir(argv);
    const auto resolveExeRelative = [&exeDir](const std::string& value) {
        const std::filesystem::path path(value);
        return (path.is_absolute() ? path : exeDir / path).lexically_normal();
    };
    if (!args.compactDenomCachePath.empty()) {
        if (args.compactDenomCachePathExplicit) {
            std::error_code ec;
            const auto absolute = std::filesystem::absolute(
                args.compactDenomCachePath, ec);
            args.compactDenomCachePath =
                (ec ? std::filesystem::path(args.compactDenomCachePath)
                    : absolute).lexically_normal().string();
        } else {
            args.compactDenomCachePath =
                resolveExeRelative(args.compactDenomCachePath).string();
        }
    }
    if (!args.compactDenomCacheDir.empty()) {
        args.compactDenomCacheDir =
            resolveExeRelative(args.compactDenomCacheDir).string();
    }

    // 5. Normalize derived defaults, then handle -dumpConf and validation.
    normalizeArgs(args);
    if (args.bsremMlemEquivalenceMode) {
        args.bsrem_alpha0 = 1.0f;
        args.bsrem_gamma = 0.0f;
        args.bsremUseCurvDenom = false;
        args.useReg = false;
        args.reg.type = RegType::None;
        args.reg.beta = 0.0;
    }

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

    auto dataHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        // Treat -i/-input as aliases for -df (path normalization/drive-letter
        // handling is keyed on the arg name inside getRequiredArg).
        a.projPath = ArgumentParser::getRequiredArg(i, argc, argv, "-df");
        return true;
    };
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs&) {
        (void)ArgumentParser::getRequiredArg(i, argc, argv, "-isotope");
        return true; // Pre-scanned assertion, never an override of recipe identity.
    }, {"-isotope"});
    bindHandlerAliases(dataHandler, {"-df", "-i", "-input"});

    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.projPaths = spect::core::sortPathsByBedOrder(
            splitPathList(ArgumentParser::getRequiredArg(i, argc, argv, "-dfList")));
        return true;
    }, {"-dfList"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.chunkCSRDir = ArgumentParser::getRequiredArg(i, argc, argv, "-chunkCSRDir");
        a.chunkCSRDirExplicit = true;
        a.matrixId.clear();
        return true;
    }, {"-chunkCSRDir"});
    // Second energy window (joint dual-window MLEM).
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.chunkCSRDir2 = ArgumentParser::getRequiredArg(i, argc, argv, "-chunkCSRDir2");
        a.chunkCSRDir2Explicit = true;
        a.matrixId2.clear();
        return true;
    }, {"-chunkCSRDir2"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.projPath2 = ArgumentParser::getRequiredArg(i, argc, argv, "-df2");
        return true;
    }, {"-df2"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.projPaths2 = spect::core::sortPathsByBedOrder(
            splitPathList(ArgumentParser::getRequiredArg(i, argc, argv, "-dfList2")));
        return true;
    }, {"-dfList2"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.bProjPath2 = ArgumentParser::getRequiredArg(i, argc, argv, "-bProj2");
        return true;
    }, {"-bProj2"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.bProjPaths2 = spect::core::sortPathsByBedOrder(
            splitPathList(ArgumentParser::getRequiredArg(i, argc, argv, "-bProjList2")));
        return true;
    }, {"-bProjList2"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.csrqResidency =
            ArgumentParser::getRequiredArg(i, argc, argv, "-csrqResidency");
        return true;
    }, {"-csrqResidency"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        const auto parts =
            split(ArgumentParser::getRequiredArg(i, argc, argv, "-energyWeights"), ',');
        a.energyWindowWeights.clear();
        for (auto& p : parts) {
            if (!p.empty())
                a.energyWindowWeights.push_back(std::stod(p));
        }
        return true;
    }, {"-energyWeights"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.compactDenomCachePath =
            ArgumentParser::getRequiredArg(i, argc, argv, "-compactDenomCache");
        a.compactDenomCachePathExplicit = true;
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
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.saveInterval = parseInteger(
            ArgumentParser::getRequiredArg(i, argc, argv, "-save-every"),
            0, "-save-every");
        return true;
    }, {"-save-every"});
    // Divisibility against the compact angle count is checked once the
    // matrix metadata is known (recon loop, parseCompactSubsetSchedule).
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.subsetSchedule = ArgumentParser::getRequiredArg(i, argc, argv, "-subsets");
        if (a.subsetSchedule.empty()) {
            std::cerr << "Error: -subsets needs N or N1:i1,N2:i2,...\n";
            return false;
        }
        return true;
    }, {"-subsets"});
    auto threadsHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        a.numThreads = std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, k));
        return true;
    };
    bindHandlerAliases(threadsHandler, {"-th"});
    bindHandlerAliases(threadsHandler, {"-threads"});
    bindHandlerAliases(threadsHandler, {"-t"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.cudaDeviceValue = ArgumentParser::getRequiredArg(i, argc, argv, "-cudaDevice");
        a.cudaDeviceRequested = true;
        a.cudaDeviceFromCli = true;
        return true;
    }, {"-cudaDevice"});


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
        a.bsremSensFloorRel = (float)std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bsremSensFloor"));
        return true;
    }, {"-bsremSensFloor"});
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
        a.bsremSoftMaskLambdaOut = std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bsremSoftLambda"));
        return true;
    }, {"-bsremSoftLambda"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(
            i, argc, argv, a.bsremMlemEquivalenceMode);
    }, {"-bsremEquiv"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv,
                                                  a.bsremDiagnostics);
    }, {"-bsremDiag"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.bsremDiagInterval = std::stoi(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bsremDiagInterval"));
        return a.bsremDiagInterval > 0;
    }, {"-bsremDiagInterval"});

    // --- IO & Verbosity ---
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOutputPath(
            ArgumentParser::getRequiredArg(i, argc, argv, "-fout"), a);
    }, {"-fout"});
    auto outDirHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        // Value is the output directory itself (not a name prefix).
        const std::string k = argv[i];
        const std::string dir = ArgumentParser::getRequiredArg(i, argc, argv, k);
        std::filesystem::path p = std::filesystem::path(dir).lexically_normal();
        if (p.is_absolute()) {
            std::error_code absEc;
            auto abs = std::filesystem::absolute(p, absEc);
            a.outputDir =
                (absEc ? p : abs).lexically_normal().make_preferred().string();
        } else {
            a.outputDir = p.empty() ? std::string(".") : p.string();
        }
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
        const std::string flag = argv[i];
        if (i + 1 < argc) {
            const std::string next = argv[i + 1];
            if (!next.empty() && next[0] != '-') {
                std::cerr << "Error: " << flag
                          << " prints version and exits; use -vb <0|1|2> "
                             "for verbosity.\n";
                return false;
            }
        }
        printVersion(flag == "-v");
        a.versionRequested = true;
        return true;
    }, {"-v", "--version"});

    // --- Filtering & PSF ---
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
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string val =
            toLower(ArgumentParser::getRequiredArg(i, argc, argv, "-psfMode"));
        a.psfModeExplicit = true;
        if (val == "operator")
            a.psf.mode = PSFMode::Operator;
        else if (val == "sieve")
            a.psf.mode = PSFMode::Sieve;
        else if (val == "none")
            a.psf.mode = PSFMode::None;
        else {
            std::cerr << "Error: -psfMode expects none|operator|sieve; got '" << val << "'.\n";
            return false;
        }
        a.usePSF = a.psf.mode != PSFMode::None;
        return true;
    }, {"-psfMode"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string val = toLower(
            ArgumentParser::getRequiredArg(i, argc, argv, "-psfKernel"));
        if (val == "adjoint")
            a.psf.kernel = PSFKernel::Adjoint;
        else if (val == "castor")
            a.psf.kernel = PSFKernel::Castor;
        else {
            std::cerr << "Error: -psfKernel expects adjoint|castor; got '"
                      << val << "'.\n";
            return false;
        }
        return true;
    }, {"-psfKernel"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.useSmooth = true;
        a.smoothSigma =
            (float)std::stod(ArgumentParser::getRequiredArg(i, argc, argv, "-smooth"));
        a.smoothApplyStage = "in_recon";  // CLI filter flags override INI stage
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
        a.butterApplyStage = "in_recon";  // CLI filter flags override INI stage
        return true;
    }, {"-butter"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.butterOrder =
            std::stoi(ArgumentParser::getRequiredArg(i, argc, argv, "-butterOrder"));
        return true;
    }, {"-butterOrder"});

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

    // --- Mask & Geometry Extras ---
    auto cylHandler = [](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string k = argv[i];
        auto parts = split(ArgumentParser::getRequiredArg(i, argc, argv, k), ',');
        if (parts.size() < 2)
            return false;
        a.useCylindricalMask = true;
        a.cylinderDiameterMM = (float)std::stod(parts[0]);
        a.cylinderHeightMM = (float)std::stod(parts[1]);
        a.cylinderDimsExplicit = true;
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
        const double stepMm = std::stod(
            ArgumentParser::getRequiredArg(i, argc, argv, "-bedStepMm"));
        if (stepMm <= 0.0) {
            std::cerr << "Error: -bedStepMm must be positive.\n";
            return false;
        }
        a.bedStepMm = stepMm;
        return true;
    }, {"-bedStepMm"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(
            i, argc, argv, a.singleBedStepAligned);
    }, {"-singleBedStepAligned"});
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
        a.normFactorFileExplicit = true;
        return true;
    }, {"-normFactorFile"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        a.badDetectorFile = ArgumentParser::getRequiredArg(
            i, argc, argv, "-badDetectorFile");
        a.badDetectorFileExplicit = true;
        return true;
    }, {"-badDetectorFile"});
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
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        return ArgumentParser::parseOnOffOptional(i, argc, argv, a.seamXyAlign);
    }, {"-seamXyAlign"});

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
        a.ctacMapEnergyKeV =
            std::stod(ArgumentParser::getRequiredArg(i, argc, argv, "-ctacMapEnergy"));
        if (!(a.ctacMapEnergyKeV > 0.0)) {
            std::cerr << "Error: -ctacMapEnergy (keV) must be positive.\n";
            return false;
        }
        return true;
    }, {"-ctacMapEnergy"});
    bindHandlerAliases([](int& i, int argc, char** argv, ReconArgs& a) {
        const std::string raw =
            ArgumentParser::getRequiredArg(i, argc, argv, "-ctacWindowEnergies");
        a.ctacWindowEnergiesKeV.clear();
        for (const std::string& item : split(raw, ',')) {
            if (item.empty()) continue;
            const double keV = std::stod(item);
            if (!(keV > 0.0)) {
                std::cerr << "Error: -ctacWindowEnergies entries (keV) must be positive.\n";
                return false;
            }
            a.ctacWindowEnergiesKeV.push_back(keV);
        }
        if (a.ctacWindowEnergiesKeV.empty()) {
            std::cerr << "Error: -ctacWindowEnergies needs at least one energy.\n";
            return false;
        }
        return true;
    }, {"-ctacWindowEnergies"});
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
    a.reconSectionsOverrideCli = cfg.getBool(
        "Compatibility", "recon_sections_override_cli",
        a.reconSectionsOverrideCli);

    a.isotope = canonicalIsotope(cfg.getString("Acquisition", "isotope", a.isotope));
    a.projectionModel = cfg.getString("Acquisition", "projection_model", a.projectionModel);
    if (!a.projectionModel.empty() && a.projectionModel != "single" &&
        a.projectionModel != "merged" && a.projectionModel != "joint")
        throw std::runtime_error("projection_model must be single, merged or joint.");
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
        if (!parseIterations(iters, a))
            throw std::runtime_error("Invalid [Reconstruction] iterations.");
    }
    const auto saveEvery = cfg.getString("Reconstruction", "save_interval");
    if (!saveEvery.empty())
        a.saveInterval = parseInteger(saveEvery, 0, "save_interval");
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
    a.bsremSoftMaskLambdaOut = cfg.getDouble(
        "Reconstruction", "bsrem_soft_mask_lambda_out", a.bsremSoftMaskLambdaOut);
    a.bsremMlemEquivalenceMode = cfg.getBool(
        "Reconstruction", "bsrem_mlem_equivalence", a.bsremMlemEquivalenceMode);
    a.bsremDiagnostics =
        cfg.getBool("Reconstruction", "bsrem_diagnostics", a.bsremDiagnostics);
    a.bsremDiagInterval =
        cfg.getInt("Reconstruction", "bsrem_diag_interval", a.bsremDiagInterval);
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
    if (cfg.hasKey("CTAC", "mu_map_energy_kev"))
        a.ctacMapEnergyKeV =
            cfg.getDouble("CTAC", "mu_map_energy_kev", a.ctacMapEnergyKeV);
    const std::string windowEnergies =
        cfg.getString("CTAC", "window_energies_kev");
    if (!windowEnergies.empty()) {
        a.ctacWindowEnergiesKeV.clear();
        for (auto& p : split(windowEnergies, ',')) {
            if (!p.empty())
                a.ctacWindowEnergiesKeV.push_back(std::stod(p));
        }
    }

    // IO
    a.outputDir = cfg.getString("InputOutput", "output_dir", a.outputDir);
    a.outputName = cfg.getString("InputOutput", "output_name", a.outputName);

    // Stable matrix IDs select a deployed directory; explicit legacy paths remain supported.
    a.matrixRoot = cfg.getString("SystemMatrix", "root", a.matrixRoot);
    if (cfg.hasKey("SystemMatrix", "matrix_id")) {
        a.matrixId = cfg.getString("SystemMatrix", "matrix_id");
        a.chunkCSRDir.clear();
        a.chunkCSRDirExplicit = false;
    }
    if (cfg.hasKey("SystemMatrix", "matrix_id2")) {
        a.matrixId2 = cfg.getString("SystemMatrix", "matrix_id2");
        a.chunkCSRDir2.clear();
        a.chunkCSRDir2Explicit = false;
    }
    // System Matrix
    std::string chunkDir =
        cfg.getString("SystemMatrix", "default_chunk_csr_dir");
    if (chunkDir.empty())
        chunkDir = cfg.getString("SystemMatrix", "chunk_csr_dir");
    if (chunkDir.empty())
        chunkDir = cfg.getString("Matrix", "chunk_csr_dir");
    if (!chunkDir.empty()) {
        a.chunkCSRDir = normalizePath(chunkDir);
        a.matrixId.clear();
        a.chunkCSRDirExplicit = false;
    }
    if (cfg.hasKey("SystemMatrix", "compact_denom_cache_path"))
        a.compactDenomCachePathExplicit = false;
    a.compactDenomCachePath = normalizePath(cfg.getString(
        "SystemMatrix", "compact_denom_cache_path",
        a.compactDenomCachePath));
    a.compactDenomCacheDir = normalizePath(cfg.getString(
        "SystemMatrix", "compact_denom_cache_dir", a.compactDenomCacheDir));
    // Joint dual-window: second matrix directory and per-window weights.
    const std::string chunkDir2 = cfg.getString("SystemMatrix", "chunk_csr_dir2");
    if (!chunkDir2.empty()) {
        a.chunkCSRDir2 = normalizePath(chunkDir2);
        a.matrixId2.clear();
        a.chunkCSRDir2Explicit = false;
    }
    a.csrqResidency =
        cfg.getString("SystemMatrix", "csrq_residency", a.csrqResidency);
    const std::string windowWeights =
        cfg.getString("SystemMatrix", "energy_window_weights");
    if (!windowWeights.empty()) {
        a.energyWindowWeights.clear();
        for (auto& p : split(windowWeights, ',')) {
            if (!p.empty())
                a.energyWindowWeights.push_back(std::stod(p));
        }
    }

    // [Algorithm] section
    std::string recipeAlg = cfg.getString("Algorithm", "name");
    if (!recipeAlg.empty()) {
        std::string recipeAlgLower = toLower(recipeAlg);
        if (recipeAlgLower == "mlem")
            a.algorithm = ReconArgs::Algorithm::MLEM;
        else if (recipeAlgLower == "bsrem")
            a.algorithm = ReconArgs::Algorithm::BSREM;
        else
            throw std::runtime_error("Unknown [Algorithm] name: " + recipeAlg);
    }
    std::string recipeIters = cfg.getString("Algorithm", "iterations");
    if (!recipeIters.empty()) {
        if (!parseIterations(recipeIters, a))
            throw std::runtime_error("Invalid [Algorithm] iterations.");
    }
    const auto recipeSaveEvery = cfg.getString("Algorithm", "save_interval");
    if (!recipeSaveEvery.empty())
        a.saveInterval = parseInteger(recipeSaveEvery, 0, "save_interval");
    if (!cfg.getString("Algorithm", "subset_mode").empty())
        throw std::runtime_error("[Algorithm] subset_mode is no longer supported; this build uses angle subsets.");
    const std::string subsets = cfg.getString("Algorithm", "subsets");
    if (!subsets.empty() && recipeIters.find(':') != std::string::npos)
        throw std::runtime_error("Use iterations=A:B or iterations=A with subsets, not both.");
    if (!subsets.empty())
        a.subsetSchedule = subsets;

    // [Regularization] section
    std::string regT = cfg.getString("Regularization", "type");
    if (!regT.empty()) {
        std::string regLower = toLower(regT);
        if (regLower == "none") {
            a.useReg = false;
            a.reg.type = RegType::None;
            a.regCtPath.clear();
        } else if (regLower == "tv")
            regT = "TVIsotropic";

        RegType rt = stringToRegType(regT);
        if (rt == RegType::Unknown)
            throw std::runtime_error("Unknown regularization type: " + regT);
        if (!isPublicRegularizer(rt)) {
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
        a.bsremSensFloorRel = (float)cfg.getDouble(
            "BSREM", "sens_floor_rel", a.bsremSensFloorRel);
        std::string kappaClip = cfg.getString("BSREM", "kappa_clip");
        if (!kappaClip.empty()) {
            auto kc = split(kappaClip, ',');
            if (kc.size() == 2) {
                a.bsremKappaClipMin = std::stod(kc[0]);
                a.bsremKappaClipMax = std::stod(kc[1]);
            }
        }
        a.bsremSoftMask = cfg.getBool("BSREM", "soft_mask", a.bsremSoftMask);
        a.bsremSoftMaskLambdaOut = cfg.getDouble("BSREM", "soft_mask_lambda", a.bsremSoftMaskLambdaOut);
        a.bsremMlemEquivalenceMode = cfg.getBool(
            "BSREM", "mlem_equivalence", a.bsremMlemEquivalenceMode);
        a.bsremDiagnostics =
            cfg.getBool("BSREM", "diagnostics", a.bsremDiagnostics);
        a.bsremDiagInterval =
            cfg.getInt("BSREM", "diag_interval", a.bsremDiagInterval);
    }


    // PSF
    std::string psfModeStr = cfg.getString("PSF", "mode");
    if (!psfModeStr.empty()) {
        std::string psfModeLower = toLower(psfModeStr);
        if (psfModeLower == "none")
            a.psf.mode = PSFMode::None;
        else if (psfModeLower == "operator")
            a.psf.mode = PSFMode::Operator;
        else if (psfModeLower == "sieve")
            a.psf.mode = PSFMode::Sieve;
        else {
            throw std::runtime_error(
                "PSF mode in config must be none, operator or sieve; got '" +
                psfModeStr + "'.");
        }
        a.usePSF = a.psf.mode != PSFMode::None;
        a.psfModeExplicit = true;
    }
    a.psf.fwhm_mm = cfg.getDouble("PSF", "fwhm", a.psf.fwhm_mm);
    a.psf.z_blur = cfg.getBool("PSF", "z_blur", a.psf.z_blur);

    std::string psfKernelStr = cfg.getString("PSF", "kernel");
    if (!psfKernelStr.empty()) {
        std::string k = toLower(psfKernelStr);
        if (k == "adjoint")
            a.psf.kernel = PSFKernel::Adjoint;
        else if (k == "castor")
            a.psf.kernel = PSFKernel::Castor;
        else {
            throw std::runtime_error(
                "PSF kernel in config must be adjoint or castor; got '" +
                psfKernelStr + "'.");
        }
    }

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
            a.cylinderDimsExplicit = true;
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

    a.mergeTimeSegments = cfg.getBool(
        "InputOutput", "merge_time_segments", a.mergeTimeSegments);
    a.outputFiltersOverwritePrimary = cfg.getBool(
        "InputOutput", "output_filters_overwrite_primary",
        a.outputFiltersOverwritePrimary);

    // Filtering
    if (cfg.hasSection("Filtering")) {
        const double bwCut = cfg.getDouble("Filtering", "butterworth_cutoff", a.useButter ? a.butterCutoff : 0.0);
        a.butterCutoff = static_cast<float>(bwCut);
        a.useButter = bwCut > 0.0;
        a.butterOrder = cfg.getInt("Filtering", "butterworth_order", a.butterOrder);
        const double gSigma = cfg.getDouble("Filtering", "gaussian_sigma", a.useSmooth ? a.smoothSigma : 0.0);
        a.smoothSigma = static_cast<float>(gSigma);
        a.useSmooth = gSigma > 0.0;
        a.smoothKernel = cfg.getInt("Filtering", "gaussian_kernel", a.smoothKernel);
        std::string smoothStage =
            toLower(cfg.getString("Filtering", "smooth_apply_stage"));
        if (!smoothStage.empty())
            a.smoothApplyStage = smoothStage;
        std::string butterStage =
            toLower(cfg.getString("Filtering", "butter_apply_stage"));
        if (!butterStage.empty())
            a.butterApplyStage = butterStage;
        a.outputSmoothFwhmMm = static_cast<float>(cfg.getDouble(
            "Filtering", "output_smooth_fwhm_mm", a.outputSmoothFwhmMm));
        a.outputSmoothSigmas = static_cast<float>(cfg.getDouble(
            "Filtering", "output_smooth_sigmas", a.outputSmoothSigmas));
        a.outputButterCutoffCpcm = static_cast<float>(cfg.getDouble(
            "Filtering", "output_butter_cutoff_cpcm",
            a.outputButterCutoffCpcm));
        a.outputButterOrder = cfg.getInt(
            "Filtering", "output_butter_order", a.outputButterOrder);
        a.outputNonnegativeClamp = cfg.getBool(
            "Filtering", "output_nonnegative_clamp",
            a.outputNonnegativeClamp);
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
        a.detZEdgeCensor = cfg.getInt(
            "MultiBed", "detz_edge_censor", a.detZEdgeCensor);
        a.detZRowWeightsFile = cfg.getString(
            "MultiBed", "detz_row_weights_file", a.detZRowWeightsFile);
        a.normFactorFile = cfg.getString(
            "MultiBed", "normalization_factor_file", a.normFactorFile);
        a.zBedFlip = cfg.getBool("MultiBed", "z_bed_flip", a.zBedFlip);
        const double iniStepMm = cfg.getDouble("MultiBed", "bed_step_mm", 0.0);
        if (iniStepMm > 0.0)
            a.bedStepMm = iniStepMm;
        a.singleBedStepAligned = cfg.getBool(
            "MultiBed", "single_bed_step_aligned", a.singleBedStepAligned);
        a.zOutFlip = cfg.getBool("MultiBed", "z_out_flip", a.zOutFlip);
        a.xOutFlip = cfg.getBool("MultiBed", "x_out_flip", a.xOutFlip);
        a.seamXyAlign = cfg.getBool("MultiBed", "seam_xy_align", a.seamXyAlign);
    }
    if (cfg.hasKey("Normalization", "factor_file") || cfg.hasKey("MultiBed", "normalization_factor_file"))
        a.normFactorFileExplicit = false;
    a.normFactorFile = cfg.getString("Normalization", "factor_file", a.normFactorFile);
    a.normFactorFile2 = cfg.getString("Normalization", "factor_file2", a.normFactorFile2);

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
         argName == "-df2" || argName == "-dfList2" || argName == "-bProj2" ||
         argName == "-bProjList2" || argName == "-chunkCSRDir2" ||
         argName == "-fout" || argName == "-o" || argName == "-output" ||
         argName == "-chunkCSRDir" ||
         argName == "-compactDenomCache" || argName == "-ctacFactorCacheDir" ||
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
    const auto colon = value.find(':');
    args.numIters = parseInteger(value.substr(0, colon), 1, "-it iterations");
    if (colon != std::string::npos)
        args.subsetSchedule = std::to_string(
            parseInteger(value.substr(colon + 1), 1, "-it subsets"));
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
    // Absolute output directories stay absolute so log + MHD/RAW never fall back to
    // a CWD-relative folder when `-fout`/`-o` carried a drive letter.
    if (!args.outputDir.empty()) {
        std::filesystem::path dir(args.outputDir);
        if (dir.is_absolute()) {
            std::error_code absEc;
            auto abs = std::filesystem::absolute(dir, absEc);
            args.outputDir =
                (absEc ? dir : abs).lexically_normal().make_preferred().string();
        }
    }
}

bool ArgumentParser::parseOutputPath(const std::string& path, ReconArgs& args) {
    try {
        const bool trailingSep =
            !path.empty() && (path.back() == '/' || path.back() == '\\');
        std::filesystem::path p = std::filesystem::path(path).lexically_normal();
        const bool inputAbsolute = p.is_absolute();

        std::error_code dirEc;
        const bool existingDir = std::filesystem::is_directory(p, dirEc);

        if (trailingSep || existingDir) {
            // `-fout D:/out/case/` or an existing directory: write into that
            // folder with the current/default name prefix.
            args.outputDir = p.string();
            if (args.outputName.empty())
                args.outputName = "recon";
        } else {
            // Strip volume extensions so `-fout out/run.mhd` becomes prefix
            // `run` → `run_itN.mhd`, not `run.mhd_itN.mhd`.
            std::filesystem::path name = p.filename();
            const auto ext = name.extension().string();
            if (ext == ".mhd" || ext == ".MHD" || ext == ".raw" ||
                ext == ".RAW" || ext == ".img" || ext == ".IMG")
                name = name.stem();
            args.outputName = name.string();
            const auto parent = p.parent_path();
            if (parent.empty()) {
                // `D:/run` → dir `D:/`; relative `run` → `.`
                args.outputDir =
                    inputAbsolute ? p.root_path().string() : std::string(".");
            } else {
                args.outputDir = parent.string();
            }
        }

        if (args.outputDir.empty())
            args.outputDir = inputAbsolute ? p.root_path().string() : ".";

        // Absolute -fout: keep an absolute outputDir so log and MHD/RAW both
        // land under that drive/UNC path, independent of process CWD.
        std::filesystem::path dir(args.outputDir);
        if (inputAbsolute || dir.is_absolute()) {
            if (dir.empty())
                dir = p.root_path();
            std::error_code absEc;
            auto abs = std::filesystem::absolute(dir, absEc);
            args.outputDir =
                (absEc ? dir : abs).lexically_normal().make_preferred().string();
        }
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
    if (args.csrqResidency != "auto" && args.csrqResidency != "streamed") {
        std::cerr << "Error: -csrqResidency expects auto|streamed, got '"
                  << args.csrqResidency << "'.\n";
        return false;
    }
    // Joint dual-window: the second window must mirror the first one bed for
    // bed and carry explicit weights (a silent default would bias the joint
    // likelihood toward whichever window the matrices happen to favour).
    const bool hasProj2 = !args.projPath2.empty() || !args.projPaths2.empty();
    const bool hasBProj2 = !args.bProjPath2.empty() || !args.bProjPaths2.empty();
    // CTAC energy scaling: the factor table is computed at the mu map's energy
    // and raised to mu_w(E_k)/mu_w(E_map) per window. Both energies must be
    // inside the water table when given.
    const size_t windowCount = args.dualWindow() ? 2u : 1u;
    if (!args.ctacWindowEnergiesKeV.empty()) {
        if (args.ctacWindowEnergiesKeV.size() != windowCount) {
            std::cerr << "Error: -ctacWindowEnergies lists " << args.ctacWindowEnergiesKeV.size()
                      << " energies for " << windowCount << " window(s).\n";
            return false;
        }
        double mu = 0.0;
        for (double keV : args.ctacWindowEnergiesKeV) {
            if (!spect::core::waterMuPerMMAtEnergyKeV(keV, mu)) {
                std::cerr << "Error: -ctacWindowEnergies " << keV
                          << " keV is outside the water attenuation table (20-1000 keV).\n";
                return false;
            }
        }
        if (args.ctacMapEnergyKeV > 0.0 &&
            !spect::core::waterMuPerMMAtEnergyKeV(args.ctacMapEnergyKeV, mu)) {
            std::cerr << "Error: -ctacMapEnergy " << args.ctacMapEnergyKeV
                      << " keV is outside the water attenuation table (20-1000 keV).\n";
            return false;
        }
        if (args.ctacMapEnergyKeV <= 0.0 &&
            spect::core::waterAttenuationPreset(args.ctacAutoMuNuclide) == nullptr) {
            std::cerr << "Error: -ctacWindowEnergies needs the mu map's energy: give "
                         "-ctacMapEnergy <keV> or a known [CTAC] auto_mu_nuclide.\n";
            return false;
        }
    }
    if (!args.dualWindow()) {
        if (hasProj2 || hasBProj2 || !args.energyWindowWeights.empty()) {
            std::cerr << "Error: -df2/-dfList2, -bProj2/-bProjList2 and "
                         "-energyWeights require -chunkCSRDir2.\n";
            return false;
        }
        return true;
    }
    if (!args.projPaths2.empty() && !args.projPath2.empty()) {
        std::cerr << "Error: Use either -df2 or -dfList2, not both.\n";
        return false;
    }
    if (!hasProj2) {
        std::cerr << "Error: -chunkCSRDir2 requires the second window's "
                     "projections (-df2 or -dfList2).\n";
        return false;
    }
    const size_t beds1 = args.projPaths.empty() ? 1u : args.projPaths.size();
    const size_t beds2 = args.projPaths2.empty() ? 1u : args.projPaths2.size();
    if (beds1 != beds2) {
        std::cerr << "Error: the second window must list the same number of "
                     "beds as the first (" << beds1 << " vs " << beds2 << ").\n";
        return false;
    }
    if (!args.bProjPaths2.empty() && !args.bProjPath2.empty()) {
        std::cerr << "Error: Use either -bProj2 or -bProjList2, not both.\n";
        return false;
    }
    if (args.energyWindowWeights.size() != 2 ||
        !(args.energyWindowWeights[0] > 0.0) ||
        !(args.energyWindowWeights[1] > 0.0)) {
        std::cerr << "Error: -energyWeights <w1,w2> (both > 0) is required with "
                     "-chunkCSRDir2; Lu recipe channel order 208/113: 0.1036,0.0617 when "
                     "the matrices are per emitted photon.\n";
        return false;
    }
    if (args.doCTAC && args.ctacWindowEnergiesKeV.size() != 2) {
        std::cerr << "Error: joint dual-window CTAC needs -ctacWindowEnergies "
                     "<keV1,keV2> (photopeak of each window, e.g. Lu-177 "
                     "208.4,112.9): mu differs per photopeak and the factor "
                     "table is scaled per window as att^(mu_w(E_k)/mu_w(E_map)).\n";
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
    if ((args.dualWindow() && (args.normFactorFile.empty() != args.normFactorFile2.empty())) ||
        (!args.dualWindow() && !args.normFactorFile2.empty())) {
        std::cerr << "Error: joint normalization requires both factor_file and factor_file2; single-window uses factor_file only.\n";
        return false;
    }
    if (!args.normFactorFile2.empty()) {
        if (!args.bProjPath2.empty() || !args.bProjPaths2.empty())
            throw std::runtime_error("Joint normalization currently requires background-free reconstruction.");
        const auto values = spect::io::loadDetZRowWeights(
            args.normFactorFile2, args.detNz, args.detNx, args.numAngles);
        if (values.size() != static_cast<size_t>(args.detNz) * args.detNx * args.numAngles)
            throw std::runtime_error("factor_file2 must contain one correction per projection bin.");
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
    if (args.bsremSensFloorRel < 0.0f || args.bsremSensFloorRel >= 1.0f) {
        std::cerr << "Error: BSREM sensitivity floor must be in [0, 1).\n";
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
    if (args.bsremSoftMask && args.bsremSoftMaskLambdaOut < 0.0) {
        std::cerr << "Error: BSREM soft-mask lambda must be >= 0.\n";
        return false;
    }
    if (args.bsremSoftMask && std::isnan(args.maskFromAttThreshold)) {
        std::cerr << "Error: BSREM soft-mask requires -maskFromAtt.\n";
        return false;
    }
    if (args.bsremDiagInterval <= 0) {
        std::cerr << "Error: -bsremDiagInterval must be > 0.\n";
        return false;
    }
    if (args.bsremMlemEquivalenceMode) {
        if (args.smoothApplyStage == "in_recon" ||
            args.butterApplyStage == "in_recon") {
            std::cerr << "Error: -bsremEquiv forbids in-recon smoothing/"
                         "Butterworth.\n";
            return false;
        }
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

    const auto isValidApplyStage = [](const std::string& stage) {
        return stage == "none" || stage == "in_recon" || stage == "output";
    };
    if (!isValidApplyStage(args.smoothApplyStage)) {
        std::cerr << "Error: smooth_apply_stage must be none|in_recon|output; got '"
                  << args.smoothApplyStage << "'.\n";
        return false;
    }
    if (!isValidApplyStage(args.butterApplyStage)) {
        std::cerr << "Error: butter_apply_stage must be none|in_recon|output; got '"
                  << args.butterApplyStage << "'.\n";
        return false;
    }
    if (args.smoothApplyStage == "in_recon" && args.smoothSigma <= 0) {
        std::cerr << "Error: smooth_apply_stage=in_recon requires "
                     "gaussian_sigma > 0 (or CLI -smooth).\n";
        return false;
    }
    if (args.butterApplyStage == "in_recon" && args.butterCutoff <= 0) {
        std::cerr << "Error: butter_apply_stage=in_recon requires "
                     "butterworth_cutoff > 0 (or CLI -butter).\n";
        return false;
    }
    if (args.smoothApplyStage == "output" &&
        (!std::isfinite(args.outputSmoothFwhmMm) ||
         args.outputSmoothFwhmMm <= 0.0f ||
         !std::isfinite(args.outputSmoothSigmas) ||
         args.outputSmoothSigmas <= 0.0f)) {
        std::cerr << "Error: smooth_apply_stage=output requires positive finite "
                     "output_smooth_fwhm_mm and output_smooth_sigmas.\n";
        return false;
    }
    if (args.butterApplyStage == "output" &&
        (!std::isfinite(args.outputButterCutoffCpcm) ||
         args.outputButterCutoffCpcm <= 0.0f ||
         args.outputButterOrder < 1)) {
        std::cerr << "Error: butter_apply_stage=output requires positive finite "
                     "output_butter_cutoff_cpcm and output_butter_order >= 1.\n";
        return false;
    }
    if (args.psf.mode == PSFMode::Sieve &&
        (args.smoothApplyStage == "in_recon" ||
         args.butterApplyStage == "in_recon")) {
        std::cerr << "Error: PSF mode sieve forbids in_recon filtering "
                     "(smooth/butter); sieves never filter the iteration "
                     "trajectory. Use smooth_apply_stage/butter_apply_stage "
                     "output or none.\n";
        return false;
    }

    if (args.psf.kernel == PSFKernel::Castor &&
        args.psf.mode != PSFMode::None) {
        // CASToR stationary Gaussian：横断面 x/y 必须各向同性（原型校验规则）。
        const double fx =
            args.psf.fwhm_x > 0.0 ? args.psf.fwhm_x : args.psf.fwhm_mm;
        const double fy =
            args.psf.fwhm_y > 0.0 ? args.psf.fwhm_y : args.psf.fwhm_mm;
        if (std::fabs(fx - fy) > 1e-6 * std::max(1.0, fx)) {
            std::cerr << "Error: -psfKernel castor requires equal x/y "
                         "transaxial FWHM (CASToR stationary Gaussian).\n";
            return false;
        }
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

    std::cout << "[Compatibility]\n";
    std::cout << "recon_sections_override_cli = "
              << (a.reconSectionsOverrideCli ? "true" : "false") << "\n\n";

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
    std::cout << "iterations = " << a.numIters << "\n";
    std::cout << "save_interval = " << a.saveInterval << "\n";
    std::cout << "subsets    = " << a.subsetSchedule << "\n\n";

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
        std::cout << "sens_floor_rel    = " << a.bsremSensFloorRel << "\n";
        std::cout << "kappa_clip        = " << a.bsremKappaClipMin << "," << a.bsremKappaClipMax << "\n";
        std::cout << "soft_mask         = " << (a.bsremSoftMask ? "true" : "false") << "\n";
        std::cout << "soft_mask_lambda  = " << a.bsremSoftMaskLambdaOut << "\n";
        std::cout << "mlem_equivalence  = "
                  << (a.bsremMlemEquivalenceMode ? "true" : "false") << "\n";
        std::cout << "diagnostics       = "
                  << (a.bsremDiagnostics ? "true" : "false") << "\n\n";
    }
    std::cout << "[Filtering]\n";
    std::cout << "butterworth_cutoff  = " << a.butterCutoff << "\n";
    std::cout << "butterworth_order   = " << a.butterOrder << "\n";
    std::cout << "gaussian_sigma      = " << a.smoothSigma << "\n";
    std::cout << "gaussian_kernel     = " << a.smoothKernel << "\n";
    std::cout << "smooth_apply_stage  = " << a.smoothApplyStage << "\n";
    std::cout << "butter_apply_stage  = " << a.butterApplyStage << "\n";
    std::cout << "output_smooth_fwhm_mm     = " << a.outputSmoothFwhmMm << "\n";
    std::cout << "output_smooth_sigmas      = " << a.outputSmoothSigmas << "\n";
    std::cout << "output_butter_cutoff_cpcm = " << a.outputButterCutoffCpcm << "\n";
    std::cout << "output_butter_order       = " << a.outputButterOrder << "\n";
    std::cout << "output_nonnegative_clamp  = "
              << (a.outputNonnegativeClamp ? "true" : "false") << "\n";
    std::cout << "\n";

    std::cout << "[PSF]\n";
    const char* psfModes[] = {"none", "operator", "sieve"};
    int psfIdx = static_cast<int>(a.psf.mode);
    std::cout << "mode = " << (psfIdx >= 0 && psfIdx <= 2 ? psfModes[psfIdx] : "unknown") << "\n";
    std::cout << "kernel = " << (a.psf.kernel == PSFKernel::Castor ? "castor" : "adjoint") << "\n";
    std::cout << "fwhm = " << a.psf.fwhm_mm << "\n";
    std::cout << "z_blur = " << (a.psf.z_blur ? "true" : "false") << "\n";
    std::cout << "min_sigmas = " << a.psf.min_sigmas << "\n\n";

    std::cout << "[CTAC]\n";
    std::cout << "auto_mu_nuclide = " << a.ctacAutoMuNuclide << "\n";
    std::cout << "auto_mu_value   = " << a.ctacAutoMuValue << "\n";
    std::cout << "auto_mu_source  = " << a.ctacAutoMuValueSource << "\n";
    std::cout << "mu_map_energy_kev = "
              << (a.ctacMapEnergyKeV > 0.0 ? std::to_string(a.ctacMapEnergyKeV)
                                            : std::string("(nuclide preset)"))
              << "\n";
    std::cout << "window_energies_kev = ";
    if (a.ctacWindowEnergiesKeV.empty()) {
        std::cout << "(none: table used as is)";
    } else {
        for (size_t k = 0; k < a.ctacWindowEnergiesKeV.size(); ++k)
            std::cout << (k ? "," : "") << a.ctacWindowEnergiesKeV[k];
    }
    std::cout << "\n\n";

    std::cout << "[Mask]\n";
    std::cout << "use_cylinder   = " << (a.useCylindricalMask ? "true" : "false") << "\n";
    std::cout << "cylinder_dims  = " << a.cylinderDiameterMM << "," << a.cylinderHeightMM << "\n\n";
    std::cout << "[Fixed6ModuleGeometry]\n";
    std::cout << "single_roi_mm = "
              << spect::core::sixmodule::kSingleBedRoiDiameterMm << ","
              << spect::core::sixmodule::kSingleBedRoiHeightMm << "\n";
    std::cout << "single_mask_xyz = "
              << spect::core::sixmodule::kOutputMaskNx << ","
              << spect::core::sixmodule::kOutputMaskNy << ","
              << (a.singleBedStepAligned
                      ? std::string("step+") + std::to_string(
                            spect::core::sixmodule::kDefaultBlendMarginLayers)
                      : std::to_string(
                            spect::core::sixmodule::kSingleBedMaskNz))
              << " layers\n";
    std::cout << "multibed_roi_mm = "
              << spect::core::sixmodule::kMultiBedRoiDiameterMm << ","
              << spect::core::sixmodule::kMultiBedRoiHeightMm << "\n";
    std::cout << "multibed_crop_xyz = "
              << spect::core::sixmodule::kOutputMaskNx << ","
              << spect::core::sixmodule::kOutputMaskNy << ","
              << "step+" << spect::core::sixmodule::kDefaultBlendMarginLayers << " layers\n";
    std::cout << "multibed_step_mm = "
              << spect::core::sixmodule::kMultiBedStepMm << "\n";

    std::cout << "[Backend]\n";
    std::cout << "sparse = cuda\n";
    std::cout << "butter = cufft\n\n";

    std::cout << "[Acquisition]\n";
    std::cout << "isotope = " << a.isotope << "\nprojection_model = " << a.projectionModel << "\n";
    std::cout << "[Normalization]\nfactor_file = " << a.normFactorFile
              << "\nfactor_file2 = " << a.normFactorFile2 << "\n";
    std::cout << "[SystemMatrix]\n";
    std::cout << "root = " << a.matrixRoot << "\nmatrix_id = " << a.matrixId
              << "\nmatrix_id2 = " << a.matrixId2 << "\n";
    std::cout << "compact_denom_cache_path = " << (a.compactDenomCachePath.empty() ? "(auto)" : a.compactDenomCachePath) << "\n";
    std::cout << "compact_denom_cache_dir = " << (a.compactDenomCacheDir.empty() ? "recon_out/compact_denom_cache (cwd legacy)" : a.compactDenomCacheDir) << "\n";
    std::cout << "chunk_csr_dir = " << (a.chunkCSRDir.empty() ? "(none)" : a.chunkCSRDir) << "\n";
    std::cout << "chunk_csr_dir2 = " << (a.chunkCSRDir2.empty() ? "(none)" : a.chunkCSRDir2) << "\n";
    std::cout << "energy_window_weights = ";
    if (a.energyWindowWeights.empty()) {
        std::cout << "(single window)";
    } else {
        for (size_t k = 0; k < a.energyWindowWeights.size(); ++k)
            std::cout << (k ? "," : "") << a.energyWindowWeights[k];
    }
    std::cout << "\n\n";

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
    std::cout << "stitch_method = center66_cosine10  # canonical fixed output stitch\n";
    std::cout << "bed_step_mm   = "
              << (a.bedStepMm > 0.0 ? a.bedStepMm
                                    : spect::core::sixmodule::kMultiBedStepMm)
              << "  # scanner bed step; crop = step+10, stride = step\n";
    std::cout << "single_bed_step_aligned = "
              << (a.singleBedStepAligned ? "true" : "false")
              << "  # single-bed crop = step+10; false keeps legacy 72 layers\n";
    std::cout << "detz_edge_censor = " << a.detZEdgeCensor
              << "  # detZ edge rows censored per side (0=off; compact chunk layout only)\n";
    std::cout << "detz_row_weights_file = "
              << (a.detZRowWeightsFile.empty() ? "(none)" : a.detZRowWeightsFile)
              << "  # detZ efficiency weights (detNz or detNz*detNx floats); sensitivity only\n";
    std::cout << "normalization_factor_file = "
              << (a.normFactorFile.empty() ? "(none)" : a.normFactorFile)
              << "  # full projection correction c; reciprocal 1/c enters sensitivity\n";
    std::cout << "crop_end     = " << a.cropEndStart << "," << a.cropEndEnd << "  # top,end slices zeroed\n";
    std::cout << "z_bed_flip   = " << (a.zBedFlip ? 1 : 0) << "\n";
    std::cout << "z_out_flip   = " << (a.zOutFlip ? 1 : 0) << "\n";
    std::cout << "x_out_flip   = " << (a.xOutFlip ? 1 : 0) << "\n";
    std::cout << "seam_xy_align = " << (a.seamXyAlign ? 1 : 0)
              << "  # overlap NCC integer XY shift; 0 keeps unshifted stitch\n\n";

    std::cout << "[InputOutput]\n";
    std::cout << "merge_time_segments = "
              << (a.mergeTimeSegments ? "true" : "false") << "\n";
    std::cout << "output_filters_overwrite_primary = "
              << (a.outputFiltersOverwritePrimary ? "true" : "false")
              << "\n";
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
    std::cout << "  merge_time_segments      [InputOutput] true + -df <timestamp/SPECT>: sum matching sibling BedN.bin inputs\n";
    std::cout << "  output_filters_overwrite_primary  [InputOutput] write output Smooth->Butter chain to the unsuffixed volume\n";
    std::cout << "  -chunkCSRDir <dir>       Directory with metadata.bin + chunk_*.csrb (or chunk_*.csrq)\n";
    std::cout << "                           Default: selected recipe matrix_id under EXE-local matrix root; explicit CLI follows config priority\n";
    std::cout << "  -chunkCSRDir2 <dir>      Second energy window CSRQ matrix for joint MLEM/OSEM/BSREM;\n";
    std::cout << "                           pair with -df2/-dfList2 (same beds), optional -bProj2/-bProjList2, and -energyWeights\n";
    std::cout << "  -energyWeights <w1,w2>   Window weights (mean_k = w_k*A_k*x + b_k), Lu recipe channel order 208/113: 0.1036,0.0617\n";
    std::cout << "  -csrqResidency <auto|streamed>  CSRQ matrices resident when they fit (auto) or always streamed from pinned host memory\n";
    std::cout << "  -compactDenomCache <mhd> Exact compact denominator cache path (subject to config priority)\n";
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
    std::cout << "  -it <A[:B]>             A full iterations, B subsets (not save interval)\n";
    std::cout << "  -save-every <N>         Save every N full iterations; 0 = final only\n";
    std::cout << "  -subsets <N|N1:i1,...>   Ordered subsets over compact angles (CSRB or joint CSRQ);\n";
    std::cout << "                           N | compact angles, later N | earlier N,\n";
    std::cout << "                           stage iterations sum to -it (default 1)\n";
    std::cout << "  -cudaDevice <id>         Select CUDA device (overrides SPECT_CUDA_DEVICE)\n\n";

    std::cout << "=== Algorithm: BSREM ===\n";
    std::cout << "  -bsrem <a0[,g]>          BSREM alpha0 and gamma\n";
    std::cout << "  -bsremCurv on|off        Curvature denominator D=S+beta*Dreg\n";
    std::cout << "  -bsremCurvFreeze on|off  Freeze curvature after warmup\n";
    std::cout << "  -bsremCurvWarmup <n>     Warmup iterations (default 2)\n";
    std::cout << "  -bsremCurvEps <v>        Stability epsilon (default 1e-6)\n";
    std::cout << "  -bsremSensFloor <rel>    Freeze voxels with s < rel*max(s) (default 0=off)\n";
    std::cout << "  -bsremKappa on|off       Kappa map beta_j=beta*kappa_j\n";
    std::cout << "  -bsremKappaEps <v>       Kappa epsilon (default 1e-6)\n";
    std::cout << "  -bsremKappaClip <min,max>  Kappa clip range (default 0.1,10)\n";
    std::cout << "  -bsremSoftMask on|off    Soft-mask penalty boosting\n";
    std::cout << "  -bsremSoftLambda <v>     Outside penalty boost (default 100)\n";
    std::cout << "  -bsremEquiv on|off       MLEM degeneration mode (beta=0, alpha=1, no clip)\n";
    std::cout << "  -bsremDiag on|off        Write iteration_metrics.csv\n";
    std::cout << "  -bsremDiagInterval <n>   Diagnostic stride after iter 10 (default 5)\n\n";

    std::cout << "=== Regularization ===\n";
    std::cout << "  -reg <type[,beta,delta]> TV | RDP\n";
    std::cout << "  -regBeta <beta>          Regularization weight\n";
    std::cout << "  -regDelta <delta>        Transition parameter\n\n";
    std::cout << "  -regCT <path>            Registered HU CT for TV/RDP edge weights\n";
    std::cout << "  -regCtSigma <value>      HU edge scale; set explicitly (e.g. 100)\n";
    std::cout << "  -regCtFloor <0..1>       Minimum CT edge weight (default 0.05)\n\n";

    std::cout << "=== Filtering ===\n";
    std::cout << "  -butter <fc[,order]>     Butterworth low-pass (default: 0.35,2); implies butter_apply_stage=in_recon\n";
    std::cout << "  -smooth <sigma>          Gaussian smoothing sigma (mm); implies smooth_apply_stage=in_recon\n";
    std::cout << "  -smoothKernel <n>        Gaussian kernel half-width\n";
    std::cout << "  INI [Filtering] stages:  smooth_apply_stage / butter_apply_stage = none|in_recon|output\n";
    std::cout << "  Output filters:          FWHM/sigmas in mm and cutoff in cycles/cm; suffixed copies stay independent\n";
    std::cout << "  Output clamp:            output_nonnegative_clamp=true|false (applied after each filter)\n\n";

    std::cout << "=== PSF ===\n";
    std::cout << "  -psf [fwhm_mm[,N]]       PSF Gaussian FWHM (mm); N*sigma truncation (defaults: 1.3,3)\n";
    std::cout << "  -psfMode <mode>          none | operator | sieve (sieve: operator in-loop + PSF-blurred save copies)\n";
    std::cout << "  -psfKernel <kind>        adjoint | castor (castor: strict CASToR erf kernel, z-only boundary norm, forward-reuse backward)\n";
    std::cout << "\n";
    std::cout << "  SPECT_CUDA_PSF_DOUBLE=1  Force double-precision CUDA PSF (default float)\n";
    std::cout << "  SPECT_CUDA_VOLUME_OPS=0  Disable CUDA volume ops; PSF-enabled reconstruction then fails\n";

    std::cout << "=== Geometry ===\n";
    std::cout << "  -dim <Nx,Ny,Nz>          Reconstruction grid (voxels)\n";
    std::cout << "  -vox <Vx,Vy,Vz>          Voxel size (mm)\n";
    std::cout << "  -ang <N>                 Projection angles\n";
    std::cout << "  -detN <Nz[,Nx]>          Detector channels\n\n";
    std::cout << "  -badDetectorFile <file>  Skip global 0..95 or link/FPGA/dector blocks in compact identity reconstruction\n\n";

    std::cout << "=== Multi-Bed ===\n";
    std::cout << "  -bedOverlap <b1,b2,b3,b4,b5> Fixed Bed1..Bed5 overlap table; slot b1=Bed1-Bed2\n";
    std::cout << "  -bedStepMm <mm>              Scanner bed step in mm (default 16.8); stride = round(mm/voxel_z)\n";
    std::cout << "  -singleBedStepAligned <0|1>  Single-bed crop = step+10 (default 1); 0 keeps legacy 72 layers\n";
    std::cout << "  -detZEdgeCensor <k>       Censor k detZ edge rows per side in compact numerator+denominator (default 0)\n";
    std::cout << "  -detZRowWeightsFile <f>   Efficiency weights: text detZ/per-bin/full projection; sensitivity only\n";
    std::cout << "  -normFactorFile <f>       Full projection correction factors c; applies reciprocal 1/c to sensitivity\n";
    std::cout << "  -cropEnd <top[,end]>      Zero final Z top/end slices after output flips\n";
    std::cout << "  -zbedflip <0|1>           Reverse local dfList bed Z (default 1; inside compact joint mapping for multi-bed)\n";
    std::cout << "  -zoutflip <0|1>           Flip whole output volume on Z after per-bed flip (default 0)\n";
    std::cout << "  -xoutflip <0|1>           Flip whole output volume on X (default 0)\n";
    std::cout << "  -seamXyAlign <0|1>        Integer XY align from overlap NCC before cosine blend (default 0)\n";
    std::cout << "  -ctac                     Enable iterative attenuation correction (uses -attMap; mu in mm^-1)\n";
    std::cout << "  -ctacModel <m>            line-integral (default; angle-dependent parallel-ray Int mu ds)\n";
    std::cout << "  -ctacRotSign <1|-1>       Rotation sign matching sysmat generator --voxel-rotation-sign (default 1)\n";
    std::cout << "  -ctacAutoMu <vol>         Build water mu-map from an early-iteration recon volume\n";
    std::cout << "                            (threshold + closing + hole fill; for activity-filled phantoms;\n";
    std::cout << "                            mutually exclusive with -attMap; generated map saved as *_automu)\n";
    std::cout << "  -ctacAutoMuThresh <f>     Relative threshold vs P99.9 activity (default 0.05)\n";
    std::cout << "  -ctacAutoMuValue <mu>     Debug override for auto-mu body fill, mm^-1\n";
    std::cout << "                            Default comes from [CTAC] auto_mu_nuclide (Tc99m => 0.0152)\n";
    std::cout << "  -ctacMapEnergy <keV>      Photon energy the mu map / auto-mu value refers to\n";
    std::cout << "                            (default: the auto_mu_nuclide preset energy)\n";
    std::cout << "  -ctacWindowEnergies <keV,...>  Photopeak energy per window; the CTAC factor table\n";
    std::cout << "                            is scaled per window as att^(mu_w(E_k)/mu_w(E_map)) (water\n";
    std::cout << "                            NIST table). Required for dual-window CTAC, e.g. 208.4,112.9\n\n";

    std::cout << "=== Masking ===\n";
    std::cout << "  -cyl <diam,height>        Cylindrical mask (mm)\n";
    std::cout << "  -cylc <cx,cy,cz>          Cylinder center (1-based voxel)\n";
    std::cout << "  -maskFromAtt <thresh>     Body mask from mu-map\n\n";

    std::cout << "=== Advanced ===\n";
    std::cout << "  -config <path>            Global INI or complete recipe (default: EXE-local config/spect_config.ini)\n";
    std::cout << "  -dumpConf                 Print merged config and exit\n";
    std::cout << "  Unknown/deprecated CLI arguments are noted and ignored.\n";
    std::cout << "  INI recon_sections_override_cli=true: recipe > global > CLI for configured reconstruction fields; false: CLI > recipe > global.\n";
    std::cout << "  -projectionResidual [on|off]  Write final compact projection residual CSV\n";
    std::cout << "  -vb <0|1|2>               Verbosity level (default 1)\n";
    std::cout << "  -v                        Print version and complete changelog, then exit\n";
    std::cout << "                            (-v <n> is rejected; use -vb)\n";
    std::cout << "  --version                 Print one-line version string and exit\n";
    std::cout << "  -th <N>                   Thread count\n";
    std::cout << "  -h, -help, --help         Show this help\n\n";
    std::cout << "  SPECT_FAIL_ON_ZERO_CHECKPOINT=1  Abort before writing an all-zero checkpoint\n";

    std::cout << "  Algorithm-specific defaults are loaded beside the global config or from config/algo_<name>.conf.\n";
    std::cout << "  Override any value with CLI flags. Use -dumpConf to inspect.\n";
    std::cout << "  See: multipinhole/doc/USAGE_CN.md for full parameter reference.\n\n";
}

} // namespace cli
} // namespace spect
