// ArgumentParser.h - Command-line argument parser for SPECT reconstruction
// Extracted from SPECT-recon.cpp for modular CLI architecture
#pragma once

#include "ConfigParser.h"
#include "ReconArgs.h"
#include "StringUtils.h"
#include "core/BedPathUtils.h"
#include "core/ReconLimits.h"

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
    enum class ParseResult {
        Success,
        ExitSuccess,
        ExitFailure,
    };

    ArgumentParser() { registerHandlers(); }

    /**
     * @brief Parse command-line arguments.
     * @param argc Argument count.
     * @param argv Argument vector.
     * @param args Output ReconArgs structure.
     * @return ParseResult describing whether reconstruction should continue.
     */
    ParseResult parse(int argc, char** argv, ReconArgs& args);

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
     * @brief Parse iteration specification (iterations[:subsets]).
     */
    static bool parseIterations(const std::string& value, ReconArgs& args);

    /**
     * @brief Parse output path into directory and filename.
     */
    static bool parseOutputPath(const std::string& path, ReconArgs& args);

    /**
     * @brief Validate all arguments after parsing.
     */
    static bool validateArgs(const ReconArgs& args);

    static bool validateIoArgs(const ReconArgs& args);
    static bool validateGeometryArgs(const ReconArgs& args);
    static bool validateAlgorithmArgs(const ReconArgs& args);

    /**
     * @brief Apply layout-specific defaults after parsing (e.g. compact → CUDA).
     */
    static void normalizeArgs(ReconArgs& args);

    void bindHandlerAliases(const Handler& handler,
                            std::initializer_list<const char*> flags);

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

};

} // namespace cli
} // namespace spect
