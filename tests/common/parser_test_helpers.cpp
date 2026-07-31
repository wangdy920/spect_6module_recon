#include "tests/common/parser_test_helpers.h"

#include "cli/ArgumentParser.h"

#include <filesystem>
#include <fstream>

namespace spect {
namespace io {
namespace test {

bool parseReconArgsForTest(const std::vector<std::string>& args,
                           spect::cli::ReconArgs& parsed) {
    std::vector<std::string> storage = args;
    const auto emptyConfig = std::filesystem::temp_directory_path() /
                             "spect_parser_empty_config.ini";
    bool usesEmptyConfig = false;
    for (std::string& arg : storage) {
        if (arg == "__missing_compact_test_config__.ini") {
            std::ofstream(emptyConfig, std::ios::trunc).close();
            arg = emptyConfig.string();
            usesEmptyConfig = true;
        }
    }
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (std::string& arg : storage) {
        argv.push_back(arg.data());
    }
    spect::cli::ArgumentParser parser;
    const bool ok = parser.parse(static_cast<int>(argv.size()), argv.data(), parsed) ==
                    spect::cli::ArgumentParser::ParseResult::Success;
    if (usesEmptyConfig) {
        std::error_code ec;
        std::filesystem::remove(emptyConfig, ec);
    }
    return ok;
}

std::vector<std::string> compactBaseArgs() {
    return {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-df", "proj.bin",
        "-chunkCSRDir", "chunks",
        "-opti", "mlem",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };
}

} // namespace test
} // namespace io
} // namespace spect
