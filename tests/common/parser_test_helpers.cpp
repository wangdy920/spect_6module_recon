#include "tests/common/parser_test_helpers.h"

#include "cli/ArgumentParser.h"

namespace spect {
namespace io {
namespace test {

bool parseReconArgsForTest(const std::vector<std::string>& args,
                           spect::cli::ReconArgs& parsed) {
    std::vector<std::string> storage = args;
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (std::string& arg : storage) {
        argv.push_back(arg.data());
    }
    spect::cli::ArgumentParser parser;
    return parser.parse(static_cast<int>(argv.size()), argv.data(), parsed) ==
           spect::cli::ArgumentParser::ParseResult::Success;
}

std::vector<std::string> compactBaseArgs() {
    return {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-df", "proj.bin",
        "-loadCSR", "matrix.csrb",
        "-matrixLayout", "compact-symmetric",
        "-sparseBackend", "cuda",
        "-alg", "mlem",
        "-subsets", "1",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };
}

} // namespace test
} // namespace io
} // namespace spect
