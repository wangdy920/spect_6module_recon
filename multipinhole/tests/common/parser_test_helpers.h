#pragma once

#include <string>
#include <vector>

namespace spect {
namespace cli {
struct ReconArgs;
} // namespace cli
namespace io {
namespace test {

bool parseReconArgsForTest(const std::vector<std::string>& args,
                           spect::cli::ReconArgs& parsed);

std::vector<std::string> compactBaseArgs();

} // namespace test
} // namespace io
} // namespace spect
