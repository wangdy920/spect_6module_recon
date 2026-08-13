#pragma once

namespace spect::cli {

// Refactor contract, 2026-07-03:
// 0 = successful completion or help/dump-only command.
// 1 = user/configuration/input validation error.
// 2 = runtime failure after valid inputs were accepted.
// Current CLI still exposes the historical 0/1 process contract; future parser
// cleanup may map internal failures onto these names without changing scripts.
inline constexpr int kExitSuccess = 0;
inline constexpr int kExitUserError = 1;
inline constexpr int kExitRuntimeError = 2;

} // namespace spect::cli
