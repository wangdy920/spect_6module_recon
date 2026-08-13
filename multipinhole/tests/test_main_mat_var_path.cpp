#include <gtest/gtest.h>

#include "io/MatVarPath.h"

#include <string>

namespace spect {
namespace main {
namespace test {

TEST(MatVarPathTest, ExtractsVarFromMatPath) {
  std::string path = "proj.mat:matrix";
  const std::string var = extractMatVarName(path);
  EXPECT_EQ(path, "proj.mat");
  EXPECT_EQ(var, "matrix");
}

TEST(MatVarPathTest, KeepsWindowsDriveLetterPath) {
  std::string path = "D:/data/proj.mat";
  const std::string var = extractMatVarName(path);
  EXPECT_EQ(path, "D:/data/proj.mat");
  EXPECT_TRUE(var.empty());
}

TEST(MatVarPathTest, RejectsPathLikeVarSuffix) {
  std::string path = "proj.mat:sub/dir";
  const std::string var = extractMatVarName(path);
  EXPECT_EQ(path, "proj.mat:sub/dir");
  EXPECT_TRUE(var.empty());
}

TEST(MatVarPathTest, ResolveFallsBackWhenMissingVar) {
  std::string path = "proj.mat";
  const std::string var = resolveMatVarName(path, "matrix");
  EXPECT_EQ(path, "proj.mat");
  EXPECT_EQ(var, "matrix");
}

} // namespace test
} // namespace main
} // namespace spect

