#include <gtest/gtest.h>

#include "postprocess/SmoothingOps.h"

#include <vector>

namespace spect::main::test {

TEST(SmoothingOpsTest, ClampNonNegativeInPlaceZerosNegativeValues) {
  std::vector<double> values = {-1.0, 0.5, -0.25, 2.0, 0.0};

  clampNonNegativeInPlace(values, 1);

  ASSERT_EQ(values.size(), static_cast<size_t>(5));
  EXPECT_DOUBLE_EQ(values[0], 0.0);
  EXPECT_DOUBLE_EQ(values[1], 0.5);
  EXPECT_DOUBLE_EQ(values[2], 0.0);
  EXPECT_DOUBLE_EQ(values[3], 2.0);
  EXPECT_DOUBLE_EQ(values[4], 0.0);
}

} // namespace spect::main::test
