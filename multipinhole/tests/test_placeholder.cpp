/**
 * @file test_placeholder.cpp
 * @brief 占位测试文件，验证GoogleTest框架配置正确
 * 
 * 后续任务将添加具体测试:
 * - T0.2: CSRBinary/CSRMap测试
 * - T0.3: SpMV正确性测试
 * - T0.4: Golden重建测试
 */

#include <gtest/gtest.h>

namespace spect {
namespace test {

/**
 * @brief 验证测试框架正常工作
 */
TEST(FrameworkTest, BasicAssertion) {
    EXPECT_EQ(1 + 1, 2);
    EXPECT_TRUE(true);
}

/**
 * @brief 验证浮点比较功能
 */
TEST(FrameworkTest, FloatingPointComparison) {
    float a = 1.0f / 3.0f;
    float b = 0.333333f;
    EXPECT_NEAR(a, b, 1e-5f);
}

/**
 * @brief 验证字符串比较功能
 */
TEST(FrameworkTest, StringComparison) {
    std::string expected = "SPECT-recon";
    std::string actual = "SPECT-recon";
    EXPECT_EQ(expected, actual);
}

}  // namespace test
}  // namespace spect
