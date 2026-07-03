/**
 * @file test_main.cpp
 * @brief GoogleTest 主入口文件
 * 
 * 所有测试套件共享此入口点
 */

#include <gtest/gtest.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
