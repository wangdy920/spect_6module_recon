/**
 * @file Types.h
 * @brief 核心类型定义、状态码、基础宏
 * 
 * 此文件不应包含任何第三方头文件(matio/cufft/omp等)
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace spect {

/**
 * @brief 操作状态码
 */
enum class Status {
    Ok = 0,
    InvalidArgument,
    FileNotFound,
    IOError,
    OutOfMemory,
    InvalidFormat,
    NotImplemented,
    InternalError
};

/**
 * @brief 状态码转字符串
 */
inline const char* statusToString(Status s) {
    switch (s) {
        case Status::Ok: return "Ok";
        case Status::InvalidArgument: return "InvalidArgument";
        case Status::FileNotFound: return "FileNotFound";
        case Status::IOError: return "IOError";
        case Status::OutOfMemory: return "OutOfMemory";
        case Status::InvalidFormat: return "InvalidFormat";
        case Status::NotImplemented: return "NotImplemented";
        case Status::InternalError: return "InternalError";
        default: return "Unknown";
    }
}

/**
 * @brief 执行后端类型
 */
enum class BackendType {
    Reference,  // 标量参考实现
    CPU,        // AVX2 + OpenMP
    CUDA        // GPU (预留)
};

/**
 * @brief 精度类型
 */
enum class Precision {
    Float32,
    Float64
};

/**
 * @brief 重建算法类型
 */
enum class AlgorithmType {
    MLEM,
    OSEM,
    BSREM
};

/**
 * @brief 三维尺寸
 */
struct Dims3D {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    
    size_t total() const { return static_cast<size_t>(nx) * ny * nz; }
};

/**
 * @brief 体素尺寸
 */
struct VoxelSize {
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;
};

/**
 * @brief 坐标偏移
 */
struct Offset3D {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

} // namespace spect
