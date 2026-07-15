/**
 * @file MHDWriter.h
 * @brief MetaImage (MHD/RAW) format writer
 * 
 * Migrated from original_source_backup/dataLoader.h
 */

#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <cstdint>
#include <type_traits>
#include <cstring>

namespace spect {
namespace io {

/**
 * @brief Get element type string for MHD
 */
inline const char* elementTypeTag(bool isFloat) {
    return isFloat ? "MET_FLOAT" : "MET_DOUBLE";
}

inline bool isLittleEndianHost() {
    const uint16_t x = 1;
    return *reinterpret_cast<const uint8_t*>(&x) == 1;
}

template <typename T>
inline T byteswapValue(T value) {
    static_assert(std::is_trivially_copyable_v<T>, "byteswapValue requires trivially copyable type");
    T out{};
    const auto* src = reinterpret_cast<const unsigned char*>(&value);
    auto* dst = reinterpret_cast<unsigned char*>(&out);
    for (size_t i = 0; i < sizeof(T); ++i)
        dst[i] = src[sizeof(T) - 1 - i];
    return out;
}

/**
 * @brief Write binary file
 */
inline bool writeFile(const std::string& filePath, const char* data, size_t bytes) {
    std::ofstream f(filePath, std::ios::binary);
    if (!f) return false;
    f.write(data, static_cast<std::streamsize>(bytes));
    return f.good();
}

template <typename T>
inline bool writeRawWithEndian(const std::string& filePath,
                               const std::vector<T>& data,
                               bool little_endian) {
    std::ofstream f(filePath, std::ios::binary);
    if (!f) return false;

    const bool hostLittle = isLittleEndianHost();
    if (little_endian == hostLittle) {
        f.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size() * sizeof(T)));
        return f.good();
    }

    std::vector<T> swapped;
    swapped.reserve(data.size());
    for (const auto& v : data)
        swapped.push_back(byteswapValue(v));
    f.write(reinterpret_cast<const char*>(swapped.data()),
            static_cast<std::streamsize>(swapped.size() * sizeof(T)));
    return f.good();
}

/**
 * @brief 写入 3D 图像为 MHD/RAW 格式
 * 
 * @param path 输出文件路径（不含扩展名）
 * @param data 图像数据（float，列主序）
 * @param Nx, Ny, Nz 图像尺寸
 * @param spacing_x, spacing_y, spacing_z 体素间距（mm）
 * @param little_endian 字节序
 */
void writeMHD(
    const std::string &path,
    const std::vector<float> &data,
    int Nx, int Ny, int Nz,
    float spacing_x = 1.0f,
    float spacing_y = 1.0f,
    float spacing_z = 1.0f,
    bool little_endian = true
);

/**
 * @brief 写入 3D 图像为 MHD/RAW 格式（double 版本）
 */
void writeMHD(
    const std::string &path,
    const std::vector<double> &data,
    int Nx, int Ny, int Nz,
    float spacing_x = 1.0f,
    float spacing_y = 1.0f,
    float spacing_z = 1.0f,
    bool little_endian = true
);

/**
 * @brief 保存 MHD/RAW 文件（带迭代次数后缀）
 * 
 * @param vol 图像数据
 * @param Nx, Ny, Nz 图像尺寸
 * @param vx, vy, vz 体素间距（mm）
 * @param outdir 输出目录
 * @param prefix 文件名前缀
 * @param iter 迭代次数（<0 表示不加后缀）
 * @param little_endian 字节序
 */
inline void saveMHDRAW(
    const std::vector<double>& vol,
    int Nx, int Ny, int Nz,
    double vx, double vy, double vz,
    const std::string& outdir,
    const std::string& prefix,
    int iter,
    bool little_endian = true)
{
    namespace fs = std::filesystem;
    std::string stem = prefix;
    if (iter >= 0)
        stem += std::string("_it") + std::to_string(iter);
    writeMHD((fs::path(outdir) / stem).string(), vol, Nx, Ny, Nz,
             static_cast<float>(vx), static_cast<float>(vy),
             static_cast<float>(vz), little_endian);
}

} // namespace io
} // namespace spect
