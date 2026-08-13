/**
 * @file VolumeLoader.h
 * @brief Volume image loading (MHD/HDR/Interfile formats)
 * 
 * Migrated from original_source_backup/dataLoader.h
 */

#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include <cmath>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace spect {
namespace io {

/**
 * @brief Volume metadata
 */
struct VolumeInfo {
    int Nx = 0, Ny = 0, Nz = 0;
    float dx = 1.0f, dy = 1.0f, dz = 1.0f;
    std::string rawPath;
    bool isFloat = true;
    bool littleEndian = true;
    int headerOffset = 0;
};

/**
 * @brief Parse MHD (MetaImage) header file
 */
inline VolumeInfo parseMHD(const std::string &path) {
    VolumeInfo info;
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open MHD file: " + path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("NDims") != std::string::npos &&
            line.find("3") == std::string::npos)
            throw std::runtime_error("MHD must be 3D");
        if (line.rfind("DimSize", 0) == 0) {
            std::stringstream ss(line.substr(line.find('=') + 1));
            ss >> info.Nx >> info.Ny >> info.Nz;
        }
        if (line.rfind("ElementSpacing", 0) == 0) {
            std::stringstream ss(line.substr(line.find('=') + 1));
            ss >> info.dx >> info.dy >> info.dz;
        }
        if (line.rfind("ElementType", 0) == 0) {
            if (line.find("MET_DOUBLE") != std::string::npos)
                info.isFloat = false;
        }
        if (line.rfind("ElementDataFile", 0) == 0) {
            std::string raw = line.substr(line.find('=') + 1);
            raw.erase(0, raw.find_first_not_of(" \t\r\n"));
            raw.erase(raw.find_last_not_of(" \t\r\n") + 1);
            std::filesystem::path p(path);
            info.rawPath = (p.parent_path() / raw).string();
        }
    }
    return info;
}

/**
 * @brief Parse Analyze 7.5 HDR header file
 */
inline VolumeInfo parseHDR(const std::string &path) {
    VolumeInfo info;
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open HDR file: " + path);

    auto readShort = [&](int off) -> int16_t {
        f.seekg(off, std::ios::beg);
        int16_t v;
        f.read((char*)&v, 2);
        return v;
    };

    int16_t dim[8];
    f.seekg(40, std::ios::beg);
    f.read((char*)dim, 16);
    info.Nx = dim[1];
    info.Ny = dim[2];
    info.Nz = dim[3];

    float pixdim[8];
    f.seekg(72, std::ios::beg);
    f.read((char*)pixdim, 32);
    info.dx = pixdim[1];
    info.dy = pixdim[2];
    info.dz = pixdim[3];

    int16_t dtype = readShort(70);
    if (dtype == 16) info.isFloat = true;       // FLOAT
    else if (dtype == 64) info.isFloat = false; // DOUBLE
    else info.isFloat = false;

    std::filesystem::path p(path);
    info.rawPath = (p.parent_path() / p.stem()).string() + ".img";
    return info;
}

/**
 * @brief Parse Interfile 3.3 header
 */
inline VolumeInfo parseInterfile(const std::string &path) {
    VolumeInfo info;
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open Interfile: " + path);

    std::string line;
    std::string dataFileName;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == ';')
            continue;
        size_t assignPos = line.find(":=");
        if (assignPos == std::string::npos)
            continue;

        std::string key = line.substr(0, assignPos);
        std::string val = line.substr(assignPos + 2);

        auto trim = [](std::string &s) {
            s.erase(0, s.find_first_not_of(" ![]\t\r\n"));
            if (!s.empty())
                s.erase(s.find_last_not_of(" \t\r\n") + 1);
        };
        trim(key);
        trim(val);

        if (key == "matrix size [1]") info.Nx = std::stoi(val);
        else if (key == "matrix size [2]") info.Ny = std::stoi(val);
        else if (key == "matrix size [3]") info.Nz = std::stoi(val);
        else if (key == "scaling factor (mm/pixel) [1]") info.dx = std::stof(val);
        else if (key == "scaling factor (mm/pixel) [2]") info.dy = std::stof(val);
        else if (key == "scaling factor (mm/pixel) [3]") info.dz = std::stof(val);
        else if (key == "number format") {
            info.isFloat = (val.find("float") != std::string::npos);
        }
        else if (key == "name of data file") {
            dataFileName = val;
        }
    }

    std::filesystem::path p(path);
    if (!dataFileName.empty()) {
        info.rawPath = (p.parent_path() / dataFileName).string();
    } else {
        info.rawPath = (p.parent_path() / p.stem()).string() + ".img";
    }
    return info;
}

/**
 * @brief Check if file is binary Analyze format (vs text Interfile)
 */
inline bool isBinaryAnalyze(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    char buf[4] = {0};
    if (!f.read(buf, 4))
        return false;
    if (buf[0] == '!')
        return false;
    return true;
}

/**
 * @brief Load 3D volume from MHD/HDR/Interfile
 * 
 * @param path Path to header file (.mhd or .hdr)
 * @param Nx, Ny, Nz Output dimensions
 * @return Volume data as double vector (column-major)
 */
inline std::vector<double> loadVolume(const std::string &path, int &Nx, int &Ny, int &Nz) {
    VolumeInfo info;
    std::string ext = std::filesystem::path(path).extension().string();
    
    if (ext == ".mhd") {
        info = parseMHD(path);
    } else if (ext == ".hdr") {
        if (isBinaryAnalyze(path)) {
            info = parseHDR(path);
        } else {
            info = parseInterfile(path);
        }
    } else {
        throw std::runtime_error("Unknown volume format: " + ext);
    }

    Nx = info.Nx;
    Ny = info.Ny;
    Nz = info.Nz;
    size_t numel = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz);

    if (numel > 1024ULL * 1024ULL * 1024ULL) {
        std::cerr << "[NOTE] Loading huge volume: " << numel << " voxels.\n";
    }

    std::vector<double> out(numel);
    std::ifstream raw(info.rawPath, std::ios::binary);
    if (!raw)
        throw std::runtime_error("Cannot open RAW file: " + info.rawPath);

    if (info.isFloat) {
        const size_t CHUNK_SIZE = 1024 * 1024;
        std::vector<float> buf(std::min(numel, CHUNK_SIZE));
        size_t processed = 0;

        while (processed < numel) {
            size_t remain = numel - processed;
            size_t nRead = std::min(remain, buf.size());
            raw.read(reinterpret_cast<char*>(buf.data()), nRead * sizeof(float));
            if (!raw)
                throw std::runtime_error("Unexpected EOF reading RAW file");
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(nRead); ++i)
                out[processed + i] = static_cast<double>(buf[i]);
            processed += nRead;
        }
    } else {
        raw.read(reinterpret_cast<char*>(out.data()), numel * sizeof(double));
        if (!raw)
            throw std::runtime_error("Unexpected EOF reading RAW file");
    }
    return out;
}

/**
 * @brief Load volume with spacing info
 */
inline std::vector<double> loadVolumeWithSpacing(
    const std::string &path,
    int &Nx, int &Ny, int &Nz,
    float &dx, float &dy, float &dz
) {
    VolumeInfo info;
    std::string ext = std::filesystem::path(path).extension().string();
    
    if (ext == ".mhd") {
        info = parseMHD(path);
    } else if (ext == ".hdr") {
        if (isBinaryAnalyze(path)) {
            info = parseHDR(path);
        } else {
            info = parseInterfile(path);
        }
    } else {
        throw std::runtime_error("Unknown volume format: " + ext);
    }

    Nx = info.Nx;
    Ny = info.Ny;
    Nz = info.Nz;
    dx = info.dx;
    dy = info.dy;
    dz = info.dz;
    
    size_t numel = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz);
    std::vector<double> out(numel);
    std::ifstream raw(info.rawPath, std::ios::binary);
    if (!raw)
        throw std::runtime_error("Cannot open RAW file: " + info.rawPath);

    if (info.isFloat) {
        std::vector<float> buf(numel);
        raw.read(reinterpret_cast<char*>(buf.data()), numel * sizeof(float));
        if (!raw)
            throw std::runtime_error("Unexpected EOF reading RAW file: " + info.rawPath);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(numel); ++i)
            out[i] = static_cast<double>(buf[i]);
    } else {
        raw.read(reinterpret_cast<char*>(out.data()), numel * sizeof(double));
        if (!raw)
            throw std::runtime_error("Unexpected EOF reading RAW file: " + info.rawPath);
    }
    return out;
}

} // namespace io
} // namespace spect
