// DetZRowWeights.h - Projection-domain efficiency weights loader
// Weights normalize the compact sensitivity (denominator) only; the measured
// projection data is never scaled by them.
#pragma once

#include <cmath>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace spect {
namespace io {

/**
 * @brief Load detector efficiency weights.
 *
 * Text files hold whitespace-separated values: detNz values (one per detZ
 * row), detNz*detNx values (one per detector bin), or
 * numAngles*detNz*detNx values (one per projection bin). A .bin file uses the
 * projection layout <uint32 numAngles><uint32 raysPerAngle><float64 values>.
 * All values must be finite and > 0.
 *
 * Detector-bin order is detZ-major, and full projection order is angle-major.
 * Short forms are broadcast over angle and/or detX by the runtime caller.
 *
 * @param path  Weights file path.
 * @param detNz Detector Z rows (0 = unknown, defer count check).
 * @param detNx Detector X channels (0 = unknown, defer count check).
 * @param numAngles Projection angles (0 = unknown, defer angle/count check).
 * @return The loaded weight values.
 * @throws std::runtime_error on unreadable file, empty file, invalid token,
 *         non-finite or non-positive value, or wrong value count.
 */
inline std::vector<double> loadDetZRowWeights(const std::string& path,
                                              int detNz, int detNx,
                                              int numAngles = 0) {
    auto validateValues = [&path](const std::vector<double>& values) {
        if (values.empty()) {
            throw std::runtime_error(
                "normalization weights file is empty: " + path);
        }
        for (double value : values) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::runtime_error(
                    "normalization weights must be finite and > 0: " + path);
            }
        }
    };

    std::string extension;
    const size_t dot = path.find_last_of('.');
    if (dot != std::string::npos)
        extension = path.substr(dot);
    for (char& ch : extension) {
        ch = static_cast<char>(
            std::tolower(static_cast<unsigned char>(ch)));
    }
    if (extension == ".bin") {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) {
            throw std::runtime_error(
                "normalization weights file is not readable: " + path);
        }
        const std::streamoff fileSize = in.tellg();
        in.seekg(0);
        std::uint32_t header[2] = {};
        in.read(reinterpret_cast<char*>(header), sizeof(header));
        if (!in || header[0] == 0 || header[1] == 0) {
            throw std::runtime_error(
                "normalization projection BIN has an invalid header: " +
                path);
        }
        const std::uint64_t count =
            static_cast<std::uint64_t>(header[0]) * header[1];
        const std::uint64_t expectedSize =
            sizeof(header) + count * sizeof(double);
        if (expectedSize >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::streamoff>::max()) ||
            fileSize != static_cast<std::streamoff>(expectedSize)) {
            throw std::runtime_error(
                "normalization projection BIN size does not match its "
                "header: " + path);
        }
        if (detNz > 0 && detNx > 0 &&
            header[1] != static_cast<std::uint32_t>(detNz * detNx)) {
            throw std::runtime_error(
                "normalization projection BIN rays_per_angle does not match "
                "detNz*detNx: " + path);
        }
        if (numAngles > 0 &&
            header[0] != static_cast<std::uint32_t>(numAngles)) {
            throw std::runtime_error(
                "normalization projection BIN num_angles does not match the "
                "reconstruction: " + path);
        }
        if (count > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error(
                "normalization projection BIN is too large: " + path);
        }
        std::vector<double> values(static_cast<size_t>(count));
        in.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(count * sizeof(double)));
        if (!in) {
            throw std::runtime_error(
                "failed reading normalization projection BIN: " + path);
        }
        validateValues(values);
        return values;
    }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("detZ row weights file is not readable: " +
                                 path);
    }
    std::vector<double> values;
    std::string token;
    while (in >> token) {
        double value = 0.0;
        try {
            size_t parsed = 0;
            value = std::stod(token, &parsed);
            if (parsed != token.size())
                throw std::runtime_error("trailing characters");
        } catch (const std::exception&) {
            throw std::runtime_error(
                "detZ row weights file has an invalid value '" + token +
                "': " + path);
        }
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error(
                "detZ row weights must be finite and > 0 (got '" + token +
                "'): " + path);
        }
        values.push_back(value);
    }
    if (!in.eof()) {
        throw std::runtime_error("failed reading detZ row weights file: " +
                                 path);
    }
    validateValues(values);
    if (detNz > 0 && detNx > 0) {
        const size_t rows = static_cast<size_t>(detNz);
        const size_t bins = rows * static_cast<size_t>(detNx);
        const size_t projectionBins =
            numAngles > 0 ? bins * static_cast<size_t>(numAngles) : 0;
        if (values.size() != rows && values.size() != bins &&
            values.size() != projectionBins) {
            throw std::runtime_error(
                "detZ row weights file must hold exactly " +
                std::to_string(rows) + " (per-row) or " +
                std::to_string(bins) + " (per-bin)" +
                (numAngles > 0
                     ? " or " + std::to_string(projectionBins) +
                           " (per-projection-bin)"
                     : "") +
                " values, got " +
                std::to_string(values.size()) + ": " + path);
        }
    }
    return values;
}

} // namespace io
} // namespace spect
