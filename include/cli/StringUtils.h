// StringUtils.h - String utility functions for CLI parsing
// Extracted from SPECT-recon.cpp for modular CLI architecture
#pragma once

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

namespace spect {
namespace cli {

/**
 * @brief Split a string by a single character separator.
 * @param s Input string.
 * @param sep Separator character.
 * @return Vector of non-empty tokens.
 */
inline std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, sep)) {
        if (!tok.empty())
            out.push_back(tok);
    }
    return out;
}

/**
 * @brief Trim whitespace from both ends of a string.
 * @param s Input string.
 * @return Trimmed copy of the string.
 */
inline std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
        ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])))
        --end;
    return s.substr(start, end - start);
}

/**
 * @brief Split a path list by comma or semicolon.
 * @param s Input string with paths separated by ',' or ';'.
 * @return Vector of trimmed, non-empty paths.
 */
inline std::vector<std::string> splitPathList(const std::string& s) {
    std::vector<std::string> out;
    std::string current;
    for (char ch : s) {
        if (ch == ',' || ch == ';') {
            std::string trimmed = trim(current);
            if (!trimmed.empty())
                out.push_back(trimmed);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    std::string trimmed = trim(current);
    if (!trimmed.empty())
        out.push_back(trimmed);
    return out;
}

/**
 * @brief Normalize path separators to forward slashes.
 * @param path Input path string.
 * @return Path with all backslashes converted to forward slashes.
 */
inline std::string normalizePath(const std::string& path) {
    std::string normalized = path;
    for (size_t i = 0; i < normalized.length(); ++i) {
        if (normalized[i] == '\\')
            normalized[i] = '/';
    }
    return normalized;
}

/**
 * @brief Convert string to lowercase.
 * @param s Input string.
 * @return Lowercase copy of the string.
 */
inline std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

/**
 * @brief Extract an optional MATLAB variable name from "path:var" format.
 * 
 * Handles Windows drive letters (e.g., "D:/file.mat:matrix") correctly.
 * 
 * @param path Input path (modified to remove the :var suffix if found).
 * @return The variable name, or empty string if not found.
 */
inline std::string takeVarName(std::string& path) {
    const size_t pos = path.rfind(':');
    if (pos == std::string::npos)
        return {};

    // Ignore Windows drive letters like "C:/..." or "C:\\..."
    if (pos == 1 && path.size() >= 3 &&
        std::isalpha(static_cast<unsigned char>(path[0])) &&
        (path[2] == '/' || path[2] == '\\')) {
        return {};
    }

    const std::string var = path.substr(pos + 1);
    if (var.empty())
        return {};
    if (var.find('/') != std::string::npos || var.find('\\') != std::string::npos)
        return {};

    path = path.substr(0, pos);
    return var;
}

/**
 * @brief Parse a boolean value from string.
 * @param s Input string (on/off, true/false, yes/no, 1/0).
 * @param out Output boolean value.
 * @return true if parsing succeeded, false otherwise.
 */
inline bool parseOnOff(const std::string& s, bool& out) {
    const std::string v = toLower(s);
    if (v == "on" || v == "true" || v == "1" || v == "yes") {
        out = true;
        return true;
    }
    if (v == "off" || v == "false" || v == "0" || v == "no") {
        out = false;
        return true;
    }
    return false;
}

} // namespace cli
} // namespace spect
