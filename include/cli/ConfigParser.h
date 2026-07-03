// ConfigParser.h - INI-style configuration file parser
// Extracted from SPECT-recon.h for modular CLI architecture
#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <string>

namespace spect {
namespace cli {

/**
 * @brief Simple INI-style configuration file parser.
 * 
 * Supports section-based configuration files with key=value pairs.
 * Comments starting with ';' or '#' are ignored.
 * 
 * Example INI file:
 * @code
 * [Basic]
 * dimensions = 200,200,200
 * voxel_size = 1.0,1.0,1.0
 * threads = 8
 * @endcode
 */
class ConfigParser {
public:
    using SectionMap = std::map<std::string, std::string>;
    using ConfigMap = std::map<std::string, SectionMap>;

    ConfigParser() = default;

    /**
     * @brief Load configuration from an INI file.
     * @param filename Path to the configuration file.
     * @return true if file was successfully loaded, false otherwise.
     */
    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open())
            return false;

        std::string line, currentSection;
        while (std::getline(file, line)) {
            stripInlineComment(line);
            trimInPlace(line);

            if (line.empty() || line[0] == ';' || line[0] == '#')
                continue;

            if (line[0] == '[' && line.back() == ']') {
                currentSection = line.substr(1, line.length() - 2);
            } else {
                size_t eq = line.find('=');
                if (eq != std::string::npos) {
                    std::string key = line.substr(0, eq);
                    std::string val = line.substr(eq + 1);
                    trimInPlace(key);
                    trimInPlace(val);

                    if (!currentSection.empty()) {
                        sections[currentSection][key] = val;
                    }
                }
            }
        }
        return true;
    }

    /**
     * @brief Get a string value from the configuration.
     * @param sec Section name.
     * @param key Key name within the section.
     * @param def Default value if key not found.
     * @return The value or default.
     */
    std::string getString(const std::string& sec, const std::string& key,
                          const std::string& def = "") const {
        auto sit = sections.find(sec);
        if (sit != sections.end()) {
            auto kit = sit->second.find(key);
            if (kit != sit->second.end())
                return kit->second;
        }
        return def;
    }

    /**
     * @brief Get an integer value from the configuration.
     * @param sec Section name.
     * @param key Key name within the section.
     * @param def Default value if key not found or invalid.
     * @return The integer value or default.
     */
    int getInt(const std::string& sec, const std::string& key, int def) const {
        std::string val = getString(sec, key);
        if (val.empty())
            return def;
        try {
            return std::stoi(val);
        } catch (...) {
            return def;
        }
    }

    /**
     * @brief Get a double value from the configuration.
     * @param sec Section name.
     * @param key Key name within the section.
     * @param def Default value if key not found or invalid.
     * @return The double value or default.
     */
    double getDouble(const std::string& sec, const std::string& key,
                     double def) const {
        std::string val = getString(sec, key);
        if (val.empty())
            return def;
        try {
            return std::stod(val);
        } catch (...) {
            return def;
        }
    }

    /**
     * @brief Get a boolean value from the configuration.
     * @param sec Section name.
     * @param key Key name within the section.
     * @param def Default value if key not found or invalid.
     * @return The boolean value or default.
     * 
     * Recognizes: true/True/1/on as true, false/False/0/off as false.
     */
    bool getBool(const std::string& sec, const std::string& key, bool def) const {
        std::string val = getString(sec, key);
        if (val.empty())
            return def;
        if (val == "true" || val == "True" || val == "1" || val == "on")
            return true;
        if (val == "false" || val == "False" || val == "0" || val == "off")
            return false;
        return def;
    }

    /**
     * @brief Check if a section exists.
     * @param sec Section name.
     * @return true if section exists.
     */
    bool hasSection(const std::string& sec) const {
        return sections.find(sec) != sections.end();
    }

    /**
     * @brief Check if a key exists in a section.
     * @param sec Section name.
     * @param key Key name.
     * @return true if key exists in section.
     */
    bool hasKey(const std::string& sec, const std::string& key) const {
        auto sit = sections.find(sec);
        if (sit != sections.end()) {
            return sit->second.find(key) != sit->second.end();
        }
        return false;
    }

    /**
     * @brief Get all sections.
     * @return Reference to the internal sections map.
     */
    const ConfigMap& getSections() const { return sections; }

private:
    static void trimInPlace(std::string& s) {
        const size_t first = s.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            s.clear();
            return;
        }
        const size_t last = s.find_last_not_of(" \t\r\n");
        s = s.substr(first, last - first + 1);
    }

    static void stripInlineComment(std::string& s) {
        const size_t semi = s.find(';');
        const size_t hash = s.find('#');
        size_t pos = std::string::npos;
        if (semi != std::string::npos)
            pos = semi;
        if (hash != std::string::npos)
            pos = (pos == std::string::npos) ? hash : std::min(pos, hash);
        if (pos != std::string::npos)
            s.erase(pos);
    }

    ConfigMap sections;
};

} // namespace cli
} // namespace spect
