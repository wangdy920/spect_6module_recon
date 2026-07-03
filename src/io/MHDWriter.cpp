#include "io/MHDWriter.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace spect {
namespace io {

void writeMHD(const std::string &path, const std::vector<float> &data, int Nx,
              int Ny, int Nz, float spacing_x, float spacing_y,
              float spacing_z, bool little_endian) {
  namespace fs = std::filesystem;
  fs::path outPath(path);
  fs::create_directories(outPath.parent_path());

  fs::path rawPath = outPath;
  rawPath.replace_extension(".raw");
  fs::path mhdPath = outPath;
  mhdPath.replace_extension(".mhd");

  if (!writeRawWithEndian(rawPath.string(), data, little_endian))
    throw std::runtime_error("Failed writing raw file: " + rawPath.string());

  std::ofstream mhd(mhdPath.string());
  if (!mhd)
    throw std::runtime_error("Failed writing MHD file: " + mhdPath.string());

  mhd << "ObjectType = Image\n";
  mhd << "NDims = 3\n";
  mhd << "DimSize = " << Nx << ' ' << Ny << ' ' << Nz << "\n";
  mhd << "ElementSpacing = " << spacing_x << ' ' << spacing_y << ' ' << spacing_z << "\n";
  mhd << "ElementType = " << elementTypeTag(true) << "\n";
  mhd << "ElementByteOrderMSB = " << (little_endian ? "False" : "True") << "\n";
  mhd << "HeaderSize = 0\n";
  mhd << "ElementDataFile = " << rawPath.filename().string() << "\n";
}

void writeMHD(const std::string &path, const std::vector<double> &data, int Nx,
              int Ny, int Nz, float spacing_x, float spacing_y,
              float spacing_z, bool little_endian) {
  namespace fs = std::filesystem;
  fs::path outPath(path);
  fs::create_directories(outPath.parent_path());

  fs::path rawPath = outPath;
  rawPath.replace_extension(".raw");
  fs::path mhdPath = outPath;
  mhdPath.replace_extension(".mhd");

  if (!writeRawWithEndian(rawPath.string(), data, little_endian))
    throw std::runtime_error("Failed writing raw file: " + rawPath.string());

  std::ofstream mhd(mhdPath.string());
  if (!mhd)
    throw std::runtime_error("Failed writing MHD file: " + mhdPath.string());

  mhd << "ObjectType = Image\n";
  mhd << "NDims = 3\n";
  mhd << "DimSize = " << Nx << ' ' << Ny << ' ' << Nz << "\n";
  mhd << "ElementSpacing = " << spacing_x << ' ' << spacing_y << ' ' << spacing_z << "\n";
  mhd << "ElementType = " << elementTypeTag(false) << "\n";
  mhd << "ElementByteOrderMSB = " << (little_endian ? "False" : "True") << "\n";
  mhd << "HeaderSize = 0\n";
  mhd << "ElementDataFile = " << rawPath.filename().string() << "\n";
}

} // namespace io
} // namespace spect
