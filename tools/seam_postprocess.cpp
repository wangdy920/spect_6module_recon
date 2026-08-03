#include "io/MHDWriter.h"
#include "io/VolumeLoader.h"
#include "postprocess/SeamPostprocess.h"
#include "pipeline/VolumeOutput.h"

#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<double> parseTriplet(const std::string &text) {
  std::vector<double> values;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::size_t stop = comma == std::string::npos ? text.size() : comma;
    values.push_back(std::stod(text.substr(start, stop - start)));
    if (comma == std::string::npos)
      break;
    start = stop + 1;
  }
  return values;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4 || argc > 5) {
    std::cerr << "Usage: spect_seam_postprocess <input.mhd> "
                 "<output-stem> "
                 "<phantom|zimu|mouse|derenzo|adhesion|classify|unknown> "
                 "[derenzo-strength | inner,outer,damp]\n";
    return 1;
  }
  try {
    int nx = 0, ny = 0, nz = 0;
    auto volume = spect::io::loadVolume(argv[1], nx, ny, nz);
    const auto info = spect::io::parseMHD(argv[1]);
    const std::string profile = argv[3];
    if (profile == "classify") {
      std::cout << spect::main::classifyOutputObjectProfile(volume, nx, ny, nz,
                                                            info.dx, info.dy)
                << '\n';
      return 0;
    }
    if (profile == "adhesion") {
      double inner = 0.02;
      double outer = 0.10;
      double damp = 0.55;
      if (argc == 5) {
        const auto values = parseTriplet(argv[4]);
        if (values.size() != 3)
          throw std::invalid_argument(
              "adhesion mode expects inner,outer,damp");
        inner = values[0];
        outer = values[1];
        damp = values[2];
      }
      spect::main::postprocessDerenzoAdhesion(volume, nx, ny, nz, info.dx,
                                              info.dy, inner, outer, damp);
      spect::io::writeMHD(argv[2], volume, nx, ny, nz, info.dx, info.dy,
                          info.dz, true);
      return 0;
    }
    if (profile == "derenzo") {
      const double strength = argc == 5 ? std::stod(argv[4]) : 0.30;
      spect::main::postprocessDerenzoOutput(
          volume, nx, ny, nz, info.dx, info.dy, info.dz, strength);
      spect::io::writeMHD(argv[2], volume, nx, ny, nz, info.dx, info.dy,
                          info.dz, true);
      return 0;
    }
    if (nz != 272 && nz != 320)
      throw std::invalid_argument(
          "non-Derenzo mode expects Z=272 fused or Z=320 full-bed volume");

    spect::main::OutputTransform transform;
    transform.objectAwareSeamPostprocess = true;
    transform.objectProfile = profile;
    transform.phantomCloseStrength = 1.0;
    transform.zimuCloseStrength = 0.5;
    transform.seamHalfWidth = 8;
    transform.baseNz = 80;
    transform.bedZStarts = {0, 80, 160, 240};
    transform.outputBedOverlaps = {16, 16, 16};
    if (nz == 320) {
      transform.sigmoidBlendOutput = true;
      int outputNz = 0;
      volume = spect::main::prepareOutputVolume(
          volume, nx, ny, nz, nullptr, spect::main::CropSettings{}, &outputNz,
          &transform);
      nz = outputNz;
    } else {
      spect::main::postprocessStitchedOutput(volume, nx, ny, nz, transform);
    }

    spect::io::writeMHD(argv[2], volume, nx, ny, nz, info.dx, info.dy,
                        info.dz, true);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 2;
  }
}
