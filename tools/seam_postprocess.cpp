#include "io/MHDWriter.h"
#include "io/VolumeLoader.h"
#include "main/VolumeOutput.h"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc < 4 || argc > 5) {
    std::cerr << "Usage: spect_seam_postprocess <input.mhd> "
                 "<output-stem> <phantom|zimu|mouse|derenzo|unknown> "
                 "[derenzo-strength]\n";
    return 1;
  }
  try {
    int nx = 0, ny = 0, nz = 0;
    auto volume = spect::io::loadVolume(argv[1], nx, ny, nz);
    const auto info = spect::io::parseMHD(argv[1]);
    const std::string profile = argv[3];
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
