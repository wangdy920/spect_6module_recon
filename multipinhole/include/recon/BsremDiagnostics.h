#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace spect {

struct PoissonObjectiveStats {
  double dataObjective = 0.0;
  size_t lowMuCount = 0;
  size_t invalidMuCount = 0;
};

/// Φ_data = Σ_i [μ_i - y_i log μ_i], ignoring log(y!).
/// y=0 contributes μ. y>0 and μ<=muMin counts as invalid and is not replaced
/// with a silent epsilon.
PoissonObjectiveStats accumulatePoissonDataObjective(
    const double *mu, const double *y, size_t count, double muMin = 0.0);

struct BsremIterationMetrics {
  int iteration = 0;
  double alpha = 1.0;
  double beta = 0.0;
  double kappaMean = 1.0;
  double lambdaMean = 0.0;
  double dataObjective = 0.0;
  bool dataObjectiveValid = false;
  double regularizerRaw = 0.0;
  double regularizerWeighted = 0.0;
  double totalObjective = 0.0;
  bool totalObjectiveValid = false;
  double projectedGradL2OverS = 0.0;
  double projectedGradLinf = 0.0;
  double relativeImageChange = 0.0;
  double roiMean = 0.0;
  double roiSum = 0.0;
  double roiCv = 0.0;
  double imageMin = 0.0;
  double imageMax = 0.0;
  size_t nonzeroCount = 0;
  size_t atBoundCount = 0;
  size_t lowMuCount = 0;
  size_t invalidMuCount = 0;
  size_t invalidGradCount = 0;
  size_t invalidCurvatureCount = 0;
  size_t negativeCurvatureCount = 0;
  size_t clampNegativeCount = 0;
  size_t clampPositiveCount = 0;
  size_t clampNegativeRoiCount = 0;
  size_t clampPositiveRoiCount = 0;
  double relativeStepP50 = 0.0;
  double relativeStepP95 = 0.0;
  double relativeStepP99 = 0.0;
  double relativeStepMax = 0.0;
  double preconditionedAbsP50 = 0.0;
  double preconditionedAbsP95 = 0.0;
  double preconditionedAbsP99 = 0.0;
  double preconditionedAbsMax = 0.0;
  double curvatureOverS_p50 = 0.0;
  double curvatureOverS_p95 = 0.0;
  double curvatureOverS_p99 = 0.0;
};

struct BsremIterationInputs {
  const std::vector<double> *image = nullptr;
  const std::vector<double> *imagePrev = nullptr;
  const std::vector<double> *numerator = nullptr;
  const std::vector<double> *globalDenom = nullptr;
  const std::vector<double> *gradient = nullptr;
  const std::vector<double> *curvature = nullptr;
  const std::vector<double> *kappa = nullptr;
  const std::vector<double> *rawRelativeSteps = nullptr;
  const std::vector<double> *preconditionedSteps = nullptr;
  const std::vector<uint8_t> *mask = nullptr;
  double alpha = 1.0;
  double beta = 0.0;
  double xMin = 0.0;
  double xBoundEps = 1e-12;
  double sMin = 1e-12;
  double clipMin = -0.9;
  double clipMax = 1.0;
  double regularizerRaw = 0.0;
  const PoissonObjectiveStats *poisson = nullptr;
};

BsremIterationMetrics computeBsremIterationMetrics(
    int iteration, const BsremIterationInputs &in);

std::string bsremIterationMetricsCsvHeader();
std::string formatBsremIterationMetricsCsv(const BsremIterationMetrics &m);

bool shouldRecordBsremDiagnostics(int iteration, int numIters, int interval);

} // namespace spect
