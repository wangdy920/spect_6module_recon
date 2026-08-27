#include "recon/BsremDiagnostics.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace spect {
namespace {

double percentileSortedAbs(std::vector<double> values, double p) {
  if (values.empty())
    return 0.0;
  for (double &v : values)
    v = std::abs(v);
  std::sort(values.begin(), values.end());
  const double idx = p * static_cast<double>(values.size() - 1);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = std::min(lo + 1, values.size() - 1);
  const double w = idx - static_cast<double>(lo);
  return values[lo] * (1.0 - w) + values[hi] * w;
}

bool inRoi(const std::vector<uint8_t> *mask, const std::vector<double> *s,
           size_t i, double sMin) {
  if (mask && (*mask)[i] == 0)
    return false;
  if (s && (*s)[i] <= sMin)
    return false;
  return true;
}

} // namespace

PoissonObjectiveStats accumulatePoissonDataObjective(
    const double *mu, const double *y, size_t count, double muMin) {
  if ((count > 0) && (mu == nullptr || y == nullptr))
    throw std::runtime_error("accumulatePoissonDataObjective: null buffer");
  PoissonObjectiveStats stats;
  for (size_t i = 0; i < count; ++i) {
    const double mui = mu[i];
    const double yi = y[i];
    if (!std::isfinite(mui) || !std::isfinite(yi) || mui < 0.0) {
      ++stats.invalidMuCount;
      continue;
    }
    if (yi == 0.0) {
      stats.dataObjective += mui;
      if (mui <= muMin)
        ++stats.lowMuCount;
      continue;
    }
    if (yi < 0.0) {
      ++stats.invalidMuCount;
      continue;
    }
    if (mui <= muMin) {
      ++stats.invalidMuCount;
      continue;
    }
    stats.dataObjective += mui - yi * std::log(mui);
  }
  return stats;
}

BsremIterationMetrics computeBsremIterationMetrics(
    int iteration, const BsremIterationInputs &in) {
  if (!in.image || !in.globalDenom)
    throw std::runtime_error(
        "computeBsremIterationMetrics: image and denom required");
  const size_t n = in.image->size();
  if (in.globalDenom->size() != n)
    throw std::runtime_error(
        "computeBsremIterationMetrics: denom size mismatch");

  BsremIterationMetrics m;
  m.iteration = iteration;
  m.alpha = in.alpha;
  m.beta = in.beta;
  m.regularizerRaw = in.regularizerRaw;
  if (in.poisson) {
    m.dataObjective = in.poisson->dataObjective;
    m.dataObjectiveValid = true;
    m.lowMuCount = in.poisson->lowMuCount;
    m.invalidMuCount = in.poisson->invalidMuCount;
  }

  double kappaSum = 0.0;
  size_t kappaCount = 0;
  double gP2 = 0.0;
  double gPinf = 0.0;
  double s2 = 0.0;
  double x2 = 0.0;
  double dx2 = 0.0;
  double roiSum = 0.0;
  double roiSum2 = 0.0;
  size_t roiCount = 0;
  m.imageMin = std::numeric_limits<double>::infinity();
  m.imageMax = -std::numeric_limits<double>::infinity();

  std::vector<double> curvOverS;
  curvOverS.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    const double xi = (*in.image)[i];
    const double si = (*in.globalDenom)[i];
    s2 += si * si;
    x2 += xi * xi;
    m.imageMin = std::min(m.imageMin, xi);
    m.imageMax = std::max(m.imageMax, xi);
    if (xi > 0.0)
      ++m.nonzeroCount;
    if (xi <= in.xMin + in.xBoundEps)
      ++m.atBoundCount;
    if (in.imagePrev && in.imagePrev->size() == n) {
      const double d = xi - (*in.imagePrev)[i];
      dx2 += d * d;
    }
    const double kappa = in.kappa ? (*in.kappa)[i] : 1.0;
    if (std::isfinite(kappa)) {
      kappaSum += kappa;
      ++kappaCount;
    }
    if (in.gradient && !std::isfinite((*in.gradient)[i]))
      ++m.invalidGradCount;
    if (in.curvature) {
      const double c = (*in.curvature)[i];
      if (!std::isfinite(c))
        ++m.invalidCurvatureCount;
      else if (c < 0.0)
        ++m.negativeCurvatureCount;
      const double lambda = in.beta * kappa;
      if (si + 1e-30 > 0.0)
        curvOverS.push_back(lambda * std::max(c, 0.0) / (si + 1e-30));
    }

    const double nval = (in.numerator && in.numerator->size() == n)
                            ? (*in.numerator)[i]
                            : 0.0;
    const double r = (in.gradient && in.gradient->size() == n)
                         ? (*in.gradient)[i]
                         : 0.0;
    const double g = si - nval + in.beta * kappa * r;
    const bool frozen = (xi <= in.xMin + in.xBoundEps) && (g > 0.0);
    const double gP = frozen ? 0.0 : g;
    gP2 += gP * gP;
    gPinf = std::max(gPinf, std::abs(gP));

    const bool roi = inRoi(in.mask, in.globalDenom, i, in.sMin);
    if (roi) {
      roiSum += xi;
      roiSum2 += xi * xi;
      ++roiCount;
    }
    if (in.rawRelativeSteps && in.rawRelativeSteps->size() == n) {
      const double u = (*in.rawRelativeSteps)[i];
      if (u < in.clipMin) {
        ++m.clampNegativeCount;
        if (roi)
          ++m.clampNegativeRoiCount;
      }
      if (u > in.clipMax) {
        ++m.clampPositiveCount;
        if (roi)
          ++m.clampPositiveRoiCount;
      }
    }
  }

  m.kappaMean =
      kappaCount > 0 ? kappaSum / static_cast<double>(kappaCount) : 1.0;
  m.lambdaMean = in.beta * m.kappaMean;
  m.regularizerWeighted = m.lambdaMean * m.regularizerRaw;
  if (m.dataObjectiveValid) {
    m.totalObjective = m.dataObjective + m.regularizerWeighted;
    m.totalObjectiveValid = true;
  }
  m.projectedGradL2OverS = std::sqrt(gP2) / (std::sqrt(s2) + 1e-30);
  m.projectedGradLinf = gPinf;
  m.relativeImageChange = std::sqrt(dx2) / (std::sqrt(x2) + 1e-30);
  if (roiCount > 0) {
    m.roiSum = roiSum;
    m.roiMean = roiSum / static_cast<double>(roiCount);
    const double mean2 = m.roiMean * m.roiMean;
    const double var =
        std::max(0.0, roiSum2 / static_cast<double>(roiCount) - mean2);
    m.roiCv = m.roiMean != 0.0 ? std::sqrt(var) / std::abs(m.roiMean) : 0.0;
  }
  if (!std::isfinite(m.imageMin)) {
    m.imageMin = 0.0;
    m.imageMax = 0.0;
  }

  if (in.rawRelativeSteps && in.rawRelativeSteps->size() == n) {
    std::vector<double> steps = *in.rawRelativeSteps;
    m.relativeStepP50 = percentileSortedAbs(steps, 0.50);
    m.relativeStepP95 = percentileSortedAbs(steps, 0.95);
    m.relativeStepP99 = percentileSortedAbs(steps, 0.99);
    m.relativeStepMax = percentileSortedAbs(steps, 1.0);
  }
  if (in.preconditionedSteps && in.preconditionedSteps->size() == n) {
    std::vector<double> steps = *in.preconditionedSteps;
    m.preconditionedAbsP50 = percentileSortedAbs(steps, 0.50);
    m.preconditionedAbsP95 = percentileSortedAbs(steps, 0.95);
    m.preconditionedAbsP99 = percentileSortedAbs(steps, 0.99);
    m.preconditionedAbsMax = percentileSortedAbs(steps, 1.0);
  }
  if (!curvOverS.empty()) {
    m.curvatureOverS_p50 = percentileSortedAbs(curvOverS, 0.50);
    m.curvatureOverS_p95 = percentileSortedAbs(curvOverS, 0.95);
    m.curvatureOverS_p99 = percentileSortedAbs(curvOverS, 0.99);
  }
  return m;
}

std::string bsremIterationMetricsCsvHeader() {
  return "iteration,alpha,beta,kappa,lambda,data_objective,regularizer_raw,"
         "regularizer_weighted,total_objective,projected_grad_l2,"
         "projected_grad_linf,relative_image_change,roi_mean,roi_sum,roi_cv,"
         "image_min,image_max,low_mu_count,invalid_mu_count,invalid_grad_count,"
         "invalid_curvature_count,negative_curvature_count,"
         "clamp_negative_count,clamp_positive_count,clamp_negative_roi_count,"
         "clamp_positive_roi_count,relative_step_p50,relative_step_p95,"
         "relative_step_p99,relative_step_max,curvature_over_s_p50,"
         "curvature_over_s_p95,curvature_over_s_p99";
}

std::string formatBsremIterationMetricsCsv(const BsremIterationMetrics &m) {
  std::ostringstream os;
  os << std::setprecision(17) << m.iteration << ',' << m.alpha << ',' << m.beta
     << ',' << m.kappaMean << ',' << m.lambdaMean << ',';
  if (m.dataObjectiveValid)
    os << m.dataObjective;
  os << ',' << m.regularizerRaw << ',' << m.regularizerWeighted << ',';
  if (m.totalObjectiveValid)
    os << m.totalObjective;
  os << ',' << m.projectedGradL2OverS << ',' << m.projectedGradLinf << ','
     << m.relativeImageChange << ',' << m.roiMean << ',' << m.roiSum << ','
     << m.roiCv << ',' << m.imageMin << ',' << m.imageMax << ','
     << m.lowMuCount << ',' << m.invalidMuCount << ',' << m.invalidGradCount
     << ',' << m.invalidCurvatureCount << ',' << m.negativeCurvatureCount
     << ',' << m.clampNegativeCount << ',' << m.clampPositiveCount << ','
     << m.clampNegativeRoiCount << ',' << m.clampPositiveRoiCount << ','
     << m.relativeStepP50 << ',' << m.relativeStepP95 << ','
     << m.relativeStepP99 << ',' << m.relativeStepMax << ','
     << m.curvatureOverS_p50 << ',' << m.curvatureOverS_p95 << ','
     << m.curvatureOverS_p99;
  return os.str();
}

bool shouldRecordBsremDiagnostics(int iteration, int numIters, int interval) {
  if (iteration <= 10 || iteration == numIters)
    return true;
  return interval > 0 && (iteration % interval == 0);
}

} // namespace spect
