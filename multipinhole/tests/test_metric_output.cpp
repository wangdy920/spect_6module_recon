#include <gtest/gtest.h>

#include <cmath>
#include <regex>

#include "recon/ReconMetrics.h"

namespace {

TEST(MetricOutput, MatchesScriptRegexAndHasObjective) {
    const double beta = 1e-3;
    const double likelihood = -12.5;
    const double penalty = 3.25;

    const std::string line =
        spect::app::formatMetricLine(beta, likelihood, penalty, "runtime");

    const std::regex metric_re(
        R"(\[METRIC\] Beta=([\d\.e\-\+]+) Likelihood=([\d\.e\-\+]+) Penalty=([\d\.e\-\+]+))");

    EXPECT_TRUE(std::regex_search(line, metric_re));
    EXPECT_NE(line.find(" Objective="), std::string::npos);
    EXPECT_NE(line.find(" Model=runtime"), std::string::npos);
}

TEST(MetricOutput, RuntimeLikelihoodUsesBedOffsetsBackgroundAndAttenuation) {
    spect::CSR matrix;
    matrix.rows = 1;
    matrix.cols = 1;
    matrix.row_ptr = {0, 1};
    matrix.col_idx = {0};
    matrix.val = {2.0f};

    spect::ReconOpts opts;
    opts.threads = 1;
    opts.bedCount = 2;
    opts.num_angles = 1;
    opts.bedOffsetVoxels = {0, 1};
    opts.b_proj = nullptr;
    opts.bedAttenFactors = {
        std::vector<uint16_t>{32768},
        std::vector<uint16_t>{65535},
    };

    const std::vector<double> image = {3.0, 4.0};
    const std::vector<double> background = {1.0, 2.0};
    opts.b_proj = &background;

    const double scale0 = static_cast<double>(opts.bedAttenFactors[0][0]) / 65535.0;
    const double scale1 = static_cast<double>(opts.bedAttenFactors[1][0]) / 65535.0;
    const double yhat0 = 2.0 * 3.0 * scale0 + 1.0;
    const double yhat1 = 2.0 * 4.0 * scale1 + 2.0;
    const std::vector<double> projections = {yhat0, yhat1};

    const auto metric = spect::app::computePoissonLogLikelihoodRuntime(
        matrix, image, projections, opts, 1, 1, 2, nullptr, 1);

    const double expected = yhat0 * std::log(yhat0) - yhat0 +
                            yhat1 * std::log(yhat1) - yhat1;
    EXPECT_NEAR(metric.likelihood, expected, 1e-9);
    EXPECT_STREQ(metric.modelTag, "runtime");
}

} // namespace
