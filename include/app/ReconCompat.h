// ReconCompat.h - Compatibility layer for legacy code
// This header provides global-namespace access to modular reconstructors
#pragma once

#include "app/OSEMReconstructor.h"
#include "app/BSREMReconstructor.h"
#include "app/ReconConfig.h"
#include "app/ProjectionOps.h"
#include "filters/PSFConfig.h"

// Import spect namespace types into global namespace for legacy compatibility
using spect::ReconOpts;
using spect::PSFKernels;
using spect::PSFMode;
using spect::PSFConf;

// Import CSR types
using spect::io::CSR;
using spect::io::CSRView;
using spect::io::CSRMap;

// Import reconstructor functions into global namespace
using spect::OSEM_step;
using spect::BSREM_step;

// Import projection ops
using spect::computeNormAT;
using spect::computeNormAT_Atten;
using spect::computeSubsetSensitivity;

// Import filter functions
using spect::filters::conv3d_separable_inplace;
using spect::filters::conv2d_separable_inplace;

