/**
 * @file SPECT.h
 * @brief Master include for SPECT reconstruction library
 * 
 * This header provides access to all modular SPECT reconstruction functionality.
 * Use this instead of individual original_source_backup headers.
 */

#pragma once

// ================= Core Types =================
#include "core/Types.h"
#include "core/Masks.h"

// ================= I/O =================
#include "io/CSRBuilder.h"
#include "io/MatFileReader.h"
#include "io/MHDWriter.h"
#include "io/VolumeLoader.h"
#include "io/ProjectionLoader.h"

// ================= Filters =================
#include "filters/PSFConfig.h"
#include "filters/Convolution.h"
#include "filters/ButterworthFilter.h"

// ================= Regularizers =================
#include "regularizer/RegConfig.h"
#include "regularizer/AllRegularizers.h"
#include "regularizer/RegularizerCompat.h"

// ================= CLI =================
#include "cli/ReconArgs.h"
#include "cli/ArgumentParser.h"
#include "cli/ConfigParser.h"
#include "cli/StringUtils.h"

// ================= Reconstruction Algorithms =================
#include "app/ReconConfig.h"
#include "app/ReconContext.h"
#include "app/ReconContextAdapter.h"
#include "app/ProjectionOps.h"
#include "app/OSEMReconstructor.h"
#include "app/BSREMReconstructor.h"

// ================= Sparse Operations =================
#include "sparse/SparseOpsCPU.h"
#include "sparse/SparseOpsRef.h"
#include "sparse/SparseOpsBackend.h"

// ================= Interfaces =================
#include "interfaces/IRegularizer.h"
#include "interfaces/IProjector.h"
#include "interfaces/ISparseOps.h"

// ================= Namespace Aliases =================
namespace spect {

// Import commonly used types to top-level namespace
using io::CSR;
using CSRViewF = io::CSRView;
using io::CSRMap;
using io::MatArray;
using io::readMat;
using io::loadCSRBinary;
using io::saveCSRBinary;
using io::saveMHDRAW;
using io::loadVolume;

using filters::PSFConf;
using filters::PSFMode;
using filters::PSFKernels;
using filters::ButterworthConf;
using filters::BWType;
using filters::Spacing3;

using core::makeCylMask_mm;
using core::applyMaskInplace;

using cli::ReconArgs;

} // namespace spect


