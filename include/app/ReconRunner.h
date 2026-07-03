// ReconRunner.h - Reconstruction runner interface
// 【T2.3】封装重建执行流程
#pragma once

#include "ReconContext.h"
#include "core/Types.h"
#include <functional>
#include <memory>
#include <string>

namespace spect {
namespace app {

/**
 * @brief Progress callback type.
 * 
 * Called during reconstruction with:
 * - iteration: Current iteration (1-based)
 * - totalIters: Total iterations
 * - subset: Current subset (0-based, -1 if not applicable)
 * - message: Optional status message
 */
using ProgressCallback = std::function<void(int iteration, int totalIters, 
                                             int subset, const std::string& message)>;

/**
 * @brief Reconstruction runner interface.
 * 
 * Encapsulates the entire reconstruction workflow:
 * 1. Load data (projections, system matrix)
 * 2. Initialize reconstruction
 * 3. Run iterations
 * 4. Save results
 */
class IReconRunner {
public:
    virtual ~IReconRunner() = default;
    
    /**
     * @brief Initialize the runner with context.
     * @param ctx Reconstruction context with all parameters.
     * @return Status code.
     */
    virtual Status initialize(ReconContext& ctx) = 0;
    
    /**
     * @brief Run the reconstruction.
     * @param progress Optional progress callback.
     * @return Status code.
     */
    virtual Status run(ProgressCallback progress = nullptr) = 0;
    
    /**
     * @brief Get the reconstructed image.
     * @return Pointer to image data (Nx * Ny * Nz doubles).
     */
    virtual const double* getImage() const = 0;
    
    /**
     * @brief Get image dimensions.
     */
    virtual void getImageDims(int& nx, int& ny, int& nz) const = 0;
    
    /**
     * @brief Save current image to file.
     * @param path Output path (without extension).
     * @param format Output format ("mhd", "raw", etc.).
     * @return Status code.
     */
    virtual Status saveImage(const std::string& path, 
                             const std::string& format = "mhd") = 0;
};

/**
 * @brief Factory function to create a reconstruction runner.
 * 
 * @param algorithm Algorithm type from ReconArgs.
 * @return Unique pointer to runner instance.
 */
std::unique_ptr<IReconRunner> createReconRunner(
    spect::cli::ReconArgs::Algorithm algorithm);

} // namespace app
} // namespace spect
