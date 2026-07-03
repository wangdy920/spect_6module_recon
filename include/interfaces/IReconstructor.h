/**
 * @file IReconstructor.h
 * @brief 重建算法统一接口
 * 
 * 支持initialize/iterate/finalize模式，支持callback和checkpoint。
 */

#pragma once

#include "../core/Types.h"
#include <vector>
#include <functional>
#include <memory>

namespace spect {

// 前向声明
class IProjector;
class IRegularizer;
class IPSFOperator;

/**
 * @brief 迭代回调函数类型
 * 
 * @param iteration 当前迭代次数
 * @param image 当前图像估计
 * @param objective 目标函数值 (可选，-1表示未计算)
 * @return true 继续迭代，false 停止
 */
using IterationCallback = std::function<bool(int iteration, 
                                             const std::vector<double>& image,
                                             double objective)>;

/**
 * @brief 重建配置
 */
struct ReconConfig {
    AlgorithmType algorithm = AlgorithmType::MLEM;
    int num_iterations = 30;
    int num_subsets = 1;
    bool enforce_nonnegativity = true;
    double initial_value = 1.0;
    int save_interval = 0;  ///< 每N次迭代保存 (0=不保存)
    int threads = 1;
};

/**
 * @brief 重建器接口
 */
class IReconstructor {
public:
    virtual ~IReconstructor() = default;
    
    /**
     * @brief 获取算法名称
     */
    virtual std::string algorithmName() const = 0;
    
    /**
     * @brief 初始化重建器
     * 
     * @param projector 投影器
     * @param projection 投影数据
     * @param config 重建配置
     * @return Status
     */
    virtual Status initialize(const IProjector* projector,
                             const std::vector<double>& projection,
                             const ReconConfig& config) = 0;
    
    /**
     * @brief 设置正则化器 (可选)
     */
    virtual void setRegularizer(const IRegularizer* reg) = 0;
    
    /**
     * @brief 设置PSF算子 (可选)
     */
    virtual void setPSFOperator(const IPSFOperator* psf) = 0;
    
    /**
     * @brief 设置初始图像 (可选，否则使用均匀初始化)
     */
    virtual void setInitialImage(const std::vector<double>& image) = 0;
    
    /**
     * @brief 设置掩码 (可选)
     */
    virtual void setMask(const std::vector<uint8_t>& mask) = 0;
    
    /**
     * @brief 执行单次迭代
     * 
     * @return Status
     */
    virtual Status iterate() = 0;
    
    /**
     * @brief 执行多次迭代
     * 
     * @param num_iterations 迭代次数
     * @param callback 可选回调
     * @return Status
     */
    virtual Status run(int num_iterations = -1,
                      IterationCallback callback = nullptr) = 0;
    
    /**
     * @brief 获取当前迭代次数
     */
    virtual int currentIteration() const = 0;
    
    /**
     * @brief 获取当前图像估计
     */
    virtual const std::vector<double>& currentImage() const = 0;
    
    /**
     * @brief 计算当前目标函数值
     */
    virtual double computeObjective() const = 0;
    
    /**
     * @brief 完成重建 (清理资源)
     */
    virtual void finalize() = 0;
    
    /**
     * @brief 保存检查点
     */
    virtual Status saveCheckpoint(const std::string& path) const = 0;
    
    /**
     * @brief 加载检查点
     */
    virtual Status loadCheckpoint(const std::string& path) = 0;
};

/**
 * @brief 重建器工厂
 */
class ReconstructorFactory {
public:
    static std::unique_ptr<IReconstructor> create(AlgorithmType type);
    static std::unique_ptr<IReconstructor> create(const std::string& name);
};

} // namespace spect
