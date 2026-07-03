/**
 * @file IGeometry.h
 * @brief 几何变换与坐标系统接口
 * 
 * 封装多探头、多针孔、多床位上下文。
 */

#pragma once

#include "../core/Types.h"
#include <vector>

namespace spect {

/**
 * @brief 床位信息
 */
struct BedPosition {
    int bed_id = 0;          ///< 床位ID
    Offset3D offset;         ///< 相对于参考点的偏移
    double overlap_fraction = 0.0; ///< 与相邻床位的重叠比例
    bool is_active = true;   ///< 是否参与重建
};

/**
 * @brief 探头信息
 */
struct DetectorHead {
    int head_id = 0;         ///< 探头ID
    double angle = 0.0;      ///< 角度位置 (弧度)
    double radius = 0.0;     ///< 旋转半径
    bool is_active = true;   ///< 是否参与重建
};

/**
 * @brief 几何接口
 */
class IGeometry {
public:
    virtual ~IGeometry() = default;
    
    /**
     * @brief 获取图像维度
     */
    virtual Dims3D imageDims() const = 0;
    
    /**
     * @brief 获取体素尺寸
     */
    virtual VoxelSize voxelSize() const = 0;
    
    /**
     * @brief 获取床位数
     */
    virtual int numBeds() const = 0;
    
    /**
     * @brief 获取床位信息
     */
    virtual const BedPosition& bedPosition(int idx) const = 0;
    
    /**
     * @brief 获取探头数
     */
    virtual int numHeads() const = 0;
    
    /**
     * @brief 获取探头信息
     */
    virtual const DetectorHead& detectorHead(int idx) const = 0;
    
    /**
     * @brief 世界坐标到体素坐标转换
     */
    virtual void worldToVoxel(double wx, double wy, double wz,
                             double& vx, double& vy, double& vz) const = 0;
    
    /**
     * @brief 体素坐标到世界坐标转换
     */
    virtual void voxelToWorld(double vx, double vy, double vz,
                             double& wx, double& wy, double& wz) const = 0;
    
    /**
     * @brief 获取多床位合并后的总图像维度
     */
    virtual Dims3D combinedImageDims() const = 0;
    
    /**
     * @brief 计算床位重叠区域的权重
     */
    virtual void computeOverlapWeights(int bed1, int bed2,
                                      std::vector<double>& weights) const = 0;
};

} // namespace spect
