/**
 * @file Forward.h
 * @brief 前向声明汇总
 * 
 * 用于减少头文件依赖，加快编译速度。
 */

#pragma once

namespace spect {

// Core types
struct CSRHeader;
struct CSRView;
struct Dims3D;
struct VoxelSize;
struct Offset3D;
enum class Status;
enum class BackendType;
enum class Precision;
enum class AlgorithmType;

// Interfaces
class ISparseOps;
class IProjector;
class IRegularizer;
class IPSFOperator;
class IGeometry;
class IReconstructor;

// Geometry
struct BedPosition;
struct DetectorHead;

// Config
struct ReconConfig;

// Factories
class RegularizerFactory;
class ReconstructorFactory;

} // namespace spect
