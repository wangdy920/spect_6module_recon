/*
SPECT System Matrix PSC-YSO v7.6 CSR-ADAPTIVE RECT3-ROI CUDA (CSRB direct output + dense/hash source aggregation)

v730 vs v727 主要变化：
  1. 修复 DOI 间隙衰减累积错误：物理间隙不再贡献衰减路径长度。
  2. 修复包围盒预筛选的潜在精度风险：改用参数t范围过滤，数学上严格正确。
  3. DOI 使用全探测器范围均匀采样，ROI 在连续采样网格上裁剪。
  4. 使用 x_edges_flat / z_edges_flat 存储每个晶体真实的物理左右边界。
  5. 非 DOI 前表面采样只覆盖晶体本体，不再对间隙区域单独布点。
  6. 立体角权重不再乘 detector sample 面积因子。

编译：
  nvcc -O3 -std=c++14 -arch=sm_86 spect_sysmat_PSC_6module_24pin_YSO_rect3_DOIlin_direct.cu -o spect_psc_yso_rect3_doi
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <dirent.h>
#endif

#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <utility>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
   编译期常量
======================================================================== */
#ifndef S_VOX
#define S_VOX 3
#endif

#ifndef S_DET
#define S_DET 10
#endif

#define N_SUB_VOX  (S_VOX*S_VOX*S_VOX)
#define INV_N_SUB_VOX (1.0 / (double)N_SUB_VOX)

#define MAX_INTERVALS   64
#define MAX_DOI_SEG     32
#define BLOCK_SIZE_VA   256
/*
  v7.4-opt 支持三种输出模式：
    v2d:    推荐。先按 source=(angle,voxel) 聚合到 detector，再离线转置成旧 row 格式。
    direct: 旧 direct row buffer，DOI 下会放大 row 内查重/归并开销。
    coo:    旧 detector-row COO + sort/reduce，作为验证/回退路径。
*/
#define MAX_NNZ_PER_ROW 30000

#ifndef MAX_PINHOLES_TOTAL
#define MAX_PINHOLES_TOTAL 128
#endif

/*
  v7.3.1: collimator geometry is rectangular only.  The old cylindrical-shell
  branch is removed from the hot path; all planes and bounding spheres are also
  cached in constant memory to avoid repeated global-memory reads in every ray.
  For 6 modules x 4 pinholes/module x 3 channel sections plus 6 rectangular
  bodies, n_poly = 6 + 6*4*3 = 78, so the following limits are conservative.
*/
#ifndef MAX_COLL_POLY
#define MAX_COLL_POLY 256
#endif
#ifndef MAX_COLL_PLANES
#define MAX_COLL_PLANES (MAX_COLL_POLY*6)
#endif

#define COLL_GEOM_RECT 1

/*
  单个 thread 处理一个 voxel-angle-detector candidate。
  DOI 模式下，一个 candidate 的多个子射线可能命中多个 DOI 像素。
  先在线程局部聚合同一 candidate 内的 DOI 像素，再写全局 COO，减少原子写和后续 reduce 压力。
*/
#ifndef LOCAL_DOI_ACCUM
#define LOCAL_DOI_ACCUM 64
#endif

#define D_EPS 1e-15
#define F_EPS 1e-12f

#ifndef DOI_RADIAL_SAMPLES_DEFAULT
#define DOI_RADIAL_SAMPLES_DEFAULT 1024
#endif

#ifndef ENABLE_PRECHECK_STATS
#define ENABLE_PRECHECK_STATS 0
#endif

#ifndef COMPILE_ENABLE_DOI
#define COMPILE_ENABLE_DOI 1
#endif

/*
  准直器穿透衰减 exp(-mu*mp) 的计算模式:
    0: 使用 CUDA double exp，最保守
    1: double range reduction + 16阶泰勒，目标误差约 1e-13 量级
    2: __expf(float)，最快但不满足 1e-10 精度要求
*/
#ifndef COLL_EXP_MODE
#define COLL_EXP_MODE 0
#endif

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(_e));       \
        exit(1);                                                   \
    }                                                             \
} while(0)

static std::mutex g_print_mutex;

/* ========================================================================
   GPU统计
======================================================================== */
__device__ unsigned int  g_stat_nnz_overflow;
__device__ unsigned int  g_stat_hash_overflow;
__device__ unsigned long long g_stat_precheck_skipped;
__device__ unsigned long long g_stat_precheck_passed;

/* ========================================================================
   Constant Memory 结构体
======================================================================== */
#ifndef DET_NX_DEFAULT
#define DET_NX_DEFAULT 48
#endif
#ifndef DET_NZ_DEFAULT
#define DET_NZ_DEFAULT 48
#endif

#ifndef MAX_DET_PIX_X
#define MAX_DET_PIX_X DET_NX_DEFAULT
#endif
#ifndef MAX_DET_PIX_Z
#define MAX_DET_PIX_Z DET_NZ_DEFAULT
#endif

#if MAX_DET_PIX_X < DET_NX_DEFAULT
#error "MAX_DET_PIX_X must be >= DET_NX_DEFAULT"
#endif
#if MAX_DET_PIX_Z < DET_NZ_DEFAULT
#error "MAX_DET_PIX_Z must be >= DET_NZ_DEFAULT"
#endif

#define DET_PIX_X MAX_DET_PIX_X
#define DET_PIX_Z MAX_DET_PIX_Z

/*
  DetectorConst:
    x_edges_flat[2*i]   = 第i个晶体的物理左边界 (X轴)
    x_edges_flat[2*i+1] = 第i个晶体的物理右边界 (X轴)
    z_edges_flat 同理

  用于 DOI 追踪时判断射线当前线段是否处于真实晶体内部，
  从而避免将间隙空气区域计入衰减路径。
*/
struct DetectorConst {
    float x_edges_flat[DET_PIX_X * 2];   /* 每晶体左右边界，共 2*DET_PIX_X 个 float */
    float z_edges_flat[DET_PIX_Z * 2];   /* 每晶体上下边界，共 2*DET_PIX_Z 个 float */
    float x_centers[DET_PIX_X];          /* 每个晶体中心，用于 ROI 投影落点映射；投影可落在间隙中，取最近晶体中心 */
    float z_centers[DET_PIX_Z];
    float det_y;                          /* 探测器前表面 Y 坐标 */
    float back_y;                         /* 探测器后表面 Y 坐标 */
    float thickness;                      /* 探测器厚度 */
    float doi_x0;                         /* DOI 均匀采样：第一个 X 采样点坐标 */
    float doi_z0;                         /* DOI 均匀采样：第一个 Z 采样点坐标 */
    float doi_dx;                         /* DOI 均匀采样 X 间距 */
    float doi_dz;                         /* DOI 均匀采样 Z 间距 */
    int   nx, nz;                         /* 晶体阵列尺寸 */
    int   compact_det_z;                  /* Z 对称后有效像素行数 */
    int   doi_nx;                         /* DOI 均匀采样 X 点数 */
    int   doi_nz;                         /* DOI 均匀采样 Z 点数 */
};

struct PhysicsConst {
    double mu_collimator;
    double coll_exp_cutoff_x;
    double coll_exp_cutoff_len;
    double center_lor_mp_threshold;

    float mu_detector;
    float penetration_length_threshold;
    float penetration_probability_threshold;
    float weight_threshold;

    int solid_angle_power;
    int enable_penetration;
    int enable_DOI;
    int enable_center_lor_precheck;

    /*
      DOI 几何权重模式：
        0: front-plane 模式，保持旧结果：先在探测器前表面计算
           cos(theta) / r_front^2，DOI 只返回晶体内首次相互作用概率。
        1: DOI 线性积分模式：Siddon 每个晶体段内用 4 点 Gauss-Legendre
           积分 mu*exp(-mu*l) * (t_front/t)^2。调用端仍乘 front-plane
           立体角，因此这里返回的是相对 front-plane 立体角修正后的 alpha。
           该模式用于补偿 DOI 深度导致的 1/r^2 几何变化。
    */
    int enable_doi_solid_angle_integral;

    float pixel_size;
    float voxel_size;
};

struct CollConst {
    int n_poly;              /* total = rectangular bodies + pinhole void polytopes */
    int n_solid_poly;        /* rectangular collimator bodies, stored first */
    int n_void_poly;         /* 3-section pinhole channel polytopes, stored after solids */
    int geom_kind;           /* fixed to COLL_GEOM_RECT for metadata/debug; no cylinder branch */
    double R_inner, R_outer, H_half;    /* reserved, kept for ABI/metadata compatibility */
    double R_inner2, R_outer2;          /* reserved */
};

struct SysConfig {
    int total_voxels;
    int num_angles;
    int compact_num_angles;
    int full_det_x;
    int compact_det_z;
    int compact_rays_per_angle;
    int Nx, Ny, Nz;
    int roi_radius;       /* ROI 半窗口，窗口边长 = 2*roi_radius+1 */
    int roi_side;
    int roi_area;
    int n_pinholes;

    /* v7.4-opt: 准直器拓扑信息，用于 pinhole-aware fast penetration。 */
    int pinholes_per_module;
    int pinhole_sections;
    int fast_pinhole_penetration;  /* 1: 只算当前 pinhole 所属 module solid + 当前 pinhole 3段 void */

    int enable_doi_sample_roi;      /* 1: DOI 均匀采样只取投影点附近矩形 ROI；0: 全探测器面 */
    int doi_roi_radius_x;           /* DOI ROI 在均匀采样格上的 X 半径 */
    int doi_roi_radius_z;           /* DOI ROI 在均匀采样格上的 Z 半径 */
    int doi_roi_side_x;
    int doi_roi_side_z;
    int doi_task_area;              /* ROI on: side_x*side_z; ROI off: doi_nx*doi_nz */
};

__constant__ DetectorConst c_det;
__constant__ PhysicsConst  c_phys;
__constant__ CollConst     c_coll;
__constant__ SysConfig     c_cfg;

/*
  pm 面针孔中心，已经在 Host 端复制/旋转成所有 module 的全局坐标。
  ROI 只负责粗筛；最终权重仍由 analytical_penetration_d + DOI 精确计算。
*/
__constant__ float c_pin_x[MAX_PINHOLES_TOTAL];
__constant__ float c_pin_y[MAX_PINHOLES_TOTAL];
__constant__ float c_pin_z[MAX_PINHOLES_TOTAL];

/*
  几何常量缓存：准直器所有 polyhedron 的 6 个平面与包围球。
  kernel 仍保留旧参数列表以减少外层代码改动，但实际解析穿透计算使用
  analytical_penetration_const() 直接读取 constant memory。
*/
__constant__ double c_pnx_c[MAX_COLL_PLANES];
__constant__ double c_pny_c[MAX_COLL_PLANES];
__constant__ double c_pnz_c[MAX_COLL_PLANES];
__constant__ double c_pd_c [MAX_COLL_PLANES];
__constant__ double c_bcx_c[MAX_COLL_POLY];
__constant__ double c_bcy_c[MAX_COLL_POLY];
__constant__ double c_bcz_c[MAX_COLL_POLY];
__constant__ double c_br2_c[MAX_COLL_POLY];

__constant__ double c_vso_x[N_SUB_VOX];
__constant__ double c_vso_y[N_SUB_VOX];
__constant__ double c_vso_z[N_SUB_VOX];

/* ========================================================================
   Device: high-accuracy exp(-x) for collimator attenuation
======================================================================== */
__device__ __forceinline__ double exp_neg_collimator(double x)
{
#if COLL_EXP_MODE == 0
    return exp(-x);
#elif COLL_EXP_MODE == 2
    return (double)__expf((float)(-x));
#else
    if (x <= 0.0) return 1.0;
    if (x > 745.0) return 0.0;

    /*
      exp(-x) = 2^-n * exp(r), r = n*ln2 - x, n = round(x/ln2).
      Thus r is bounded by roughly [-ln2/2, ln2/2], where a 16th-order
      Taylor polynomial is comfortably below the requested 1e-10 error.
    */
    const double inv_ln2 = 1.44269504088896340736;
    const double ln2     = 0.69314718055994530942;

    int n = (int)(x * inv_ln2 + 0.5);
    double r = (double)n * ln2 - x;

    double p = 4.7794773323873852974e-14;  /* 1/16! */
    p = p*r + 7.6471637318198164759e-13;   /* 1/15! */
    p = p*r + 1.1470745597729724714e-11;   /* 1/14! */
    p = p*r + 1.6059043836821614599e-10;   /* 1/13! */
    p = p*r + 2.0876756987868098979e-09;   /* 1/12! */
    p = p*r + 2.5052108385441718775e-08;   /* 1/11! */
    p = p*r + 2.7557319223985890653e-07;   /* 1/10! */
    p = p*r + 2.7557319223985890653e-06;   /* 1/9! */
    p = p*r + 2.4801587301587301587e-05;   /* 1/8! */
    p = p*r + 1.9841269841269841270e-04;   /* 1/7! */
    p = p*r + 1.3888888888888888890e-03;   /* 1/6! */
    p = p*r + 8.3333333333333333332e-03;   /* 1/5! */
    p = p*r + 4.1666666666666666664e-02;   /* 1/4! */
    p = p*r + 1.6666666666666666666e-01;   /* 1/3! */
    p = p*r + 5.0000000000000000000e-01;   /* 1/2! */
    p = p*r + 1.0000000000000000000e+00;   /* 1/1! */
    p = p*r + 1.0000000000000000000e+00;   /* 1/0! */

    return ldexp(p, -n);
#endif
}

/* ========================================================================
   Device: analytical penetration (不变，保持 double 精度)
======================================================================== */
__device__ __forceinline__ bool ray_poly_interval_d(
    int p,
    double ox, double oy, double oz,
    double dx, double dy, double dz,
    const double* __restrict__ planes_nx,
    const double* __restrict__ planes_ny,
    const double* __restrict__ planes_nz,
    const double* __restrict__ planes_d,
    const double* __restrict__ bc_x,
    const double* __restrict__ bc_y,
    const double* __restrict__ bc_z,
    const double* __restrict__ br2,
    double* te_out,
    double* tx_out)
{
    double ocx = bc_x[p]-ox, ocy = bc_y[p]-oy, ocz = bc_z[p]-oz;
    double proj = ocx*dx + ocy*dy + ocz*dz;
    double dist2 = ocx*ocx + ocy*ocy + ocz*ocz - proj*proj;
    if (dist2 > br2[p]) return false;

    double te = -1e30, tx = 1e30;
    int base = p * 6;
    for (int f = 0; f < 6; f++) {
        int idx = base + f;
        double nx_ = planes_nx[idx], ny_ = planes_ny[idx];
        double nz_ = planes_nz[idx], dd  = planes_d[idx];
        double denom = nx_*dx + ny_*dy + nz_*dz;
        double numer = dd - (nx_*ox + ny_*oy + nz_*oz);

        if (fabs(denom) < D_EPS) {
            if (numer < -1e-10) return false;
        } else if (denom < 0.0) {
            double t = numer/denom; if (t > te) te = t;
        } else {
            double t = numer/denom; if (t < tx) tx = t;
        }
        if (te >= tx) return false;
    }

    if (tx <= 0.0) return false;
    if (te < 0.0) te = 0.0;
    if (te >= tx) return false;

    *te_out = te;
    *tx_out = tx;
    return true;
}

__device__ __forceinline__ void add_union_interval_d(
    double mi[MAX_INTERVALS][2],
    int* mi_n,
    double a,
    double b)
{
    if (b <= a || *mi_n >= MAX_INTERVALS) return;

    int n = *mi_n;
    mi[n][0] = a;
    mi[n][1] = b;
    n++;

    for (int i = 1; i < n; i++) {
        double ka = mi[i][0], kb = mi[i][1];
        int j = i - 1;
        while (j >= 0 && mi[j][0] > ka) {
            mi[j+1][0] = mi[j][0];
            mi[j+1][1] = mi[j][1];
            j--;
        }
        mi[j+1][0] = ka;
        mi[j+1][1] = kb;
    }

    int out = 0;
    for (int i = 0; i < n; i++) {
        if (out == 0 || mi[i][0] > mi[out-1][1]) {
            mi[out][0] = mi[i][0];
            mi[out][1] = mi[i][1];
            out++;
        } else if (mi[i][1] > mi[out-1][1]) {
            mi[out-1][1] = mi[i][1];
        }
    }
    *mi_n = out;
}

__device__ double analytical_penetration_d(
    double ox, double oy, double oz,
    double dx, double dy, double dz,
    const double* __restrict__ planes_nx,
    const double* __restrict__ planes_ny,
    const double* __restrict__ planes_nz,
    const double* __restrict__ planes_d,
    const double* __restrict__ bc_x,
    const double* __restrict__ bc_y,
    const double* __restrict__ bc_z,
    const double* __restrict__ br2)
{
    /*
      v7.3.1 rectangular-only collimator penetration model.

      准直器实体 = 每个模块一个矩形体；针孔 = v2_PSC 坐标定义的三段通道
      (pb->pmb, pmb->pmt, pmt->pt)。穿透长度计算为：

          material_length = union(rectangular solid bodies) - union(pinhole voids)

      旧圆柱壳分支已删除，避免每条射线先做圆柱求交再进入多面体扣减的额外开销。
    */
    double mi[MAX_INTERVALS][2];
    int mi_n = 0;

    /* 1) 与所有矩形实体求交，并做区间 union。n_solid_poly = 模块数，通常为 6。 */
    for (int p = 0; p < c_coll.n_solid_poly; p++) {
        double te, tx;
        if (ray_poly_interval_d(
                p, ox, oy, oz, dx, dy, dz,
                planes_nx, planes_ny, planes_nz, planes_d,
                bc_x, bc_y, bc_z, br2, &te, &tx)) {
            add_union_interval_d(mi, &mi_n, te, tx);
        }
    }
    if (mi_n == 0) return 0.0;

    /* 2) 逐个扣除 pinhole void。void 多面体紧跟在 solid bodies 之后。 */
    int void_start = c_coll.n_solid_poly;
    int void_end   = c_coll.n_solid_poly + c_coll.n_void_poly;
    for (int p = void_start; p < void_end; p++) {
        double te, tx;
        if (!ray_poly_interval_d(
                p, ox, oy, oz, dx, dy, dz,
                planes_nx, planes_ny, planes_nz, planes_d,
                bc_x, bc_y, bc_z, br2, &te, &tx)) {
            continue;
        }

        double new_mi[MAX_INTERVALS][2];
        int new_n = 0;
        for (int k = 0; k < mi_n; k++) {
            double a1=mi[k][0], a2=mi[k][1];
            double v1=fmax(te,a1), v2=fmin(tx,a2);
            if (v1 >= v2) {
                if (new_n < MAX_INTERVALS) { new_mi[new_n][0]=a1; new_mi[new_n][1]=a2; new_n++; }
            } else {
                if (a1 < v1 && new_n < MAX_INTERVALS) { new_mi[new_n][0]=a1; new_mi[new_n][1]=v1; new_n++; }
                if (v2 < a2 && new_n < MAX_INTERVALS) { new_mi[new_n][0]=v2; new_mi[new_n][1]=a2; new_n++; }
            }
        }
        mi_n = new_n;
        for (int k=0; k<mi_n; k++) { mi[k][0]=new_mi[k][0]; mi[k][1]=new_mi[k][1]; }
        if (mi_n == 0) return 0.0;
    }

    double mat = 0.0;
    for (int k=0; k<mi_n; k++) mat += mi[k][1] - mi[k][0];
    return mat;
}

__device__ __forceinline__ double analytical_penetration_const(
    double ox, double oy, double oz,
    double dx, double dy, double dz)
{
    return analytical_penetration_d(
        ox, oy, oz, dx, dy, dz,
        c_pnx_c, c_pny_c, c_pnz_c, c_pd_c,
        c_bcx_c, c_bcy_c, c_bcz_c, c_br2_c);
}

/* ========================================================================
   v7.4-opt: pinhole-aware penetration

   精确模式 analytical_penetration_const() 会对所有 module solid 和所有
   pinhole void 做 union-minus-void。对于当前 ROI candidate，射线来自某个
   pinhole 的投影窗口，绝大多数有效射线只与该 pinhole 所属 module 和该
   pinhole 的 3 段通道相关。

   fast_pinhole_penetration=1 时：
       material = 当前 pinhole 所属 module solid - 当前 pinhole 的 sections
   这将每条子射线的 polyhedron 测试从 6 + 72 降为 1 + 3。

   建议流程：
     1) 用 --fast-pin-penetration 0 生成小角度/小ROI基准；
     2) 用 --fast-pin-penetration 1 生成同样配置；
     3) 比较 row sum / forward projection 误差；
     4) 若误差可接受，再用于全量生成。
======================================================================== */
__device__ double analytical_penetration_pinhole_const(
    int pin_id,
    double ox, double oy, double oz,
    double dx, double dy, double dz)
{
    int pins_per_module = c_cfg.pinholes_per_module;
    int sections        = c_cfg.pinhole_sections;
    if (pins_per_module <= 0 || sections <= 0) {
        return analytical_penetration_const(ox, oy, oz, dx, dy, dz);
    }

    int module_id = pin_id / pins_per_module;
    if (module_id < 0) module_id = 0;
    if (module_id >= c_coll.n_solid_poly && c_coll.n_solid_poly > 0) {
        module_id = module_id % c_coll.n_solid_poly;
    }

    double mi[MAX_INTERVALS][2];
    int mi_n = 0;

    double te, tx;
    if (!ray_poly_interval_d(
            module_id, ox, oy, oz, dx, dy, dz,
            c_pnx_c, c_pny_c, c_pnz_c, c_pd_c,
            c_bcx_c, c_bcy_c, c_bcz_c, c_br2_c, &te, &tx)) {
        return 0.0;
    }
    mi[0][0] = te;
    mi[0][1] = tx;
    mi_n = 1;

    int void_start = c_coll.n_solid_poly + pin_id * sections;
    int void_end   = void_start + sections;
    int max_void_end = c_coll.n_solid_poly + c_coll.n_void_poly;
    if (void_end > max_void_end) void_end = max_void_end;

    for (int p = void_start; p < void_end; p++) {
        double ve, vx;
        if (!ray_poly_interval_d(
                p, ox, oy, oz, dx, dy, dz,
                c_pnx_c, c_pny_c, c_pnz_c, c_pd_c,
                c_bcx_c, c_bcy_c, c_bcz_c, c_br2_c, &ve, &vx)) {
            continue;
        }

        double new_mi[MAX_INTERVALS][2];
        int new_n = 0;
        for (int k = 0; k < mi_n; k++) {
            double a1 = mi[k][0], a2 = mi[k][1];
            double v1 = fmax(ve, a1), v2 = fmin(vx, a2);
            if (v1 >= v2) {
                if (new_n < MAX_INTERVALS) {
                    new_mi[new_n][0] = a1;
                    new_mi[new_n][1] = a2;
                    new_n++;
                }
            } else {
                if (a1 < v1 && new_n < MAX_INTERVALS) {
                    new_mi[new_n][0] = a1;
                    new_mi[new_n][1] = v1;
                    new_n++;
                }
                if (v2 < a2 && new_n < MAX_INTERVALS) {
                    new_mi[new_n][0] = v2;
                    new_mi[new_n][1] = a2;
                    new_n++;
                }
            }
        }
        mi_n = new_n;
        for (int k = 0; k < mi_n; k++) {
            mi[k][0] = new_mi[k][0];
            mi[k][1] = new_mi[k][1];
        }
        if (mi_n == 0) return 0.0;
    }

    double mat = 0.0;
    for (int k = 0; k < mi_n; k++) mat += mi[k][1] - mi[k][0];
    return mat;
}

__device__ __forceinline__ double analytical_penetration_selected_const(
    int pin_id,
    double ox, double oy, double oz,
    double dx, double dy, double dz)
{
    if (c_cfg.fast_pinhole_penetration) {
        return analytical_penetration_pinhole_const(pin_id, ox, oy, oz, dx, dy, dz);
    }
    return analytical_penetration_const(ox, oy, oz, dx, dy, dz);
}

/* ========================================================================
   Device: DOI 精确追踪 (修复版)

   核心修复：
     旧版错误：将射线穿越探测器的完整路径长度（含间隙）作为衰减厚度计算。
     新版正确：
       1. 射线从 vox 出发，与探测器区域 [det_y, back_y] 求参数 t 的范围。
       2. 用 x_edges_flat / z_edges_flat 求所有 X 和 Z 方向的晶体边界对应的 t 值。
          — 注意：这里只插入"晶体边界"（即晶体的进出面），而不是网格的外框边界。
          — 间隙区域在两个晶体边界之间自然形成，不需要额外插入间隙端点。
       3. 对每个子线段，用中点判断是否在某个晶体内部：
          a. 若中点在晶体内：该段长度累加到 L_accum（衰减路径），
             计算该段的探测概率贡献，归属至对应像素。
          b. 若中点在间隙中：该段为空气，L_accum 不增加，不贡献任何探测概率。

   包围盒筛选修正：
     旧版：用空间坐标 min_x/max_x 过滤 x_edges_flat，存在精度风险。
     新版：用参数 t 的范围 [t_front, t_back] 过滤——即判断各晶体边界对应的
           参数 t 是否在射线的有效范围内，这是数学上严格正确的做法。

   L_accum 语义：
     当前射线已经穿过的闪烁体材料总厚度（mm 等效在射线方向上的长度）。
     只有真正在晶体内的线段才累加到 L_accum。
     间隙（空气）区域不参与 L_accum，从而保证后续晶体层的穿透概率计算正确。
======================================================================== */
__device__ __forceinline__ int lower_bound_float_device(const float* arr, int n, float x)
{
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int upper_bound_float_device(const float* arr, int n, float x)
{
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] <= x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int nearest_center_index_device(const float* centers, int n, float x)
{
    /*
      将连续坐标映射到最近晶体中心。注意：x 可能落在晶体间隙，
      ROI 中心仍然要落到最近的实际探测器像素作为窗口中心。
    */
    int pos = lower_bound_float_device(centers, n, x);
    if (pos <= 0) return 0;
    if (pos >= n) return n - 1;
    float dl = fabsf(x - centers[pos - 1]);
    float dr = fabsf(centers[pos] - x);
    return (dr < dl) ? pos : (pos - 1);
}

__device__ __forceinline__ int find_crystal_index_device(
    const float* centers, const float* edges_flat, int n, float x)
{
    /*
      先用中心点二分找到最接近的晶体，再只检查相邻少数几个晶体边界。
      这样避免 DOI 中对 96 个晶体逐个线性扫描。
    */
    int c = nearest_center_index_device(centers, n, x);
    #pragma unroll
    for (int off = -1; off <= 1; off++) {
        int i = c + off;
        if (i >= 0 && i < n) {
            if (x >= edges_flat[2*i] && x <= edges_flat[2*i + 1]) return i;
        }
    }
    return -1;  /* 落在物理间隙中 */
}


__device__ __forceinline__ float detector_segment_alpha(
    float t_front,
    float t1,
    float dt,
    float L_accum)
{
    /*
      返回当前 GAGG/YSO 晶体线段对 DOI pixel 的探测 alpha。

      front-plane 兼容模式：
          alpha = exp(-mu*L) - exp(-mu*(L+dt))

      DOI 线性积分模式：
          alpha = ∫_0^dt mu * exp[-mu*(L+u)] * (t_front/(t1+u))^2 du

      调用端已经乘了 detector front plane 的几何权重 cos(theta)/t_front^2，
      因此这里的 (t_front/t)^2 是把几何权重移动到 DOI 深度处的修正因子。
      4 点 Gauss-Legendre 在 dt<=晶体/边界分段长度时精度足够，且不改变
      Siddon 分段边界和 DOI pixel 归属。
    */
    float mu = c_phys.mu_detector;
    if (mu <= 0.0f || dt <= 0.0f) return 0.0f;

    if (!c_phys.enable_doi_solid_angle_integral) {
        return expf(-mu * L_accum) - expf(-mu * (L_accum + dt));
    }

    const float x1 = -0.8611363115940526f;
    const float x2 = -0.3399810435848563f;
    const float x3 =  0.3399810435848563f;
    const float x4 =  0.8611363115940526f;
    const float w1 =  0.3478548451374538f;
    const float w2 =  0.6521451548625461f;
    const float w3 =  0.6521451548625461f;
    const float w4 =  0.3478548451374538f;

    float half = 0.5f * dt;
    float mid  = 0.5f * dt;
    float tf2  = t_front * t_front;
    float sum  = 0.0f;

    float u = mid + half * x1;
    float t = t1 + u;
    if (t > 1e-8f) sum += w1 * expf(-mu * (L_accum + u)) * (tf2 / (t*t));

    u = mid + half * x2;
    t = t1 + u;
    if (t > 1e-8f) sum += w2 * expf(-mu * (L_accum + u)) * (tf2 / (t*t));

    u = mid + half * x3;
    t = t1 + u;
    if (t > 1e-8f) sum += w3 * expf(-mu * (L_accum + u)) * (tf2 / (t*t));

    u = mid + half * x4;
    t = t1 + u;
    if (t > 1e-8f) sum += w4 * expf(-mu * (L_accum + u)) * (tf2 / (t*t));

    return mu * half * sum;
}

/* ========================================================================
   Device: DOI 精确追踪 v7.3.0

   优化点：
     1. 先计算射线穿过探测器前/后表面的 X/Z 坐标范围。
     2. 只插入落在 [x_min, x_max] / [z_min, z_max] 内的晶体边界，
        不再无条件遍历所有 96 个晶体边界。
     3. 线段中点找晶体时使用“中心点二分 + 相邻边界确认”，
        避免每段再次线性扫描 96 个晶体。

   物理逻辑保持不变：只有晶体内路径累加 L_accum；间隙不贡献探测概率。
======================================================================== */
__device__ int compute_doi_spread(
    float vox_x, float vox_y, float vox_z,
    float ldr_x, float ldr_y, float ldr_z,
    int* pix_out,
    float* alpha_out)
{
    if (ldr_y < 1e-8f) return 0;

    float t_front = (c_det.det_y  - vox_y) / ldr_y;
    float t_back  = (c_det.back_y - vox_y) / ldr_y;
    if (t_front <= 0.f || t_back <= t_front) return 0;

    float x_front = vox_x + t_front * ldr_x;
    float x_back  = vox_x + t_back  * ldr_x;
    float z_front = vox_z + t_front * ldr_z;
    float z_back  = vox_z + t_back  * ldr_z;

    float x_min = fminf(x_front, x_back) - 1e-5f;
    float x_max = fmaxf(x_front, x_back) + 1e-5f;
    float z_min = fminf(z_front, z_back) - 1e-5f;
    float z_max = fmaxf(z_front, z_back) + 1e-5f;

    const int MAX_T = 2*DET_PIX_X + 2*DET_PIX_Z + 4;
    float t_list[MAX_T];
    int   t_cnt = 0;

    t_list[t_cnt++] = t_front;
    t_list[t_cnt++] = t_back;

    if (fabsf(ldr_x) > 1e-10f) {
        float inv_lx = 1.f / ldr_x;
        int e0 = lower_bound_float_device(c_det.x_edges_flat, 2*c_det.nx, x_min);
        int e1 = upper_bound_float_device(c_det.x_edges_flat, 2*c_det.nx, x_max);
        for (int e = e0; e < e1; e++) {
            float t = (c_det.x_edges_flat[e] - vox_x) * inv_lx;
            if (t > t_front && t < t_back && t_cnt < MAX_T) t_list[t_cnt++] = t;
        }
    }

    if (fabsf(ldr_z) > 1e-10f) {
        float inv_lz = 1.f / ldr_z;
        int e0 = lower_bound_float_device(c_det.z_edges_flat, 2*c_det.nz, z_min);
        int e1 = upper_bound_float_device(c_det.z_edges_flat, 2*c_det.nz, z_max);
        for (int e = e0; e < e1; e++) {
            float t = (c_det.z_edges_flat[e] - vox_z) * inv_lz;
            if (t > t_front && t < t_back && t_cnt < MAX_T) t_list[t_cnt++] = t;
        }
    }

    /* t_cnt 通常很小，插入排序比通用排序开销低。 */
    for (int i = 1; i < t_cnt; i++) {
        float key = t_list[i];
        int   j   = i - 1;
        while (j >= 0 && t_list[j] > key) {
            t_list[j+1] = t_list[j];
            j--;
        }
        t_list[j+1] = key;
    }

    int   n_pix   = 0;
    float L_accum = 0.f;

    for (int s = 0; s < t_cnt - 1; s++) {
        float t1 = t_list[s];
        float t2 = t_list[s+1];
        float dt = t2 - t1;
        if (dt < 1e-6f) continue;

        float tm = 0.5f * (t1 + t2);
        float xm = vox_x + tm * ldr_x;
        float zm = vox_z + tm * ldr_z;

        int ix = find_crystal_index_device(c_det.x_centers, c_det.x_edges_flat, c_det.nx, xm);
        if (ix < 0) continue;

        int iz = find_crystal_index_device(c_det.z_centers, c_det.z_edges_flat, c_det.nz, zm);
        if (iz < 0) continue;

        float prob = detector_segment_alpha(t_front, t1, dt, L_accum);

        if (prob > 1e-8f) {
            int lin = iz * c_det.nx + ix;
            bool found = false;
            for (int k = 0; k < n_pix; k++) {
                if (pix_out[k] == lin) {
                    alpha_out[k] += prob;
                    found = true;
                    break;
                }
            }
            if (!found && n_pix < MAX_DOI_SEG) {
                pix_out[n_pix]   = lin;
                alpha_out[n_pix] = prob;
                n_pix++;
            }
        }

        L_accum += dt;
    }

    return n_pix;
}

/* ========================================================================
   Device: center LOR precheck (不变)
======================================================================== */
__device__ __forceinline__ bool center_lor_passcheck(
    double vcx, double vcy, double vcz,
    double dkx, double dky, double dkz,
    const double* __restrict__ pnx,
    const double* __restrict__ pny,
    const double* __restrict__ pnz,
    const double* __restrict__ pd,
    const double* __restrict__ bcx,
    const double* __restrict__ bcy,
    const double* __restrict__ bcz,
    const double* __restrict__ br2)
{
    if (!c_phys.enable_center_lor_precheck) return true;

    double lx = dkx-vcx, ly = dky-vcy, lz = dkz-vcz;
    double r2 = lx*lx + ly*ly + lz*lz;
    if (r2 < 1e-12) return false;
    double inv_r = 1.0 / sqrt(r2);

    double ldx = lx*inv_r, ldy = ly*inv_r, ldz = lz*inv_r;
    if (ldy < 1e-8) return false;

    (void)pnx; (void)pny; (void)pnz; (void)pd;
    (void)bcx; (void)bcy; (void)bcz; (void)br2;
    double mp = analytical_penetration_const(vcx, vcy, vcz, ldx, ldy, ldz);

    return (mp <= c_phys.center_lor_mp_threshold);
}


__device__ __forceinline__ bool center_lor_passcheck_pin(
    int pin_id,
    double vcx, double vcy, double vcz,
    double dkx, double dky, double dkz,
    const double* __restrict__ pnx,
    const double* __restrict__ pny,
    const double* __restrict__ pnz,
    const double* __restrict__ pd,
    const double* __restrict__ bcx,
    const double* __restrict__ bcy,
    const double* __restrict__ bcz,
    const double* __restrict__ br2)
{
    if (!c_phys.enable_center_lor_precheck) return true;

    double lx = dkx-vcx, ly = dky-vcy, lz = dkz-vcz;
    double r2 = lx*lx + ly*ly + lz*lz;
    if (r2 < 1e-12) return false;
    double inv_r = 1.0 / sqrt(r2);

    double ldx = lx*inv_r, ldy = ly*inv_r, ldz = lz*inv_r;
    if (ldy < 1e-8) return false;

    (void)pnx; (void)pny; (void)pnz; (void)pd;
    (void)bcx; (void)bcy; (void)bcz; (void)br2;
    double mp = analytical_penetration_selected_const(pin_id, vcx, vcy, vcz, ldx, ldy, ldz);

    return (mp <= c_phys.center_lor_mp_threshold);
}

/* ========================================================================
   Device: ROI 投影与 COO 输出辅助函数
======================================================================== */
__device__ __forceinline__ bool project_voxel_to_detector_by_pin(
    double vcx, double vcy, double vcz,
    int pin_id,
    float* proj_x,
    float* proj_z,
    int* center_ix,
    int* center_iz)
{
    /*
      体素中心 V 与 pm 针孔中心 P 连线，继续延长到探测器前表面 y=det_y。
      投影点可能落在晶体间隙中，因此后续只把它映射为“最近晶体中心”作为 ROI 窗口中心。
    */
    double px = (double)c_pin_x[pin_id];
    double py = (double)c_pin_y[pin_id];
    double pz = (double)c_pin_z[pin_id];

    double denom = py - vcy;
    if (denom <= 1e-10) return false;

    double t = ((double)c_det.det_y - vcy) / denom;
    if (t <= 1.0) return false;  /* 探测器应在针孔之后 */

    float x = (float)(vcx + t * (px - vcx));
    float z = (float)(vcz + t * (pz - vcz));

    *proj_x = x;
    *proj_z = z;
    *center_ix = nearest_center_index_device(c_det.x_centers, c_det.nx, x);
    *center_iz = nearest_center_index_device(c_det.z_centers, c_det.nz, z);
    return true;
}

__device__ __forceinline__ int roi_owner_pinhole(
    double vcx, double vcy, double vcz,
    int ix_pix, int iz_pix)
{
    /*
      同一个 detector pixel 可能同时落入多个 pinhole 的 ROI 窗口。
      为避免重复计算/重复计数，这里把该 pixel 归属给“所有包含它的 ROI 中编号最小的 pinhole”。
      这样得到的是 ROI 并集，而不是多个 ROI 的重复叠加。
    */
    int owner = -1;
    for (int p = 0; p < c_cfg.n_pinholes; p++) {
        float px, pz;
        int cx, cz;
        if (!project_voxel_to_detector_by_pin(vcx, vcy, vcz, p, &px, &pz, &cx, &cz)) continue;
        if (abs(ix_pix - cx) <= c_cfg.roi_radius &&
            abs(iz_pix - cz) <= c_cfg.roi_radius) {
            owner = p;
            break;
        }
    }
    return owner;
}

__device__ __forceinline__ void append_coo_entry(
    unsigned long long row,
    unsigned long long col,
    float val,
    unsigned long long total_voxels,
    unsigned long long* __restrict__ coo_keys,
    float* __restrict__ coo_vals,
    unsigned long long* __restrict__ coo_nnz,
    unsigned long long coo_cap)
{
    if (val < c_phys.weight_threshold) return;
    unsigned long long pos = atomicAdd(coo_nnz, 1ULL);
    if (pos < coo_cap) {
        coo_keys[pos] = row * total_voxels + col;
        coo_vals[pos] = val;
    } else {
        atomicAdd(&g_stat_nnz_overflow, 1u);
    }
}



__device__ __forceinline__ void append_v2d_entry(
    unsigned long long source_id,
    unsigned int det_id,
    float val,
    unsigned long long* __restrict__ v2d_keys,
    float* __restrict__ v2d_vals,
    unsigned long long* __restrict__ v2d_nnz,
    unsigned long long v2d_cap)
{
    /*
      source-oriented COO:
        source_id = local_angle * total_voxels + voxel_id
        key       = source_id * compact_rays_per_angle + det_id

      该 key 先在 GPU 上按 (source,det) reduce，消除 DOI/子射线导致的重复；
      保存 chunk 时再转置成旧格式 row=(angle,det), col=voxel。
    */
    if (val < c_phys.weight_threshold) return;
    if ((int)det_id < 0 || det_id >= (unsigned int)c_cfg.compact_rays_per_angle) return;

    unsigned long long pos = atomicAdd(v2d_nnz, 1ULL);
    if (pos < v2d_cap) {
        v2d_keys[pos] = source_id * (unsigned long long)c_cfg.compact_rays_per_angle +
                        (unsigned long long)det_id;
        v2d_vals[pos] = val;
    } else {
        atomicAdd(&g_stat_nnz_overflow, 1u);
    }
}

__device__ __forceinline__ void append_direct_entry(
    int row,
    int voxel_index0,
    float val,
    int* __restrict__ vox_out,
    float* __restrict__ wgt_out,
    int* __restrict__ nnz_out)
{
    /*
      direct 输出模式：每个 chunk 在 GPU 上保留一个 row-major 定长矩阵缓冲区：

          vox_out[row * MAX_NNZ_PER_ROW + slot]
          wgt_out[row * MAX_NNZ_PER_ROW + slot]
          nnz_out[row]

      与 COO 模式相比，这里不再将所有贡献写成 COO，也不做 thrust sort/reduce。
      显存占用约为：rows_per_chunk * MAX_NNZ_PER_ROW * (sizeof(int)+sizeof(float))。
      优点是省去 COO 全局排序和 reduce，通常更适合 48GB 级别 GPU。

      为减少同一 voxel 的重复项，写入前沿用旧版 row 内线性查找；该查找只影响速度，
      不改变系统矩阵物理模型。线程内已有 LOCAL_DOI_ACCUM 聚合，因此实际 atomic 次数
      已明显少于逐子射线直接写 row buffer。
    */
    if (val < c_phys.weight_threshold) return;

    int*   rv = vox_out + (long long)row * MAX_NNZ_PER_ROW;
    float* rw = wgt_out + (long long)row * MAX_NNZ_PER_ROW;

    int voxel_index1 = voxel_index0 + 1;  /* 文件保存时会减 1，保持与旧版 chunk 格式一致。 */
    int cur = nnz_out[row];

    for (int s = 0; s < cur && s < MAX_NNZ_PER_ROW; s++) {
        if (rv[s] == voxel_index1) {
            atomicAdd(&rw[s], val);
            return;
        }
    }

    int slot = atomicAdd(&nnz_out[row], 1);
    if (slot < MAX_NNZ_PER_ROW) {
        rv[slot] = voxel_index1;
        rw[slot] = val;
    } else {
        atomicAdd(&nnz_out[row], -1);
        atomicAdd(&g_stat_nnz_overflow, 1u);
    }
}

__device__ __forceinline__ int nearest_uniform_sample_index(
    float first,
    float step,
    int n,
    float x)
{
    if (n <= 1 || step <= 0.0f) return 0;
    int idx = (int)floorf((x - first) / step + 0.5f);
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    return idx;
}

__device__ __forceinline__ bool doi_uniform_sample_indices(
    int rid,
    float proj_x,
    float proj_z,
    int* sx_idx,
    int* sz_idx)
{
    if (c_det.doi_nx <= 0 || c_det.doi_nz <= 0) return false;

    if (!c_cfg.enable_doi_sample_roi) {
        int area = c_det.doi_nx * c_det.doi_nz;
        if (rid < 0 || rid >= area) return false;
        *sx_idx = rid % c_det.doi_nx;
        *sz_idx = rid / c_det.doi_nx;
        return true;
    }

    int side_x = c_cfg.doi_roi_side_x;
    int side_z = c_cfg.doi_roi_side_z;
    if (side_x <= 0 || side_z <= 0 || rid < 0 || rid >= side_x * side_z) return false;

    int center_x = nearest_uniform_sample_index(c_det.doi_x0, c_det.doi_dx, c_det.doi_nx, proj_x);
    int center_z = nearest_uniform_sample_index(c_det.doi_z0, c_det.doi_dz, c_det.doi_nz, proj_z);
    int dx = (rid % side_x) - c_cfg.doi_roi_radius_x;
    int dz = (rid / side_x) - c_cfg.doi_roi_radius_z;

    int ix = center_x + dx;
    int iz = center_z + dz;
    if (ix < 0 || ix >= c_det.doi_nx) return false;
    if (iz < 0 || iz >= c_det.doi_nz) return false;

    *sx_idx = ix;
    *sz_idx = iz;
    return true;
}

__device__ __forceinline__ int compute_doi_sample_contrib(
    int pin_id,
    bool use_pin_context,
    double vx,
    double vy,
    double vz,
    float sx,
    float sz,
    int* pix_out,
    float* w_out)
{
    double lx = (double)sx - vx;
    double ly = (double)c_det.det_y - vy;
    double lz = (double)sz - vz;
    if (ly < 1e-8) return 0;

    double r2 = lx*lx + ly*ly + lz*lz;
    if (r2 < 1e-12) return 0;

    double inv_r = 1.0 / sqrt(r2);
    double ldx = lx*inv_r, ldy = ly*inv_r, ldz = lz*inv_r;
    if (ldy < 1e-8) return 0;

    double mp = use_pin_context
        ? analytical_penetration_selected_const(pin_id, vx, vy, vz, ldx, ldy, ldz)
        : analytical_penetration_const(vx, vy, vz, ldx, ldy, ldz);
    int    pinhole = (mp < (double)c_phys.penetration_length_threshold);
    double P_coll  = 1.0;

    if (!pinhole) {
        if (!c_phys.enable_penetration) return 0;

        double x_coll = c_phys.mu_collimator * mp;
        if (x_coll >= c_phys.coll_exp_cutoff_x ||
            mp     >= c_phys.coll_exp_cutoff_len) {
            return 0;
        }

        P_coll = exp_neg_collimator(x_coll);
        if (P_coll <= (double)c_phys.penetration_probability_threshold) return 0;
    }

    double cp;
    switch (c_phys.solid_angle_power) {
        case 1:  cp = ldy;          break;
        case 2:  cp = ldy*ldy;      break;
        case 3:  cp = ldy*ldy*ldy;  break;
        default: cp = pow(ldy, (double)c_phys.solid_angle_power);
    }

    double w_sub = cp * inv_r * inv_r * P_coll * INV_N_SUB_VOX;

    int   doi_pix[MAX_DOI_SEG];
    float doi_a  [MAX_DOI_SEG];
    int ns = compute_doi_spread(
        (float)vx, (float)vy, (float)vz,
        (float)ldx, (float)ldy, (float)ldz,
        doi_pix, doi_a);

    int nout = 0;
    for (int dd = 0; dd < ns; dd++) {
        int jl = doi_pix[dd];
        if (jl / c_det.nx >= c_det.compact_det_z) continue;
        float w_f = (float)(w_sub * (double)doi_a[dd]);
        if (w_f < c_phys.weight_threshold) continue;
        pix_out[nout] = jl;
        w_out[nout] = w_f;
        nout++;
    }
    return nout;
}

__global__ void kernel_subray_uniform_doi_direct(
    const float*  __restrict__ vox_x,
    const float*  __restrict__ vox_y,
    const float*  __restrict__ vox_z,
    const float*  __restrict__ tx_val,
    const float*  __restrict__ tz_val,
    const double* __restrict__ angle_cos,
    const double* __restrict__ angle_sin,
    const double* __restrict__ pnx,
    const double* __restrict__ pny,
    const double* __restrict__ pnz,
    const double* __restrict__ pd,
    const double* __restrict__ bcx,
    const double* __restrict__ bcy,
    const double* __restrict__ bcz,
    const double* __restrict__ br2,
    int  a_off,
    int  a_cnt,
    int*   __restrict__ vox_out,
    float* __restrict__ wgt_out,
    int*   __restrict__ nnz_out)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long nva  = (long long)c_cfg.total_voxels * a_cnt;
    bool use_pin_context = (c_cfg.enable_doi_sample_roi != 0);
    int pin_factor = use_pin_context ? c_cfg.n_pinholes : 1;
    long long ntot = nva * (long long)pin_factor * (long long)c_cfg.doi_task_area;
    if (gid >= ntot) return;

    int rid = (int)(gid % c_cfg.doi_task_area);
    long long tmp = gid / c_cfg.doi_task_area;
    int pin_id = 0;
    if (use_pin_context) {
        pin_id = (int)(tmp % c_cfg.n_pinholes);
        tmp /= c_cfg.n_pinholes;
    }
    int la = (int)(tmp % a_cnt);
    int i  = (int)(tmp / a_cnt);
    if (i >= c_cfg.total_voxels) return;

    int angle_idx = a_off + la;
    double ct = angle_cos[angle_idx];
    double st = angle_sin[angle_idx];

    double wx = (double)vox_x[i];
    double wy = (double)vox_y[i];
    double wz = (double)vox_z[i];
    double vcx =  wx*ct + wy*st;
    double vcy = -wx*st + wy*ct;
    double vcz =  wz;

    float proj_x = 0.0f, proj_z = 0.0f;
    if (use_pin_context) {
        int cx_idx, cz_idx;
        if (!project_voxel_to_detector_by_pin(vcx, vcy, vcz, pin_id,
                                               &proj_x, &proj_z, &cx_idx, &cz_idx)) return;
    }

    int sx_idx, sz_idx;
    if (!doi_uniform_sample_indices(rid, proj_x, proj_z, &sx_idx, &sz_idx)) return;
    float sx = tx_val[sx_idx];
    float sz = tz_val[sz_idx];

    if (vcy >= (double)c_det.det_y - 1e-6) return;
    if ((double)c_det.det_y - vcy < 1e-8)  return;

    bool pass = use_pin_context
        ? center_lor_passcheck_pin(
            pin_id, vcx, vcy, vcz, (double)sx, (double)c_det.det_y, (double)sz,
            pnx, pny, pnz, pd, bcx, bcy, bcz, br2)
        : center_lor_passcheck(
            vcx, vcy, vcz, (double)sx, (double)c_det.det_y, (double)sz,
            pnx, pny, pnz, pd, bcx, bcy, bcz, br2);
    if (!pass) {
#if ENABLE_PRECHECK_STATS
        atomicAdd(&g_stat_precheck_skipped, 1ULL);
#endif
        return;
    }
#if ENABLE_PRECHECK_STATS
    atomicAdd(&g_stat_precheck_passed, 1ULL);
#endif

    int rpa = c_cfg.full_det_x * c_cfg.compact_det_z;
    int local_row_base = la * rpa;
    int pix[MAX_DOI_SEG];
    float w[MAX_DOI_SEG];

    for (int sv = 0; sv < N_SUB_VOX; sv++) {
        double vx = vcx + c_vso_x[sv]*ct + c_vso_y[sv]*st;
        double vy = vcy + (-c_vso_x[sv]*st + c_vso_y[sv]*ct);
        double vz = vcz + c_vso_z[sv];
        int ns = compute_doi_sample_contrib(pin_id, use_pin_context, vx, vy, vz, sx, sz, pix, w);
        for (int dd = 0; dd < ns; dd++) {
            append_direct_entry(local_row_base + pix[dd], i, w[dd], vox_out, wgt_out, nnz_out);
        }
    }
}

__global__ void kernel_subray_uniform_doi_v2d(
    const float*  __restrict__ vox_x,
    const float*  __restrict__ vox_y,
    const float*  __restrict__ vox_z,
    const float*  __restrict__ tx_val,
    const float*  __restrict__ tz_val,
    const double* __restrict__ angle_cos,
    const double* __restrict__ angle_sin,
    const double* __restrict__ pnx,
    const double* __restrict__ pny,
    const double* __restrict__ pnz,
    const double* __restrict__ pd,
    const double* __restrict__ bcx,
    const double* __restrict__ bcy,
    const double* __restrict__ bcz,
    const double* __restrict__ br2,
    int  a_off,
    int  a_cnt,
    unsigned long long* __restrict__ v2d_keys,
    float*              __restrict__ v2d_vals,
    unsigned long long* __restrict__ v2d_nnz,
    unsigned long long v2d_cap)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long nva  = (long long)c_cfg.total_voxels * a_cnt;
    bool use_pin_context = (c_cfg.enable_doi_sample_roi != 0);
    int pin_factor = use_pin_context ? c_cfg.n_pinholes : 1;
    long long ntot = nva * (long long)pin_factor * (long long)c_cfg.doi_task_area;
    if (gid >= ntot) return;

    int rid = (int)(gid % c_cfg.doi_task_area);
    long long tmp = gid / c_cfg.doi_task_area;
    int pin_id = 0;
    if (use_pin_context) {
        pin_id = (int)(tmp % c_cfg.n_pinholes);
        tmp /= c_cfg.n_pinholes;
    }
    int la = (int)(tmp % a_cnt);
    int i  = (int)(tmp / a_cnt);
    if (i >= c_cfg.total_voxels) return;

    int angle_idx = a_off + la;
    double ct = angle_cos[angle_idx];
    double st = angle_sin[angle_idx];

    double wx = (double)vox_x[i];
    double wy = (double)vox_y[i];
    double wz = (double)vox_z[i];
    double vcx =  wx*ct + wy*st;
    double vcy = -wx*st + wy*ct;
    double vcz =  wz;

    float proj_x = 0.0f, proj_z = 0.0f;
    if (use_pin_context) {
        int cx_idx, cz_idx;
        if (!project_voxel_to_detector_by_pin(vcx, vcy, vcz, pin_id,
                                               &proj_x, &proj_z, &cx_idx, &cz_idx)) return;
    }

    int sx_idx, sz_idx;
    if (!doi_uniform_sample_indices(rid, proj_x, proj_z, &sx_idx, &sz_idx)) return;
    float sx = tx_val[sx_idx];
    float sz = tz_val[sz_idx];

    if (vcy >= (double)c_det.det_y - 1e-6) return;
    if ((double)c_det.det_y - vcy < 1e-8)  return;

    bool pass = use_pin_context
        ? center_lor_passcheck_pin(
            pin_id, vcx, vcy, vcz, (double)sx, (double)c_det.det_y, (double)sz,
            pnx, pny, pnz, pd, bcx, bcy, bcz, br2)
        : center_lor_passcheck(
            vcx, vcy, vcz, (double)sx, (double)c_det.det_y, (double)sz,
            pnx, pny, pnz, pd, bcx, bcy, bcz, br2);
    if (!pass) {
#if ENABLE_PRECHECK_STATS
        atomicAdd(&g_stat_precheck_skipped, 1ULL);
#endif
        return;
    }
#if ENABLE_PRECHECK_STATS
    atomicAdd(&g_stat_precheck_passed, 1ULL);
#endif

    unsigned long long source_id =
        (unsigned long long)la * (unsigned long long)c_cfg.total_voxels + (unsigned long long)i;
    int pix[MAX_DOI_SEG];
    float w[MAX_DOI_SEG];

    for (int sv = 0; sv < N_SUB_VOX; sv++) {
        double vx = vcx + c_vso_x[sv]*ct + c_vso_y[sv]*st;
        double vy = vcy + (-c_vso_x[sv]*st + c_vso_y[sv]*ct);
        double vz = vcz + c_vso_z[sv];
        int ns = compute_doi_sample_contrib(pin_id, use_pin_context, vx, vy, vz, sx, sz, pix, w);
        for (int dd = 0; dd < ns; dd++) {
            append_v2d_entry(source_id, (unsigned int)pix[dd], w[dd],
                             v2d_keys, v2d_vals, v2d_nnz, v2d_cap);
        }
    }
}

__global__ void kernel_subray_uniform_doi_coo(
    const float*  __restrict__ vox_x,
    const float*  __restrict__ vox_y,
    const float*  __restrict__ vox_z,
    const float*  __restrict__ tx_val,
    const float*  __restrict__ tz_val,
    const double* __restrict__ angle_cos,
    const double* __restrict__ angle_sin,
    const double* __restrict__ pnx,
    const double* __restrict__ pny,
    const double* __restrict__ pnz,
    const double* __restrict__ pd,
    const double* __restrict__ bcx,
    const double* __restrict__ bcy,
    const double* __restrict__ bcz,
    const double* __restrict__ br2,
    int  a_off,
    int  a_cnt,
    unsigned long long* __restrict__ coo_keys,
    float*              __restrict__ coo_vals,
    unsigned long long* __restrict__ coo_nnz,
    unsigned long long coo_cap)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long nva  = (long long)c_cfg.total_voxels * a_cnt;
    bool use_pin_context = (c_cfg.enable_doi_sample_roi != 0);
    int pin_factor = use_pin_context ? c_cfg.n_pinholes : 1;
    long long ntot = nva * (long long)pin_factor * (long long)c_cfg.doi_task_area;
    if (gid >= ntot) return;

    int rid = (int)(gid % c_cfg.doi_task_area);
    long long tmp = gid / c_cfg.doi_task_area;
    int pin_id = 0;
    if (use_pin_context) {
        pin_id = (int)(tmp % c_cfg.n_pinholes);
        tmp /= c_cfg.n_pinholes;
    }
    int la = (int)(tmp % a_cnt);
    int i  = (int)(tmp / a_cnt);
    if (i >= c_cfg.total_voxels) return;

    int angle_idx = a_off + la;
    double ct = angle_cos[angle_idx];
    double st = angle_sin[angle_idx];

    double wx = (double)vox_x[i];
    double wy = (double)vox_y[i];
    double wz = (double)vox_z[i];
    double vcx =  wx*ct + wy*st;
    double vcy = -wx*st + wy*ct;
    double vcz =  wz;

    float proj_x = 0.0f, proj_z = 0.0f;
    if (use_pin_context) {
        int cx_idx, cz_idx;
        if (!project_voxel_to_detector_by_pin(vcx, vcy, vcz, pin_id,
                                               &proj_x, &proj_z, &cx_idx, &cz_idx)) return;
    }

    int sx_idx, sz_idx;
    if (!doi_uniform_sample_indices(rid, proj_x, proj_z, &sx_idx, &sz_idx)) return;
    float sx = tx_val[sx_idx];
    float sz = tz_val[sz_idx];

    if (vcy >= (double)c_det.det_y - 1e-6) return;
    if ((double)c_det.det_y - vcy < 1e-8)  return;

    bool pass = use_pin_context
        ? center_lor_passcheck_pin(
            pin_id, vcx, vcy, vcz, (double)sx, (double)c_det.det_y, (double)sz,
            pnx, pny, pnz, pd, bcx, bcy, bcz, br2)
        : center_lor_passcheck(
            vcx, vcy, vcz, (double)sx, (double)c_det.det_y, (double)sz,
            pnx, pny, pnz, pd, bcx, bcy, bcz, br2);
    if (!pass) {
#if ENABLE_PRECHECK_STATS
        atomicAdd(&g_stat_precheck_skipped, 1ULL);
#endif
        return;
    }
#if ENABLE_PRECHECK_STATS
    atomicAdd(&g_stat_precheck_passed, 1ULL);
#endif

    int rpa = c_cfg.full_det_x * c_cfg.compact_det_z;
    int local_row_base = la * rpa;
    int pix[MAX_DOI_SEG];
    float w[MAX_DOI_SEG];

    for (int sv = 0; sv < N_SUB_VOX; sv++) {
        double vx = vcx + c_vso_x[sv]*ct + c_vso_y[sv]*st;
        double vy = vcy + (-c_vso_x[sv]*st + c_vso_y[sv]*ct);
        double vz = vcz + c_vso_z[sv];
        int ns = compute_doi_sample_contrib(pin_id, use_pin_context, vx, vy, vz, sx, sz, pix, w);
        for (int dd = 0; dd < ns; dd++) {
            unsigned long long row = (unsigned long long)(local_row_base + pix[dd]);
            append_coo_entry(row, (unsigned long long)i, w[dd],
                             (unsigned long long)c_cfg.total_voxels,
                             coo_keys, coo_vals, coo_nnz, coo_cap);
        }
    }
}

/* ========================================================================
   主计算 Kernel v7.3.0 ROI-COO

   线程映射：
     gid -> voxel i, local angle la, pinhole p, ROI offset rid

   与旧版 “voxel × angle × 全探测器像素” 不同，新版只遍历每个
   voxel-angle-pinhole 投影中心附近的 ROI 窗口。窗口大小由 --roi-radius 控制：
     roi_side = 2*roi_radius + 1
     roi_area = roi_side²

   输出：
     不再写巨大定长 row buffer。每个有效贡献写入 COO：
       key = local_row * total_voxels + voxel_index
       val = weight
     chunk 结束后在 GPU 上 sort/reduce，将相同(row, voxel)合并，再只把有效 NNZ 传回 CPU。
======================================================================== */
__global__ void kernel_subray_roi_direct(
    const float*  __restrict__ vox_x,
    const float*  __restrict__ vox_y,
    const float*  __restrict__ vox_z,
    const int*    __restrict__ tx_start,
    const int*    __restrict__ tx_count,
    const float*  __restrict__ tx_val,
    const int*    __restrict__ tz_start,
    const int*    __restrict__ tz_count,
    const float*  __restrict__ tz_val,
    const double* __restrict__ angle_cos,
    const double* __restrict__ angle_sin,
    const double* __restrict__ pnx,
    const double* __restrict__ pny,
    const double* __restrict__ pnz,
    const double* __restrict__ pd,
    const double* __restrict__ bcx,
    const double* __restrict__ bcy,
    const double* __restrict__ bcz,
    const double* __restrict__ br2,
    int  a_off,
    int  a_cnt,
    int*   __restrict__ vox_out,
    float* __restrict__ wgt_out,
    int*   __restrict__ nnz_out)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    long long nva  = (long long)c_cfg.total_voxels * a_cnt;
    long long ntot = nva * (long long)c_cfg.n_pinholes * (long long)c_cfg.roi_area;
    if (gid >= ntot) return;

    int rid = (int)(gid % c_cfg.roi_area);
    long long tmp = gid / c_cfg.roi_area;
    int pin_id = (int)(tmp % c_cfg.n_pinholes);
    tmp /= c_cfg.n_pinholes;
    int la = (int)(tmp % a_cnt);
    int i  = (int)(tmp / a_cnt);
    if (i >= c_cfg.total_voxels) return;

    int dx_roi = (rid % c_cfg.roi_side) - c_cfg.roi_radius;
    int dz_roi = (rid / c_cfg.roi_side) - c_cfg.roi_radius;

    int angle_idx = a_off + la;
    double ct = angle_cos[angle_idx];
    double st = angle_sin[angle_idx];

    double wx = (double)vox_x[i];
    double wy = (double)vox_y[i];
    double wz = (double)vox_z[i];

    /* 体素旋转到当前角度坐标系。 */
    double vcx =  wx*ct + wy*st;
    double vcy = -wx*st + wy*ct;
    double vcz =  wz;

    float proj_x, proj_z;
    int cx_idx, cz_idx;
    if (!project_voxel_to_detector_by_pin(vcx, vcy, vcz, pin_id,
                                          &proj_x, &proj_z, &cx_idx, &cz_idx)) return;

    int ix_pix = cx_idx + dx_roi;
    int iz_pix = cz_idx + dz_roi;
    if (ix_pix < 0 || ix_pix >= c_det.nx) return;
    if (iz_pix < 0 || iz_pix >= c_det.nz) return;
    if (iz_pix >= c_det.compact_det_z) return;

    /* 多个 pinhole ROI 重叠时只由 owner 负责计算，避免重复计数。 */
    if (roi_owner_pinhole(vcx, vcy, vcz, ix_pix, iz_pix) != pin_id) return;

    int k = iz_pix * c_det.nx + ix_pix;
    int rpa = c_cfg.full_det_x * c_cfg.compact_det_z;
    int local_row_base = la * rpa;

    double dkx = (double)c_det.x_centers[ix_pix];
    double dky = (double)c_det.det_y;
    double dkz = (double)c_det.z_centers[iz_pix];

    if (vcy >= dky - 1e-6) return;
    if (dky - vcy < 1e-8)  return;

    bool pass = center_lor_passcheck_pin(
        pin_id, vcx, vcy, vcz, dkx, dky, dkz,
        pnx, pny, pnz, pd, bcx, bcy, bcz, br2);

    if (!pass) {
#if ENABLE_PRECHECK_STATS
        atomicAdd(&g_stat_precheck_skipped, 1ULL);
#endif
        return;
    }
#if ENABLE_PRECHECK_STATS
    atomicAdd(&g_stat_precheck_passed, 1ULL);
#endif

    double inv_nsv = INV_N_SUB_VOX;
    double w_accum = 0.0;

#if COMPILE_ENABLE_DOI
    int   local_pix[LOCAL_DOI_ACCUM];
    float local_w  [LOCAL_DOI_ACCUM];
    int   local_n = 0;
#endif

    int ts_x = tx_start[ix_pix];
    int tc_x = tx_count[ix_pix];
    int ts_z = tz_start[iz_pix];
    int tc_z = tz_count[iz_pix];

    for (int sv = 0; sv < N_SUB_VOX; sv++) {
        double vx = vcx + c_vso_x[sv]*ct + c_vso_y[sv]*st;
        double vy = vcy + (-c_vso_x[sv]*st + c_vso_y[sv]*ct);
        double vz = vcz + c_vso_z[sv];

        if (dky - vy < 1e-8) continue;

        for (int dx_idx = 0; dx_idx < tc_x; dx_idx++) {
            float sx   = tx_val   [ts_x + dx_idx];
            for (int dz_idx = 0; dz_idx < tc_z; dz_idx++) {
                float sz   = tz_val   [ts_z + dz_idx];

                double lx = (double)sx - vx;
                double ly = dky - vy;
                double lz = (double)sz - vz;

                double r2 = lx*lx + ly*ly + lz*lz;
                if (r2 < 1e-12) continue;

                double inv_r = 1.0 / sqrt(r2);
                double ldx = lx*inv_r, ldy = ly*inv_r, ldz = lz*inv_r;
                if (ldy < 1e-8) continue;

                (void)pnx; (void)pny; (void)pnz; (void)pd;
                (void)bcx; (void)bcy; (void)bcz; (void)br2;
                double mp = analytical_penetration_selected_const(pin_id, vx, vy, vz, ldx, ldy, ldz);

                int    pinhole = (mp < (double)c_phys.penetration_length_threshold);
                double P_coll  = 1.0;

                if (!pinhole) {
                    if (!c_phys.enable_penetration) continue;

                    double x_coll = c_phys.mu_collimator * mp;
                    if (x_coll >= c_phys.coll_exp_cutoff_x ||
                        mp     >= c_phys.coll_exp_cutoff_len)
                        continue;

                    P_coll = exp_neg_collimator(x_coll);
                    if (P_coll <= (double)c_phys.penetration_probability_threshold)
                        continue;
                }

                double cp;
                switch (c_phys.solid_angle_power) {
                    case 1:  cp = ldy;          break;
                    case 2:  cp = ldy*ldy;      break;
                    case 3:  cp = ldy*ldy*ldy;  break;
                    default: cp = pow(ldy, (double)c_phys.solid_angle_power);
                }

                double W_ray = cp * inv_r * inv_r;

#if COMPILE_ENABLE_DOI
                if (c_phys.enable_DOI) {
                    double w_sub = W_ray * P_coll * inv_nsv;

                    int   doi_pix[MAX_DOI_SEG];
                    float doi_a  [MAX_DOI_SEG];

                    int ns = compute_doi_spread(
                        (float)vx, (float)vy, (float)vz,
                        (float)ldx, (float)ldy, (float)ldz,
                        doi_pix, doi_a);

                    if (ns == 0) continue;

                    for (int dd = 0; dd < ns; dd++) {
                        int jl = doi_pix[dd];
                        if (jl / c_det.nx >= c_det.compact_det_z) continue;
                        float w_f = (float)(w_sub * (double)doi_a[dd]);
                        if (w_f < c_phys.weight_threshold) continue;

                        bool found = false;
                        for (int s = 0; s < local_n; s++) {
                            if (local_pix[s] == jl) {
                                local_w[s] += w_f;
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            if (local_n < LOCAL_DOI_ACCUM) {
                                local_pix[local_n] = jl;
                                local_w  [local_n] = w_f;
                                local_n++;
                            } else {
                                unsigned long long row2 = (unsigned long long)(local_row_base + jl);
                                append_direct_entry((int)row2, i, w_f, vox_out, wgt_out, nnz_out);
                            }
                        }
                    }
                } else {
#endif
                    bool in_cx = (sx >= c_det.x_edges_flat[2*ix_pix  ] &&
                                  sx <= c_det.x_edges_flat[2*ix_pix+1]);
                    bool in_cz = (sz >= c_det.z_edges_flat[2*iz_pix  ] &&
                                  sz <= c_det.z_edges_flat[2*iz_pix+1]);
                    if (in_cx && in_cz) {
                        w_accum += W_ray * P_coll * inv_nsv;
                    }
#if COMPILE_ENABLE_DOI
                }
#endif
            }
        }
    }

#if COMPILE_ENABLE_DOI
    if (c_phys.enable_DOI) {
        for (int s = 0; s < local_n; s++) {
            if (local_w[s] < c_phys.weight_threshold) continue;
            unsigned long long row = (unsigned long long)(local_row_base + local_pix[s]);
            append_direct_entry((int)row, i, local_w[s], vox_out, wgt_out, nnz_out);
        }
    } else
#endif
    {
        if (w_accum <= 0.0) return;
        float w_f = (float)w_accum;
        if (w_f < c_phys.weight_threshold) return;

        unsigned long long row = (unsigned long long)(local_row_base + k);
        append_direct_entry((int)row, i, w_f, vox_out, wgt_out, nnz_out);
    }
}



__global__ void kernel_subray_roi_v2d(
    const float*  __restrict__ vox_x,
    const float*  __restrict__ vox_y,
    const float*  __restrict__ vox_z,
    const int*    __restrict__ tx_start,
    const int*    __restrict__ tx_count,
    const float*  __restrict__ tx_val,
    const int*    __restrict__ tz_start,
    const int*    __restrict__ tz_count,
    const float*  __restrict__ tz_val,
    const double* __restrict__ angle_cos,
    const double* __restrict__ angle_sin,
    const double* __restrict__ pnx,
    const double* __restrict__ pny,
    const double* __restrict__ pnz,
    const double* __restrict__ pd,
    const double* __restrict__ bcx,
    const double* __restrict__ bcy,
    const double* __restrict__ bcz,
    const double* __restrict__ br2,
    int  a_off,
    int  a_cnt,
    unsigned long long* __restrict__ v2d_keys,
    float*              __restrict__ v2d_vals,
    unsigned long long* __restrict__ v2d_nnz,
    unsigned long long v2d_cap)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    long long nva  = (long long)c_cfg.total_voxels * a_cnt;
    long long ntot = nva * (long long)c_cfg.n_pinholes * (long long)c_cfg.roi_area;
    if (gid >= ntot) return;

    int rid = (int)(gid % c_cfg.roi_area);
    long long tmp = gid / c_cfg.roi_area;
    int pin_id = (int)(tmp % c_cfg.n_pinholes);
    tmp /= c_cfg.n_pinholes;
    int la = (int)(tmp % a_cnt);
    int i  = (int)(tmp / a_cnt);
    if (i >= c_cfg.total_voxels) return;

    int dx_roi = (rid % c_cfg.roi_side) - c_cfg.roi_radius;
    int dz_roi = (rid / c_cfg.roi_side) - c_cfg.roi_radius;

    int angle_idx = a_off + la;
    double ct = angle_cos[angle_idx];
    double st = angle_sin[angle_idx];

    double wx = (double)vox_x[i];
    double wy = (double)vox_y[i];
    double wz = (double)vox_z[i];

    /* 体素旋转到当前角度坐标系。 */
    double vcx =  wx*ct + wy*st;
    double vcy = -wx*st + wy*ct;
    double vcz =  wz;

    float proj_x, proj_z;
    int cx_idx, cz_idx;
    if (!project_voxel_to_detector_by_pin(vcx, vcy, vcz, pin_id,
                                          &proj_x, &proj_z, &cx_idx, &cz_idx)) return;

    int ix_pix = cx_idx + dx_roi;
    int iz_pix = cz_idx + dz_roi;
    if (ix_pix < 0 || ix_pix >= c_det.nx) return;
    if (iz_pix < 0 || iz_pix >= c_det.nz) return;
    if (iz_pix >= c_det.compact_det_z) return;

    /* 多个 pinhole ROI 重叠时只由 owner 负责计算，避免重复计数。 */
    if (roi_owner_pinhole(vcx, vcy, vcz, ix_pix, iz_pix) != pin_id) return;

    int k = iz_pix * c_det.nx + ix_pix;
    int rpa = c_cfg.full_det_x * c_cfg.compact_det_z;
    int local_row_base = la * rpa;

    double dkx = (double)c_det.x_centers[ix_pix];
    double dky = (double)c_det.det_y;
    double dkz = (double)c_det.z_centers[iz_pix];

    if (vcy >= dky - 1e-6) return;
    if (dky - vcy < 1e-8)  return;

    bool pass = center_lor_passcheck_pin(
        pin_id, vcx, vcy, vcz, dkx, dky, dkz,
        pnx, pny, pnz, pd, bcx, bcy, bcz, br2);

    if (!pass) {
#if ENABLE_PRECHECK_STATS
        atomicAdd(&g_stat_precheck_skipped, 1ULL);
#endif
        return;
    }
#if ENABLE_PRECHECK_STATS
    atomicAdd(&g_stat_precheck_passed, 1ULL);
#endif

    double inv_nsv = INV_N_SUB_VOX;
    double w_accum = 0.0;

#if COMPILE_ENABLE_DOI
    int   local_pix[LOCAL_DOI_ACCUM];
    float local_w  [LOCAL_DOI_ACCUM];
    int   local_n = 0;
#endif

    int ts_x = tx_start[ix_pix];
    int tc_x = tx_count[ix_pix];
    int ts_z = tz_start[iz_pix];
    int tc_z = tz_count[iz_pix];

    for (int sv = 0; sv < N_SUB_VOX; sv++) {
        double vx = vcx + c_vso_x[sv]*ct + c_vso_y[sv]*st;
        double vy = vcy + (-c_vso_x[sv]*st + c_vso_y[sv]*ct);
        double vz = vcz + c_vso_z[sv];

        if (dky - vy < 1e-8) continue;

        for (int dx_idx = 0; dx_idx < tc_x; dx_idx++) {
            float sx   = tx_val   [ts_x + dx_idx];
            for (int dz_idx = 0; dz_idx < tc_z; dz_idx++) {
                float sz   = tz_val   [ts_z + dz_idx];

                double lx = (double)sx - vx;
                double ly = dky - vy;
                double lz = (double)sz - vz;

                double r2 = lx*lx + ly*ly + lz*lz;
                if (r2 < 1e-12) continue;

                double inv_r = 1.0 / sqrt(r2);
                double ldx = lx*inv_r, ldy = ly*inv_r, ldz = lz*inv_r;
                if (ldy < 1e-8) continue;

                (void)pnx; (void)pny; (void)pnz; (void)pd;
                (void)bcx; (void)bcy; (void)bcz; (void)br2;
                double mp = analytical_penetration_selected_const(pin_id, vx, vy, vz, ldx, ldy, ldz);

                int    pinhole = (mp < (double)c_phys.penetration_length_threshold);
                double P_coll  = 1.0;

                if (!pinhole) {
                    if (!c_phys.enable_penetration) continue;

                    double x_coll = c_phys.mu_collimator * mp;
                    if (x_coll >= c_phys.coll_exp_cutoff_x ||
                        mp     >= c_phys.coll_exp_cutoff_len)
                        continue;

                    P_coll = exp_neg_collimator(x_coll);
                    if (P_coll <= (double)c_phys.penetration_probability_threshold)
                        continue;
                }

                double cp;
                switch (c_phys.solid_angle_power) {
                    case 1:  cp = ldy;          break;
                    case 2:  cp = ldy*ldy;      break;
                    case 3:  cp = ldy*ldy*ldy;  break;
                    default: cp = pow(ldy, (double)c_phys.solid_angle_power);
                }

                double W_ray = cp * inv_r * inv_r;

#if COMPILE_ENABLE_DOI
                if (c_phys.enable_DOI) {
                    double w_sub = W_ray * P_coll * inv_nsv;

                    int   doi_pix[MAX_DOI_SEG];
                    float doi_a  [MAX_DOI_SEG];

                    int ns = compute_doi_spread(
                        (float)vx, (float)vy, (float)vz,
                        (float)ldx, (float)ldy, (float)ldz,
                        doi_pix, doi_a);

                    if (ns == 0) continue;

                    for (int dd = 0; dd < ns; dd++) {
                        int jl = doi_pix[dd];
                        if (jl / c_det.nx >= c_det.compact_det_z) continue;
                        float w_f = (float)(w_sub * (double)doi_a[dd]);
                        if (w_f < c_phys.weight_threshold) continue;

                        bool found = false;
                        for (int s = 0; s < local_n; s++) {
                            if (local_pix[s] == jl) {
                                local_w[s] += w_f;
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            if (local_n < LOCAL_DOI_ACCUM) {
                                local_pix[local_n] = jl;
                                local_w  [local_n] = w_f;
                                local_n++;
                            } else {
                                unsigned long long source_id = (unsigned long long)la * (unsigned long long)c_cfg.total_voxels + (unsigned long long)i;
                                append_v2d_entry(source_id, (unsigned int)jl, w_f,
                                                 v2d_keys, v2d_vals, v2d_nnz, v2d_cap);
                            }
                        }
                    }
                } else {
#endif
                    bool in_cx = (sx >= c_det.x_edges_flat[2*ix_pix  ] &&
                                  sx <= c_det.x_edges_flat[2*ix_pix+1]);
                    bool in_cz = (sz >= c_det.z_edges_flat[2*iz_pix  ] &&
                                  sz <= c_det.z_edges_flat[2*iz_pix+1]);
                    if (in_cx && in_cz) {
                        w_accum += W_ray * P_coll * inv_nsv;
                    }
#if COMPILE_ENABLE_DOI
                }
#endif
            }
        }
    }

#if COMPILE_ENABLE_DOI
    if (c_phys.enable_DOI) {
        for (int s = 0; s < local_n; s++) {
            if (local_w[s] < c_phys.weight_threshold) continue;
            unsigned long long source_id = (unsigned long long)la * (unsigned long long)c_cfg.total_voxels + (unsigned long long)i;
            append_v2d_entry(source_id, (unsigned int)local_pix[s], local_w[s],
                             v2d_keys, v2d_vals, v2d_nnz, v2d_cap);
        }
    } else
#endif
    {
        if (w_accum <= 0.0) return;
        float w_f = (float)w_accum;
        if (w_f < c_phys.weight_threshold) return;

        unsigned long long source_id = (unsigned long long)la * (unsigned long long)c_cfg.total_voxels + (unsigned long long)i;
        append_v2d_entry(source_id, (unsigned int)k, w_f,
                         v2d_keys, v2d_vals, v2d_nnz, v2d_cap);
    }
}


/* ========================================================================
   CSR-EDGE 自适应聚合路径

   目标：一次几何计算，block 内按 source=(angle,voxel) 聚合 detector，
   直接输出唯一 edge(row,col,val)，随后由 Host/GPU scatter 成 CSR-Binary。

   agg_mode:
     0 = dense accumulator + bitset，适合 RPA 小，例如 48x24=1152。
     1 = shared hash accumulator，适合 RPA 中/大，例如 96x96=9216。
======================================================================== */
__device__ __forceinline__ void emit_csr_edge_entry(
    unsigned int row,
    unsigned int col,
    float val,
    unsigned int* __restrict__ edge_rows,
    unsigned int* __restrict__ edge_cols,
    float*        __restrict__ edge_vals,
    unsigned long long* __restrict__ edge_nnz,
    unsigned long long edge_cap,
    unsigned long long* __restrict__ row_counts)
{
    if (val < c_phys.weight_threshold) return;
    unsigned long long pos = atomicAdd(edge_nnz, 1ULL);
    if (pos < edge_cap) {
        edge_rows[pos] = row;
        edge_cols[pos] = col;
        edge_vals[pos] = val;
        atomicAdd(&row_counts[row], 1ULL);
    } else {
        atomicAdd(&g_stat_nnz_overflow, 1u);
    }
}

__device__ __forceinline__ void csr_dense_add(
    int det_id,
    float w,
    int rpa,
    float* __restrict__ det_acc,
    unsigned int* __restrict__ det_mask)
{
    if (w < c_phys.weight_threshold) return;
    if (det_id < 0 || det_id >= rpa) return;
    atomicAdd(&det_acc[det_id], w);
    atomicOr(&det_mask[det_id >> 5], 1u << (det_id & 31));
}

__device__ __forceinline__ void csr_hash_add(
    int det_id,
    float w,
    int rpa,
    unsigned int* __restrict__ sh_key,
    float* __restrict__ sh_val,
    int hash_size,
    int* __restrict__ overflow_flag)
{
    if (w < c_phys.weight_threshold) return;
    if (det_id < 0 || det_id >= rpa) return;

    const unsigned int EMPTY = 0xFFFFFFFFu;
    unsigned int key = (unsigned int)det_id;
    unsigned int h = (key * 2654435761u) & (unsigned int)(hash_size - 1);

    for (int probe = 0; probe < hash_size; ++probe) {
        unsigned int old = atomicCAS(&sh_key[h], EMPTY, key);
        if (old == EMPTY || old == key) {
            atomicAdd(&sh_val[h], w);
            return;
        }
        h = (h + 1u) & (unsigned int)(hash_size - 1);
    }

    atomicExch(overflow_flag, 1);
    atomicAdd(&g_stat_hash_overflow, 1u);
}

__device__ __forceinline__ void csr_agg_add(
    int agg_mode,
    int det_id,
    float w,
    int rpa,
    float* __restrict__ det_acc,
    unsigned int* __restrict__ det_mask,
    unsigned int* __restrict__ sh_key,
    float* __restrict__ sh_val,
    int hash_size,
    int* __restrict__ hash_overflow)
{
    if (agg_mode == 0) {
        csr_dense_add(det_id, w, rpa, det_acc, det_mask);
    } else {
        csr_hash_add(det_id, w, rpa, sh_key, sh_val, hash_size, hash_overflow);
    }
}

__global__ void kernel_source_roi_csr_edges(
    const float*  __restrict__ vox_x,
    const float*  __restrict__ vox_y,
    const float*  __restrict__ vox_z,
    const int*    __restrict__ tx_start,
    const int*    __restrict__ tx_count,
    const float*  __restrict__ tx_val,
    const int*    __restrict__ tz_start,
    const int*    __restrict__ tz_count,
    const float*  __restrict__ tz_val,
    const double* __restrict__ angle_cos,
    const double* __restrict__ angle_sin,
    const double* __restrict__ pnx,
    const double* __restrict__ pny,
    const double* __restrict__ pnz,
    const double* __restrict__ pd,
    const double* __restrict__ bcx,
    const double* __restrict__ bcy,
    const double* __restrict__ bcz,
    const double* __restrict__ br2,
    int  a_off,
    int  a_cnt,
    int  agg_mode,
    int  hash_size,
    unsigned int* __restrict__ edge_rows,
    unsigned int* __restrict__ edge_cols,
    float*        __restrict__ edge_vals,
    unsigned long long* __restrict__ edge_nnz,
    unsigned long long edge_cap,
    unsigned long long* __restrict__ row_counts)
{
    unsigned long long source_gid = (unsigned long long)blockIdx.x;
    unsigned long long nva = (unsigned long long)c_cfg.total_voxels * (unsigned long long)a_cnt;
    if (source_gid >= nva) return;

    int i  = (int)(source_gid % (unsigned long long)c_cfg.total_voxels);
    int la = (int)(source_gid / (unsigned long long)c_cfg.total_voxels);
    int rpa = c_cfg.compact_rays_per_angle;
    int mask_words = (rpa + 31) >> 5;

    extern __shared__ unsigned char shbuf[];

    float* det_acc = nullptr;
    unsigned int* det_mask = nullptr;
    unsigned int* sh_key = nullptr;
    float* sh_val = nullptr;
    int* hash_overflow = nullptr;

    if (agg_mode == 0) {
        det_acc = (float*)shbuf;
        det_mask = (unsigned int*)(det_acc + rpa);
        for (int q = threadIdx.x; q < rpa; q += blockDim.x) det_acc[q] = 0.0f;
        for (int q = threadIdx.x; q < mask_words; q += blockDim.x) det_mask[q] = 0u;
    } else {
        sh_key = (unsigned int*)shbuf;
        sh_val = (float*)(sh_key + hash_size);
        hash_overflow = (int*)(sh_val + hash_size);
        const unsigned int EMPTY = 0xFFFFFFFFu;
        for (int q = threadIdx.x; q < hash_size; q += blockDim.x) {
            sh_key[q] = EMPTY;
            sh_val[q] = 0.0f;
        }
        if (threadIdx.x == 0) *hash_overflow = 0;
    }
    __syncthreads();

    int angle_idx = a_off + la;
    double ct = angle_cos[angle_idx];
    double st = angle_sin[angle_idx];

    double wx = (double)vox_x[i];
    double wy = (double)vox_y[i];
    double wz = (double)vox_z[i];

    /* 体素旋转到当前角度坐标系。 */
    double vcx =  wx*ct + wy*st;
    double vcy = -wx*st + wy*ct;
    double vcz =  wz;

#if COMPILE_ENABLE_DOI
    if (c_phys.enable_DOI) {
        bool use_pin_context = (c_cfg.enable_doi_sample_roi != 0);
        int pin_factor = use_pin_context ? c_cfg.n_pinholes : 1;
        int tasks = pin_factor * c_cfg.doi_task_area;
        for (int task = threadIdx.x; task < tasks; task += blockDim.x) {
            int rid = task % c_cfg.doi_task_area;
            int pin_id = use_pin_context ? (task / c_cfg.doi_task_area) : 0;

            float proj_x = 0.0f, proj_z = 0.0f;
            if (use_pin_context) {
                int cx_idx, cz_idx;
                if (!project_voxel_to_detector_by_pin(vcx, vcy, vcz, pin_id,
                                                       &proj_x, &proj_z, &cx_idx, &cz_idx)) continue;
            }

            int sx_idx, sz_idx;
            if (!doi_uniform_sample_indices(rid, proj_x, proj_z, &sx_idx, &sz_idx)) continue;
            float sx = tx_val[sx_idx];
            float sz = tz_val[sz_idx];

            if (vcy >= (double)c_det.det_y - 1e-6) continue;
            if ((double)c_det.det_y - vcy < 1e-8)  continue;

            bool pass = use_pin_context
                ? center_lor_passcheck_pin(
                    pin_id, vcx, vcy, vcz, (double)sx, (double)c_det.det_y, (double)sz,
                    pnx, pny, pnz, pd, bcx, bcy, bcz, br2)
                : center_lor_passcheck(
                    vcx, vcy, vcz, (double)sx, (double)c_det.det_y, (double)sz,
                    pnx, pny, pnz, pd, bcx, bcy, bcz, br2);
            if (!pass) {
#if ENABLE_PRECHECK_STATS
                atomicAdd(&g_stat_precheck_skipped, 1ULL);
#endif
                continue;
            }
#if ENABLE_PRECHECK_STATS
            atomicAdd(&g_stat_precheck_passed, 1ULL);
#endif

            int pix[MAX_DOI_SEG];
            float w[MAX_DOI_SEG];
            for (int sv = 0; sv < N_SUB_VOX; sv++) {
                double vx = vcx + c_vso_x[sv]*ct + c_vso_y[sv]*st;
                double vy = vcy + (-c_vso_x[sv]*st + c_vso_y[sv]*ct);
                double vz = vcz + c_vso_z[sv];
                int ns = compute_doi_sample_contrib(pin_id, use_pin_context, vx, vy, vz, sx, sz, pix, w);
                for (int dd = 0; dd < ns; dd++) {
                    csr_agg_add(agg_mode, pix[dd], w[dd], rpa,
                                det_acc, det_mask, sh_key, sh_val, hash_size, hash_overflow);
                }
            }
        }
    } else
#endif
    {
    int tasks = c_cfg.n_pinholes * c_cfg.roi_area;
    for (int task = threadIdx.x; task < tasks; task += blockDim.x) {
        int rid = task % c_cfg.roi_area;
        int pin_id = task / c_cfg.roi_area;

        int dx_roi = (rid % c_cfg.roi_side) - c_cfg.roi_radius;
        int dz_roi = (rid / c_cfg.roi_side) - c_cfg.roi_radius;

        float proj_x, proj_z;
        int cx_idx, cz_idx;
        if (!project_voxel_to_detector_by_pin(vcx, vcy, vcz, pin_id,
                                              &proj_x, &proj_z, &cx_idx, &cz_idx)) continue;

        int ix_pix = cx_idx + dx_roi;
        int iz_pix = cz_idx + dz_roi;
        if (ix_pix < 0 || ix_pix >= c_det.nx) continue;
        if (iz_pix < 0 || iz_pix >= c_det.nz) continue;
        if (iz_pix >= c_det.compact_det_z) continue;

        /* 多个 pinhole ROI 重叠时只由 owner 负责计算，避免重复计数。 */
        if (roi_owner_pinhole(vcx, vcy, vcz, ix_pix, iz_pix) != pin_id) continue;

        int k = iz_pix * c_det.nx + ix_pix;

        double dkx = (double)c_det.x_centers[ix_pix];
        double dky = (double)c_det.det_y;
        double dkz = (double)c_det.z_centers[iz_pix];

        if (vcy >= dky - 1e-6) continue;
        if (dky - vcy < 1e-8)  continue;

        bool pass = center_lor_passcheck_pin(
            pin_id, vcx, vcy, vcz, dkx, dky, dkz,
            pnx, pny, pnz, pd, bcx, bcy, bcz, br2);

        if (!pass) {
#if ENABLE_PRECHECK_STATS
            atomicAdd(&g_stat_precheck_skipped, 1ULL);
#endif
            continue;
        }
#if ENABLE_PRECHECK_STATS
        atomicAdd(&g_stat_precheck_passed, 1ULL);
#endif

        double inv_nsv = INV_N_SUB_VOX;
        double w_accum = 0.0;

#if COMPILE_ENABLE_DOI
        int   local_pix[LOCAL_DOI_ACCUM];
        float local_w  [LOCAL_DOI_ACCUM];
        int   local_n = 0;
#endif

        int ts_x = tx_start[ix_pix];
        int tc_x = tx_count[ix_pix];
        int ts_z = tz_start[iz_pix];
        int tc_z = tz_count[iz_pix];

        for (int sv = 0; sv < N_SUB_VOX; sv++) {
            double vx = vcx + c_vso_x[sv]*ct + c_vso_y[sv]*st;
            double vy = vcy + (-c_vso_x[sv]*st + c_vso_y[sv]*ct);
            double vz = vcz + c_vso_z[sv];

            if (dky - vy < 1e-8) continue;

            for (int dx_idx = 0; dx_idx < tc_x; dx_idx++) {
                float sx   = tx_val   [ts_x + dx_idx];
                for (int dz_idx = 0; dz_idx < tc_z; dz_idx++) {
                    float sz   = tz_val   [ts_z + dz_idx];

                    double lx = (double)sx - vx;
                    double ly = dky - vy;
                    double lz = (double)sz - vz;

                    double r2 = lx*lx + ly*ly + lz*lz;
                    if (r2 < 1e-12) continue;

                    double inv_r = 1.0 / sqrt(r2);
                    double ldx = lx*inv_r, ldy = ly*inv_r, ldz = lz*inv_r;
                    if (ldy < 1e-8) continue;

                    (void)pnx; (void)pny; (void)pnz; (void)pd;
                    (void)bcx; (void)bcy; (void)bcz; (void)br2;
                    double mp = analytical_penetration_selected_const(pin_id, vx, vy, vz, ldx, ldy, ldz);

                    int    pinhole = (mp < (double)c_phys.penetration_length_threshold);
                    double P_coll  = 1.0;

                    if (!pinhole) {
                        if (!c_phys.enable_penetration) continue;

                        double x_coll = c_phys.mu_collimator * mp;
                        if (x_coll >= c_phys.coll_exp_cutoff_x ||
                            mp     >= c_phys.coll_exp_cutoff_len)
                            continue;

                        P_coll = exp_neg_collimator(x_coll);
                        if (P_coll <= (double)c_phys.penetration_probability_threshold)
                            continue;
                    }

                    double cp;
                    switch (c_phys.solid_angle_power) {
                        case 1:  cp = ldy;          break;
                        case 2:  cp = ldy*ldy;      break;
                        case 3:  cp = ldy*ldy*ldy;  break;
                        default: cp = pow(ldy, (double)c_phys.solid_angle_power);
                    }

                    double W_ray = cp * inv_r * inv_r;

#if COMPILE_ENABLE_DOI
                    if (c_phys.enable_DOI) {
                        double w_sub = W_ray * P_coll * inv_nsv;

                        int   doi_pix[MAX_DOI_SEG];
                        float doi_a  [MAX_DOI_SEG];

                        int ns = compute_doi_spread(
                            (float)vx, (float)vy, (float)vz,
                            (float)ldx, (float)ldy, (float)ldz,
                            doi_pix, doi_a);

                        if (ns == 0) continue;

                        for (int dd = 0; dd < ns; dd++) {
                            int jl = doi_pix[dd];
                            if (jl / c_det.nx >= c_det.compact_det_z) continue;
                            float w_f = (float)(w_sub * (double)doi_a[dd]);
                            if (w_f < c_phys.weight_threshold) continue;

                            bool found = false;
                            for (int s = 0; s < local_n; s++) {
                                if (local_pix[s] == jl) {
                                    local_w[s] += w_f;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                if (local_n < LOCAL_DOI_ACCUM) {
                                    local_pix[local_n] = jl;
                                    local_w  [local_n] = w_f;
                                    local_n++;
                                } else {
                                    csr_agg_add(agg_mode, jl, w_f, rpa,
                                                det_acc, det_mask, sh_key, sh_val, hash_size, hash_overflow);
                                }
                            }
                        }
                    } else {
#endif
                        bool in_cx = (sx >= c_det.x_edges_flat[2*ix_pix  ] &&
                                      sx <= c_det.x_edges_flat[2*ix_pix+1]);
                        bool in_cz = (sz >= c_det.z_edges_flat[2*iz_pix  ] &&
                                      sz <= c_det.z_edges_flat[2*iz_pix+1]);
                        if (in_cx && in_cz) {
                            w_accum += W_ray * P_coll * inv_nsv;
                        }
#if COMPILE_ENABLE_DOI
                    }
#endif
                }
            }
        }

#if COMPILE_ENABLE_DOI
        if (c_phys.enable_DOI) {
            for (int s = 0; s < local_n; s++) {
                if (local_w[s] < c_phys.weight_threshold) continue;
                csr_agg_add(agg_mode, local_pix[s], local_w[s], rpa,
                            det_acc, det_mask, sh_key, sh_val, hash_size, hash_overflow);
            }
        } else
#endif
        {
            if (w_accum > 0.0) {
                float w_f = (float)w_accum;
                if (w_f >= c_phys.weight_threshold) {
                    csr_agg_add(agg_mode, k, w_f, rpa,
                                det_acc, det_mask, sh_key, sh_val, hash_size, hash_overflow);
                }
            }
        }
    }
    }

    __syncthreads();

    unsigned int col = (unsigned int)i;
    if (agg_mode == 0) {
        for (int word = threadIdx.x; word < mask_words; word += blockDim.x) {
            unsigned int m = det_mask[word];
            while (m) {
                int b = __ffs(m) - 1;
                int det_id = word * 32 + b;
                if (det_id < rpa) {
                    float w = det_acc[det_id];
                    if (w >= c_phys.weight_threshold) {
                        unsigned int row = (unsigned int)(la * rpa + det_id);
                        emit_csr_edge_entry(row, col, w,
                                            edge_rows, edge_cols, edge_vals, edge_nnz, edge_cap, row_counts);
                    }
                }
                m &= (m - 1u);
            }
        }
    } else {
        if (hash_overflow && *hash_overflow) {
            /* 已在 csr_hash_add 中累计 g_stat_hash_overflow。这里不做 fallback，调用端必须检查。 */
        }
        const unsigned int EMPTY = 0xFFFFFFFFu;
        for (int h = threadIdx.x; h < hash_size; h += blockDim.x) {
            unsigned int det_id = sh_key[h];
            if (det_id != EMPTY && (int)det_id < rpa) {
                float w = sh_val[h];
                if (w >= c_phys.weight_threshold) {
                    unsigned int row = (unsigned int)(la * rpa + (int)det_id);
                    emit_csr_edge_entry(row, col, w,
                                        edge_rows, edge_cols, edge_vals, edge_nnz, edge_cap, row_counts);
                }
            }
        }
    }
}




__global__ void kernel_subray_roi_coo(
    const float*  __restrict__ vox_x,
    const float*  __restrict__ vox_y,
    const float*  __restrict__ vox_z,
    const int*    __restrict__ tx_start,
    const int*    __restrict__ tx_count,
    const float*  __restrict__ tx_val,
    const int*    __restrict__ tz_start,
    const int*    __restrict__ tz_count,
    const float*  __restrict__ tz_val,
    const double* __restrict__ angle_cos,
    const double* __restrict__ angle_sin,
    const double* __restrict__ pnx,
    const double* __restrict__ pny,
    const double* __restrict__ pnz,
    const double* __restrict__ pd,
    const double* __restrict__ bcx,
    const double* __restrict__ bcy,
    const double* __restrict__ bcz,
    const double* __restrict__ br2,
    int  a_off,
    int  a_cnt,
    unsigned long long* __restrict__ coo_keys,
    float*              __restrict__ coo_vals,
    unsigned long long* __restrict__ coo_nnz,
    unsigned long long coo_cap)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    long long nva  = (long long)c_cfg.total_voxels * a_cnt;
    long long ntot = nva * (long long)c_cfg.n_pinholes * (long long)c_cfg.roi_area;
    if (gid >= ntot) return;

    int rid = (int)(gid % c_cfg.roi_area);
    long long tmp = gid / c_cfg.roi_area;
    int pin_id = (int)(tmp % c_cfg.n_pinholes);
    tmp /= c_cfg.n_pinholes;
    int la = (int)(tmp % a_cnt);
    int i  = (int)(tmp / a_cnt);
    if (i >= c_cfg.total_voxels) return;

    int dx_roi = (rid % c_cfg.roi_side) - c_cfg.roi_radius;
    int dz_roi = (rid / c_cfg.roi_side) - c_cfg.roi_radius;

    int angle_idx = a_off + la;
    double ct = angle_cos[angle_idx];
    double st = angle_sin[angle_idx];

    double wx = (double)vox_x[i];
    double wy = (double)vox_y[i];
    double wz = (double)vox_z[i];

    /* 体素旋转到当前角度坐标系。 */
    double vcx =  wx*ct + wy*st;
    double vcy = -wx*st + wy*ct;
    double vcz =  wz;

    float proj_x, proj_z;
    int cx_idx, cz_idx;
    if (!project_voxel_to_detector_by_pin(vcx, vcy, vcz, pin_id,
                                          &proj_x, &proj_z, &cx_idx, &cz_idx)) return;

    int ix_pix = cx_idx + dx_roi;
    int iz_pix = cz_idx + dz_roi;
    if (ix_pix < 0 || ix_pix >= c_det.nx) return;
    if (iz_pix < 0 || iz_pix >= c_det.nz) return;
    if (iz_pix >= c_det.compact_det_z) return;

    /* 多个 pinhole ROI 重叠时只由 owner 负责计算，避免重复计数。 */
    if (roi_owner_pinhole(vcx, vcy, vcz, ix_pix, iz_pix) != pin_id) return;

    int k = iz_pix * c_det.nx + ix_pix;
    int rpa = c_cfg.full_det_x * c_cfg.compact_det_z;
    int local_row_base = la * rpa;

    double dkx = (double)c_det.x_centers[ix_pix];
    double dky = (double)c_det.det_y;
    double dkz = (double)c_det.z_centers[iz_pix];

    if (vcy >= dky - 1e-6) return;
    if (dky - vcy < 1e-8)  return;

    bool pass = center_lor_passcheck_pin(
        pin_id, vcx, vcy, vcz, dkx, dky, dkz,
        pnx, pny, pnz, pd, bcx, bcy, bcz, br2);

    if (!pass) {
#if ENABLE_PRECHECK_STATS
        atomicAdd(&g_stat_precheck_skipped, 1ULL);
#endif
        return;
    }
#if ENABLE_PRECHECK_STATS
    atomicAdd(&g_stat_precheck_passed, 1ULL);
#endif

    double inv_nsv = INV_N_SUB_VOX;
    double w_accum = 0.0;

#if COMPILE_ENABLE_DOI
    int   local_pix[LOCAL_DOI_ACCUM];
    float local_w  [LOCAL_DOI_ACCUM];
    int   local_n = 0;
#endif

    int ts_x = tx_start[ix_pix];
    int tc_x = tx_count[ix_pix];
    int ts_z = tz_start[iz_pix];
    int tc_z = tz_count[iz_pix];

    for (int sv = 0; sv < N_SUB_VOX; sv++) {
        double vx = vcx + c_vso_x[sv]*ct + c_vso_y[sv]*st;
        double vy = vcy + (-c_vso_x[sv]*st + c_vso_y[sv]*ct);
        double vz = vcz + c_vso_z[sv];

        if (dky - vy < 1e-8) continue;

        for (int dx_idx = 0; dx_idx < tc_x; dx_idx++) {
            float sx   = tx_val   [ts_x + dx_idx];
            for (int dz_idx = 0; dz_idx < tc_z; dz_idx++) {
                float sz   = tz_val   [ts_z + dz_idx];

                double lx = (double)sx - vx;
                double ly = dky - vy;
                double lz = (double)sz - vz;

                double r2 = lx*lx + ly*ly + lz*lz;
                if (r2 < 1e-12) continue;

                double inv_r = 1.0 / sqrt(r2);
                double ldx = lx*inv_r, ldy = ly*inv_r, ldz = lz*inv_r;
                if (ldy < 1e-8) continue;

                (void)pnx; (void)pny; (void)pnz; (void)pd;
                (void)bcx; (void)bcy; (void)bcz; (void)br2;
                double mp = analytical_penetration_selected_const(pin_id, vx, vy, vz, ldx, ldy, ldz);

                int    pinhole = (mp < (double)c_phys.penetration_length_threshold);
                double P_coll  = 1.0;

                if (!pinhole) {
                    if (!c_phys.enable_penetration) continue;

                    double x_coll = c_phys.mu_collimator * mp;
                    if (x_coll >= c_phys.coll_exp_cutoff_x ||
                        mp     >= c_phys.coll_exp_cutoff_len)
                        continue;

                    P_coll = exp_neg_collimator(x_coll);
                    if (P_coll <= (double)c_phys.penetration_probability_threshold)
                        continue;
                }

                double cp;
                switch (c_phys.solid_angle_power) {
                    case 1:  cp = ldy;          break;
                    case 2:  cp = ldy*ldy;      break;
                    case 3:  cp = ldy*ldy*ldy;  break;
                    default: cp = pow(ldy, (double)c_phys.solid_angle_power);
                }

                double W_ray = cp * inv_r * inv_r;

#if COMPILE_ENABLE_DOI
                if (c_phys.enable_DOI) {
                    double w_sub = W_ray * P_coll * inv_nsv;

                    int   doi_pix[MAX_DOI_SEG];
                    float doi_a  [MAX_DOI_SEG];

                    int ns = compute_doi_spread(
                        (float)vx, (float)vy, (float)vz,
                        (float)ldx, (float)ldy, (float)ldz,
                        doi_pix, doi_a);

                    if (ns == 0) continue;

                    for (int dd = 0; dd < ns; dd++) {
                        int jl = doi_pix[dd];
                        if (jl / c_det.nx >= c_det.compact_det_z) continue;
                        float w_f = (float)(w_sub * (double)doi_a[dd]);
                        if (w_f < c_phys.weight_threshold) continue;

                        bool found = false;
                        for (int s = 0; s < local_n; s++) {
                            if (local_pix[s] == jl) {
                                local_w[s] += w_f;
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            if (local_n < LOCAL_DOI_ACCUM) {
                                local_pix[local_n] = jl;
                                local_w  [local_n] = w_f;
                                local_n++;
                            } else {
                                unsigned long long row2 = (unsigned long long)(local_row_base + jl);
                                append_coo_entry(row2, (unsigned long long)i, w_f,
                                                 (unsigned long long)c_cfg.total_voxels,
                                                 coo_keys, coo_vals, coo_nnz, coo_cap);
                            }
                        }
                    }
                } else {
#endif
                    bool in_cx = (sx >= c_det.x_edges_flat[2*ix_pix  ] &&
                                  sx <= c_det.x_edges_flat[2*ix_pix+1]);
                    bool in_cz = (sz >= c_det.z_edges_flat[2*iz_pix  ] &&
                                  sz <= c_det.z_edges_flat[2*iz_pix+1]);
                    if (in_cx && in_cz) {
                        w_accum += W_ray * P_coll * inv_nsv;
                    }
#if COMPILE_ENABLE_DOI
                }
#endif
            }
        }
    }

#if COMPILE_ENABLE_DOI
    if (c_phys.enable_DOI) {
        for (int s = 0; s < local_n; s++) {
            if (local_w[s] < c_phys.weight_threshold) continue;
            unsigned long long row = (unsigned long long)(local_row_base + local_pix[s]);
            append_coo_entry(row, (unsigned long long)i, local_w[s],
                             (unsigned long long)c_cfg.total_voxels,
                             coo_keys, coo_vals, coo_nnz, coo_cap);
        }
    } else
#endif
    {
        if (w_accum <= 0.0) return;
        float w_f = (float)w_accum;
        if (w_f < c_phys.weight_threshold) return;

        unsigned long long row = (unsigned long long)(local_row_base + k);
        append_coo_entry(row, (unsigned long long)i, w_f,
                         (unsigned long long)c_cfg.total_voxels,
                         coo_keys, coo_vals, coo_nnz, coo_cap);
    }
}

/* ========================================================================
   Host: 几何辅助函数 (与 v726 完全一致)
======================================================================== */
struct Plane6 {
    double nx[6], ny[6], nz[6], d[6];
};

typedef std::array<double,3>  Vec3h;
typedef std::array<Vec3h,4>   Quad;
typedef std::vector<Quad>     PinLayer;
typedef std::vector<PinLayer> PinLayers;

static inline void cross3(const double a[3], const double b[3], double r[3]) {
    r[0]=a[1]*b[2]-a[2]*b[1]; r[1]=a[2]*b[0]-a[0]*b[2]; r[2]=a[0]*b[1]-a[1]*b[0];
}
static inline double dot3(const double a[3], const double b[3]) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
static inline double norm3(const double a[3]) { return sqrt(dot3(a,a)); }

static void make_plane_quad(
    const double v[4][3], const double center[3],
    double* nx, double* ny, double* nz, double* d)
{
    double cx=0.25*(v[0][0]+v[1][0]+v[2][0]+v[3][0]);
    double cy=0.25*(v[0][1]+v[1][1]+v[2][1]+v[3][1]);
    double cz=0.25*(v[0][2]+v[1][2]+v[2][2]+v[3][2]);

    double d1[3]={v[2][0]-v[0][0], v[2][1]-v[0][1], v[2][2]-v[0][2]};
    double d2[3]={v[3][0]-v[1][0], v[3][1]-v[1][1], v[3][2]-v[1][2]};
    double n[3]; cross3(d1,d2,n);
    double nl=norm3(n);

    if (nl < 1e-15) {
        double e1[3]={v[1][0]-v[0][0],v[1][1]-v[0][1],v[1][2]-v[0][2]};
        double e2[3]={v[2][0]-v[0][0],v[2][1]-v[0][1],v[2][2]-v[0][2]};
        cross3(e1,e2,n); nl=norm3(n);
        if (nl < 1e-15) {
            double e3[3]={v[3][0]-v[0][0],v[3][1]-v[0][1],v[3][2]-v[0][2]};
            cross3(e1,e3,n); nl=norm3(n);
        }
    }
    n[0]/=nl; n[1]/=nl; n[2]/=nl;
    double dd=n[0]*cx+n[1]*cy+n[2]*cz;
    double cd[3]={center[0]-cx, center[1]-cy, center[2]-cz};
    if (dot3(n,cd) > 1e-10) { n[0]=-n[0]; n[1]=-n[1]; n[2]=-n[2]; dd=-dd; }
    *nx=n[0]; *ny=n[1]; *nz=n[2]; *d=dd;
}

static Plane6 hex_planes(const double vlo[4][3], const double vhi[4][3]) {
    Plane6 P;
    double center[3]={0,0,0};
    for (int i=0;i<4;i++) for (int d=0;d<3;d++) center[d]+=vlo[i][d]+vhi[i][d];
    center[0]/=8; center[1]/=8; center[2]/=8;

    double q[4][3];
    for (int i=0;i<4;i++){q[i][0]=vlo[i][0];q[i][1]=vlo[i][1];q[i][2]=vlo[i][2];}
    make_plane_quad(q,center,&P.nx[0],&P.ny[0],&P.nz[0],&P.d[0]);
    for (int i=0;i<4;i++){q[i][0]=vhi[i][0];q[i][1]=vhi[i][1];q[i][2]=vhi[i][2];}
    make_plane_quad(q,center,&P.nx[1],&P.ny[1],&P.nz[1],&P.d[1]);
    for (int s=0;s<4;s++) {
        int i0=s, i1=(s+1)%4;
        q[0][0]=vlo[i0][0];q[0][1]=vlo[i0][1];q[0][2]=vlo[i0][2];
        q[1][0]=vlo[i1][0];q[1][1]=vlo[i1][1];q[1][2]=vlo[i1][2];
        q[2][0]=vhi[i1][0];q[2][1]=vhi[i1][1];q[2][2]=vhi[i1][2];
        q[3][0]=vhi[i0][0];q[3][1]=vhi[i0][1];q[3][2]=vhi[i0][2];
        make_plane_quad(q,center,&P.nx[2+s],&P.ny[2+s],&P.nz[2+s],&P.d[2+s]);
    }
    return P;
}

static PinLayers match_vertex_correspondence(const PinLayers& in) {
    PinLayers out = in;
    int n_layers=(int)out.size(); if (n_layers<2) return out;
    int n_pin=(int)out[0].size();
    static const int perms[8][4]={
        {0,1,2,3},{1,2,3,0},{2,3,0,1},{3,0,1,2},
        {3,2,1,0},{0,3,2,1},{1,0,3,2},{2,1,0,3}
    };
    for (int p=0;p<n_pin;p++) {
        double c_ref[3]={0,0,0};
        for (int i=0;i<4;i++) for (int d=0;d<3;d++) c_ref[d]+=out[0][p][i][d];
        for (int d=0;d<3;d++) c_ref[d]*=0.25;
        for (int li=1;li<n_layers;li++) {
            double c_tgt[3]={0,0,0};
            for (int i=0;i<4;i++) for (int d=0;d<3;d++) c_tgt[d]+=out[li][p][i][d];
            for (int d=0;d<3;d++) c_tgt[d]*=0.25;
            double axis[3]={c_tgt[0]-c_ref[0],c_tgt[1]-c_ref[1],c_tgt[2]-c_ref[2]};
            double axn=norm3(axis); if (axn<1e-12) continue;
            axis[0]/=axn; axis[1]/=axn; axis[2]/=axn;
            double u0[3]={out[0][p][0][0]-c_ref[0],out[0][p][0][1]-c_ref[1],out[0][p][0][2]-c_ref[2]};
            double proj=dot3(u0,axis);
            u0[0]-=proj*axis[0]; u0[1]-=proj*axis[1]; u0[2]-=proj*axis[2];
            double ul=norm3(u0); if (ul<1e-12) continue;
            u0[0]/=ul; u0[1]/=ul; u0[2]/=ul;
            double v0[3]; cross3(axis,u0,v0);
            double ref_ang[4], tgt_ang[4];
            for (int i=0;i<4;i++) {
                double dv[3]={out[0][p][i][0]-c_ref[0],out[0][p][i][1]-c_ref[1],out[0][p][i][2]-c_ref[2]};
                ref_ang[i]=atan2(dot3(dv,v0),dot3(dv,u0));
                double dv2[3]={out[li][p][i][0]-c_tgt[0],out[li][p][i][1]-c_tgt[1],out[li][p][i][2]-c_tgt[2]};
                tgt_ang[i]=atan2(dot3(dv2,v0),dot3(dv2,u0));
            }
            int best_perm=0; double best_cost=1e30;
            for (int pi=0;pi<8;pi++) {
                double cost=0;
                for (int i=0;i<4;i++) {
                    double diff=tgt_ang[perms[pi][i]]-ref_ang[i];
                    diff=fmod(diff+3*M_PI,2*M_PI)-M_PI;
                    cost+=fabs(diff);
                }
                if (cost<best_cost){best_cost=cost;best_perm=pi;}
            }
            if (best_perm!=0) {
                Quad reordered;
                for (int i=0;i<4;i++) reordered[i]=out[li][p][perms[best_perm][i]];
                out[li][p]=reordered;
                printf("[match_vertex] pinhole %d layer %d: reorder (%d,%d,%d,%d)\n",
                       p,li,perms[best_perm][0],perms[best_perm][1],
                       perms[best_perm][2],perms[best_perm][3]);
            }
            for (int d=0;d<3;d++) c_ref[d]=c_tgt[d];
        }
    }
    return out;
}

struct CollimatorHost {
    int n_poly = 0;
    int n_solid_poly = 0;
    int n_void_poly = 0;
    int geom_kind = COLL_GEOM_RECT;
    std::vector<double> planes_nx, planes_ny, planes_nz, planes_d;
    std::vector<double> bc_x, bc_y, bc_z, br2;
    double R_inner, R_outer, H_half;
    std::vector<double> pin_x, pin_y, pin_z;
};

static void rotate_xy(double x, double y, double cp, double sp, double* xr, double* yr)
{
    *xr = x*cp - y*sp;
    *yr = x*sp + y*cp;
}

static void store_hexahedron(CollimatorHost& C, int pi, const double vlo[4][3], const double vhi[4][3])
{
    Plane6 P6=hex_planes(vlo,vhi);
    int base=pi*6;
    for (int f=0;f<6;f++) {
        C.planes_nx[base+f]=P6.nx[f]; C.planes_ny[base+f]=P6.ny[f];
        C.planes_nz[base+f]=P6.nz[f]; C.planes_d [base+f]=P6.d [f];
    }
    double cx=0,cy=0,cz=0;
    for (int v=0;v<4;v++){cx+=vlo[v][0]+vhi[v][0];cy+=vlo[v][1]+vhi[v][1];cz+=vlo[v][2]+vhi[v][2];}
    cx/=8;cy/=8;cz/=8;
    double mr2=0;
    for (int v=0;v<4;v++) for (int w=0;w<2;w++) {
        const double (*vv)[3]=(w==0)?vlo:vhi;
        double dx=vv[v][0]-cx,dy=vv[v][1]-cy,dz=vv[v][2]-cz;
        double r2=dx*dx+dy*dy+dz*dz; if(r2>mr2)mr2=r2;
    }
    C.bc_x[pi]=cx; C.bc_y[pi]=cy; C.bc_z[pi]=cz; C.br2[pi]=mr2;
}

static CollimatorHost build_rect_collimator(
    double box_width_x, double box_length_z,
    double box_y_min, double box_y_max,
    const PinLayers& layers_in, int n_modules)
{
    CollimatorHost C;
    int n_layers=(int)layers_in.size();
    int n_pin=(int)layers_in[0].size();
    PinLayers layers=match_vertex_correspondence(layers_in);
    int nfpp=n_layers-1;
    if (n_modules <= 0) n_modules = 1;

    C.geom_kind=COLL_GEOM_RECT;
    C.R_inner=0.0;
    C.R_outer=0.0;
    C.H_half=0.0;
    C.n_solid_poly=n_modules;
    C.n_void_poly=n_pin*n_modules*nfpp;
    C.n_poly=C.n_solid_poly+C.n_void_poly;
    C.planes_nx.resize((size_t)C.n_poly*6); C.planes_ny.resize((size_t)C.n_poly*6);
    C.planes_nz.resize((size_t)C.n_poly*6); C.planes_d .resize((size_t)C.n_poly*6);
    C.bc_x.resize(C.n_poly); C.bc_y.resize(C.n_poly);
    C.bc_z.resize(C.n_poly); C.br2 .resize(C.n_poly);

    double ylo = fmin(box_y_min, box_y_max);
    double yhi = fmax(box_y_min, box_y_max);
    double aperture_y_min = 1.0e300;
    double aperture_y_max = -1.0e300;
    for (int k=0;k<n_layers;k++) for (int p=0;p<n_pin;p++) for (int v=0;v<4;v++) {
        double y = layers[k][p][v][1];
        if (y < aperture_y_min) aperture_y_min = y;
        if (y > aperture_y_max) aperture_y_max = y;
    }

    double hx = 0.5 * box_width_x;
    double hz = 0.5 * box_length_z;
    int pi=0;
    for (int m=0;m<n_modules;m++) {
        double phi=m*2.0*M_PI/(double)n_modules;
        double cp=cos(phi), sp=sin(phi);

        double box_lo[4][3]={
            {-hx,ylo, hz},{ hx,ylo, hz},{ hx,ylo,-hz},{-hx,ylo,-hz}
        };
        double box_hi[4][3]={
            {-hx,yhi, hz},{ hx,yhi, hz},{ hx,yhi,-hz},{-hx,yhi,-hz}
        };
        for (int v=0;v<4;v++) {
            rotate_xy(box_lo[v][0], box_lo[v][1], cp, sp, &box_lo[v][0], &box_lo[v][1]);
            rotate_xy(box_hi[v][0], box_hi[v][1], cp, sp, &box_hi[v][0], &box_hi[v][1]);
        }
        store_hexahedron(C, pi++, box_lo, box_hi);
    }

    for (int m=0;m<n_modules;m++) {
        double phi=m*2.0*M_PI/(double)n_modules;
        double cp=cos(phi), sp=sin(phi);
        for (int p=0;p<n_pin;p++) {
            std::vector<std::array<std::array<double,3>,4>> rl(n_layers);
            for (int k=0;k<n_layers;k++) for (int v=0;v<4;v++) {
                double x0=layers[k][p][v][0], y0=layers[k][p][v][1];
                rotate_xy(x0, y0, cp, sp, &rl[k][v][0], &rl[k][v][1]);
                rl[k][v][2]=layers[k][p][v][2];
            }

            int mid_layer0 = (n_layers >= 4) ? 1 : ((n_layers >= 3) ? 1 : 0);
            int mid_layer1 = (n_layers >= 4) ? 2 : mid_layer0;
            double pcx=0.0, pcy=0.0, pcz=0.0;
            for (int ml=mid_layer0; ml<=mid_layer1; ml++) {
                for (int v=0; v<4; v++) {
                    pcx += rl[ml][v][0];
                    pcy += rl[ml][v][1];
                    pcz += rl[ml][v][2];
                }
            }
            double inv_mid_count = 1.0 / (4.0 * (double)(mid_layer1 - mid_layer0 + 1));
            C.pin_x.push_back(pcx*inv_mid_count);
            C.pin_y.push_back(pcy*inv_mid_count);
            C.pin_z.push_back(pcz*inv_mid_count);

            for (int s=0;s<nfpp;s++) {
                double vlo[4][3], vhi[4][3];
                for (int v=0;v<4;v++) for (int d=0;d<3;d++) {
                    vlo[v][d]=rl[s  ][v][d]; vhi[v][d]=rl[s+1][v][d];
                }
                store_hexahedron(C, pi++, vlo, vhi);
            }
        }
    }
    printf("Collimator: rect %.3f(x) x %.3f(z), box y=[%.6f, %.6f], aperture y=[%.6f, %.6f], modules=%d, pins/module=%d, solids=%d, voids=%d\n",
           box_width_x, box_length_z, ylo, yhi, aperture_y_min, aperture_y_max,
           n_modules, n_pin, C.n_solid_poly, C.n_void_poly);
    return C;
}

template<int N_PIN>
static PinLayer make_pin_layer(const double arr[N_PIN][4][3]) {
    PinLayer L(N_PIN);
    for (int p=0;p<N_PIN;p++) for (int v=0;v<4;v++) for (int d=0;d<3;d++) L[p][v][d]=arr[p][v][d];
    return L;
}

/* ========================================================================
   Host: 体素与探测器构建
======================================================================== */
struct VoxelHost {
    std::vector<float> x, y, z;
    int Nx, Ny, Nz, total;
};

static VoxelHost build_voxels(int Nx, int Ny, int Nz, float d) {
    VoxelHost v;
    v.Nx=Nx; v.Ny=Ny; v.Nz=Nz; v.total=Nx*Ny*Nz;
    v.x.resize(v.total); v.y.resize(v.total); v.z.resize(v.total);
    int idx=0;
    for (int iz=0;iz<Nz;iz++) {
        float fz=(iz-(Nz-1)/2.0f)*d;
        for (int iy=0;iy<Ny;iy++) {
            float fy=(iy-(Ny-1)/2.0f)*d;
            for (int ix=0;ix<Nx;ix++) {
                v.x[idx]=(ix-(Nx-1)/2.0f)*d; v.y[idx]=fy; v.z[idx]=fz; idx++;
            }
        }
    }
    return v;
}

struct AxisTargets {
    std::vector<int>   start;   /* 每个像素对应的采样起始索引 */
    std::vector<int>   count;   /* 每个像素对应的采样点数 */
    std::vector<float> val;     /* 采样点坐标 */
};

static AxisTargets build_crystal_axis_targets(
    int n,                          /* 晶体数 */
    const std::vector<float>& centers, /* 各晶体中心坐标 */
    float ps,                       /* 晶体物理尺寸（像素大小） */
    int S_det)                      /* 晶体区域等分数 */
{
    AxisTargets T;
    T.start.resize(n);
    T.count.resize(n);

    for (int i = 0; i < n; i++) {
        T.start[i] = (int)T.val.size();

        float c_left = centers[i] - ps * 0.5f;
        float step_c = ps / (float)S_det;
        for (int j = 0; j < S_det; j++) {
            T.val.push_back(c_left + (j + 0.5f) * step_c);
        }

        T.count[i] = (int)T.val.size() - T.start[i];
    }

    return T;
}

/*
  DOI 专用均匀轴采样：在整个探测器物理范围内取等间距中点。
  这里不再把采样点分配到任何晶体编号；ROI 也在连续采样网格上裁剪。
*/
static std::vector<float> build_uniform_axis_values(
    float lo,
    float hi,
    int n_samples,
    float* step_out)
{
    std::vector<float> val;
    if (n_samples < 1) n_samples = 1;
    float span = hi - lo;
    if (step_out) *step_out = 0.0f;
    if (span <= 0.0f) {
        val.push_back(lo);
        return val;
    }

    float step = span / (float)n_samples;
    if (step_out) *step_out = step;
    val.reserve(n_samples);
    for (int j = 0; j < n_samples; j++) {
        val.push_back(lo + (j + 0.5f) * step);
    }
    return val;
}

struct DetectorHost {
    std::vector<float> cx, cy, cz;
    std::vector<float> x_centers, z_centers;
    std::vector<float> x_edges_flat;  /* [2*nx]：每晶体物理左右边界 */
    std::vector<float> z_edges_flat;  /* [2*nz]：每晶体物理上下边界 */
    AxisTargets tx, tz;               /* 非 DOI：每个晶体本体内的前表面采样点 */
    std::vector<float> doi_x, doi_z;  /* DOI 模式：全探测器范围均匀采样点 */
    float doi_x0=0.0f, doi_z0=0.0f, doi_dx=0.0f, doi_dz=0.0f;
    float x_span=0.0f, z_span=0.0f;
    float det_y, back_y, thickness;
    int nx, nz;
};

static DetectorHost build_detector(
    int nx, int nz, float ps, float gs,
    float dy, float th, float mg, int ms,
    int doi_radial_samples,
    bool build_crystal_targets,
    bool build_doi_targets)
{
    DetectorHost D;
    D.nx=nx; D.nz=nz;
    D.det_y=dy; D.thickness=th; D.back_y=dy+th;

    /* 生成各晶体中心坐标（含模块内小间隙和模块间大间隙） */
    auto make_centers = [&](int n) -> std::vector<float> {
        std::vector<float> c(n, 0.f);
        for (int i=1;i<n;i++) {
            bool cr=(i%ms==0);
            c[i]=c[i-1]+ps+(cr?mg:gs);
        }
        float m=(c[0]+c[n-1])*0.5f;
        for (auto& v:c) v-=m;
        return c;
    };

    auto xc=make_centers(nx);
    auto zc=make_centers(nz);
    D.x_centers = xc;
    D.z_centers = zc;

    /* 存储每个晶体的物理边界（用于 DOI 精确判断） */
    D.x_edges_flat.resize(2*nx);
    D.z_edges_flat.resize(2*nz);
    for (int i=0;i<nx;i++) {
        D.x_edges_flat[2*i  ]=xc[i]-ps*0.5f;
        D.x_edges_flat[2*i+1]=xc[i]+ps*0.5f;
    }
    for (int i=0;i<nz;i++) {
        D.z_edges_flat[2*i  ]=zc[i]-ps*0.5f;
        D.z_edges_flat[2*i+1]=zc[i]+ps*0.5f;
    }

    /* 探测器像素中心（用于主射线方向的中心 LOR） */
    D.cx.resize(nx*nz); D.cy.resize(nx*nz); D.cz.resize(nx*nz);
    for (int iz=0;iz<nz;iz++) for (int ix=0;ix<nx;ix++) {
        int l=iz*nx+ix;
        D.cx[l]=xc[ix]; D.cy[l]=dy; D.cz[l]=zc[iz];
    }

    if (build_crystal_targets) {
        D.tx = build_crystal_axis_targets(nx, xc, ps, S_DET);
        D.tz = build_crystal_axis_targets(nz, zc, ps, S_DET);
    }

    float x_lo = D.x_edges_flat[0];
    float x_hi = D.x_edges_flat[2*(nx - 1) + 1];
    float z_lo = D.z_edges_flat[0];
    float z_hi = D.z_edges_flat[2*(nz - 1) + 1];
    D.x_span = x_hi - x_lo;
    D.z_span = z_hi - z_lo;

    if (build_doi_targets) {
        if (doi_radial_samples < 1) doi_radial_samples = 1;
        int doi_axial_samples = (D.x_span > 0.0f)
            ? (int)roundf((float)doi_radial_samples * D.z_span / D.x_span)
            : doi_radial_samples;
        if (doi_axial_samples < 1) doi_axial_samples = 1;

        D.doi_x = build_uniform_axis_values(x_lo, x_hi, doi_radial_samples, &D.doi_dx);
        D.doi_z = build_uniform_axis_values(z_lo, z_hi, doi_axial_samples, &D.doi_dz);
        D.doi_x0 = D.doi_x.empty() ? x_lo : D.doi_x[0];
        D.doi_z0 = D.doi_z.empty() ? z_lo : D.doi_z[0];
    }

    if (build_crystal_targets) {
        int total_tx = (int)D.tx.val.size();
        int total_tz = (int)D.tz.val.size();
        printf("✓ Detector crystal targets: X axis %d samples (avg %.1f/pixel), "
               "Z axis %d samples (avg %.1f/pixel)\n",
               total_tx, (float)total_tx/nx,
               total_tz, (float)total_tz/nz);
    }

    if (build_doi_targets) {
        printf("✓ DOI uniform targets: X axis %d samples over %.3f mm (avg %.1f/pixel), "
               "Z axis %d samples over %.3f mm (avg %.1f/pixel)\n",
               (int)D.doi_x.size(), D.x_span, (float)D.doi_x.size()/nx,
               (int)D.doi_z.size(), D.z_span, (float)D.doi_z.size()/nz);
    }

    return D;
}

/* ========================================================================
   工具函数
======================================================================== */
template<typename T>
static void gpu_up(T** d, const std::vector<T>& h) {
    CUDA_CHECK(cudaMalloc(d, h.size()*sizeof(T)));
    CUDA_CHECK(cudaMemcpy(*d, h.data(), h.size()*sizeof(T), cudaMemcpyHostToDevice));
}



static const uint32_t CSRB_MAGIC = 0x42525343u; /* 'CSRB' little-endian */
static const uint32_t CSRB_VERSION = 1u;

static bool save_chunk_csr_binary(
    const char* fn,
    int rt,
    int ao,
    int ac,
    const std::vector<int64_t>& row_ptr,
    const std::vector<int32_t>& cols,
    const std::vector<float>& wg)
{
    /*
      CSRB chunk format, written directly for reconstruction:
        uint32 magic      = 'CSRB'
        uint32 version    = 1
        int32  num_rows   = angle_count * compact_rays_per_angle
        int32  angle_offset
        int32  angle_count
        int64  nnz
        int64  row_ptr[num_rows + 1]
        int32  col_idx[nnz]
        float  values[nnz]

      该格式避免旧 chunk_*.csrb 中 “每行 n + col/val” 在重建端再次扫描转换 CSR。
    */
    if ((int)row_ptr.size() != rt + 1) return false;
    int64_t nnz = row_ptr.back();
    if (nnz < 0 || (size_t)nnz != cols.size() || cols.size() != wg.size()) return false;

    FILE* f = fopen(fn, "wb");
    if (!f) return false;
    fwrite(&CSRB_MAGIC,   4, 1, f);
    fwrite(&CSRB_VERSION, 4, 1, f);
    fwrite(&rt,           4, 1, f);
    fwrite(&ao,           4, 1, f);
    fwrite(&ac,           4, 1, f);
    fwrite(&nnz,          8, 1, f);
    fwrite(row_ptr.data(), 8, (size_t)rt + 1, f);
    if (nnz > 0) {
        fwrite(cols.data(), 4, (size_t)nnz, f);
        fwrite(wg.data(),   4, (size_t)nnz, f);
    }
    fclose(f);
    return true;
}

static void save_chunk_direct(
    const char* fn,
    const int* vi_1based,
    const float* wg_in,
    int* nz,
    int ao,
    int ac,
    int rt)
{
    /*
      DIRECT row buffer -> CSRB。
      direct kernel 内部 voxel index 是 1-based；写入文件时转换为 0-based。
    */
    std::vector<int64_t> row_ptr((size_t)rt + 1, 0);
    int64_t nnz = 0;
    for (int r = 0; r < rt; r++) {
        int n = nz[r];
        if (n > MAX_NNZ_PER_ROW) n = MAX_NNZ_PER_ROW;
        if (n < 0) n = 0;
        row_ptr[(size_t)r] = nnz;
        nnz += (int64_t)n;
    }
    row_ptr[(size_t)rt] = nnz;

    std::vector<int32_t> cols((size_t)nnz);
    std::vector<float>   wg((size_t)nnz);
    for (int r = 0; r < rt; r++) {
        int64_t out = row_ptr[(size_t)r];
        int n = (int)(row_ptr[(size_t)r + 1] - row_ptr[(size_t)r]);
        long long base = (long long)r * MAX_NNZ_PER_ROW;
        for (int k = 0; k < n; k++) {
            cols[(size_t)out + k] = vi_1based[base + k] - 1;
            wg  [(size_t)out + k] = wg_in[base + k];
        }
    }
    save_chunk_csr_binary(fn, rt, ao, ac, row_ptr, cols, wg);
}

static void save_chunk_from_coo(
    const char* fn,
    std::vector<unsigned long long>& keys,
    std::vector<float>& vals,
    int total_voxels,
    int ao,
    int ac,
    int rt,
    float weight_threshold)
{
    /*
      Reduced detector-row COO -> CSRB。
      key = row * total_voxels + col。
      为兼容调试输入，这里仍检查排序；正式路径 GPU 已 sort/reduce。
    */
    if (keys.size() != vals.size()) return;

    bool sorted = true;
    for (size_t i = 1; i < keys.size(); i++) {
        if (keys[i-1] > keys[i]) { sorted = false; break; }
    }
    if (!sorted) {
        std::vector<size_t> idx(keys.size());
        for (size_t i = 0; i < idx.size(); i++) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return keys[a] < keys[b]; });
        std::vector<unsigned long long> k2;
        std::vector<float> v2;
        k2.reserve(keys.size());
        v2.reserve(vals.size());
        for (size_t id : idx) {
            if (!k2.empty() && k2.back() == keys[id]) v2.back() += vals[id];
            else { k2.push_back(keys[id]); v2.push_back(vals[id]); }
        }
        keys.swap(k2);
        vals.swap(v2);
    }

    std::vector<int64_t> row_ptr((size_t)rt + 1, 0);
    std::vector<int32_t> cols;
    std::vector<float>   wg;
    cols.reserve(keys.size());
    wg.reserve(vals.size());

    size_t pos = 0;
    for (int r = 0; r < rt; r++) {
        row_ptr[(size_t)r] = (int64_t)cols.size();
        while (pos < keys.size()) {
            unsigned long long row = keys[pos] / (unsigned long long)total_voxels;
            if ((int)row < r) { pos++; continue; }
            if ((int)row > r) break;
            float v = vals[pos];
            unsigned long long col = keys[pos] % (unsigned long long)total_voxels;
            if (v >= weight_threshold && col < (unsigned long long)total_voxels) {
                cols.push_back((int32_t)col);
                wg.push_back(v);
            }
            pos++;
        }
    }
    row_ptr[(size_t)rt] = (int64_t)cols.size();
    save_chunk_csr_binary(fn, rt, ao, ac, row_ptr, cols, wg);
}


static void save_chunk_from_v2d(
    const char* fn,
    const std::vector<unsigned long long>& keys,
    const std::vector<float>& vals,
    int total_voxels,
    int rpa,
    int ao,
    int ac,
    int rt,
    float weight_threshold)
{
    /*
      V2D reduced keys:
          key = source_id * rpa + det_id
          source_id = local_angle * total_voxels + voxel_id

      这里直接转置并保存为 CSRB：
          row = local_angle * rpa + det_id
          col = voxel_id

      GPU reduce by V2D key 后，每个 (local_angle, voxel, detector) 唯一，
      转置得到的 (row=(local_angle,detector), col=voxel) 也天然唯一。
    */
    if (keys.size() != vals.size() || rpa <= 0 || total_voxels <= 0) return;

    std::vector<int64_t> row_counts((size_t)rt, 0);
    size_t valid = 0;
    for (size_t p = 0; p < keys.size(); p++) {
        float v = vals[p];
        if (v < weight_threshold) continue;
        unsigned long long source_det = keys[p];
        unsigned long long source_id = source_det / (unsigned long long)rpa;
        unsigned int det_id = (unsigned int)(source_det % (unsigned long long)rpa);
        unsigned long long la = source_id / (unsigned long long)total_voxels;
        unsigned long long col = source_id % (unsigned long long)total_voxels;
        if (la >= (unsigned long long)ac || det_id >= (unsigned int)rpa || col >= (unsigned long long)total_voxels) continue;
        int row = (int)la * rpa + (int)det_id;
        if (row < 0 || row >= rt) continue;
        row_counts[(size_t)row]++;
        valid++;
    }

    std::vector<int64_t> row_ptr((size_t)rt + 1, 0);
    for (int r = 0; r < rt; r++) row_ptr[(size_t)r + 1] = row_ptr[(size_t)r] + row_counts[(size_t)r];

    std::vector<int64_t> write_pos = row_ptr;
    std::vector<int32_t> cols(valid);
    std::vector<float> wg(valid);

    for (size_t p = 0; p < keys.size(); p++) {
        float v = vals[p];
        if (v < weight_threshold) continue;
        unsigned long long source_det = keys[p];
        unsigned long long source_id = source_det / (unsigned long long)rpa;
        unsigned int det_id = (unsigned int)(source_det % (unsigned long long)rpa);
        unsigned long long la = source_id / (unsigned long long)total_voxels;
        unsigned long long col = source_id % (unsigned long long)total_voxels;
        if (la >= (unsigned long long)ac || det_id >= (unsigned int)rpa || col >= (unsigned long long)total_voxels) continue;
        int row = (int)la * rpa + (int)det_id;
        if (row < 0 || row >= rt) continue;
        int64_t q = write_pos[(size_t)row]++;
        cols[(size_t)q] = (int32_t)col;
        wg[(size_t)q] = v;
    }

    save_chunk_csr_binary(fn, rt, ao, ac, row_ptr, cols, wg);
}


static void save_chunk_from_edges(
    const char* fn,
    const std::vector<unsigned int>& edge_rows,
    const std::vector<unsigned int>& edge_cols,
    const std::vector<float>& edge_vals,
    const std::vector<unsigned long long>& h_row_counts,
    int total_voxels,
    int rt,
    int ao,
    int ac,
    float weight_threshold)
{
    /*
      CSR-EDGE 路径：kernel 已经保证每个 source=(angle,voxel) 内部按 det_id 聚合，
      因此 edge(row,col,val) 理论上不重复。这里只按 row_counts 做 prefix，然后散射写 CSRB。

      为了避免一次性排序，这里不保证 row 内 col 有序。重建 SpMV/自写 forward/backward
      通常不要求列有序；若后续需要二分/压缩，可增加可选 row 内排序。
    */
    size_t edge_nnz = edge_vals.size();
    if (edge_rows.size() != edge_nnz || edge_cols.size() != edge_nnz) return;
    if ((int)h_row_counts.size() != rt) return;

    std::vector<int64_t> row_ptr((size_t)rt + 1, 0);
    for (int r = 0; r < rt; r++) {
        row_ptr[(size_t)r + 1] = row_ptr[(size_t)r] + (int64_t)h_row_counts[(size_t)r];
    }
    int64_t expected_nnz = row_ptr[(size_t)rt];

    std::vector<int64_t> write_pos = row_ptr;
    std::vector<int32_t> cols((size_t)expected_nnz);
    std::vector<float>   wg((size_t)expected_nnz);

    size_t dropped = 0;
    for (size_t e = 0; e < edge_nnz; e++) {
        float v = edge_vals[e];
        if (v < weight_threshold) { dropped++; continue; }
        unsigned int row = edge_rows[e];
        unsigned int col = edge_cols[e];
        if ((int)row >= rt || (int)col >= total_voxels) { dropped++; continue; }
        int64_t q = write_pos[(size_t)row]++;
        if (q < 0 || q >= expected_nnz) { dropped++; continue; }
        cols[(size_t)q] = (int32_t)col;
        wg[(size_t)q]   = v;
    }

    /* 如果由于过滤/异常导致实际写入少于 row_counts，压缩一次。正常情况下不会进入。 */
    bool mismatch = false;
    for (int r = 0; r < rt; r++) {
        if (write_pos[(size_t)r] != row_ptr[(size_t)r + 1]) { mismatch = true; break; }
    }
    if (mismatch || dropped) {
        std::vector<int64_t> row_counts2((size_t)rt, 0);
        for (size_t e = 0; e < edge_nnz; e++) {
            float v = edge_vals[e];
            unsigned int row = edge_rows[e];
            unsigned int col = edge_cols[e];
            if (v >= weight_threshold && (int)row < rt && (int)col < total_voxels) row_counts2[(size_t)row]++;
        }
        row_ptr.assign((size_t)rt + 1, 0);
        for (int r = 0; r < rt; r++) row_ptr[(size_t)r + 1] = row_ptr[(size_t)r] + row_counts2[(size_t)r];
        write_pos = row_ptr;
        cols.assign((size_t)row_ptr[(size_t)rt], 0);
        wg.assign((size_t)row_ptr[(size_t)rt], 0.0f);
        for (size_t e = 0; e < edge_nnz; e++) {
            float v = edge_vals[e];
            unsigned int row = edge_rows[e];
            unsigned int col = edge_cols[e];
            if (v < weight_threshold || (int)row >= rt || (int)col >= total_voxels) continue;
            int64_t q = write_pos[(size_t)row]++;
            cols[(size_t)q] = (int32_t)col;
            wg[(size_t)q] = v;
        }
    }

    save_chunk_csr_binary(fn, rt, ao, ac, row_ptr, cols, wg);
}


static bool starts_with(const std::string& s, const std::string& p) {
    return s.size()>=p.size()&&s.compare(0,p.size(),p)==0;
}
static bool ends_with(const std::string& s, const std::string& p) {
    return s.size()>=p.size()&&s.compare(s.size()-p.size(),p.size(),p)==0;
}
static void clean_old_output(const std::string& dir) {
#ifdef _WIN32
    std::string pattern = dir + "\\*";
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        std::string name(fd.cFileName);
        if ((starts_with(name,"chunk_")&&(ends_with(name,".bin")||ends_with(name,".csrb")))||
             name=="metadata.bin"||name=="metadata.txt"||name=="symmetry_info.bin") {
            std::string path=dir+"\\"+name; remove(path.c_str());
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR* d=opendir(dir.c_str()); if(!d) return;
    struct dirent* ent;
    while((ent=readdir(d))!=NULL) {
        std::string name(ent->d_name);
        if ((starts_with(name,"chunk_")&&(ends_with(name,".bin")||ends_with(name,".csrb")))||
             name=="metadata.bin"||name=="metadata.txt"||name=="symmetry_info.bin") {
            std::string path=dir+"/"+name; remove(path.c_str());
        }
    }
    closedir(d);
#endif
}
static std::vector<int> parse_gpu_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s); std::string item;
    while(std::getline(ss,item,',')) if(!item.empty()) out.push_back(atoi(item.c_str()));
    return out;
}

static int parse_voxel_rotation_sign(const char* s) {
    if (!strcmp(s, "cw") || !strcmp(s, "+") || !strcmp(s, "+1") || !strcmp(s, "1")) return 1;
    if (!strcmp(s, "ccw") || !strcmp(s, "-") || !strcmp(s, "-1")) return -1;
    int v = atoi(s);
    return (v < 0) ? -1 : 1;
}

/* ========================================================================
   RunConfig & Metadata
======================================================================== */
struct RunConfig {
    std::string output_dir = "matrix_PSC_6modules_24pin_YSO_1024x1024_DOI_PEN_141keV";
    std::string sym_mode   = "90";  /* 90/45/none */

    int Nx=104, Ny=104, Nz=74;
    int det_nx=DET_NX_DEFAULT, det_nz=DET_NZ_DEFAULT;
    int num_angles=360;

    float voxel_size=0.3f;
    float pixel_size=1.55f;
    float gap_size=0.16f;
    float detector_y=94.0f;
    float detector_thickness=9.0f;
    float module_gap_f=1.34f;
    int   module_size=12;
    int   doi_radial_samples=DOI_RADIAL_SAMPLES_DEFAULT;
    int   enable_doi_sample_roi=0;

    /* 6 模块是当前硬件的物理真相；此前默认 1 会静默生成单模块矩阵。 */
    int   coll_modules=6;
    int   fast_pinhole_penetration=0;
    float coll_box_width_x=68.2f;
    float coll_box_length_z=100.0f;
    float coll_box_y_min=32.0f;
    float coll_box_y_max=60.0f;
    int   voxel_rotation_sign=1;

    bool use_zsym=true;
    int  chunk_angles=23;
    int  clean_output=1;

    /* ROI 半窗口，窗口边长=2*roi_radius+1。建议先用 6~8 做保守验证，再逐步缩小到 4~5。 */
    int  roi_radius=12;

    /*
      输出模式：
        direct: GPU 常驻定长 row buffer，跳过 COO sort/reduce；显存占用高，速度通常更快。
        coo:    COO + sort/reduce；显存更可控，适合作为验证/回退模式。
    */
    std::string output_mode = "csr";

    /*
      CSR 自适应聚合模式：
        adaptive: RPA <= dense_rpa_limit 时用 dense accumulator，否则用 shared hash。
        dense:    强制 dense accumulator + bitset。
        hash:     强制 shared hash。
      output_mode=csr/csr-adaptive/csr-dense/csr-hash 时使用。
    */
    std::string csr_agg_mode = "adaptive";
    int dense_rpa_limit = 2048;
    int hash_size = 4096;
    int edge_capacity_million = 2400;

    /* 每个 chunk 的 COO/V2D 原始条目容量。单位：百万条。旧 v2d/coo 回退路径使用。 */
    int  coo_capacity_million=120;
    int  gpu_reduce=1;
    int  num_gpus=2;
    std::vector<int> gpu_list;

    PhysicsConst phys;
};

struct WorkerStats {
    unsigned int      nnz_overflow=0;
    unsigned int      hash_overflow=0;
    unsigned long long precheck_skipped=0;
    unsigned long long precheck_passed=0;
    double seconds=0.0;
};

static void save_metadata(const RunConfig& cfg, int compact_na, int compact_dz, int rpa, int total_rows, int total_voxels) {
    std::string mf=cfg.output_dir+"/metadata.bin";
    FILE* f=fopen(mf.c_str(),"wb"); if(!f) return;
    int mn=MAX_NNZ_PER_ROW;  /* 兼容旧读取器；v7.3.0 实际输出不再受该值限制 */
    int full_na=cfg.num_angles;
    int use90=(cfg.sym_mode=="90")?1:0;
    int use45=(cfg.sym_mode=="45")?1:0;
    int zsym=cfg.use_zsym?1:0;
    int sym_code = use45 ? 45 : (use90 ? 90 : 0);
    int symmetry_order = use45 ? 8 : (use90 ? 4 : 1);

    /* 旧字段顺序保持不变，兼容已有读取程序。 */
    fwrite(&total_rows,4,1,f); fwrite(&total_voxels,4,1,f); fwrite(&mn,4,1,f);
    fwrite(&compact_na,4,1,f); fwrite(&cfg.det_nx,4,1,f); fwrite(&compact_dz,4,1,f);
    fwrite(&rpa,4,1,f); fwrite(&full_na,4,1,f);
    fwrite(&cfg.Nx,4,1,f); fwrite(&cfg.Ny,4,1,f); fwrite(&cfg.Nz,4,1,f);
    fwrite(&cfg.voxel_size,4,1,f); fwrite(&cfg.det_nx,4,1,f); fwrite(&cfg.det_nz,4,1,f);
    fwrite(&cfg.pixel_size,4,1,f); fwrite(&cfg.detector_y,4,1,f);
    fwrite(&cfg.detector_thickness,4,1,f); fwrite(&use90,4,1,f); fwrite(&zsym,4,1,f);

    /* v7.3.2 新增字段：明确写入 45° 对称信息。 */
    fwrite(&use45,4,1,f);
    fwrite(&sym_code,4,1,f);
    fwrite(&symmetry_order,4,1,f);
    fclose(f);

    /* 同步写一个文本版 metadata，方便人工检查和后处理脚本读取。 */
    std::string tf=cfg.output_dir+"/metadata.txt";
    FILE* t=fopen(tf.c_str(),"w");
    if (t) {
        fprintf(t,"version=v7.6_csrb_csr_adaptive_rect3_roi\n");
        fprintf(t,"chunk_format=csrb\n");
        fprintf(t,"total_rows=%d\n",total_rows);
        fprintf(t,"total_voxels=%d\n",total_voxels);
        fprintf(t,"compact_angles=%d\n",compact_na);
        fprintf(t,"full_angles=%d\n",full_na);
        fprintf(t,"det_nx=%d\n",cfg.det_nx);
        fprintf(t,"det_nz=%d\n",cfg.det_nz);
        fprintf(t,"compact_det_z=%d\n",compact_dz);
        fprintf(t,"rays_per_angle=%d\n",rpa);
        fprintf(t,"sym_mode=%s\n",cfg.sym_mode.c_str());
        fprintf(t,"sym_code=%d\n",sym_code);
        fprintf(t,"symmetry_order=%d\n",symmetry_order);
        fprintf(t,"use_45deg_symmetry=%d\n",use45);
        fprintf(t,"use_90deg_symmetry=%d\n",use90);
        fprintf(t,"use_z_symmetry=%d\n",zsym);
        fprintf(t,"roi_radius=%d\n",cfg.roi_radius);
        fprintf(t,"roi_side=%d\n",2*cfg.roi_radius+1);
        fprintf(t,"doi_solid_angle_integral=%d\n",cfg.phys.enable_doi_solid_angle_integral);
        fprintf(t,"collimator=rectangular_v2_PSC_three_section\n");
        fprintf(t,"output_mode=%s\n",cfg.output_mode.c_str());
        fprintf(t,"csr_agg_mode=%s\n",cfg.csr_agg_mode.c_str());
        fprintf(t,"dense_rpa_limit=%d\n",cfg.dense_rpa_limit);
        fprintf(t,"hash_size=%d\n",cfg.hash_size);
        fprintf(t,"edge_capacity_million=%d\n",cfg.edge_capacity_million);
        fprintf(t,"coo_capacity_million=%d\n",cfg.coo_capacity_million);
        fprintf(t,"coll_modules=%d\n",cfg.coll_modules);
        fprintf(t,"fast_pinhole_penetration=%d\n",cfg.fast_pinhole_penetration);
        fprintf(t,"note=CSRB direct output; csr mode uses source-local dense/hash aggregation + edge-to-CSR.\n");
        fclose(t);
    }
}


struct PinholeHost {
    std::vector<float> x, y, z;
    int total = 0;
};

[[maybe_unused]] static PinholeHost build_pinhole_centers_from_pm(
    const double pm[][4][3], int n_pin_per_module, int n_modules)
{
    /*
      使用 pm 面四顶点中心作为 ROI 投影针孔中心。
      注意：ROI 只做候选筛选；准直器穿透长度仍使用完整 pb-pm-pt polytope 精确计算。
    */
    PinholeHost P;
    P.total = n_pin_per_module * n_modules;
    P.x.resize(P.total);
    P.y.resize(P.total);
    P.z.resize(P.total);

    int idx = 0;
    for (int m = 0; m < n_modules; m++) {
        double phi = m * 2.0 * M_PI / (double)n_modules;
        double cp = cos(phi), sp = sin(phi);
        for (int p = 0; p < n_pin_per_module; p++) {
            double cx = 0.0, cy = 0.0, cz = 0.0;
            for (int v = 0; v < 4; v++) {
                cx += pm[p][v][0];
                cy += pm[p][v][1];
                cz += pm[p][v][2];
            }
            cx *= 0.25; cy *= 0.25; cz *= 0.25;

            P.x[idx] = (float)(cx * cp - cy * sp);
            P.y[idx] = (float)(cx * sp + cy * cp);
            P.z[idx] = (float)cz;
            idx++;
        }
    }
    return P;
}

static PinholeHost build_pinhole_centers_from_collimator(const CollimatorHost& coll)
{
    PinholeHost P;
    P.total = (int)coll.pin_x.size();
    P.x.resize(P.total);
    P.y.resize(P.total);
    P.z.resize(P.total);
    for (int i = 0; i < P.total; i++) {
        P.x[i] = (float)coll.pin_x[i];
        P.y[i] = (float)coll.pin_y[i];
        P.z[i] = (float)coll.pin_z[i];
    }
    return P;
}

/* ========================================================================
   GPU Worker
======================================================================== */
static void run_gpu_worker(
    int worker_id, int gpu_id, int num_workers,
    const RunConfig& cfg,
    const VoxelHost& vox, const DetectorHost& det, const CollimatorHost& coll,
    const PinholeHost& pins,
    const std::vector<double>& angles,
    int compact_na, int compact_dz, int rpa, int nchunks,
    WorkerStats& stats)
{
    auto t_start=std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaSetDevice(gpu_id));

    /* 上传基础数据 */
    float  *d_vx=nullptr,*d_vy=nullptr,*d_vz=nullptr;
    double *d_ang_cos=nullptr,*d_ang_sin=nullptr;
    double *d_pnx=nullptr,*d_pny=nullptr,*d_pnz=nullptr,*d_pd=nullptr;
    double *d_bcx=nullptr,*d_bcy=nullptr,*d_bcz=nullptr,*d_br2=nullptr;
    int    *d_tx_s=nullptr,*d_tx_c=nullptr,*d_tz_s=nullptr,*d_tz_c=nullptr;
    float  *d_tx_v=nullptr,*d_tz_v=nullptr;
#if COMPILE_ENABLE_DOI
    const bool doi_runtime_enabled = (cfg.phys.enable_DOI != 0);
#else
    const bool doi_runtime_enabled = false;
#endif

    gpu_up(&d_vx, vox.x); gpu_up(&d_vy, vox.y); gpu_up(&d_vz, vox.z);
    std::vector<double> angle_cos(angles.size()), angle_sin(angles.size());
    for (size_t i = 0; i < angles.size(); i++) {
        angle_cos[i] = cos(angles[i]);
        angle_sin[i] = sin(angles[i]);
    }
    gpu_up(&d_ang_cos, angle_cos);
    gpu_up(&d_ang_sin, angle_sin);
    gpu_up(&d_pnx,coll.planes_nx); gpu_up(&d_pny,coll.planes_ny);
    gpu_up(&d_pnz,coll.planes_nz); gpu_up(&d_pd, coll.planes_d);
    gpu_up(&d_bcx,coll.bc_x); gpu_up(&d_bcy,coll.bc_y);
    gpu_up(&d_bcz,coll.bc_z); gpu_up(&d_br2,coll.br2);

    /*
      准直器几何上传到 constant memory。kernel 热路径调用
      analytical_penetration_const()，从常量缓存读取多面体平面，避免每条
      子射线反复从 global memory 读取相同几何。旧 d_pnx/d_bcx 等仍保留为
      参数占位，便于最小化外层接口改动。
    */
    if (coll.n_poly > MAX_COLL_POLY || (int)coll.planes_nx.size() > MAX_COLL_PLANES) {
        fprintf(stderr, "Collimator too large for constant cache: n_poly=%d planes=%zu, limits poly=%d planes=%d\n",
                coll.n_poly, coll.planes_nx.size(), MAX_COLL_POLY, MAX_COLL_PLANES);
        exit(1);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_pnx_c, coll.planes_nx.data(), coll.planes_nx.size()*sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_pny_c, coll.planes_ny.data(), coll.planes_ny.size()*sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_pnz_c, coll.planes_nz.data(), coll.planes_nz.size()*sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_pd_c,  coll.planes_d .data(), coll.planes_d .size()*sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_bcx_c, coll.bc_x.data(),      coll.bc_x.size()*sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_bcy_c, coll.bc_y.data(),      coll.bc_y.size()*sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_bcz_c, coll.bc_z.data(),      coll.bc_z.size()*sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_br2_c, coll.br2 .data(),      coll.br2 .size()*sizeof(double)));

    if (!doi_runtime_enabled) {
        gpu_up(&d_tx_s,det.tx.start); gpu_up(&d_tx_c,det.tx.count);
        gpu_up(&d_tz_s,det.tz.start); gpu_up(&d_tz_c,det.tz.count);
    }
#if COMPILE_ENABLE_DOI
    const std::vector<float>& upload_tx_val = doi_runtime_enabled ? det.doi_x : det.tx.val;
    const std::vector<float>& upload_tz_val = doi_runtime_enabled ? det.doi_z : det.tz.val;
#else
    const std::vector<float>& upload_tx_val = det.tx.val;
    const std::vector<float>& upload_tz_val = det.tz.val;
#endif
    gpu_up(&d_tx_v,upload_tx_val);
    gpu_up(&d_tz_v,upload_tz_val);

    /* 设置 constant memory */
    DetectorConst dc; memset(&dc,0,sizeof(dc));
    for (int i=0;i<cfg.det_nx*2;i++) dc.x_edges_flat[i]=det.x_edges_flat[i];
    for (int i=0;i<cfg.det_nz*2;i++) dc.z_edges_flat[i]=det.z_edges_flat[i];
    for (int i=0;i<cfg.det_nx;i++) dc.x_centers[i]=det.x_centers[i];
    for (int i=0;i<cfg.det_nz;i++) dc.z_centers[i]=det.z_centers[i];
    dc.det_y=det.det_y; dc.back_y=det.back_y; dc.thickness=det.thickness;
    dc.nx=cfg.det_nx; dc.nz=cfg.det_nz; dc.compact_det_z=compact_dz;
    dc.doi_x0=det.doi_x0; dc.doi_z0=det.doi_z0;
    dc.doi_dx=det.doi_dx; dc.doi_dz=det.doi_dz;
    dc.doi_nx=(int)det.doi_x.size(); dc.doi_nz=(int)det.doi_z.size();
    CUDA_CHECK(cudaMemcpyToSymbol(c_det, &dc, sizeof(dc)));

    CUDA_CHECK(cudaMemcpyToSymbol(c_phys, &cfg.phys, sizeof(cfg.phys)));

    CollConst cc;
    cc.n_poly=coll.n_poly;
    cc.n_solid_poly=coll.n_solid_poly;
    cc.n_void_poly=coll.n_void_poly;
    cc.geom_kind=coll.geom_kind;
    cc.R_inner=coll.R_inner; cc.R_outer=coll.R_outer; cc.H_half=coll.H_half;
    cc.R_inner2=cc.R_inner*cc.R_inner; cc.R_outer2=cc.R_outer*cc.R_outer;
    CUDA_CHECK(cudaMemcpyToSymbol(c_coll, &cc, sizeof(cc)));

    SysConfig sc;
    sc.total_voxels=vox.total; sc.num_angles=cfg.num_angles;
    sc.compact_num_angles=compact_na; sc.full_det_x=cfg.det_nx;
    sc.compact_det_z=compact_dz; sc.compact_rays_per_angle=rpa;
    sc.Nx=cfg.Nx; sc.Ny=cfg.Ny; sc.Nz=cfg.Nz;
    sc.roi_radius=cfg.roi_radius; sc.roi_side=2*cfg.roi_radius+1;
    sc.roi_area=sc.roi_side*sc.roi_side; sc.n_pinholes=pins.total;
    sc.pinholes_per_module = (coll.n_solid_poly > 0) ? (pins.total / coll.n_solid_poly) : pins.total;
    sc.pinhole_sections = (pins.total > 0) ? (coll.n_void_poly / pins.total) : 0;
    sc.fast_pinhole_penetration = cfg.fast_pinhole_penetration;
    sc.enable_doi_sample_roi = cfg.enable_doi_sample_roi ? 1 : 0;
    float x_pitch = (cfg.det_nx > 0 && det.x_span > 0.0f) ? det.x_span / (float)cfg.det_nx : cfg.pixel_size;
    float z_pitch = (cfg.det_nz > 0 && det.z_span > 0.0f) ? det.z_span / (float)cfg.det_nz : cfg.pixel_size;
    float roi_half_x = ((float)cfg.roi_radius + 0.5f) * x_pitch;
    float roi_half_z = ((float)cfg.roi_radius + 0.5f) * z_pitch;
    sc.doi_roi_radius_x = (det.doi_dx > 0.0f) ? (int)ceilf(roi_half_x / det.doi_dx) : 0;
    sc.doi_roi_radius_z = (det.doi_dz > 0.0f) ? (int)ceilf(roi_half_z / det.doi_dz) : 0;
    if (sc.doi_roi_radius_x < 0) sc.doi_roi_radius_x = 0;
    if (sc.doi_roi_radius_z < 0) sc.doi_roi_radius_z = 0;
    sc.doi_roi_side_x = 2 * sc.doi_roi_radius_x + 1;
    sc.doi_roi_side_z = 2 * sc.doi_roi_radius_z + 1;
    long long full_sample_area = (long long)det.doi_x.size() * (long long)det.doi_z.size();
    long long roi_sample_area = (long long)sc.doi_roi_side_x * (long long)sc.doi_roi_side_z;
    long long doi_area = sc.enable_doi_sample_roi ? roi_sample_area : full_sample_area;
    if (doi_area > INT_MAX) {
        fprintf(stderr, "DOI sample task area too large: %lld > INT_MAX. Reduce --doi-radial-samples or enable ROI.\n", doi_area);
        exit(1);
    }
    sc.doi_task_area = (int)doi_area;
    CUDA_CHECK(cudaMemcpyToSymbol(c_cfg, &sc, sizeof(sc)));
    int doi_task_area_host = sc.doi_task_area;

    if (pins.total > MAX_PINHOLES_TOTAL) {
        fprintf(stderr, "Too many pinholes: %d > MAX_PINHOLES_TOTAL=%d\n", pins.total, MAX_PINHOLES_TOTAL);
        exit(1);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_pin_x, pins.x.data(), pins.total*sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_pin_y, pins.y.data(), pins.total*sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_pin_z, pins.z.data(), pins.total*sizeof(float)));

    double vso_x[N_SUB_VOX], vso_y[N_SUB_VOX], vso_z[N_SUB_VOX];
    int kk=0;
    for (int sz=1;sz<=S_VOX;sz++) for (int sy=1;sy<=S_VOX;sy++) for (int sx=1;sx<=S_VOX;sx++) {
        vso_x[kk]=((2.0*sx-S_VOX-1)/(2.0*S_VOX))*cfg.voxel_size;
        vso_y[kk]=((2.0*sy-S_VOX-1)/(2.0*S_VOX))*cfg.voxel_size;
        vso_z[kk]=((2.0*sz-S_VOX-1)/(2.0*S_VOX))*cfg.voxel_size;
        kk++;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_vso_x,vso_x,sizeof(double)*N_SUB_VOX));
    CUDA_CHECK(cudaMemcpyToSymbol(c_vso_y,vso_y,sizeof(double)*N_SUB_VOX));
    CUDA_CHECK(cudaMemcpyToSymbol(c_vso_z,vso_z,sizeof(double)*N_SUB_VOX));

    unsigned int z32=0; unsigned long long z64=0;
    CUDA_CHECK(cudaMemcpyToSymbol(g_stat_nnz_overflow,    &z32,sizeof(z32)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_stat_hash_overflow,   &z32,sizeof(z32)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_stat_precheck_skipped,&z64,sizeof(z64)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_stat_precheck_passed, &z64,sizeof(z64)));

    bool csr_mode = (cfg.output_mode == "csr" || cfg.output_mode == "csr-adaptive" ||
                     cfg.output_mode == "csr-dense" || cfg.output_mode == "csr-hash" ||
                     cfg.output_mode == "edgecsr");

    if (csr_mode) {
        /*
          CSR-EDGE 模式：
            一个 block 负责一个 source=(local_angle, voxel)。
            block 内用 dense accumulator 或 shared hash 聚合同一 source 下所有 DOI/子射线贡献，
            每个 source-detector 只输出一条 edge(row,col,val)，再在 Host 侧按 row_counts 直接写 CSRB。

          与 v2d raw COO 相比：
            - 不保存 raw DOI/subray 级 V2D entry；
            - 不分配 reduced key/value 双缓冲；
            - 不调用 thrust sort/reduce；
            - 只需要 edge_cap 容纳 unique source-det 级 edge。
        */
        unsigned long long edge_cap = (unsigned long long)cfg.edge_capacity_million * 1000000ULL;
        if (edge_cap == 0ULL) edge_cap = (unsigned long long)cfg.coo_capacity_million * 1000000ULL;

        std::string agg = cfg.csr_agg_mode;
        if (cfg.output_mode == "csr-dense") agg = "dense";
        if (cfg.output_mode == "csr-hash")  agg = "hash";
        if (cfg.output_mode == "csr-adaptive" || cfg.output_mode == "csr" || cfg.output_mode == "edgecsr") {
            if (agg != "dense" && agg != "hash") agg = (rpa <= cfg.dense_rpa_limit) ? "dense" : "hash";
        }
        int agg_mode = (agg == "dense") ? 0 : 1;

        int hash_size = cfg.hash_size;
        if (hash_size < 64) hash_size = 64;
        /* hash_size 必须为 2 的幂，便于用 & 取模。 */
        if ((hash_size & (hash_size - 1)) != 0) {
            int hs = 1; while (hs < hash_size) hs <<= 1; hash_size = hs;
        }

        size_t dense_shmem = (size_t)rpa * sizeof(float) + (size_t)((rpa + 31) >> 5) * sizeof(unsigned int);
        size_t hash_shmem  = (size_t)hash_size * (sizeof(unsigned int) + sizeof(float)) + sizeof(int);
        size_t shmem_bytes = (agg_mode == 0) ? dense_shmem : hash_shmem;

        /* 如果使用较大的 dynamic shared memory，尝试放宽 kernel 限制；老架构失败不致命。 */
        cudaFuncSetAttribute(kernel_source_roi_csr_edges, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_bytes);

        int max_ac = std::min(cfg.chunk_angles, compact_na);
        int max_rt = max_ac * rpa;

        unsigned int *d_edge_rows=nullptr, *d_edge_cols=nullptr;
        float *d_edge_vals=nullptr;
        unsigned long long *d_edge_nnz=nullptr, *d_row_counts=nullptr;
        CUDA_CHECK(cudaMalloc(&d_edge_rows, edge_cap * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_edge_cols, edge_cap * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_edge_vals, edge_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_edge_nnz, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_row_counts, (size_t)max_rt * sizeof(unsigned long long)));

        {
            std::lock_guard<std::mutex> lock(g_print_mutex);
            printf("[GPU%d/W%d] CSR-EDGE mode: agg=%s, RPA=%d, dense_limit=%d, hash_size=%d, shmem=%.1f KB, edge_cap=%dM, edge_buffer≈%.2f GiB\n",
                   gpu_id, worker_id, agg.c_str(), rpa, cfg.dense_rpa_limit, hash_size,
                   (double)shmem_bytes/1024.0, cfg.edge_capacity_million,
                   (double)edge_cap * (sizeof(unsigned int)*2 + sizeof(float)) / (1024.0*1024.0*1024.0));
            fflush(stdout);
        }

        unsigned int hash_prev = 0;
        CUDA_CHECK(cudaMemcpyFromSymbol(&hash_prev, g_stat_hash_overflow, sizeof(hash_prev)));

        for (int ch=0; ch<nchunks; ch++) {
            if ((ch % num_workers) != worker_id) continue;
            int as=ch*cfg.chunk_angles;
            int ac=std::min(cfg.chunk_angles, compact_na-as);
            int rt=ac*rpa;

            auto t0=std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaMemset(d_edge_nnz, 0, sizeof(unsigned long long)));
            CUDA_CHECK(cudaMemset(d_row_counts, 0, (size_t)rt * sizeof(unsigned long long)));

            unsigned long long n_sources = (unsigned long long)vox.total * (unsigned long long)ac;
            dim3 grid((unsigned int)n_sources, 1, 1);

            kernel_source_roi_csr_edges<<<grid, BLOCK_SIZE_VA, shmem_bytes>>>(
                d_vx,d_vy,d_vz,
                d_tx_s,d_tx_c,d_tx_v,
                d_tz_s,d_tz_c,d_tz_v,
                d_ang_cos,d_ang_sin,
                d_pnx,d_pny,d_pnz,d_pd,
                d_bcx,d_bcy,d_bcz,d_br2,
                as,ac, agg_mode, hash_size,
                d_edge_rows,d_edge_cols,d_edge_vals,d_edge_nnz,edge_cap,d_row_counts);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            unsigned long long raw_edges = 0;
            CUDA_CHECK(cudaMemcpy(&raw_edges, d_edge_nnz, sizeof(raw_edges), cudaMemcpyDeviceToHost));
            unsigned long long used_edges = std::min(raw_edges, edge_cap);

            std::vector<unsigned long long> h_row_counts((size_t)rt);
            CUDA_CHECK(cudaMemcpy(h_row_counts.data(), d_row_counts, (size_t)rt*sizeof(unsigned long long), cudaMemcpyDeviceToHost));

            std::vector<unsigned int> h_rows((size_t)used_edges);
            std::vector<unsigned int> h_cols((size_t)used_edges);
            std::vector<float> h_vals((size_t)used_edges);
            if (used_edges > 0) {
                CUDA_CHECK(cudaMemcpy(h_rows.data(), d_edge_rows, (size_t)used_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_cols.data(), d_edge_cols, (size_t)used_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_vals.data(), d_edge_vals, (size_t)used_edges*sizeof(float), cudaMemcpyDeviceToHost));
            }

            unsigned int hash_now = 0;
            CUDA_CHECK(cudaMemcpyFromSymbol(&hash_now, g_stat_hash_overflow, sizeof(hash_now)));
            unsigned int hash_delta = hash_now - hash_prev;
            hash_prev = hash_now;

            float dt_s=std::chrono::duration<float>(
                std::chrono::high_resolution_clock::now()-t0).count();

            char fn[1024];
            snprintf(fn,sizeof(fn),"%s/chunk_%03d.csrb",cfg.output_dir.c_str(),ch);
            save_chunk_from_edges(fn,h_rows,h_cols,h_vals,h_row_counts,vox.total,rt,as,ac,cfg.phys.weight_threshold);

            {
                std::lock_guard<std::mutex> lock(g_print_mutex);
                printf("[GPU%d/W%d] Chunk %03d/%03d  angles %d-%d  %.2fs  CSR_EDGE=%.2fM%s%s\n",
                       gpu_id,worker_id,ch,nchunks-1,as,as+ac-1,dt_s,
                       (double)used_edges/1e6,
                       (raw_edges>edge_cap)?"  *** EDGE CAP OVERFLOW: increase --edge-cap-m or reduce --chunk-angles ***":"",
                       (hash_delta>0)?"  *** HASH OVERFLOW: increase --hash-size or use dense/tile ***":"");
                fflush(stdout);
            }
        }

        cudaFree(d_edge_rows); cudaFree(d_edge_cols); cudaFree(d_edge_vals);
        cudaFree(d_edge_nnz); cudaFree(d_row_counts);

    } else if (cfg.output_mode == "v2d") {
        /*
          V2D 模式：推荐用于 DOI。
          kernel 输出 key=(source=(angle,voxel), detector)，先按 source-det reduce，
          再用 histogram+scatter 转置为旧 detector-row chunk 格式。
        */
        unsigned long long v2d_cap = (unsigned long long)cfg.coo_capacity_million * 1000000ULL;
        unsigned long long *d_keys = nullptr, *d_keys_red = nullptr, *d_raw_nnz = nullptr;
        float *d_vals = nullptr, *d_vals_red = nullptr;
        CUDA_CHECK(cudaMalloc(&d_keys,     v2d_cap * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_vals,     v2d_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_keys_red, v2d_cap * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_vals_red, v2d_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_raw_nnz,  sizeof(unsigned long long)));

        for (int ch=0; ch<nchunks; ch++) {
            if ((ch % num_workers) != worker_id) continue;
            int as=ch*cfg.chunk_angles;
            int ac=std::min(cfg.chunk_angles, compact_na-as);
            int rt=ac*rpa;

            auto t0=std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaMemset(d_raw_nnz,0,sizeof(unsigned long long)));

            long long roi_side = 2LL*cfg.roi_radius + 1LL;
            long long sample_area = doi_runtime_enabled ? (long long)doi_task_area_host : roi_side * roi_side;
            long long pin_factor = doi_runtime_enabled
                ? (cfg.enable_doi_sample_roi ? (long long)pins.total : 1LL)
                : (long long)pins.total;
            long long nthreads=(long long)vox.total * ac * pin_factor * sample_area;
            int nb=(int)((nthreads+BLOCK_SIZE_VA-1)/BLOCK_SIZE_VA);

#if COMPILE_ENABLE_DOI
            if (doi_runtime_enabled) {
                kernel_subray_uniform_doi_v2d<<<nb, BLOCK_SIZE_VA>>>(
                    d_vx,d_vy,d_vz,
                    d_tx_v,d_tz_v,
                    d_ang_cos,d_ang_sin,
                    d_pnx,d_pny,d_pnz,d_pd,
                    d_bcx,d_bcy,d_bcz,d_br2,
                    as,ac, d_keys,d_vals,d_raw_nnz, v2d_cap);
            } else
#endif
            {
                kernel_subray_roi_v2d<<<nb, BLOCK_SIZE_VA>>>(
                    d_vx,d_vy,d_vz,
                    d_tx_s,d_tx_c,d_tx_v,
                    d_tz_s,d_tz_c,d_tz_v,
                    d_ang_cos,d_ang_sin,
                    d_pnx,d_pny,d_pnz,d_pd,
                    d_bcx,d_bcy,d_bcz,d_br2,
                    as,ac, d_keys,d_vals,d_raw_nnz, v2d_cap);
            }

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            unsigned long long raw_nnz = 0;
            CUDA_CHECK(cudaMemcpy(&raw_nnz, d_raw_nnz, sizeof(raw_nnz), cudaMemcpyDeviceToHost));
            unsigned long long used_nnz = std::min(raw_nnz, v2d_cap);

            size_t reduced_nnz = 0;
            if (used_nnz > 0) {
                thrust::device_ptr<unsigned long long> k_begin(d_keys);
                thrust::device_ptr<unsigned long long> k_end(d_keys + used_nnz);
                thrust::device_ptr<float> v_begin(d_vals);

                thrust::sort_by_key(thrust::device, k_begin, k_end, v_begin);

                thrust::device_ptr<unsigned long long> kr_begin(d_keys_red);
                thrust::device_ptr<float> vr_begin(d_vals_red);
                auto new_end = thrust::reduce_by_key(thrust::device, k_begin, k_end, v_begin, kr_begin, vr_begin);
                reduced_nnz = (size_t)(new_end.first - kr_begin);
            }

            std::vector<unsigned long long> h_keys(reduced_nnz);
            std::vector<float> h_vals(reduced_nnz);
            if (reduced_nnz > 0) {
                CUDA_CHECK(cudaMemcpy(h_keys.data(), d_keys_red, reduced_nnz*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_vals.data(), d_vals_red, reduced_nnz*sizeof(float), cudaMemcpyDeviceToHost));
            }

            float dt_s=std::chrono::duration<float>(
                std::chrono::high_resolution_clock::now()-t0).count();

            char fn[1024];
            snprintf(fn,sizeof(fn),"%s/chunk_%03d.csrb",cfg.output_dir.c_str(),ch);
            save_chunk_from_v2d(fn,h_keys,h_vals,vox.total,rpa,as,ac,rt,cfg.phys.weight_threshold);

            {
                std::lock_guard<std::mutex> lock(g_print_mutex);
                printf("[GPU%d/W%d] Chunk %03d/%03d  angles %d-%d  %.2fs  rawV2D=%.2fM  reducedV2D=%.2fM%s\n",
                       gpu_id,worker_id,ch,nchunks-1,as,as+ac-1,dt_s,
                       (double)used_nnz/1e6, (double)reduced_nnz/1e6,
                       (raw_nnz>v2d_cap)?"  *** V2D CAP OVERFLOW: increase --coo-cap-m or reduce --chunk-angles ***":"");
                fflush(stdout);
            }
        }

        cudaFree(d_keys); cudaFree(d_vals); cudaFree(d_keys_red); cudaFree(d_vals_red); cudaFree(d_raw_nnz);
    } else if (cfg.output_mode == "direct") {
        /*
          DIRECT 模式：
            - 每个 worker/GPU 只分配一次最大 chunk 的定长 row buffer；
            - 每个 chunk 只清零 nnz_out，不再 memset 巨大的 vox_out/wgt_out；
            - 计算完成后按旧 chunk 格式写出；
            - 省去 COO 原始条目写入、thrust sort_by_key 与 reduce_by_key。

          显存占用估计：
            max_rt * MAX_NNZ_PER_ROW * (sizeof(int)+sizeof(float)) + max_rt*sizeof(int)
          其中 max_rt = chunk_angles * rays_per_angle。该模式要求较大显存，
          但符合“把数据直接放在 GPU 中，用显存换速度”的需求。
        */
        int max_ac = std::min(cfg.chunk_angles, compact_na);
        long long max_rt = (long long)max_ac * rpa;
        long long max_buf = max_rt * (long long)MAX_NNZ_PER_ROW;
        double direct_gb = (double)max_buf * (sizeof(int)+sizeof(float)) / (1024.0*1024.0*1024.0);
        {
            std::lock_guard<std::mutex> lock(g_print_mutex);
            printf("[GPU%d/W%d] DIRECT mode: max rows=%lld, MAX_NNZ_PER_ROW=%d, row buffer≈%.2f GiB\n",
                   gpu_id,worker_id,max_rt,MAX_NNZ_PER_ROW,direct_gb);
            fflush(stdout);
        }

        int *d_vo=nullptr, *d_no=nullptr;
        float *d_wo=nullptr;
        CUDA_CHECK(cudaMalloc(&d_vo, max_buf*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_wo, max_buf*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_no, max_rt*sizeof(int)));

        /* Host 侧缓冲区也复用，避免每个 chunk 反复分配大数组。 */
        std::vector<int>   h_vo(max_buf);
        std::vector<float> h_wo(max_buf);
        std::vector<int>   h_no(max_rt);

        for (int ch=0; ch<nchunks; ch++) {
            if ((ch % num_workers) != worker_id) continue;
            int as=ch*cfg.chunk_angles;
            int ac=std::min(cfg.chunk_angles, compact_na-as);
            int rt=ac*rpa;
            long long buf=(long long)rt*MAX_NNZ_PER_ROW;

            auto t0=std::chrono::high_resolution_clock::now();

            /*
              只清零每行 nnz 计数。vox_out/wgt_out 中旧数据会被新 slot 覆盖；
              保存时只读取 [0, nnz[row])，所以无需清零整个巨大 row buffer。
            */
            CUDA_CHECK(cudaMemset(d_no,0,rt*sizeof(int)));

            long long roi_side = 2LL*cfg.roi_radius + 1LL;
            long long sample_area = doi_runtime_enabled ? (long long)doi_task_area_host : roi_side * roi_side;
            long long pin_factor = doi_runtime_enabled
                ? (cfg.enable_doi_sample_roi ? (long long)pins.total : 1LL)
                : (long long)pins.total;
            long long nthreads=(long long)vox.total * ac * pin_factor * sample_area;
            int nb=(int)((nthreads+BLOCK_SIZE_VA-1)/BLOCK_SIZE_VA);

#if COMPILE_ENABLE_DOI
            if (doi_runtime_enabled) {
                kernel_subray_uniform_doi_direct<<<nb, BLOCK_SIZE_VA>>>(
                    d_vx,d_vy,d_vz,
                    d_tx_v,d_tz_v,
                    d_ang_cos,d_ang_sin,
                    d_pnx,d_pny,d_pnz,d_pd,
                    d_bcx,d_bcy,d_bcz,d_br2,
                    as,ac, d_vo,d_wo,d_no);
            } else
#endif
            {
                kernel_subray_roi_direct<<<nb, BLOCK_SIZE_VA>>>(
                    d_vx,d_vy,d_vz,
                    d_tx_s,d_tx_c,d_tx_v,
                    d_tz_s,d_tz_c,d_tz_v,
                    d_ang_cos,d_ang_sin,
                    d_pnx,d_pny,d_pnz,d_pd,
                    d_bcx,d_bcy,d_bcz,d_br2,
                    as,ac, d_vo,d_wo,d_no);
            }

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(h_no.data(),d_no,rt*sizeof(int),cudaMemcpyDeviceToHost));
            long long ns=0;
            for (int r=0;r<rt;r++) {
                if (h_no[r] > MAX_NNZ_PER_ROW) h_no[r]=MAX_NNZ_PER_ROW;
                if (h_no[r] < 0) h_no[r]=0;
                ns += h_no[r];
            }

            if (ns > 0) {
                CUDA_CHECK(cudaMemcpy(h_vo.data(),d_vo,buf*sizeof(int),cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_wo.data(),d_wo,buf*sizeof(float),cudaMemcpyDeviceToHost));
            }

            float dt_s=std::chrono::duration<float>(
                std::chrono::high_resolution_clock::now()-t0).count();

            char fn[1024];
            snprintf(fn,sizeof(fn),"%s/chunk_%03d.csrb",cfg.output_dir.c_str(),ch);
            save_chunk_direct(fn,h_vo.data(),h_wo.data(),h_no.data(),as,ac,rt);

            {
                std::lock_guard<std::mutex> lock(g_print_mutex);
                printf("[GPU%d/W%d] Chunk %03d/%03d  angles %d-%d  %.2fs  DIRECT_NNZ=%.2fM\n",
                       gpu_id,worker_id,ch,nchunks-1,as,as+ac-1,dt_s,(double)ns/1e6);
                fflush(stdout);
            }
        }

        cudaFree(d_vo); cudaFree(d_wo); cudaFree(d_no);
    } else {
        /*
          COO 模式保留为可选验证/回退路径。它使用较小、可控的 COO 缓冲区，
          但每个 chunk 需要 sort/reduce，通常比 direct 模式慢。
        */
        unsigned long long coo_cap = (unsigned long long)cfg.coo_capacity_million * 1000000ULL;
        unsigned long long *d_keys = nullptr, *d_keys_red = nullptr, *d_raw_nnz = nullptr;
        float *d_vals = nullptr, *d_vals_red = nullptr;
        CUDA_CHECK(cudaMalloc(&d_keys,     coo_cap * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_vals,     coo_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_keys_red, coo_cap * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_vals_red, coo_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_raw_nnz,  sizeof(unsigned long long)));

        for (int ch=0; ch<nchunks; ch++) {
            if ((ch % num_workers) != worker_id) continue;
            int as=ch*cfg.chunk_angles;
            int ac=std::min(cfg.chunk_angles, compact_na-as);
            int rt=ac*rpa;

            auto t0=std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaMemset(d_raw_nnz,0,sizeof(unsigned long long)));

            long long roi_side = 2LL*cfg.roi_radius + 1LL;
            long long sample_area = doi_runtime_enabled ? (long long)doi_task_area_host : roi_side * roi_side;
            long long pin_factor = doi_runtime_enabled
                ? (cfg.enable_doi_sample_roi ? (long long)pins.total : 1LL)
                : (long long)pins.total;
            long long nthreads=(long long)vox.total * ac * pin_factor * sample_area;
            int nb=(int)((nthreads+BLOCK_SIZE_VA-1)/BLOCK_SIZE_VA);

#if COMPILE_ENABLE_DOI
            if (doi_runtime_enabled) {
                kernel_subray_uniform_doi_coo<<<nb, BLOCK_SIZE_VA>>>(
                    d_vx,d_vy,d_vz,
                    d_tx_v,d_tz_v,
                    d_ang_cos,d_ang_sin,
                    d_pnx,d_pny,d_pnz,d_pd,
                    d_bcx,d_bcy,d_bcz,d_br2,
                    as,ac, d_keys,d_vals,d_raw_nnz, coo_cap);
            } else
#endif
            {
                kernel_subray_roi_coo<<<nb, BLOCK_SIZE_VA>>>(
                    d_vx,d_vy,d_vz,
                    d_tx_s,d_tx_c,d_tx_v,
                    d_tz_s,d_tz_c,d_tz_v,
                    d_ang_cos,d_ang_sin,
                    d_pnx,d_pny,d_pnz,d_pd,
                    d_bcx,d_bcy,d_bcz,d_br2,
                    as,ac, d_keys,d_vals,d_raw_nnz, coo_cap);
            }

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            unsigned long long raw_nnz = 0;
            CUDA_CHECK(cudaMemcpy(&raw_nnz, d_raw_nnz, sizeof(raw_nnz), cudaMemcpyDeviceToHost));
            unsigned long long used_nnz = std::min(raw_nnz, coo_cap);

            size_t reduced_nnz = 0;
            if (used_nnz > 0) {
                thrust::device_ptr<unsigned long long> k_begin(d_keys);
                thrust::device_ptr<unsigned long long> k_end(d_keys + used_nnz);
                thrust::device_ptr<float> v_begin(d_vals);

                thrust::sort_by_key(thrust::device, k_begin, k_end, v_begin);

                thrust::device_ptr<unsigned long long> kr_begin(d_keys_red);
                thrust::device_ptr<float> vr_begin(d_vals_red);
                auto new_end = thrust::reduce_by_key(thrust::device, k_begin, k_end, v_begin, kr_begin, vr_begin);
                reduced_nnz = (size_t)(new_end.first - kr_begin);
            }

            std::vector<unsigned long long> h_keys(reduced_nnz);
            std::vector<float> h_vals(reduced_nnz);
            if (reduced_nnz > 0) {
                CUDA_CHECK(cudaMemcpy(h_keys.data(), d_keys_red, reduced_nnz*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_vals.data(), d_vals_red, reduced_nnz*sizeof(float), cudaMemcpyDeviceToHost));
            }

            float dt_s=std::chrono::duration<float>(
                std::chrono::high_resolution_clock::now()-t0).count();

            char fn[1024];
            snprintf(fn,sizeof(fn),"%s/chunk_%03d.csrb",cfg.output_dir.c_str(),ch);
            save_chunk_from_coo(fn,h_keys,h_vals,vox.total,as,ac,rt,cfg.phys.weight_threshold);

            {
                std::lock_guard<std::mutex> lock(g_print_mutex);
                printf("[GPU%d/W%d] Chunk %03d/%03d  angles %d-%d  %.2fs  rawCOO=%.2fM  reducedNNZ=%.2fM%s\n",
                       gpu_id,worker_id,ch,nchunks-1,as,as+ac-1,dt_s,
                       (double)used_nnz/1e6, (double)reduced_nnz/1e6,
                       (raw_nnz>coo_cap)?"  *** COO CAP OVERFLOW: increase --coo-cap-m ***":"");
                fflush(stdout);
            }
        }

        cudaFree(d_keys); cudaFree(d_vals); cudaFree(d_keys_red); cudaFree(d_vals_red); cudaFree(d_raw_nnz);
    }

    CUDA_CHECK(cudaMemcpyFromSymbol(&stats.nnz_overflow,
        g_stat_nnz_overflow,sizeof(stats.nnz_overflow)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&stats.hash_overflow,
        g_stat_hash_overflow,sizeof(stats.hash_overflow)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&stats.precheck_skipped,
        g_stat_precheck_skipped,sizeof(stats.precheck_skipped)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&stats.precheck_passed,
        g_stat_precheck_passed,sizeof(stats.precheck_passed)));

    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_ang_cos); cudaFree(d_ang_sin);
    cudaFree(d_pnx); cudaFree(d_pny); cudaFree(d_pnz); cudaFree(d_pd);
    cudaFree(d_bcx); cudaFree(d_bcy); cudaFree(d_bcz); cudaFree(d_br2);
    cudaFree(d_tx_s); cudaFree(d_tx_c); cudaFree(d_tx_v);
    cudaFree(d_tz_s); cudaFree(d_tz_c); cudaFree(d_tz_v);
    CUDA_CHECK(cudaDeviceSynchronize());

    stats.seconds=std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now()-t_start).count();
    {
        std::lock_guard<std::mutex> lock(g_print_mutex);
        printf("[Worker %d / GPU %d] finished in %.2f s\n",worker_id,gpu_id,stats.seconds);
    }
}

/* ========================================================================
   main
======================================================================== */
int main(int argc, char** argv)
{
    RunConfig cfg;
    memset(&cfg.phys, 0, sizeof(cfg.phys));

    cfg.phys.mu_collimator                   = 3.202f;
    cfg.phys.center_lor_mp_threshold         = 5.0;
    cfg.phys.mu_detector                     = 0.1368f;
    cfg.phys.penetration_length_threshold    = 1e-3f;
    cfg.phys.penetration_probability_threshold = 0.01f;
    cfg.phys.weight_threshold                = 1e-12f;
    cfg.phys.solid_angle_power               = 1;
    cfg.phys.enable_penetration              = 1;
    cfg.phys.enable_DOI                      = 1;
    cfg.phys.enable_center_lor_precheck      = 1;
    cfg.phys.enable_doi_solid_angle_integral = 1;
    cfg.phys.pixel_size                      = cfg.pixel_size;
    cfg.phys.voxel_size                      = cfg.voxel_size;

    for (int ai=1; ai<argc; ai++) {
        if (!strcmp(argv[ai],"--num-gpus")&&ai+1<argc)     cfg.num_gpus=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--gpu-list")&&ai+1<argc) cfg.gpu_list=parse_gpu_list(argv[++ai]);
        else if (!strcmp(argv[ai],"--nx")&&ai+1<argc) cfg.Nx=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--ny")&&ai+1<argc) cfg.Ny=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--nz")&&ai+1<argc) cfg.Nz=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--det-nx")&&ai+1<argc) cfg.det_nx=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--det-nz")&&ai+1<argc) cfg.det_nz=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--num-angles")&&ai+1<argc) cfg.num_angles=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--voxel-size")&&ai+1<argc) cfg.voxel_size=(float)atof(argv[++ai]);
        else if (!strcmp(argv[ai],"--pixel-size")&&ai+1<argc) cfg.pixel_size=(float)atof(argv[++ai]);
        else if (!strcmp(argv[ai],"--gap-size")&&ai+1<argc) cfg.gap_size=(float)atof(argv[++ai]);
        else if (!strcmp(argv[ai],"--det-y")&&ai+1<argc) cfg.detector_y=(float)atof(argv[++ai]);
        else if (!strcmp(argv[ai],"--det-thickness")&&ai+1<argc) cfg.detector_thickness=(float)atof(argv[++ai]);
        else if (!strcmp(argv[ai],"--sym")&&ai+1<argc)      cfg.sym_mode=argv[++ai];
        else if (!strcmp(argv[ai],"--zsym")&&ai+1<argc)     cfg.use_zsym=atoi(argv[++ai])!=0;
        else if (!strcmp(argv[ai],"--chunk-angles")&&ai+1<argc) cfg.chunk_angles=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--out")&&ai+1<argc)      cfg.output_dir=argv[++ai];
        else if (!strcmp(argv[ai],"--clean")&&ai+1<argc)    cfg.clean_output=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--precheck")&&ai+1<argc) cfg.phys.enable_center_lor_precheck=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--center-threshold")&&ai+1<argc) cfg.phys.center_lor_mp_threshold=atof(argv[++ai]);
        else if (!strcmp(argv[ai],"--penetration")&&ai+1<argc) cfg.phys.enable_penetration=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--weight-threshold")&&ai+1<argc) cfg.phys.weight_threshold=(float)atof(argv[++ai]);
        else if (!strcmp(argv[ai],"--roi-radius")&&ai+1<argc) cfg.roi_radius=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--matrix-output")&&ai+1<argc) cfg.output_mode=argv[++ai];
        else if (!strcmp(argv[ai],"--output-mode")&&ai+1<argc) cfg.output_mode=argv[++ai];
        else if (!strcmp(argv[ai],"--use-coo")&&ai+1<argc) cfg.output_mode=(atoi(argv[++ai])?"coo":"direct");
        else if (!strcmp(argv[ai],"--coo-cap-m")&&ai+1<argc) cfg.coo_capacity_million=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--edge-cap-m")&&ai+1<argc) cfg.edge_capacity_million=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--csr-agg")&&ai+1<argc) cfg.csr_agg_mode=argv[++ai];
        else if (!strcmp(argv[ai],"--dense-rpa-limit")&&ai+1<argc) cfg.dense_rpa_limit=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--hash-size")&&ai+1<argc) cfg.hash_size=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--doi")&&ai+1<argc) cfg.phys.enable_DOI=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--doi-solid-angle-integral")&&ai+1<argc) cfg.phys.enable_doi_solid_angle_integral=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--doi-line-integral")&&ai+1<argc) cfg.phys.enable_doi_solid_angle_integral=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--doi-radial-samples")&&ai+1<argc) cfg.doi_radial_samples=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--doi-samples")&&ai+1<argc) cfg.doi_radial_samples=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--doi-roi")&&ai+1<argc) cfg.enable_doi_sample_roi=atoi(argv[++ai])!=0;
        else if (!strcmp(argv[ai],"--sample-roi")&&ai+1<argc) cfg.enable_doi_sample_roi=atoi(argv[++ai])!=0;
        else if (!strcmp(argv[ai],"--coll-modules")&&ai+1<argc) cfg.coll_modules=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--fast-pin-penetration")&&ai+1<argc) cfg.fast_pinhole_penetration=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--coll-box-width-x")&&ai+1<argc) cfg.coll_box_width_x=(float)atof(argv[++ai]);
        else if (!strcmp(argv[ai],"--coll-box-length-z")&&ai+1<argc) cfg.coll_box_length_z=(float)atof(argv[++ai]);
        else if (!strcmp(argv[ai],"--coll-box-y-min")&&ai+1<argc) cfg.coll_box_y_min=(float)atof(argv[++ai]);
        else if (!strcmp(argv[ai],"--coll-box-y-max")&&ai+1<argc) cfg.coll_box_y_max=(float)atof(argv[++ai]);
        else if (!strcmp(argv[ai],"--coll-box-height-y")&&ai+1<argc) cfg.coll_box_y_max=cfg.coll_box_y_min+(float)atof(argv[++ai]);
        else if (!strcmp(argv[ai],"--voxel-rotation")&&ai+1<argc) cfg.voxel_rotation_sign=parse_voxel_rotation_sign(argv[++ai]);
        else if (!strcmp(argv[ai],"--voxel-rotation-sign")&&ai+1<argc) cfg.voxel_rotation_sign=parse_voxel_rotation_sign(argv[++ai]);
        else { fprintf(stderr,"Unknown arg: %s\n",argv[ai]); return 1; }
    }

    cfg.phys.pixel_size   = cfg.pixel_size;
    cfg.phys.voxel_size   = cfg.voxel_size;
    if (cfg.coll_modules < 1) cfg.coll_modules = 1;
    if (cfg.roi_radius < 0) cfg.roi_radius = 0;
    if (cfg.coo_capacity_million < 1) cfg.coo_capacity_million = 1;
    if (cfg.edge_capacity_million < 1) cfg.edge_capacity_million = cfg.coo_capacity_million;
    if (cfg.doi_radial_samples < 1) cfg.doi_radial_samples = DOI_RADIAL_SAMPLES_DEFAULT;
    if (cfg.dense_rpa_limit < 1) cfg.dense_rpa_limit = 2048;
    if (cfg.hash_size < 64) cfg.hash_size = 64;
    if ((cfg.hash_size & (cfg.hash_size - 1)) != 0) {
        int hs = 1; while (hs < cfg.hash_size) hs <<= 1; cfg.hash_size = hs;
    }
    bool valid_output = (cfg.output_mode == "csr" || cfg.output_mode == "csr-adaptive" ||
                         cfg.output_mode == "csr-dense" || cfg.output_mode == "csr-hash" ||
                         cfg.output_mode == "edgecsr" ||
                         cfg.output_mode == "v2d" || cfg.output_mode == "direct" || cfg.output_mode == "coo");
    if (!valid_output) {
        fprintf(stderr,"Invalid --matrix-output %s, expected csr, csr-dense, csr-hash, v2d, direct or coo.\n", cfg.output_mode.c_str());
        return 1;
    }
    if (cfg.csr_agg_mode != "adaptive" && cfg.csr_agg_mode != "dense" && cfg.csr_agg_mode != "hash") {
        fprintf(stderr,"Invalid --csr-agg %s, expected adaptive, dense or hash.\n", cfg.csr_agg_mode.c_str());
        return 1;
    }

    if (cfg.phys.enable_penetration &&
        cfg.phys.penetration_probability_threshold > 0.0f &&
        cfg.phys.mu_collimator > 0.0) {
        cfg.phys.coll_exp_cutoff_x =
            -log((double)cfg.phys.penetration_probability_threshold);
        cfg.phys.coll_exp_cutoff_len =
            cfg.phys.coll_exp_cutoff_x / cfg.phys.mu_collimator;
    } else {
        cfg.phys.coll_exp_cutoff_x   = 1.0e300;
        cfg.phys.coll_exp_cutoff_len = 1.0e300;
    }

    int dev_count=0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count<=0) { fprintf(stderr,"No CUDA device.\n"); return 1; }

    if (cfg.gpu_list.empty()) {
        if (cfg.num_gpus<=0) cfg.num_gpus=dev_count;
        if (cfg.num_gpus>dev_count) cfg.num_gpus=dev_count;
        for (int i=0;i<cfg.num_gpus;i++) cfg.gpu_list.push_back(i);
    }
    int num_workers=(int)cfg.gpu_list.size();

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║ SPECT System Matrix v7.6 CSR-ADAPTIVE CUDA                  ║\n");
    if (cfg.phys.enable_DOI) {
        printf("║ S_VOX=%d³=%d, detector=uniform DOI plane                     ║\n",
               S_VOX,N_SUB_VOX);
    } else {
        printf("║ S_VOX=%d³=%d, S_DET=%d (晶体本体等分)                    ║\n",
               S_VOX,N_SUB_VOX,S_DET);
    }
    printf("║ GPUs=%d, sym=%s, zsym=%d                                    ║\n",
           num_workers,cfg.sym_mode.c_str(),(int)cfg.use_zsym);
    printf("║ ROI radius=%d, output=%s, edge cap=%dM, COO cap=%dM            ║\n",
           cfg.roi_radius,cfg.output_mode.c_str(),cfg.edge_capacity_million,cfg.coo_capacity_million);
    printf("║ COLL_EXP_MODE=%d, cutoff x=%.6g, length=%.6g mm              ║\n",
           COLL_EXP_MODE,
           cfg.phys.coll_exp_cutoff_x,
           cfg.phys.coll_exp_cutoff_len);
    printf("║ DOI line integral=%d  collimator=RECT3(v2_PSC)                ║\n",
           cfg.phys.enable_doi_solid_angle_integral);
    printf("║ DOI sampling=%s, radial samples=%d                            ║\n",
           cfg.phys.enable_DOI ? "uniform-axis" : "crystal-only",
           cfg.doi_radial_samples);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

#ifdef _WIN32
    _mkdir(cfg.output_dir.c_str());
#else
    mkdir(cfg.output_dir.c_str(), 0755);
#endif
    if (cfg.clean_output) clean_old_output(cfg.output_dir);

    /* ====== Pinhole definition: v2_PSC rectangular 3-section channel ======
       4 CAD layers define 3 physical channel sections:
         section 1: pb  -> pmb
         section 2: pmb -> pmt
         section 3: pmt -> pt
       Coordinates are copied from v2_PSC.cu.  Detector and voxel geometry below
       are kept from spect_sysmat_PSC_6module_24pin_YSO.cu. */
    static const int N_PIN_RECT = 4;

    double pb_rect[N_PIN_RECT][4][3]={
        {{  1.15598970, 29.97771986,  8.98864435},{ 10.95540720, 27.92810508,  8.98864435},{ 10.95540720, 27.92810508, -1.17296307},{  1.15598970, 29.97771986, -1.17296307}},
        {{  1.15598970, 29.97771986, -8.98864435},{ 10.95540720, 27.92810508, -8.98864435},{ 10.95540720, 27.92810508,  1.17296307},{  1.15598970, 29.97771986,  1.17296307}},
        {{ -1.15598970, 29.97771986,  8.98864435},{-10.95540720, 27.92810508,  8.98864435},{-10.95540720, 27.92810508, -1.17296307},{ -1.15598970, 29.97771986, -1.17296307}},
        {{ -1.15598970, 29.97771986, -8.98864435},{-10.95540720, 27.92810508, -8.98864435},{-10.95540720, 27.92810508,  1.17296307},{ -1.15598970, 29.97771986,  1.17296307}},
    };

    double pmb_rect[N_PIN_RECT][4][3]={
        {{ 7.81793237,40.55329744, 7.86492332},{ 9.09039730,40.28715275, 7.86492332},{ 9.09039730,40.28715275, 6.46492332},{ 7.81793237,40.55329744, 6.46492332}},
        {{ 7.81793237,40.55329744,-7.86492332},{ 9.09039730,40.28715275,-7.86492332},{ 9.09039730,40.28715275,-6.46492332},{ 7.81793237,40.55329744,-6.46492332}},
        {{-7.81793237,40.55329744, 7.86492332},{-9.09039730,40.28715275, 7.86492332},{-9.09039730,40.28715275, 6.46492332},{-7.81793237,40.55329744, 6.46492332}},
        {{-7.81793237,40.55329744,-7.86492332},{-9.09039730,40.28715275,-7.86492332},{-9.09039730,40.28715275,-6.46492332},{-7.81793237,40.55329744,-6.46492332}}
    };

    double pmt_rect[N_PIN_RECT][4][3]={
        {{ 7.85888275,40.74908541, 7.89962886},{ 9.13134768,40.48294072, 7.89962886},{ 9.13134768,40.48294072, 6.49962886},{ 7.85888275,40.74908541, 6.49962886}},
        {{ 7.85888275,40.74908541,-7.89962886},{ 9.13134768,40.48294072,-7.89962886},{ 9.13134768,40.48294072,-6.49962886},{ 7.85888275,40.74908541,-6.49962886}},
        {{-7.85888275,40.74908541, 7.89962886},{-9.13134768,40.48294072, 7.89962886},{-9.13134768,40.48294072, 6.49962886},{-7.85888275,40.74908541, 6.49962886}},
        {{-7.85888275,40.74908541,-7.89962886},{-9.13134768,40.48294072,-7.89962886},{-9.13134768,40.48294072,-6.49962886},{-7.85888275,40.74908541,-6.49962886}}
    };

    double pt_rect[N_PIN_RECT][4][3]={
        {{ 5.39501394,64.77571941, 21.89109222},{ 21.01809401,61.50804601, 21.89109222},{ 21.01809401,61.50804601, 4.64907371},{ 5.39501394,64.77571941, 4.64907371}},
        {{ 5.39501394,64.77571941,-21.89109222},{ 21.01809401,61.50804601,-21.89109222},{ 21.01809401,61.50804601,-4.64907371},{ 5.39501394,64.77571941,-4.64907371}},
        {{-5.39501394,64.77571941, 21.89109222},{-21.01809401,61.50804601, 21.89109222},{-21.01809401,61.50804601, 4.64907371},{-5.39501394,64.77571941, 4.64907371}},
        {{-5.39501394,64.77571941,-21.89109222},{-21.01809401,61.50804601,-21.89109222},{-21.01809401,61.50804601,-4.64907371},{-5.39501394,64.77571941,-4.64907371}}
    };

    PinLayers layers;
    layers.push_back(make_pin_layer<N_PIN_RECT>(pb_rect));
    layers.push_back(make_pin_layer<N_PIN_RECT>(pmb_rect));
    layers.push_back(make_pin_layer<N_PIN_RECT>(pmt_rect));
    layers.push_back(make_pin_layer<N_PIN_RECT>(pt_rect));

    CollimatorHost coll = build_rect_collimator(
        cfg.coll_box_width_x,
        cfg.coll_box_length_z,
        cfg.coll_box_y_min,
        cfg.coll_box_y_max,
        layers,
        cfg.coll_modules);
    PinholeHost pins = build_pinhole_centers_from_collimator(coll);
    printf("ROI pinhole centers: %d centers from four-layer channel midpoint\n", pins.total);

    VoxelHost vox = build_voxels(cfg.Nx, cfg.Ny, cfg.Nz, cfg.voxel_size);

    if (cfg.det_nx > DET_PIX_X || cfg.det_nz > DET_PIX_Z) {
        fprintf(stderr, "Detector size %dx%d exceeds compile-time capacity MAX_DET_PIX_X/Z=%dx%d. Recompile with -DDET_NX_DEFAULT=<nx> -DDET_NZ_DEFAULT=<nz>, or raise -DMAX_DET_PIX_X/-DMAX_DET_PIX_Z for runtime overrides.\n",
                cfg.det_nx, cfg.det_nz, DET_PIX_X, DET_PIX_Z);
        return 1;
    }

#if COMPILE_ENABLE_DOI
    bool build_crystal_targets = (cfg.phys.enable_DOI == 0);
    bool build_doi_targets = (cfg.phys.enable_DOI != 0);
#else
    bool build_crystal_targets = true;
    bool build_doi_targets = false;
#endif

    DetectorHost det = build_detector(
        cfg.det_nx, cfg.det_nz,
        cfg.pixel_size, cfg.gap_size,
        cfg.detector_y, cfg.detector_thickness,
        cfg.module_gap_f, cfg.module_size,
        cfg.doi_radial_samples,
        build_crystal_targets,
        build_doi_targets);

    int compact_na;
    if      (cfg.sym_mode=="90")   compact_na=cfg.num_angles/4;
    else if (cfg.sym_mode=="45")   compact_na=cfg.num_angles/8+1;
    else                           compact_na=cfg.num_angles;

    int compact_dz = cfg.use_zsym ? (cfg.det_nz+1)/2 : cfg.det_nz;
    int rpa        = cfg.det_nx * compact_dz;
    int total_rows = compact_na * rpa;

    std::vector<double> angles(compact_na);
    for (int i=0;i<compact_na;i++) angles[i]=(double)cfg.voxel_rotation_sign*i*(2.0*M_PI/cfg.num_angles);

    int nchunks=(compact_na+cfg.chunk_angles-1)/cfg.chunk_angles;

    printf("Config:\n");
    printf("  Voxels:    %d×%d×%d = %d\n",cfg.Nx,cfg.Ny,cfg.Nz,vox.total);
    printf("  Detector:  %d×%d, compact Z=%d\n",cfg.det_nx,cfg.det_nz,compact_dz);
    printf("  Angles:    compact=%d, full=%d\n",compact_na,cfg.num_angles);
    printf("  Rows:      %d\n",total_rows);
    printf("  Chunks:    %d (each %d angles)\n",nchunks,cfg.chunk_angles);
    printf("  ROI:       radius=%d, side=%d, area=%d per pinhole\n", cfg.roi_radius, 2*cfg.roi_radius+1, (2*cfg.roi_radius+1)*(2*cfg.roi_radius+1));
    printf("  Output:    %s\n", cfg.output_mode.c_str());
    printf("  DOI samp:  %s, radial_samples=%d, axial_samples=%zu, roi=%d\n",
           cfg.phys.enable_DOI ? "uniform-full-detector-axis" : "crystal-only",
           cfg.doi_radial_samples, det.doi_z.size(), cfg.enable_doi_sample_roi);
    printf("  CSR agg:   %s, dense_rpa_limit=%d, hash_size=%d\n", cfg.csr_agg_mode.c_str(), cfg.dense_rpa_limit, cfg.hash_size);
    printf("  Edge cap:  %dM unique source-det edges/GPU chunk\n", cfg.edge_capacity_million);
    printf("  COO cap:   %dM raw entries/GPU chunk for v2d/coo fallback\n", cfg.coo_capacity_million);
    printf("  Collimator: modules=%d, pinholes=%d, fast-pin=%d\n",
           cfg.coll_modules, pins.total, cfg.fast_pinhole_penetration);
    printf("  Geometry:  voxel=%.4f mm, pixel=%.4f mm, det_y=%.2f mm, thickness=%.2f mm\n",
           cfg.voxel_size, cfg.pixel_size, cfg.detector_y, cfg.detector_thickness);
    printf("  Rotation:  voxel_rotation_sign=%+d, sym=%s, zsym=%d\n",
           cfg.voxel_rotation_sign, cfg.sym_mode.c_str(), cfg.use_zsym ? 1 : 0);
    if (cfg.coll_modules != 6) {
        printf("  *** WARNING: coll_modules=%d != 6. This does NOT match the 6-module\n"
               "      hardware; the resulting system matrix will only model %d module(s).\n"
               "      Pass --coll-modules 6 unless this is intentional.\n",
               cfg.coll_modules, cfg.coll_modules);
    }
    printf("  Output:    %s\n\n",cfg.output_dir.c_str());

    save_metadata(cfg,compact_na,compact_dz,rpa,total_rows,vox.total);

    std::vector<std::thread> threads;
    std::vector<WorkerStats> stats(num_workers);
    auto total_t0=std::chrono::high_resolution_clock::now();

    for (int w=0;w<num_workers;w++) {
        int gid=cfg.gpu_list[w];
        threads.emplace_back(run_gpu_worker,
            w,gid,num_workers,
            std::cref(cfg),std::cref(vox),std::cref(det),std::cref(coll),
            std::cref(pins),
            std::cref(angles),compact_na,compact_dz,rpa,nchunks,
            std::ref(stats[w]));
    }
    for (auto& th:threads) th.join();

    double total_sec=std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now()-total_t0).count();

    unsigned long long ts=0,tp=0; unsigned int to=0, thash=0;
    for (int w=0;w<num_workers;w++) { ts+=stats[w].precheck_skipped; tp+=stats[w].precheck_passed; to+=stats[w].nnz_overflow; thash+=stats[w].hash_overflow; }

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║ Complete  Total=%.2f min                                     ║\n",total_sec/60.0);
    printf("║ Entry overflow entries: %u%s\n",to,to?" *** increase --edge-cap-m/--coo-cap-m or reduce --chunk-angles ***":"");
    printf("║ Hash overflow events: %u%s\n",thash,thash?" *** increase --hash-size or use --csr-agg dense/tile fallback ***":"");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    return 0;
}
