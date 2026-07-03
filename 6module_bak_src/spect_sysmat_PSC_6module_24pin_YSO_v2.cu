/*
SPECT System Matrix PSC-YSO v7.3.2 RECT3-ROI-DIRECT/COO CUDA (v2_PSC rectangular collimator + optional DOI line integral)

v730 vs v727 主要变化：
  1. 修复 DOI 间隙衰减累积错误：物理间隙不再贡献衰减路径长度。
  2. 修复包围盒预筛选的潜在精度风险：改用参数t范围过滤，数学上严格正确。
  3. 自适应间隙采样：大间隙按面积比例分配采样点，小间隙取单点。
  4. 使用 x_edges_flat / z_edges_flat 存储每个晶体真实的物理左右边界。
  5. 探测器采样点由 Host 端 build_axis_targets 生成，含晶体区和间隙区。
  6. 采样权重即该采样段代表的物理宽度，用于正确的立体角面积积分。

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
#define S_DET 5
#endif

#define N_SUB_VOX  (S_VOX*S_VOX*S_VOX)
#define INV_N_SUB_VOX (1.0 / (double)N_SUB_VOX)

#define MAX_INTERVALS   64
#define MAX_DOI_SEG     32
#define BLOCK_SIZE_VA   256
/*
  v7.3.2 支持两种输出模式：
    direct: 使用 GPU 常驻定长 row buffer，避免 COO sort/reduce，显存占用较高但通常更快。
    coo:    使用 COO + sort/reduce，显存更可控，作为验证/回退路径。
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

/*
  大间隙采样点数上限：防止极端比例时过度采样
  小间隙阈值：小于此宽度的间隙只取单点
*/
#define GAP_LARGE_MAX_SAMPLES 10
#define GAP_SMALL_THRESHOLD   0.5f   /* mm，小于此值视为"小间隙"，只取1点 */

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
__device__ unsigned long long g_stat_precheck_skipped;
__device__ unsigned long long g_stat_precheck_passed;

/* ========================================================================
   Constant Memory 结构体
======================================================================== */
#define DET_PIX_X 48
#define DET_PIX_Z 48

/*
  DetectorConst:
    x_edges_flat[2*i]   = 第i个晶体的物理左边界 (X轴)
    x_edges_flat[2*i+1] = 第i个晶体的物理右边界 (X轴)
    z_edges_flat 同理

  用于 DOI 追踪时判断射线当前线段是否处于真实晶体内部，
  从而避免将间隙空气区域计入衰减路径。
*/
struct DetectorConst {
    float x_edges_flat[DET_PIX_X * 2];   /* 每晶体左右边界，共 192 个 float */
    float z_edges_flat[DET_PIX_Z * 2];   /* 每晶体上下边界，共 192 个 float */
    float x_centers[DET_PIX_X];          /* 每个晶体中心，用于 ROI 投影落点映射；投影可落在间隙中，取最近晶体中心 */
    float z_centers[DET_PIX_Z];
    float det_y;                          /* 探测器前表面 Y 坐标 */
    float back_y;                         /* 探测器后表面 Y 坐标 */
    float thickness;                      /* 探测器厚度 */
    int   nx, nz;                         /* 晶体阵列尺寸 */
    int   compact_det_z;                  /* Z 对称后有效像素行数 */
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
           area * cos(theta) / r_front^2，DOI 只返回晶体内首次相互作用概率。
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

      调用端已经乘了 detector front plane 的立体角 area*cos(theta)/t_front^2，
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
    const float*  __restrict__ tx_weight,
    const int*    __restrict__ tz_start,
    const int*    __restrict__ tz_count,
    const float*  __restrict__ tz_val,
    const float*  __restrict__ tz_weight,
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

    bool pass = center_lor_passcheck(
        vcx, vcy, vcz, dkx, dky, dkz,
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
            float wx_w = tx_weight[ts_x + dx_idx];

            for (int dz_idx = 0; dz_idx < tc_z; dz_idx++) {
                float sz   = tz_val   [ts_z + dz_idx];
                float wz_w = tz_weight[ts_z + dz_idx];

                double target_area = (double)wx_w * (double)wz_w;

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
                double mp = analytical_penetration_const(vx, vy, vz, ldx, ldy, ldz);

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

                double W_ray = target_area * cp * inv_r * inv_r;

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



__global__ void kernel_subray_roi_coo(
    const float*  __restrict__ vox_x,
    const float*  __restrict__ vox_y,
    const float*  __restrict__ vox_z,
    const int*    __restrict__ tx_start,
    const int*    __restrict__ tx_count,
    const float*  __restrict__ tx_val,
    const float*  __restrict__ tx_weight,
    const int*    __restrict__ tz_start,
    const int*    __restrict__ tz_count,
    const float*  __restrict__ tz_val,
    const float*  __restrict__ tz_weight,
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

    bool pass = center_lor_passcheck(
        vcx, vcy, vcz, dkx, dky, dkz,
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
            float wx_w = tx_weight[ts_x + dx_idx];

            for (int dz_idx = 0; dz_idx < tc_z; dz_idx++) {
                float sz   = tz_val   [ts_z + dz_idx];
                float wz_w = tz_weight[ts_z + dz_idx];

                double target_area = (double)wx_w * (double)wz_w;

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
                double mp = analytical_penetration_const(vx, vy, vz, ldx, ldy, ldz);

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

                double W_ray = target_area * cp * inv_r * inv_r;

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

/*
  AxisTargets：为单个坐标轴方向生成包含晶体区和间隙区的完整采样点表。

  每个"像素"线程负责：
    1. 本晶体表面（ps 宽度）的 S_det 个等分采样点。
    2. 本晶体右侧（或下侧）的间隙区域的采样点：
       - 大间隙（>= GAP_SMALL_THRESHOLD）：按照 round(S_det * gap/ps) 分配，
         最大不超过 GAP_LARGE_MAX_SAMPLES。
       - 小间隙（< GAP_SMALL_THRESHOLD）：只取 1 个中心点。
    这样可以正确对探测器整个连续表面（含间隙）做面积加权的立体角积分。

  注意：最后一个像素（ix == nx-1）没有右侧间隙，只有晶体本身的采样点。
*/
struct AxisTargets {
    std::vector<int>   start;   /* 每个像素对应的采样起始索引 */
    std::vector<int>   count;   /* 每个像素对应的采样点数 */
    std::vector<float> val;     /* 采样点坐标 */
    std::vector<float> weight;  /* 采样点代表的物理宽度（用于面积积分） */
};

static AxisTargets build_axis_targets(
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

        /* --- 1. 晶体本身：S_det 个等分采样点 --- */
        float c_left = centers[i] - ps * 0.5f;
        float step_c = ps / (float)S_det;
        for (int j = 0; j < S_det; j++) {
            T.val.push_back(c_left + (j + 0.5f) * step_c);
            T.weight.push_back(step_c);
        }

        /* --- 2. 右侧间隙（若存在） --- */
        if (i < n - 1) {
            float gap = centers[i+1] - centers[i] - ps;
            if (gap > 1e-4f) {
                float g_left = centers[i] + ps * 0.5f;

                if (gap < GAP_SMALL_THRESHOLD) {
                    /* 小间隙：仅取中心一点 */
                    T.val.push_back(g_left + gap * 0.5f);
                    T.weight.push_back(gap);
                } else {
                    /* 大间隙：按比例分配采样点数，并限制上限 */
                    int N_gap = (int)roundf((float)S_det * gap / ps);
                    if (N_gap < 1) N_gap = 1;
                    if (N_gap > GAP_LARGE_MAX_SAMPLES) N_gap = GAP_LARGE_MAX_SAMPLES;

                    float step_g = gap / (float)N_gap;
                    for (int j = 0; j < N_gap; j++) {
                        T.val.push_back(g_left + (j + 0.5f) * step_g);
                        T.weight.push_back(step_g);
                    }
                }
            }
        }

        T.count[i] = (int)T.val.size() - T.start[i];
    }

    return T;
}

struct DetectorHost {
    std::vector<float> cx, cy, cz;
    std::vector<float> x_centers, z_centers;
    std::vector<float> x_edges_flat;  /* [2*nx]：每晶体物理左右边界 */
    std::vector<float> z_edges_flat;  /* [2*nz]：每晶体物理上下边界 */
    AxisTargets tx, tz;               /* X/Z 轴含间隙采样点表 */
    float det_y, back_y, thickness;
    int nx, nz;
};

static DetectorHost build_detector(
    int nx, int nz, float ps, float gs,
    float dy, float th, float mg, int ms)
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

    /* 构建含间隙的自适应采样点表 */
    D.tx = build_axis_targets(nx, xc, ps, S_DET);
    D.tz = build_axis_targets(nz, zc, ps, S_DET);

    /* 打印采样点统计 */
    int total_tx = (int)D.tx.val.size();
    int total_tz = (int)D.tz.val.size();
    printf("✓ Detector targets: X axis %d samples (avg %.1f/pixel), "
           "Z axis %d samples (avg %.1f/pixel)\n",
           total_tx, (float)total_tx/nx,
           total_tz, (float)total_tz/nz);

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


static void save_chunk_direct(
    const char* fn,
    const int* vi_1based,
    const float* wg,
    int* nz,
    int ao,
    int ac,
    int rt)
{
    /*
      保存 direct row buffer 到旧版 chunk 格式。

      direct kernel 为兼容旧版内部约定，将 voxel index 存成 1-based；
      文件格式仍写 0-based voxel index，因此这里写出时减 1。

      注意：vi_1based / wg 是 host 端完整 row-major 缓冲区，大小为
      rt * MAX_NNZ_PER_ROW；nz 是每行有效项数。
    */
    FILE* f=fopen(fn,"wb");
    if(!f) return;
    fwrite(&rt,4,1,f);
    fwrite(&ao,4,1,f);
    fwrite(&ac,4,1,f);
    for (int r=0;r<rt;r++) {
        int n=nz[r];
        if (n > MAX_NNZ_PER_ROW) n = MAX_NNZ_PER_ROW;
        if (n < 0) n = 0;
        fwrite(&n,4,1,f);
        if (n>0) {
            std::vector<int> v0(n);
            long long base = (long long)r * MAX_NNZ_PER_ROW;
            for (int k=0;k<n;k++) v0[k]=vi_1based[base+k]-1;
            fwrite(v0.data(),4,n,f);
            fwrite(&wg[base],4,n,f);
        }
    }
    fclose(f);
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
      keys 已经建议在 GPU 端按 key 排序并 reduce。为了鲁棒性，
      如果调用方关闭 GPU reduce 或调试时传入未排序 COO，这里再做一次 Host 端排序。
      文件格式保持旧版 chunk 格式：
        int rt, ao, ac
        for each row:
          int n
          int voxel_indices[n]   // 0-based
          float weights[n]
      因此原有后处理/重建程序通常无需修改。
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
            if (!k2.empty() && k2.back() == keys[id]) {
                v2.back() += vals[id];
            } else {
                k2.push_back(keys[id]);
                v2.push_back(vals[id]);
            }
        }
        keys.swap(k2);
        vals.swap(v2);
    }

    FILE* f = fopen(fn, "wb");
    if (!f) return;
    fwrite(&rt, 4, 1, f);
    fwrite(&ao, 4, 1, f);
    fwrite(&ac, 4, 1, f);

    size_t pos = 0;
    for (int r = 0; r < rt; r++) {
        std::vector<int>   cols;
        std::vector<float> wg;

        while (pos < keys.size()) {
            unsigned long long row = keys[pos] / (unsigned long long)total_voxels;
            if ((int)row < r) { pos++; continue; }
            if ((int)row > r) break;

            float v = vals[pos];
            unsigned long long col = keys[pos] % (unsigned long long)total_voxels;
            if (v >= weight_threshold && col < (unsigned long long)total_voxels) {
                cols.push_back((int)col);       /* 0-based voxel index */
                wg.push_back(v);
            }
            pos++;
        }

        int n = (int)cols.size();
        fwrite(&n, 4, 1, f);
        if (n > 0) {
            fwrite(cols.data(), 4, n, f);
            fwrite(wg.data(),   4, n, f);
        }
    }
    fclose(f);
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
        if ((starts_with(name,"chunk_")&&ends_with(name,".bin"))||
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
        if ((starts_with(name,"chunk_")&&ends_with(name,".bin"))||
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
    std::string output_dir = "matrix_PSC_6modules_24pin_YSO_3X5_PEN_DOI_V2";
    std::string sym_mode   = "45";

    int Nx=108, Ny=108, Nz=80;
    int det_nx=48, det_nz=48;
    int num_angles=360;

    float voxel_size=0.3f;
    float pixel_size=1.55f;
    float gap_size=0.16f;
    float detector_y=94.0f;
    float detector_thickness=9.0f;
    float module_gap_f=1.34f;
    int   module_size=12;

    int   coll_modules=6;
    float coll_box_width_x=68.2f;
    float coll_box_length_z=100.0f;
    float coll_box_y_min=32.0f;
    float coll_box_y_max=60.0f;
    int   voxel_rotation_sign=1;

    bool use_zsym=true;
    int  chunk_angles=23;
    int  clean_output=1;

    /* ROI 半窗口，窗口边长=2*roi_radius+1。建议先用 6~8 做保守验证，再逐步缩小到 4~5。 */
    int  roi_radius=7;

    /*
      输出模式：
        direct: GPU 常驻定长 row buffer，跳过 COO sort/reduce；显存占用高，速度通常更快。
        coo:    COO + sort/reduce；显存更可控，适合作为验证/回退模式。
    */
    std::string output_mode = "direct";

    /* 每个 chunk 的 COO 原始条目容量。单位：百万条。仅 output_mode=coo 时使用。 */
    int  coo_capacity_million=120;
    int  gpu_reduce=1;
    int  num_gpus=2;
    std::vector<int> gpu_list;

    PhysicsConst phys;
};

struct WorkerStats {
    unsigned int      nnz_overflow=0;
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
    int roi_radius = cfg.roi_radius;
    int roi_side = 2*cfg.roi_radius + 1;

    /* 旧字段顺序保持不变，兼容已有读取程序。 */
    fwrite(&total_rows,4,1,f); fwrite(&total_voxels,4,1,f); fwrite(&mn,4,1,f);
    fwrite(&compact_na,4,1,f); fwrite(&cfg.det_nx,4,1,f); fwrite(&compact_dz,4,1,f);
    fwrite(&rpa,4,1,f); fwrite(&full_na,4,1,f);
    fwrite(&cfg.Nx,4,1,f); fwrite(&cfg.Ny,4,1,f); fwrite(&cfg.Nz,4,1,f);
    fwrite(&cfg.voxel_size,4,1,f); fwrite(&cfg.det_nx,4,1,f); fwrite(&cfg.det_nz,4,1,f);
    fwrite(&cfg.pixel_size,4,1,f); fwrite(&cfg.detector_y,4,1,f);
    fwrite(&cfg.detector_thickness,4,1,f); fwrite(&use90,4,1,f); fwrite(&zsym,4,1,f);

    /* v7.3.2 新增字段：明确写入 45° 对称、ROI 与输出模式信息。 */
    fwrite(&use45,4,1,f);
    fwrite(&sym_code,4,1,f);
    fwrite(&symmetry_order,4,1,f);
    fwrite(&roi_radius,4,1,f);
    fwrite(&roi_side,4,1,f);
    fclose(f);

    /* 同步写一个文本版 metadata，方便人工检查和后处理脚本读取。 */
    std::string tf=cfg.output_dir+"/metadata.txt";
    FILE* t=fopen(tf.c_str(),"w");
    if (t) {
        fprintf(t,"version=v7.3.2_rect3_roi_direct_coo\n");
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
        fprintf(t,"roi_radius=%d\n",roi_radius);
        fprintf(t,"roi_side=%d\n",roi_side);
        fprintf(t,"doi_solid_angle_integral=%d\n",cfg.phys.enable_doi_solid_angle_integral);
        fprintf(t,"collimator=rectangular_v2_PSC_three_section\n");
        fprintf(t,"output_mode=%s\n",cfg.output_mode.c_str());
        fprintf(t,"coo_capacity_million=%d\n",cfg.coo_capacity_million);
        fprintf(t,"note=chunk bin remains old row-wise format; fixed MAX_NNZ row buffer removed internally.\n");
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
    float  *d_vx,*d_vy,*d_vz;
    double *d_ang_cos,*d_ang_sin;
    double *d_pnx,*d_pny,*d_pnz,*d_pd;
    double *d_bcx,*d_bcy,*d_bcz,*d_br2;
    int    *d_tx_s,*d_tx_c,*d_tz_s,*d_tz_c;
    float  *d_tx_v,*d_tx_w,*d_tz_v,*d_tz_w;

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

    gpu_up(&d_tx_s,det.tx.start); gpu_up(&d_tx_c,det.tx.count);
    gpu_up(&d_tx_v,det.tx.val);   gpu_up(&d_tx_w,det.tx.weight);
    gpu_up(&d_tz_s,det.tz.start); gpu_up(&d_tz_c,det.tz.count);
    gpu_up(&d_tz_v,det.tz.val);   gpu_up(&d_tz_w,det.tz.weight);

    /* 设置 constant memory */
    DetectorConst dc; memset(&dc,0,sizeof(dc));
    for (int i=0;i<cfg.det_nx*2;i++) dc.x_edges_flat[i]=det.x_edges_flat[i];
    for (int i=0;i<cfg.det_nz*2;i++) dc.z_edges_flat[i]=det.z_edges_flat[i];
    for (int i=0;i<cfg.det_nx;i++) dc.x_centers[i]=det.x_centers[i];
    for (int i=0;i<cfg.det_nz;i++) dc.z_centers[i]=det.z_centers[i];
    dc.det_y=det.det_y; dc.back_y=det.back_y; dc.thickness=det.thickness;
    dc.nx=cfg.det_nx; dc.nz=cfg.det_nz; dc.compact_det_z=compact_dz;
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
    CUDA_CHECK(cudaMemcpyToSymbol(c_cfg, &sc, sizeof(sc)));

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
    CUDA_CHECK(cudaMemcpyToSymbol(g_stat_precheck_skipped,&z64,sizeof(z64)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_stat_precheck_passed, &z64,sizeof(z64)));

    if (cfg.output_mode == "direct") {
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
            long long nthreads=(long long)vox.total * ac * pins.total * roi_side * roi_side;
            int nb=(int)((nthreads+BLOCK_SIZE_VA-1)/BLOCK_SIZE_VA);

            kernel_subray_roi_direct<<<nb, BLOCK_SIZE_VA>>>(
                d_vx,d_vy,d_vz,
                d_tx_s,d_tx_c,d_tx_v,d_tx_w,
                d_tz_s,d_tz_c,d_tz_v,d_tz_w,
                d_ang_cos,d_ang_sin,
                d_pnx,d_pny,d_pnz,d_pd,
                d_bcx,d_bcy,d_bcz,d_br2,
                as,ac, d_vo,d_wo,d_no);

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
            snprintf(fn,sizeof(fn),"%s/chunk_%03d.bin",cfg.output_dir.c_str(),ch);
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
            long long nthreads=(long long)vox.total * ac * pins.total * roi_side * roi_side;
            int nb=(int)((nthreads+BLOCK_SIZE_VA-1)/BLOCK_SIZE_VA);

            kernel_subray_roi_coo<<<nb, BLOCK_SIZE_VA>>>(
                d_vx,d_vy,d_vz,
                d_tx_s,d_tx_c,d_tx_v,d_tx_w,
                d_tz_s,d_tz_c,d_tz_v,d_tz_w,
                d_ang_cos,d_ang_sin,
                d_pnx,d_pny,d_pnz,d_pd,
                d_bcx,d_bcy,d_bcz,d_br2,
                as,ac, d_keys,d_vals,d_raw_nnz, coo_cap);

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
            snprintf(fn,sizeof(fn),"%s/chunk_%03d.bin",cfg.output_dir.c_str(),ch);
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
    CUDA_CHECK(cudaMemcpyFromSymbol(&stats.precheck_skipped,
        g_stat_precheck_skipped,sizeof(stats.precheck_skipped)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&stats.precheck_passed,
        g_stat_precheck_passed,sizeof(stats.precheck_passed)));

    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_ang_cos); cudaFree(d_ang_sin);
    cudaFree(d_pnx); cudaFree(d_pny); cudaFree(d_pnz); cudaFree(d_pd);
    cudaFree(d_bcx); cudaFree(d_bcy); cudaFree(d_bcz); cudaFree(d_br2);
    cudaFree(d_tx_s); cudaFree(d_tx_c); cudaFree(d_tx_v); cudaFree(d_tx_w);
    cudaFree(d_tz_s); cudaFree(d_tz_c); cudaFree(d_tz_v); cudaFree(d_tz_w);
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

    cfg.phys.mu_collimator                   = 3.202;
    cfg.phys.center_lor_mp_threshold         = 5.0;
    cfg.phys.mu_detector                     = 0.1368f;
    cfg.phys.penetration_length_threshold    = 1e-3f;
    cfg.phys.penetration_probability_threshold = 0.05f;
    cfg.phys.weight_threshold                = 1e-12f;
    cfg.phys.solid_angle_power               = 1;
    cfg.phys.enable_penetration              = 1;
    cfg.phys.enable_DOI                      = 1;
    cfg.phys.enable_center_lor_precheck      = 1;
    cfg.phys.enable_doi_solid_angle_integral = 0;
    cfg.phys.pixel_size                      = cfg.pixel_size;
    cfg.phys.voxel_size                      = cfg.voxel_size;

    for (int ai=1; ai<argc; ai++) {
        if (!strcmp(argv[ai],"--num-gpus")&&ai+1<argc)     cfg.num_gpus=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--gpu-list")&&ai+1<argc) cfg.gpu_list=parse_gpu_list(argv[++ai]);
        else if (!strcmp(argv[ai],"--sym")&&ai+1<argc)      cfg.sym_mode=argv[++ai];
        else if (!strcmp(argv[ai],"--zsym")&&ai+1<argc)     cfg.use_zsym=atoi(argv[++ai])!=0;
        else if (!strcmp(argv[ai],"--chunk-angles")&&ai+1<argc) cfg.chunk_angles=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--out")&&ai+1<argc)      cfg.output_dir=argv[++ai];
        else if (!strcmp(argv[ai],"--clean")&&ai+1<argc)    cfg.clean_output=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--precheck")&&ai+1<argc) cfg.phys.enable_center_lor_precheck=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--center-threshold")&&ai+1<argc) cfg.phys.center_lor_mp_threshold=atof(argv[++ai]);
        else if (!strcmp(argv[ai],"--penetration")&&ai+1<argc) cfg.phys.enable_penetration=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--roi-radius")&&ai+1<argc) cfg.roi_radius=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--matrix-output")&&ai+1<argc) cfg.output_mode=argv[++ai];
        else if (!strcmp(argv[ai],"--output-mode")&&ai+1<argc) cfg.output_mode=argv[++ai];
        else if (!strcmp(argv[ai],"--use-coo")&&ai+1<argc) cfg.output_mode=(atoi(argv[++ai])?"coo":"direct");
        else if (!strcmp(argv[ai],"--coo-cap-m")&&ai+1<argc) cfg.coo_capacity_million=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--doi")&&ai+1<argc) cfg.phys.enable_DOI=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--doi-solid-angle-integral")&&ai+1<argc) cfg.phys.enable_doi_solid_angle_integral=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--doi-line-integral")&&ai+1<argc) cfg.phys.enable_doi_solid_angle_integral=atoi(argv[++ai]);
        else if (!strcmp(argv[ai],"--coll-modules")&&ai+1<argc) cfg.coll_modules=atoi(argv[++ai]);
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
    if (cfg.roi_radius < 0) cfg.roi_radius = 0;
    if (cfg.coo_capacity_million < 1) cfg.coo_capacity_million = 1;
    if (cfg.output_mode != "direct" && cfg.output_mode != "coo") {
        fprintf(stderr,"Invalid --matrix-output %s, expected direct or coo.\n", cfg.output_mode.c_str());
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
    printf("║ SPECT System Matrix v7.3.2 ROI-DIRECT/COO CUDA              ║\n");
    printf("║ S_VOX=%d³=%d, S_DET=%d (晶体等分), GAP_SMALL<%.2fmm→1pt  ║\n",
           S_VOX,N_SUB_VOX,S_DET,GAP_SMALL_THRESHOLD);
    printf("║ GPUs=%d, sym=%s, zsym=%d                                    ║\n",
           num_workers,cfg.sym_mode.c_str(),(int)cfg.use_zsym);
    printf("║ ROI radius=%d, output=%s, COO cap=%dM entries/GPU chunk       ║\n",
           cfg.roi_radius,cfg.output_mode.c_str(),cfg.coo_capacity_million);
    printf("║ COLL_EXP_MODE=%d, cutoff x=%.6g, length=%.6g mm              ║\n",
           COLL_EXP_MODE,
           cfg.phys.coll_exp_cutoff_x,
           cfg.phys.coll_exp_cutoff_len);
    printf("║ DOI line integral=%d  collimator=RECT3(v2_PSC)                ║\n",
           cfg.phys.enable_doi_solid_angle_integral);
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

    DetectorHost det = build_detector(
        cfg.det_nx, cfg.det_nz,
        cfg.pixel_size, cfg.gap_size,
        cfg.detector_y, cfg.detector_thickness,
        cfg.module_gap_f, cfg.module_size);

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
    printf("  COO cap:   %dM raw entries per GPU chunk (coo mode only)\n", cfg.coo_capacity_million);
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

    unsigned long long ts=0,tp=0; unsigned int to=0;
    for (int w=0;w<num_workers;w++) { ts+=stats[w].precheck_skipped; tp+=stats[w].precheck_passed; to+=stats[w].nnz_overflow; }

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║ Complete  Total=%.2f min                                     ║\n",total_sec/60.0);
    printf("║ COO overflow entries: %u%s\n",to,to?" *** increase --coo-cap-m or reduce --chunk-angles ***":"");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    return 0;
}
