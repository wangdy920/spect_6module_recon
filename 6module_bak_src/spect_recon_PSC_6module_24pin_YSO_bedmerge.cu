/*
 * SPECT 多床位 MLEM/OSL 重建、图像卷积、后处理和余弦加权拼接程序。
 *
 * 功能与算法边界
 * --------------
 * 每个 BedN 投影拥有独立的局部 Nx*Ny*Nz 重建状态；不同床位只共享系统
 * 矩阵和角度分块，不共享似然函数或先验。圆柱 ROI 同时限定有效体素和 CSR
 * 非零元；每个保存轮次再中心裁剪用户指定的三维 mask，在相邻床位重叠区
 * 独立估计标量增益，并用 cos^2/sin^2 权重完成轴向拼接。默认配置为
 * --algorithm mlem --implementation omega，且关闭卷积、先验和滤波。
 *
 * MLEM/OSL 的参考实现
 * -------------------
 * --implementation 是 EM 数值策略和先验实现的唯一版本开关，避免把不同软件
 * 的公式混入同一轮更新：
 *
 * 1. OMEGA 使用 epsilon 数值保护。投影域计算
 *      r_i = g_i*y_i/(g_i*(A*x)_i + epsilon)，
 *    sensitivity 小于 epsilon 时按 epsilon 使用；MLEM 更新分子再加 epsilon。
 *    OSL 在 sensitivity+beta*prior_gradient 后加 epsilon；非正/非有限分母或
 *    异常更新写入 epsilon positivity 下限，并输出逐类计数、范围和 beta 诊断。
 * 2. CASToR 使用前向阈值。当 g_i*(A*x)_i 不大于 threshold 时令 r_i=g_i；
 *    这里的 g_i 对应 CASToR 在隐式响应模型中使用的中性 ratio=1。MLEM 和
 *    OSL 都对更新因子应用相同的最小/最大限制。OSL 分母非正时保持旧体素并
 *    计数报警。
 *
 * MLEM 是无先验的最大似然更新；OSL 是将 prior 一阶导数延迟一轮计算的惩罚
 * 似然更新。MAP-EM 是一类目标和算法族，不是一条由 OMEGA 与 CASToR 共同
 * 定义的独立更新公式，因此本程序不把 --algorithm map-em 静默映射为 OSL。
 * OMEGA 可配 TV 或 MRP；CASToR 3.2 只有本程序所参考的 MRP，没有 TV penalty，
 * 所以 CASToR+TV 会在启动前报错。当前普通 TV/MRP 均不需要 CT；本文件不实现
 * 解剖引导 TV、APLS 或其他需要配准先验图像的算法。
 *
 * PSF 与后处理
 * ------------
 * --conv-impl omega 使用 OMEGA 中心采样 Gaussian 和 symmetric 边界；castor
 * 严格复现 CASToR 的 erf 体素积分核、仅 z 边界补偿以及“反向再次调用正向
 * convolver”的行为；castor-adjoint 保持 CASToR 前向 C=D_z^-1*K 不变，但
 * 将反向修正为真正的 C^T=K*D_z^-1。PSF 版本可独立选择以便对照实验，不参与
 * EM/prior 版本绑定。post 只处理保存副本，sieve 严格复用同一个 CASToR
 * convolver，intra/smooth 在迭代之间回写状态。smooth 可从指定轮次开始，
 * 选择每轮、周期或仅最终轮执行；post 则仍只处理每个保存轮次的副本。
 *
 * 投影格式、坐标和线性索引
 * --------------------------
 * 投影文件头为两个小端 int32：full_num_angles 和 det_nx*det_nz。数据区
 * 必须完整保存为 float32 或 float64，按角度优先展开：
 *   row = angle*(det_nz*det_nx) + det_z*det_nx + det_x。
 * 局部图像统一采用 0-based 索引并按
 *   voxel = z*(Nx*Ny) + y*Nx + x
 * 展平，x 连续。Bed1 位于最大物理 z，床号增加时沿负 z 移动
 * --bed-step-mm；裁剪、增益拟合和拼接均遵循同一约定。
 *
 * 实测与仿真数据
 * --------------
 * 实测数据默认屏蔽 LINK4/FPGA2/DET4 探测器 bin，并可加载实测归一化/
 * 系统响应因子 g_i。仿真模式默认关闭该坏道 mask；没有显式提供
 * --norm-factor 时也关闭归一化因子自动发现，防止理想投影误用实测响应。
 * 上传 GPU 前会拒绝 NaN、Inf 和负投影值。
 *
 * 调度、显存和卷积时机
 * --------------------
 * 调度顺序为 iteration -> angle slab -> Bed1..BedN -> 局部更新。
 * --angle-batch 0 将完整 CSR 分配到多个 GPU；大于 0 时由 GPU0 分批加载
 * compact angle slab。基础状态和反投影显存约为
 * 2*beds*Nx*Ny*Nz*sizeof(float)。只有实际启用 PSF、OSL prior、Gaussian 或
 * Butterworth 时才分配各自缓存；OMEGA 不分配 CASToR threshold 计数器或
 * z-normalization 体积。Butterworth 截止频率使用 cycles/cm，并通过每轴
 * 2 倍补零的 R2C/C2R FFT 减小循环串扰；可按 every、final、periodic 或
 * last_n 调度；--butter-start-iter 可让前若干轮保持纯 MLEM/OSL，其中除
 * final 外的滤波都会改变后续迭代轨迹。--stitch-filter 则只在增益拟合和
 * cos^2/sin^2 拼接完成后生成派生体积，不回写任何单床重建状态。
 *
 * 输出目录
 * --------
 * 相对 --out 名称自动放入当前目录 recon_output/；绝对路径以及已以
 * recon_output/ 开头的相对路径保持不变。single_bed_raw/BedN 暂存完整局部
 * 图像，single_bed/BedN 保存中心 mask，stitched 保存各轮 NIfTI-1 hdr/img。
 * 启用拼接后滤波时，额外写入 stitched_smooth_fwhm0p45/ 和/或
 * stitched_butter_fc3cpcm_o8/ 等参数化目录，未滤波 stitched/ 始终保留。
 * 默认每 50 次保存为 recon50、recon100
 * 等，不生成 recon_final。未指定 --keep-raw 时，完整尺寸中间结果在 mask
 * 写出后删除。文件中保留的全局联合引擎只用于底层诊断，公开 --proj-dir
 * 入口始终使用上述独立局部状态调度器。
 *
 * Windows 编译示例：
 *   nvcc -O3 -std=c++17 -arch=sm_120 ^
 *     -Xcompiler "/utf-8,/Zc:preprocessor" ^
 *     spect_recon_PSC_6module_24pin_YSO_bedmerge.cu ^
 *     -o spect_recon_PSC_6module_24pin_YSO_bedmerge.exe -lcusparse -lcufft
 *
 * 仿真数据与 OMEGA OSL-TV 示例：
 *   spect_recon_PSC_6module_24pin_YSO_bedmerge.exe ^
 *     --matrix-dir MCmatrix_PSC --proj-dir simulation_preprocess ^
 *     --out recon_tv --data-mode simulation --iters 50 ^
 *     --algorithm osl --implementation omega --prior tv ^
 *     --map-beta 1e-3 --tv-smoothing 1e-4
 */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cufft.h>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cinttypes>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <dirent.h>
#endif
#include <algorithm>
#include <vector>
#include <string>
#include <chrono>
#include <future>
#include <thread>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <cctype>
#include <limits>
#include <utility>

namespace fs = std::filesystem;

#ifdef _WIN32
#define MKDIR(p) _mkdir(p)
#else
#define MKDIR(p) mkdir(p, 0755)
#endif

#define BLK 256
#define MAX_KERNEL_LEN 129

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[CUDA] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t _s = (call); \
    if (_s != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "[cuSPARSE] %s:%d status=%d\n", __FILE__, __LINE__, (int)_s); \
        exit(1); \
    } \
} while(0)

#define CUFFT_CHECK(call) do { \
    cufftResult _r = (call); \
    if (_r != CUFFT_SUCCESS) { \
        fprintf(stderr, "[cuFFT] %s:%d result=%d\n", __FILE__, __LINE__, (int)_r); \
        exit(1); \
    } \
} while(0)

#define IO_CHECK(call, expected) do { \
    size_t _got = (size_t)(call); \
    if (_got != (size_t)(expected)) { \
        fprintf(stderr, "[I/O] %s:%d expected=%zu got=%zu\n", \
                __FILE__, __LINE__, (size_t)(expected), _got); \
        exit(1); \
    } \
} while(0)

// 系统矩阵的对称性模式和图像后处理映射类型。
enum SymMode {
    SYM_NONE = 0,
    SYM_ZONLY,
    SYM_180,
    SYM_Z180,
    SYM_90,
    SYM_45,
    SYM_45Z
};

enum MapType {
    MAP_NONE = 0,
    MAP_QUADRATIC,
    MAP_TV,
    MAP_MRP,
    MAP_HUBER,
    MAP_RELATIVE,
    MAP_LOGCOSH,
    MAP_HYPERBOLIC
};

// 公开多床位入口使用的算法与参考实现。枚举值会传入 CUDA kernel，必须保持为
// 简单稳定的整数。ReferenceImplementation 是 EM 和 prior 的唯一版本来源。
enum ReconstructionAlgorithm {
    ALGORITHM_MLEM = 0,
    ALGORITHM_OSL = 1
};

enum ReferenceImplementation {
    IMPLEMENTATION_OMEGA = 0,
    IMPLEMENTATION_CASTOR = 1
};

enum ConvolutionImplementation {
    CONV_IMPLEMENTATION_CASTOR_ADJOINT = 0,
    CONV_IMPLEMENTATION_OMEGA = 1,
    CONV_IMPLEMENTATION_CASTOR = 2
};

enum MRPNeighborhoodShape {
    MRP_NEIGHBORHOOD_AUTO = -1,
    MRP_NEIGHBORHOOD_SIX_NEAREST = 0,
    MRP_NEIGHBORHOOD_BOX = 1,
    MRP_NEIGHBORHOOD_SPHERE = 2
};

// CUDA 重建引擎的参数。该结构只描述单次床位重建，不包含拼接 mask。
struct ReconConfig {
    // 输入、输出和归一化因子路径。
    std::string matrix_dir;
    std::string proj_path;
    std::string output_dir = "recon_multibed_output";
    bool output_dir_explicit = false;
    std::string norm_factor_path;
    std::string norm_factor_mode = "auto";
    bool norm_factor_invert = false;
    bool norm_factor_enable = true;

    // 圆柱 ROI、数据模式和坏探测器掩码。
    bool roi_enable = true;
    float roi_diameter_mm = 30.0f;
    float roi_height_mm = 66.0f;
    float roi_center_mm[3] = {0.0f, 0.0f, 0.0f};
    std::string data_mode = "measured";
    bool detector_mask_enable = true;

    // 迭代次数、保存周期和 GPU 数量。
    bool check_sens_zsym = false;
    bool save_sens_edge_png = false;
    bool save_raw1 = true;
    float sens_zsym_tol = 1e-4f;

    int num_iters = 50;
    int save_interval = 50;
    int num_gpus = 1;

    int angle_batch = 0;

    // 系统矩阵分块驻留与预取策略。
    bool slab_prefetch = true;
    bool slab_prefetch_early = false;

    int gpu_slab_cache = 0;

    // 系统矩阵对称性覆盖设置。
    std::string sym_override = "auto";

    // 迭代中的 PSF 卷积设置。
    std::string conv_mode = "none";
    float psf_fwhm_mm[3] = {1.5f, 1.5f, 1.5f};
    float psf_nb_sigmas = 3.0f;

    // 迭代后的后处理平滑调度。
    std::string post_schedule = "constant";
    float post_fwhm_mm = 0.45f;
    float post_fwhm_start_mm = 0.60f;
    float post_fwhm_end_mm = 0.15f;
    int post_start_iter = 1;
    int post_apply_every = 1;

    // 映射正则化参数。
    MapType map_type = MAP_NONE;
    float map_beta = 0.0f;
    int map_neigh26 = 1;
    float map_delta = 1e-6f;
    float tv_smoothing = 1e-4f;
    int mrp_radius[3] = {1, 1, 1};
    float mrp_epsilon = 1e-8f;
    bool mrp_no_norm = false;
    float map_huber_threshold = 0.01f;
    float map_relative_gamma = 2.0f;
    float map_relative_epsilon = 1e-6f;

    /* 正值约束和 Butterworth 频域滤波；截止频率使用物理单位 cycles/cm。 */
    bool positivity = false;
    bool butter_enable = false;
    float butter_cutoff_cycles_per_cm = 0.0f;
    int butter_order = 8;
    std::string butter_mode = "final";
    int butter_start_iter = 0;
    int butter_period = 5;
    int butter_last_n = 10;

    // 投影角度抽样步长。
    int recon_angle_step = 1;

    // 多床位投影名称、床位步进及显式床位 z 偏移。
    std::string bed_projection_name = "projection.bin";
    float bed_step_mm = 0.0f;
    bool bed_step_set = false;
    std::vector<float> bed_offsets_mm;
    // 多床时在两端床位中心外各保留的轴向层数。
    int multibed_edge_layers = 40;

    bool apply_roi_each_iter = true;
};

// 底层重建引擎使用的床位几何信息。
struct BedInfo {
    std::string name;
    std::string projection_path;
    float center_mm = 0.0f;

    float shift_vox = 0.0f;
    int shift_floor = 0;
    float shift_frac = 0.0f;

    double raw_counts = 0.0;
};

// 多床联合坐标系的尺寸、体素大小和 z 方向物理范围。
struct GlobalGeometry {
    int Nx = 0, Ny = 0;
    int local_Nz = 0;
    int global_Nz = 0;
    int64_t local_voxels = 0;
    int64_t global_voxels = 0;
    float voxel_mm = 0.0f;

    float origin_z_mm = 0.0f;
    float min_bed_center_mm = 0.0f;
    float max_bed_center_mm = 0.0f;
};

// 系统矩阵 metadata；Nx、Ny、Nz 与 CSR 列索引必须保持一致。
struct MatrixMeta {
    int total_rows = 0;
    int total_voxels = 0;
    int max_nnz = 0;
    int compact_num_angles = 0;
    int det_nx = 0;
    int compact_det_z = 0;
    int compact_rpa = 0;
    int full_num_angles = 0;
    int Nx = 0, Ny = 0, Nz = 0;
    float voxel_size = 0.0f;
    int det_nx_dup = 0;
    int det_nz = 0;
    float pixel_size = 0.0f;
    float detector_y = 0.0f;
    float detector_thickness = 0.0f;
    int use_90sym = 0;
    int use_zsym = 0;

    int use_45sym = 0;
    int sym_code = 0;
    int symmetry_order = 1;
};

// 一个 CSR 分块文件的索引信息。
struct ChunkInfo {
    std::string path;
    int num_rows = 0;
    int angle_offset = 0;
    int angle_count = 0;
    int64_t nnz = -1;
    bool is_csrb = false;
};

struct ChunkCSR {
    // 已加载到主机内存的 CSR 分块，row_ptr 使用 64 位以避免大矩阵溢出。
    int num_rows = 0;
    int angle_offset = 0;
    int angle_count = 0;
    std::vector<int64_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<float> values;
};

struct HostCSR64 {
    // 供单个 GPU worker 使用的完整 CSR 分块视图。
    int64_t num_rows = 0;
    int64_t num_cols = 0;
    int64_t nnz = 0;
    int start_ck = 0;
    int end_ck = 0;
    std::vector<int64_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<float> values;
};

// -------------------- 通用路径与字符串工具 --------------------
static bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

static bool ends_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(s.size()-p.size(), p.size(), p) == 0;
}

static void make_dir(const std::string& d) {
    if (!d.empty()) MKDIR(d.c_str());
}

static bool file_exists(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

static int64_t file_size_bytes(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return -1;
    return (int64_t)st.st_size;
}

static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep)) out.push_back(item);
    return out;
}

static std::string to_lower_copy(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    char last = a.back();
    if (last == '/' || last == '\\') return a + b;
#ifdef _WIN32
    return a + "\\" + b;
#else
    return a + "/" + b;
#endif
}

static std::string path_basename(const std::string& p) {
    size_t pos = p.find_last_of("/\\");
    return (pos == std::string::npos) ? p : p.substr(pos + 1);
}

static bool is_directory(const std::string& path);

static std::string trim_trailing_separators(std::string p) {
    while (p.size() > 1 && (p.back() == '/' || p.back() == '\\')) {
#ifdef _WIN32
        if (p.size() == 3 && p[1] == ':') break;
#endif
        p.pop_back();
    }
    return p;
}

static std::string safe_output_component(std::string s) {
    for (char& c : s) {
        unsigned char uc = (unsigned char)c;
        if (uc < 32 || c == '<' || c == '>' || c == ':' || c == '"' ||
            c == '/' || c == '\\' || c == '|' || c == '?' || c == '*') {
            c = '_';
        }
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '.')) s.back() = '_';
    return s.empty() ? "input" : s;
}

static std::string path_stem(const std::string& path) {
    std::string name = path_basename(trim_trailing_separators(path));
    size_t dot = name.find_last_of('.');
    if (dot != std::string::npos && dot > 0) name.erase(dot);
    return name;
}

static std::string default_output_dir_from_projection(const std::string& proj_path) {
    std::string clean = trim_trailing_separators(proj_path);
    std::string name = path_basename(clean);
    if (!is_directory(clean)) {
        name = path_stem(clean);
        return "recon_" + safe_output_component(name);
    }
    return "recon_multibed_" + safe_output_component(name);
}

static std::string path_parent(const std::string& p) {
    size_t pos = p.find_last_of("/\\");
    if (pos == std::string::npos) return ".";
    if (pos == 0) return p.substr(0, 1);
    return p.substr(0, pos);
}

static bool is_directory(const std::string& path) {
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(path.c_str());
    return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

static std::vector<std::string> list_directory_entries(const std::string& dir,
                                                        bool want_directories)
{
    std::vector<std::string> out;
#ifdef _WIN32
    std::string pattern = join_path(dir, "*");
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return out;
    do {
        std::string name = fd.cFileName;
        if (name == "." || name == "..") continue;
        bool isdir = (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        if (isdir == want_directories) out.push_back(join_path(dir, name));
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR* dp = opendir(dir.c_str());
    if (!dp) return out;
    while (dirent* ent = readdir(dp)) {
        std::string name = ent->d_name;
        if (name == "." || name == "..") continue;
        std::string full = join_path(dir, name);
        if (is_directory(full) == want_directories) out.push_back(full);
    }
    closedir(dp);
#endif
    std::sort(out.begin(), out.end());
    return out;
}

static int trailing_integer_or_max(const std::string& path) {
    std::string name = path_basename(path);
    int end = (int)name.size() - 1;
    while (end >= 0 && std::isspace((unsigned char)name[(size_t)end])) end--;
    int begin = end;
    while (begin >= 0 && std::isdigit((unsigned char)name[(size_t)begin])) begin--;
    begin++;
    if (begin > end) return std::numeric_limits<int>::max();
    return atoi(name.substr((size_t)begin, (size_t)(end - begin + 1)).c_str());
}

static void natural_sort_bed_paths(std::vector<std::string>& paths) {
    std::sort(paths.begin(), paths.end(), [](const std::string& a, const std::string& b) {
        int ia = trailing_integer_or_max(a);
        int ib = trailing_integer_or_max(b);
        if (ia != ib) return ia < ib;
        return to_lower_copy(path_basename(a)) < to_lower_copy(path_basename(b));
    });
}

static bool parse_flat_bed_bin_index(const std::string& path, int* bed_index) {
    std::string name = to_lower_copy(path_basename(path));
    const std::string prefix = "bed";
    const std::string suffix = ".bin";
    if (!starts_with(name, prefix) || !ends_with(name, suffix)) return false;

    size_t begin = prefix.size();
    size_t end = name.size() - suffix.size();
    if (begin >= end) return false;

    int64_t value = 0;
    for (size_t i = begin; i < end; ++i) {
        unsigned char c = (unsigned char)name[i];
        if (!std::isdigit(c)) return false;
        int digit = name[i] - '0';
        if (value > (std::numeric_limits<int>::max() - digit) / 10) return false;
        value = value * 10 + digit;
    }
    if (value <= 0) return false;
    if (bed_index) *bed_index = (int)value;
    return true;
}

static void sort_and_validate_flat_bed_bins(std::vector<std::string>& paths) {
    std::sort(paths.begin(), paths.end(), [](const std::string& a, const std::string& b) {
        int ia = 0, ib = 0;
        parse_flat_bed_bin_index(a, &ia);
        parse_flat_bed_bin_index(b, &ib);
        if (ia != ib) return ia < ib;
        return to_lower_copy(path_basename(a)) < to_lower_copy(path_basename(b));
    });
    for (size_t i = 1; i < paths.size(); ++i) {
        int prev = 0, current = 0;
        parse_flat_bed_bin_index(paths[i - 1], &prev);
        parse_flat_bed_bin_index(paths[i], &current);
        if (prev == current) {
            fprintf(stderr,
                    "Duplicate flat bed index %d: %s and %s\n",
                    current, paths[i - 1].c_str(), paths[i].c_str());
            exit(1);
        }
    }
}

static std::string find_projection_in_bed_dir(const std::string& bed_dir,
                                               const std::string& preferred_name)
{
    std::vector<std::string> candidates;
    if (!preferred_name.empty()) candidates.push_back(join_path(bed_dir, preferred_name));
    candidates.push_back(join_path(bed_dir, "projection.bin"));
    candidates.push_back(join_path(bed_dir, "proj.bin"));
    candidates.push_back(join_path(bed_dir, "data.bin"));

    for (const std::string& p : candidates) {
        if (file_exists(p)) return p;
    }

    std::vector<std::string> bins;
    for (const std::string& p : list_directory_entries(bed_dir, false)) {
        std::string name = to_lower_copy(path_basename(p));
        if (!ends_with(name, ".bin")) continue;
        if (name.find("norm") != std::string::npos ||
            name.find("normalization") != std::string::npos ||
            name == "metadata.bin") continue;
        bins.push_back(p);
    }
    if (bins.size() == 1) return bins[0];
    if (bins.size() > 1) {
        fprintf(stderr,
                "Multiple .bin files found in %s; specify --bed-proj-name explicitly.\n",
                bed_dir.c_str());
        for (const auto& p : bins) fprintf(stderr, "  candidate: %s\n", p.c_str());
        exit(1);
    }
    return "";
}

static std::vector<std::string> discover_projection_paths(const ReconConfig& cfg) {
    if (!is_directory(cfg.proj_path)) {
        if (!file_exists(cfg.proj_path)) {
            fprintf(stderr, "Projection path not found: %s\n", cfg.proj_path.c_str());
            exit(1);
        }
        return {cfg.proj_path};
    }

    std::vector<std::string> flat_bed_bins;
    for (const std::string& p : list_directory_entries(cfg.proj_path, false)) {
        if (parse_flat_bed_bin_index(p, nullptr)) flat_bed_bins.push_back(p);
    }

    std::vector<std::string> subdirs = list_directory_entries(cfg.proj_path, true);
    std::vector<std::string> bed_like;
    for (const std::string& d : subdirs) {
        std::string n = to_lower_copy(path_basename(d));
        if (starts_with(n, "bed")) bed_like.push_back(d);
    }

    if (!flat_bed_bins.empty()) {
        if (!bed_like.empty()) {
            fprintf(stderr,
                    "Ambiguous bed layout under %s: both flat Bed<N>.bin files "
                    "and bed<N> directories were found. Keep only one layout.\n",
                    cfg.proj_path.c_str());
            exit(1);
        }
        sort_and_validate_flat_bed_bins(flat_bed_bins);
        return flat_bed_bins;
    }

    bool explicit_bed_layout = !bed_like.empty();
    if (explicit_bed_layout) subdirs.swap(bed_like);
    natural_sort_bed_paths(subdirs);

    std::vector<std::string> projections;
    for (const std::string& d : subdirs) {
        std::string p = find_projection_in_bed_dir(d, cfg.bed_projection_name);
        if (!p.empty()) {
            projections.push_back(p);
        } else if (explicit_bed_layout) {
            fprintf(stderr,
                    "Bed directory %s does not contain projection file %s.\n",
                    d.c_str(), cfg.bed_projection_name.c_str());
            exit(1);
        }
    }

    if (projections.empty()) {
        std::string direct = find_projection_in_bed_dir(cfg.proj_path,
                                                        cfg.bed_projection_name);
        if (!direct.empty()) projections.push_back(direct);
    }

    if (projections.empty()) {
        fprintf(stderr,
                "No projection files found under %s. Expected flat Bed<N>.bin files "
                "or bed1/bed2/.../%s.\n",
                cfg.proj_path.c_str(), cfg.bed_projection_name.c_str());
        exit(1);
    }
    return projections;
}

// -------------------- 床位发现与几何分配 --------------------
static std::vector<float> parse_float_list(const std::string& s) {
    std::vector<float> out;
    for (const std::string& item : split(s, ',')) {
        if (item.empty()) continue;
        out.push_back((float)atof(item.c_str()));
    }
    return out;
}

static GlobalGeometry assign_bed_geometry(std::vector<BedInfo>& beds,
                                          const ReconConfig& cfg,
                                          const MatrixMeta& meta)
{
    if (beds.empty()) {
        fprintf(stderr, "No beds were discovered.\n");
        exit(1);
    }

    const int B = (int)beds.size();
    if (!cfg.bed_offsets_mm.empty()) {
        if ((int)cfg.bed_offsets_mm.size() != B) {
            fprintf(stderr,
                    "--bed-offsets-mm count mismatch: got %zu values for %d beds.\n",
                    cfg.bed_offsets_mm.size(), B);
            exit(1);
        }
        for (int b = 0; b < B; ++b) beds[b].center_mm = cfg.bed_offsets_mm[(size_t)b];
    } else if (B == 1) {
        beds[0].center_mm = 0.0f;
    } else {
        if (!cfg.bed_step_set || !(cfg.bed_step_mm > 0.0f) ||
            !std::isfinite(cfg.bed_step_mm)) {
            fprintf(stderr,
                    "Multiple beds require --bed-step-mm D or explicit --bed-offsets-mm z1,z2,...\n");
            exit(1);
        }
        float mid = 0.5f * (float)(B - 1);
        for (int b = 0; b < B; ++b) {

            beds[b].center_mm = (mid - (float)b) * cfg.bed_step_mm;
        }
    }

    float min_center = beds[0].center_mm;
    float max_center = beds[0].center_mm;
    for (const BedInfo& bed : beds) {
        min_center = std::min(min_center, bed.center_mm);
        max_center = std::max(max_center, bed.center_mm);
    }

    GlobalGeometry gg;
    gg.Nx = meta.Nx;
    gg.Ny = meta.Ny;
    gg.local_Nz = meta.Nz;
    gg.local_voxels = meta.total_voxels;
    gg.voxel_mm = meta.voxel_size;
    gg.min_bed_center_mm = min_center;
    gg.max_bed_center_mm = max_center;
    const float local_half_center = 0.5f * (float)(meta.Nz - 1);
    if (B == 1) {
        gg.global_Nz = 72;
        gg.origin_z_mm = beds[0].center_mm - 35.5f * meta.voxel_size;
    } else {
        const float span_raw = (max_center - min_center) / meta.voxel_size;
        const float nearest_span = roundf(span_raw);
        const float span_vox = fabsf(span_raw - nearest_span) < 1e-4f
            ? nearest_span : ceilf(span_raw);
        gg.global_Nz = (int)span_vox + 2 * cfg.multibed_edge_layers;
        gg.origin_z_mm = min_center -
            ((float)cfg.multibed_edge_layers - 0.5f) * meta.voxel_size;
    }

    for (BedInfo& bed : beds) {
        const float local_origin_mm =
            bed.center_mm - local_half_center * meta.voxel_size;
        float shift = (local_origin_mm - gg.origin_z_mm) / meta.voxel_size;

        int floor_shift = (int)floorf(shift);
        float frac = shift - (float)floor_shift;

        const float snap_tol_vox = 1e-4f;
        if (frac < snap_tol_vox) frac = 0.0f;
        if (frac > 1.0f - snap_tol_vox) {
            floor_shift += 1;
            frac = 0.0f;
        }
        bed.shift_vox = (float)floor_shift + frac;
        bed.shift_floor = floor_shift;
        bed.shift_frac = frac;
    }
    gg.global_voxels = (int64_t)meta.Nx * meta.Ny * gg.global_Nz;
    if (gg.global_Nz <= 0 || gg.global_voxels <= 0) {
        fprintf(stderr, "Invalid global geometry: Nz=%d voxels=%" PRId64 "\n",
                gg.global_Nz, gg.global_voxels);
        exit(1);
    }
    return gg;
}

static void parse_fwhm3(const std::string& s, float out[3]) {
    auto parts = split(s, ',');
    if (parts.size() == 1) {
        float v = (float)atof(parts[0].c_str());
        out[0] = out[1] = out[2] = v;
    } else if (parts.size() >= 3) {
        out[0] = (float)atof(parts[0].c_str());
        out[1] = (float)atof(parts[1].c_str());
        out[2] = (float)atof(parts[2].c_str());
    }
}

static MapType parse_map_type(const std::string& s) {
    if (s == "none" || s == "off") return MAP_NONE;
    if (s == "quadratic") return MAP_QUADRATIC;
    if (s == "tv" || s == "TV") return MAP_TV;
    if (s == "mrp" || s == "MRP") return MAP_MRP;
    if (s == "huber") return MAP_HUBER;
    if (s == "relative" || s == "relative_difference" || s == "rd") return MAP_RELATIVE;
    if (s == "logcosh") return MAP_LOGCOSH;
    if (s == "hyperbolic") return MAP_HYPERBOLIC;
    fprintf(stderr, "Unknown MAP type: %s\n", s.c_str());
    exit(1);
}

static const char* sym_name(SymMode m) {
    switch (m) {
        case SYM_NONE: return "none";
        case SYM_ZONLY: return "zonly";
        case SYM_180: return "180";
        case SYM_Z180: return "z180";
        case SYM_90: return "90";
        case SYM_45: return "45";
        case SYM_45Z: return "45z";
        default: return "unknown";
    }
}

static SymMode parse_sym_mode(const std::string& s, const MatrixMeta& m) {
    if (s != "auto") {
        if (s == "none") return SYM_NONE;
        if (s == "zonly") return SYM_ZONLY;
        if (s == "180") return SYM_180;
        if (s == "z180") return SYM_Z180;
        if (s == "90") return SYM_90;
        if (s == "45") return SYM_45;
        if (s == "45z" || s == "45deg_z") return SYM_45Z;
        fprintf(stderr, "Unknown --sym %s\n", s.c_str());
        exit(1);
    }

    bool looks_45 = (m.full_num_angles > 0) &&
                    ((m.full_num_angles % 8) == 0) &&
                    (m.compact_num_angles == m.full_num_angles / 8 + 1);
    bool looks_90 = (m.full_num_angles > 0) &&
                    ((m.full_num_angles % 4) == 0) &&
                    (m.compact_num_angles == m.full_num_angles / 4);
    bool looks_180 = (m.full_num_angles > 0) &&
                     ((m.full_num_angles % 2) == 0) &&
                     (m.compact_num_angles == m.full_num_angles / 2);

    if (m.use_45sym || m.sym_code == 45 || m.symmetry_order == 8 || looks_45) {
        return m.use_zsym ? SYM_45Z : SYM_45;
    }
    if (m.use_90sym || looks_90) return SYM_90;
    if (looks_180) return m.use_zsym ? SYM_Z180 : SYM_180;
    if (m.use_zsym)  return SYM_ZONLY;
    return SYM_NONE;
}

static int xy_sym_count(SymMode mode) {
    switch (mode) {
        case SYM_45:
        case SYM_45Z: return 8;
        case SYM_90:  return 4;
        case SYM_180:
        case SYM_Z180:return 2;
        default:      return 1;
    }
}

static bool mode_uses_z(SymMode mode) {
    return mode == SYM_ZONLY || mode == SYM_Z180 || mode == SYM_45Z;
}

// -------------------- 系统矩阵 metadata 与 CSR 读取 --------------------
static const uint32_t CSRB_MAGIC = 0x42525343u;
static const uint32_t CSRB_VERSION = 1u;

static MatrixMeta read_metadata(const std::string& path) {
    MatrixMeta m;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open metadata: %s\n", path.c_str()); exit(1); }

    IO_CHECK(fread(&m.total_rows, 4, 1, f), 1);
    IO_CHECK(fread(&m.total_voxels, 4, 1, f), 1);
    IO_CHECK(fread(&m.max_nnz, 4, 1, f), 1);
    IO_CHECK(fread(&m.compact_num_angles, 4, 1, f), 1);
    IO_CHECK(fread(&m.det_nx, 4, 1, f), 1);
    IO_CHECK(fread(&m.compact_det_z, 4, 1, f), 1);
    IO_CHECK(fread(&m.compact_rpa, 4, 1, f), 1);
    IO_CHECK(fread(&m.full_num_angles, 4, 1, f), 1);
    IO_CHECK(fread(&m.Nx, 4, 1, f), 1);
    IO_CHECK(fread(&m.Ny, 4, 1, f), 1);
    IO_CHECK(fread(&m.Nz, 4, 1, f), 1);
    IO_CHECK(fread(&m.voxel_size, 4, 1, f), 1);
    IO_CHECK(fread(&m.det_nx_dup, 4, 1, f), 1);
    IO_CHECK(fread(&m.det_nz, 4, 1, f), 1);
    IO_CHECK(fread(&m.pixel_size, 4, 1, f), 1);
    IO_CHECK(fread(&m.detector_y, 4, 1, f), 1);
    IO_CHECK(fread(&m.detector_thickness, 4, 1, f), 1);
    IO_CHECK(fread(&m.use_90sym, 4, 1, f), 1);
    IO_CHECK(fread(&m.use_zsym, 4, 1, f), 1);

    IO_CHECK(fread(&m.use_45sym, 4, 1, f), 1);
    IO_CHECK(fread(&m.sym_code, 4, 1, f), 1);
    IO_CHECK(fread(&m.symmetry_order, 4, 1, f), 1);
    fclose(f);

    if (m.det_nz <= 0) {

        m.det_nz = m.use_zsym ? (2 * m.compact_det_z) : m.compact_det_z;
    }
    if (m.compact_rpa <= 0) m.compact_rpa = m.det_nx * m.compact_det_z;
    if (m.total_voxels <= 0) m.total_voxels = m.Nx * m.Ny * m.Nz;
    return m;
}

static std::vector<ChunkInfo> scan_chunks(const std::string& dir) {
    std::vector<std::string> paths_csrb;
    std::vector<std::string> paths_bin;
#ifdef _WIN32
    {
        std::string pattern = dir + "\\chunk_*.csrb";
        WIN32_FIND_DATAA fd;
        HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do { if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) paths_csrb.push_back(dir + "/" + fd.cFileName); }
            while (FindNextFileA(h, &fd));
            FindClose(h);
        }
    }
    {
        std::string pattern = dir + "\\chunk_*.bin";
        WIN32_FIND_DATAA fd;
        HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do { if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) paths_bin.push_back(dir + "/" + fd.cFileName); }
            while (FindNextFileA(h, &fd));
            FindClose(h);
        }
    }
#else
    DIR* d = opendir(dir.c_str());
    if (!d) { fprintf(stderr, "Cannot open matrix dir: %s\n", dir.c_str()); exit(1); }
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        std::string name(ent->d_name);
        if (starts_with(name, "chunk_") && ends_with(name, ".csrb")) paths_csrb.push_back(dir + "/" + name);
        else if (starts_with(name, "chunk_") && ends_with(name, ".bin")) paths_bin.push_back(dir + "/" + name);
    }
    closedir(d);
#endif
    std::sort(paths_csrb.begin(), paths_csrb.end());
    std::sort(paths_bin.begin(), paths_bin.end());

    const bool use_csrb = !paths_csrb.empty();
    const std::vector<std::string>& paths = use_csrb ? paths_csrb : paths_bin;
    if (paths.empty()) { fprintf(stderr, "No chunk_*.csrb or chunk_*.bin found in %s\n", dir.c_str()); exit(1); }

    std::vector<ChunkInfo> infos;
    for (const auto& p : paths) {
        FILE* f = fopen(p.c_str(), "rb");
        if (!f) continue;
        ChunkInfo ci;
        ci.path = p;
        ci.is_csrb = use_csrb;
        if (use_csrb) {
            uint32_t magic = 0, version = 0;
            int64_t nnz = 0;
            IO_CHECK(fread(&magic,   4, 1, f), 1);
            IO_CHECK(fread(&version, 4, 1, f), 1);
            if (magic != CSRB_MAGIC || version != CSRB_VERSION) {
                fprintf(stderr, "Invalid CSRB chunk header: %s\n", p.c_str());
                exit(1);
            }
            IO_CHECK(fread(&ci.num_rows,     4, 1, f), 1);
            IO_CHECK(fread(&ci.angle_offset, 4, 1, f), 1);
            IO_CHECK(fread(&ci.angle_count,  4, 1, f), 1);
            IO_CHECK(fread(&nnz,             8, 1, f), 1);
            ci.nnz = nnz;
        } else {
            IO_CHECK(fread(&ci.num_rows,     4, 1, f), 1);
            IO_CHECK(fread(&ci.angle_offset, 4, 1, f), 1);
            IO_CHECK(fread(&ci.angle_count,  4, 1, f), 1);
            ci.nnz = -1;
        }
        fclose(f);
        infos.push_back(ci);
    }
    std::sort(infos.begin(), infos.end(), [](const ChunkInfo& a, const ChunkInfo& b){
        return a.angle_offset < b.angle_offset;
    });
    return infos;
}

static ChunkCSR load_chunk_csr_legacy_bin(const std::string& path) {
    ChunkCSR csr;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open chunk: %s\n", path.c_str()); exit(1); }

    IO_CHECK(fread(&csr.num_rows,     4, 1, f), 1);
    IO_CHECK(fread(&csr.angle_offset, 4, 1, f), 1);
    IO_CHECK(fread(&csr.angle_count,  4, 1, f), 1);

    long payload_pos = ftell(f);
    csr.row_ptr.resize((size_t)csr.num_rows + 1, 0);
    int64_t total_nnz = 0;
    for (int r = 0; r < csr.num_rows; r++) {
        int32_t n = 0;
        IO_CHECK(fread(&n, 4, 1, f), 1);
        csr.row_ptr[(size_t)r] = total_nnz;
        total_nnz += (int64_t)n;
        if (n > 0) fseek(f, (long)n * 8L, SEEK_CUR);
    }
    csr.row_ptr[(size_t)csr.num_rows] = total_nnz;
    csr.col_idx.resize((size_t)total_nnz);
    csr.values.resize((size_t)total_nnz);

    fseek(f, payload_pos, SEEK_SET);
    for (int r = 0; r < csr.num_rows; r++) {
        int32_t n = 0;
        IO_CHECK(fread(&n, 4, 1, f), 1);
        int64_t off = csr.row_ptr[(size_t)r];
        if (n > 0) {
            IO_CHECK(fread(csr.col_idx.data() + off, 4, n, f), n);
            IO_CHECK(fread(csr.values.data()  + off, 4, n, f), n);
        }
    }
    fclose(f);
    return csr;
}

static ChunkCSR load_chunk_csr_csrb(const std::string& path) {
    ChunkCSR csr;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open CSRB chunk: %s\n", path.c_str()); exit(1); }

    uint32_t magic = 0, version = 0;
    int64_t nnz = 0;
    IO_CHECK(fread(&magic,   4, 1, f), 1);
    IO_CHECK(fread(&version, 4, 1, f), 1);
    if (magic != CSRB_MAGIC || version != CSRB_VERSION) {
        fprintf(stderr, "Invalid CSRB chunk header: %s\n", path.c_str());
        exit(1);
    }
    IO_CHECK(fread(&csr.num_rows,     4, 1, f), 1);
    IO_CHECK(fread(&csr.angle_offset, 4, 1, f), 1);
    IO_CHECK(fread(&csr.angle_count,  4, 1, f), 1);
    IO_CHECK(fread(&nnz,              8, 1, f), 1);
    if (csr.num_rows < 0 || nnz < 0) {
        fprintf(stderr, "Invalid CSRB dimensions in %s\n", path.c_str());
        exit(1);
    }

    csr.row_ptr.resize((size_t)csr.num_rows + 1);
    csr.col_idx.resize((size_t)nnz);
    csr.values.resize((size_t)nnz);
    IO_CHECK(fread(csr.row_ptr.data(), 8, (size_t)csr.num_rows + 1, f), (size_t)csr.num_rows + 1);
    if (nnz > 0) {
        IO_CHECK(fread(csr.col_idx.data(), 4, (size_t)nnz, f), (size_t)nnz);
        IO_CHECK(fread(csr.values.data(),  4, (size_t)nnz, f), (size_t)nnz);
    }
    fclose(f);

    if (csr.row_ptr.empty() || csr.row_ptr.front() != 0 || csr.row_ptr.back() != nnz) {
        fprintf(stderr, "Invalid CSRB row_ptr in %s\n", path.c_str());
        exit(1);
    }
    return csr;
}

static ChunkCSR load_chunk_csr(const ChunkInfo& ci) {
    return ci.is_csrb ? load_chunk_csr_csrb(ci.path) : load_chunk_csr_legacy_bin(ci.path);
}

struct CsrbSlice {
    int64_t source_nnz = 0;
    int64_t source_row0 = 0;
    std::vector<int64_t> row_ptr;
};

static int seek_file_64(FILE* f, int64_t offset) {
#ifdef _WIN32
    return _fseeki64(f, offset, SEEK_SET);
#else
    return fseeko(f, (off_t)offset, SEEK_SET);
#endif
}

static CsrbSlice read_csrb_slice_row_ptr(const ChunkInfo& ci,
                                         int a0, int a1,
                                         int compact_rpa)
{

    FILE* f = fopen(ci.path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Cannot open CSRB chunk: %s\n", ci.path.c_str());
        exit(1);
    }

    uint32_t magic = 0, version = 0;
    int num_rows = 0, angle_offset = 0, angle_count = 0;
    int64_t source_nnz = 0;
    IO_CHECK(fread(&magic,        4, 1, f), 1);
    IO_CHECK(fread(&version,      4, 1, f), 1);
    IO_CHECK(fread(&num_rows,     4, 1, f), 1);
    IO_CHECK(fread(&angle_offset, 4, 1, f), 1);
    IO_CHECK(fread(&angle_count,  4, 1, f), 1);
    IO_CHECK(fread(&source_nnz,   8, 1, f), 1);

    if (magic != CSRB_MAGIC || version != CSRB_VERSION ||
        num_rows != ci.num_rows || angle_offset != ci.angle_offset ||
        angle_count != ci.angle_count || source_nnz != ci.nnz ||
        a0 < angle_offset || a1 > angle_offset + angle_count || a0 >= a1 ||
        num_rows != angle_count * compact_rpa) {
        fprintf(stderr, "Invalid CSRB slab request/header: %s angles=[%d,%d)\n",
                ci.path.c_str(), a0, a1);
        fclose(f);
        exit(1);
    }

    const int64_t source_row0 = (int64_t)(a0 - angle_offset) * compact_rpa;
    const int64_t slab_rows = (int64_t)(a1 - a0) * compact_rpa;
    const int64_t header_bytes = 4 + 4 + 4 + 4 + 4 + 8;
    if (seek_file_64(f, header_bytes + source_row0 * (int64_t)sizeof(int64_t)) != 0) {
        fprintf(stderr, "Cannot seek CSRB row_ptr: %s\n", ci.path.c_str());
        fclose(f);
        exit(1);
    }

    CsrbSlice slab;
    slab.source_nnz = source_nnz;
    slab.source_row0 = source_row0;
    slab.row_ptr.resize((size_t)slab_rows + 1);
    IO_CHECK(fread(slab.row_ptr.data(), sizeof(int64_t),
                   (size_t)slab_rows + 1, f), (size_t)slab_rows + 1);
    fclose(f);

    if (slab.row_ptr.front() < 0 || slab.row_ptr.back() > source_nnz) {
        fprintf(stderr, "Invalid CSRB slab row_ptr: %s angles=[%d,%d)\n",
                ci.path.c_str(), a0, a1);
        exit(1);
    }
    return slab;
}

static void read_csrb_slice_payload(const ChunkInfo& ci,
                                    const CsrbSlice& slab,
                                    int32_t* col_dst, float* val_dst)
{
    const int64_t nnz0 = slab.row_ptr.front();
    const int64_t slab_nnz = slab.row_ptr.back() - nnz0;
    if (slab_nnz <= 0) return;

    FILE* f = fopen(ci.path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Cannot open CSRB chunk: %s\n", ci.path.c_str());
        exit(1);
    }
    const int64_t header_bytes = 4 + 4 + 4 + 4 + 4 + 8;
    const int64_t row_ptr_bytes = ((int64_t)ci.num_rows + 1) * (int64_t)sizeof(int64_t);
    const int64_t col_offset = header_bytes + row_ptr_bytes + nnz0 * (int64_t)sizeof(int32_t);
    const int64_t val_offset = header_bytes + row_ptr_bytes +
                               slab.source_nnz * (int64_t)sizeof(int32_t) +
                               nnz0 * (int64_t)sizeof(float);

    if (seek_file_64(f, col_offset) != 0) {
        fprintf(stderr, "Cannot seek CSRB col_idx: %s\n", ci.path.c_str());
        fclose(f);
        exit(1);
    }
    IO_CHECK(fread(col_dst, sizeof(int32_t), (size_t)slab_nnz, f), (size_t)slab_nnz);
    if (seek_file_64(f, val_offset) != 0) {
        fprintf(stderr, "Cannot seek CSRB values: %s\n", ci.path.c_str());
        fclose(f);
        exit(1);
    }
    IO_CHECK(fread(val_dst, sizeof(float), (size_t)slab_nnz, f), (size_t)slab_nnz);
    fclose(f);
}

static HostCSR64 build_worker_csr(const std::vector<ChunkInfo>& chunks,
                                  int start_ck, int end_ck,
                                  int compact_rpa, int total_voxels)
{

    HostCSR64 out;
    out.start_ck = start_ck;
    out.end_ck = end_ck;
    out.num_rows = (int64_t)(end_ck - start_ck) * compact_rpa;
    out.num_cols = total_voxels;
    out.row_ptr.assign((size_t)out.num_rows + 1, 0);

    std::vector<int64_t> row_counts((size_t)out.num_rows, 0);

    for (const auto& ci : chunks) {
        int ch_start = ci.angle_offset;
        int ch_end = ci.angle_offset + ci.angle_count;
        int a0 = std::max(start_ck, ch_start);
        int a1 = std::min(end_ck, ch_end);
        if (a0 >= a1) continue;

        if (ci.is_csrb) {
            CsrbSlice slab = read_csrb_slice_row_ptr(ci, a0, a1, compact_rpa);
            int worker_row0 = (a0 - start_ck) * compact_rpa;
            int slab_rows = (a1 - a0) * compact_rpa;
            for (int rr = 0; rr < slab_rows; rr++) {
                row_counts[(size_t)worker_row0 + rr] =
                    slab.row_ptr[(size_t)rr + 1] - slab.row_ptr[(size_t)rr];
            }
        } else {
            ChunkCSR ch = load_chunk_csr(ci);
            for (int ck = a0; ck < a1; ck++) {
                int local_angle_in_chunk = ck - ch.angle_offset;
                int chunk_row0 = local_angle_in_chunk * compact_rpa;
                int worker_row0 = (ck - start_ck) * compact_rpa;
                for (int rr = 0; rr < compact_rpa; rr++) {
                    int cr = chunk_row0 + rr;
                    int wr = worker_row0 + rr;
                    row_counts[(size_t)wr] += ch.row_ptr[(size_t)cr + 1] - ch.row_ptr[(size_t)cr];
                }
            }
        }
    }

    int64_t nnz = 0;
    for (int64_t r = 0; r < out.num_rows; r++) {
        out.row_ptr[(size_t)r] = nnz;
        nnz += row_counts[(size_t)r];
    }
    out.row_ptr[(size_t)out.num_rows] = nnz;
    out.nnz = nnz;
    out.col_idx.resize((size_t)nnz);
    out.values.resize((size_t)nnz);

    std::vector<int64_t> write_pos = out.row_ptr;
    for (const auto& ci : chunks) {
        int ch_start = ci.angle_offset;
        int ch_end = ci.angle_offset + ci.angle_count;
        int a0 = std::max(start_ck, ch_start);
        int a1 = std::min(end_ck, ch_end);
        if (a0 >= a1) continue;

        if (ci.is_csrb) {
            CsrbSlice slab = read_csrb_slice_row_ptr(ci, a0, a1, compact_rpa);
            int worker_row0 = (a0 - start_ck) * compact_rpa;
            int slab_rows = (a1 - a0) * compact_rpa;
            int64_t dst0 = out.row_ptr[(size_t)worker_row0];
            read_csrb_slice_payload(ci, slab,
                                    out.col_idx.data() + dst0,
                                    out.values.data() + dst0);
            for (int rr = 0; rr < slab_rows; rr++) {
                write_pos[(size_t)worker_row0 + rr] =
                    out.row_ptr[(size_t)worker_row0 + rr + 1];
            }
        } else {
            ChunkCSR ch = load_chunk_csr(ci);
            for (int ck = a0; ck < a1; ck++) {
                int local_angle_in_chunk = ck - ch.angle_offset;
                int chunk_row0 = local_angle_in_chunk * compact_rpa;
                int worker_row0 = (ck - start_ck) * compact_rpa;
                for (int rr = 0; rr < compact_rpa; rr++) {
                    int cr = chunk_row0 + rr;
                    int wr = worker_row0 + rr;
                    int64_t ps = ch.row_ptr[(size_t)cr];
                    int64_t pe = ch.row_ptr[(size_t)cr + 1];
                    int64_t n = pe - ps;
                    if (n <= 0) continue;
                    int64_t dst = write_pos[(size_t)wr];
                    memcpy(out.col_idx.data() + dst, ch.col_idx.data() + ps, (size_t)n * sizeof(int32_t));
                    memcpy(out.values.data()  + dst, ch.values.data()  + ps, (size_t)n * sizeof(float));
                    write_pos[(size_t)wr] += n;
                }
            }
        }
    }
    return out;
}

struct TimedHostCSR {
    HostCSR64 csr;
    double build_sec = 0.0;
};

// 构造以矩阵几何中心为原点的圆柱 ROI。体素线性索引固定为
// col = z*(Nx*Ny) + y*Nx + x（0-based，x 最快变化），与 CSR 列号一致。
// mask[col]=1 表示该体素及指向该列的系统矩阵非零元参与重建。
static std::vector<uint8_t> build_centered_cylinder_roi_mask(
    int Nx, int Ny, int Nz, float voxel_mm,
    float diameter_mm, float height_mm, int64_t& active_voxels)
{
    const int64_t total_voxels = (int64_t)Nx * Ny * Nz;
    std::vector<uint8_t> mask((size_t)total_voxels, 0);
    // 使用与 CUDA ROI kernel 相同的 float 运算，避免恰好位于边界的体素在
    // Host CSR 筛选和 Device 状态清零之间产生精度分歧。
    const float radius = 0.5f * diameter_mm;
    const float radius2 = radius * radius;
    const float half_height = 0.5f * height_mm;
    active_voxels = 0;
    for (int z = 0; z < Nz; ++z) {
        const float zz = ((float)z - 0.5f * (float)(Nz - 1)) * voxel_mm;
        if (fabsf(zz) > half_height) continue;
        for (int y = 0; y < Ny; ++y) {
            const float yy = ((float)y - 0.5f * (float)(Ny - 1)) * voxel_mm;
            for (int x = 0; x < Nx; ++x) {
                const float xx = ((float)x - 0.5f * (float)(Nx - 1)) * voxel_mm;
                if (xx * xx + yy * yy > radius2) continue;
                const int64_t col = (int64_t)z * Nx * Ny + (int64_t)y * Nx + x;
                mask[(size_t)col] = 1;
                ++active_voxels;
            }
        }
    }
    return mask;
}

// 原地删除 CSR 中指向 ROI 外体素的非零元。num_cols 保持完整矩阵体素数，
// 因而图像缓冲、对称变换和输出索引均不改变；只有实际 SpMV 的 nnz 被裁剪。
static int64_t prune_host_csr_to_roi(HostCSR64& csr,
                                     const std::vector<uint8_t>& roi_mask)
{
    if ((int64_t)roi_mask.size() != csr.num_cols) {
        throw std::runtime_error("ROI mask length does not match CSR column count");
    }
    const int64_t original_nnz = csr.nnz;
    int64_t write = 0;
    int64_t old_begin = 0;
    for (int64_t row = 0; row < csr.num_rows; ++row) {
        const int64_t old_end = csr.row_ptr[(size_t)row + 1];
        csr.row_ptr[(size_t)row] = write;
        for (int64_t p = old_begin; p < old_end; ++p) {
            const int32_t col = csr.col_idx[(size_t)p];
            if (col < 0 || (int64_t)col >= csr.num_cols) {
                throw std::runtime_error("system-matrix CSR column index is out of range");
            }
            if (roi_mask[(size_t)col] == 0) continue;
            csr.col_idx[(size_t)write] = col;
            csr.values[(size_t)write] = csr.values[(size_t)p];
            ++write;
        }
        old_begin = old_end;
    }
    csr.row_ptr[(size_t)csr.num_rows] = write;
    csr.nnz = write;
    csr.col_idx.resize((size_t)write);
    csr.values.resize((size_t)write);
    return original_nnz - write;
}

static TimedHostCSR build_worker_csr_timed(
    const std::vector<ChunkInfo>& chunks,
    int start_ck, int end_ck,
    int compact_rpa, int total_voxels)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    TimedHostCSR out;
    out.csr = build_worker_csr(chunks, start_ck, end_ck,
                               compact_rpa, total_voxels);
    out.build_sec = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    return out;
}

static std::vector<double> read_projection_bin_double(const std::string& path, int* na, int* rpa) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open projection: %s\n", path.c_str()); exit(1); }
    IO_CHECK(fread(na, 4, 1, f), 1);
    IO_CHECK(fread(rpa, 4, 1, f), 1);
    int64_t total = (int64_t)(*na) * (*rpa);
    if (*na <= 0 || *rpa <= 0 || total <= 0) {
        fprintf(stderr, "Invalid projection header in %s: na=%d rpa=%d\n",
                path.c_str(), *na, *rpa);
        exit(1);
    }

    int64_t size = file_size_bytes(path);
    if (size < 8) {
        fprintf(stderr, "Projection file is too small: %s\n", path.c_str());
        exit(1);
    }
    int64_t payload = size - 8;

    fclose(f);
    std::vector<double> out((size_t)total);
    f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot reopen projection: %s\n", path.c_str()); exit(1); }
    fseek(f, 8, SEEK_SET);
    if (payload == total * (int64_t)sizeof(double)) {
        IO_CHECK(fread(out.data(), sizeof(double), (size_t)total, f), (size_t)total);
        printf("  read %s as float64 payload\n", path.c_str());
    } else if (payload == total * (int64_t)sizeof(float)) {
        std::vector<float> tmp((size_t)total);
        IO_CHECK(fread(tmp.data(), sizeof(float), (size_t)total, f), (size_t)total);
        for (int64_t i = 0; i < total; i++) out[(size_t)i] = (double)tmp[(size_t)i];
        printf("  read %s as float32 payload\n", path.c_str());
    } else {
        fprintf(stderr,
                "Projection payload size mismatch in %s: payload=%" PRId64
                ", expected %" PRId64 " bytes for float32 or %" PRId64 " bytes for float64\n",
                path.c_str(), payload,
                total * (int64_t)sizeof(float),
                total * (int64_t)sizeof(double));
        exit(1);
    }
    fclose(f);
    return out;
}

// -------------------- 投影读取、归一化和坏探测器掩码 --------------------
static constexpr int BAD_DET_ANGLE_BEGIN = 0;
static constexpr int BAD_DET_ANGLE_END   = 60;
static constexpr int BAD_DET_Z_BEGIN     = 24;
static constexpr int BAD_DET_Z_END       = 36;
static constexpr int BAD_DET_X_BEGIN     = 24;
static constexpr int BAD_DET_X_END       = 36;

struct BadDetectorMaskStats {
    int64_t bins = 0;
    int64_t nonzero_bins = 0;
    double removed_sum = 0.0;
};

// Poisson MLEM 要求测量计数有限且非负。在转换为 float 前校验，防止异常仿真
// 文件把 NaN、Inf 或负的比值分子静默传入 cuSPARSE。
static void validate_projection_counts(const std::vector<double>& data,
                                       const std::string& path) {
    int64_t nonfinite = 0;
    int64_t negative = 0;
    for (double value : data) {
        if (!std::isfinite(value)) ++nonfinite;
        else if (value < 0.0) ++negative;
    }
    if (nonfinite != 0 || negative != 0) {
        fprintf(stderr,
                "Invalid projection counts in %s: nonfinite=%" PRId64
                " negative=%" PRId64 "; expected finite non-negative bins.\n",
                path.c_str(), nonfinite, negative);
        exit(1);
    }
}

template <typename T>
static BadDetectorMaskStats zero_link4_fpga2_det4(
    std::vector<T>& data, int full_num_angles, int det_nx, int det_nz)
{
    if (full_num_angles < BAD_DET_ANGLE_END ||
        det_nx < BAD_DET_X_END || det_nz < BAD_DET_Z_END) {
        fprintf(stderr,
                "Cannot apply LINK4/FPGA2/DET4 mask: projection geometry "
                "angles=%d det=%dx%d does not contain angles 0..59, z 24..35, x 24..35\n",
                full_num_angles, det_nx, det_nz);
        exit(1);
    }
    const int64_t expected = (int64_t)full_num_angles * det_nz * det_nx;
    if ((int64_t)data.size() != expected) {
        fprintf(stderr,
                "Cannot apply LINK4/FPGA2/DET4 mask: data length=%zu expected=%" PRId64 "\n",
                data.size(), expected);
        exit(1);
    }

    BadDetectorMaskStats stats;
    const int full_rpa = det_nx * det_nz;
    for (int angle = BAD_DET_ANGLE_BEGIN; angle < BAD_DET_ANGLE_END; ++angle) {
        for (int iz = BAD_DET_Z_BEGIN; iz < BAD_DET_Z_END; ++iz) {
            for (int ix = BAD_DET_X_BEGIN; ix < BAD_DET_X_END; ++ix) {
                const int64_t idx = (int64_t)angle * full_rpa + iz * det_nx + ix;
                const double old_value = (double)data[(size_t)idx];
                stats.bins++;
                if (old_value != 0.0) stats.nonzero_bins++;
                stats.removed_sum += old_value;
                data[(size_t)idx] = (T)0;
            }
        }
    }
    return stats;
}

static std::string find_normalization_factor(const ReconConfig& cfg) {
    if (!cfg.norm_factor_enable) return "";
    if (!cfg.norm_factor_path.empty()) {
        if (!file_exists(cfg.norm_factor_path)) {
            fprintf(stderr, "Normalization factor file not found: %s\n",
                    cfg.norm_factor_path.c_str());
            exit(1);
        }
        return cfg.norm_factor_path;
    }

    std::vector<std::string> candidates;
    candidates.push_back(cfg.matrix_dir + "/normalization_factors/normalization_factor.bin");
    candidates.push_back(cfg.matrix_dir + "/normalization_factors/norm_factor.bin");
    candidates.push_back("normalization_factors/normalization_factor.bin");
    candidates.push_back("normalization_factors/norm_factor.bin");
    for (const auto& p : candidates) {
        if (file_exists(p)) return p;
    }
    return "";
}

static std::string resolve_normalization_factor_mode(const ReconConfig& cfg,
                                                     const std::string& factor_path)
{

    if (cfg.norm_factor_invert) {
        if (cfg.norm_factor_mode != "auto" && cfg.norm_factor_mode != "projection") {
            fprintf(stderr,
                    "--norm-factor-invert conflicts with --norm-factor-mode %s\n",
                    cfg.norm_factor_mode.c_str());
            exit(1);
        }
        return "projection";
    }

    if (cfg.norm_factor_mode == "matrix" || cfg.norm_factor_mode == "projection") {
        return cfg.norm_factor_mode;
    }
    if (cfg.norm_factor_mode != "auto") {
        fprintf(stderr,
                "Invalid --norm-factor-mode %s; expected auto, matrix, or projection.\n",
                cfg.norm_factor_mode.c_str());
        exit(1);
    }

    std::string mode_path = factor_path + ".mode";
    if (file_exists(mode_path)) {
        FILE* f = fopen(mode_path.c_str(), "r");
        if (!f) {
            fprintf(stderr, "Cannot open normalization factor mode file: %s\n",
                    mode_path.c_str());
            exit(1);
        }
        char mode_buf[64] = {0};
        int read_count = fscanf(f, "%63s", mode_buf);
        fclose(f);
        std::string mode = read_count == 1 ? std::string(mode_buf) : std::string();
        if (mode != "matrix" && mode != "projection") {
            fprintf(stderr,
                    "Invalid normalization factor mode in %s: '%s'; expected matrix or projection.\n",
                    mode_path.c_str(), mode.c_str());
            exit(1);
        }
        printf("[2b] Normalization factor mode auto-detected from %s: %s\n",
               mode_path.c_str(), mode.c_str());
        return mode;
    }

    fprintf(stderr,
            "Warning: normalization factor mode file not found: %s; "
            "assuming legacy matrix mode. Use --norm-factor-mode projection for "
            "a projection-precorrection factor.\n",
            mode_path.c_str());
    return "matrix";
}

static std::vector<float> read_system_response_factor(int proj_na,
                                                     int proj_rpa,
                                                     const std::string& factor_path,
                                                     const std::string& factor_mode)
{

    bool invert_factor = (factor_mode == "projection");
    int fac_na = 0, fac_rpa = 0;
    std::vector<double> factor = read_projection_bin_double(factor_path, &fac_na, &fac_rpa);
    if (fac_na != proj_na || fac_rpa != proj_rpa) {
        fprintf(stderr,
                "Normalization factor shape mismatch: factor=(%d,%d), projection=(%d,%d)\n",
                fac_na, fac_rpa, proj_na, proj_rpa);
        exit(1);
    }

    std::vector<float> g(factor.size(), 1.0f);
    int64_t bad_nonfinite = 0;
    int64_t bad_negative = 0;
    int64_t bad_invert = 0;
    int64_t zero_count = 0;
    double sum_g = 0.0;
    double min_g = std::numeric_limits<double>::infinity();
    double max_g = 0.0;

    for (size_t i = 0; i < factor.size(); i++) {
        double f = factor[i];
        if (!std::isfinite(f)) {
            bad_nonfinite++;
            continue;
        }

        double gi = f;
        if (invert_factor) {
            if (f <= 0.0) {
                bad_invert++;
                continue;
            }
            gi = 1.0 / f;
        } else {
            if (f < 0.0) {
                bad_negative++;
                continue;
            }
        }

        if (!std::isfinite(gi) || gi < 0.0) {
            bad_nonfinite++;
            continue;
        }
        if (gi == 0.0) zero_count++;
        g[i] = (float)gi;
        sum_g += gi;
        if (gi < min_g) min_g = gi;
        if (gi > max_g) max_g = gi;
    }

    if (bad_nonfinite > 0 || bad_negative > 0 || bad_invert > 0) {
        fprintf(stderr,
                "Invalid normalization/system response factor %s: nonfinite=%" PRId64
                " negative=%" PRId64 " nonpositive_for_invert=%" PRId64 "\n",
                factor_path.c_str(), bad_nonfinite, bad_negative, bad_invert);
        exit(1);
    }

    if (factor.empty()) {
        fprintf(stderr, "Normalization factor is empty: %s\n", factor_path.c_str());
        exit(1);
    }

    printf("[2b] System-matrix normalization factor loaded: %s\n", factor_path.c_str());
    printf("     model: W_ij = g_i * A_ij^ideal, factor mode=%s, g source=%s\n",
           factor_mode.c_str(), invert_factor ? "1 / factor_file" : "factor_file");
    printf("     g mean=%.6e min=%.6e max=%.6e zero_bins=%" PRId64 "\n",
           sum_g / (double)g.size(), min_g, max_g, zero_count);
    return g;
}

// -------------------- 底层重建结果输出与诊断 --------------------
static void put_le16(unsigned char* p, uint16_t v) {
    p[0] = (unsigned char)(v & 0xffu);
    p[1] = (unsigned char)((v >> 8) & 0xffu);
}

static void put_le32(unsigned char* p, uint32_t v) {
    p[0] = (unsigned char)(v & 0xffu);
    p[1] = (unsigned char)((v >> 8) & 0xffu);
    p[2] = (unsigned char)((v >> 16) & 0xffu);
    p[3] = (unsigned char)((v >> 24) & 0xffu);
}

static void put_le_float(unsigned char* p, float v) {
    uint32_t u = 0;
    static_assert(sizeof(float) == sizeof(uint32_t), "float32 required");
    memcpy(&u, &v, sizeof(u));
    put_le32(p, u);
}

static bool host_is_little_endian() {
    const uint16_t one = 1;
    return *((const unsigned char*)&one) == 1;
}

static std::string remove_known_volume_extension(const std::string& path) {
    const char* exts[] = {".hdr", ".img", ".raw"};
    for (const char* ext : exts) {
        const size_t n = strlen(ext);
        if (path.size() >= n && path.compare(path.size() - n, n, ext) == 0) {
            return path.substr(0, path.size() - n);
        }
    }
    return path;
}

static void write_float64_le(FILE* f, const float* data, size_t n) {
    const size_t block = 1u << 17;
    std::vector<double> converted(block);
    std::vector<unsigned char> bytes;
    if (!host_is_little_endian()) bytes.resize(block * sizeof(double));

    for (size_t off = 0; off < n; off += block) {
        const size_t count = std::min(block, n - off);
        for (size_t i = 0; i < count; ++i) converted[i] = (double)data[off + i];
        if (host_is_little_endian()) {
            IO_CHECK(fwrite(converted.data(), sizeof(double), count, f), count);
        } else {
            for (size_t i = 0; i < count; ++i) {
                uint64_t u = 0;
                memcpy(&u, converted.data() + i, sizeof(u));
                for (int b = 0; b < 8; ++b) {
                    bytes[8*i + (size_t)b] = (unsigned char)((u >> (8*b)) & 0xffu);
                }
            }
            IO_CHECK(fwrite(bytes.data(), sizeof(double), count, f), count);
        }
    }
}

static void save_volume_nifti1_pair(const std::string& base_path,
                                    const float* data,
                                    int Nx, int Ny, int Nz,
                                    float voxel_size,
                                    float origin_z_mm)
{
    const std::string base = remove_known_volume_extension(base_path);
    const std::string hdr_path = base + ".hdr";
    const std::string img_path = base + ".img";

    if (Nx <= 0 || Ny <= 0 || Nz <= 0 ||
        Nx > 32767 || Ny > 32767 || Nz > 32767 ||
        !(voxel_size > 0.0f) || !std::isfinite(voxel_size) ||
        !std::isfinite(origin_z_mm)) {
        fprintf(stderr,
                "Invalid NIfTI volume geometry: %d x %d x %d, voxel=%g mm, origin_z=%g mm\n",
                Nx, Ny, Nz, voxel_size, origin_z_mm);
        exit(1);
    }

    const size_t n = (size_t)Nx * Ny * Nz;
    float data_min = std::numeric_limits<float>::infinity();
    float data_max = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(data[i])) {
            const size_t plane = (size_t)Nx * (size_t)Ny;
            const int z = (int)(i / plane);
            const size_t rem = i - (size_t)z * plane;
            const int y = (int)(rem / (size_t)Nx);
            const int x = (int)(rem - (size_t)y * (size_t)Nx);
            const double x_mm = -0.5 * (double)(Nx - 1) * voxel_size + x * voxel_size;
            const double y_mm = -0.5 * (double)(Ny - 1) * voxel_size + y * voxel_size;
            const double z_mm = origin_z_mm + z * voxel_size;
            fprintf(stderr,
                    "Cannot save non-finite NIfTI voxel at index %zu (x=%d,y=%d,z=%d; %.6f,%.6f,%.6f mm)\n",
                    i, x, y, z, x_mm, y_mm, z_mm);
            exit(1);
        }
        data_min = std::min(data_min, data[i]);
        data_max = std::max(data_max, data[i]);
    }

    unsigned char hdr[348];
    memset(hdr, 0, sizeof(hdr));
    put_le32(hdr + 0, 348u);
    hdr[38] = 'r';
    put_le16(hdr + 40, 3);
    put_le16(hdr + 42, (uint16_t)Nx);
    put_le16(hdr + 44, (uint16_t)Ny);
    put_le16(hdr + 46, (uint16_t)Nz);
    put_le16(hdr + 48, 1);
    put_le16(hdr + 70, 64);
    put_le16(hdr + 72, 64);
    put_le_float(hdr + 76, 1.0f);
    put_le_float(hdr + 80, voxel_size);
    put_le_float(hdr + 84, voxel_size);
    put_le_float(hdr + 88, voxel_size);
    put_le_float(hdr + 92, 1.0f);
    put_le_float(hdr + 108, 0.0f);
    put_le_float(hdr + 112, 1.0f);
    put_le_float(hdr + 116, 0.0f);
    hdr[123] = 2;
    put_le_float(hdr + 124, data_max);
    put_le_float(hdr + 128, data_min);

    const char* desc = "SPECT multi-bed reconstruction float64; x-fastest";
    memcpy(hdr + 148, desc, std::min((size_t)79, strlen(desc)));
    put_le16(hdr + 252, 0);
    put_le16(hdr + 254, 1);
    const float x0 = -0.5f * (float)(Nx - 1) * voxel_size;
    const float y0 = -0.5f * (float)(Ny - 1) * voxel_size;
    put_le_float(hdr + 280, voxel_size); put_le_float(hdr + 292, x0);
    put_le_float(hdr + 300, voxel_size); put_le_float(hdr + 308, y0);
    put_le_float(hdr + 320, voxel_size); put_le_float(hdr + 324, origin_z_mm);
    memcpy(hdr + 344, "ni1\0", 4);

    FILE* hf = fopen(hdr_path.c_str(), "wb");
    if (!hf) { fprintf(stderr, "Cannot write %s\n", hdr_path.c_str()); exit(1); }
    IO_CHECK(fwrite(hdr, 1, sizeof(hdr), hf), sizeof(hdr));
    fclose(hf);

    FILE* imf = fopen(img_path.c_str(), "wb");
    if (!imf) { fprintf(stderr, "Cannot write %s\n", img_path.c_str()); exit(1); }
    write_float64_le(imf, data, n);
    fclose(imf);
}

static void save_volume_raw1(const std::string& path, const float* data,
                             int Nx, int Ny, int Nz, float voxel_size)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); exit(1); }
    int magic = 0x52415731;
    IO_CHECK(fwrite(&Nx, 4, 1, f), 1);
    IO_CHECK(fwrite(&Ny, 4, 1, f), 1);
    IO_CHECK(fwrite(&Nz, 4, 1, f), 1);
    IO_CHECK(fwrite(&magic, 4, 1, f), 1);
    IO_CHECK(fwrite(&voxel_size, 4, 1, f), 1);
    const size_t n = (size_t)Nx * Ny * Nz;
    IO_CHECK(fwrite(data, sizeof(float), n, f), n);
    fclose(f);
}

static void save_volume_outputs(const std::string& base_path,
                                const float* data,
                                int Nx, int Ny, int Nz,
                                float voxel_size,
                                float origin_z_mm,
                                bool save_raw1)
{
    const std::string base = remove_known_volume_extension(base_path);
    save_volume_nifti1_pair(base, data, Nx, Ny, Nz, voxel_size, origin_z_mm);
    if (save_raw1) {
        save_volume_raw1(base + ".raw", data, Nx, Ny, Nz, voxel_size);
    }
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                    out += buf;
                } else {
                    out.push_back((char)c);
                }
        }
    }
    return out;
}

static void save_multibed_geometry_json(const std::string& path,
                                        const GlobalGeometry& gg,
                                        const MatrixMeta& meta,
                                        const ReconConfig& cfg,
                                        const std::vector<BedInfo>& beds)
{
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "Cannot write geometry JSON: %s\n", path.c_str());
        return;
    }
    fprintf(f, "{\n");
    fprintf(f, "  \"Nx\": %d,\n", gg.Nx);
    fprintf(f, "  \"Ny\": %d,\n", gg.Ny);
    fprintf(f, "  \"Nz\": %d,\n", gg.global_Nz);
    fprintf(f, "  \"voxel_size_mm\": %.9g,\n", gg.voxel_mm);
    fprintf(f, "  \"origin_x_mm\": %.9g,\n",
            -0.5f * (float)(gg.Nx - 1) * gg.voxel_mm);
    fprintf(f, "  \"origin_y_mm\": %.9g,\n",
            -0.5f * (float)(gg.Ny - 1) * gg.voxel_mm);
    fprintf(f, "  \"origin_z_mm\": %.9g,\n", gg.origin_z_mm);
    fprintf(f, "  \"valid_z_min_mm\": %.9g,\n",
            gg.min_bed_center_mm - 0.5f * cfg.roi_height_mm);
    fprintf(f, "  \"valid_z_max_mm\": %.9g,\n",
            gg.max_bed_center_mm + 0.5f * cfg.roi_height_mm);
    fprintf(f, "  \"local_matrix_Nz\": %d,\n", meta.Nz);
    fprintf(f, "  \"local_matrix_length_mm\": %.9g,\n", meta.Nz * meta.voxel_size);
    fprintf(f, "  \"local_valid_fov_mm\": %.9g,\n", cfg.roi_height_mm);
    fprintf(f, "  \"normalization_shared_across_beds\": true,\n");
    fprintf(f, "  \"beds\": [\n");
    for (size_t i = 0; i < beds.size(); ++i) {
        const BedInfo& bed = beds[i];
        fprintf(f,
                "    {\"name\": \"%s\", \"projection\": \"%s\", "
                "\"center_mm\": %.9g, \"shift_vox\": %.9g, "
                "\"shift_floor\": %d, \"shift_fraction\": %.9g, "
                "\"raw_counts\": %.17g}%s\n",
                json_escape(bed.name).c_str(),
                json_escape(bed.projection_path).c_str(),
                bed.center_mm, bed.shift_vox, bed.shift_floor, bed.shift_frac,
                bed.raw_counts, (i + 1 == beds.size()) ? "" : ",");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
}

static void append_be32(std::vector<unsigned char>& out, uint32_t v) {
    out.push_back((unsigned char)((v >> 24) & 255u));
    out.push_back((unsigned char)((v >> 16) & 255u));
    out.push_back((unsigned char)((v >> 8) & 255u));
    out.push_back((unsigned char)(v & 255u));
}

static uint32_t crc32_bytes(const unsigned char* data, size_t n) {
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < n; i++) {
        c ^= (uint32_t)data[i];
        for (int k = 0; k < 8; k++) {
            c = (c & 1u) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
        }
    }
    return c ^ 0xffffffffu;
}

static uint32_t adler32_bytes(const unsigned char* data, size_t n) {
    const uint32_t MOD = 65521u;
    uint32_t a = 1u, b = 0u;
    for (size_t i = 0; i < n; i++) {
        a += (uint32_t)data[i];
        if (a >= MOD) a -= MOD;
        b += a;
        if (b >= MOD) b %= MOD;
    }
    return (b << 16) | a;
}

static void append_png_chunk(std::vector<unsigned char>& png,
                             const char type[4],
                             const std::vector<unsigned char>& data)
{
    append_be32(png, (uint32_t)data.size());
    size_t type_pos = png.size();
    png.push_back((unsigned char)type[0]);
    png.push_back((unsigned char)type[1]);
    png.push_back((unsigned char)type[2]);
    png.push_back((unsigned char)type[3]);
    png.insert(png.end(), data.begin(), data.end());
    uint32_t crc = crc32_bytes(png.data() + type_pos, 4 + data.size());
    append_be32(png, crc);
}

static bool save_png_gray8(const std::string& path,
                           int width,
                           int height,
                           const std::vector<unsigned char>& gray)
{
    if (width <= 0 || height <= 0 || gray.size() != (size_t)width * height) {
        fprintf(stderr, "Invalid PNG image dimensions for %s\n", path.c_str());
        return false;
    }

    std::vector<unsigned char> filtered;
    filtered.reserve((size_t)height * ((size_t)width + 1));
    for (int y = 0; y < height; y++) {
        filtered.push_back(0);
        const unsigned char* row = gray.data() + (size_t)y * width;
        filtered.insert(filtered.end(), row, row + width);
    }

    std::vector<unsigned char> z;
    z.reserve(filtered.size() + filtered.size() / 65535u * 5u + 16u);
    z.push_back(0x78); z.push_back(0x01);
    size_t pos = 0;
    while (pos < filtered.size()) {
        size_t n = std::min((size_t)65535, filtered.size() - pos);
        bool final_block = (pos + n == filtered.size());
        z.push_back(final_block ? 0x01 : 0x00);
        uint16_t len = (uint16_t)n;
        uint16_t nlen = (uint16_t)~len;
        z.push_back((unsigned char)(len & 255u));
        z.push_back((unsigned char)((len >> 8) & 255u));
        z.push_back((unsigned char)(nlen & 255u));
        z.push_back((unsigned char)((nlen >> 8) & 255u));
        z.insert(z.end(), filtered.begin() + pos, filtered.begin() + pos + n);
        pos += n;
    }
    append_be32(z, adler32_bytes(filtered.data(), filtered.size()));

    std::vector<unsigned char> png;
    const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    png.insert(png.end(), sig, sig + 8);

    std::vector<unsigned char> ihdr;
    append_be32(ihdr, (uint32_t)width);
    append_be32(ihdr, (uint32_t)height);
    ihdr.push_back(8);
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    append_png_chunk(png, "IHDR", ihdr);
    append_png_chunk(png, "IDAT", z);
    std::vector<unsigned char> empty;
    append_png_chunk(png, "IEND", empty);

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Cannot write PNG %s\n", path.c_str());
        return false;
    }
    bool ok = fwrite(png.data(), 1, png.size(), f) == png.size();
    fclose(f);
    if (!ok) fprintf(stderr, "Short write while saving PNG %s\n", path.c_str());
    return ok;
}

static void save_sensitivity_edge_pngs(const float* a,
                                       int Nx, int Ny, int Nz,
                                       const std::string& out_dir,
                                       const char* prefix)
{
    if (Nz <= 0) return;
    int z0 = 0;
    int z1 = Nz - 1;
    int64_t nxy = (int64_t)Nx * Ny;

    double vmin = std::numeric_limits<double>::infinity();
    double vmax = -std::numeric_limits<double>::infinity();
    for (int pass = 0; pass < 2; pass++) {
        int z = pass == 0 ? z0 : z1;
        const float* sl = a + (int64_t)z * nxy;
        for (int64_t i = 0; i < nxy; i++) {
            double v = (double)sl[i];
            if (!std::isfinite(v)) continue;
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
    }
    if (!(vmax > vmin)) {
        vmin = 0.0;
        vmax = 1.0;
    }

    auto make_image = [&](int z, std::vector<unsigned char>& img) {
        img.assign((size_t)nxy, 0);
        const float* sl = a + (int64_t)z * nxy;
        double inv = 255.0 / (vmax - vmin);
        for (int iy = 0; iy < Ny; iy++) {
            for (int ix = 0; ix < Nx; ix++) {
                int64_t idx = (int64_t)iy * Nx + ix;
                double v = (double)sl[idx];
                double t = std::isfinite(v) ? (v - vmin) * inv : 0.0;
                if (t < 0.0) t = 0.0;
                if (t > 255.0) t = 255.0;
                img[(size_t)idx] = (unsigned char)(t + 0.5);
            }
        }
    };

    std::vector<unsigned char> img;
    char fn0[1024], fn1[1024];
    snprintf(fn0, sizeof(fn0), "%s/%s_z%03d.png", out_dir.c_str(), prefix, z0);
    snprintf(fn1, sizeof(fn1), "%s/%s_z%03d.png", out_dir.c_str(), prefix, z1);
    make_image(z0, img);
    bool ok0 = save_png_gray8(fn0, Nx, Ny, img);
    make_image(z1, img);
    bool ok1 = save_png_gray8(fn1, Nx, Ny, img);
    if (ok0 && ok1) {
        printf("[8c] Saved sensitivity edge PNGs (%s): %s, %s, shared scale=[%.6e, %.6e]\n",
               prefix, fn0, fn1, vmin, vmax);
    }
}

// -------------------- CUDA 图像、ROI 和 MLEM kernels --------------------
__global__ void k_fill(float* a, float v, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = v;
}

__global__ void k_zero(float* a, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0.0f;
}

__global__ void k_add_inplace(float* dst, const float* src, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

__global__ void k_clamp_min(float* a, float v, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && a[i] < v) a[i] = v;
}

__global__ void k_clamp_pos(float* a, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && a[i] < 0.0f) a[i] = 0.0f;
}

__global__ void k_gather_bed_linear(const float* __restrict__ global_img,
                                    float* __restrict__ local_img,
                                    int Nx, int Ny,
                                    int local_Nz, int global_Nz,
                                    int shift_floor, float shift_frac,
                                    int64_t local_n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= local_n) return;

    int plane = Nx * Ny;
    int iz = (int)(idx / plane);
    int rem = (int)(idx - (int64_t)iz * plane);
    int gz0 = iz + shift_floor;
    int gz1 = gz0 + 1;

    float v0 = 0.0f, v1 = 0.0f;
    if (gz0 >= 0 && gz0 < global_Nz) {
        v0 = global_img[(int64_t)gz0 * plane + rem];
    }
    if (shift_frac > 0.0f && gz1 >= 0 && gz1 < global_Nz) {
        v1 = global_img[(int64_t)gz1 * plane + rem];
    }
    local_img[idx] = (1.0f - shift_frac) * v0 + shift_frac * v1;
}

__global__ void k_scatter_bed_linear_add(const float* __restrict__ local_img,
                                         float* __restrict__ global_img,
                                         int Nx, int Ny,
                                         int local_Nz, int global_Nz,
                                         int shift_floor, float shift_frac,
                                         int64_t local_n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= local_n) return;

    int plane = Nx * Ny;
    int iz = (int)(idx / plane);
    int rem = (int)(idx - (int64_t)iz * plane);
    int gz0 = iz + shift_floor;
    int gz1 = gz0 + 1;
    float v = local_img[idx];

    if (shift_frac == 0.0f) {
        if (gz0 >= 0 && gz0 < global_Nz) {
            global_img[(int64_t)gz0 * plane + rem] += v;
        }
        return;
    }

    if (gz0 >= 0 && gz0 < global_Nz) {
        atomicAdd(&global_img[(int64_t)gz0 * plane + rem],
                  (1.0f - shift_frac) * v);
    }
    if (shift_frac > 0.0f && gz1 >= 0 && gz1 < global_Nz) {
        atomicAdd(&global_img[(int64_t)gz1 * plane + rem],
                  shift_frac * v);
    }
}

__global__ void k_apply_multibed_roi(float* a,
                                     int Nx, int Ny, int global_Nz,
                                     float voxel_mm, float origin_z_mm,
                                     float diameter_mm, float valid_height_mm,
                                     float center_x_mm, float center_y_mm,
                                     const float* __restrict__ bed_centers_mm,
                                     int num_beds, int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int plane = Nx * Ny;
    int iz = (int)(idx / plane);
    int rem = (int)(idx - (int64_t)iz * plane);
    int iy = rem / Nx;
    int ix = rem - iy * Nx;

    float x = ((float)ix - 0.5f * (float)(Nx - 1)) * voxel_mm - center_x_mm;
    float y = ((float)iy - 0.5f * (float)(Ny - 1)) * voxel_mm - center_y_mm;
    float r = 0.5f * diameter_mm;
    bool inside_xy = (x*x + y*y <= r*r);
    if (!inside_xy) a[idx] = 0.0f;
}

static int64_t count_multibed_roi_voxels(int Nx, int Ny, int global_Nz,
                                         float voxel_mm, float origin_z_mm,
                                         float diameter_mm, float valid_height_mm,
                                         float center_x_mm, float center_y_mm,
                                         const std::vector<BedInfo>& beds)
{
    double r = 0.5 * (double)diameter_mm;
    double r2 = r * r;
    int64_t cnt = 0;
    for (int iz = 0; iz < global_Nz; ++iz) {
        for (int iy = 0; iy < Ny; ++iy) {
            double y = ((double)iy - 0.5 * (double)(Ny - 1)) * voxel_mm - center_y_mm;
            for (int ix = 0; ix < Nx; ++ix) {
                double x = ((double)ix - 0.5 * (double)(Nx - 1)) * voxel_mm - center_x_mm;
                if (x*x + y*y <= r2) ++cnt;
            }
        }
    }
    return cnt;
}

static void apply_multibed_roi_host(float* a,
                                    int Nx, int Ny, int global_Nz,
                                    float voxel_mm, float origin_z_mm,
                                    float diameter_mm, float valid_height_mm,
                                    float center_x_mm, float center_y_mm,
                                    const std::vector<BedInfo>& beds)
{
    double r = 0.5 * (double)diameter_mm;
    double r2 = r * r;
    for (int iz = 0; iz < global_Nz; ++iz) {
        for (int iy = 0; iy < Ny; ++iy) {
            double y = ((double)iy - 0.5 * (double)(Ny - 1)) * voxel_mm - center_y_mm;
            for (int ix = 0; ix < Nx; ++ix) {
                double x = ((double)ix - 0.5 * (double)(Nx - 1)) * voxel_mm - center_x_mm;
                if (x*x + y*y > r2) {
                    int64_t idx = (int64_t)iz * Nx * Ny + (int64_t)iy * Nx + ix;
                    a[idx] = 0.0f;
                }
            }
        }
    }
}

// 独立床位调度器为每个床位保存一幅局部图像，因此这里的 ROI 使用局部矩阵坐标；
// 上面的 k_apply_multibed_roi 则用于拼接后的全局坐标。
__global__ void k_apply_local_cylinder_roi(float* a,
                                           int Nx, int Ny, int Nz,
                                           float voxel_mm,
                                           float diameter_mm, float height_mm,
                                           float center_x_mm, float center_y_mm,
                                           float center_z_mm, int64_t n)
{
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int iz = (int)(idx / ((int64_t)Nx * Ny));
    const int rem = (int)(idx - (int64_t)iz * Nx * Ny);
    const int iy = rem / Nx;
    const int ix = rem - iy * Nx;
    const float x = ((float)ix - 0.5f * (float)(Nx - 1)) * voxel_mm - center_x_mm;
    const float y = ((float)iy - 0.5f * (float)(Ny - 1)) * voxel_mm - center_y_mm;
    const float z = ((float)iz - 0.5f * (float)(Nz - 1)) * voxel_mm - center_z_mm;
    const float radius = 0.5f * diameter_mm;
    const float half_height = 0.5f * height_mm;
    if (x*x + y*y > radius*radius || fabsf(z) > half_height) a[idx] = 0.0f;
}

static void check_sensitivity_zsym_host(const float* a,
                                        int Nx, int Ny, int Nz,
                                        float rel_tol,
                                        const char* label)
{
    int64_t nxy = (int64_t)Nx * Ny;
    int z_pairs = Nz / 2;
    int64_t pair_voxels = (int64_t)z_pairs * nxy;
    if (pair_voxels <= 0) {
        printf("[8b] Sensitivity axial symmetry (%s): skipped, Nz=%d\n", label, Nz);
        return;
    }

    double sum_abs = 0.0;
    double sum_scale = 0.0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    int worst_ix = 0, worst_iy = 0, worst_z = 0;
    double worst_a = 0.0, worst_b = 0.0;

    double worst_slice_rel = 0.0;
    double worst_slice_abs = 0.0;
    int worst_slice_z = 0;
    double worst_slice_sum0 = 0.0;
    double worst_slice_sum1 = 0.0;

    const double eps = 1e-20;
    for (int z = 0; z < z_pairs; z++) {
        int mz = Nz - 1 - z;
        double slice0 = 0.0;
        double slice1 = 0.0;
        for (int iy = 0; iy < Ny; iy++) {
            for (int ix = 0; ix < Nx; ix++) {
                int64_t i0 = (int64_t)z * nxy + (int64_t)iy * Nx + ix;
                int64_t i1 = (int64_t)mz * nxy + (int64_t)iy * Nx + ix;
                double v0 = (double)a[i0];
                double v1 = (double)a[i1];
                double d = fabs(v0 - v1);
                double sc = 0.5 * (fabs(v0) + fabs(v1));
                double rel = d / fmax(sc, eps);
                sum_abs += d;
                sum_scale += sc;
                slice0 += v0;
                slice1 += v1;
                if (d > max_abs) max_abs = d;
                if (rel > max_rel) {
                    max_rel = rel;
                    worst_ix = ix; worst_iy = iy; worst_z = z;
                    worst_a = v0; worst_b = v1;
                }
            }
        }

        double sd = fabs(slice0 - slice1);
        double ss = 0.5 * (fabs(slice0) + fabs(slice1));
        double srel = sd / fmax(ss, eps);
        if (srel > worst_slice_rel) {
            worst_slice_rel = srel;
            worst_slice_abs = sd;
            worst_slice_z = z;
            worst_slice_sum0 = slice0;
            worst_slice_sum1 = slice1;
        }
    }

    double mean_abs = sum_abs / (double)pair_voxels;
    double global_rel = sum_abs / fmax(sum_scale, eps);
    const char* status = (global_rel <= (double)rel_tol && worst_slice_rel <= (double)rel_tol)
        ? "PASS" : "WARN";

    printf("[8b] Sensitivity axial symmetry (%s): %s\n", label, status);
    printf("     voxel-pair: global_rel=%.6e mean_abs=%.6e max_abs=%.6e max_rel=%.6e"
           " at (x=%d,y=%d,z=%d<->%d), vals=%.6e/%.6e\n",
           global_rel, mean_abs, max_abs, max_rel,
           worst_ix, worst_iy, worst_z, Nz - 1 - worst_z, worst_a, worst_b);
    printf("     slice-sum:  worst_rel=%.6e abs=%.6e at z=%d<->%d, sums=%.6e/%.6e, tol=%.3e\n",
           worst_slice_rel, worst_slice_abs, worst_slice_z, Nz - 1 - worst_slice_z,
           worst_slice_sum0, worst_slice_sum1, (double)rel_tol);
}

__global__ void k_mlem_update(float* f, const float* bp, const float* sens,
                              const float* map_grad, float beta,
                              int64_t n)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float denom = sens[i];
        if (!isfinite(denom)) { f[i] = 0.0f; return; }
        if (map_grad && beta > 0.0f) {
            const float grad = map_grad[i];
            if (!isfinite(grad)) { f[i] = 0.0f; return; }
            denom += beta * grad;
        }
        if (!isfinite(denom) || denom < 1e-12f) denom = 1e-12f;
        const float old = f[i];
        const float back = bp[i];
        if (!isfinite(old) || !isfinite(back)) { f[i] = 0.0f; return; }
        const float v = old * back / denom;
        f[i] = (isfinite(v) && v > 0.0f) ? v : 0.0f;
    }
}

enum OslDiagnosticCounter {
    OSL_DIAG_ACTIVE_ROI = 0,
    OSL_DIAG_OLD_NONFINITE,
    OSL_DIAG_OLD_NONPOSITIVE,
    OSL_DIAG_BP_NONFINITE,
    OSL_DIAG_BP_NEGATIVE,
    OSL_DIAG_SENS_NONFINITE,
    OSL_DIAG_SENS_NONPOSITIVE,
    OSL_DIAG_GRAD_NONFINITE,
    OSL_DIAG_PENALTY_NONFINITE,
    OSL_DIAG_RAW_DENOM_NONFINITE,
    OSL_DIAG_RAW_DENOM_NONPOSITIVE,
    OSL_DIAG_EFFECTIVE_DENOM_NONFINITE,
    OSL_DIAG_EFFECTIVE_DENOM_NONPOSITIVE,
    OSL_DIAG_OMEGA_EPSILON_FLOOR,
    OSL_DIAG_CASTOR_KEEP_OLD,
    OSL_DIAG_COUNTER_COUNT
};

enum OslDiagnosticRange {
    OSL_RANGE_SENSITIVITY = 0,
    OSL_RANGE_PENALTY,
    OSL_RANGE_RAW_DENOMINATOR,
    OSL_RANGE_EFFECTIVE_DENOMINATOR,
    OSL_RANGE_BETA_LIMIT,
    OSL_RANGE_NEGATIVE_PENALTY_FRACTION,
    OSL_RANGE_COUNT
};

struct OslDiagnostics {
    unsigned long long counts[OSL_DIAG_COUNTER_COUNT];
    float minima[OSL_RANGE_COUNT];
    float maxima[OSL_RANGE_COUNT];
};

static OslDiagnostics empty_osl_diagnostics()
{
    OslDiagnostics diagnostics{};
    for (int i = 0; i < OSL_RANGE_COUNT; ++i) {
        diagnostics.minima[i] = std::numeric_limits<float>::infinity();
        diagnostics.maxima[i] = -std::numeric_limits<float>::infinity();
    }
    return diagnostics;
}

__device__ void atomic_min_float(float* address, float value)
{
    int* integer_address = reinterpret_cast<int*>(address);
    int old = *integer_address;
    while (value < __int_as_float(old)) {
        const int assumed = old;
        old = atomicCAS(integer_address, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
}

__device__ void atomic_max_float(float* address, float value)
{
    int* integer_address = reinterpret_cast<int*>(address);
    int old = *integer_address;
    while (value > __int_as_float(old)) {
        const int assumed = old;
        old = atomicCAS(integer_address, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
}

/*
 * OSL 诊断独立于更新公式，只读当前轮更新前的数据。每个 block 先在共享内存中
 * 归约，再对全局统计做一次原子操作，避免为数百万体素逐点争用全局原子变量。
 * raw denominator 始终指 sensitivity+beta*gradient；effective denominator
 * 则包含 OMEGA 的 sensitivity 下限和末尾 epsilon，或 CASToR 的原始分母。
 */
__global__ void k_collect_osl_diagnostics(
    const float* old_image, const float* backprojection,
    const float* sensitivity, const float* prior_gradient,
    const uint8_t* roi_mask, float beta, float omega_epsilon,
    int reference_implementation, OslDiagnostics* diagnostics, int64_t n)
{
    __shared__ unsigned int shared_counts[OSL_DIAG_COUNTER_COUNT][BLK];
    __shared__ float shared_minima[OSL_RANGE_COUNT][BLK];
    __shared__ float shared_maxima[OSL_RANGE_COUNT][BLK];

    const int lane = threadIdx.x;
    const int64_t index = (int64_t)blockIdx.x * blockDim.x + lane;
    unsigned int local_counts[OSL_DIAG_COUNTER_COUNT] = {};
    float local_minima[OSL_RANGE_COUNT];
    float local_maxima[OSL_RANGE_COUNT];
    #pragma unroll
    for (int range = 0; range < OSL_RANGE_COUNT; ++range) {
        local_minima[range] = FLT_MAX;
        local_maxima[range] = -FLT_MAX;
    }

    if (index < n && (!roi_mask || roi_mask[index] != 0)) {
        local_counts[OSL_DIAG_ACTIVE_ROI] = 1;
        const float old_value = old_image[index];
        const float bp_value = backprojection[index];
        const float sens_value = sensitivity[index];
        const float grad_value = prior_gradient ? prior_gradient[index] : 0.0f;

        if (!isfinite(old_value)) local_counts[OSL_DIAG_OLD_NONFINITE] = 1;
        else if (old_value <= 0.0f) local_counts[OSL_DIAG_OLD_NONPOSITIVE] = 1;
        if (!isfinite(bp_value)) local_counts[OSL_DIAG_BP_NONFINITE] = 1;
        else if (bp_value < 0.0f) local_counts[OSL_DIAG_BP_NEGATIVE] = 1;

        if (!isfinite(sens_value)) {
            local_counts[OSL_DIAG_SENS_NONFINITE] = 1;
        } else {
            if (sens_value <= 0.0f) local_counts[OSL_DIAG_SENS_NONPOSITIVE] = 1;
            local_minima[OSL_RANGE_SENSITIVITY] = sens_value;
            local_maxima[OSL_RANGE_SENSITIVITY] = sens_value;
        }

        if (!isfinite(grad_value)) {
            local_counts[OSL_DIAG_GRAD_NONFINITE] = 1;
        } else {
            const float penalty = beta * grad_value;
            if (!isfinite(penalty)) {
                local_counts[OSL_DIAG_PENALTY_NONFINITE] = 1;
            } else {
                local_minima[OSL_RANGE_PENALTY] = penalty;
                local_maxima[OSL_RANGE_PENALTY] = penalty;
                if (isfinite(sens_value)) {
                    const float raw_denominator = sens_value + penalty;
                    if (!isfinite(raw_denominator)) {
                        local_counts[OSL_DIAG_RAW_DENOM_NONFINITE] = 1;
                    } else {
                        if (raw_denominator <= 0.0f) {
                            local_counts[OSL_DIAG_RAW_DENOM_NONPOSITIVE] = 1;
                        }
                        local_minima[OSL_RANGE_RAW_DENOMINATOR] = raw_denominator;
                        local_maxima[OSL_RANGE_RAW_DENOMINATOR] = raw_denominator;
                    }

                    const float base_sensitivity =
                        reference_implementation == IMPLEMENTATION_OMEGA
                            ? fmaxf(sens_value, omega_epsilon) + omega_epsilon
                            : sens_value;
                    const float effective_denominator = base_sensitivity + penalty;
                    if (!isfinite(effective_denominator)) {
                        local_counts[OSL_DIAG_EFFECTIVE_DENOM_NONFINITE] = 1;
                    } else {
                        if (effective_denominator <= 0.0f) {
                            local_counts[OSL_DIAG_EFFECTIVE_DENOM_NONPOSITIVE] = 1;
                        }
                        local_minima[OSL_RANGE_EFFECTIVE_DENOMINATOR] =
                            effective_denominator;
                        local_maxima[OSL_RANGE_EFFECTIVE_DENOMINATOR] =
                            effective_denominator;
                    }
                    if (grad_value < 0.0f && base_sensitivity > 0.0f &&
                        isfinite(base_sensitivity)) {
                        const float beta_limit = base_sensitivity / (-grad_value);
                        const float negative_penalty_fraction =
                            (-penalty) / base_sensitivity;
                        if (isfinite(beta_limit)) {
                            local_minima[OSL_RANGE_BETA_LIMIT] = beta_limit;
                            local_maxima[OSL_RANGE_BETA_LIMIT] = beta_limit;
                        }
                        if (isfinite(negative_penalty_fraction)) {
                            local_minima[OSL_RANGE_NEGATIVE_PENALTY_FRACTION] =
                                negative_penalty_fraction;
                            local_maxima[OSL_RANGE_NEGATIVE_PENALTY_FRACTION] =
                                negative_penalty_fraction;
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int counter = 0; counter < OSL_DIAG_COUNTER_COUNT; ++counter) {
        shared_counts[counter][lane] = local_counts[counter];
    }
    #pragma unroll
    for (int range = 0; range < OSL_RANGE_COUNT; ++range) {
        shared_minima[range][lane] = local_minima[range];
        shared_maxima[range][lane] = local_maxima[range];
    }
    __syncthreads();

    for (int stride = BLK / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            #pragma unroll
            for (int counter = 0; counter < OSL_DIAG_COUNTER_COUNT; ++counter) {
                shared_counts[counter][lane] += shared_counts[counter][lane + stride];
            }
            #pragma unroll
            for (int range = 0; range < OSL_RANGE_COUNT; ++range) {
                shared_minima[range][lane] = fminf(
                    shared_minima[range][lane], shared_minima[range][lane + stride]);
                shared_maxima[range][lane] = fmaxf(
                    shared_maxima[range][lane], shared_maxima[range][lane + stride]);
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        #pragma unroll
        for (int counter = 0; counter < OSL_DIAG_COUNTER_COUNT; ++counter) {
            if (shared_counts[counter][0] != 0) {
                atomicAdd(&diagnostics->counts[counter],
                          (unsigned long long)shared_counts[counter][0]);
            }
        }
        #pragma unroll
        for (int range = 0; range < OSL_RANGE_COUNT; ++range) {
            if (shared_minima[range][0] != FLT_MAX) {
                atomic_min_float(&diagnostics->minima[range],
                                 shared_minima[range][0]);
            }
            if (shared_maxima[range][0] != -FLT_MAX) {
                atomic_max_float(&diagnostics->maxima[range],
                                 shared_maxima[range][0]);
            }
        }
    }
}

/*
 * 公开入口的版本化 EM 图像域更新。reference_implementation 与投影域 ratio
 * 使用同一个枚举，主机端不能分别选择，从结构上阻止混搭。
 *
 * OMEGA MLEM（MLEM_im.m、algorithms.h、subiterStep.h）:
 *   x_new = x_old * (backprojection + epsilon)
 *                   / max(sensitivity, epsilon)
 * OMEGA OSL:
 *   x_new = x_old * (backprojection + epsilon)
 *                   / (max(sensitivity, epsilon)
 *                      + beta*prior_gradient + epsilon)
 * OMEGA 没有 CASToR 的更新因子上下限。原始 sensitivity+penalty 非正或任一
 * 更新输入非有限时，把该体素设为 omega_epsilon，等价于 OMEGA 更新后的
 * positivity constraint 下限，避免 0 体素在乘法 EM 中永久无法恢复。
 *
 * CASToR MLEM/OSL（iOptimizerMLEM.cc、iOptimizerOneStepLate.cc）:
 *   update_factor = backprojection / (sensitivity [+ penalty])
 * MLEM 与 OSL 都应用相同的最小/最大更新因子。OSL 分母非正时 update_factor=1，
 * 即保持旧体素，并计数报警。最大更新因子为 0 表示关闭上限。
 */
__global__ void k_em_update_versioned(
    float* f, const float* bp, const float* sens, const float* prior_grad,
    int algorithm, int reference_implementation, float beta,
    float omega_epsilon,
    float castor_min_update, float castor_max_update,
    unsigned long long* nonpositive_denom_count,
    OslDiagnostics* diagnostics, int64_t n)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float old_value = f[i];
    if (!isfinite(old_value) || old_value <= 0.0f) {
        f[i] = 0.0f;
        return;
    }

    const float backprojection = bp[i];
    const float raw_sensitivity = sens[i];
    if (!isfinite(backprojection) || backprojection < 0.0f ||
        !isfinite(raw_sensitivity)) {
        if (algorithm == ALGORITHM_OSL && nonpositive_denom_count) {
            atomicAdd(nonpositive_denom_count, 1ULL);
        }
        if (algorithm == ALGORITHM_OSL &&
            reference_implementation == IMPLEMENTATION_CASTOR) {
            f[i] = old_value;
            if (diagnostics) {
                atomicAdd(&diagnostics->counts[OSL_DIAG_CASTOR_KEEP_OLD], 1ULL);
            }
        } else if (algorithm == ALGORITHM_OSL &&
                   reference_implementation == IMPLEMENTATION_OMEGA) {
            f[i] = omega_epsilon;
            if (diagnostics) {
                atomicAdd(&diagnostics->counts[OSL_DIAG_OMEGA_EPSILON_FLOOR], 1ULL);
            }
        } else {
            f[i] = 0.0f;
        }
        return;
    }

    if (reference_implementation == IMPLEMENTATION_OMEGA) {
        const float sensitivity = fmaxf(raw_sensitivity, omega_epsilon);
        float denominator = sensitivity;
        if (algorithm == ALGORITHM_OSL) {
            const float gradient = prior_grad ? prior_grad[i] : 0.0f;
            if (!isfinite(gradient)) {
                if (nonpositive_denom_count) {
                    atomicAdd(nonpositive_denom_count, 1ULL);
                }
                f[i] = omega_epsilon;
                if (diagnostics) {
                    atomicAdd(&diagnostics->counts[OSL_DIAG_OMEGA_EPSILON_FLOOR],
                              1ULL);
                }
                return;
            }
            const float raw_denominator = sensitivity + beta * gradient;
            if (!isfinite(raw_denominator) || raw_denominator <= 0.0f) {
                if (nonpositive_denom_count) {
                    atomicAdd(nonpositive_denom_count, 1ULL);
                }
                f[i] = omega_epsilon;
                if (diagnostics) {
                    atomicAdd(&diagnostics->counts[OSL_DIAG_OMEGA_EPSILON_FLOOR],
                              1ULL);
                }
                return;
            }
            denominator = raw_denominator + omega_epsilon;
        }
        const float numerator = backprojection + omega_epsilon;
        if (!isfinite(denominator) || denominator <= 0.0f) {
            f[i] = algorithm == ALGORITHM_OSL ? omega_epsilon : 0.0f;
            if (algorithm == ALGORITHM_OSL && diagnostics) {
                atomicAdd(&diagnostics->counts[OSL_DIAG_OMEGA_EPSILON_FLOOR], 1ULL);
            }
            return;
        }
        const float value = old_value * numerator / denominator;
        if (algorithm == ALGORITHM_OSL) {
            if (!isfinite(value) || value <= 0.0f) {
                if (nonpositive_denom_count) {
                    atomicAdd(nonpositive_denom_count, 1ULL);
                }
                f[i] = omega_epsilon;
                if (diagnostics) {
                    atomicAdd(&diagnostics->counts[OSL_DIAG_OMEGA_EPSILON_FLOOR],
                              1ULL);
                }
            } else {
                f[i] = fmaxf(value, omega_epsilon);
                if (value < omega_epsilon && diagnostics) {
                    atomicAdd(&diagnostics->counts[OSL_DIAG_OMEGA_EPSILON_FLOOR],
                              1ULL);
                }
            }
        } else {
            f[i] = (isfinite(value) && value > 0.0f) ? value : 0.0f;
        }
        return;
    }

    // CASToR 的 MLEM 和 OSL 共用更新因子上下限；只有 OSL 有 penalty 分母规则。
    float denominator = raw_sensitivity;
    if (algorithm == ALGORITHM_OSL) {
        const float gradient = prior_grad ? prior_grad[i] : 0.0f;
        if (!isfinite(gradient)) {
            if (nonpositive_denom_count) {
                atomicAdd(nonpositive_denom_count, 1ULL);
            }
            f[i] = old_value;
            if (diagnostics) {
                atomicAdd(&diagnostics->counts[OSL_DIAG_CASTOR_KEEP_OLD], 1ULL);
            }
            return;
        }
        denominator += beta * gradient;
        if (!isfinite(denominator) || denominator <= 0.0f) {
            if (nonpositive_denom_count) {
                atomicAdd(nonpositive_denom_count, 1ULL);
            }
            f[i] = old_value;
            if (diagnostics) {
                atomicAdd(&diagnostics->counts[OSL_DIAG_CASTOR_KEEP_OLD], 1ULL);
            }
            return;
        }
    } else if (denominator <= 0.0f) {
        f[i] = 0.0f;
        return;
    }

    float update_factor = backprojection / denominator;
    if (castor_min_update > 0.0f && update_factor < castor_min_update) {
        update_factor = castor_min_update;
    }
    if (castor_max_update > 0.0f && update_factor > castor_max_update) {
        update_factor = castor_max_update;
    }
    const float value = old_value * update_factor;
    if (algorithm == ALGORITHM_OSL && (!isfinite(value) || value <= 0.0f)) {
        if (nonpositive_denom_count) atomicAdd(nonpositive_denom_count, 1ULL);
        f[i] = old_value;
        if (diagnostics) {
            atomicAdd(&diagnostics->counts[OSL_DIAG_CASTOR_KEEP_OLD], 1ULL);
        }
    } else {
        f[i] = (isfinite(value) && value > 0.0f) ? value : 0.0f;
    }
}

__device__ __forceinline__ void map_xy_90(int ix, int iy, int Nx, int Ny,
                                          int sym, int& ox, int& oy)
{

    switch (sym) {
        case 1: ox = Ny - 1 - iy; oy = ix; break;
        case 2: ox = Nx - 1 - ix; oy = Ny - 1 - iy; break;
        case 3: ox = iy; oy = Nx - 1 - ix; break;
        default: ox = ix; oy = iy; break;
    }
}

__device__ __forceinline__ void map_xy_45_d4(int ix, int iy, int N,
                                             int sym, int& ox, int& oy)
{

    switch (sym) {
        case 1: ox = ix;       oy = N-1-iy;   break;
        case 2: ox = iy;       oy = ix;       break;
        case 3: ox = N-1-iy;   oy = ix;       break;
        case 4: ox = N-1-ix;   oy = iy;       break;
        case 5: ox = N-1-ix;   oy = N-1-iy;   break;
        case 6: ox = N-1-iy;   oy = N-1-ix;   break;
        case 7: ox = iy;       oy = N-1-ix;   break;
        default: ox = ix;      oy = iy;       break;
    }
}

__device__ __forceinline__ void map_coords_sym(int ix, int iy, int iz,
                                               int Nx, int Ny, int Nz,
                                               int sym_mode, int sym_id, int z_flip,
                                               int& ox, int& oy, int& oz)
{
    if (sym_mode == SYM_45 || sym_mode == SYM_45Z) {
        map_xy_45_d4(ix, iy, Nx, sym_id, ox, oy);
    } else if (sym_mode == SYM_90) {
        map_xy_90(ix, iy, Nx, Ny, sym_id, ox, oy);
    } else if (sym_mode == SYM_180 || sym_mode == SYM_Z180) {
        if (sym_id == 1) { ox = Nx - 1 - ix; oy = Ny - 1 - iy; }
        else             { ox = ix;          oy = iy; }
    } else {
        ox = ix; oy = iy;
    }
    oz = z_flip ? (Nz - 1 - iz) : iz;
}

__global__ void k_transform_fwd(const float* __restrict__ fin,
                                float* __restrict__ fout,
                                int Nx, int Ny, int Nz,
                                int sym_mode, int sym_id, int z_flip)
{
    int64_t N2 = (int64_t)Nx * Ny * Nz;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    int iz = (int)(idx / ((int64_t)Nx * Ny));
    int rem = (int)(idx - (int64_t)iz * Nx * Ny);
    int iy = rem / Nx;
    int ix = rem - iy * Nx;

    int ox, oy, oz;
    map_coords_sym(ix, iy, iz, Nx, Ny, Nz, sym_mode, sym_id, z_flip, ox, oy, oz);
    fout[idx] = fin[(int64_t)oz * Nx * Ny + oy * Nx + ox];
}

__global__ void k_transform_bwd_add(const float* __restrict__ bp_rot,
                                    float* __restrict__ bp_orig,
                                    int Nx, int Ny, int Nz,
                                    int sym_mode, int sym_id, int z_flip)
{
    int64_t N2 = (int64_t)Nx * Ny * Nz;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    int iz = (int)(idx / ((int64_t)Nx * Ny));
    int rem = (int)(idx - (int64_t)iz * Nx * Ny);
    int iy = rem / Nx;
    int ix = rem - iy * Nx;

    int ox, oy, oz;
    map_coords_sym(ix, iy, iz, Nx, Ny, Nz, sym_mode, sym_id, z_flip, ox, oy, oz);
    bp_orig[(int64_t)oz * Nx * Ny + oy * Nx + ox] += bp_rot[idx];
}

__device__ __forceinline__ int full_angle_from_compact(int ck, int sym_mode,
                                                       int sym_id,
                                                       int compact_num_angles,
                                                       int full_num_angles,
                                                       int* valid)
{
    *valid = 1;
    if (sym_mode == SYM_NONE || sym_mode == SYM_ZONLY) {
        return ck;
    }
    if (sym_mode == SYM_180 || sym_mode == SYM_Z180) {
        return ck + sym_id * compact_num_angles;
    }
    if (sym_mode == SYM_90) {
        return ck + sym_id * compact_num_angles;
    }
    if (sym_mode == SYM_45 || sym_mode == SYM_45Z) {
        int Q = full_num_angles / 8;
        int na = full_num_angles;
        if (compact_num_angles != Q + 1 || Q <= 0) {
            *valid = 0;
            return 0;
        }

        switch (sym_id) {
            case 0:
                if (ck < 0 || ck > Q) { *valid = 0; return 0; }
                return ck;
            case 1:
                if (ck <= 0 || ck >= Q) { *valid = 0; return 0; }
                return na - ck;
            case 2:
                if (ck < 0 || ck >= Q) { *valid = 0; return 0; }
                return 2 * Q - ck;
            case 3:
                if (ck <= 0 || ck > Q) { *valid = 0; return 0; }
                return 2 * Q + ck;
            case 4:
                if (ck < 0 || ck >= Q) { *valid = 0; return 0; }
                return 4 * Q - ck;
            case 5:
                if (ck <= 0 || ck > Q) { *valid = 0; return 0; }
                return 4 * Q + ck;
            case 6:
                if (ck < 0 || ck >= Q) { *valid = 0; return 0; }
                return 6 * Q - ck;
            case 7:
                if (ck <= 0 || ck > Q) { *valid = 0; return 0; }
                return 6 * Q + ck;
            default:
                *valid = 0;
                return 0;
        }
    }
    return ck;
}

__global__ void k_calc_ratio_batched(const float* __restrict__ y,
                                     float* __restrict__ ratio,
                                     const float* __restrict__ P_full,
                                     const float* __restrict__ G_full,
                                     int local_rows,
                                     int worker_start_ck,
                                     int compact_rpa,
                                     int det_nx,
                                     int compact_det_z,
                                     int det_nz,
                                     int full_rpa,
                                     int sym_mode,
                                     int sym_id,
                                     int z_mirror,
                                      int compact_num_angles,
                                      int full_num_angles,
                                      int recon_angle_step,
                                      int reference_implementation,
                                      float omega_epsilon,
                                      float castor_threshold,
                                      unsigned long long* low_forward_count)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= local_rows) return;

    int local_ck = r / compact_rpa;
    int rr = r - local_ck * compact_rpa;
    int ck = worker_start_ck + local_ck;

    int ix = rr % det_nx;
    int izc = rr / det_nx;

    int valid_angle = 1;
    int fa = full_angle_from_compact(ck, sym_mode, sym_id, compact_num_angles,
                                     full_num_angles, &valid_angle);
    if (!valid_angle || fa < 0 || fa >= full_num_angles) {
        ratio[r] = 0.0f;
        return;
    }
    if (recon_angle_step > 1 && (fa % recon_angle_step) != 0) {
        ratio[r] = 0.0f;
        return;
    }

    int iz_full;
    if (!z_mirror) {
        iz_full = izc;
    } else {
        iz_full = det_nz - 1 - izc;

        if (iz_full < compact_det_z) {
            ratio[r] = 0.0f;
            return;
        }
    }
    if (iz_full < 0 || iz_full >= det_nz) {
        ratio[r] = 0.0f;
        return;
    }

    int64_t full_idx = (int64_t)fa * full_rpa + iz_full * det_nx + ix;
    float g = G_full[full_idx];
    float p = P_full[full_idx];
    float yy = y[r];
    if (!isfinite(g) || !isfinite(p) || !isfinite(yy) ||
        g <= 0.0f || p < 0.0f || yy < 0.0f) {
        ratio[r] = 0.0f;
        return;
    }

    const float pred = g * yy;
    if (!isfinite(pred) || pred < 0.0f) {
        ratio[r] = 0.0f;
        return;
    }
    if (reference_implementation == IMPLEMENTATION_OMEGA) {
        // OMEGA 在投影分母中加 epsilon，不使用 CASToR 的阈值中性分支。
        const float denominator = pred + omega_epsilon;
        const float value = (g * p) / denominator;
        ratio[r] = (isfinite(value) && value >= 0.0f) ? value : 0.0f;
        return;
    }
    if (pred <= castor_threshold) {
        /*
         * CASToR 在 forwardModel 过小时令 data-space ratio=1，因为该行仍包含
         * 在 sensitivity 中。本工程将响应因子 g 显式放入 sensitivity=A^T*g，
         * 所以相同的中性反投影贡献必须是 ratio=g；只有 g=1 时字面值才为 1。
         */
        ratio[r] = g;
        if (low_forward_count) atomicAdd(low_forward_count, 1ULL);
        return;
    }
    const float value = (g * p) / pred;
    ratio[r] = (isfinite(value) && value >= 0.0f) ? value : 0.0f;
}

__global__ void k_make_unit_ratio_batched(float* __restrict__ ratio,
                                          const float* __restrict__ G_full,
                                          int local_rows,
                                          int worker_start_ck,
                                          int compact_rpa,
                                          int det_nx,
                                          int compact_det_z,
                                          int det_nz,
                                          int full_rpa,
                                          int sym_mode,
                                          int sym_id,
                                          int z_mirror,
                                          int compact_num_angles,
                                          int full_num_angles,
                                          int recon_angle_step)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= local_rows) return;

    int local_ck = r / compact_rpa;
    int rr = r - local_ck * compact_rpa;
    int ck = worker_start_ck + local_ck;

    int ix = rr % det_nx;
    int izc = rr / det_nx;

    int valid_angle = 1;
    int fa = full_angle_from_compact(ck, sym_mode, sym_id, compact_num_angles,
                                     full_num_angles, &valid_angle);
    if (!valid_angle || fa < 0 || fa >= full_num_angles) { ratio[r] = 0.0f; return; }
    if (recon_angle_step > 1 && (fa % recon_angle_step) != 0) { ratio[r] = 0.0f; return; }

    int iz_full;
    if (!z_mirror) {
        iz_full = izc;
    } else {
        iz_full = det_nz - 1 - izc;
        if (iz_full < compact_det_z) { ratio[r] = 0.0f; return; }
    }
    if (iz_full < 0 || iz_full >= det_nz) { ratio[r] = 0.0f; return; }

    int64_t full_idx = (int64_t)fa * full_rpa + iz_full * det_nx + ix;
    float g = G_full[full_idx];
    ratio[r] = (isfinite(g) && g > 0.0f) ? g : 0.0f;
}

__constant__ float c_kx[MAX_KERNEL_LEN];
__constant__ float c_ky[MAX_KERNEL_LEN];
__constant__ float c_kz[MAX_KERNEL_LEN];
__constant__ int c_hx;
__constant__ int c_hy;
__constant__ int c_hz;

__global__ void k_conv_x(const float* __restrict__ in, float* __restrict__ out,
                         int Nx, int Ny, int Nz)
{
    int64_t N2 = (int64_t)Nx * Ny * Nz;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    int iz = (int)(idx / ((int64_t)Nx * Ny));
    int rem = (int)(idx - (int64_t)iz * Nx * Ny);
    int iy = rem / Nx;
    int ix = rem - iy * Nx;
    float s = 0.0f;
    int h = c_hx;
    for (int k = -h; k <= h; k++) {
        int x = ix + k;
        if (x >= 0 && x < Nx) s += in[(int64_t)iz * Nx * Ny + iy * Nx + x] * c_kx[k + h];
    }
    out[idx] = s;
}

__global__ void k_conv_y(const float* __restrict__ in, float* __restrict__ out,
                         int Nx, int Ny, int Nz)
{
    int64_t N2 = (int64_t)Nx * Ny * Nz;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    int iz = (int)(idx / ((int64_t)Nx * Ny));
    int rem = (int)(idx - (int64_t)iz * Nx * Ny);
    int iy = rem / Nx;
    int ix = rem - iy * Nx;
    float s = 0.0f;
    int h = c_hy;
    for (int k = -h; k <= h; k++) {
        int y = iy + k;
        if (y >= 0 && y < Ny) s += in[(int64_t)iz * Nx * Ny + y * Nx + ix] * c_ky[k + h];
    }
    out[idx] = s;
}

__global__ void k_conv_z(const float* __restrict__ in, float* __restrict__ out,
                         int Nx, int Ny, int Nz)
{
    int64_t N2 = (int64_t)Nx * Ny * Nz;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    int iz = (int)(idx / ((int64_t)Nx * Ny));
    int rem = (int)(idx - (int64_t)iz * Nx * Ny);
    int iy = rem / Nx;
    int ix = rem - iy * Nx;
    float s = 0.0f;
    int h = c_hz;
    for (int k = -h; k <= h; k++) {
        int z = iz + k;
        if (z >= 0 && z < Nz) s += in[(int64_t)z * Nx * Ny + iy * Nx + ix] * c_kz[k + h];
    }
    out[idx] = s;
}

// OMEGA padding.m 使用包含边界体素的 symmetric 镜像，例如 -1->0、N->N-1。
__device__ __forceinline__ int symmetric_reflect_index(int value, int size) {
    while (value < 0 || value >= size) {
        value = value < 0 ? -value - 1 : 2 * size - value - 1;
    }
    return value;
}

__global__ void k_conv_x_symmetric(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int Nx, int Ny, int Nz) {
    const int64_t n = (int64_t)Nx * Ny * Nz;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int z = (int)(idx / ((int64_t)Nx * Ny));
    const int rem = (int)(idx - (int64_t)z * Nx * Ny);
    const int y = rem / Nx;
    const int x = rem - y * Nx;
    float sum = 0.0f;
    for (int k = -c_hx; k <= c_hx; ++k) {
        const int xx = symmetric_reflect_index(x + k, Nx);
        sum += in[(int64_t)z * Nx * Ny + (int64_t)y * Nx + xx]
             * c_kx[k + c_hx];
    }
    out[idx] = sum;
}

__global__ void k_conv_y_symmetric(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int Nx, int Ny, int Nz) {
    const int64_t n = (int64_t)Nx * Ny * Nz;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int z = (int)(idx / ((int64_t)Nx * Ny));
    const int rem = (int)(idx - (int64_t)z * Nx * Ny);
    const int y = rem / Nx;
    const int x = rem - y * Nx;
    float sum = 0.0f;
    for (int k = -c_hy; k <= c_hy; ++k) {
        const int yy = symmetric_reflect_index(y + k, Ny);
        sum += in[(int64_t)z * Nx * Ny + (int64_t)yy * Nx + x]
             * c_ky[k + c_hy];
    }
    out[idx] = sum;
}

__global__ void k_conv_z_symmetric(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int Nx, int Ny, int Nz) {
    const int64_t n = (int64_t)Nx * Ny * Nz;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int z = (int)(idx / ((int64_t)Nx * Ny));
    const int rem = (int)(idx - (int64_t)z * Nx * Ny);
    const int y = rem / Nx;
    const int x = rem - y * Nx;
    float sum = 0.0f;
    for (int k = -c_hz; k <= c_hz; ++k) {
        const int zz = symmetric_reflect_index(z + k, Nz);
        sum += in[(int64_t)zz * Nx * Ny + (int64_t)y * Nx + x]
             * c_kz[k + c_hz];
    }
    out[idx] = sum;
}

// CASToR stationary convolver 只补偿轴向 z 截断，不补偿横断面 x/y 边界。
__global__ void k_build_castor_z_norm(float* norm, int Nx, int Ny, int Nz) {
    const int64_t n = (int64_t)Nx * Ny * Nz;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int z = (int)(idx / ((int64_t)Nx * Ny));
    float sum = 0.0f;
    for (int k = -c_hz; k <= c_hz; ++k) {
        const int zz = z + k;
        if (zz >= 0 && zz < Nz) sum += c_kz[k + c_hz];
    }
    norm[idx] = (isfinite(sum) && sum > 1e-12f) ? sum : 1.0f;
}

__global__ void k_div_norm(float* a, const float* norm, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float d = norm[i];
        if (d < 1e-12f) d = 1e-12f;
        a[i] /= d;
    }
}

__global__ void k_div_norm_to(const float* __restrict__ in,
                              const float* __restrict__ norm,
                              float* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float d = norm[i];
        if (!isfinite(d) || d < 1e-12f) d = 1e-12f;
        float v = in[i] / d;
        out[i] = isfinite(v) ? v : 0.0f;
    }
}

static int castor_round_host(float x) {
    int h = (int)floorf(x);
    if ((x - h) > 0.5f) h++;
    // CASToR 允许半核为 0，对很小的 FWHM 得到 1 个体素的恒等核。
    if (h < 0) h = 0;
    return h;
}

static std::vector<float> build_castor_erf_1d(float fwhm_mm, float voxel_mm, float nb_sigmas) {
    float sigma = fwhm_mm / (2.0f * sqrtf(2.0f * logf(2.0f)));
    int half = castor_round_host(sigma * nb_sigmas / voxel_mm);
    if (2*half + 1 > MAX_KERNEL_LEN) {
        fprintf(stderr, "Gaussian kernel too large: %d > %d, reduce FWHM or increase MAX_KERNEL_LEN\n",
                2*half+1, MAX_KERNEL_LEN);
        exit(1);
    }
    float denom = sqrtf(2.0f) * sigma;
    std::vector<float> k((size_t)2*half + 1);
    float sum = 0.0f;
    for (int i = -half; i <= half; i++) {
        float v = 0.5f * (erff((i + 0.5f) * voxel_mm / denom) -
                          erff((i - 0.5f) * voxel_mm / denom));
        k[(size_t)(i + half)] = v;
        sum += v;
    }
    if (sum <= 0.0f) sum = 1.0f;
    for (auto& v : k) v /= sum;
    return k;
}

// OMEGA PSFKernel.m：半径 ceil(2*sigma/voxel)，在体素中心采样 Gaussian。
static std::vector<float> build_omega_sampled_1d(float fwhm_mm,
                                                  float voxel_mm) {
    const float sigma = fwhm_mm /
        (2.0f * sqrtf(2.0f * logf(2.0f)));
    const int half = (int)ceilf(2.0f * sigma / voxel_mm);
    if (2 * half + 1 > MAX_KERNEL_LEN) {
        fprintf(stderr,
                "OMEGA Gaussian kernel too large: %d > %d, reduce FWHM or increase MAX_KERNEL_LEN\n",
                2 * half + 1, MAX_KERNEL_LEN);
        exit(1);
    }
    std::vector<float> kernel((size_t)2 * half + 1);
    float sum = 0.0f;
    for (int i = -half; i <= half; ++i) {
        const float distance = (float)i * voxel_mm;
        const float value = expf(-0.5f * distance * distance / (sigma * sigma));
        kernel[(size_t)(i + half)] = value;
        sum += value;
    }
    if (!isfinite(sum) || sum <= 0.0f) sum = 1.0f;
    for (float& value : kernel) value /= sum;
    return kernel;
}

static void upload_gaussian_kernel(const std::vector<float>& kx,
                                   const std::vector<float>& ky,
                                   const std::vector<float>& kz)
{
    int hx = (int)kx.size() / 2;
    int hy = (int)ky.size() / 2;
    int hz = (int)kz.size() / 2;
    CUDA_CHECK(cudaMemcpyToSymbol(c_kx, kx.data(), kx.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ky, ky.data(), ky.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_kz, kz.data(), kz.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_hx, &hx, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_hy, &hy, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_hz, &hz, sizeof(int)));
}

/*
 * 以下全边界归一化卷积是旧的底层联合诊断引擎所用辅助函数。它以
 * C=D_xyz^-1*K 保持整个有限图像上的常量，但这并不是 CASToR 3.2 的前向
 * 模型，因为 CASToR 只补偿 z 边界。公开 --proj-dir 入口不再把它暴露为
 * “adjoint”版本；真正的 CASToR 伴随修正版见 gpu_conv3d_castor_adjoint()。
 */
static void gpu_conv3d_forward_renorm(const float* d_in, float* d_out,
                                      float* d_tmp, const float* d_norm,
                                      int Nx, int Ny, int Nz,
                                      const std::vector<float>& kx,
                                      const std::vector<float>& ky,
                                      const std::vector<float>& kz)
{
    int64_t N2 = (int64_t)Nx * Ny * Nz;
    int nb = (int)((N2 + BLK - 1) / BLK);
    upload_gaussian_kernel(kx, ky, kz);
    k_conv_x<<<nb, BLK>>>(d_in, d_tmp, Nx, Ny, Nz);
    k_conv_y<<<nb, BLK>>>(d_tmp, d_out, Nx, Ny, Nz);
    k_conv_z<<<nb, BLK>>>(d_out, d_tmp, Nx, Ny, Nz);
    CUDA_CHECK(cudaMemcpy(d_out, d_tmp, (size_t)N2*sizeof(float), cudaMemcpyDeviceToDevice));
    k_div_norm<<<nb, BLK>>>(d_out, d_norm, N2);
    CUDA_CHECK(cudaGetLastError());
}

static void gpu_conv3d_transpose_renorm(const float* d_in, float* d_out,
                                        float* d_tmp, const float* d_norm,
                                        int Nx, int Ny, int Nz,
                                        const std::vector<float>& kx,
                                        const std::vector<float>& ky,
                                        const std::vector<float>& kz)
{
    int64_t N2 = (int64_t)Nx * Ny * Nz;
    int nb = (int)((N2 + BLK - 1) / BLK);
    upload_gaussian_kernel(kx, ky, kz);
    k_div_norm_to<<<nb, BLK>>>(d_in, d_norm, d_out, N2);
    k_conv_x<<<nb, BLK>>>(d_out, d_tmp, Nx, Ny, Nz);
    k_conv_y<<<nb, BLK>>>(d_tmp, d_out, Nx, Ny, Nz);
    k_conv_z<<<nb, BLK>>>(d_out, d_tmp, Nx, Ny, Nz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(d_out, d_tmp, (size_t)N2*sizeof(float),
                          cudaMemcpyDeviceToDevice));
}

static void build_conv_norm(float* d_norm, float* d_tmp1, float* d_tmp2,
                            int Nx, int Ny, int Nz,
                            const std::vector<float>& kx,
                            const std::vector<float>& ky,
                            const std::vector<float>& kz)
{
    int64_t N2 = (int64_t)Nx * Ny * Nz;
    int nb = (int)((N2 + BLK - 1) / BLK);
    upload_gaussian_kernel(kx, ky, kz);
    k_fill<<<nb, BLK>>>(d_norm, 1.0f, N2);
    k_conv_x<<<nb, BLK>>>(d_norm, d_tmp1, Nx, Ny, Nz);
    k_conv_y<<<nb, BLK>>>(d_tmp1, d_tmp2, Nx, Ny, Nz);
    k_conv_z<<<nb, BLK>>>(d_tmp2, d_norm, Nx, Ny, Nz);
    k_clamp_min<<<nb, BLK>>>(d_norm, 1e-12f, N2);
    CUDA_CHECK(cudaGetLastError());
}

static void gpu_conv3d_zero_padded(const float* d_in, float* d_out,
                                   float* d_tmp, int Nx, int Ny, int Nz,
                                   const std::vector<float>& kx,
                                   const std::vector<float>& ky,
                                   const std::vector<float>& kz) {
    const int64_t n = (int64_t)Nx * Ny * Nz;
    const int blocks = (int)((n + BLK - 1) / BLK);
    upload_gaussian_kernel(kx, ky, kz);
    k_conv_x<<<blocks, BLK>>>(d_in, d_tmp, Nx, Ny, Nz);
    k_conv_y<<<blocks, BLK>>>(d_tmp, d_out, Nx, Ny, Nz);
    k_conv_z<<<blocks, BLK>>>(d_out, d_tmp, Nx, Ny, Nz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(d_out, d_tmp, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToDevice));
}

static void gpu_conv3d_omega_symmetric(const float* d_in, float* d_out,
                                       float* d_tmp, int Nx, int Ny, int Nz,
                                       const std::vector<float>& kx,
                                       const std::vector<float>& ky,
                                       const std::vector<float>& kz) {
    const int64_t n = (int64_t)Nx * Ny * Nz;
    const int blocks = (int)((n + BLK - 1) / BLK);
    upload_gaussian_kernel(kx, ky, kz);
    k_conv_x_symmetric<<<blocks, BLK>>>(d_in, d_tmp, Nx, Ny, Nz);
    k_conv_y_symmetric<<<blocks, BLK>>>(d_tmp, d_out, Nx, Ny, Nz);
    k_conv_z_symmetric<<<blocks, BLK>>>(d_out, d_tmp, Nx, Ny, Nz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(d_out, d_tmp, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToDevice));
}

static void build_castor_z_norm(float* d_norm, int Nx, int Ny, int Nz,
                                const std::vector<float>& kx,
                                const std::vector<float>& ky,
                                const std::vector<float>& kz) {
    const int64_t n = (int64_t)Nx * Ny * Nz;
    const int blocks = (int)((n + BLK - 1) / BLK);
    upload_gaussian_kernel(kx, ky, kz);
    k_build_castor_z_norm<<<blocks, BLK>>>(d_norm, Nx, Ny, Nz);
    CUDA_CHECK(cudaGetLastError());
}

static void gpu_conv3d_castor(const float* d_in, float* d_out,
                              float* d_tmp, const float* d_z_norm,
                              int Nx, int Ny, int Nz,
                              const std::vector<float>& kx,
                              const std::vector<float>& ky,
                              const std::vector<float>& kz) {
    const int64_t n = (int64_t)Nx * Ny * Nz;
    const int blocks = (int)((n + BLK - 1) / BLK);
    gpu_conv3d_zero_padded(d_in, d_out, d_tmp, Nx, Ny, Nz, kx, ky, kz);
    k_div_norm<<<blocks, BLK>>>(d_out, d_z_norm, n);
    CUDA_CHECK(cudaGetLastError());
}

/*
 * CASToR stationary Gaussian 的正向为 C=D_z^-1*K，其中 K 是 x/y/z 零扩展
 * 卷积，D_z 只补偿 z 方向有限边界。严格伴随必须按相反顺序执行
 * C^T=K^T*D_z^-1；Gaussian 核对称，所以 K^T=K。先把输入除以 D_z，再做
 * 零扩展卷积。d_out 可同时作为缩放后的输入和最终输出，因为第一轴卷积写入
 * 独立的 d_tmp，kernel 启动顺序由同一 CUDA stream 保证。
 */
static void gpu_conv3d_castor_adjoint(
    const float* d_in, float* d_out, float* d_tmp,
    const float* d_z_norm, int Nx, int Ny, int Nz,
    const std::vector<float>& kx, const std::vector<float>& ky,
    const std::vector<float>& kz) {
    const int64_t n = (int64_t)Nx * Ny * Nz;
    const int blocks = (int)((n + BLK - 1) / BLK);
    k_div_norm_to<<<blocks, BLK>>>(d_in, d_z_norm, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    gpu_conv3d_zero_padded(d_out, d_out, d_tmp,
                           Nx, Ny, Nz, kx, ky, kz);
}

static bool convolution_requires_castor_norm(int implementation) {
    return implementation == CONV_IMPLEMENTATION_CASTOR ||
           implementation == CONV_IMPLEMENTATION_CASTOR_ADJOINT;
}

static void build_gaussian_kernel_set(
    int implementation, const float fwhm_mm[3], float voxel_mm,
    float nb_sigmas, std::vector<float>& kx, std::vector<float>& ky,
    std::vector<float>& kz) {
    if (implementation == CONV_IMPLEMENTATION_OMEGA) {
        kx = build_omega_sampled_1d(fwhm_mm[0], voxel_mm);
        ky = build_omega_sampled_1d(fwhm_mm[1], voxel_mm);
        kz = build_omega_sampled_1d(fwhm_mm[2], voxel_mm);
    } else {
        kx = build_castor_erf_1d(fwhm_mm[0], voxel_mm, nb_sigmas);
        ky = build_castor_erf_1d(fwhm_mm[1], voxel_mm, nb_sigmas);
        kz = build_castor_erf_1d(fwhm_mm[2], voxel_mm, nb_sigmas);
    }
}

static void prepare_convolution_normalization(
    int implementation, float* d_norm, int Nx, int Ny, int Nz,
    const std::vector<float>& kx, const std::vector<float>& ky,
    const std::vector<float>& kz) {
    if (convolution_requires_castor_norm(implementation)) {
        build_castor_z_norm(d_norm, Nx, Ny, Nz, kx, ky, kz);
    }
}

static void apply_convolution_forward(
    int implementation, const float* d_in, float* d_out, float* d_tmp,
    const float* d_norm, int Nx, int Ny, int Nz,
    const std::vector<float>& kx, const std::vector<float>& ky,
    const std::vector<float>& kz) {
    if (implementation == CONV_IMPLEMENTATION_OMEGA) {
        gpu_conv3d_omega_symmetric(d_in, d_out, d_tmp,
                                   Nx, Ny, Nz, kx, ky, kz);
    } else {
        // strict CASToR 与伴随修正版共享完全相同的前向模型。
        gpu_conv3d_castor(d_in, d_out, d_tmp, d_norm,
                          Nx, Ny, Nz, kx, ky, kz);
    }
}

static void apply_convolution_backward(
    int implementation, const float* d_in, float* d_out, float* d_tmp,
    const float* d_norm, int Nx, int Ny, int Nz,
    const std::vector<float>& kx, const std::vector<float>& ky,
    const std::vector<float>& kz) {
    if (implementation == CONV_IMPLEMENTATION_CASTOR_ADJOINT) {
        gpu_conv3d_castor_adjoint(d_in, d_out, d_tmp, d_norm,
                                  Nx, Ny, Nz, kx, ky, kz);
    } else {
        // OMEGA 与严格 CASToR 均按各自源码在反投影/sensitivity 后重用正向卷积。
        apply_convolution_forward(implementation, d_in, d_out, d_tmp, d_norm,
                                  Nx, Ny, Nz, kx, ky, kz);
    }
}

__global__ void k_butter_mul(cufftComplex* data, const float* filt, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float f = filt[i];
        data[i].x *= f;
        data[i].y *= f;
    }
}

/*
 * SPECT 文献中常用的三维 Butterworth 后滤波：
 *
 *   H(f) = 1 / (1 + (f/fc)^(2n)),
 *   f = sqrt(fx^2 + fy^2 + fz^2)。
 *
 * fc 使用 cycles/cm，转换为 cycles/mm 后与每轴真实物理频率比较。旧实现把
 * 三维对角 Nyquist 归一化为 1，同一个参数会随维数定义改变，无法与论文或不同
 * 体素大小的数据比较，因此不再保留。每轴使用 2 倍零填充，并把原图放在补零
 * 体积中央，避免未补零 FFT 把图像两侧当作相邻位置进行循环卷积。
 */
struct ButterworthWorkspace {
    int nx = 0, ny = 0, nz = 0;
    int padded_nx = 0, padded_ny = 0, padded_nz = 0;
    int offset_x = 0, offset_y = 0, offset_z = 0;
    int64_t padded_real_count = 0;
    int64_t spectrum_count = 0;
    float* d_padded_real = nullptr;
    cufftComplex* d_spectrum = nullptr;
    float* d_response = nullptr;
    cufftHandle forward_plan = 0;
    cufftHandle inverse_plan = 0;
};

__global__ void k_butter_embed_centered(const float* input, float* padded,
                                        int nx, int ny, int nz,
                                        int padded_nx, int padded_ny,
                                        int offset_x, int offset_y, int offset_z,
                                        int64_t input_count)
{
    const int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= input_count) return;
    const int z = (int)(index / ((int64_t)nx * ny));
    const int rem = (int)(index - (int64_t)z * nx * ny);
    const int y = rem / nx;
    const int x = rem - y * nx;
    const int64_t padded_index =
        (int64_t)(z + offset_z) * padded_nx * padded_ny +
        (int64_t)(y + offset_y) * padded_nx + x + offset_x;
    padded[padded_index] = input[index];
}

__global__ void k_butter_crop_centered(const float* padded, float* output,
                                       int nx, int ny, int nz,
                                       int padded_nx, int padded_ny,
                                       int offset_x, int offset_y, int offset_z,
                                       float inverse_fft_size,
                                       bool clamp_for_next_em_iteration,
                                       int64_t output_count)
{
    const int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_count) return;
    const int z = (int)(index / ((int64_t)nx * ny));
    const int rem = (int)(index - (int64_t)z * nx * ny);
    const int y = rem / nx;
    const int x = rem - y * nx;
    const int64_t padded_index =
        (int64_t)(z + offset_z) * padded_nx * padded_ny +
        (int64_t)(y + offset_y) * padded_nx + x + offset_x;
    float value = padded[padded_index] * inverse_fft_size;
    if (!isfinite(value)) value = 0.0f;
    // 只有还要进入下一轮 EM 时才恢复非负定义域；最终后滤波保持线性结果。
    if (clamp_for_next_em_iteration && value < 0.0f) value = 0.0f;
    output[index] = value;
}

static std::vector<float> build_butterworth_response(
    int padded_nx, int padded_ny, int padded_nz,
    float voxel_mm, float cutoff_cycles_per_cm, int order)
{
    const int freq_nx = padded_nx / 2 + 1;
    const int64_t spectrum_count =
        (int64_t)padded_nz * padded_ny * freq_nx;
    std::vector<float> response((size_t)spectrum_count, 0.0f);
    const double cutoff_cycles_per_mm =
        (double)cutoff_cycles_per_cm / 10.0;
    for (int z = 0; z < padded_nz; ++z) {
        const int kz = (z <= padded_nz / 2) ? z : z - padded_nz;
        const double fz = (double)kz / ((double)padded_nz * voxel_mm);
        for (int y = 0; y < padded_ny; ++y) {
            const int ky = (y <= padded_ny / 2) ? y : y - padded_ny;
            const double fy = (double)ky / ((double)padded_ny * voxel_mm);
            for (int x = 0; x < freq_nx; ++x) {
                const double fx = (double)x / ((double)padded_nx * voxel_mm);
                const double frequency = std::sqrt(fx * fx + fy * fy + fz * fz);
                const double ratio = frequency / cutoff_cycles_per_mm;
                const double power = std::pow(ratio, 2.0 * (double)order);
                const int64_t index =
                    (int64_t)z * padded_ny * freq_nx + (int64_t)y * freq_nx + x;
                response[(size_t)index] =
                    std::isfinite(power) ? (float)(1.0 / (1.0 + power)) : 0.0f;
            }
        }
    }
    return response;
}

static void initialize_butterworth_workspace(
    ButterworthWorkspace& workspace, int nx, int ny, int nz,
    float voxel_mm, float cutoff_cycles_per_cm, int order)
{
    workspace.nx = nx;
    workspace.ny = ny;
    workspace.nz = nz;
    workspace.padded_nx = 2 * nx;
    workspace.padded_ny = 2 * ny;
    workspace.padded_nz = 2 * nz;
    workspace.offset_x = (workspace.padded_nx - nx) / 2;
    workspace.offset_y = (workspace.padded_ny - ny) / 2;
    workspace.offset_z = (workspace.padded_nz - nz) / 2;
    workspace.padded_real_count =
        (int64_t)workspace.padded_nx * workspace.padded_ny * workspace.padded_nz;
    workspace.spectrum_count =
        (int64_t)workspace.padded_nz * workspace.padded_ny *
        (workspace.padded_nx / 2 + 1);

    const std::vector<float> response = build_butterworth_response(
        workspace.padded_nx, workspace.padded_ny, workspace.padded_nz,
        voxel_mm, cutoff_cycles_per_cm, order);
    CUDA_CHECK(cudaMalloc(&workspace.d_padded_real,
                          (size_t)workspace.padded_real_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&workspace.d_spectrum,
                          (size_t)workspace.spectrum_count * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&workspace.d_response,
                          (size_t)workspace.spectrum_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(workspace.d_response, response.data(),
                          response.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUFFT_CHECK(cufftPlan3d(&workspace.forward_plan,
                            workspace.padded_nz,
                            workspace.padded_ny,
                            workspace.padded_nx,
                            CUFFT_R2C));
    CUFFT_CHECK(cufftPlan3d(&workspace.inverse_plan,
                            workspace.padded_nz,
                            workspace.padded_ny,
                            workspace.padded_nx,
                            CUFFT_C2R));
}

static void apply_butterworth_filter(ButterworthWorkspace& workspace,
                                     const float* input, float* output,
                                     bool clamp_for_next_em_iteration)
{
    const int64_t input_count =
        (int64_t)workspace.nx * workspace.ny * workspace.nz;
    const int input_blocks = (int)((input_count + BLK - 1) / BLK);
    const int spectrum_blocks =
        (int)((workspace.spectrum_count + BLK - 1) / BLK);
    CUDA_CHECK(cudaMemset(workspace.d_padded_real, 0,
                          (size_t)workspace.padded_real_count * sizeof(float)));
    k_butter_embed_centered<<<input_blocks, BLK>>>(
        input, workspace.d_padded_real,
        workspace.nx, workspace.ny, workspace.nz,
        workspace.padded_nx, workspace.padded_ny,
        workspace.offset_x, workspace.offset_y, workspace.offset_z,
        input_count);
    CUDA_CHECK(cudaGetLastError());
    CUFFT_CHECK(cufftExecR2C(workspace.forward_plan, workspace.d_padded_real,
                             workspace.d_spectrum));
    k_butter_mul<<<spectrum_blocks, BLK>>>(
        workspace.d_spectrum, workspace.d_response, workspace.spectrum_count);
    CUDA_CHECK(cudaGetLastError());
    CUFFT_CHECK(cufftExecC2R(workspace.inverse_plan, workspace.d_spectrum,
                             workspace.d_padded_real));
    k_butter_crop_centered<<<input_blocks, BLK>>>(
        workspace.d_padded_real, output,
        workspace.nx, workspace.ny, workspace.nz,
        workspace.padded_nx, workspace.padded_ny,
        workspace.offset_x, workspace.offset_y, workspace.offset_z,
        1.0f / (float)workspace.padded_real_count,
        clamp_for_next_em_iteration, input_count);
    CUDA_CHECK(cudaGetLastError());
}

static void destroy_butterworth_workspace(ButterworthWorkspace& workspace)
{
    if (workspace.inverse_plan) CUFFT_CHECK(cufftDestroy(workspace.inverse_plan));
    if (workspace.forward_plan) CUFFT_CHECK(cufftDestroy(workspace.forward_plan));
    if (workspace.d_response) CUDA_CHECK(cudaFree(workspace.d_response));
    if (workspace.d_spectrum) CUDA_CHECK(cudaFree(workspace.d_spectrum));
    if (workspace.d_padded_real) CUDA_CHECK(cudaFree(workspace.d_padded_real));
    workspace = ButterworthWorkspace{};
}

// Butterworth 的 start=0 保留旧版调度：periodic 按全局迭代编号整除。
// start>0 时，start 是第一次允许滤波的轮次，periodic 也从该轮开始计周期。
static bool should_apply_filter(int iteration, int total_iterations, const std::string& schedule_mode, int first_iteration, int interval, int trailing_count)
{
    if (first_iteration > 0 && iteration < first_iteration) return false;
    if (schedule_mode == "every") return true;
    if (schedule_mode == "final") return iteration == total_iterations;
    if (schedule_mode == "periodic") {
        const bool on_period = interval > 0 &&
            (first_iteration > 0
                ? (iteration - first_iteration) % interval == 0
                : iteration % interval == 0);
        return on_period || iteration == total_iterations;
    }
    if (schedule_mode == "last_n") {
        return iteration > total_iterations - trailing_count;
    }
    return false;
}

// smooth 的历史行为本来就是从 smooth_start_iter 起按相对周期执行，因此默认
// periodic 完全兼容旧命令；final 只在最后一次 EM 更新后回写一次平滑结果。
static bool should_apply_smooth(int iteration, int total_iterations, const std::string& schedule_mode, int first_iteration, int interval)
{
    if (iteration < first_iteration) return false;
    if (schedule_mode == "every") return true;
    if (schedule_mode == "final") return iteration == total_iterations;
    return interval > 0 &&
        (iteration - first_iteration) % interval == 0;
}

__device__ __forceinline__ void tv_forward_normalized(
    const float* __restrict__ f, int x, int y, int z,
    int Nx, int Ny, int Nz, float smoothing,
    float& px, float& py, float& pz)
{
    const int64_t plane = (int64_t)Nx * Ny;
    const int64_t idx = (int64_t)z * plane + (int64_t)y * Nx + x;
    const float center = f[idx];
    const float dx = (x + 1 < Nx) ? f[idx + 1] - center : 0.0f;
    const float dy = (y + 1 < Ny) ? f[idx + Nx] - center : 0.0f;
    const float dz = (z + 1 < Nz) ? f[idx + plane] - center : 0.0f;
    float denom = sqrtf(dx*dx + dy*dy + dz*dz + smoothing);
    if (!isfinite(denom) || denom < 1e-12f) denom = 1e-12f;
    px = dx / denom;
    py = dy / denom;
    pz = dz / denom;
}

/*
 * OMEGA 普通非解剖 isotropic TV：先计算 x/y/z 前向差分并以
 * sqrt(dx^2+dy^2+dz^2+TVsmoothing) 归一化，再取负散度。边界使用零法向
 * 前向差分，相当于 Neumann 边界；常量图像的梯度严格为零。
 */
__global__ void k_tv_gradient_omega(const float* __restrict__ f,
                                    float* __restrict__ grad,
                                    int Nx, int Ny, int Nz,
                                    float smoothing)
{
    const int64_t n = (int64_t)Nx * Ny * Nz;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int64_t plane = (int64_t)Nx * Ny;
    const int z = (int)(idx / plane);
    const int rem = (int)(idx - (int64_t)z * plane);
    const int y = rem / Nx;
    const int x = rem - y * Nx;

    float px, py, pz;
    tv_forward_normalized(f, x, y, z, Nx, Ny, Nz, smoothing, px, py, pz);
    float value = -(px + py + pz);
    if (x > 0) {
        float qx, qy, qz;
        tv_forward_normalized(f, x - 1, y, z, Nx, Ny, Nz,
                              smoothing, qx, qy, qz);
        value += qx;
    }
    if (y > 0) {
        float qx, qy, qz;
        tv_forward_normalized(f, x, y - 1, z, Nx, Ny, Nz,
                              smoothing, qx, qy, qz);
        value += qy;
    }
    if (z > 0) {
        float qx, qy, qz;
        tv_forward_normalized(f, x, y, z - 1, Nx, Ny, Nz,
                              smoothing, qx, qy, qz);
        value += qz;
    }
    grad[idx] = isfinite(value) ? value : 0.0f;
}

#define MAX_MRP_RADIUS 2
#define MAX_MRP_WINDOW 125

__device__ __forceinline__ int mrp_symmetric_index(int value, int size) {
    while (value < 0 || value >= size) {
        value = (value < 0) ? -value - 1 : 2 * size - value - 1;
    }
    return value;
}

/*
 * OMEGA MRP：对称填充后取 (2*rx+1)*(2*ry+1)*(2*rz+1) 邻域中值。
 * 默认输出 (x-median)/(median+epsilon)；mrp_no_norm=true 时只输出差值。
 * 固定上限避免运行时动态数组，半径 0..2 对应最多 125 个样本。
 */
__global__ void k_mrp_gradient_omega(const float* __restrict__ f,
                                     float* __restrict__ grad,
                                     int Nx, int Ny, int Nz,
                                     int rx, int ry, int rz,
                                     float epsilon, bool mrp_no_norm)
{
    const int64_t n = (int64_t)Nx * Ny * Nz;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int64_t plane = (int64_t)Nx * Ny;
    const int z = (int)(idx / plane);
    const int rem = (int)(idx - (int64_t)z * plane);
    const int y = rem / Nx;
    const int x = rem - y * Nx;

    float values[MAX_MRP_WINDOW];
    int count = 0;
    for (int dz = -rz; dz <= rz; ++dz) {
        const int zz = mrp_symmetric_index(z + dz, Nz);
        for (int dy = -ry; dy <= ry; ++dy) {
            const int yy = mrp_symmetric_index(y + dy, Ny);
            for (int dx = -rx; dx <= rx; ++dx) {
                const int xx = mrp_symmetric_index(x + dx, Nx);
                float value = f[(int64_t)zz * plane + (int64_t)yy * Nx + xx];
                if (!isfinite(value)) value = 0.0f;
                int insert = count;
                while (insert > 0 && values[insert - 1] > value) {
                    values[insert] = values[insert - 1];
                    --insert;
                }
                values[insert] = value;
                ++count;
            }
        }
    }
    const float median = values[count / 2];
    const float diff = f[idx] - median;
    const float value = mrp_no_norm ? diff : diff / (median + epsilon);
    grad[idx] = isfinite(value) ? value : 0.0f;
}

__device__ __forceinline__ bool voxel_inside_centered_cylinder(
    int x, int y, int z, int Nx, int Ny, int Nz, float voxel_mm,
    float roi_diameter_mm, float roi_height_mm) {
    const float xx = ((float)x - 0.5f * (float)(Nx - 1)) * voxel_mm;
    const float yy = ((float)y - 0.5f * (float)(Ny - 1)) * voxel_mm;
    const float zz = ((float)z - 0.5f * (float)(Nz - 1)) * voxel_mm;
    const float radius = 0.5f * roi_diameter_mm;
    return xx * xx + yy * yy <= radius * radius &&
           fabsf(zz) <= 0.5f * roi_height_mm;
}

/*
 * CASToR iPenaltyMedianRootPrior 的邻域和边界语义：
 * - 中心体素包含在邻域中；
 * - 图像范围外或被重建 ROI/mask 屏蔽的邻居直接丢弃，不做镜像/补零；
 * - 有偶数个有效值时，中值取中间两项平均；
 * - median==0 时一阶导数为 0，否则为 (x-median)/median；
 * - sphere 包围半径按 max(1,floor(radius_mm/voxel_mm)) 计算，体素中心距离
 *   必须严格小于 radius，等于半径的体素不进入邻域。
 *
 * 为控制 CUDA 每线程局部数组，box order 和 sphere 的轴向包围半径均限制为 2，
 * 因而有效样本最多 5^3=125。该限制只缩小可配置范围，不改变范围内的公式。
 */
__global__ void k_mrp_gradient_castor(
    const float* __restrict__ f, float* __restrict__ grad,
    int Nx, int Ny, int Nz, int neighborhood_shape, int box_order,
    bool box_exclude_corners, float sphere_radius_mm, float voxel_mm,
    float roi_diameter_mm, float roi_height_mm) {
    const int64_t n = (int64_t)Nx * Ny * Nz;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int64_t plane = (int64_t)Nx * Ny;
    const int z = (int)(idx / plane);
    const int rem = (int)(idx - (int64_t)z * plane);
    const int y = rem / Nx;
    const int x = rem - y * Nx;
    if (!voxel_inside_centered_cylinder(
            x, y, z, Nx, Ny, Nz, voxel_mm,
            roi_diameter_mm, roi_height_mm)) {
        grad[idx] = 0.0f;
        return;
    }

    int extent = 1;
    if (neighborhood_shape == MRP_NEIGHBORHOOD_BOX) {
        extent = box_order;
    } else if (neighborhood_shape == MRP_NEIGHBORHOOD_SPHERE) {
        const int radius_voxels = (int)(sphere_radius_mm / voxel_mm);
        extent = radius_voxels > 1 ? radius_voxels : 1;
    }
    if (extent > MAX_MRP_RADIUS) {
        grad[idx] = 0.0f;
        return;
    }

    float values[MAX_MRP_WINDOW];
    int count = 0;
    for (int dz = -extent; dz <= extent; ++dz) {
        for (int dy = -extent; dy <= extent; ++dy) {
            for (int dx = -extent; dx <= extent; ++dx) {
                bool include = false;
                if (neighborhood_shape == MRP_NEIGHBORHOOD_SIX_NEAREST) {
                    include = abs(dx) + abs(dy) + abs(dz) <= 1;
                } else if (neighborhood_shape == MRP_NEIGHBORHOOD_BOX) {
                    include = true;
                    if (box_exclude_corners && box_order > 0 &&
                        abs(dx) == box_order && abs(dy) == box_order &&
                        abs(dz) == box_order) {
                        include = false;
                    }
                } else {
                    const float distance2 = (float)(dx * dx + dy * dy + dz * dz)
                                          * voxel_mm * voxel_mm;
                    include = distance2 < sphere_radius_mm * sphere_radius_mm;
                }
                if (!include) continue;
                const int xx = x + dx;
                const int yy = y + dy;
                const int zz = z + dz;
                if (xx < 0 || xx >= Nx || yy < 0 || yy >= Ny ||
                    zz < 0 || zz >= Nz) continue;
                if (!voxel_inside_centered_cylinder(
                        xx, yy, zz, Nx, Ny, Nz, voxel_mm,
                        roi_diameter_mm, roi_height_mm)) continue;

                float value = f[(int64_t)zz * plane + (int64_t)yy * Nx + xx];
                if (!isfinite(value)) value = 0.0f;
                int insert = count;
                while (insert > 0 && values[insert - 1] > value) {
                    values[insert] = values[insert - 1];
                    --insert;
                }
                values[insert] = value;
                ++count;
            }
        }
    }

    if (count <= 0) {
        grad[idx] = 0.0f;
        return;
    }
    float median = values[count / 2];
    if ((count & 1) == 0) {
        median = 0.5f * (median + values[count / 2 - 1]);
    }
    const float value = median != 0.0f ? (f[idx] - median) / median : 0.0f;
    grad[idx] = isfinite(value) ? value : 0.0f;
}

__global__ void k_map_gradient(const float* __restrict__ f,
                               float* __restrict__ grad,
                               int Nx, int Ny, int Nz,
                               int map_type,
                               int neigh26,
                               float delta,
                               float huber_T,
                               float rel_gamma,
                               float rel_eps)
{
    int64_t N2 = (int64_t)Nx * Ny * Nz;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;

    int iz = (int)(idx / ((int64_t)Nx * Ny));
    int rem = (int)(idx - (int64_t)iz * Nx * Ny);
    int iy = rem / Nx;
    int ix = rem - iy * Nx;
    float fi = f[idx];

    float g = 0.0f;

    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        if (!neigh26 && abs(dx) + abs(dy) + abs(dz) != 1) continue;
        int x = ix + dx, y = iy + dy, z = iz + dz;
        if (x < 0 || x >= Nx || y < 0 || y >= Ny || z < 0 || z >= Nz) continue;
        float w = rsqrtf((float)(dx*dx + dy*dy + dz*dz));
        float fj = f[(int64_t)z*Nx*Ny + y*Nx + x];
        float d = fi - fj;

        if (map_type == MAP_QUADRATIC) {
            g += 2.0f * w * d;
        } else if (map_type == MAP_HUBER) {
            float ad = fabsf(d);
            float gc = (ad <= huber_T) ? (2.0f * d) : (2.0f * huber_T * ((d >= 0) ? 1.0f : -1.0f));
            g += w * gc;
        } else if (map_type == MAP_RELATIVE) {
            float dn = fi + fj + rel_gamma * fabsf(d) + rel_eps;
            float sgn = (d > 0) ? 1.0f : ((d < 0) ? -1.0f : 0.0f);
            float gc = 2.0f*d/dn - d*d*(1.0f + rel_gamma*sgn)/(dn*dn);
            g += w * gc;
        } else if (map_type == MAP_LOGCOSH) {
            float dl = fmaxf(delta, 1e-12f);
            g += w * tanhf(d / dl) / dl;
        } else if (map_type == MAP_HYPERBOLIC) {
            float dl = fmaxf(delta, 1e-12f);
            float den = sqrtf(d*d + dl*dl);
            g += w * d / den;
        }
    }
    grad[idx] = g;
}

class GPUWorker {
public:
    int dev = 0;
    int start_ck = 0;
    int end_ck = 0;
    int local_rows = 0;
    int N2 = 0;
    int Nx = 0, Ny = 0, Nz = 0;
    int compact_rpa = 0;
    int det_nx = 0, det_nz = 0, compact_det_z = 0;

    int64_t nnz = 0;
    bool csr_index64 = false;
    int32_t* d_row32 = nullptr;
    int64_t* d_row64 = nullptr;
    int32_t* d_col32 = nullptr;
    int64_t* d_col64 = nullptr;
    float* d_val = nullptr;

    float* d_img_in = nullptr;
    float* d_img_rot = nullptr;
    float* d_y = nullptr;
    float* d_ratio = nullptr;
    unsigned long long* d_low_forward_count = nullptr;
    float* d_bp_rot = nullptr;
    float* d_bp_part = nullptr;
    float* d_sens_part = nullptr;

    float* d_P_all = nullptr;
    size_t proj_stride = 0;
    int num_beds = 0;
    float* d_G_full = nullptr;

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t mat = nullptr;
    cusparseDnVecDescr_t vec_img = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    cusparseDnVecDescr_t vec_ratio = nullptr;
    cusparseDnVecDescr_t vec_bp = nullptr;
    void* d_spmv_buf = nullptr;
    size_t spmv_buf_size = 0;

    void init(int device_id, const HostCSR64& hcsr, const MatrixMeta& meta,
              const std::vector<float>& P_all, size_t projection_stride,
              int bed_count, const std::vector<float>& G_full,
              bool track_low_forward = false)
    {
        dev = device_id;
        start_ck = hcsr.start_ck;
        end_ck = hcsr.end_ck;
        if (hcsr.num_rows > 2147483647LL || hcsr.num_cols > 2147483647LL) {
            fprintf(stderr,
                    "Local compact CSR rows/cols are too large for current cuSPARSE usage: rows=%" PRId64 " cols=%" PRId64 " nnz=%" PRId64 "\n",
                    hcsr.num_rows, hcsr.num_cols, hcsr.nnz);
            exit(1);
        }
        local_rows = (int)hcsr.num_rows;
        nnz = hcsr.nnz;
        Nx = meta.Nx; Ny = meta.Ny; Nz = meta.Nz;
        N2 = meta.total_voxels;
        compact_rpa = meta.compact_rpa;
        det_nx = meta.det_nx;
        det_nz = meta.det_nz;
        compact_det_z = meta.compact_det_z;
        proj_stride = projection_stride;
        num_beds = bed_count;
        if (num_beds <= 0 || proj_stride == 0 ||
            P_all.size() != proj_stride * (size_t)num_beds) {
            fprintf(stderr,
                    "Invalid multi-bed projection buffer: beds=%d stride=%zu total=%zu\n",
                    num_beds, proj_stride, P_all.size());
            exit(1);
        }

        CUDA_CHECK(cudaSetDevice(dev));
        CUSPARSE_CHECK(cusparseCreate(&handle));

        csr_index64 = (nnz > 2147483647LL);

        if (csr_index64) {

            CUDA_CHECK(cudaMalloc(&d_row64, ((size_t)local_rows + 1) * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&d_col64, (size_t)nnz * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&d_val,   (size_t)nnz * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_row64, hcsr.row_ptr.data(),
                                  ((size_t)local_rows + 1)*sizeof(int64_t),
                                  cudaMemcpyHostToDevice));

            const size_t CHUNK = 1u << 24;
            std::vector<int64_t> tmp64;
            tmp64.resize(std::min((size_t)nnz, CHUNK));
            size_t done = 0;
            while (done < (size_t)nnz) {
                size_t ncopy = std::min(CHUNK, (size_t)nnz - done);
                for (size_t i = 0; i < ncopy; ++i) {
                    tmp64[i] = (int64_t)hcsr.col_idx[done + i];
                }
                CUDA_CHECK(cudaMemcpy(d_col64 + done, tmp64.data(),
                                      ncopy * sizeof(int64_t),
                                      cudaMemcpyHostToDevice));
                done += ncopy;
            }
            CUDA_CHECK(cudaMemcpy(d_val, hcsr.values.data(),
                                  (size_t)nnz*sizeof(float),
                                  cudaMemcpyHostToDevice));
        } else {
            std::vector<int32_t> row32((size_t)local_rows + 1);
            for (int i = 0; i <= local_rows; i++) {
                row32[(size_t)i] = (int32_t)hcsr.row_ptr[(size_t)i];
            }
            CUDA_CHECK(cudaMalloc(&d_row32, ((size_t)local_rows + 1) * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_col32, (size_t)nnz * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_val,   (size_t)nnz * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_row32, row32.data(),
                                  ((size_t)local_rows + 1)*sizeof(int32_t),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_col32, hcsr.col_idx.data(),
                                  (size_t)nnz*sizeof(int32_t),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_val, hcsr.values.data(),
                                  (size_t)nnz*sizeof(float),
                                  cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaMalloc(&d_img_in,    (size_t)N2*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_img_rot,   (size_t)N2*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bp_rot,    (size_t)N2*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bp_part,   (size_t)N2*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sens_part, (size_t)N2*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y,         (size_t)local_rows*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ratio,     (size_t)local_rows*sizeof(float)));
        // 只有 CASToR threshold 路径需要该计数器；OMEGA 不分配未使用的显存。
        if (track_low_forward) {
            CUDA_CHECK(cudaMalloc(&d_low_forward_count,
                                  sizeof(unsigned long long)));
            CUDA_CHECK(cudaMemset(d_low_forward_count, 0,
                                  sizeof(unsigned long long)));
        }
        CUDA_CHECK(cudaMalloc(&d_P_all,     P_all.size()*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_P_all, P_all.data(), P_all.size()*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_G_full,    G_full.size()*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_G_full, G_full.data(), G_full.size()*sizeof(float), cudaMemcpyHostToDevice));

        if (csr_index64) {
            CUSPARSE_CHECK(cusparseCreateCsr(&mat,
                local_rows, N2, nnz,
                d_row64, d_col64, d_val,
                CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        } else {
            CUSPARSE_CHECK(cusparseCreateCsr(&mat,
                local_rows, N2, nnz,
                d_row32, d_col32, d_val,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        }

        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_img,   N2,         d_img_rot, CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y,     local_rows, d_y,       CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_ratio, local_rows, d_ratio,   CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_bp,    N2,         d_bp_rot,  CUDA_R_32F));

        float alpha = 1.0f, beta = 0.0f;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat, vec_img, &beta, vec_y, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buf_size));
        size_t bwd_buf_size = 0;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, mat, vec_ratio, &beta, vec_bp, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, &bwd_buf_size));
        spmv_buf_size = std::max(spmv_buf_size, bwd_buf_size);
        if (spmv_buf_size > 0) CUDA_CHECK(cudaMalloc(&d_spmv_buf, spmv_buf_size));

        printf("  GPU%d compact angles [%d,%d), rows=%d, nnz=%.2fM, csr_index=%s\n",
               dev, start_ck, end_ck, local_rows, (double)nnz/1e6,
               csr_index64 ? "64/64" : "32/32");
    }

    void spmv_forward() {
        float alpha = 1.0f, beta = 0.0f;
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat, vec_img, &beta, vec_y, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buf));
    }

    void spmv_backward() {
        float alpha = 1.0f, beta = 0.0f;
        CUDA_CHECK(cudaMemset(d_bp_rot, 0, (size_t)N2*sizeof(float)));
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, mat, vec_ratio, &beta, vec_bp, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buf));
    }

    // 每轮开始前清零；读取时同步当前 device，供主机端按轮次汇总报警。
    void reset_low_forward_count() {
        if (!d_low_forward_count) return;
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaMemset(d_low_forward_count, 0,
                              sizeof(unsigned long long)));
    }

    unsigned long long read_low_forward_count() const {
        if (!d_low_forward_count) return 0;
        CUDA_CHECK(cudaSetDevice(dev));
        unsigned long long count = 0;
        CUDA_CHECK(cudaMemcpy(&count, d_low_forward_count, sizeof(count),
                              cudaMemcpyDeviceToHost));
        return count;
    }

    // 多床位复用同一系统矩阵时，只替换投影和响应因子，不重新创建 CSR。
    void update_projection(const std::vector<float>& P_all_new,
                           size_t projection_stride_new,
                           const std::vector<float>& G_full_new) {
        if (projection_stride_new != proj_stride ||
            P_all_new.size() != proj_stride ||
            G_full_new.size() != proj_stride) {
            fprintf(stderr, "Persistent GPU matrix projection shape mismatch\n");
            exit(1);
        }
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaMemcpy(d_P_all, P_all_new.data(),
                              P_all_new.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_G_full, G_full_new.data(),
                              G_full_new.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        num_beds = 1;
    }

    void destroy() {
        CUDA_CHECK(cudaSetDevice(dev));
        if (vec_bp)    cusparseDestroyDnVec(vec_bp);
        if (vec_ratio) cusparseDestroyDnVec(vec_ratio);
        if (vec_y)     cusparseDestroyDnVec(vec_y);
        if (vec_img)   cusparseDestroyDnVec(vec_img);
        if (mat)       cusparseDestroySpMat(mat);
        if (handle)    cusparseDestroy(handle);
        if (d_spmv_buf) cudaFree(d_spmv_buf);
        if (d_G_full) cudaFree(d_G_full);
        if (d_P_all) cudaFree(d_P_all);
        if (d_low_forward_count) cudaFree(d_low_forward_count);
        if (d_ratio) cudaFree(d_ratio);
        if (d_y) cudaFree(d_y);
        if (d_sens_part) cudaFree(d_sens_part);
        if (d_bp_part) cudaFree(d_bp_part);
        if (d_bp_rot) cudaFree(d_bp_rot);
        if (d_img_rot) cudaFree(d_img_rot);
        if (d_img_in) cudaFree(d_img_in);
        if (d_val) cudaFree(d_val);
        if (d_col64) cudaFree(d_col64);
        if (d_col32) cudaFree(d_col32);
        if (d_row64) cudaFree(d_row64);
        if (d_row32) cudaFree(d_row32);
    }
};

// bedmerge 的每个 BedN 只更换投影数据，CSR 和 cuSPARSE 对象在整个进程内驻留。
struct ResidentMatrixCache {
    bool valid = false;
    std::string matrix_dir;
    int gpus = 0;
    int nx = 0, ny = 0, nz = 0;
    int compact_angles = 0;
    std::vector<GPUWorker> workers;

    void clear() {
        for (auto& worker : workers) worker.destroy();
        workers.clear();
        valid = false;
        matrix_dir.clear();
        gpus = 0;
    }
};

static ResidentMatrixCache g_bedmerge_matrix_cache;

// -------------------- 底层 CUDA 重建命令行 --------------------
static void print_usage() {
    printf("Usage:\n");
    printf("  spect_recon_PSC_6module_24pin_YSO_multibed --matrix-dir DIR --proj FILE_OR_BED_ROOT [options]\n\n");
    printf("Required:\n");
    printf("  --matrix-dir DIR          directory containing metadata.bin and chunk_*.csrb (or legacy chunk_*.bin)\n");
    printf("  --proj PATH               single projection file, flat Bed<N>.bin root, or bed1/bed2/... root\n\n");
    printf("Options:\n");
    printf("  --out DIR                 output directory [auto: recon_<file> or recon_multibed_<folder>]\n");
    printf("  output                    NIfTI-1 paired .hdr/.img float64 plus RAW1 .raw\n");
    printf("  --no-raw1                 do not save the default RAW1 .raw companion file\n");
    printf("  --norm-factor FILE        optional normalization factor bin; if omitted, search matrix_dir/normalization_factors/\n");
    printf("  --norm-factor-mode MODE   auto|matrix|projection [auto]; auto reads <factor_file>.mode\n");
    printf("                             matrix: g_i=factor; projection: g_i=1/factor\n");
    printf("  --norm-factor-invert      legacy alias of --norm-factor-mode projection\n");
    printf("  --bed-proj-name NAME      projection filename inside each bed directory [projection.bin]\n");
    printf("  --bed-step-mm F           equal spacing; Bed1=max z and bed number increases toward -z\n");
    printf("                             required for >1 bed unless offsets are given\n");
    printf("  --bed-offsets-mm LIST     explicit centers in Bed1,Bed2,... order, e.g. 18,0,-18\n");
    printf("                             overrides --bed-step-mm and preserves the supplied signs\n");
    printf("  --roi-cylinder            enable cylindrical image support crop [default on]\n");
    printf("  --no-roi, --no-roi-cylinder disable cylindrical image support crop\n");
    printf("  --roi-diameter-mm F       cylindrical support diameter in mm [30.0]\n");
    printf("  --roi-height-mm F         valid axial FOV per bed in mm [66.0]; multi-bed uses interval union\n");
    printf("  --multibed-edge-layers N  axial layers retained beyond each end bed center [40]\n");
    printf("  --no-detector-mask        keep LINK4/FPGA2/DET4 bins (for ideal projections)\n");
    printf("  --data-mode MODE          measured or simulation\n");
    printf("  --no-norm-factor          use g_i=1\n");
    printf("  --roi-center-mm X,Y,Z     cylindrical support center offset in mm [0,0,0]\n");
    printf("  --no-roi-each-iter        only mask initialization/saved output, not every iterative update\n");
    printf("                             also accepts: --roi-center-mm X Y Z\n");
    printf("  --check-sens-zsym         report sensitivity-image z <-> Nz-1-z symmetry and save edge PNGs\n");
    printf("  --save-sens-edge-png      save sensitivity axial first/last slice PNGs without running z-symmetry report\n");
    printf("  --sens-zsym-tol F         relative tolerance for sensitivity z-symmetry report [1e-4]\n");
    printf("  --iters N                 iterations [50]\n");
    printf("  --gpus N                  number of GPUs [1]\n");
    printf("  --angle-batch N           single-GPU compact-angle batch size; 0=disabled [0]\n");
    printf("  --slab-prefetch           overlap next Host CSR build with current GPU slab [default on]\n");
    printf("  --no-slab-prefetch        disable overlap for A/B timing or constrained host memory\n");
    printf("  --slab-prefetch-early     also overlap GPU setup; holds current+next Host CSR simultaneously\n");
    printf("  --gpu-slab-cache N        keep first N angle slabs resident on GPU [0]\n");
    printf("  --save-every N            save reconN every N iterations; no recon_final [50]\n");
    printf("  --sym auto|none|zonly|180|z180|90|45|45z [auto]\n");
    printf("  --recon-angle-step N      use every N-th full angle [1]\n");
    printf("  --conv-mode none|psf|post|sieve [none]\n");
    printf("  --psf-fwhm F or Fx,Fy,Fz  PSF FWHM in mm [1.5,1.5,1.5]\n");
    printf("  --psf-sigmas F            PSF truncation nbSigmas [3]\n");
    printf("  --post-fwhm F             post Gaussian FWHM [0.45]\n");
    printf("  --post-schedule constant|linear|exponential [constant]\n");
    printf("  --post-fwhm-start F       linear/exponential start [0.60]\n");
    printf("  --post-fwhm-end F         linear/exponential end [0.15]\n");
    printf("  --post-start-iter N       first post iteration [1]\n");
    printf("  --post-every N            post apply period [1]\n");
    printf("  --map-type none|quadratic|tv|huber|relative|logcosh|hyperbolic [none]\n");
    printf("  --map-beta F              MAP beta [0]\n");
    printf("  --map-neigh 6|26          MAP neighborhood [26]\n");
    printf("  --map-delta F             TV/logcosh/hyperbolic delta [1e-6]\n");
    printf("  --map-huber-threshold F   Huber threshold [0.01]\n");
    printf("  --map-relative-gamma F    relative difference gamma [2]\n");
    printf("  --map-relative-epsilon F  relative difference epsilon [1e-6]\n");
    printf("  --butter-cutoff-cm F      3D Butterworth cutoff, cycles/cm; omitted=off\n");
    printf("  --butter-order N          Butterworth order [8]\n");
    printf("  --butter-mode every|final|periodic|last_n [final]\n");
    printf("  --butter-start-iter N     first eligible filtering iteration [0=legacy schedule]\n");
    printf("  --butter-period N         periodic interval [5]\n");
    printf("  --butter-last-n N         last_n window [10]\n");
}

static ReconConfig parse_args(int argc, char** argv) {
    ReconConfig c;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char* name){
            if (i+1 >= argc) { fprintf(stderr, "%s needs a value\n", name); exit(1); }
            return argv[++i];
        };
        if      (a == "--matrix-dir") c.matrix_dir = need("--matrix-dir");
        else if (a == "--proj") c.proj_path = need("--proj");
        else if (a == "--bed-proj-name") c.bed_projection_name = need("--bed-proj-name");
        else if (a == "--bed-step-mm" || a == "--bed-step") {
            c.bed_step_mm = (float)atof(need(a.c_str()));
            c.bed_step_set = true;
        }
        else if (a == "--bed-offsets-mm" || a == "--bed-centers-mm") {
            c.bed_offsets_mm = parse_float_list(need(a.c_str()));
        }
        else if (a == "--multibed-edge-layers") {
            c.multibed_edge_layers = atoi(need("--multibed-edge-layers"));
        }
        else if (a == "--out") {
            c.output_dir = need("--out");
            c.output_dir_explicit = true;
        }
        else if (a == "--legacy-raw1" || a == "--legacy-raw" || a == "--raw1") c.save_raw1 = true;
        else if (a == "--no-raw1" || a == "--no-raw") c.save_raw1 = false;
        else if (a == "--norm-factor") c.norm_factor_path = need("--norm-factor");
        else if (a == "--norm-factor-mode" || a == "--norm-mode") c.norm_factor_mode = need(a.c_str());
        else if (a == "--norm-factor-invert" || a == "--norm-invert") c.norm_factor_invert = true;
        else if (a == "--roi-cylinder" || a == "--roi") c.roi_enable = true;
        else if (a == "--no-roi" || a == "--no-roi-cylinder" || a == "--roi-off") c.roi_enable = false;
        else if (a == "--no-roi-each-iter") c.apply_roi_each_iter = false;
        else if (a == "--roi-each-iter") c.apply_roi_each_iter = true;
        else if (a == "--roi-diameter-mm" || a == "--roi-diameter") c.roi_diameter_mm = (float)atof(need(a.c_str()));
        else if (a == "--roi-height-mm" || a == "--roi-height") c.roi_height_mm = (float)atof(need(a.c_str()));
        else if (a == "--data-mode") {
            c.data_mode = need("--data-mode");
            if (c.data_mode != "measured" && c.data_mode != "simulation") { fprintf(stderr, "--data-mode must be measured or simulation\n"); exit(1); }
            if (c.data_mode == "simulation") c.detector_mask_enable = false;
        }
        else if (a == "--no-detector-mask") c.detector_mask_enable = false;
        else if (a == "--no-norm-factor") c.norm_factor_enable = false;
        else if (a == "--roi-center-mm" || a == "--roi-center") {
            const char* v = need(a.c_str());
            if (strchr(v, ',')) {
                parse_fwhm3(v, c.roi_center_mm);
            } else {
                c.roi_center_mm[0] = (float)atof(v);
                if (i + 2 >= argc) { fprintf(stderr, "%s needs X Y Z or X,Y,Z\n", a.c_str()); exit(1); }
                c.roi_center_mm[1] = (float)atof(argv[++i]);
                c.roi_center_mm[2] = (float)atof(argv[++i]);
            }
        }
        else if (a == "--check-sens-zsym" || a == "--check-sensitivity-zsym" ||
                 a == "--check-sensitivity-axial-symmetry") {
            c.check_sens_zsym = true;
            c.save_sens_edge_png = true;
        }
        else if (a == "--save-sens-edge-png" || a == "--save-sensitivity-edge-png") c.save_sens_edge_png = true;
        else if (a == "--sens-zsym-tol" || a == "--sensitivity-zsym-tol") c.sens_zsym_tol = (float)atof(need(a.c_str()));
        else if (a == "--iters") c.num_iters = atoi(need("--iters"));
        else if (a == "--gpus") c.num_gpus = atoi(need("--gpus"));
        else if (a == "--angle-batch" || a == "--compact-angle-batch") c.angle_batch = atoi(need(a.c_str()));
        else if (a == "--slab-prefetch") {
            c.slab_prefetch = true;
            c.slab_prefetch_early = false;
        }
        else if (a == "--slab-prefetch-early") {
            c.slab_prefetch = true;
            c.slab_prefetch_early = true;
        }
        else if (a == "--no-slab-prefetch") {
            c.slab_prefetch = false;
            c.slab_prefetch_early = false;
        }
        else if (a == "--gpu-slab-cache" || a == "--resident-slabs") {
            c.gpu_slab_cache = atoi(need(a.c_str()));
        }
        else if (a == "--save-every") c.save_interval = atoi(need("--save-every"));
        else if (a == "--sym") c.sym_override = need("--sym");
        else if (a == "--recon-angle-step" || a == "--angle-step") c.recon_angle_step = atoi(need(a.c_str()));
        else if (a == "--conv-mode") c.conv_mode = need("--conv-mode");
        else if (a == "--psf-fwhm") parse_fwhm3(need("--psf-fwhm"), c.psf_fwhm_mm);
        else if (a == "--psf-sigmas") c.psf_nb_sigmas = (float)atof(need("--psf-sigmas"));
        else if (a == "--post-fwhm") c.post_fwhm_mm = (float)atof(need("--post-fwhm"));
        else if (a == "--post-schedule") c.post_schedule = need("--post-schedule");
        else if (a == "--post-fwhm-start") c.post_fwhm_start_mm = (float)atof(need("--post-fwhm-start"));
        else if (a == "--post-fwhm-end") c.post_fwhm_end_mm = (float)atof(need("--post-fwhm-end"));
        else if (a == "--post-start-iter") c.post_start_iter = atoi(need("--post-start-iter"));
        else if (a == "--post-every") c.post_apply_every = atoi(need("--post-every"));
        else if (a == "--map-type") c.map_type = parse_map_type(need("--map-type"));
        else if (a == "--map-beta") c.map_beta = (float)atof(need("--map-beta"));
        else if (a == "--map-neigh") c.map_neigh26 = atoi(need("--map-neigh")) == 26;
        else if (a == "--map-delta") c.map_delta = (float)atof(need("--map-delta"));
        else if (a == "--map-huber-threshold") c.map_huber_threshold = (float)atof(need("--map-huber-threshold"));
        else if (a == "--map-relative-gamma") c.map_relative_gamma = (float)atof(need("--map-relative-gamma"));
        else if (a == "--map-relative-epsilon") c.map_relative_epsilon = (float)atof(need("--map-relative-epsilon"));
        else if (a == "--butter-cutoff-cm") {
            c.butter_cutoff_cycles_per_cm =
                (float)atof(need("--butter-cutoff-cm"));
            if (!std::isfinite(c.butter_cutoff_cycles_per_cm) ||
                c.butter_cutoff_cycles_per_cm <= 0.0f) {
                fprintf(stderr, "--butter-cutoff-cm must be finite and > 0\n");
                exit(1);
            }
            c.butter_enable = true;
        }
        else if (a == "--butter-order") c.butter_order = atoi(need("--butter-order"));
        else if (a == "--butter-mode") c.butter_mode = need("--butter-mode");
        else if (a == "--butter-start-iter") c.butter_start_iter = atoi(need("--butter-start-iter"));
        else if (a == "--butter-period") c.butter_period = atoi(need("--butter-period"));
        else if (a == "--butter-last-n") c.butter_last_n = atoi(need("--butter-last-n"));
        else if (a == "--help" || a == "-h") { print_usage(); exit(0); }
        else { fprintf(stderr, "Unknown option: %s\n", a.c_str()); print_usage(); exit(1); }
    }
    if (c.matrix_dir.empty() || c.proj_path.empty()) { print_usage(); exit(1); }
    if (c.num_gpus < 1) c.num_gpus = 1;
    if (c.angle_batch < 0) c.angle_batch = 0;
    if (c.multibed_edge_layers < 1) {
        fprintf(stderr, "--multibed-edge-layers must be a positive integer\n");
        exit(1);
    }
    if (c.gpu_slab_cache < 0) c.gpu_slab_cache = 0;
    if (c.recon_angle_step < 1) c.recon_angle_step = 1;
    if (c.butter_order < 1 || c.butter_order > 32) {
        fprintf(stderr, "--butter-order must be in [1,32]\n");
        exit(1);
    }
    if (c.butter_start_iter < 0 || c.butter_start_iter > c.num_iters ||
        c.butter_period < 1 || c.butter_last_n < 1) {
        fprintf(stderr,
                "Butterworth schedule requires start in [0,iters], period >=1, last_n >=1\n");
        exit(1);
    }
    if (c.save_interval <= 0 || c.save_interval > c.num_iters) {
        fprintf(stderr, "--save-every must be in [1, --iters]; recon_final is disabled\n");
        exit(1);
    }
    if (c.roi_diameter_mm <= 0.0f || c.roi_height_mm <= 0.0f) {
        fprintf(stderr, "Invalid cylinder ROI size: diameter=%.6g height=%.6g\n",
                c.roi_diameter_mm, c.roi_height_mm);
        exit(1);
    }
    if (c.sens_zsym_tol <= 0.0f || !std::isfinite(c.sens_zsym_tol)) c.sens_zsym_tol = 1e-4f;
    if (c.data_mode == "simulation" && c.norm_factor_enable &&
        c.norm_factor_path.empty()) {
        c.norm_factor_enable = false;
        printf("[simulation] no explicit --norm-factor: disable automatic "
               "normalisation-factor discovery (g_i=1)\n");
    }
    return c;
}

static float scheduled_post_fwhm(const ReconConfig& cfg, int iter) {
    if (cfg.post_schedule == "constant") return cfg.post_fwhm_mm;
    float t = (cfg.num_iters <= 1) ? 0.0f : (float)(iter - 1) / (float)(cfg.num_iters - 1);
    if (cfg.post_schedule == "linear") {
        return cfg.post_fwhm_start_mm + t * (cfg.post_fwhm_end_mm - cfg.post_fwhm_start_mm);
    }
    if (cfg.post_schedule == "exponential") {
        float start = fmaxf(cfg.post_fwhm_start_mm, 1e-6f);
        float end = fmaxf(cfg.post_fwhm_end_mm, 1e-6f);
        float lam = logf(start / end);
        return start * expf(-lam * t);
    }
    return cfg.post_fwhm_mm;
}

static void enable_peer_access(int num_gpus) {
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < num_gpus; j++) {
            if (i == j) continue;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, i, j);
            if (can) {
                cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                    fprintf(stderr, "Warning: peer access %d->%d failed: %s\n", i, j, cudaGetErrorString(e));
                }
                cudaGetLastError();
            }
        }
    }
}

static void broadcast_image_from_gpu0(const float* d_src0,
                                      std::vector<GPUWorker>& workers,
                                      int64_t n) {
    for (auto& w : workers) {
        if (w.dev == 0) {
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaMemcpy(w.d_img_in, d_src0, (size_t)n*sizeof(float), cudaMemcpyDeviceToDevice));
        } else {
            CUDA_CHECK(cudaMemcpyPeer(w.d_img_in, w.dev, d_src0, 0, (size_t)n*sizeof(float)));
        }
    }
}

static void reduce_parts_to_gpu0(float* d_out0, float* d_tmp0,
                                 const std::vector<GPUWorker>& workers,
                                 float* GPUWorker::*member,
                                 int64_t n)
{
    int nb = (int)((n + BLK - 1) / BLK);
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemset(d_out0, 0, (size_t)n*sizeof(float)));
    for (const auto& w : workers) {
        float* src = w.*member;
        if (w.dev == 0) {
            k_add_inplace<<<nb, BLK>>>(d_out0, src, n);
        } else {
            CUDA_CHECK(cudaMemcpyPeer(d_tmp0, 0, src, w.dev, (size_t)n*sizeof(float)));
            k_add_inplace<<<nb, BLK>>>(d_out0, d_tmp0, n);
        }
    }
}

// CUDA 重建引擎入口：解析底层重建参数并输出系统矩阵完整尺寸的 NIfTI。
// 底层入口：完成一次床位的完整系统矩阵重建并写出完整尺寸图像。
// 该历史全局联合实现仅保留给源码级诊断。公开入口固定调用后面的独立状态调度器，
// 使每床似然定义与既有基线保持一致。
[[maybe_unused]] static int bedmerge_reconstruct_main(int argc, char** argv) {
    ReconConfig cfg = parse_args(argc, argv);
    if (!cfg.output_dir_explicit) {
        cfg.output_dir = default_output_dir_from_projection(cfg.proj_path);
    }
    make_dir(cfg.output_dir);

    printf("\n==============================================================\n");
    printf("  SPECT MLEM/MAP-EM v4 Multi-Bed Joint Reconstruction CUDA (Windows)\n");
    printf("==============================================================\n");
    printf("  cuSPARSE: ALG_DEFAULT, shared workspace, no transpose preprocess\n");
    printf("  output directory: %s%s\n",
           cfg.output_dir.c_str(),
           cfg.output_dir_explicit ? " (explicit)" : " (auto)");

    MatrixMeta meta = read_metadata(cfg.matrix_dir + "/metadata.bin");
    if ((int64_t)meta.Nx * meta.Ny * meta.Nz != meta.total_voxels) {
        fprintf(stderr,
                "metadata total_voxels mismatch: Nx*Ny*Nz=%" PRId64 " total_voxels=%d\n",
                (int64_t)meta.Nx * meta.Ny * meta.Nz, meta.total_voxels);
        exit(1);
    }
    if (cfg.butter_enable) {
        const float nyquist_cycles_per_cm = 5.0f / meta.voxel_size;
        if (cfg.butter_cutoff_cycles_per_cm > nyquist_cycles_per_cm) {
            fprintf(stderr,
                    "--butter-cutoff-cm %.9g exceeds voxel Nyquist %.9g cycles/cm "
                    "(voxel=%.9g mm)\n",
                    (double)cfg.butter_cutoff_cycles_per_cm,
                    (double)nyquist_cycles_per_cm, (double)meta.voxel_size);
            exit(1);
        }
    }

    SymMode sym_mode = parse_sym_mode(cfg.sym_override, meta);
    bool use_z_branches = (mode_uses_z(sym_mode) || meta.use_zsym != 0) &&
                          (meta.compact_det_z < meta.det_nz);
    int sym_count = xy_sym_count(sym_mode);
    int z_count = use_z_branches ? 2 : 1;
    int full_rpa = meta.det_nx * meta.det_nz;
    int64_t local_N = meta.total_voxels;

    if ((sym_mode == SYM_45 || sym_mode == SYM_45Z || sym_mode == SYM_90) &&
        meta.Nx != meta.Ny) {
        fprintf(stderr, "45/90 symmetry requires Nx==Ny, got Nx=%d Ny=%d\n",
                meta.Nx, meta.Ny);
        exit(1);
    }
    if ((sym_mode == SYM_45 || sym_mode == SYM_45Z) &&
        meta.compact_num_angles != meta.full_num_angles / 8 + 1) {
        fprintf(stderr,
                "Warning: 45 symmetry expects compact_angles=full/8+1, got compact=%d full=%d\n",
                meta.compact_num_angles, meta.full_num_angles);
    }

    printf("[1] Local system-matrix geometry\n");
    printf("  local image: %d x %d x %d = %d voxels, voxel=%.4f mm\n",
           meta.Nx, meta.Ny, meta.Nz, meta.total_voxels, meta.voxel_size);
    printf("  matrix axial length: %.4f mm; valid axial FOV per bed: %.4f mm\n",
           meta.Nz * meta.voxel_size, cfg.roi_height_mm);
    printf("  detector: det_nx=%d det_nz=%d compact_det_z=%d\n",
           meta.det_nx, meta.det_nz, meta.compact_det_z);
    printf("  angles: full=%d compact=%d; compact_rpa=%d\n",
           meta.full_num_angles, meta.compact_num_angles, meta.compact_rpa);
    printf("  symmetry: %s, xy branches=%d, z branches=%d\n",
           sym_name(sym_mode), sym_count, z_count);

    std::vector<std::string> projection_paths = discover_projection_paths(cfg);
    std::vector<BedInfo> beds(projection_paths.size());
    std::vector<float> P_all;
    size_t proj_stride = 0;
    int proj_na = 0, proj_rpa = 0;

    printf("[2] Projection input: %zu bed(s)\n", projection_paths.size());
    for (size_t b = 0; b < projection_paths.size(); ++b) {
        int na = 0, rpa = 0;
        std::vector<double> p_double = read_projection_bin_double(projection_paths[b],
                                                                  &na, &rpa);
        if (na != meta.full_num_angles || rpa != full_rpa) {
            fprintf(stderr,
                    "Projection shape mismatch for %s: got=(%d,%d), expected=(%d,%d)\n",
                    projection_paths[b].c_str(), na, rpa,
                    meta.full_num_angles, full_rpa);
            exit(1);
        }
        validate_projection_counts(p_double, projection_paths[b]);
        if (b == 0) {
            proj_na = na;
            proj_rpa = rpa;
            proj_stride = p_double.size();
            P_all.reserve(proj_stride * projection_paths.size());
        } else if (p_double.size() != proj_stride) {
            fprintf(stderr,
                    "Projection length mismatch for bed %zu: got=%zu expected=%zu\n",
                    b + 1, p_double.size(), proj_stride);
            exit(1);
        }

        BadDetectorMaskStats bad_stats{};
        if (cfg.detector_mask_enable) {
            bad_stats = zero_link4_fpga2_det4(
                p_double, na, meta.det_nx, meta.det_nz);
        }

        beds[b].projection_path = projection_paths[b];
        if (projection_paths.size() == 1) {
            beds[b].name = "bed1";
        } else if (parse_flat_bed_bin_index(projection_paths[b], nullptr)) {
            beds[b].name = path_stem(projection_paths[b]);
        } else {
            beds[b].name = path_basename(path_parent(projection_paths[b]));
        }
        double sum = 0.0;
        for (double v : p_double) sum += v;
        beds[b].raw_counts = sum;

        size_t old_size = P_all.size();
        P_all.resize(old_size + proj_stride);
        for (size_t i = 0; i < proj_stride; ++i) {
            P_all[old_size + i] = (float)p_double[i];
        }
        printf("  %-12s counts=%.6e file=%s\n",
               beds[b].name.c_str(), beds[b].raw_counts,
               beds[b].projection_path.c_str());
        if (cfg.detector_mask_enable) {
            printf("               masked LINK4/FPGA2/DET4: angles=0..59 z=24..35 x=24..35, "
                   "bins=%" PRId64 ", prior_nonzero=%" PRId64 ", removed_counts=%.6e\n",
                   bad_stats.bins, bad_stats.nonzero_bins, bad_stats.removed_sum);
        } else {
            printf("               detector mask disabled; all projection bins retained\n");
        }
    }

    GlobalGeometry gg = assign_bed_geometry(beds, cfg, meta);
    int64_t global_N = gg.global_voxels;

    printf("[3] Global multi-bed geometry\n");
    printf("  global image: %d x %d x %d = %" PRId64 " voxels\n",
           gg.Nx, gg.Ny, gg.global_Nz, gg.global_voxels);
    printf("  global z centers: [%.4f, %.4f] mm; origin_z=%.4f mm\n",
           gg.origin_z_mm,
           gg.origin_z_mm + (gg.global_Nz - 1) * gg.voxel_mm,
           gg.origin_z_mm);
    for (const BedInfo& bed : beds) {
        printf("  %-12s center=%9.4f mm shift=%9.5f vox (floor=%d frac=%.5f)\n",
               bed.name.c_str(), bed.center_mm, bed.shift_vox,
               bed.shift_floor, bed.shift_frac);
    }

    std::vector<int> order((size_t)beds.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return beds[(size_t)a].center_mm < beds[(size_t)b].center_mm;
    });
    for (size_t i = 1; i < order.size(); ++i) {
        const BedInfo& a = beds[(size_t)order[i - 1]];
        const BedInfo& b = beds[(size_t)order[i]];
        float distance = b.center_mm - a.center_mm;
        float valid_overlap = cfg.roi_height_mm - distance;
        float matrix_overlap = meta.Nz * meta.voxel_size - distance;
        printf("  overlap %-12s <-> %-12s: distance=%.4f mm, "
               "valid_overlap=%.4f mm, matrix_overlap=%.4f mm\n",
               a.name.c_str(), b.name.c_str(), distance,
               valid_overlap, matrix_overlap);
        if (valid_overlap < 0.0f) {
            fprintf(stderr,
                    "Warning: %.4f mm axial gap between valid FOVs of %s and %s.\n",
                    -valid_overlap, a.name.c_str(), b.name.c_str());
        }
    }

    std::vector<float> G_full(proj_stride, 1.0f);
    std::string norm_factor_path = find_normalization_factor(cfg);
    if (!norm_factor_path.empty()) {
        std::string norm_factor_mode = resolve_normalization_factor_mode(cfg,
                                                                         norm_factor_path);
        G_full = read_system_response_factor(proj_na, proj_rpa,
                                             norm_factor_path,
                                             norm_factor_mode);
        if (G_full.size() != proj_stride) {
            fprintf(stderr,
                    "Normalization factor length mismatch: got %zu expected %zu\n",
                    G_full.size(), proj_stride);
            exit(1);
        }
        printf("[4] Shared normalization factor: %s\n", norm_factor_path.c_str());
    } else {
        printf("[4] Shared normalization factor: none found, using g_i=1\n");
    }
    if (cfg.detector_mask_enable) {
        const BadDetectorMaskStats response_mask_stats = zero_link4_fpga2_det4(
            G_full, proj_na, meta.det_nx, meta.det_nz);
        printf("[4] Disabled system response for LINK4/FPGA2/DET4: "
               "angles=0..59 z=24..35 x=24..35, bins=%" PRId64
               ", prior_nonzero=%" PRId64 ", removed_response_sum=%.6e\n",
               response_mask_stats.bins, response_mask_stats.nonzero_bins,
               response_mask_stats.removed_sum);
    } else {
        printf("[4] Detector mask disabled; ideal projection bins and system response retained\n");
    }

    save_multibed_geometry_json(cfg.output_dir + "/recon_geometry.json",
                                gg, meta, cfg, beds);

    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0) {
        fprintf(stderr, "No CUDA device found\n");
        exit(1);
    }
    cfg.num_gpus = std::min(cfg.num_gpus, dev_count);
    bool use_angle_batch = (cfg.angle_batch > 0);
    if (use_angle_batch && cfg.num_gpus != 1) {
        fprintf(stderr,
                "Warning: --angle-batch is a single-GPU fallback path; forcing --gpus 1.\n");
        cfg.num_gpus = 1;
    }
    enable_peer_access(cfg.num_gpus);

    std::vector<ChunkInfo> chunks = scan_chunks(cfg.matrix_dir);
    printf("[5] Found %zu system-matrix chunks\n", chunks.size());

    std::vector<GPUWorker> transient_workers;
    std::vector<GPUWorker>* worker_storage = &transient_workers;
    if (!use_angle_batch) {
        if (cfg.gpu_slab_cache > 0) {
            fprintf(stderr,
                    "Warning: --gpu-slab-cache only applies with --angle-batch; ignoring it.\n");
        }
        const std::string matrix_key = fs::absolute(cfg.matrix_dir).string();
        const bool can_reuse = g_bedmerge_matrix_cache.valid &&
                               g_bedmerge_matrix_cache.matrix_dir == matrix_key &&
                               g_bedmerge_matrix_cache.gpus == cfg.num_gpus &&
                               g_bedmerge_matrix_cache.nx == meta.Nx &&
                               g_bedmerge_matrix_cache.ny == meta.Ny &&
                               g_bedmerge_matrix_cache.nz == meta.Nz &&
                               g_bedmerge_matrix_cache.compact_angles == meta.compact_num_angles;
        if (can_reuse) {
            printf("[5b] Reusing resident GPU CSR system matrix; updating Bed projection only\n");
            for (auto& worker : g_bedmerge_matrix_cache.workers) {
                worker.update_projection(P_all, proj_stride, G_full);
            }
        } else {
            g_bedmerge_matrix_cache.clear();
            std::vector<HostCSR64> hcsrs((size_t)cfg.num_gpus);
            for (int g = 0; g < cfg.num_gpus; ++g) {
                int start_ck = g * meta.compact_num_angles / cfg.num_gpus;
                int end_ck = (g + 1) * meta.compact_num_angles / cfg.num_gpus;
                auto t0 = std::chrono::high_resolution_clock::now();
                hcsrs[(size_t)g] = build_worker_csr(chunks, start_ck, end_ck,
                                                    meta.compact_rpa,
                                                    meta.total_voxels);
                double sec = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t0).count();
                printf("  host CSR GPU%d angles [%d,%d): rows=%" PRId64
                       " nnz=%.2fM, %.2fs\n",
                       g, start_ck, end_ck, hcsrs[(size_t)g].num_rows,
                       (double)hcsrs[(size_t)g].nnz / 1e6, sec);
            }
            g_bedmerge_matrix_cache.workers.resize((size_t)cfg.num_gpus);
            for (int g = 0; g < cfg.num_gpus; ++g) {
                g_bedmerge_matrix_cache.workers[(size_t)g].init(
                    g, hcsrs[(size_t)g], meta, P_all, proj_stride,
                    1, G_full);
            }
            g_bedmerge_matrix_cache.valid = true;
            g_bedmerge_matrix_cache.matrix_dir = matrix_key;
            g_bedmerge_matrix_cache.gpus = cfg.num_gpus;
            g_bedmerge_matrix_cache.nx = meta.Nx;
            g_bedmerge_matrix_cache.ny = meta.Ny;
            g_bedmerge_matrix_cache.nz = meta.Nz;
            g_bedmerge_matrix_cache.compact_angles = meta.compact_num_angles;
            hcsrs.clear();
            hcsrs.shrink_to_fit();
        }
        worker_storage = &g_bedmerge_matrix_cache.workers;
    } else {
        const char* prefetch_mode = !cfg.slab_prefetch ? "off" :
                                    (cfg.slab_prefetch_early ? "early" : "safe");
        printf("  single-GPU compact-angle batching: batch=%d, slab_prefetch=%s, "
               "gpu_slab_cache=%d\n",
               cfg.angle_batch, prefetch_mode, cfg.gpu_slab_cache);
    }
    std::vector<GPUWorker>& workers = *worker_storage;

    CUDA_CHECK(cudaSetDevice(0));
    int nb_local = (int)((local_N + BLK - 1) / BLK);
    int nb_global = (int)((global_N + BLK - 1) / BLK);

    float *d_f = nullptr, *d_f_eff = nullptr;
    float *d_global_bp = nullptr, *d_global_sens = nullptr;
    float *d_local_current = nullptr, *d_local_bp = nullptr;
    float *d_local_sens = nullptr, *d_tmp_local = nullptr;
    float *d_conv_tmp1 = nullptr, *d_conv_tmp2 = nullptr;
    float *d_psf_norm = nullptr, *d_post_norm = nullptr, *d_map_grad = nullptr;
    float *d_bed_centers = nullptr;

    CUDA_CHECK(cudaMalloc(&d_f,             (size_t)global_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_f_eff,         (size_t)global_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_global_bp,     (size_t)global_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_global_sens,   (size_t)global_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_local_current, (size_t)local_N  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_local_bp,      (size_t)local_N  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_local_sens,    (size_t)local_N  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp_local,     (size_t)local_N  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_tmp1,     (size_t)global_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_tmp2,     (size_t)global_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_psf_norm,      (size_t)global_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_post_norm,     (size_t)global_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_map_grad,      (size_t)global_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bed_centers,   beds.size() * sizeof(float)));

    std::vector<BedInfo> roi_beds = beds;
    std::vector<float> h_roi_centers(beds.size());
    for (size_t b = 0; b < beds.size(); ++b) {
        roi_beds[b].center_mm += cfg.roi_center_mm[2];
        h_roi_centers[b] = roi_beds[b].center_mm;
    }
    CUDA_CHECK(cudaMemcpy(d_bed_centers, h_roi_centers.data(),
                          h_roi_centers.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    k_fill<<<nb_global, BLK>>>(d_f, 1.0f, global_N);
    if (cfg.roi_enable) {
        int64_t roi_voxels = count_multibed_roi_voxels(
            gg.Nx, gg.Ny, gg.global_Nz, gg.voxel_mm, gg.origin_z_mm,
            cfg.roi_diameter_mm, cfg.roi_height_mm,
            cfg.roi_center_mm[0], cfg.roi_center_mm[1], roi_beds);
        printf("[6] Multi-bed ROI: diameter=%.4f mm, per-bed valid height=%.4f mm, "
               "voxels=%" PRId64 " / %" PRId64 " (%.2f%%)\n",
               cfg.roi_diameter_mm, cfg.roi_height_mm,
               roi_voxels, global_N,
               100.0 * (double)roi_voxels / (double)global_N);
        if (roi_voxels <= 0) {
            fprintf(stderr, "Multi-bed ROI contains no voxels.\n");
            exit(1);
        }
        k_apply_multibed_roi<<<nb_global, BLK>>>(
            d_f, gg.Nx, gg.Ny, gg.global_Nz,
            gg.voxel_mm, gg.origin_z_mm,
            cfg.roi_diameter_mm, cfg.roi_height_mm,
            cfg.roi_center_mm[0], cfg.roi_center_mm[1],
            d_bed_centers, (int)beds.size(), global_N);
    } else {
        printf("[6] Multi-bed ROI disabled\n");
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    bool use_psf = (cfg.conv_mode == "psf" || cfg.conv_mode == "sieve");
    bool use_post = (cfg.conv_mode == "post" || cfg.conv_mode == "sieve");
    if (cfg.conv_mode == "none") {
        use_psf = false;
        use_post = false;
    }

    std::vector<float> psf_kx, psf_ky, psf_kz;
    if (use_psf) {
        psf_kx = build_castor_erf_1d(cfg.psf_fwhm_mm[0], gg.voxel_mm,
                                     cfg.psf_nb_sigmas);
        psf_ky = build_castor_erf_1d(cfg.psf_fwhm_mm[1], gg.voxel_mm,
                                     cfg.psf_nb_sigmas);
        psf_kz = build_castor_erf_1d(cfg.psf_fwhm_mm[2], gg.voxel_mm,
                                     cfg.psf_nb_sigmas);
        build_conv_norm(d_psf_norm, d_conv_tmp1, d_conv_tmp2,
                        gg.Nx, gg.Ny, gg.global_Nz,
                        psf_kx, psf_ky, psf_kz);
        printf("[7] Global PSF on: FWHM=[%.3f %.3f %.3f] mm\n",
               cfg.psf_fwhm_mm[0], cfg.psf_fwhm_mm[1],
               cfg.psf_fwhm_mm[2]);
    } else {
        k_fill<<<nb_global, BLK>>>(d_psf_norm, 1.0f, global_N);
        printf("[7] Global PSF off\n");
    }

    ButterworthWorkspace butter_workspace;
    if (cfg.butter_enable) {
        initialize_butterworth_workspace(
            butter_workspace, gg.Nx, gg.Ny, gg.global_Nz, gg.voxel_mm,
            cfg.butter_cutoff_cycles_per_cm, cfg.butter_order);
        const double workspace_mib =
            ((double)butter_workspace.padded_real_count * sizeof(float) +
             (double)butter_workspace.spectrum_count *
                 (sizeof(cufftComplex) + sizeof(float))) /
            (1024.0 * 1024.0);
        printf("[8] Global Butterworth on: cutoff=%.6g cycles/cm order=%d "
               "mode=%s padding=%dx%dx%d workspace=%.1f MiB\n",
               cfg.butter_cutoff_cycles_per_cm, cfg.butter_order,
               cfg.butter_mode.c_str(), butter_workspace.padded_nx,
               butter_workspace.padded_ny, butter_workspace.padded_nz,
               workspace_mib);
    } else {
        printf("[8] Global Butterworth off\n");
    }

    std::vector<GPUWorker> cached_slab_workers;
    int cached_slab_count = 0;
    int cached_end_ck = 0;
    if (use_angle_batch && cfg.gpu_slab_cache > 0) {
        const int batch = std::max(1, cfg.angle_batch);
        const int total_slabs =
            (meta.compact_num_angles + batch - 1) / batch;
        cached_slab_count = std::min(cfg.gpu_slab_cache, total_slabs);
        cached_slab_workers.resize((size_t)cached_slab_count);

        printf("[8b] Build persistent GPU slab cache: requested=%d actual=%d\n",
               cfg.gpu_slab_cache, cached_slab_count);
        for (int i = 0; i < cached_slab_count; ++i) {
            const int start_ck = i * batch;
            const int end_ck = std::min(meta.compact_num_angles,
                                        start_ck + batch);
            TimedHostCSR built = build_worker_csr_timed(
                chunks, start_ck, end_ck,
                meta.compact_rpa, meta.total_voxels);
            auto setup_t0 = std::chrono::high_resolution_clock::now();
            cached_slab_workers[(size_t)i].init(
                0, built.csr, meta, P_all, proj_stride,
                (int)beds.size(), G_full);
            double setup_sec = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - setup_t0).count();
            std::vector<int64_t>().swap(built.csr.row_ptr);
            std::vector<int32_t>().swap(built.csr.col_idx);
            std::vector<float>().swap(built.csr.values);

            size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            printf("  cached slab %d/%d angles [%d,%d): build=%.3fs "
                   "setup=%.3fs GPU memory=%.1f/%.1f MiB used\n",
                   i + 1, cached_slab_count, start_ck, end_ck,
                   built.build_sec, setup_sec,
                   (double)(total_bytes - free_bytes) / (1024.0 * 1024.0),
                   (double)total_bytes / (1024.0 * 1024.0));
        }
        cached_end_ck = std::min(meta.compact_num_angles,
                                 cached_slab_count * batch);
    }

    printf("[9] Compute joint multi-bed sensitivity\n");
    CUDA_CHECK(cudaMemset(d_global_sens, 0,
                          (size_t)global_N * sizeof(float)));

    if (!use_angle_batch) {
        std::vector<std::thread> threads;
        for (int g = 0; g < cfg.num_gpus; ++g) {
            threads.emplace_back([&, g]() {
                GPUWorker& w = workers[(size_t)g];
                CUDA_CHECK(cudaSetDevice(w.dev));
                int nb_img = (int)(((int64_t)w.N2 + BLK - 1) / BLK);
                int nb_rows = (w.local_rows + BLK - 1) / BLK;
                CUDA_CHECK(cudaMemset(w.d_sens_part, 0,
                                      (size_t)w.N2 * sizeof(float)));
                for (int s = 0; s < sym_count; ++s) {
                    for (int zm = 0; zm < z_count; ++zm) {
                        k_make_unit_ratio_batched<<<nb_rows, BLK>>>(
                            w.d_ratio, w.d_G_full, w.local_rows,
                            w.start_ck, w.compact_rpa,
                            w.det_nx, w.compact_det_z, w.det_nz,
                            full_rpa, (int)sym_mode, s, zm,
                            meta.compact_num_angles,
                            meta.full_num_angles,
                            cfg.recon_angle_step);
                        w.spmv_backward();
                        k_transform_bwd_add<<<nb_img, BLK>>>(
                            w.d_bp_rot, w.d_sens_part,
                            w.Nx, w.Ny, w.Nz,
                            (int)sym_mode, s, zm);
                    }
                }
                CUDA_CHECK(cudaDeviceSynchronize());
            });
        }
        for (auto& t : threads) t.join();

        reduce_parts_to_gpu0(d_local_sens, d_tmp_local, workers,
                             &GPUWorker::d_sens_part, local_N);
        for (const BedInfo& bed : beds) {
            k_scatter_bed_linear_add<<<nb_local, BLK>>>(
                d_local_sens, d_global_sens,
                gg.Nx, gg.Ny, gg.local_Nz, gg.global_Nz,
                bed.shift_floor, bed.shift_frac, local_N);
        }
    } else {
        int batch = std::max(1, cfg.angle_batch);
        double sens_build_work_sec = 0.0;
        double sens_build_wait_sec = 0.0;
        double sens_setup_sec = 0.0;
        double sens_compute_sec = 0.0;
        double sens_teardown_sec = 0.0;

        auto accumulate_sensitivity_slab = [&](GPUWorker& w) {
            int nb_img = (int)(((int64_t)w.N2 + BLK - 1) / BLK);
            int nb_rows = (w.local_rows + BLK - 1) / BLK;
            CUDA_CHECK(cudaMemset(w.d_sens_part, 0,
                                  (size_t)w.N2 * sizeof(float)));
            for (int s = 0; s < sym_count; ++s) {
                for (int zm = 0; zm < z_count; ++zm) {
                    k_make_unit_ratio_batched<<<nb_rows, BLK>>>(
                        w.d_ratio, w.d_G_full, w.local_rows,
                        w.start_ck, w.compact_rpa,
                        w.det_nx, w.compact_det_z, w.det_nz,
                        full_rpa, (int)sym_mode, s, zm,
                        meta.compact_num_angles,
                        meta.full_num_angles,
                        cfg.recon_angle_step);
                    w.spmv_backward();
                    k_transform_bwd_add<<<nb_img, BLK>>>(
                        w.d_bp_rot, w.d_sens_part,
                        w.Nx, w.Ny, w.Nz,
                        (int)sym_mode, s, zm);
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            for (const BedInfo& bed : beds) {
                k_scatter_bed_linear_add<<<nb_local, BLK>>>(
                    w.d_sens_part, d_global_sens,
                    gg.Nx, gg.Ny, gg.local_Nz, gg.global_Nz,
                    bed.shift_floor, bed.shift_frac, local_N);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        };

        auto launch_slab_build = [&](int start_ck) {
            int end_ck = std::min(meta.compact_num_angles,
                                  start_ck + batch);
            return std::async(std::launch::async,
                              [&, start_ck, end_ck]() {
                return build_worker_csr_timed(chunks, start_ck, end_ck,
                                              meta.compact_rpa,
                                              meta.total_voxels);
            });
        };
        std::future<TimedHostCSR> pending_build;
        if (cfg.slab_prefetch && cached_end_ck < meta.compact_num_angles) {
            pending_build = launch_slab_build(cached_end_ck);
        }

        auto cached_compute_t0 = std::chrono::high_resolution_clock::now();
        for (GPUWorker& w : cached_slab_workers) {
            accumulate_sensitivity_slab(w);
        }
        sens_compute_sec += std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - cached_compute_t0).count();

        for (int start_ck = cached_end_ck;
             start_ck < meta.compact_num_angles;
             start_ck += batch) {
            int end_ck = std::min(meta.compact_num_angles,
                                  start_ck + batch);
            auto wait_t0 = std::chrono::high_resolution_clock::now();
            TimedHostCSR built = cfg.slab_prefetch
                ? pending_build.get()
                : build_worker_csr_timed(chunks, start_ck, end_ck,
                                         meta.compact_rpa,
                                         meta.total_voxels);
            double wait_sec = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - wait_t0).count();
            sens_build_work_sec += built.build_sec;
            sens_build_wait_sec += wait_sec;
            HostCSR64 hcsr = std::move(built.csr);

            if (cfg.slab_prefetch_early &&
                end_ck < meta.compact_num_angles) {
                pending_build = launch_slab_build(end_ck);
            }

            auto setup_t0 = std::chrono::high_resolution_clock::now();
            GPUWorker w;
            w.init(0, hcsr, meta, P_all, proj_stride,
                   (int)beds.size(), G_full);
            std::vector<int64_t>().swap(hcsr.row_ptr);
            std::vector<int32_t>().swap(hcsr.col_idx);
            std::vector<float>().swap(hcsr.values);
            sens_setup_sec += std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - setup_t0).count();

            if (cfg.slab_prefetch && !cfg.slab_prefetch_early &&
                end_ck < meta.compact_num_angles) {
                pending_build = launch_slab_build(end_ck);
            }

            auto compute_t0 = std::chrono::high_resolution_clock::now();
            accumulate_sensitivity_slab(w);
            sens_compute_sec += std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - compute_t0).count();

            auto teardown_t0 = std::chrono::high_resolution_clock::now();
            w.destroy();
            sens_teardown_sec += std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - teardown_t0).count();
        }
        printf("  sensitivity slab timing: cached=%d build_work=%.3fs wait=%.3fs "
               "setup=%.3fs compute=%.3fs teardown=%.3fs overlap_saved=%.3fs\n",
               cached_slab_count,
               sens_build_work_sec, sens_build_wait_sec,
               sens_setup_sec, sens_compute_sec, sens_teardown_sec,
               std::max(0.0, sens_build_work_sec - sens_build_wait_sec));
    }

    if (use_psf) {
        gpu_conv3d_transpose_renorm(d_global_sens, d_f_eff, d_conv_tmp1,
                                    d_psf_norm,
                                    gg.Nx, gg.Ny, gg.global_Nz,
                                    psf_kx, psf_ky, psf_kz);
        CUDA_CHECK(cudaMemcpy(d_global_sens, d_f_eff,
                              (size_t)global_N * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }
    k_clamp_min<<<nb_global, BLK>>>(d_global_sens, 1e-12f, global_N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_sens((size_t)global_N);
    CUDA_CHECK(cudaMemcpy(h_sens.data(), d_global_sens,
                          (size_t)global_N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    if (cfg.check_sens_zsym) {
        check_sensitivity_zsym_host(h_sens.data(), gg.Nx, gg.Ny,
                                    gg.global_Nz,
                                    cfg.sens_zsym_tol, "global-unmasked");
    }
    if (cfg.save_sens_edge_png) {
        save_sensitivity_edge_pngs(h_sens.data(), gg.Nx, gg.Ny,
                                   gg.global_Nz,
                                   cfg.output_dir,
                                   "sensitivity_global_unmasked");
    }
    if (cfg.roi_enable) {
        apply_multibed_roi_host(h_sens.data(),
                                gg.Nx, gg.Ny, gg.global_Nz,
                                gg.voxel_mm, gg.origin_z_mm,
                                cfg.roi_diameter_mm, cfg.roi_height_mm,
                                cfg.roi_center_mm[0], cfg.roi_center_mm[1],
                                roi_beds);
    }
    save_volume_outputs(cfg.output_dir + "/sensitivity",
                        h_sens.data(), gg.Nx, gg.Ny, gg.global_Nz,
                        gg.voxel_mm, gg.origin_z_mm, cfg.save_raw1);

    printf("[10] Iterate: beds=%zu iters=%d conv=%s MAP=%s beta=%.3e\n",
           beds.size(), cfg.num_iters, cfg.conv_mode.c_str(),
           cfg.map_type == MAP_NONE ? "none" : "on",
           cfg.map_beta);

    std::vector<float> h_out((size_t)global_N);
    auto total_t0 = std::chrono::high_resolution_clock::now();

    for (int iter = 1; iter <= cfg.num_iters; ++iter) {
        auto iter_t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaSetDevice(0));

        float* d_current = d_f;
        if (use_psf) {
            gpu_conv3d_forward_renorm(d_f, d_f_eff, d_conv_tmp1,
                                      d_psf_norm,
                                      gg.Nx, gg.Ny, gg.global_Nz,
                                      psf_kx, psf_ky, psf_kz);
            d_current = d_f_eff;
        }

        CUDA_CHECK(cudaMemset(d_global_bp, 0,
                              (size_t)global_N * sizeof(float)));

        if (!use_angle_batch) {

            for (size_t b = 0; b < beds.size(); ++b) {
                const BedInfo& bed = beds[b];

                k_gather_bed_linear<<<nb_local, BLK>>>(
                    d_current, d_local_current,
                    gg.Nx, gg.Ny, gg.local_Nz, gg.global_Nz,
                    bed.shift_floor, bed.shift_frac, local_N);
                CUDA_CHECK(cudaDeviceSynchronize());
                broadcast_image_from_gpu0(d_local_current, workers, local_N);

                std::vector<std::thread> threads;
                for (int g = 0; g < cfg.num_gpus; ++g) {
                    threads.emplace_back([&, g, b]() {
                        GPUWorker& w = workers[(size_t)g];
                        CUDA_CHECK(cudaSetDevice(w.dev));
                        int nb_img = (int)(((int64_t)w.N2 + BLK - 1) / BLK);
                        int nb_rows = (w.local_rows + BLK - 1) / BLK;
                        CUDA_CHECK(cudaMemset(w.d_bp_part, 0,
                                              (size_t)w.N2 * sizeof(float)));

                        const float* d_P_bed =
                            w.d_P_all + b * w.proj_stride;
                        for (int s = 0; s < sym_count; ++s) {
                            for (int zm = 0; zm < z_count; ++zm) {
                                k_transform_fwd<<<nb_img, BLK>>>(
                                    w.d_img_in, w.d_img_rot,
                                    w.Nx, w.Ny, w.Nz,
                                    (int)sym_mode, s, zm);
                                w.spmv_forward();
                                k_calc_ratio_batched<<<nb_rows, BLK>>>(
                                    w.d_y, w.d_ratio,
                                    d_P_bed, w.d_G_full,
                                    w.local_rows, w.start_ck,
                                    w.compact_rpa,
                                    w.det_nx, w.compact_det_z,
                                    w.det_nz, full_rpa,
                                    (int)sym_mode, s, zm,
                                    meta.compact_num_angles,
                                    meta.full_num_angles,
                                    cfg.recon_angle_step,
                                    IMPLEMENTATION_CASTOR,
                                    1e-5f, 1e-10f, nullptr);
                                w.spmv_backward();
                                k_transform_bwd_add<<<nb_img, BLK>>>(
                                    w.d_bp_rot, w.d_bp_part,
                                    w.Nx, w.Ny, w.Nz,
                                    (int)sym_mode, s, zm);
                            }
                        }
                        CUDA_CHECK(cudaDeviceSynchronize());
                    });
                }
                for (auto& t : threads) t.join();

                reduce_parts_to_gpu0(d_local_bp, d_tmp_local, workers,
                                     &GPUWorker::d_bp_part, local_N);
                k_scatter_bed_linear_add<<<nb_local, BLK>>>(
                    d_local_bp, d_global_bp,
                    gg.Nx, gg.Ny, gg.local_Nz, gg.global_Nz,
                    bed.shift_floor, bed.shift_frac, local_N);
            }
        } else {

            int batch = std::max(1, cfg.angle_batch);
            double iter_build_work_sec = 0.0;
            double iter_build_wait_sec = 0.0;
            double iter_setup_sec = 0.0;
            double iter_compute_sec = 0.0;
            double iter_teardown_sec = 0.0;

            auto accumulate_backprojection_slab = [&](GPUWorker& w) {
                int nb_img = (int)(((int64_t)w.N2 + BLK - 1) / BLK);
                int nb_rows = (w.local_rows + BLK - 1) / BLK;

                for (size_t b = 0; b < beds.size(); ++b) {
                    const BedInfo& bed = beds[b];
                    k_gather_bed_linear<<<nb_local, BLK>>>(
                        d_current, d_local_current,
                        gg.Nx, gg.Ny, gg.local_Nz, gg.global_Nz,
                        bed.shift_floor, bed.shift_frac, local_N);
                    CUDA_CHECK(cudaMemcpy(w.d_img_in, d_local_current,
                                          (size_t)local_N * sizeof(float),
                                          cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemset(w.d_bp_part, 0,
                                          (size_t)w.N2 * sizeof(float)));

                    const float* d_P_bed =
                        w.d_P_all + b * w.proj_stride;
                    for (int s = 0; s < sym_count; ++s) {
                        for (int zm = 0; zm < z_count; ++zm) {
                            k_transform_fwd<<<nb_img, BLK>>>(
                                w.d_img_in, w.d_img_rot,
                                w.Nx, w.Ny, w.Nz,
                                (int)sym_mode, s, zm);
                            w.spmv_forward();
                            k_calc_ratio_batched<<<nb_rows, BLK>>>(
                                w.d_y, w.d_ratio,
                                d_P_bed, w.d_G_full,
                                w.local_rows, w.start_ck,
                                w.compact_rpa,
                                w.det_nx, w.compact_det_z,
                                w.det_nz, full_rpa,
                                (int)sym_mode, s, zm,
                                meta.compact_num_angles,
                                meta.full_num_angles,
                                cfg.recon_angle_step,
                                IMPLEMENTATION_CASTOR,
                                1e-5f, 1e-10f, nullptr);
                            w.spmv_backward();
                            k_transform_bwd_add<<<nb_img, BLK>>>(
                                w.d_bp_rot, w.d_bp_part,
                                w.Nx, w.Ny, w.Nz,
                                (int)sym_mode, s, zm);
                        }
                    }
                    k_scatter_bed_linear_add<<<nb_local, BLK>>>(
                        w.d_bp_part, d_global_bp,
                        gg.Nx, gg.Ny, gg.local_Nz, gg.global_Nz,
                        bed.shift_floor, bed.shift_frac, local_N);
                }
                CUDA_CHECK(cudaDeviceSynchronize());
            };

            auto launch_slab_build = [&](int start_ck) {
                int end_ck = std::min(meta.compact_num_angles,
                                      start_ck + batch);
                return std::async(std::launch::async,
                                  [&, start_ck, end_ck]() {
                    return build_worker_csr_timed(chunks, start_ck, end_ck,
                                                  meta.compact_rpa,
                                                  meta.total_voxels);
                });
            };
            std::future<TimedHostCSR> pending_build;
            if (cfg.slab_prefetch && cached_end_ck < meta.compact_num_angles) {
                pending_build = launch_slab_build(cached_end_ck);
            }

            auto cached_compute_t0 = std::chrono::high_resolution_clock::now();
            for (GPUWorker& w : cached_slab_workers) {
                accumulate_backprojection_slab(w);
            }
            iter_compute_sec += std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - cached_compute_t0).count();

            for (int start_ck = cached_end_ck;
                 start_ck < meta.compact_num_angles;
                 start_ck += batch) {
                int end_ck = std::min(meta.compact_num_angles,
                                      start_ck + batch);
                auto wait_t0 = std::chrono::high_resolution_clock::now();
                TimedHostCSR built = cfg.slab_prefetch
                    ? pending_build.get()
                    : build_worker_csr_timed(chunks, start_ck, end_ck,
                                             meta.compact_rpa,
                                             meta.total_voxels);
                double wait_sec = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - wait_t0).count();
                iter_build_work_sec += built.build_sec;
                iter_build_wait_sec += wait_sec;
                HostCSR64 hcsr = std::move(built.csr);
                printf("  iter %d slab [%d,%d): rows=%" PRId64
                       " nnz=%.2fM build=%.2fs wait=%.2fs\n",
                       iter, start_ck, end_ck, hcsr.num_rows,
                       (double)hcsr.nnz / 1e6,
                       built.build_sec, wait_sec);

                if (cfg.slab_prefetch_early &&
                    end_ck < meta.compact_num_angles) {
                    pending_build = launch_slab_build(end_ck);
                }

                auto setup_t0 = std::chrono::high_resolution_clock::now();
                GPUWorker w;
                w.init(0, hcsr, meta, P_all, proj_stride,
                       (int)beds.size(), G_full);
                std::vector<int64_t>().swap(hcsr.row_ptr);
                std::vector<int32_t>().swap(hcsr.col_idx);
                std::vector<float>().swap(hcsr.values);
                iter_setup_sec += std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - setup_t0).count();

                if (cfg.slab_prefetch && !cfg.slab_prefetch_early &&
                    end_ck < meta.compact_num_angles) {
                    pending_build = launch_slab_build(end_ck);
                }

                auto compute_t0 = std::chrono::high_resolution_clock::now();
                accumulate_backprojection_slab(w);
                iter_compute_sec += std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - compute_t0).count();

                auto teardown_t0 = std::chrono::high_resolution_clock::now();
                w.destroy();
                iter_teardown_sec += std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - teardown_t0).count();
            }
            printf("  iter %d slab timing: cached=%d build_work=%.3fs wait=%.3fs "
                   "setup=%.3fs compute=%.3fs teardown=%.3fs overlap_saved=%.3fs\n",
                   iter, cached_slab_count,
                   iter_build_work_sec, iter_build_wait_sec,
                   iter_setup_sec, iter_compute_sec, iter_teardown_sec,
                   std::max(0.0, iter_build_work_sec - iter_build_wait_sec));
        }

        if (use_psf) {
            gpu_conv3d_transpose_renorm(d_global_bp, d_f_eff, d_conv_tmp1,
                                        d_psf_norm,
                                        gg.Nx, gg.Ny, gg.global_Nz,
                                        psf_kx, psf_ky, psf_kz);
            CUDA_CHECK(cudaMemcpy(d_global_bp, d_f_eff,
                                  (size_t)global_N * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }

        float* map_ptr = nullptr;
        if (cfg.map_type != MAP_NONE && cfg.map_beta > 0.0f) {
            if (cfg.map_type == MAP_TV) {
                k_tv_gradient_omega<<<nb_global, BLK>>>(
                    d_f, d_map_grad, gg.Nx, gg.Ny, gg.global_Nz,
                    cfg.tv_smoothing);
            } else if (cfg.map_type == MAP_MRP) {
                k_mrp_gradient_omega<<<nb_global, BLK>>>(
                    d_f, d_map_grad, gg.Nx, gg.Ny, gg.global_Nz,
                    cfg.mrp_radius[0], cfg.mrp_radius[1], cfg.mrp_radius[2],
                    cfg.mrp_epsilon, cfg.mrp_no_norm);
            } else {
                k_map_gradient<<<nb_global, BLK>>>(
                    d_f, d_map_grad,
                    gg.Nx, gg.Ny, gg.global_Nz,
                    (int)cfg.map_type, cfg.map_neigh26,
                    cfg.map_delta, cfg.map_huber_threshold,
                    cfg.map_relative_gamma,
                    cfg.map_relative_epsilon);
            }
            map_ptr = d_map_grad;
        }

        k_mlem_update<<<nb_global, BLK>>>(
            d_f, d_global_bp, d_global_sens,
            map_ptr, cfg.map_beta, global_N);
        if (cfg.positivity) {
            k_clamp_pos<<<nb_global, BLK>>>(d_f, global_N);
        }

        if (cfg.roi_enable && cfg.apply_roi_each_iter) {
            k_apply_multibed_roi<<<nb_global, BLK>>>(
                d_f, gg.Nx, gg.Ny, gg.global_Nz,
                gg.voxel_mm, gg.origin_z_mm,
                cfg.roi_diameter_mm, cfg.roi_height_mm,
                cfg.roi_center_mm[0], cfg.roi_center_mm[1],
                d_bed_centers, (int)beds.size(), global_N);
        }

        if (use_post && iter >= cfg.post_start_iter &&
            cfg.post_apply_every > 0 &&
            ((iter - cfg.post_start_iter) % cfg.post_apply_every == 0)) {
            float fw = scheduled_post_fwhm(cfg, iter);
            if (fw > 0.01f) {
                std::vector<float> pkx = build_castor_erf_1d(
                    fw, gg.voxel_mm, cfg.psf_nb_sigmas);
                std::vector<float> pky = build_castor_erf_1d(
                    fw, gg.voxel_mm, cfg.psf_nb_sigmas);
                std::vector<float> pkz = build_castor_erf_1d(
                    fw, gg.voxel_mm, cfg.psf_nb_sigmas);
                build_conv_norm(d_post_norm, d_conv_tmp1, d_conv_tmp2,
                                gg.Nx, gg.Ny, gg.global_Nz,
                                pkx, pky, pkz);
                gpu_conv3d_forward_renorm(d_f, d_f_eff, d_conv_tmp1,
                                          d_post_norm,
                                          gg.Nx, gg.Ny, gg.global_Nz,
                                          pkx, pky, pkz);
                CUDA_CHECK(cudaMemcpy(d_f, d_f_eff,
                                      (size_t)global_N * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
                if (cfg.positivity) {
                    k_clamp_pos<<<nb_global, BLK>>>(d_f, global_N);
                }
            }
        }

        if (cfg.butter_enable &&
            should_apply_filter(iter, cfg.num_iters,
                                cfg.butter_mode,
                                cfg.butter_start_iter,
                                cfg.butter_period,
                                cfg.butter_last_n)) {
            apply_butterworth_filter(
                butter_workspace, d_f, d_f,
                iter < cfg.num_iters || cfg.positivity);
        }

        if (cfg.roi_enable && cfg.apply_roi_each_iter) {
            k_apply_multibed_roi<<<nb_global, BLK>>>(
                d_f, gg.Nx, gg.Ny, gg.global_Nz,
                gg.voxel_mm, gg.origin_z_mm,
                cfg.roi_diameter_mm, cfg.roi_height_mm,
                cfg.roi_center_mm[0], cfg.roi_center_mm[1],
                d_bed_centers, (int)beds.size(), global_N);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        double iter_sec = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - iter_t0).count();

        bool do_save = cfg.save_interval > 0 &&
                       iter % cfg.save_interval == 0;
        if (do_save) {
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_f,
                                  (size_t)global_N * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            if (cfg.roi_enable) {
                apply_multibed_roi_host(
                    h_out.data(), gg.Nx, gg.Ny, gg.global_Nz,
                    gg.voxel_mm, gg.origin_z_mm,
                    cfg.roi_diameter_mm, cfg.roi_height_mm,
                    cfg.roi_center_mm[0], cfg.roi_center_mm[1],
                    roi_beds);
            }
            char fn[1024];
            snprintf(fn, sizeof(fn), "%s/recon%d",
                     cfg.output_dir.c_str(), iter);
            save_volume_outputs(fn, h_out.data(),
                                gg.Nx, gg.Ny, gg.global_Nz,
                                gg.voxel_mm, gg.origin_z_mm,
                                cfg.save_raw1);
            printf("  Iter %3d/%d | saved | %.3fs\n",
                   iter, cfg.num_iters, iter_sec);
        } else {
            printf("  Iter %3d/%d | %.3fs\n",
                   iter, cfg.num_iters, iter_sec);
        }
    }

    double total_sec = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - total_t0).count();
    printf("\nDone. Total %.2f min, avg %.3f s/iter\n",
           total_sec / 60.0,
           total_sec / std::max(1, cfg.num_iters));

    if (cfg.butter_enable) destroy_butterworth_workspace(butter_workspace);

    cudaFree(d_bed_centers);
    cudaFree(d_map_grad);
    cudaFree(d_post_norm);
    cudaFree(d_psf_norm);
    cudaFree(d_conv_tmp2);
    cudaFree(d_conv_tmp1);
    cudaFree(d_tmp_local);
    cudaFree(d_local_sens);
    cudaFree(d_local_bp);
    cudaFree(d_local_current);
    cudaFree(d_global_sens);
    cudaFree(d_global_bp);
    cudaFree(d_f_eff);
    cudaFree(d_f);

    for (auto& w : cached_slab_workers) w.destroy();
    if (worker_storage != &g_bedmerge_matrix_cache.workers) {
        for (auto& w : transient_workers) w.destroy();
    }
    return 0;
}

// -------------------- mask 裁剪与多床拼接入口 --------------------
// 外层参数：负责发现多个 BedN.bin、裁剪单床图像并拼接。
struct BedMergeOptions {
    // 所有路径由命令行提供；默认矩阵目录相对于当前工作目录。
    fs::path proj_dir;
    fs::path matrix_dir = fs::path("MCmatrix_PSC");
    fs::path out;
    double bed_step_mm = 16.8;
    // 中心 mask 的体素尺寸，顺序为 Nx、Ny、Nz。
    int mask_nx = 100; // 输出 mask 的 X 尺寸（体素）。
    int mask_ny = 100; // 输出 mask 的 Y 尺寸（体素）。
    int mask_nz = 72;  // 输出 mask 的 Z 尺寸（体素）。
    double gain_roi_diameter_mm = 30.0;
    double roi_diameter_mm = 30.0;
    // ROI controls which full system-matrix voxels participate in reconstruction.
    // It is independent of --mask-size, which only center-crops saved images.
    double roi_height_mm = 66.0;
    // 全局多床视野在两端床中心外各保留的层数。
    // 单床保持72层；多床拼接窗口轴向尺寸为2*edge_layers。
    int multibed_edge_layers = 40;
    int iterations = 50;
    // 只保存 save_interval 的整数倍迭代，不额外生成 recon_final。
    int save_interval = 50;
    int gpus = 1;
    int angle_batch = 0;
    int gpu_slab_cache = 6;

    // 重建算法。默认 OMEGA 纯 MLEM；只有 OSL 才分配和计算 prior gradient。
    int algorithm = ALGORITHM_MLEM;
    bool algorithm_explicit = false;
    int implementation = IMPLEMENTATION_OMEGA;
    bool implementation_explicit = false;
    std::string implementation_source;
    bool legacy_version_alias_used = false;
    bool legacy_numeric_alias_used = false;
    float omega_epsilon = 1e-5f;
    float castor_min_update = 0.01f;
    float castor_max_update = 0.0f; // 0 表示不限制最大更新因子。
    float castor_threshold = 1e-10f;

    // 卷积时机：post 仅处理保存副本，smooth/intra 会回写迭代状态。
    std::string conv_mode = "none";
    int conv_implementation = CONV_IMPLEMENTATION_CASTOR_ADJOINT;
    bool conv_implementation_explicit = false;
    float psf_fwhm_mm[3] = {1.5f, 1.5f, 1.5f};
    float psf_nb_sigmas = 3.0f;
    float post_fwhm_mm = 0.45f;
    bool post_fwhm_explicit = false;
    std::string smooth_schedule = "constant";
    std::string smooth_mode = "periodic";
    float smooth_fwhm_start_mm = 0.60f;
    float smooth_fwhm_end_mm = 0.15f;
    int smooth_start_iter = 1;
    int smooth_apply_every = 1;

    // OSL prior。实现版本由 implementation 统一绑定，不能独立选择。
    MapType map_type = MAP_NONE;
    bool map_type_alias_used = false;
    bool prior_option_used = false;
    float map_beta = 0.0f;
    float tv_smoothing = 1e-4f;
    int mrp_radius[3] = {1, 1, 1};
    bool mrp_radius_explicit = false;
    float mrp_epsilon = 1e-8f;
    bool mrp_epsilon_explicit = false;
    bool mrp_no_norm = false;
    int mrp_neighborhood = MRP_NEIGHBORHOOD_AUTO;
    bool mrp_neighborhood_explicit = false;
    int mrp_box_order = 1;
    bool mrp_box_order_explicit = false;
    bool mrp_box_exclude_corners = false;
    float mrp_sphere_radius_mm = 1.0f;
    bool mrp_sphere_radius_explicit = false;

    /* 修正版 3D Butterworth：物理截止频率，单位 cycles/cm；0 表示关闭。 */
    bool butter_enable = false;
    float butter_cutoff_cycles_per_cm = 0.0f;
    int butter_order = 8;
    std::string butter_mode = "final";
    int butter_start_iter = 0;
    int butter_period = 5;
    int butter_last_n = 10;

    // 拼接后滤波只处理已经完成增益校正和 cos^2/sin^2 融合的派生体积。
    // 默认关闭，因此纯重建不会为这些输出分配任何额外 GPU 缓冲。
    std::string stitch_filter = "none";
    float stitch_smooth_fwhm_mm = 0.45f;
    float stitch_smooth_sigmas = 3.0f;
    float stitch_butter_cutoff_cycles_per_cm = 3.0f;
    int stitch_butter_order = 8;
    std::vector<std::pair<float, int>> stitch_butter_variants;

    std::string data_mode = "measured";
    bool no_norm_factor = false;
    fs::path norm_factor;
    std::string norm_factor_mode = "projection";
    bool no_detector_mask = false;
    // 默认删除重建后用于裁剪的完整尺寸中间结果，只保留 mask 和拼接结果。
    bool keep_raw = false;
    bool dry_run = false;
    bool self_test_algorithms = false;
};

struct BedInput {
    int number = 0;
    fs::path path;
};

struct Volume {
    // 数据按 z*(nx*ny)+y*nx+x 展平，x 维连续；所有值读写为 double。
    int nx = 0;
    int ny = 0;
    int nz = 0;
    double voxel_mm = 0.0;
    double origin_x_mm = 0.0;
    double origin_y_mm = 0.0;
    double origin_z_mm = 0.0;
    std::vector<double> data;
};

static void fail(const std::string& message) {
    throw std::runtime_error(message);
}

static bool finite_positive(double value) {
    return std::isfinite(value) && value > 0.0;
}

static int parse_int(const char* value, const char* name) {
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > 2147483647L) {
        fail(std::string(name) + " must be a positive integer");
    }
    return static_cast<int>(parsed);
}

static int parse_nonnegative_int(const char* value, const char* name) {
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed < 0 || parsed > 2147483647L) {
        fail(std::string(name) + " must be a non-negative integer");
    }
    return static_cast<int>(parsed);
}

static double parse_positive_double(const char* value, const char* name) {
    char* end = nullptr;
    double parsed = std::strtod(value, &end);
    if (!end || *end != '\0' || !finite_positive(parsed)) {
        fail(std::string(name) + " must be a finite positive number");
    }
    return parsed;
}

static double parse_nonnegative_double(const char* value, const char* name) {
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (!end || *end != '\0' || !std::isfinite(parsed) || parsed < 0.0) {
        fail(std::string(name) + " must be a finite non-negative number");
    }
    return parsed;
}

static const char* algorithm_name(int value) {
    return value == ALGORITHM_OSL ? "osl" : "mlem";
}

static const char* reference_implementation_name(int value) {
    return value == IMPLEMENTATION_CASTOR ? "castor" : "omega";
}

static const char* conv_implementation_name(int value) {
    if (value == CONV_IMPLEMENTATION_OMEGA) return "omega";
    if (value == CONV_IMPLEMENTATION_CASTOR) return "castor";
    return "castor-adjoint";
}

static const char* mrp_neighborhood_name(int value) {
    if (value == MRP_NEIGHBORHOOD_AUTO) return "auto";
    if (value == MRP_NEIGHBORHOOD_SIX_NEAREST) return "6-nearest";
    if (value == MRP_NEIGHBORHOOD_SPHERE) return "sphere";
    return "box";
}

static int parse_reference_implementation(const std::string& value,
                                          const char* option_name) {
    if (value == "omega") return IMPLEMENTATION_OMEGA;
    if (value == "castor") return IMPLEMENTATION_CASTOR;
    fail(std::string(option_name) + " must be omega or castor");
    return IMPLEMENTATION_OMEGA;
}

// 新旧版本参数最终都写入同一个字段。重复指定同一版本允许，指定不同版本则
// 立即报错，避免旧脚本在 OSL 与 MRP 之间形成静默混搭。
static void select_reference_implementation(BedMergeOptions& options,
                                            int implementation,
                                            const char* option_name,
                                            bool legacy_alias) {
    if (options.implementation_explicit &&
        options.implementation != implementation) {
        fail(std::string(option_name) + "=" +
             reference_implementation_name(implementation) +
             " conflicts with " + options.implementation_source + "=" +
             reference_implementation_name(options.implementation));
    }
    options.implementation = implementation;
    options.implementation_explicit = true;
    if (options.implementation_source.empty()) {
        options.implementation_source = option_name;
    }
    options.legacy_version_alias_used =
        options.legacy_version_alias_used || legacy_alias;
}

static void parse_positive_float3(const char* text, const char* name,
                                  float out[3]) {
    std::vector<std::string> fields = split(text, ',');
    if (fields.size() != 1 && fields.size() != 3) {
        fail(std::string(name) + " must be F or Fx,Fy,Fz");
    }
    for (size_t i = 0; i < fields.size(); ++i) {
        char* end = nullptr;
        const double value = std::strtod(fields[i].c_str(), &end);
        if (!end || *end != '\0' || !finite_positive(value) ||
            value > std::numeric_limits<float>::max()) {
            fail(std::string(name) + " must contain finite positive values");
        }
        out[i] = static_cast<float>(value);
    }
    if (fields.size() == 1) out[1] = out[2] = out[0];
}

static void parse_mrp_radius3(const char* text, int out[3]) {
    std::vector<std::string> fields = split(text, ',');
    if (fields.size() != 1 && fields.size() != 3) {
        fail("--mrp-radius must be R or Rx,Ry,Rz");
    }
    for (size_t i = 0; i < fields.size(); ++i) {
        char* end = nullptr;
        const long value = std::strtol(fields[i].c_str(), &end, 10);
        if (!end || *end != '\0' || value < 0 || value > MAX_MRP_RADIUS) {
            fail("--mrp-radius values must be integers in [0,2]");
        }
        out[i] = static_cast<int>(value);
    }
    if (fields.size() == 1) out[1] = out[2] = out[0];
}

// 解析中心 mask 尺寸，支持 100,100,72、100x100x72 和 100X100X72。
static void parse_mask_size(const char* text, int& nx, int& ny, int& nz) {
    std::string value(text);
    for (char& c : value) {
        if (c == 'x' || c == 'X' || c == ';') c = ',';
    }
    std::stringstream stream(value);
    std::vector<std::string> fields;
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    if (fields.size() != 3) {
        fail("--mask-size must be Nx,Ny,Nz, for example 100,100,72");
    }
    nx = parse_int(fields[0].c_str(), "mask Nx");
    ny = parse_int(fields[1].c_str(), "mask Ny");
    nz = parse_int(fields[2].c_str(), "mask Nz");
}

static const char* need_value(int argc, char** argv, int& index, const char* name) {
    if (index + 1 >= argc) fail(std::string(name) + " needs a value");
    return argv[++index];
}

static void bedmerge_print_usage() {
    // NVCC 13.1 会误解析紧跟在 UTF-8 中文字符后的转义换行，因此以 ASCII %s
    // 结束这些格式串，并把换行作为独立参数传入。
    std::printf(
        "用法: spect_recon_PSC_6module_24pin_YSO_bedmerge "
        "--proj-dir DIR --out DIR [options]\n\n"
        "输入目录必须包含 Bed1.bin，可继续包含 Bed2.bin、Bed3.bin 等。%s"
        "各床位保持独立局部图像，共享常驻 CSR/角度分块，最后在 CPU 端拼接。%s"
        "基础参数:\n"
        "  --matrix-dir DIR          系统矩阵目录 [MCmatrix_PSC]\n"
        "  --out DIR                 输出目录；相对名称自动置于 recon_output/\n"
        "  --bed-step-mm F           床位中心间距，单位 mm [16.8]\n"
        "  --mask-size Nx,Ny,Nz      中心输出 mask，单位体素 [100,100,72]\n"
        "  --gain-roi-diameter-mm F  增益拟合 ROI 直径，单位 mm [30]\n"
        "  --roi-diameter-mm F       重建圆柱 ROI 直径，单位 mm [30]\n"
        "  --roi-height-mm F         每床有效高度，单位 mm [66.0]\n"
        "  --multibed-edge-layers N  全局视野两端延长层数 [40]\n"
        "  --iters N                 MLEM/OSL 迭代次数 [50]\n"
        "  --save-every N            每 N 次保存 reconN；不生成 recon_final [50]\n"
        "  --gpus N                  CUDA GPU 数量 [1]\n"
        "  --angle-batch N           compact angle 分批；0=完整 CSR 常驻 [0]\n"
        "  --gpu-slab-cache N        GPU 常驻角度分块数；0=关闭 [6]\n\n"
        "重建算法与数值阈值:\n"
        "  --algorithm mlem|osl      重建更新；纯 MLEM 不分配 prior 缓冲 [mlem]\n"
        "  --implementation MODE     omega|castor；同时绑定 EM 和 prior [omega]\n"
        "  --omega-epsilon F         OMEGA 投影/图像域防零常数，必须 >0 [1e-5]\n"
        "  --castor-threshold F      CASToR g*A*x 阈值，必须 >0 [1e-10]\n"
        "  --castor-min-update F     CASToR MLEM/OSL 最小更新因子；0=关闭 [0.01]\n"
        "  --castor-max-update F     CASToR MLEM/OSL 最大更新因子；0=关闭 [0]\n"
        "  兼容别名: --osl-version/--mrp-version、--osl-epsilon,\n"
        "            --forward-threshold、--osl-min-update/--osl-max-update%s"
        "Gaussian 卷积:\n"
        "  --conv-mode MODE          none|psf|post|sieve|intra|smooth [none]\n"
        "                            psf=系统模型；post=仅处理保存副本；sieve=psf+post\n"
        "                            intra/smooth=迭代间平滑并回写状态%s"
        "  --conv-impl MODE          omega|castor|castor-adjoint [castor-adjoint]\n"
        "                            omega=中心采样+镜像；castor=严格源码行为\n"
        "                            castor-adjoint=CASToR 前向+严格伴随%s"
        "  --psf-fwhm F|Fx,Fy,Fz    PSF FWHM，单位 mm [1.5,1.5,1.5]\n"
        "                            CASToR 要求 Fx=Fy；FWHM 取决于系统空间分辨率\n"
        "  --psf-sigmas F            CASToR/castor-adjoint 截断 sigma 数 (0,10] [3]\n"
        "                            OMEGA 固定使用 2 sigma，因此忽略此参数\n"
        "  --post-fwhm F             post/intra Gaussian FWHM，单位 mm [0.45]\n"
        "  --smooth-mode MODE        every|periodic|final [periodic]\n"
        "  --smooth-schedule MODE    constant|linear|exponential [constant]\n"
        "  --smooth-fwhm-start F     linear/exponential 起始 FWHM，mm [0.60]\n"
        "  --smooth-fwhm-end F       linear/exponential 结束 FWHM，mm [0.15]\n"
        "  --smooth-start-iter N     intra 首次执行迭代 [1]\n"
        "  --smooth-every N          intra 执行周期 [1]\n\n"
        "OSL 空间 prior（版本必须与 --implementation 一致）:\n"
        "  --prior none|tv|mrp       空间 prior [none]\n"
        "  --map-type none|tv|mrp    --prior 的旧兼容名称\n"
        "  --map-beta F              OSL 分母中的先验权重 beta；启用先验时必须 >0\n"
        "  --tv-smoothing F          TV 根号内平滑量，必须 >0 [1e-4]\n"
        "                            仅 OMEGA；与图像强度尺度和 beta 共同调节\n"
        "  --mrp-radius R|Rx,Ry,Rz   MRP 半径，单位体素，每轴范围 [0,2] [1,1,1]\n"
        "  --mrp-epsilon F           OMEGA MRP 分母防零常数 [1e-8]\n"
        "  --mrp-no-norm             OMEGA 使用 x-median，不做 median 归一化%s"
        "  --mrp-neighborhood MODE   CASToR 6-nearest|box|sphere [6-nearest]\n"
        "  --mrp-box-order N         CASToR box 阶数，范围 [0,2] [1]\n"
        "  --mrp-box-exclude-corners CASToR box 丢弃最外层 8 个角点%s"
        "  --mrp-sphere-radius-mm F  CASToR sphere 半径，单位 mm [1]\n"
        "                            sphere 包围盒每轴最多 2 体素%s"
        "Butterworth 后处理:\n"
        "  --butter-cutoff-cm F      三维径向截止频率，cycles/cm；不填=关闭\n"
        "                            合法上限由体素 Nyquist=5/voxel_mm 决定\n"
        "  --butter-order N          阶数，范围 [1,32] [8]\n"
        "  --butter-mode MODE        every|final|periodic|last_n [final]\n"
        "  --butter-start-iter N     首次允许滤波轮次；0=旧调度，范围 [0,iters] [0]\n"
        "  --butter-period N         periodic 周期 [5]\n"
        "  --butter-last-n N         last_n 窗口 [10]\n\n"
        "拼接后一次性滤波（不回写单床重建）:\n"
        "  --stitch-filter MODE      none|smooth|butterworth|both [none]\n"
        "  --stitch-smooth-fwhm-mm F Gaussian FWHM，单位 mm [0.45]\n"
        "  --stitch-smooth-sigmas F  CASToR Gaussian 截断 sigma 数 (0,10] [3]\n"
        "  --stitch-butter-cutoff-cm F Butterworth 截止频率，cycles/cm [3]\n"
        "  --stitch-butter-order N   Butterworth 阶数，范围 [1,32] [8]\n\n"
        "  --stitch-butter-variant F,N 可重复指定多组截止频率和阶数\n\n"
        "数据与其他参数:\n"
        "  --data-mode measured|simulation [measured]；仿真默认 g_i=1\n"
        "  --no-detector-mask        关闭实测坏探测器 mask\n"
        "  --keep-raw                保留完整矩阵尺寸的单床中间体积%s"
        "  --no-norm-factor          强制 g_i=1，关闭因子自动搜索%s"
        "  --norm-factor FILE        显式归一化因子%s"
        "  --norm-factor-mode MODE   auto|matrix|projection [projection]\n"
        "  --dry-run                 只校验并打印配置\n"
        "  --self-test-algorithms    运行 PSF、ratio、MLEM/OSL、TV/MRP、Butterworth 与 ROI 数值自检\n"
        "  --help                    显示帮助\n",
        "\n\n", "\n", "\n", "\n\n", "\n", "\n", "\n", "\n",
        "\n", "\n", "\n");
}

static BedMergeOptions parse_options(int argc, char** argv) {
    BedMergeOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            bedmerge_print_usage();
            std::exit(0);
        } else if (arg == "--proj-dir") {
            options.proj_dir = need_value(argc, argv, i, "--proj-dir");
        } else if (arg == "--matrix-dir") {
            options.matrix_dir = need_value(argc, argv, i, "--matrix-dir");
        } else if (arg == "--out") {
            options.out = need_value(argc, argv, i, "--out");
        } else if (arg == "--bed-step-mm") {
            options.bed_step_mm = parse_positive_double(
                need_value(argc, argv, i, "--bed-step-mm"), "--bed-step-mm");
        } else if (arg == "--mask-size") {
            parse_mask_size(need_value(argc, argv, i, "--mask-size"),
                            options.mask_nx, options.mask_ny, options.mask_nz);
        } else if (arg == "--gain-roi-diameter-mm") {
            options.gain_roi_diameter_mm = parse_positive_double(
                need_value(argc, argv, i, "--gain-roi-diameter-mm"), "--gain-roi-diameter-mm");
        } else if (arg == "--roi-diameter-mm") {
            options.roi_diameter_mm = parse_positive_double(
                need_value(argc, argv, i, "--roi-diameter-mm"), "--roi-diameter-mm");
        } else if (arg == "--roi-height-mm") {
            options.roi_height_mm = parse_positive_double(
                need_value(argc, argv, i, "--roi-height-mm"), "--roi-height-mm");
        } else if (arg == "--multibed-edge-layers") {
            options.multibed_edge_layers = parse_int(
                need_value(argc, argv, i, "--multibed-edge-layers"),
                "--multibed-edge-layers");
        } else if (arg == "--iters") {
            options.iterations = parse_int(need_value(argc, argv, i, "--iters"), "--iters");
        } else if (arg == "--save-every") {
            options.save_interval = parse_int(
                need_value(argc, argv, i, "--save-every"), "--save-every");
        } else if (arg == "--gpus") {
            options.gpus = parse_int(need_value(argc, argv, i, "--gpus"), "--gpus");
        } else if (arg == "--angle-batch") {
            options.angle_batch = parse_nonnegative_int(
                need_value(argc, argv, i, "--angle-batch"), "--angle-batch");
        } else if (arg == "--gpu-slab-cache") {
            options.gpu_slab_cache = parse_nonnegative_int(
                need_value(argc, argv, i, "--gpu-slab-cache"), "--gpu-slab-cache");
        } else if (arg == "--algorithm") {
            const std::string value = need_value(argc, argv, i, "--algorithm");
            if (value == "mlem") options.algorithm = ALGORITHM_MLEM;
            else if (value == "osl") options.algorithm = ALGORITHM_OSL;
            else fail("--algorithm must be mlem or osl");
            options.algorithm_explicit = true;
        } else if (arg == "--implementation" || arg == "--osl-version" ||
                   arg == "--mrp-version") {
            const std::string value = need_value(argc, argv, i, arg.c_str());
            select_reference_implementation(
                options, parse_reference_implementation(value, arg.c_str()),
                arg.c_str(), arg != "--implementation");
        } else if (arg == "--omega-epsilon" || arg == "--osl-epsilon") {
            options.omega_epsilon = static_cast<float>(parse_positive_double(
                need_value(argc, argv, i, arg.c_str()), arg.c_str()));
            options.legacy_numeric_alias_used =
                options.legacy_numeric_alias_used || arg == "--osl-epsilon";
        } else if (arg == "--castor-min-update" || arg == "--osl-min-update") {
            options.castor_min_update = static_cast<float>(parse_nonnegative_double(
                need_value(argc, argv, i, arg.c_str()), arg.c_str()));
            options.legacy_numeric_alias_used =
                options.legacy_numeric_alias_used || arg == "--osl-min-update";
        } else if (arg == "--castor-max-update" || arg == "--osl-max-update") {
            options.castor_max_update = static_cast<float>(parse_nonnegative_double(
                need_value(argc, argv, i, arg.c_str()), arg.c_str()));
            options.legacy_numeric_alias_used =
                options.legacy_numeric_alias_used || arg == "--osl-max-update";
        } else if (arg == "--castor-threshold" ||
                   arg == "--forward-threshold") {
            options.castor_threshold = static_cast<float>(parse_positive_double(
                need_value(argc, argv, i, arg.c_str()), arg.c_str()));
            options.legacy_numeric_alias_used =
                options.legacy_numeric_alias_used || arg == "--forward-threshold";
        } else if (arg == "--conv-mode") {
            options.conv_mode = need_value(argc, argv, i, "--conv-mode");
        } else if (arg == "--conv-impl") {
            const std::string value = need_value(argc, argv, i, "--conv-impl");
            if (value == "castor-adjoint" || value == "adjoint") {
                options.conv_implementation = CONV_IMPLEMENTATION_CASTOR_ADJOINT;
            } else if (value == "omega") {
                options.conv_implementation = CONV_IMPLEMENTATION_OMEGA;
            } else if (value == "castor") {
                options.conv_implementation = CONV_IMPLEMENTATION_CASTOR;
            } else {
                fail("--conv-impl must be omega, castor, or castor-adjoint");
            }
            options.conv_implementation_explicit = true;
        } else if (arg == "--psf-fwhm") {
            parse_positive_float3(need_value(argc, argv, i, "--psf-fwhm"),
                                  "--psf-fwhm", options.psf_fwhm_mm);
        } else if (arg == "--psf-sigmas") {
            options.psf_nb_sigmas = static_cast<float>(parse_positive_double(
                need_value(argc, argv, i, "--psf-sigmas"), "--psf-sigmas"));
        } else if (arg == "--post-fwhm") {
            options.post_fwhm_mm = static_cast<float>(parse_positive_double(
                need_value(argc, argv, i, "--post-fwhm"), "--post-fwhm"));
            options.post_fwhm_explicit = true;
        } else if (arg == "--smooth-mode") {
            options.smooth_mode = need_value(argc, argv, i, "--smooth-mode");
        } else if (arg == "--smooth-schedule" || arg == "--post-schedule") {
            options.smooth_schedule = need_value(argc, argv, i, arg.c_str());
        } else if (arg == "--smooth-fwhm-start" || arg == "--post-fwhm-start") {
            options.smooth_fwhm_start_mm = static_cast<float>(parse_positive_double(
                need_value(argc, argv, i, arg.c_str()), arg.c_str()));
        } else if (arg == "--smooth-fwhm-end" || arg == "--post-fwhm-end") {
            options.smooth_fwhm_end_mm = static_cast<float>(parse_positive_double(
                need_value(argc, argv, i, arg.c_str()), arg.c_str()));
        } else if (arg == "--smooth-start-iter" || arg == "--post-start-iter") {
            options.smooth_start_iter = parse_int(
                need_value(argc, argv, i, arg.c_str()), arg.c_str());
        } else if (arg == "--smooth-every" || arg == "--post-every") {
            options.smooth_apply_every = parse_int(
                need_value(argc, argv, i, arg.c_str()), arg.c_str());
        } else if (arg == "--prior" || arg == "--map-type") {
            options.map_type = parse_map_type(
                need_value(argc, argv, i, arg.c_str()));
            if (options.map_type != MAP_NONE && options.map_type != MAP_TV &&
                options.map_type != MAP_MRP) {
                fail("--prior/--map-type supports only none, tv, or mrp");
            }
            if (arg == "--map-type") options.map_type_alias_used = true;
            else options.prior_option_used = true;
        } else if (arg == "--map-beta") {
            options.map_beta = static_cast<float>(parse_positive_double(
                need_value(argc, argv, i, "--map-beta"), "--map-beta"));
        } else if (arg == "--tv-smoothing") {
            options.tv_smoothing = static_cast<float>(parse_positive_double(
                need_value(argc, argv, i, "--tv-smoothing"), "--tv-smoothing"));
        } else if (arg == "--mrp-radius") {
            parse_mrp_radius3(need_value(argc, argv, i, "--mrp-radius"),
                              options.mrp_radius);
            options.mrp_radius_explicit = true;
        } else if (arg == "--mrp-epsilon") {
            options.mrp_epsilon = static_cast<float>(parse_positive_double(
                need_value(argc, argv, i, "--mrp-epsilon"), "--mrp-epsilon"));
            options.mrp_epsilon_explicit = true;
        } else if (arg == "--mrp-no-norm") {
            options.mrp_no_norm = true;
        } else if (arg == "--mrp-neighborhood") {
            const std::string value = need_value(argc, argv, i,
                                                 "--mrp-neighborhood");
            if (value == "6-nearest") {
                options.mrp_neighborhood = MRP_NEIGHBORHOOD_SIX_NEAREST;
            } else if (value == "box") {
                options.mrp_neighborhood = MRP_NEIGHBORHOOD_BOX;
            } else if (value == "sphere") {
                options.mrp_neighborhood = MRP_NEIGHBORHOOD_SPHERE;
            } else {
                fail("--mrp-neighborhood must be 6-nearest, box, or sphere");
            }
            options.mrp_neighborhood_explicit = true;
        } else if (arg == "--mrp-box-order") {
            options.mrp_box_order = parse_nonnegative_int(
                need_value(argc, argv, i, "--mrp-box-order"),
                "--mrp-box-order");
            options.mrp_box_order_explicit = true;
        } else if (arg == "--mrp-box-exclude-corners") {
            options.mrp_box_exclude_corners = true;
        } else if (arg == "--mrp-sphere-radius-mm") {
            options.mrp_sphere_radius_mm = static_cast<float>(parse_positive_double(
                need_value(argc, argv, i, "--mrp-sphere-radius-mm"),
                "--mrp-sphere-radius-mm"));
            options.mrp_sphere_radius_explicit = true;
        } else if (arg == "--butter-cutoff-cm") {
            options.butter_cutoff_cycles_per_cm =
                static_cast<float>(parse_positive_double(
                    need_value(argc, argv, i, "--butter-cutoff-cm"),
                    "--butter-cutoff-cm"));
            options.butter_enable = true;
        } else if (arg == "--butter-order") {
            options.butter_order = parse_int(
                need_value(argc, argv, i, "--butter-order"), "--butter-order");
        } else if (arg == "--butter-mode") {
            options.butter_mode = need_value(argc, argv, i, "--butter-mode");
        } else if (arg == "--butter-start-iter") {
            options.butter_start_iter = parse_int(
                need_value(argc, argv, i, "--butter-start-iter"),
                "--butter-start-iter");
        } else if (arg == "--butter-period") {
            options.butter_period = parse_int(
                need_value(argc, argv, i, "--butter-period"), "--butter-period");
        } else if (arg == "--butter-last-n") {
            options.butter_last_n = parse_int(
                need_value(argc, argv, i, "--butter-last-n"), "--butter-last-n");
        } else if (arg == "--stitch-filter") {
            options.stitch_filter = need_value(argc, argv, i, "--stitch-filter");
        } else if (arg == "--stitch-smooth-fwhm-mm") {
            options.stitch_smooth_fwhm_mm = static_cast<float>(parse_positive_double(
                need_value(argc, argv, i, "--stitch-smooth-fwhm-mm"),
                "--stitch-smooth-fwhm-mm"));
        } else if (arg == "--stitch-smooth-sigmas") {
            options.stitch_smooth_sigmas = static_cast<float>(parse_positive_double(
                need_value(argc, argv, i, "--stitch-smooth-sigmas"),
                "--stitch-smooth-sigmas"));
        } else if (arg == "--stitch-butter-cutoff-cm") {
            options.stitch_butter_cutoff_cycles_per_cm =
                static_cast<float>(parse_positive_double(
                    need_value(argc, argv, i, "--stitch-butter-cutoff-cm"),
                    "--stitch-butter-cutoff-cm"));
        } else if (arg == "--stitch-butter-order") {
            options.stitch_butter_order = parse_int(
                need_value(argc, argv, i, "--stitch-butter-order"),
                "--stitch-butter-order");
        } else if (arg == "--stitch-butter-variant") {
            const std::string value = need_value(
                argc, argv, i, "--stitch-butter-variant");
            const std::vector<std::string> fields = split(value, ',');
            if (fields.size() != 2) {
                fail("--stitch-butter-variant must be cutoff_cycles_per_cm,order");
            }
            const float cutoff = static_cast<float>(parse_positive_double(
                fields[0].c_str(), "stitch Butterworth cutoff"));
            const int order = parse_int(
                fields[1].c_str(), "stitch Butterworth order");
            if (order > 32) {
                fail("stitch Butterworth variant order must be in [1,32]");
            }
            options.stitch_butter_variants.emplace_back(cutoff, order);
        } else if (arg == "--data-mode") {
            options.data_mode = need_value(argc, argv, i, "--data-mode");
            if (options.data_mode != "measured" && options.data_mode != "simulation") {
                fail("--data-mode must be measured or simulation");
            }
        } else if (arg == "--no-detector-mask") {
            options.no_detector_mask = true;
        } else if (arg == "--keep-raw") {
            options.keep_raw = true;
        } else if (arg == "--no-norm-factor") {
            options.no_norm_factor = true;
        } else if (arg == "--norm-factor") {
            options.norm_factor = need_value(argc, argv, i, "--norm-factor");
        } else if (arg == "--norm-factor-mode") {
            options.norm_factor_mode = need_value(argc, argv, i, "--norm-factor-mode");
            if (options.norm_factor_mode != "auto" &&
                options.norm_factor_mode != "matrix" &&
                options.norm_factor_mode != "projection") {
                fail("--norm-factor-mode must be auto, matrix, or projection");
            }
        } else if (arg == "--dry-run") {
            options.dry_run = true;
        } else if (arg == "--self-test-algorithms") {
            options.self_test_algorithms = true;
        } else {
            fail("unknown option: " + arg);
        }
    }
    if (options.self_test_algorithms) return options;
    if (options.legacy_version_alias_used) {
        std::fprintf(stderr,
                     "[compatibility] --osl-version/--mrp-version now select the complete EM+prior implementation; prefer --implementation.\n");
    }
    if (options.legacy_numeric_alias_used) {
        std::fprintf(stderr,
                     "[compatibility] legacy OSL numeric option accepted; prefer --omega-epsilon or --castor-*.\n");
    }
    if (options.map_type_alias_used && options.prior_option_used) {
        fail("do not combine --prior with the legacy --map-type alias");
    }
    if (options.map_type_alias_used && options.map_type != MAP_NONE &&
        !options.algorithm_explicit) {
        options.algorithm = ALGORITHM_OSL;
        std::fprintf(stderr,
                     "[compatibility] --map-type is deprecated; selecting --algorithm osl. Use --prior in new commands.\n");
    }
    if (options.prior_option_used && options.map_type != MAP_NONE &&
        !options.algorithm_explicit) {
        fail("--prior tv/mrp requires explicit --algorithm osl");
    }
    if (options.algorithm == ALGORITHM_MLEM && options.map_type != MAP_NONE) {
        fail("pure MLEM cannot include TV/MRP in its update; use --algorithm osl");
    }
    if (options.algorithm == ALGORITHM_OSL && options.map_type == MAP_NONE) {
        fail("--algorithm osl requires --prior tv or mrp");
    }
    if (options.algorithm == ALGORITHM_OSL && !(options.map_beta > 0.0f)) {
        fail("--algorithm osl requires --map-beta >0");
    }
    if (options.algorithm == ALGORITHM_MLEM && options.map_beta > 0.0f) {
        fail("--map-beta is only valid with --algorithm osl and a prior");
    }
    if (options.castor_max_update > 0.0f &&
        options.castor_max_update < options.castor_min_update) {
        fail("--castor-max-update must be 0 or >= --castor-min-update");
    }
    if (options.algorithm == ALGORITHM_OSL && options.map_type == MAP_TV &&
        options.implementation == IMPLEMENTATION_CASTOR) {
        fail("CASToR 3.2 has no TV penalty; use --implementation omega for TV");
    }
    if (options.algorithm == ALGORITHM_OSL && options.map_type == MAP_MRP) {
        if (options.implementation == IMPLEMENTATION_OMEGA) {
            if (options.mrp_neighborhood_explicit ||
                options.mrp_box_order_explicit ||
                options.mrp_box_exclude_corners ||
                options.mrp_sphere_radius_explicit) {
                fail("CASToR MRP neighborhood options cannot be used with --implementation omega");
            }
        } else {
            if (options.mrp_radius_explicit || options.mrp_epsilon_explicit ||
                options.mrp_no_norm) {
                fail("OMEGA MRP radius/epsilon/no-norm options cannot be used with --implementation castor");
            }
            if (options.mrp_neighborhood == MRP_NEIGHBORHOOD_AUTO) {
                // CASToR 配套 MRP.conf 的第一个有效配置是 6-nearest。
                options.mrp_neighborhood = MRP_NEIGHBORHOOD_SIX_NEAREST;
            }
        }
    }
    if (options.proj_dir.empty() || options.out.empty()) {
        fail("--proj-dir and --out are required");
    }
    // Keep output clutter out of the working directory without overriding an
    // explicit absolute destination.  Normalising first also accepts
    // ./recon_output/name without creating recon_output/recon_output/name.
    if (options.out.is_relative()) {
        const fs::path normal_out = options.out.lexically_normal();
        const auto first = normal_out.begin();
        if (first == normal_out.end() || *first != fs::path("recon_output")) {
            options.out = fs::path("recon_output") / normal_out;
        } else {
            options.out = normal_out;
        }
    }
    if (options.save_interval > options.iterations) {
        fail("--save-every must not exceed --iters because recon_final is not written");
    }
    const bool valid_conv_mode =
        options.conv_mode == "none" || options.conv_mode == "psf" ||
        options.conv_mode == "post" || options.conv_mode == "sieve" ||
        options.conv_mode == "intra" || options.conv_mode == "smooth";
    if (!valid_conv_mode) {
        fail("--conv-mode must be none, psf, post, sieve, intra, or smooth");
    }
    if (options.conv_mode == "sieve") {
        if (options.conv_implementation_explicit &&
            options.conv_implementation != CONV_IMPLEMENTATION_CASTOR) {
            fail("strict CASToR sieve requires --conv-impl castor");
        }
        options.conv_implementation = CONV_IMPLEMENTATION_CASTOR;
        if (options.post_fwhm_explicit) {
            fail("strict CASToR sieve reuses --psf-fwhm and rejects --post-fwhm");
        }
    }
    if ((options.conv_mode == "psf" || options.conv_mode == "sieve") &&
        convolution_requires_castor_norm(options.conv_implementation) &&
        fabsf(options.psf_fwhm_mm[0] - options.psf_fwhm_mm[1]) >
            1e-6f * fmaxf(1.0f, options.psf_fwhm_mm[0])) {
        fail("CASToR stationary Gaussian requires equal x/y transaxial FWHM");
    }
    if (options.psf_nb_sigmas > 10.0f) {
        fail("--psf-sigmas must be in (0,10]");
    }
    if (options.smooth_schedule != "constant" &&
        options.smooth_schedule != "linear" &&
        options.smooth_schedule != "exponential") {
        fail("--smooth-schedule must be constant, linear, or exponential");
    }
    if (options.smooth_mode != "every" && options.smooth_mode != "periodic" &&
        options.smooth_mode != "final") {
        fail("--smooth-mode must be every, periodic, or final");
    }
    if (options.smooth_start_iter < 1 ||
        options.smooth_start_iter > options.iterations) {
        fail("--smooth-start-iter must be in [1,iters]");
    }
    if (options.smooth_apply_every < 1) {
        fail("--smooth-every must be >=1");
    }
    if (options.mrp_box_order > MAX_MRP_RADIUS) {
        fail("--mrp-box-order must be in [0,2]");
    }
    if (options.map_type == MAP_MRP &&
        options.implementation == IMPLEMENTATION_CASTOR &&
        options.mrp_no_norm) {
        fail("--mrp-no-norm is an OMEGA-only option");
    }
    if (options.butter_order < 1 || options.butter_order > 32) {
        fail("--butter-order must be in [1,32]");
    }
    if (options.butter_mode != "every" && options.butter_mode != "final" &&
        options.butter_mode != "periodic" && options.butter_mode != "last_n") {
        fail("--butter-mode must be every, final, periodic, or last_n");
    }
    if (options.butter_start_iter < 0 ||
        options.butter_start_iter > options.iterations) {
        fail("--butter-start-iter must be in [0,iters]");
    }
    if (options.butter_period < 1) fail("--butter-period must be >=1");
    if (options.butter_last_n < 1) fail("--butter-last-n must be >=1");
    if (options.stitch_filter != "none" &&
        options.stitch_filter != "smooth" &&
        options.stitch_filter != "butterworth" &&
        options.stitch_filter != "both") {
        fail("--stitch-filter must be none, smooth, butterworth, or both");
    }
    if (options.stitch_smooth_sigmas > 10.0f) {
        fail("--stitch-smooth-sigmas must be in (0,10]");
    }
    if (options.stitch_butter_order < 1 || options.stitch_butter_order > 32) {
        fail("--stitch-butter-order must be in [1,32]");
    }
    if (options.stitch_butter_variants.empty()) {
        options.stitch_butter_variants.emplace_back(
            options.stitch_butter_cutoff_cycles_per_cm,
            options.stitch_butter_order);
    }
    if (!fs::is_directory(options.proj_dir)) fail("projection directory not found: " + options.proj_dir.string());
    if (!fs::is_directory(options.matrix_dir)) fail("matrix directory not found: " + options.matrix_dir.string());
    if (fs::exists(options.out)) fail("output directory already exists: " + options.out.string());
    if (options.no_norm_factor && (!options.norm_factor.empty() || options.norm_factor_mode != "projection")) {
        fail("--no-norm-factor conflicts with --norm-factor or an explicit non-default mode");
    }
    if (!options.norm_factor.empty() && !fs::is_regular_file(options.norm_factor)) {
        fail("normalization factor not found: " + options.norm_factor.string());
    }
    if (options.data_mode == "simulation" && !options.no_norm_factor &&
        options.norm_factor.empty()) {
        options.no_norm_factor = true;
        std::printf("[simulation] no explicit --norm-factor: disable automatic "
                    "normalisation-factor discovery (g_i=1)\n");
    }
    return options;
}

// 只接收连续编号的 Bed1.bin、Bed2.bin...，避免床位顺序与步进对应错误。
static std::vector<BedInput> discover_beds(const fs::path& root) {
    std::regex pattern("^[Bb]ed([0-9]+)\\.bin$");
    std::vector<BedInput> beds;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        std::smatch match;
        const std::string name = entry.path().filename().string();
        if (!std::regex_match(name, match, pattern)) continue;
        long number = std::stol(match[1].str());
        if (number <= 0 || number > 1000000) fail("invalid bed number: " + name);
        beds.push_back({static_cast<int>(number), fs::absolute(entry.path())});
    }
    std::sort(beds.begin(), beds.end(), [](const BedInput& a, const BedInput& b) {
        return a.number < b.number;
    });
    if (beds.empty()) fail("at least one BedN.bin file is required");
    for (size_t i = 0; i < beds.size(); ++i) {
        if (beds[i].number != static_cast<int>(i + 1)) {
            fail("bed numbers must be continuous from Bed1");
        }
    }
    return beds;
}

static void make_parent_dirs(const fs::path& path) {
    std::error_code error;
    fs::create_directories(path, error);
    if (error) fail("cannot create directory: " + path.string() + ": " + error.message());
}

static double read_metadata_value(const fs::path& metadata_path,
                                  const std::string& key) {
    std::ifstream input(metadata_path);
    if (!input) fail("missing matrix metadata: " + metadata_path.string());
    std::string line;
    while (std::getline(input, line)) {
        const size_t separator = line.find('=');
        if (separator == std::string::npos || line.substr(0, separator) != key) continue;
        const std::string value_text = line.substr(separator + 1);
        char* end = nullptr;
        const double value = std::strtod(value_text.c_str(), &end);
        if (!end || *end != '\0' || !finite_positive(value)) {
            fail("invalid metadata value for " + key);
        }
        return value;
    }
    // 某些系统矩阵生成器只在 metadata.bin 中保存权威几何信息。CUDA 读取器已使用
    // 该二进制记录；这里读取同一组值，保证裁剪和拼接包装层兼容这些矩阵。
    if (key == "Nx" || key == "Ny" || key == "Nz" || key == "voxel_size_mm") {
        const fs::path binary_path = metadata_path.parent_path() / "metadata.bin";
        std::ifstream binary(binary_path, std::ios::binary);
        if (binary) {
            const std::streamoff offset =
                (key == "Nx") ? 32 :
                (key == "Ny") ? 36 :
                (key == "Nz") ? 40 : 44;
            binary.seekg(offset, std::ios::beg);
            unsigned char bytes[4] = {0, 0, 0, 0};
            binary.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
            if (binary.gcount() == static_cast<std::streamsize>(sizeof(bytes))) {
                const uint32_t bits = static_cast<uint32_t>(bytes[0]) |
                                      (static_cast<uint32_t>(bytes[1]) << 8) |
                                      (static_cast<uint32_t>(bytes[2]) << 16) |
                                      (static_cast<uint32_t>(bytes[3]) << 24);
                if (key == "voxel_size_mm") {
                    float value = 0.0f;
                    std::memcpy(&value, &bits, sizeof(value));
                    if (finite_positive(value)) {
                        std::fprintf(stderr,
                                     "[metadata] %s missing from metadata.txt; using metadata.bin value %.9g\n",
                                     key.c_str(), static_cast<double>(value));
                        return static_cast<double>(value);
                    }
                } else {
                    const uint32_t value = bits;
                    if (value > 0 && value <= 2147483647u) {
                        std::fprintf(stderr,
                                     "[metadata] %s missing from metadata.txt; using metadata.bin value %u\n",
                                     key.c_str(), value);
                        return static_cast<double>(value);
                    }
                }
            }
        }
    }
    fail("metadata key not found: " + key);
    return 0.0;
}

static uint16_t read_le16(const unsigned char* p) {
    return static_cast<uint16_t>(p[0]) |
           (static_cast<uint16_t>(p[1]) << 8);
}

static uint32_t read_le32(const unsigned char* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

static float read_le_float(const unsigned char* p) {
    uint32_t bits = read_le32(p);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// 读取指定保存轮次的 NIfTI-1 成对 hdr/img，并检查尺寸、体素大小和 NaN/Inf。
static Volume read_nifti(const fs::path& directory, const std::string& stem) {
    const fs::path hdr_path = directory / (stem + ".hdr");
    const fs::path img_path = directory / (stem + ".img");
    std::ifstream header_file(hdr_path, std::ios::binary);
    if (!header_file) fail("missing NIfTI header: " + hdr_path.string());
    unsigned char header[348] = {};
    header_file.read(reinterpret_cast<char*>(header), sizeof(header));
    if (header_file.gcount() != static_cast<std::streamsize>(sizeof(header)) ||
        read_le32(header) != 348u) {
        fail("invalid NIfTI-1 header: " + hdr_path.string());
    }
    Volume volume;
    volume.nx = static_cast<int>(read_le16(header + 42));
    volume.ny = static_cast<int>(read_le16(header + 44));
    volume.nz = static_cast<int>(read_le16(header + 46));
    volume.voxel_mm = read_le_float(header + 80);
    volume.origin_x_mm = read_le_float(header + 292);
    volume.origin_y_mm = read_le_float(header + 308);
    volume.origin_z_mm = read_le_float(header + 324);
    if (volume.nx <= 0 || volume.ny <= 0 || volume.nz <= 0 ||
        !finite_positive(volume.voxel_mm)) {
        fail("invalid NIfTI geometry: " + hdr_path.string());
    }
    const size_t count = static_cast<size_t>(volume.nx) * volume.ny * volume.nz;
    volume.data.resize(count);
    std::ifstream image_file(img_path, std::ios::binary);
    if (!image_file) fail("missing NIfTI image: " + img_path.string());
    image_file.read(reinterpret_cast<char*>(volume.data.data()),
                    static_cast<std::streamsize>(count * sizeof(double)));
    if (image_file.gcount() != static_cast<std::streamsize>(count * sizeof(double))) {
        fail("NIfTI image has unexpected size: " + img_path.string());
    }
    for (double value : volume.data) {
        if (!std::isfinite(value) || value < 0.0) fail("invalid voxel in " + img_path.string());
    }
    return volume;
}

static float scheduled_smooth_fwhm(const BedMergeOptions& options, int iter) {
    if (options.smooth_schedule == "constant") return options.post_fwhm_mm;
    const float t = options.iterations <= 1
        ? 0.0f
        : (float)(iter - 1) / (float)(options.iterations - 1);
    if (options.smooth_schedule == "linear") {
        return options.smooth_fwhm_start_mm +
               t * (options.smooth_fwhm_end_mm - options.smooth_fwhm_start_mm);
    }
    const float start = options.smooth_fwhm_start_mm;
    const float end = options.smooth_fwhm_end_mm;
    return start * expf(-logf(start / end) * t);
}

// 一个 CUDA 调度上下文保存所有床位，但每个 BedN 仍拥有独立的局部似然和图像状态。
// 床位重叠不会在迭代中相加，跨床操作只发生在最终 CPU 增益拟合与拼接阶段。
static int run_independent_iteration_major_reconstruction(
    const BedMergeOptions& options, const std::vector<BedInput>& beds,
    const fs::path& raw_root) {
    const MatrixMeta meta = read_metadata(
        (options.matrix_dir / "metadata.bin").string());
    const int64_t local_n = meta.total_voxels;
    if ((int64_t)meta.Nx * meta.Ny * meta.Nz != local_n) {
        fail("metadata.bin has inconsistent Nx*Ny*Nz and total_voxels");
    }
    if (options.butter_enable) {
        const float nyquist_cycles_per_cm = 5.0f / meta.voxel_size;
        if (options.butter_cutoff_cycles_per_cm > nyquist_cycles_per_cm) {
            std::ostringstream message;
            message << "--butter-cutoff-cm "
                    << options.butter_cutoff_cycles_per_cm
                    << " exceeds voxel Nyquist " << nyquist_cycles_per_cm
                    << " cycles/cm (voxel=" << meta.voxel_size << " mm)";
            fail(message.str());
        }
    }

    const SymMode sym_mode = parse_sym_mode("auto", meta);
    const int sym_count = xy_sym_count(sym_mode);
    const int z_count = ((mode_uses_z(sym_mode) || meta.use_zsym != 0) &&
                         meta.compact_det_z < meta.det_nz) ? 2 : 1;
    const int full_rpa = meta.det_nx * meta.det_nz;
    const size_t bed_count = beds.size();
    const float effective_roi_height_mm = bed_count == 1
        ? 72.0f * meta.voxel_size
        : (float)meta.Nz * meta.voxel_size;
    const bool use_detector_mask =
        options.data_mode == "measured" && !options.no_detector_mask;
    const bool use_psf =
        options.conv_mode == "psf" || options.conv_mode == "sieve";
    const bool use_post =
        options.conv_mode == "post" || options.conv_mode == "sieve";
    const bool use_intra =
        options.conv_mode == "intra" || options.conv_mode == "smooth";
    const bool track_castor_threshold =
        options.implementation == IMPLEMENTATION_CASTOR;
    const bool need_convolution_norm =
        convolution_requires_castor_norm(options.conv_implementation);

    if (options.algorithm == ALGORITHM_OSL && options.map_type == MAP_MRP &&
        options.implementation == IMPLEMENTATION_CASTOR &&
        options.mrp_neighborhood == MRP_NEIGHBORHOOD_SPHERE &&
        std::max(1, (int)(options.mrp_sphere_radius_mm / meta.voxel_size)) >
        MAX_MRP_RADIUS) {
        fail("CASToR MRP sphere radius exceeds the supported two-voxel bounding radius");
    }

    if ((sym_mode == SYM_45 || sym_mode == SYM_45Z || sym_mode == SYM_90) &&
        meta.Nx != meta.Ny) {
        fail("45/90 degree system-matrix symmetry requires Nx == Ny");
    }

    // 所有投影在主机端只保存一份，每个活动 worker 也只上传一份。worker 通过
    // d_P_all + bed_index*projection_stride 选择当前床位。
    std::vector<float> p_all;
    size_t projection_stride = 0;
    int projection_angles = 0;
    int projection_rpa = 0;
    for (size_t b = 0; b < bed_count; ++b) {
        int na = 0;
        int rpa = 0;
        std::vector<double> projection = read_projection_bin_double(
            beds[b].path.string(), &na, &rpa);
        if (na != meta.full_num_angles || rpa != full_rpa) {
            fail("projection shape mismatch for " + beds[b].path.string() +
                 ": expected (angles,rpa)=(" +
                 std::to_string(meta.full_num_angles) + "," +
                 std::to_string(full_rpa) + ")");
        }
        validate_projection_counts(projection, beds[b].path.string());
        if (b == 0) {
            projection_angles = na;
            projection_rpa = rpa;
            projection_stride = projection.size();
            p_all.reserve(projection_stride * bed_count);
        } else if (projection.size() != projection_stride) {
            fail("projection length mismatch between beds");
        }

        if (use_detector_mask) {
            const BadDetectorMaskStats stats = zero_link4_fpga2_det4(
                projection, na, meta.det_nx, meta.det_nz);
            std::printf("[input] Bed%zu measured detector mask: bins=%" PRId64
                        " removed_counts=%.6e\n",
                        b + 1, stats.bins, stats.removed_sum);
        }
        double counts = 0.0;
        for (double value : projection) counts += value;
        const size_t offset = p_all.size();
        p_all.resize(offset + projection_stride);
        for (size_t i = 0; i < projection_stride; ++i) {
            p_all[offset + i] = static_cast<float>(projection[i]);
        }
        std::printf("[input] Bed%zu counts=%.6e file=%s\n", b + 1, counts,
                    beds[b].path.string().c_str());
    }

    // 复用底层因子读取器，确保 matrix/projection 因子方向与底层引擎一致。除非用户
    // 显式提供因子，仿真数据进入此处时 no_norm_factor=true。
    ReconConfig factor_cfg;
    factor_cfg.matrix_dir = options.matrix_dir.string();
    factor_cfg.norm_factor_enable = !options.no_norm_factor;
    factor_cfg.norm_factor_path = options.norm_factor.string();
    // 自动发现实测因子时保留历史 auto 语义；外层默认 projection 只用于用户显式
    // 指定因子文件的情况。
    factor_cfg.norm_factor_mode = options.norm_factor.empty()
        ? "auto" : options.norm_factor_mode;
    std::vector<float> g_full(projection_stride, 1.0f);
    const std::string factor_path = find_normalization_factor(factor_cfg);
    if (!factor_path.empty()) {
        const std::string factor_mode = resolve_normalization_factor_mode(
            factor_cfg, factor_path);
        g_full = read_system_response_factor(projection_angles, projection_rpa,
                                             factor_path, factor_mode);
        std::printf("[normalisation] %s (%s)\n", factor_path.c_str(),
                    factor_mode.c_str());
    } else {
        std::printf("[normalisation] g_i=1\n");
    }
    if (use_detector_mask) {
        zero_link4_fpga2_det4(g_full, projection_angles,
                              meta.det_nx, meta.det_nz);
    }

    int available_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&available_gpus));
    if (available_gpus <= 0) fail("no CUDA device is available");
    int num_gpus = std::min(options.gpus, available_gpus);
    const bool use_angle_batch = options.angle_batch > 0;
    if (use_angle_batch && num_gpus != 1) {
        std::printf("[scheduler] --angle-batch uses GPU0 only; forcing --gpus 1\n");
        num_gpus = 1;
    }
    enable_peer_access(num_gpus);

    const std::vector<ChunkInfo> chunks = scan_chunks(
        options.matrix_dir.string());
    int64_t active_roi_voxels = 0;
    const std::vector<uint8_t> roi_mask = build_centered_cylinder_roi_mask(
        meta.Nx, meta.Ny, meta.Nz, meta.voxel_size,
        (float)options.roi_diameter_mm, effective_roi_height_mm,
        active_roi_voxels);
    if (active_roi_voxels <= 0) fail("cylindrical reconstruction ROI contains no voxels");
    std::printf("[roi] active voxels=%" PRId64 "/%d (%.3f%%), "
                "diameter=%.6g mm height=%.6g mm\n",
                active_roi_voxels, meta.total_voxels,
                100.0 * (double)active_roi_voxels / (double)meta.total_voxels,
                options.roi_diameter_mm, (double)effective_roi_height_mm);

    // 每个驻留 CSR 或临时角度分块都使用同一 ROI 列表裁剪。图像列空间保持完整，
    // 但 ROI 外非零元不会上传 GPU，也不会进入 sensitivity/前投影/反投影。
    auto build_roi_csr = [&](int start_ck, int end_ck) {
        TimedHostCSR built = build_worker_csr_timed(
            chunks, start_ck, end_ck, meta.compact_rpa, meta.total_voxels);
        const int64_t original_nnz = built.csr.nnz;
        const auto prune_start = std::chrono::high_resolution_clock::now();
        const int64_t removed = prune_host_csr_to_roi(built.csr, roi_mask);
        if (built.csr.nnz <= 0) {
            fail("cylindrical ROI leaves an empty system-matrix angle block");
        }
        built.build_sec += std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - prune_start).count();
        const double kept_percent = original_nnz > 0
            ? 100.0 * (double)built.csr.nnz / (double)original_nnz : 100.0;
        std::printf("[roi] CSR angles [%d,%d): nnz=%" PRId64
                    "/%" PRId64 " (%.3f%%), removed=%" PRId64 "\n",
                    start_ck, end_ck, built.csr.nnz, original_nnz,
                    kept_percent, removed);
        return built;
    };
    std::vector<GPUWorker> workers;
    std::vector<GPUWorker> cached_slab_workers;
    int cached_end_ck = 0;
    if (!use_angle_batch) {
        workers.resize((size_t)num_gpus);
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            const int start_ck = gpu * meta.compact_num_angles / num_gpus;
            const int end_ck = (gpu + 1) * meta.compact_num_angles / num_gpus;
            const TimedHostCSR built = build_roi_csr(start_ck, end_ck);
            workers[(size_t)gpu].init(gpu, built.csr, meta, p_all,
                                      projection_stride, (int)bed_count, g_full,
                                      track_castor_threshold);
            std::printf("[scheduler] resident CSR GPU%d angles [%d,%d), "
                        "build=%.3fs\n", gpu, start_ck, end_ck,
                        built.build_sec);
        }
    } else if (options.gpu_slab_cache > 0) {
        const int batch = std::max(1, options.angle_batch);
        const int total_slabs =
            (meta.compact_num_angles + batch - 1) / batch;
        const int cached_count = std::min(options.gpu_slab_cache, total_slabs);
        cached_slab_workers.resize((size_t)cached_count);
        for (int slab = 0; slab < cached_count; ++slab) {
            const int start_ck = slab * batch;
            const int end_ck = std::min(meta.compact_num_angles,
                                        start_ck + batch);
            const TimedHostCSR built = build_roi_csr(start_ck, end_ck);
            cached_slab_workers[(size_t)slab].init(
                0, built.csr, meta, p_all, projection_stride,
                (int)bed_count, g_full, track_castor_threshold);
            std::printf("[scheduler] cached angle slab [%d,%d), build=%.3fs\n",
                        start_ck, end_ck, built.build_sec);
        }
        cached_end_ck = std::min(meta.compact_num_angles, cached_count * batch);
    }

    CUDA_CHECK(cudaSetDevice(0));
    const int blocks_local = (int)((local_n + BLK - 1) / BLK);
    float* d_states = nullptr;
    float* d_backprojections = nullptr;
    float* d_sensitivity = nullptr;
    float* d_reduce_tmp = nullptr;
    float* d_effective_states = nullptr;
    float* d_conv_work = nullptr;
    float* d_conv_tmp = nullptr;
    float* d_psf_norm = nullptr;
    float* d_post_norm = nullptr;
    float* d_map_grad = nullptr;
    unsigned long long* d_nonpositive_osl_count = nullptr;
    OslDiagnostics* d_osl_diagnostics = nullptr;
    uint8_t* d_osl_roi_mask = nullptr;
    ButterworthWorkspace butter_workspace;
    CUDA_CHECK(cudaMalloc(&d_states, bed_count * (size_t)local_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_backprojections,
                          bed_count * (size_t)local_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sensitivity, (size_t)local_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_reduce_tmp, (size_t)local_n * sizeof(float)));

    const bool any_gaussian = use_psf || use_post || use_intra;
    if (any_gaussian) {
        CUDA_CHECK(cudaMalloc(&d_conv_work, (size_t)local_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv_tmp, (size_t)local_n * sizeof(float)));
    }
    if (use_psf) {
        CUDA_CHECK(cudaMalloc(&d_effective_states,
                              bed_count * (size_t)local_n * sizeof(float)));
        if (need_convolution_norm) {
            CUDA_CHECK(cudaMalloc(&d_psf_norm,
                                  (size_t)local_n * sizeof(float)));
        }
    }
    if (need_convolution_norm &&
        ((use_post && options.conv_mode != "sieve") || use_intra)) {
        CUDA_CHECK(cudaMalloc(&d_post_norm, (size_t)local_n * sizeof(float)));
    }
    if (options.algorithm == ALGORITHM_OSL) {
        CUDA_CHECK(cudaMalloc(&d_map_grad, (size_t)local_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nonpositive_osl_count,
                              sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_osl_diagnostics, sizeof(OslDiagnostics)));
        CUDA_CHECK(cudaMalloc(&d_osl_roi_mask,
                              (size_t)local_n * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(d_osl_roi_mask, roi_mask.data(),
                              (size_t)local_n * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
    }
    if (options.butter_enable) {
        initialize_butterworth_workspace(
            butter_workspace, meta.Nx, meta.Ny, meta.Nz, meta.voxel_size,
            options.butter_cutoff_cycles_per_cm, options.butter_order);
        const double workspace_mib =
            ((double)butter_workspace.padded_real_count * sizeof(float) +
             (double)butter_workspace.spectrum_count *
                 (sizeof(cufftComplex) + sizeof(float))) /
            (1024.0 * 1024.0);
        std::printf("[butterworth] cutoff=%.6g cycles/cm order=%d "
                    "padding=%dx%dx%d workspace=%.1f MiB\n",
                    options.butter_cutoff_cycles_per_cm,
                    options.butter_order,
                    butter_workspace.padded_nx,
                    butter_workspace.padded_ny,
                    butter_workspace.padded_nz,
                    workspace_mib);
    }

    std::vector<float> psf_kx, psf_ky, psf_kz;
    if (use_psf) {
        build_gaussian_kernel_set(
            options.conv_implementation, options.psf_fwhm_mm,
            meta.voxel_size, options.psf_nb_sigmas,
            psf_kx, psf_ky, psf_kz);
        prepare_convolution_normalization(
            options.conv_implementation, d_psf_norm,
            meta.Nx, meta.Ny, meta.Nz, psf_kx, psf_ky, psf_kz);
    }

    std::vector<float> post_kx, post_ky, post_kz;
    if (use_post) {
        if (options.conv_mode == "sieve") {
            // CASToR sieve 复用同一个 convolver：核、截断和边界归一化完全相同。
            post_kx = psf_kx;
            post_ky = psf_ky;
            post_kz = psf_kz;
        } else {
            const float post_fwhm[3] = {
                options.post_fwhm_mm, options.post_fwhm_mm,
                options.post_fwhm_mm};
            build_gaussian_kernel_set(
                options.conv_implementation, post_fwhm, meta.voxel_size,
                options.psf_nb_sigmas, post_kx, post_ky, post_kz);
            prepare_convolution_normalization(
                options.conv_implementation, d_post_norm,
                meta.Nx, meta.Ny, meta.Nz,
                post_kx, post_ky, post_kz);
        }
    }

    std::printf("[algorithm] algorithm=%s implementation=%s conv=%s/%s "
                "psf=%s post=%s intra=%s prior=%s beta=%.6g "
                "butterworth=%s\n",
                algorithm_name(options.algorithm),
                reference_implementation_name(options.implementation),
                options.conv_mode.c_str(),
                conv_implementation_name(options.conv_implementation),
                use_psf ? "on" : "off",
                use_post ? "on" : "off", use_intra ? "on" : "off",
                options.map_type == MAP_TV ? "tv" :
                (options.map_type == MAP_MRP ? "mrp" : "none"),
                options.map_beta, options.butter_enable ? "on" : "off");

    for (size_t b = 0; b < bed_count; ++b) {
        float* state = d_states + b * (size_t)local_n;
        k_fill<<<blocks_local, BLK>>>(state, 1.0f, local_n);
        if (options.roi_diameter_mm > 0.0) {
            k_apply_local_cylinder_roi<<<blocks_local, BLK>>>(
                state, meta.Nx, meta.Ny, meta.Nz, meta.voxel_size,
                (float)options.roi_diameter_mm, effective_roi_height_mm,
                0.0f, 0.0f, 0.0f, local_n);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto accumulate_worker_sensitivity = [&](GPUWorker& worker) {
        CUDA_CHECK(cudaSetDevice(worker.dev));
        const int blocks_rows = (worker.local_rows + BLK - 1) / BLK;
        const int blocks_image = (int)(((int64_t)worker.N2 + BLK - 1) / BLK);
        CUDA_CHECK(cudaMemset(worker.d_sens_part, 0,
                              (size_t)worker.N2 * sizeof(float)));
        for (int s = 0; s < sym_count; ++s) {
            for (int zm = 0; zm < z_count; ++zm) {
                k_make_unit_ratio_batched<<<blocks_rows, BLK>>>(
                    worker.d_ratio, worker.d_G_full, worker.local_rows,
                    worker.start_ck, worker.compact_rpa, worker.det_nx,
                    worker.compact_det_z, worker.det_nz, full_rpa,
                    (int)sym_mode, s, zm, meta.compact_num_angles,
                    meta.full_num_angles, 1);
                worker.spmv_backward();
                k_transform_bwd_add<<<blocks_image, BLK>>>(
                    worker.d_bp_rot, worker.d_sens_part,
                    worker.Nx, worker.Ny, worker.Nz, (int)sym_mode, s, zm);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    std::printf("[scheduler] sensitivity: one local A^T 1 shared by %zu beds\n",
                bed_count);
    if (!use_angle_batch) {
        std::vector<std::thread> sensitivity_threads;
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            sensitivity_threads.emplace_back([&, gpu]() {
                accumulate_worker_sensitivity(workers[(size_t)gpu]);
            });
        }
        for (std::thread& thread : sensitivity_threads) thread.join();
        reduce_parts_to_gpu0(d_sensitivity, d_reduce_tmp, workers,
                             &GPUWorker::d_sens_part, local_n);
    } else {
        CUDA_CHECK(cudaMemset(d_sensitivity, 0, (size_t)local_n * sizeof(float)));
        auto add_sensitivity_slab = [&](GPUWorker& worker) {
            accumulate_worker_sensitivity(worker);
            CUDA_CHECK(cudaSetDevice(0));
            k_add_inplace<<<blocks_local, BLK>>>(d_sensitivity,
                                                  worker.d_sens_part, local_n);
            CUDA_CHECK(cudaDeviceSynchronize());
        };
        for (GPUWorker& worker : cached_slab_workers) add_sensitivity_slab(worker);
        const int batch = std::max(1, options.angle_batch);
        for (int start_ck = cached_end_ck; start_ck < meta.compact_num_angles;
             start_ck += batch) {
            const int end_ck = std::min(meta.compact_num_angles, start_ck + batch);
            const TimedHostCSR built = build_roi_csr(start_ck, end_ck);
            GPUWorker worker;
            worker.init(0, built.csr, meta, p_all, projection_stride,
                        (int)bed_count, g_full, track_castor_threshold);
            add_sensitivity_slab(worker);
            worker.destroy();
        }
    }
    CUDA_CHECK(cudaSetDevice(0));
    if (use_psf) {
        // castor-adjoint 用真转置；OMEGA/严格 CASToR 按各自源码重用正向卷积。
        apply_convolution_backward(
            options.conv_implementation, d_sensitivity, d_conv_work,
            d_conv_tmp, d_psf_norm, meta.Nx, meta.Ny, meta.Nz,
            psf_kx, psf_ky, psf_kz);
        CUDA_CHECK(cudaMemcpy(d_sensitivity, d_conv_work,
                              (size_t)local_n * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }
    // 不在此处统一 clamp：OMEGA 在更新核内按 omega_epsilon 设下限，CASToR
    // 使用原始 sensitivity。统一 1e-12 会改变两套参考实现的零灵敏度语义。
    CUDA_CHECK(cudaDeviceSynchronize());

    auto run_worker_backprojection = [&](GPUWorker& worker, size_t bed_index) {
        CUDA_CHECK(cudaSetDevice(worker.dev));
        const int blocks_rows = (worker.local_rows + BLK - 1) / BLK;
        const int blocks_image = (int)(((int64_t)worker.N2 + BLK - 1) / BLK);
        CUDA_CHECK(cudaMemset(worker.d_bp_part, 0,
                              (size_t)worker.N2 * sizeof(float)));
        const float* projection = worker.d_P_all + bed_index * worker.proj_stride;
        for (int s = 0; s < sym_count; ++s) {
            for (int zm = 0; zm < z_count; ++zm) {
                k_transform_fwd<<<blocks_image, BLK>>>(
                    worker.d_img_in, worker.d_img_rot,
                    worker.Nx, worker.Ny, worker.Nz, (int)sym_mode, s, zm);
                worker.spmv_forward();
                k_calc_ratio_batched<<<blocks_rows, BLK>>>(
                    worker.d_y, worker.d_ratio, projection, worker.d_G_full,
                    worker.local_rows, worker.start_ck, worker.compact_rpa,
                    worker.det_nx, worker.compact_det_z, worker.det_nz,
                    full_rpa, (int)sym_mode, s, zm, meta.compact_num_angles,
                    meta.full_num_angles, 1, options.implementation,
                    options.omega_epsilon, options.castor_threshold,
                    worker.d_low_forward_count);
                worker.spmv_backward();
                k_transform_bwd_add<<<blocks_image, BLK>>>(
                    worker.d_bp_rot, worker.d_bp_part,
                    worker.Nx, worker.Ny, worker.Nz, (int)sym_mode, s, zm);
            }
        }
    };

    std::printf("[scheduler] independent local %s: order=iteration -> "
                "matrix/slab -> Bed1..Bed%zu\n",
                algorithm_name(options.algorithm), bed_count);
    std::vector<float> host_state((size_t)local_n);
    const float local_origin_z =
        -0.5f * (float)(meta.Nz - 1) * meta.voxel_size;
    const auto total_start = std::chrono::high_resolution_clock::now();
    for (int iteration = 1; iteration <= options.iterations; ++iteration) {
        const auto iteration_start = std::chrono::high_resolution_clock::now();
        unsigned long long streamed_low_forward_count = 0;
        for (GPUWorker& worker : workers) worker.reset_low_forward_count();
        for (GPUWorker& worker : cached_slab_workers) {
            worker.reset_low_forward_count();
        }
        CUDA_CHECK(cudaSetDevice(0));
        if (d_nonpositive_osl_count) {
            CUDA_CHECK(cudaMemset(d_nonpositive_osl_count, 0,
                                  sizeof(unsigned long long)));
            const OslDiagnostics empty_diagnostics = empty_osl_diagnostics();
            CUDA_CHECK(cudaMemcpy(d_osl_diagnostics, &empty_diagnostics,
                                  sizeof(empty_diagnostics),
                                  cudaMemcpyHostToDevice));
        }
        if (use_psf) {
            // 前投影使用 A_eff*x=A*(C*x)，所有床位只计算一次有效模糊状态。
            for (size_t b = 0; b < bed_count; ++b) {
                apply_convolution_forward(
                    options.conv_implementation,
                    d_states + b * (size_t)local_n,
                    d_effective_states + b * (size_t)local_n,
                    d_conv_tmp, d_psf_norm,
                    meta.Nx, meta.Ny, meta.Nz,
                    psf_kx, psf_ky, psf_kz);
            }
        }
        if (!use_angle_batch) {
            for (size_t b = 0; b < bed_count; ++b) {
                float* state = d_states + b * (size_t)local_n;
                const float* projected_state = use_psf
                    ? d_effective_states + b * (size_t)local_n
                    : state;
                broadcast_image_from_gpu0(projected_state, workers, local_n);
                std::vector<std::thread> worker_threads;
                for (int gpu = 0; gpu < num_gpus; ++gpu) {
                    worker_threads.emplace_back([&, gpu, b]() {
                        GPUWorker& worker = workers[(size_t)gpu];
                        run_worker_backprojection(worker, b);
                        CUDA_CHECK(cudaDeviceSynchronize());
                    });
                }
                for (std::thread& thread : worker_threads) thread.join();
                CUDA_CHECK(cudaSetDevice(0));
                reduce_parts_to_gpu0(d_backprojections + b * (size_t)local_n,
                                     d_reduce_tmp, workers,
                                     &GPUWorker::d_bp_part, local_n);
            }
        } else {
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaMemset(d_backprojections, 0,
                                  bed_count * (size_t)local_n * sizeof(float)));
            auto process_slab = [&](GPUWorker& worker) {
                for (size_t b = 0; b < bed_count; ++b) {
                    const float* projected_state = use_psf
                        ? d_effective_states + b * (size_t)local_n
                        : d_states + b * (size_t)local_n;
                    CUDA_CHECK(cudaMemcpy(worker.d_img_in,
                                          projected_state,
                                          (size_t)local_n * sizeof(float),
                                          cudaMemcpyDeviceToDevice));
                    run_worker_backprojection(worker, b);
                    CUDA_CHECK(cudaSetDevice(0));
                    k_add_inplace<<<blocks_local, BLK>>>(
                        d_backprojections + b * (size_t)local_n,
                        worker.d_bp_part, local_n);
                }
                CUDA_CHECK(cudaDeviceSynchronize());
            };
            for (GPUWorker& worker : cached_slab_workers) process_slab(worker);
            const int batch = std::max(1, options.angle_batch);
            for (int start_ck = cached_end_ck; start_ck < meta.compact_num_angles;
                 start_ck += batch) {
                const int end_ck = std::min(meta.compact_num_angles,
                                            start_ck + batch);
                const TimedHostCSR built = build_roi_csr(start_ck, end_ck);
                GPUWorker worker;
                worker.init(0, built.csr, meta, p_all, projection_stride,
                            (int)bed_count, g_full, track_castor_threshold);
                process_slab(worker);
                streamed_low_forward_count += worker.read_low_forward_count();
                worker.destroy();
            }
        }

        CUDA_CHECK(cudaSetDevice(0));
        const bool apply_intra = use_intra &&
            should_apply_smooth(iteration, options.iterations,
                                options.smooth_mode,
                                options.smooth_start_iter,
                                options.smooth_apply_every);
        std::vector<float> intra_kx, intra_ky, intra_kz;
        if (apply_intra) {
            const float fwhm = scheduled_smooth_fwhm(options, iteration);
            const float intra_fwhm[3] = {fwhm, fwhm, fwhm};
            build_gaussian_kernel_set(
                options.conv_implementation, intra_fwhm, meta.voxel_size,
                options.psf_nb_sigmas, intra_kx, intra_ky, intra_kz);
            prepare_convolution_normalization(
                options.conv_implementation, d_post_norm,
                meta.Nx, meta.Ny, meta.Nz,
                intra_kx, intra_ky, intra_kz);
        }
        const bool apply_butter = options.butter_enable &&
            should_apply_filter(iteration, options.iterations,
                                options.butter_mode,
                                options.butter_start_iter,
                                options.butter_period,
                                options.butter_last_n);

        for (size_t b = 0; b < bed_count; ++b) {
            float* state = d_states + b * (size_t)local_n;
            float* bp = d_backprojections + b * (size_t)local_n;

            if (use_psf) {
                apply_convolution_backward(
                    options.conv_implementation, bp, d_conv_work,
                    d_conv_tmp, d_psf_norm, meta.Nx, meta.Ny, meta.Nz,
                    psf_kx, psf_ky, psf_kz);
                CUDA_CHECK(cudaMemcpy(bp, d_conv_work,
                                      (size_t)local_n * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
            }

            const float* map_gradient = nullptr;
            if (options.algorithm == ALGORITHM_OSL &&
                options.map_type == MAP_TV) {
                k_tv_gradient_omega<<<blocks_local, BLK>>>(
                    state, d_map_grad, meta.Nx, meta.Ny, meta.Nz,
                    options.tv_smoothing);
                CUDA_CHECK(cudaGetLastError());
                map_gradient = d_map_grad;
            } else if (options.algorithm == ALGORITHM_OSL &&
                       options.map_type == MAP_MRP) {
                if (options.implementation == IMPLEMENTATION_OMEGA) {
                    k_mrp_gradient_omega<<<blocks_local, BLK>>>(
                        state, d_map_grad, meta.Nx, meta.Ny, meta.Nz,
                        options.mrp_radius[0], options.mrp_radius[1],
                        options.mrp_radius[2], options.mrp_epsilon,
                        options.mrp_no_norm);
                } else {
                    k_mrp_gradient_castor<<<blocks_local, BLK>>>(
                        state, d_map_grad, meta.Nx, meta.Ny, meta.Nz,
                        options.mrp_neighborhood, options.mrp_box_order,
                        options.mrp_box_exclude_corners,
                        options.mrp_sphere_radius_mm, meta.voxel_size,
                        (float)options.roi_diameter_mm,
                        effective_roi_height_mm);
                }
                CUDA_CHECK(cudaGetLastError());
                map_gradient = d_map_grad;
            }
            if (options.algorithm == ALGORITHM_OSL) {
                k_collect_osl_diagnostics<<<blocks_local, BLK>>>(
                    state, bp, d_sensitivity, map_gradient,
                    d_osl_roi_mask, options.map_beta,
                    options.omega_epsilon, options.implementation,
                    d_osl_diagnostics, local_n);
                CUDA_CHECK(cudaGetLastError());
            }
            k_em_update_versioned<<<blocks_local, BLK>>>(
                state, bp, d_sensitivity, map_gradient,
                options.algorithm, options.implementation,
                options.map_beta, options.omega_epsilon,
                options.castor_min_update, options.castor_max_update,
                d_nonpositive_osl_count, d_osl_diagnostics, local_n);
            CUDA_CHECK(cudaGetLastError());

            if (apply_intra) {
                apply_convolution_forward(
                    options.conv_implementation, state, d_conv_work,
                    d_conv_tmp, d_post_norm, meta.Nx, meta.Ny, meta.Nz,
                    intra_kx, intra_ky, intra_kz);
                CUDA_CHECK(cudaMemcpy(state, d_conv_work,
                                      (size_t)local_n * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
            }

            if (apply_butter) {
                apply_butterworth_filter(
                    butter_workspace, state, state,
                    iteration < options.iterations);
            }

            // 后处理可能把圆柱外体素重新填入非零值，因此最后重新施加支持域。
            k_apply_local_cylinder_roi<<<blocks_local, BLK>>>(
                state, meta.Nx, meta.Ny, meta.Nz, meta.voxel_size,
                (float)options.roi_diameter_mm, effective_roi_height_mm,
                0.0f, 0.0f, 0.0f, local_n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned long long low_forward_count = streamed_low_forward_count;
        for (const GPUWorker& worker : workers) {
            low_forward_count += worker.read_low_forward_count();
        }
        for (const GPUWorker& worker : cached_slab_workers) {
            low_forward_count += worker.read_low_forward_count();
        }
        CUDA_CHECK(cudaSetDevice(0));
        unsigned long long nonpositive_osl_count = 0;
        OslDiagnostics osl_diagnostics = empty_osl_diagnostics();
        if (d_nonpositive_osl_count) {
            CUDA_CHECK(cudaMemcpy(&nonpositive_osl_count,
                                  d_nonpositive_osl_count,
                                  sizeof(nonpositive_osl_count),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&osl_diagnostics, d_osl_diagnostics,
                                  sizeof(osl_diagnostics),
                                  cudaMemcpyDeviceToHost));
        }
        if (low_forward_count > 0) {
            std::fprintf(stderr,
                         "[warning] iteration %d: %llu CASToR data-space evaluations had g*A*x <= %.9g; neutral ratio=g was used.\n",
                         iteration, low_forward_count,
                         (double)options.castor_threshold);
        }
        if (nonpositive_osl_count > 0) {
            const unsigned long long active =
                osl_diagnostics.counts[OSL_DIAG_ACTIVE_ROI];
            const double abnormal_percent = active > 0
                ? 100.0 * (double)nonpositive_osl_count / (double)active
                : 0.0;
            std::fprintf(stderr,
                         "[osl-diagnostic] iteration=%d implementation=%s "
                         "active_roi_voxels=%llu abnormal_updates=%llu (%.6f%%)\n",
                         iteration,
                         reference_implementation_name(options.implementation),
                         active, nonpositive_osl_count, abnormal_percent);
            std::fprintf(stderr,
                         "[osl-diagnostic] counts old_nonfinite=%llu old_nonpositive=%llu "
                         "bp_nonfinite=%llu bp_negative=%llu sens_nonfinite=%llu "
                         "sens_nonpositive=%llu grad_nonfinite=%llu penalty_nonfinite=%llu\n",
                         osl_diagnostics.counts[OSL_DIAG_OLD_NONFINITE],
                         osl_diagnostics.counts[OSL_DIAG_OLD_NONPOSITIVE],
                         osl_diagnostics.counts[OSL_DIAG_BP_NONFINITE],
                         osl_diagnostics.counts[OSL_DIAG_BP_NEGATIVE],
                         osl_diagnostics.counts[OSL_DIAG_SENS_NONFINITE],
                         osl_diagnostics.counts[OSL_DIAG_SENS_NONPOSITIVE],
                         osl_diagnostics.counts[OSL_DIAG_GRAD_NONFINITE],
                         osl_diagnostics.counts[OSL_DIAG_PENALTY_NONFINITE]);
            std::fprintf(stderr,
                         "[osl-diagnostic] denominator_counts raw_nonfinite=%llu "
                         "raw_nonpositive=%llu effective_nonfinite=%llu "
                         "effective_nonpositive=%llu\n",
                         osl_diagnostics.counts[OSL_DIAG_RAW_DENOM_NONFINITE],
                         osl_diagnostics.counts[OSL_DIAG_RAW_DENOM_NONPOSITIVE],
                         osl_diagnostics.counts[OSL_DIAG_EFFECTIVE_DENOM_NONFINITE],
                         osl_diagnostics.counts[OSL_DIAG_EFFECTIVE_DENOM_NONPOSITIVE]);
            std::fprintf(stderr,
                         "[osl-diagnostic] ranges sensitivity=[%.9g,%.9g] "
                         "penalty_beta_grad=[%.9g,%.9g] raw_denominator=[%.9g,%.9g] "
                         "effective_denominator=[%.9g,%.9g]\n",
                         (double)osl_diagnostics.minima[OSL_RANGE_SENSITIVITY],
                         (double)osl_diagnostics.maxima[OSL_RANGE_SENSITIVITY],
                         (double)osl_diagnostics.minima[OSL_RANGE_PENALTY],
                         (double)osl_diagnostics.maxima[OSL_RANGE_PENALTY],
                         (double)osl_diagnostics.minima[OSL_RANGE_RAW_DENOMINATOR],
                         (double)osl_diagnostics.maxima[OSL_RANGE_RAW_DENOMINATOR],
                         (double)osl_diagnostics.minima[OSL_RANGE_EFFECTIVE_DENOMINATOR],
                         (double)osl_diagnostics.maxima[OSL_RANGE_EFFECTIVE_DENOMINATOR]);
            std::fprintf(stderr,
                         "[osl-diagnostic] beta current=%.9g estimated_safe_upper=%.9g "
                         "max_negative_penalty_fraction=%.9g actions omega_epsilon_floor=%llu "
                         "castor_keep_old=%llu omega_epsilon=%.9g\n",
                         (double)options.map_beta,
                         (double)osl_diagnostics.minima[OSL_RANGE_BETA_LIMIT],
                         (double)osl_diagnostics.maxima[
                             OSL_RANGE_NEGATIVE_PENALTY_FRACTION],
                         osl_diagnostics.counts[OSL_DIAG_OMEGA_EPSILON_FLOOR],
                         osl_diagnostics.counts[OSL_DIAG_CASTOR_KEEP_OLD],
                         (double)options.omega_epsilon);
            std::fprintf(stderr,
                         "[warning] iteration %d: %llu OSL voxels had sensitivity+penalty <=0 or non-finite inputs. implementation=%s.\n",
                         iteration, nonpositive_osl_count,
                         reference_implementation_name(options.implementation));
            std::fflush(stderr);
        }

        const bool save_iteration = iteration % options.save_interval == 0;
        if (save_iteration) {
            const std::string stem = "recon" + std::to_string(iteration);
            for (size_t b = 0; b < bed_count; ++b) {
                const float* saved_state = d_states + b * (size_t)local_n;
                if (use_post) {
                    // post 只处理保存副本；sieve 复用 PSF 的核和归一化。
                    const float* post_norm = options.conv_mode == "sieve"
                        ? d_psf_norm : d_post_norm;
                    apply_convolution_forward(
                        options.conv_implementation,
                        saved_state, d_conv_work, d_conv_tmp, post_norm,
                        meta.Nx, meta.Ny, meta.Nz,
                        post_kx, post_ky, post_kz);
                    k_apply_local_cylinder_roi<<<blocks_local, BLK>>>(
                        d_conv_work, meta.Nx, meta.Ny, meta.Nz,
                        meta.voxel_size,
                        (float)options.roi_diameter_mm,
                        effective_roi_height_mm,
                        0.0f, 0.0f, 0.0f, local_n);
                    CUDA_CHECK(cudaGetLastError());
                    saved_state = d_conv_work;
                }
                CUDA_CHECK(cudaMemcpy(host_state.data(), saved_state,
                                      (size_t)local_n * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                const fs::path bed_dir = raw_root /
                    ("Bed" + std::to_string(beds[b].number));
                make_parent_dirs(bed_dir);
                save_volume_outputs((bed_dir / stem).string(),
                                    host_state.data(),
                                    meta.Nx, meta.Ny, meta.Nz,
                                    meta.voxel_size, local_origin_z, false);
            }
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - iteration_start).count();
        std::printf("  Iter %3d/%d | %s%s | %.3fs\n", iteration,
                    options.iterations,
                    use_angle_batch ? "slabs shared across beds" : "resident CSR",
                    save_iteration ? " | saved" : "",
                    seconds);
    }
    const double total_seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - total_start).count();
    std::printf("[scheduler] complete: %.2f min, %.3f s/iteration\n",
                total_seconds / 60.0,
                total_seconds / std::max(1, options.iterations));

    CUDA_CHECK(cudaSetDevice(0));
    if (options.butter_enable) {
        destroy_butterworth_workspace(butter_workspace);
    }
    if (d_nonpositive_osl_count) {
        CUDA_CHECK(cudaFree(d_nonpositive_osl_count));
    }
    if (d_osl_diagnostics) CUDA_CHECK(cudaFree(d_osl_diagnostics));
    if (d_osl_roi_mask) CUDA_CHECK(cudaFree(d_osl_roi_mask));
    if (d_map_grad) CUDA_CHECK(cudaFree(d_map_grad));
    if (d_post_norm) CUDA_CHECK(cudaFree(d_post_norm));
    if (d_psf_norm) CUDA_CHECK(cudaFree(d_psf_norm));
    if (d_conv_tmp) CUDA_CHECK(cudaFree(d_conv_tmp));
    if (d_conv_work) CUDA_CHECK(cudaFree(d_conv_work));
    if (d_effective_states) CUDA_CHECK(cudaFree(d_effective_states));
    CUDA_CHECK(cudaFree(d_reduce_tmp));
    CUDA_CHECK(cudaFree(d_sensitivity));
    CUDA_CHECK(cudaFree(d_backprojections));
    CUDA_CHECK(cudaFree(d_states));
    for (GPUWorker& worker : cached_slab_workers) worker.destroy();
    for (GPUWorker& worker : workers) worker.destroy();
    return 0;
}

// 在 x/y/z 三个方向进行中心裁剪；原点同步移动，保持裁剪后物理坐标不变。
static Volume crop_center_mask(const Volume& source, int mask_nx,
                               int mask_ny, int mask_nz) {
    if (mask_nx <= 0 || mask_ny <= 0 || mask_nz <= 0 ||
        mask_nx > source.nx || mask_ny > source.ny || mask_nz > source.nz) {
        fail("mask size must be positive and no larger than reconstructed geometry");
    }
    const int x0 = (source.nx - mask_nx) / 2;
    const int y0 = (source.ny - mask_ny) / 2;
    const int z0 = (source.nz - mask_nz) / 2;
    Volume result = source;
    result.nx = mask_nx;
    result.ny = mask_ny;
    result.nz = mask_nz;
    result.origin_x_mm += x0 * source.voxel_mm;
    result.origin_y_mm += y0 * source.voxel_mm;
    result.origin_z_mm += z0 * source.voxel_mm;
    result.data.assign(static_cast<size_t>(mask_nx) * mask_ny * mask_nz, 0.0);
    const size_t source_plane = static_cast<size_t>(source.nx) * source.ny;
    const size_t result_plane = static_cast<size_t>(mask_nx) * mask_ny;
    for (int z = 0; z < mask_nz; ++z) {
        for (int y = 0; y < mask_ny; ++y) {
            const size_t source_offset = static_cast<size_t>(z + z0) * source_plane +
                                         static_cast<size_t>(y + y0) * source.nx + x0;
            const size_t result_offset = static_cast<size_t>(z) * result_plane +
                                         static_cast<size_t>(y) * mask_nx;
            std::copy(source.data.begin() + source_offset,
                      source.data.begin() + source_offset + mask_nx,
                      result.data.begin() + result_offset);
        }
    }
    return result;
}

static void bedmerge_put_le16(unsigned char* p, uint16_t value) {
    p[0] = static_cast<unsigned char>(value & 255u);
    p[1] = static_cast<unsigned char>((value >> 8) & 255u);
}

static void bedmerge_put_le32(unsigned char* p, uint32_t value) {
    p[0] = static_cast<unsigned char>(value & 255u);
    p[1] = static_cast<unsigned char>((value >> 8) & 255u);
    p[2] = static_cast<unsigned char>((value >> 16) & 255u);
    p[3] = static_cast<unsigned char>((value >> 24) & 255u);
}

static void bedmerge_put_le_float(unsigned char* p, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bedmerge_put_le32(p, bits);
}

// 写出 NIfTI-1 成对 hdr/img；数据为 float64，x 为最快变化维。
static void write_nifti(const fs::path& directory, const Volume& volume,
                        const char* stem = "recon_stitched_minimal") {
    make_parent_dirs(directory);
    unsigned char header[348] = {};
    bedmerge_put_le32(header, 348u);
    header[38] = 'r';
    bedmerge_put_le16(header + 40, 3);
    bedmerge_put_le16(header + 42, static_cast<uint16_t>(volume.nx));
    bedmerge_put_le16(header + 44, static_cast<uint16_t>(volume.ny));
    bedmerge_put_le16(header + 46, static_cast<uint16_t>(volume.nz));
    bedmerge_put_le16(header + 48, 1);
    bedmerge_put_le16(header + 70, 64);
    bedmerge_put_le16(header + 72, 64);
    bedmerge_put_le_float(header + 76, 1.0f);
    bedmerge_put_le_float(header + 80, static_cast<float>(volume.voxel_mm));
    bedmerge_put_le_float(header + 84, static_cast<float>(volume.voxel_mm));
    bedmerge_put_le_float(header + 88, static_cast<float>(volume.voxel_mm));
    bedmerge_put_le_float(header + 92, 1.0f);
    bedmerge_put_le_float(header + 112, 1.0f);
    header[123] = 2;
    const float min_value = static_cast<float>(*std::min_element(volume.data.begin(), volume.data.end()));
    const float max_value = static_cast<float>(*std::max_element(volume.data.begin(), volume.data.end()));
    bedmerge_put_le_float(header + 124, max_value);
    bedmerge_put_le_float(header + 128, min_value);
    const char* description = "SPECT bedmerge cropped cosine stitch; x-fastest";
    std::memcpy(header + 148, description, std::min<size_t>(79, std::strlen(description)));
    bedmerge_put_le16(header + 254, 1);
    bedmerge_put_le_float(header + 280, static_cast<float>(volume.voxel_mm));
    bedmerge_put_le_float(header + 292, static_cast<float>(volume.origin_x_mm));
    bedmerge_put_le_float(header + 300, static_cast<float>(volume.voxel_mm));
    bedmerge_put_le_float(header + 308, static_cast<float>(volume.origin_y_mm));
    bedmerge_put_le_float(header + 320, static_cast<float>(volume.voxel_mm));
    bedmerge_put_le_float(header + 324, static_cast<float>(volume.origin_z_mm));
    std::memcpy(header + 344, "ni1\0", 4);
    const std::string header_name = std::string(stem) + ".hdr";
    const std::string image_name = std::string(stem) + ".img";
    std::ofstream header_file(directory / header_name, std::ios::binary);
    header_file.write(reinterpret_cast<const char*>(header), sizeof(header));
    std::ofstream image_file(directory / image_name, std::ios::binary);
    image_file.write(reinterpret_cast<const char*>(volume.data.data()),
                     static_cast<std::streamsize>(volume.data.size() * sizeof(double)));
    if (!header_file || !image_file) fail("cannot write stitched NIfTI output");
}

static double percentile99(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const size_t index = static_cast<size_t>(0.99 * static_cast<double>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

static double median(std::vector<double> values) {
    if (values.empty()) fail("empty median sample");
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    double result = values[middle];
    if ((values.size() & 1u) == 0u) {
        std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
        result = 0.5 * (result + values[middle - 1]);
    }
    return result;
}

// 在相邻床位的 z 重叠层内估计 log(a/b) 的稳健中位数，作为相对增益。
static double pair_log_ratio(const Volume& low, const Volume& high,
                             int low_start, int high_start, int layers,
                             double voxel_mm, double roi_diameter_mm,
                             int& sample_count) {
    std::vector<double> joint;
    const double radius2 = 0.25 * roi_diameter_mm * roi_diameter_mm;
    const int plane = low.nx * low.ny;
    for (int z = 0; z < layers; ++z) {
        for (int y = 0; y < low.ny; ++y) {
            const double yy = (static_cast<double>(y) - 0.5 * (low.ny - 1)) * voxel_mm;
            for (int x = 0; x < low.nx; ++x) {
                const double xx = (static_cast<double>(x) - 0.5 * (low.nx - 1)) * voxel_mm;
                if (xx * xx + yy * yy > radius2) continue;
                const double a = low.data[static_cast<size_t>(low_start + z) * plane + y * low.nx + x];
                const double b = high.data[static_cast<size_t>(high_start + z) * plane + y * high.nx + x];
                if (std::isfinite(a) && std::isfinite(b) && std::max(a, b) > 0.0) {
                    joint.push_back(std::max(a, b));
                }
            }
        }
    }
    if (joint.empty()) fail("adjacent bed overlap has no positive signal");
    const double threshold = 0.10 * percentile99(joint);
    std::vector<double> values;
    for (int z = 0; z < layers; ++z) {
        for (int y = 0; y < low.ny; ++y) {
            const double yy = (static_cast<double>(y) - 0.5 * (low.ny - 1)) * voxel_mm;
            for (int x = 0; x < low.nx; ++x) {
                const double xx = (static_cast<double>(x) - 0.5 * (low.nx - 1)) * voxel_mm;
                if (xx * xx + yy * yy > radius2) continue;
                const double a = low.data[static_cast<size_t>(low_start + z) * plane + y * low.nx + x];
                const double b = high.data[static_cast<size_t>(high_start + z) * plane + y * high.nx + x];
                if (a > threshold && b > threshold && std::isfinite(a) && std::isfinite(b)) {
                    values.push_back(std::log(a) - std::log(b));
                }
            }
        }
    }
    if (values.size() < 100) {
        values.clear();
        for (int z = 0; z < layers; ++z) {
            for (int y = 0; y < low.ny; ++y) {
                for (int x = 0; x < low.nx; ++x) {
                    const double a = low.data[static_cast<size_t>(low_start + z) * plane + y * low.nx + x];
                    const double b = high.data[static_cast<size_t>(high_start + z) * plane + y * high.nx + x];
                    if (a > 0.0 && b > 0.0 && std::isfinite(a) && std::isfinite(b)) {
                        values.push_back(std::log(a) - std::log(b));
                    }
                }
            }
        }
    }
    if (values.empty()) fail("adjacent bed overlap has no common positive voxels");
    double center = median(values);
    std::vector<double> deviations(values.size());
    for (size_t i = 0; i < values.size(); ++i) deviations[i] = std::abs(values[i] - center);
    const double mad = median(deviations);
    if (mad > 0.0) {
        const double limit = 3.5 * 1.4826 * mad;
        std::vector<double> filtered;
        for (double value : values) if (std::abs(value - center) <= limit) filtered.push_back(value);
        if (!filtered.empty()) center = median(filtered);
        sample_count = static_cast<int>(filtered.size());
    } else {
        sample_count = static_cast<int>(values.size());
    }
    return center;
}

// 从后向前累计相邻床位增益，使重叠区的平均强度连续。
static std::vector<double> fit_gains(const std::vector<Volume>& volumes,
                                     int step_layers,
                                     double voxel_mm, double roi_diameter_mm) {
    const int count = static_cast<int>(volumes.size());
    if (!volumes.empty() && volumes[0].nz == step_layers) {
        std::printf("gain fitting skipped: mask Nz equals bed step; "
                    "adjacent beds do not overlap, using gain=1 for all beds\n");
        return std::vector<double>((size_t)count, 1.0);
    }
    std::vector<double> log_gain(count, 0.0);
    for (int low_number = count; low_number >= 2; --low_number) {
        const int low = low_number - 1;
        const int high = low_number - 2;
        // volume 已完成 mask 裁剪；重叠区为低床位 [step,Nz) 与高床位 [0,Nz-step)。
        const int overlap_start_low = step_layers;
        const int overlap_start_high = 0;
        const int overlap_layers = volumes[low].nz - step_layers;
        int samples = 0;
        const double ratio = pair_log_ratio(
            volumes[low], volumes[high], overlap_start_low,
            overlap_start_high, overlap_layers, voxel_mm, roi_diameter_mm, samples);
        log_gain[high] = log_gain[low] + ratio;
        std::printf("gain pair Bed%d/Bed%d: log_ratio=%.8g samples=%d\n",
                    low_number, low_number - 1, ratio, samples);
    }
    const double anchor = log_gain[0];
    std::vector<double> gains(count);
    for (int i = 0; i < count; ++i) gains[i] = std::exp(log_gain[i] - anchor);
    return gains;
}

// 将 mask 体积按床位步进放入全局 z 网格，并用 cos^2/sin^2 权重归一化重叠区。
static Volume stitch_volumes(const std::vector<Volume>& volumes,
                             const std::vector<double>& gains,
                             double bed_step_mm) {
    const int count = static_cast<int>(volumes.size());
    // 轴向保留层数直接使用 mask Nz，不再依赖 crop-length-mm。
    const int crop_layers = volumes[0].nz;
    const int step_layers = static_cast<int>(std::llround(bed_step_mm / volumes[0].voxel_mm));
    const int output_nz = crop_layers + (count - 1) * step_layers;
    const int plane = volumes[0].nx * volumes[0].ny;
    Volume output = volumes[0];
    output.nz = output_nz;
    output.origin_z_mm = std::numeric_limits<double>::infinity();
    output.data.assign(static_cast<size_t>(output_nz) * plane, 0.0);
    std::vector<double> weight(static_cast<size_t>(output_nz), 0.0);
    for (int bed = 0; bed < count; ++bed) {
        const int shift = (count - bed - 1) * step_layers;
        output.origin_z_mm = std::min(
            output.origin_z_mm,
            ((count - 1) * 0.5 - bed) * bed_step_mm +
            volumes[bed].origin_z_mm);
        for (int z = 0; z < crop_layers; ++z) {
            const int global_z = shift + z;
            double w = 1.0;
            if (bed > 0 && z >= step_layers) {
                const double u = static_cast<double>(z - step_layers) /
                                 (crop_layers - step_layers - 1);
                constexpr double pi = 3.14159265358979323846;
                w = std::sin(0.5 * pi * u) * std::sin(0.5 * pi * u);
            }
            if (bed + 1 < count && z < crop_layers - step_layers) {
                const double u = static_cast<double>(z) /
                                 (crop_layers - step_layers - 1);
                constexpr double pi = 3.14159265358979323846;
                w = std::cos(0.5 * pi * u) * std::cos(0.5 * pi * u);
            }
            weight[global_z] += w;
            for (int i = 0; i < plane; ++i) {
                output.data[static_cast<size_t>(global_z) * plane + i] +=
                    volumes[bed].data[static_cast<size_t>(z) * plane + i] * gains[bed] * w;
            }
        }
    }
    for (int z = 0; z < output_nz; ++z) {
        if (!(weight[z] > 0.0)) fail("stitched output contains an uncovered axial layer");
        for (int i = 0; i < plane; ++i) {
            output.data[static_cast<size_t>(z) * plane + i] /= weight[z];
        }
    }
    return output;
}

struct VolumeDiagnostics {
    double minimum = 0.0;
    double maximum = 0.0;
    double sum = 0.0;
    uint64_t negative_voxels = 0;
};

static VolumeDiagnostics diagnose_volume(const Volume& volume) {
    if (volume.data.empty()) fail("cannot diagnose an empty volume");
    VolumeDiagnostics result;
    result.minimum = std::numeric_limits<double>::infinity();
    result.maximum = -std::numeric_limits<double>::infinity();
    for (double value : volume.data) {
        if (!std::isfinite(value)) fail("filtered stitched volume contains NaN/Inf");
        result.minimum = std::min(result.minimum, value);
        result.maximum = std::max(result.maximum, value);
        result.sum += value;
        if (value < 0.0) ++result.negative_voxels;
    }
    return result;
}

static void print_stitch_filter_diagnostics(
    const char* name, const VolumeDiagnostics& before,
    const VolumeDiagnostics& after) {
    const double relative_sum_change = before.sum != 0.0
        ? (after.sum - before.sum) / before.sum : 0.0;
    std::printf(
        "[stitch-filter] %s: min %.9g -> %.9g, max %.9g -> %.9g, "
        "negative %llu -> %llu, sum %.9g -> %.9g (relative %.9g)\n",
        name, before.minimum, after.minimum, before.maximum, after.maximum,
        static_cast<unsigned long long>(before.negative_voxels),
        static_cast<unsigned long long>(after.negative_voxels),
        before.sum, after.sum, relative_sum_change);
}

static std::string decimal_path_token(double value) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6) << value;
    std::string token = stream.str();
    while (!token.empty() && token.back() == '0') token.pop_back();
    if (!token.empty() && token.back() == '.') token.pop_back();
    std::replace(token.begin(), token.end(), '.', 'p');
    return token;
}

// 拼接后 Gaussian 使用 CASToR 的 erf 体素积分核和有限边界前向语义。
// castor-adjoint 与 strict CASToR 的前向完全相同；这里没有反向步骤。输入/输出
// 转成 float32 以复用重建阶段已经验证的 CUDA kernel，NIfTI 仍按 float64 写出。
static Volume smooth_stitched_volume(const Volume& input, float fwhm_mm,
                                     float nb_sigmas) {
    const int64_t count64 = (int64_t)input.nx * input.ny * input.nz;
    if (count64 <= 0 || (uint64_t)count64 != input.data.size()) {
        fail("stitched Gaussian input geometry/data size mismatch");
    }
    std::vector<float> host((size_t)count64);
    for (size_t i = 0; i < host.size(); ++i) {
        if (!std::isfinite(input.data[i]) ||
            std::abs(input.data[i]) > std::numeric_limits<float>::max()) {
            fail("stitched Gaussian input cannot be represented as float32");
        }
        host[i] = static_cast<float>(input.data[i]);
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_tmp = nullptr;
    float* d_norm = nullptr;
    auto cleanup = [&]() {
        if (d_norm) CUDA_CHECK(cudaFree(d_norm));
        if (d_tmp) CUDA_CHECK(cudaFree(d_tmp));
        if (d_output) CUDA_CHECK(cudaFree(d_output));
        if (d_input) CUDA_CHECK(cudaFree(d_input));
    };
    try {
        const size_t bytes = host.size() * sizeof(float);
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMalloc(&d_input, bytes));
        CUDA_CHECK(cudaMalloc(&d_output, bytes));
        CUDA_CHECK(cudaMalloc(&d_tmp, bytes));
        CUDA_CHECK(cudaMalloc(&d_norm, bytes));
        CUDA_CHECK(cudaMemcpy(d_input, host.data(), bytes,
                              cudaMemcpyHostToDevice));
        const float fwhm[3] = {fwhm_mm, fwhm_mm, fwhm_mm};
        std::vector<float> kx, ky, kz;
        build_gaussian_kernel_set(CONV_IMPLEMENTATION_CASTOR_ADJOINT,
                                  fwhm, (float)input.voxel_mm, nb_sigmas,
                                  kx, ky, kz);
        prepare_convolution_normalization(
            CONV_IMPLEMENTATION_CASTOR_ADJOINT, d_norm,
            input.nx, input.ny, input.nz, kx, ky, kz);
        apply_convolution_forward(
            CONV_IMPLEMENTATION_CASTOR_ADJOINT, d_input, d_output, d_tmp,
            d_norm, input.nx, input.ny, input.nz, kx, ky, kz);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host.data(), d_output, bytes,
                              cudaMemcpyDeviceToHost));
        cleanup();
    } catch (...) {
        cleanup();
        throw;
    }

    Volume output = input;
    for (size_t i = 0; i < host.size(); ++i) {
        if (!std::isfinite(host[i])) fail("stitched Gaussian produced NaN/Inf");
        output.data[i] = host[i];
    }
    return output;
}

// 拼接后 Butterworth 复用修正版 2x 零填充实现。最后一次后处理保持线性，
// 因而不会把可能存在的负旁瓣静默截断为 0。
static Volume butterworth_stitched_volume(const Volume& input,
                                           float cutoff_cycles_per_cm,
                                           int order) {
    const int64_t count64 = (int64_t)input.nx * input.ny * input.nz;
    if (count64 <= 0 || (uint64_t)count64 != input.data.size()) {
        fail("stitched Butterworth input geometry/data size mismatch");
    }
    std::vector<float> host((size_t)count64);
    for (size_t i = 0; i < host.size(); ++i) {
        if (!std::isfinite(input.data[i]) ||
            std::abs(input.data[i]) > std::numeric_limits<float>::max()) {
            fail("stitched Butterworth input cannot be represented as float32");
        }
        host[i] = static_cast<float>(input.data[i]);
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    ButterworthWorkspace workspace;
    bool workspace_initialized = false;
    auto cleanup = [&]() {
        if (workspace_initialized) destroy_butterworth_workspace(workspace);
        if (d_output) CUDA_CHECK(cudaFree(d_output));
        if (d_input) CUDA_CHECK(cudaFree(d_input));
    };
    try {
        const size_t bytes = host.size() * sizeof(float);
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMalloc(&d_input, bytes));
        CUDA_CHECK(cudaMalloc(&d_output, bytes));
        CUDA_CHECK(cudaMemcpy(d_input, host.data(), bytes,
                              cudaMemcpyHostToDevice));
        initialize_butterworth_workspace(
            workspace, input.nx, input.ny, input.nz, (float)input.voxel_mm,
            cutoff_cycles_per_cm, order);
        workspace_initialized = true;
        apply_butterworth_filter(workspace, d_input, d_output, false);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host.data(), d_output, bytes,
                              cudaMemcpyDeviceToHost));
        cleanup();
    } catch (...) {
        cleanup();
        throw;
    }

    Volume output = input;
    for (size_t i = 0; i < host.size(); ++i) {
        if (!std::isfinite(host[i])) fail("stitched Butterworth produced NaN/Inf");
        output.data[i] = host[i];
    }
    return output;
}

static std::string json_path(const fs::path& path) {
    std::string text = path.generic_string();
    std::string escaped;
    for (char c : text) {
        if (c == '\\') escaped += "\\\\";
        else if (c == '"') escaped += "\\\"";
        else escaped += c;
    }
    return escaped;
}

static void write_report(const fs::path& path, const BedMergeOptions& options,
                         const std::vector<BedInput>& beds,
                         const std::vector<double>& gains,
                         const std::vector<int>& saved_iterations,
                         const Volume& output, int mask_layers, int step_layers) {
    std::ofstream report(path);
    report << "{\n"
           << "  \"method\": \"versioned_independent_local_mlem_or_osl_then_cosine_stitch\",\n"
           << "  \"projection_directory\": \"" << json_path(options.proj_dir) << "\",\n"
           << "  \"matrix_directory\": \"" << json_path(options.matrix_dir) << "\",\n"
           << "  \"scheduler\": {\"image_states\": \"independent_local_per_bed\", "
           << "\"matrix_traversal\": \"iteration_major\", "
           << "\"angle_batch\": " << options.angle_batch << ", "
           << "\"gpu_slab_cache\": " << options.gpu_slab_cache << "},\n"
           << "  \"reconstruction\": {\"iterations\": " << options.iterations
           << ", \"algorithm\": \"" << algorithm_name(options.algorithm)
           << "\", \"implementation\": \""
           << reference_implementation_name(options.implementation)
           << "\", \"omega_epsilon\": " << options.omega_epsilon
           << ", \"castor_threshold\": " << options.castor_threshold
           << ", \"castor_min_update\": " << options.castor_min_update
           << ", \"castor_max_update\": " << options.castor_max_update
           // 以下旧字段保留给已有报告读取脚本，值与新统一字段完全一致。
           << ", \"osl_version\": \""
           << reference_implementation_name(options.implementation)
           << "\", \"osl_epsilon\": " << options.omega_epsilon
           << ", \"osl_min_update\": " << options.castor_min_update
           << ", \"osl_max_update\": " << options.castor_max_update
           << ", \"forward_threshold\": " << options.castor_threshold
           << ", \"conv_mode\": \"" << options.conv_mode
           << "\", \"conv_impl\": \""
           << conv_implementation_name(options.conv_implementation)
           << "\", \"psf_fwhm_mm\": [" << options.psf_fwhm_mm[0] << ", "
           << options.psf_fwhm_mm[1] << ", " << options.psf_fwhm_mm[2]
           << "], \"psf_sigmas\": " << options.psf_nb_sigmas
           << ", \"post_fwhm_mm\": " << options.post_fwhm_mm
           << ", \"smooth_mode\": \"" << options.smooth_mode
           << "\", \"smooth_schedule\": \"" << options.smooth_schedule
           << "\", \"smooth_fwhm_start_mm\": " << options.smooth_fwhm_start_mm
           << ", \"smooth_fwhm_end_mm\": " << options.smooth_fwhm_end_mm
           << ", \"smooth_start_iter\": " << options.smooth_start_iter
           << ", \"smooth_every\": " << options.smooth_apply_every
           << ", \"sieve_reuses_psf_convolver\": "
           << (options.conv_mode == "sieve" ? "true" : "false")
           << ", \"prior\": \""
           << (options.map_type == MAP_TV ? "tv" :
               (options.map_type == MAP_MRP ? "mrp" : "none"))
           << "\", \"map_beta\": " << options.map_beta
           << ", \"tv_smoothing\": " << options.tv_smoothing
           << ", \"prior_implementation\": \""
           << reference_implementation_name(options.implementation)
           << "\", \"mrp_version\": \""
           << reference_implementation_name(options.implementation)
           << "\", \"mrp_radius_xyz\": [" << options.mrp_radius[0] << ", "
           << options.mrp_radius[1] << ", " << options.mrp_radius[2]
           << "], \"mrp_epsilon\": " << options.mrp_epsilon
           << ", \"mrp_no_norm\": " << (options.mrp_no_norm ? "true" : "false")
           << ", \"mrp_neighborhood\": \""
           << mrp_neighborhood_name(options.mrp_neighborhood)
           << "\", \"mrp_box_order\": " << options.mrp_box_order
           << ", \"mrp_box_exclude_corners\": "
           << (options.mrp_box_exclude_corners ? "true" : "false")
           << ", \"mrp_sphere_radius_mm\": "
           << options.mrp_sphere_radius_mm
           << ", \"butterworth_enabled\": "
           << (options.butter_enable ? "true" : "false")
           << ", \"butterworth_cutoff_cycles_per_cm\": "
           << options.butter_cutoff_cycles_per_cm
           << ", \"butterworth_order\": " << options.butter_order
           << ", \"butterworth_mode\": \"" << options.butter_mode
           << "\", \"butterworth_start_iter\": "
           << options.butter_start_iter
           << ", \"butterworth_period\": " << options.butter_period
           << ", \"butterworth_last_n\": " << options.butter_last_n
           << ", \"stitch_filter\": \"" << options.stitch_filter
           << "\", \"stitch_smooth_fwhm_mm\": "
           << options.stitch_smooth_fwhm_mm
           << ", \"stitch_smooth_sigmas\": "
           << options.stitch_smooth_sigmas
           << ", \"stitch_butterworth_cutoff_cycles_per_cm\": "
           << options.stitch_butter_cutoff_cycles_per_cm
           << ", \"stitch_butterworth_order\": "
           << options.stitch_butter_order
           << ", \"stitch_butterworth_variant_count\": "
           << options.stitch_butter_variants.size()
           << ", \"save_interval\": " << options.save_interval << "},\n"
           << "  \"roi\": {\"shape\": \"centered_cylinder\", "
           << "\"diameter_mm\": " << options.roi_diameter_mm
           << ", \"height_mm\": " << options.roi_height_mm
           << ", \"csr_columns_pruned\": true},\n"
           << "  \"bed_step_mm\": " << options.bed_step_mm << ",\n"
           << "  \"multibed_edge_layers\": "
           << options.multibed_edge_layers << ",\n"
           << "  \"mask_shape_xyz\": [" << options.mask_nx << ", "
           << options.mask_ny << ", " << options.mask_nz << "],\n"
           << "  \"mask_layers\": " << mask_layers << ",\n"
           << "  \"step_layers\": " << step_layers << ",\n"
           << "  \"overlap_layers\": " << mask_layers - step_layers << ",\n"
           << "  \"voxel_size_mm\": " << output.voxel_mm << ",\n"
           << "  \"output_shape_xyz\": [" << output.nx << ", " << output.ny << ", " << output.nz << "],\n"
           << "  \"data_mode\": \"" << options.data_mode << "\",\n"
           << "  \"normalization_factor\": \""
           << (options.no_norm_factor ? "disabled" : json_path(options.norm_factor)) << "\",\n"
           << "  \"saved_iterations\": [";
    for (size_t i = 0; i < saved_iterations.size(); ++i) {
        report << saved_iterations[i]
               << (i + 1 == saved_iterations.size() ? "" : ", ");
    }
    report << "],\n"
           << "  \"beds\": [\n";
    for (size_t i = 0; i < beds.size(); ++i) {
        report << "    {\"name\": \"Bed" << beds[i].number
               << "\", \"input\": \"" << json_path(beds[i].path)
               << "\", \"gain\": " << gains[i] << "}"
               << (i + 1 == beds.size() ? "\n" : ",\n");
    }
    report << "  ],\n"
           << "  \"output_pattern\": {"
           << (options.keep_raw
                   ? "\"full_size\": \"single_bed_raw/BedN/recon<iteration>.hdr|img\", "
                   : "")
           << "\"single_bed\": \"single_bed/BedN/recon<iteration>.hdr|img\", "
           << "\"stitched\": \"stitched/recon<iteration>.hdr|img\"";
    if (options.stitch_filter == "smooth" || options.stitch_filter == "both") {
        report << ", \"stitched_smooth\": \"stitched_smooth_fwhm"
               << decimal_path_token(options.stitch_smooth_fwhm_mm)
               << "/recon<iteration>.hdr|img\"";
    }
    if (options.stitch_filter == "butterworth" ||
        options.stitch_filter == "both") {
        report << ", \"stitched_butterworth\": [";
        for (size_t i = 0; i < options.stitch_butter_variants.size(); ++i) {
            const auto& variant = options.stitch_butter_variants[i];
            report << "\"stitched_butter_fc"
                   << decimal_path_token(variant.first)
                   << "cpcm_o" << variant.second
                   << "/recon<iteration>.hdr|img\""
                   << (i + 1 == options.stitch_butter_variants.size()
                           ? "" : ", ");
        }
        report << "]";
    }
    report << "}\n}\n";
}

// 公开调度器由一个 CUDA 上下文管理全部 BedN 状态：局部似然彼此独立，只共享完整
// CSR 或当前加载的角度分块。raw_root 只是现有裁剪/拼接流程的中间目录。
static int run_reconstruction(const BedMergeOptions& options,
                              const std::vector<BedInput>& beds,
                              const fs::path& raw_root) {
    return run_independent_iteration_major_reconstruction(options, beds, raw_root);
}

static int self_test_mirror_index(int value, int size) {
    while (value < 0 || value >= size) {
        value = value < 0 ? -value - 1 : 2 * size - value - 1;
    }
    return value;
}

static float self_test_tv_forward_denom(const std::vector<float>& image,
                                        int x, int y, int z,
                                        int Nx, int Ny, int Nz,
                                        float smoothing) {
    const int64_t plane = (int64_t)Nx * Ny;
    const int64_t index = (int64_t)z * plane + (int64_t)y * Nx + x;
    const float center = image[(size_t)index];
    const float dx = x + 1 < Nx ? image[(size_t)(index + 1)] - center : 0.0f;
    const float dy = y + 1 < Ny ? image[(size_t)(index + Nx)] - center : 0.0f;
    const float dz = z + 1 < Nz ? image[(size_t)(index + plane)] - center : 0.0f;
    return sqrtf(dx * dx + dy * dy + dz * dz + smoothing);
}

/*
 * 不读取投影和系统矩阵的快速数值自检：
 * 1. 对比严格 CASToR PSF 的原始反向和 castor-adjoint 真伴随，并验证两者
 *    前向逐点相同；检查 OMEGA symmetric 卷积保持常量；
 * 2. 分别检查 OMEGA epsilon ratio 与 CASToR threshold 中性 ratio；
 * 3. 分别检查 OMEGA/CASToR 的 MLEM、OSL、更新上下限及非正分母规则；
 * 4. 检查 TV，并将 OMEGA/CASToR MRP 与各自 CPU 参考逐体素比较；
 * 5. 检查 Butterworth 保持直流分量并衰减高频分量；
 * 6. 检查 ROI 对 Host CSR 列的裁剪；
 * 7. 检查拼接后 Gaussian/Butterworth 包装层的尺寸和有限值。
 */
static int run_algorithm_self_tests() {
    // 调度边界是 1-based：start=101 表示前100轮不滤波，第101轮首次滤波。
    if (should_apply_filter(100, 1000, "every", 101, 5, 10) ||
        !should_apply_filter(101, 1000, "every", 101, 5, 10) ||
        !should_apply_filter(101, 1000, "periodic", 101, 5, 10) ||
        should_apply_filter(102, 1000, "periodic", 101, 5, 10) ||
        !should_apply_filter(106, 1000, "periodic", 101, 5, 10) ||
        !should_apply_filter(1000, 1000, "final", 101, 5, 10) ||
        should_apply_smooth(100, 1000, "every", 101, 5) ||
        !should_apply_smooth(101, 1000, "every", 101, 5) ||
        !should_apply_smooth(101, 1000, "periodic", 101, 5) ||
        should_apply_smooth(102, 1000, "periodic", 101, 5) ||
        !should_apply_smooth(106, 1000, "periodic", 101, 5) ||
        !should_apply_smooth(1000, 1000, "final", 101, 5)) {
        fail("filter scheduling self-test failed");
    }
    std::printf("Filter scheduling boundary test: PASS\n");

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 1) fail("algorithm self-test requires a CUDA device");
    CUDA_CHECK(cudaSetDevice(0));

    const int Nx = 17, Ny = 15, Nz = 13;
    const int64_t n = (int64_t)Nx * Ny * Nz;
    const int blocks = (int)((n + BLK - 1) / BLK);
    std::vector<float> x((size_t)n), y((size_t)n), cx((size_t)n), cty((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        x[(size_t)i] = 0.3f + 0.2f * sinf((float)i * 0.017f) +
                       0.05f * cosf((float)i * 0.071f);
        y[(size_t)i] = -0.1f + 0.3f * cosf((float)i * 0.023f) -
                       0.04f * sinf((float)i * 0.113f);
    }

    float *d_x = nullptr, *d_y = nullptr, *d_cx = nullptr, *d_cty = nullptr;
    float *d_norm = nullptr, *d_tmp1 = nullptr, *d_tmp2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cx, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cty, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp1, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp2, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), (size_t)n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), (size_t)n * sizeof(float),
                          cudaMemcpyHostToDevice));

    const std::vector<float> castor_kernel =
        build_castor_erf_1d(1.5f, 0.3f, 3.0f);
    build_castor_z_norm(d_norm, Nx, Ny, Nz,
                        castor_kernel, castor_kernel, castor_kernel);
    apply_convolution_forward(
        CONV_IMPLEMENTATION_CASTOR, d_x, d_cx, d_tmp1, d_norm,
        Nx, Ny, Nz, castor_kernel, castor_kernel, castor_kernel);
    CUDA_CHECK(cudaMemcpy(cx.data(), d_cx, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::vector<float> strict_backward((size_t)n);
    apply_convolution_backward(
        CONV_IMPLEMENTATION_CASTOR, d_y, d_cty, d_tmp1, d_norm,
        Nx, Ny, Nz, castor_kernel, castor_kernel, castor_kernel);
    CUDA_CHECK(cudaMemcpy(strict_backward.data(), d_cty,
                          (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    apply_convolution_backward(
        CONV_IMPLEMENTATION_CASTOR_ADJOINT, d_y, d_cty, d_tmp1, d_norm,
        Nx, Ny, Nz, castor_kernel, castor_kernel, castor_kernel);
    CUDA_CHECK(cudaMemcpy(cty.data(), d_cty, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double lhs = 0.0, strict_rhs = 0.0, adjoint_rhs = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        lhs += (double)cx[(size_t)i] * y[(size_t)i];
        strict_rhs += (double)x[(size_t)i] * strict_backward[(size_t)i];
        adjoint_rhs += (double)x[(size_t)i] * cty[(size_t)i];
    }
    const double strict_castor_adjoint_error = fabs(lhs - strict_rhs) /
        std::max(1e-12, std::max(fabs(lhs), fabs(strict_rhs)));
    const double fixed_castor_adjoint_error = fabs(lhs - adjoint_rhs) /
        std::max(1e-12, std::max(fabs(lhs), fabs(adjoint_rhs)));

    apply_convolution_forward(
        CONV_IMPLEMENTATION_CASTOR_ADJOINT, d_x, d_cty, d_tmp1, d_norm,
        Nx, Ny, Nz, castor_kernel, castor_kernel, castor_kernel);
    CUDA_CHECK(cudaMemcpy(cty.data(), d_cty, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double castor_forward_match_error = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        castor_forward_match_error = std::max(
            castor_forward_match_error,
            fabs((double)cx[(size_t)i] - cty[(size_t)i]));
    }

    std::fill(x.begin(), x.end(), 1.0f);
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), (size_t)n * sizeof(float),
                          cudaMemcpyHostToDevice));
    const std::vector<float> omega_kernel =
        build_omega_sampled_1d(1.5f, 0.3f);
    apply_convolution_forward(
        CONV_IMPLEMENTATION_OMEGA, d_x, d_cx, d_tmp1, nullptr,
        Nx, Ny, Nz, omega_kernel, omega_kernel, omega_kernel);
    CUDA_CHECK(cudaMemcpy(cx.data(), d_cx, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double omega_constant_error = 0.0;
    for (float value : cx) {
        omega_constant_error = std::max(
            omega_constant_error, fabs((double)value - 1.0));
    }

    // 三条投影线覆盖 OMEGA epsilon、CASToR 阈值相等和正常除法三种情况。
    const int ratio_count = 3;
    const std::vector<float> ratio_forward = {0.0f, 0.05f, 2.0f};
    const std::vector<float> ratio_projection = {5.0f, 5.0f, 5.0f};
    const std::vector<float> ratio_response = {2.0f, 2.0f, 2.0f};
    CUDA_CHECK(cudaMemcpy(d_x, ratio_forward.data(),
                          ratio_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cx, ratio_projection.data(),
                          ratio_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cty, ratio_response.data(),
                          ratio_count * sizeof(float), cudaMemcpyHostToDevice));
    unsigned long long* d_test_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_test_counter, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_test_counter, 0, sizeof(unsigned long long)));

    k_calc_ratio_batched<<<1, BLK>>>(
        d_x, d_y, d_cx, d_cty, ratio_count, 0, ratio_count,
        ratio_count, 1, 1, ratio_count, SYM_NONE, 0, 0, 1, 1, 1,
        IMPLEMENTATION_OMEGA, 0.1f, 0.1f, nullptr);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> ratio_result((size_t)ratio_count);
    CUDA_CHECK(cudaMemcpy(ratio_result.data(), d_y,
                          ratio_count * sizeof(float), cudaMemcpyDeviceToHost));
    const std::vector<double> omega_ratio_expected = {
        100.0, 50.0, 10.0 / 4.1};
    double omega_ratio_error = 0.0;
    for (int i = 0; i < ratio_count; ++i) {
        omega_ratio_error = std::max(
            omega_ratio_error,
            fabs((double)ratio_result[(size_t)i] -
                 omega_ratio_expected[(size_t)i]));
    }

    k_calc_ratio_batched<<<1, BLK>>>(
        d_x, d_y, d_cx, d_cty, ratio_count, 0, ratio_count,
        ratio_count, 1, 1, ratio_count, SYM_NONE, 0, 0, 1, 1, 1,
        IMPLEMENTATION_CASTOR, 0.1f, 0.1f, d_test_counter);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(ratio_result.data(), d_y,
                          ratio_count * sizeof(float), cudaMemcpyDeviceToHost));
    unsigned long long castor_threshold_count = 0;
    CUDA_CHECK(cudaMemcpy(&castor_threshold_count, d_test_counter,
                          sizeof(castor_threshold_count),
                          cudaMemcpyDeviceToHost));
    const std::vector<double> castor_ratio_expected = {2.0, 2.0, 2.5};
    double castor_ratio_error = 0.0;
    for (int i = 0; i < ratio_count; ++i) {
        castor_ratio_error = std::max(
            castor_ratio_error,
            fabs((double)ratio_result[(size_t)i] -
                 castor_ratio_expected[(size_t)i]));
    }

    /*
     * 用包含 NaN、负 sensitivity、非正 raw denominator 和非正旧值的四体素
     * 用例直接核对详细诊断。正式更新与诊断 kernel 相互独立，因此这里既检查
     * 分类计数和范围，也在后续更新中检查两种实现的实际动作计数。
     */
    const int diagnostic_count = 4;
    const std::vector<float> diagnostic_old = {2.0f, 2.0f, 2.0f, 0.0f};
    const std::vector<float> diagnostic_bp = {
        3.0f, 3.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
    const std::vector<float> diagnostic_sensitivity = {4.0f, 0.1f, 0.1f, -0.5f};
    const std::vector<float> diagnostic_gradient = {0.5f, -0.075f, 0.0f, 0.1f};
    const std::vector<uint8_t> diagnostic_roi((size_t)diagnostic_count, 1);
    CUDA_CHECK(cudaMemcpy(d_x, diagnostic_old.data(),
                          diagnostic_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, diagnostic_bp.data(),
                          diagnostic_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cx, diagnostic_sensitivity.data(),
                          diagnostic_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cty, diagnostic_gradient.data(),
                          diagnostic_count * sizeof(float), cudaMemcpyHostToDevice));

    OslDiagnostics* d_test_osl_diagnostics = nullptr;
    uint8_t* d_test_osl_roi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_test_osl_diagnostics, sizeof(OslDiagnostics)));
    CUDA_CHECK(cudaMalloc(&d_test_osl_roi,
                          diagnostic_count * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_test_osl_roi, diagnostic_roi.data(),
                          diagnostic_count * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    const OslDiagnostics empty_test_diagnostics = empty_osl_diagnostics();
    CUDA_CHECK(cudaMemcpy(d_test_osl_diagnostics, &empty_test_diagnostics,
                          sizeof(OslDiagnostics), cudaMemcpyHostToDevice));
    k_collect_osl_diagnostics<<<1, BLK>>>(
        d_x, d_y, d_cx, d_cty, d_test_osl_roi,
        2.0f, 0.1f, IMPLEMENTATION_OMEGA,
        d_test_osl_diagnostics, diagnostic_count);
    CUDA_CHECK(cudaGetLastError());
    OslDiagnostics collected_diagnostics = empty_osl_diagnostics();
    CUDA_CHECK(cudaMemcpy(&collected_diagnostics, d_test_osl_diagnostics,
                          sizeof(OslDiagnostics), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_test_osl_roi));

    const bool osl_diagnostic_counts_pass =
        collected_diagnostics.counts[OSL_DIAG_ACTIVE_ROI] == 4 &&
        collected_diagnostics.counts[OSL_DIAG_OLD_NONFINITE] == 0 &&
        collected_diagnostics.counts[OSL_DIAG_OLD_NONPOSITIVE] == 1 &&
        collected_diagnostics.counts[OSL_DIAG_BP_NONFINITE] == 1 &&
        collected_diagnostics.counts[OSL_DIAG_BP_NEGATIVE] == 0 &&
        collected_diagnostics.counts[OSL_DIAG_SENS_NONFINITE] == 0 &&
        collected_diagnostics.counts[OSL_DIAG_SENS_NONPOSITIVE] == 1 &&
        collected_diagnostics.counts[OSL_DIAG_GRAD_NONFINITE] == 0 &&
        collected_diagnostics.counts[OSL_DIAG_PENALTY_NONFINITE] == 0 &&
        collected_diagnostics.counts[OSL_DIAG_RAW_DENOM_NONFINITE] == 0 &&
        collected_diagnostics.counts[OSL_DIAG_RAW_DENOM_NONPOSITIVE] == 2 &&
        collected_diagnostics.counts[OSL_DIAG_EFFECTIVE_DENOM_NONFINITE] == 0 &&
        collected_diagnostics.counts[OSL_DIAG_EFFECTIVE_DENOM_NONPOSITIVE] == 0;
    double osl_diagnostic_range_error = 0.0;
    auto update_diagnostic_range_error = [&](int range,
                                             double expected_minimum,
                                             double expected_maximum) {
        osl_diagnostic_range_error = std::max(
            osl_diagnostic_range_error,
            fabs((double)collected_diagnostics.minima[range] - expected_minimum));
        osl_diagnostic_range_error = std::max(
            osl_diagnostic_range_error,
            fabs((double)collected_diagnostics.maxima[range] - expected_maximum));
    };
    update_diagnostic_range_error(OSL_RANGE_SENSITIVITY, -0.5, 4.0);
    update_diagnostic_range_error(OSL_RANGE_PENALTY, -0.15, 1.0);
    update_diagnostic_range_error(OSL_RANGE_RAW_DENOMINATOR, -0.3, 5.0);
    update_diagnostic_range_error(OSL_RANGE_EFFECTIVE_DENOMINATOR, 0.05, 5.1);
    update_diagnostic_range_error(OSL_RANGE_BETA_LIMIT, 0.2 / 0.075,
                                  0.2 / 0.075);
    update_diagnostic_range_error(OSL_RANGE_NEGATIVE_PENALTY_FRACTION,
                                  0.75, 0.75);
    OslDiagnostics last_em_diagnostics = empty_osl_diagnostics();

    // 用同一启动辅助分别执行四条 EM 路径，主机端公式作为独立参考。
    auto run_em_case = [&](const std::vector<float>& old_values,
                           const std::vector<float>& backprojection,
                           const std::vector<float>& sensitivity,
                           const std::vector<float>& gradient,
                           int algorithm, int implementation, float beta,
                           float epsilon, float min_update,
                           float max_update) {
        const int64_t count = (int64_t)old_values.size();
        CUDA_CHECK(cudaMemcpy(d_x, old_values.data(),
                              (size_t)count * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, backprojection.data(),
                              (size_t)count * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cx, sensitivity.data(),
                              (size_t)count * sizeof(float),
                              cudaMemcpyHostToDevice));
        if (algorithm == ALGORITHM_OSL) {
            CUDA_CHECK(cudaMemcpy(d_cty, gradient.data(),
                                  (size_t)count * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemset(d_test_counter, 0,
                              sizeof(unsigned long long)));
        const OslDiagnostics empty_em_diagnostics = empty_osl_diagnostics();
        CUDA_CHECK(cudaMemcpy(d_test_osl_diagnostics, &empty_em_diagnostics,
                              sizeof(OslDiagnostics), cudaMemcpyHostToDevice));
        k_em_update_versioned<<<1, BLK>>>(
            d_x, d_y, d_cx,
            algorithm == ALGORITHM_OSL ? d_cty : nullptr,
            algorithm, implementation, beta, epsilon,
            min_update, max_update, d_test_counter,
            d_test_osl_diagnostics, count);
        CUDA_CHECK(cudaGetLastError());
        std::vector<float> output((size_t)count);
        CUDA_CHECK(cudaMemcpy(output.data(), d_x,
                              (size_t)count * sizeof(float),
                              cudaMemcpyDeviceToHost));
        unsigned long long warning_count = 0;
        CUDA_CHECK(cudaMemcpy(&warning_count, d_test_counter,
                              sizeof(warning_count), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last_em_diagnostics, d_test_osl_diagnostics,
                              sizeof(OslDiagnostics), cudaMemcpyDeviceToHost));
        return std::make_pair(output, warning_count);
    };

    const std::vector<float> em_old = {2.0f, 2.0f, 2.0f};
    const std::vector<float> zero_gradient = {0.0f, 0.0f, 0.0f};
    const auto omega_mlem = run_em_case(
        em_old, {3.0f, 0.0f, 1.0f}, {4.0f, 0.01f, 0.0f},
        zero_gradient, ALGORITHM_MLEM, IMPLEMENTATION_OMEGA,
        0.0f, 0.1f, 0.5f, 2.0f);
    const std::vector<double> omega_mlem_expected = {1.55, 2.0, 22.0};
    double omega_mlem_error = 0.0;
    for (size_t i = 0; i < em_old.size(); ++i) {
        omega_mlem_error = std::max(
            omega_mlem_error,
            fabs((double)omega_mlem.first[i] - omega_mlem_expected[i]));
    }

    const auto castor_mlem = run_em_case(
        em_old, {0.1f, 12.0f, 3.0f}, {1.0f, 2.0f, 4.0f},
        zero_gradient, ALGORITHM_MLEM, IMPLEMENTATION_CASTOR,
        0.0f, 0.1f, 0.5f, 2.0f);
    const std::vector<double> castor_mlem_expected = {1.0, 4.0, 1.5};
    double castor_mlem_error = 0.0;
    for (size_t i = 0; i < em_old.size(); ++i) {
        castor_mlem_error = std::max(
            castor_mlem_error,
            fabs((double)castor_mlem.first[i] - castor_mlem_expected[i]));
    }

    const auto omega_osl = run_em_case(
        em_old, {3.0f, 3.0f, 3.0f}, {4.0f, 0.1f, 0.1f},
        {0.5f, -0.075f, -0.2f}, ALGORITHM_OSL,
        IMPLEMENTATION_OMEGA, 2.0f, 0.1f, 0.5f, 2.0f);
    const unsigned long long omega_osl_floor_actions =
        last_em_diagnostics.counts[OSL_DIAG_OMEGA_EPSILON_FLOOR];
    const std::vector<double> omega_osl_expected = {
        2.0 * 3.1 / 5.1, 0.1, 0.1};
    double omega_osl_error = 0.0;
    for (size_t i = 0; i < em_old.size(); ++i) {
        omega_osl_error = std::max(
            omega_osl_error,
            fabs((double)omega_osl.first[i] - omega_osl_expected[i]));
    }

    const auto castor_osl = run_em_case(
        em_old, {3.0f, 3.0f, 20.0f}, {4.0f, 1.0f, 4.0f},
        {0.5f, -1.0f, 0.0f}, ALGORITHM_OSL,
        IMPLEMENTATION_CASTOR, 2.0f, 0.1f, 0.5f, 2.0f);
    const unsigned long long castor_osl_keep_actions =
        last_em_diagnostics.counts[OSL_DIAG_CASTOR_KEEP_OLD];
    const std::vector<double> castor_osl_expected = {1.2, 2.0, 4.0};
    double castor_osl_error = 0.0;
    for (size_t i = 0; i < em_old.size(); ++i) {
        castor_osl_error = std::max(
            castor_osl_error,
            fabs((double)castor_osl.first[i] - castor_osl_expected[i]));
    }
    CUDA_CHECK(cudaFree(d_test_osl_diagnostics));
    CUDA_CHECK(cudaFree(d_test_counter));

    std::fill(x.begin(), x.end(), 1.0f);
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), (size_t)n * sizeof(float),
                          cudaMemcpyHostToDevice));
    k_tv_gradient_omega<<<blocks, BLK>>>(d_x, d_cty, Nx, Ny, Nz, 1e-4f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(cty.data(), d_cty, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double tv_constant_error = 0.0;
    for (float value : cty) tv_constant_error = std::max(tv_constant_error,
                                                         fabs((double)value));

    for (int64_t i = 0; i < n; ++i) {
        x[(size_t)i] = 0.25f + 0.01f * (float)(i % 19) +
                       0.03f * sinf((float)i * 0.11f);
    }
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), (size_t)n * sizeof(float),
                          cudaMemcpyHostToDevice));
    k_tv_gradient_omega<<<blocks, BLK>>>(d_x, d_cty, Nx, Ny, Nz, 1e-4f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(cty.data(), d_cty, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double tv_reference_error = 0.0;
    const int64_t plane = (int64_t)Nx * Ny;
    for (int z = 0; z < Nz; ++z) {
        for (int yy = 0; yy < Ny; ++yy) {
            for (int xx = 0; xx < Nx; ++xx) {
                const int64_t index = (int64_t)z * plane + (int64_t)yy * Nx + xx;
                const float center = x[(size_t)index];
                const float xp = xx + 1 < Nx ? x[(size_t)(index + 1)] : center;
                const float yp = yy + 1 < Ny ? x[(size_t)(index + Nx)] : center;
                const float zp = z + 1 < Nz ? x[(size_t)(index + plane)] : center;
                double expected = (3.0 * center - xp - yp - zp) /
                    self_test_tv_forward_denom(x, xx, yy, z, Nx, Ny, Nz, 1e-4f);
                if (xx > 0) {
                    expected += (center - x[(size_t)(index - 1)]) /
                        self_test_tv_forward_denom(x, xx - 1, yy, z,
                                                   Nx, Ny, Nz, 1e-4f);
                }
                if (yy > 0) {
                    expected += (center - x[(size_t)(index - Nx)]) /
                        self_test_tv_forward_denom(x, xx, yy - 1, z,
                                                   Nx, Ny, Nz, 1e-4f);
                }
                if (z > 0) {
                    expected += (center - x[(size_t)(index - plane)]) /
                        self_test_tv_forward_denom(x, xx, yy, z - 1,
                                                   Nx, Ny, Nz, 1e-4f);
                }
                tv_reference_error = std::max(
                    tv_reference_error,
                    fabs((double)cty[(size_t)index] - expected));
            }
        }
    }

    k_mrp_gradient_omega<<<blocks, BLK>>>(d_x, d_cty, Nx, Ny, Nz,
                                          1, 1, 1, 1e-8f, false);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(cty.data(), d_cty, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double omega_mrp_error = 0.0;
    for (int z = 0; z < Nz; ++z) {
        for (int yy = 0; yy < Ny; ++yy) {
            for (int xx = 0; xx < Nx; ++xx) {
                std::vector<float> values;
                values.reserve(27);
                for (int dz = -1; dz <= 1; ++dz) {
                    const int zz = self_test_mirror_index(z + dz, Nz);
                    for (int dy = -1; dy <= 1; ++dy) {
                        const int y2 = self_test_mirror_index(yy + dy, Ny);
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int x2 = self_test_mirror_index(xx + dx, Nx);
                            values.push_back(x[(int64_t)zz * Nx * Ny +
                                               (int64_t)y2 * Nx + x2]);
                        }
                    }
                }
                std::sort(values.begin(), values.end());
                const int64_t index = (int64_t)z * Nx * Ny +
                                      (int64_t)yy * Nx + xx;
                const double median = values[values.size() / 2];
                const double expected = (x[(size_t)index] - median) /
                                        (median + 1e-8);
                omega_mrp_error = std::max(
                    omega_mrp_error,
                    fabs((double)cty[(size_t)index] - expected));
            }
        }
    }

    // CASToR 默认 6-nearest 包含中心体素，边界外邻居丢弃，偶数样本取中间平均。
    k_mrp_gradient_castor<<<blocks, BLK>>>(
        d_x, d_cty, Nx, Ny, Nz, MRP_NEIGHBORHOOD_SIX_NEAREST,
        1, false, 1.0f, 1.0f, 1.0e6f, 1.0e6f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(cty.data(), d_cty, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double castor_mrp_error = 0.0;
    const int neighbor_dx[7] = {0, -1, 1, 0, 0, 0, 0};
    const int neighbor_dy[7] = {0, 0, 0, -1, 1, 0, 0};
    const int neighbor_dz[7] = {0, 0, 0, 0, 0, -1, 1};
    for (int z = 0; z < Nz; ++z) {
        for (int yy = 0; yy < Ny; ++yy) {
            for (int xx = 0; xx < Nx; ++xx) {
                std::vector<float> values;
                values.reserve(7);
                for (int k = 0; k < 7; ++k) {
                    const int x2 = xx + neighbor_dx[k];
                    const int y2 = yy + neighbor_dy[k];
                    const int z2 = z + neighbor_dz[k];
                    if (x2 < 0 || x2 >= Nx || y2 < 0 || y2 >= Ny ||
                        z2 < 0 || z2 >= Nz) continue;
                    values.push_back(x[(int64_t)z2 * Nx * Ny +
                                       (int64_t)y2 * Nx + x2]);
                }
                std::sort(values.begin(), values.end());
                double median = values[values.size() / 2];
                if ((values.size() & 1u) == 0u) {
                    median = 0.5 * (median + values[values.size() / 2 - 1]);
                }
                const int64_t index = (int64_t)z * Nx * Ny +
                                      (int64_t)yy * Nx + xx;
                const double expected = median != 0.0
                    ? ((double)x[(size_t)index] - median) / median : 0.0;
                castor_mrp_error = std::max(
                    castor_mrp_error,
                    fabs((double)cty[(size_t)index] - expected));
            }
        }
    }

    // radius=voxel 时严格“距离 < radius”只保留中心体素，梯度应全为 0。
    k_mrp_gradient_castor<<<blocks, BLK>>>(
        d_x, d_cty, Nx, Ny, Nz, MRP_NEIGHBORHOOD_SPHERE,
        1, false, 1.0f, 1.0f, 1.0e6f, 1.0e6f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(cty.data(), d_cty, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double castor_sphere_strict_error = 0.0;
    for (float value : cty) {
        castor_sphere_strict_error = std::max(
            castor_sphere_strict_error, fabs((double)value));
    }

    /* 令 cutoff 精确落在补零后 x 轴第 5 个频点，直接验证 H(fc)=0.5。 */
    const float butter_voxel_mm = 1.0f;
    const int butter_order = 4;
    const float butter_cutoff_cycles_per_cm =
        10.0f * 5.0f / (float)(2 * Nx);
    ButterworthWorkspace butter_workspace;
    initialize_butterworth_workspace(
        butter_workspace, Nx, Ny, Nz, butter_voxel_mm,
        butter_cutoff_cycles_per_cm, butter_order);
    const std::vector<float> butter_response = build_butterworth_response(
        butter_workspace.padded_nx, butter_workspace.padded_ny,
        butter_workspace.padded_nz, butter_voxel_mm,
        butter_cutoff_cycles_per_cm, butter_order);
    const double butter_dc_error = fabs((double)butter_response[0] - 1.0);
    const double butter_cutoff_error = fabs((double)butter_response[5] - 0.5);
    const double butter_nyquist_gain =
        butter_response[(size_t)(butter_workspace.padded_nx / 2)];

    auto apply_self_test_butterworth = [&]() {
        CUDA_CHECK(cudaMemcpy(d_x, x.data(), (size_t)n * sizeof(float),
                              cudaMemcpyHostToDevice));
        apply_butterworth_filter(butter_workspace, d_x, d_cx, false);
        CUDA_CHECK(cudaMemcpy(cx.data(), d_cx, (size_t)n * sizeof(float),
                              cudaMemcpyDeviceToHost));
    };

    const int64_t edge_index = (int64_t)(Nz / 2) * plane +
                               (int64_t)(Ny / 2) * Nx;
    std::fill(x.begin(), x.end(), 0.0f);
    x[(size_t)edge_index] = 1.0f;
    apply_self_test_butterworth();
    const std::vector<float> positive_impulse = cx;
    const double butter_wrap_ratio =
        fabs((double)cx[(size_t)(edge_index + Nx - 1)]) /
        std::max(1e-12, fabs((double)cx[(size_t)edge_index]));

    // 正负输入应严格反号；该检查可发现旧实现的无条件负值截断。
    x[(size_t)edge_index] = -1.0f;
    apply_self_test_butterworth();
    double butter_linearity_error = 0.0;
    for (size_t i = 0; i < cx.size(); ++i) {
        butter_linearity_error = std::max(
            butter_linearity_error,
            fabs((double)positive_impulse[i] + (double)cx[i]));
    }

    double input_ac_energy = 0.0;
    for (int z = 0; z < Nz; ++z) {
        for (int yy = 0; yy < Ny; ++yy) {
            for (int xx = 0; xx < Nx; ++xx) {
                const int64_t index = (int64_t)z * plane + (int64_t)yy * Nx + xx;
                x[(size_t)index] = ((xx + yy + z) & 1) ? 1.0f : 0.0f;
                const double centered = (double)x[(size_t)index] - 0.5;
                input_ac_energy += centered * centered;
            }
        }
    }
    apply_self_test_butterworth();
    const double output_mean = std::accumulate(cx.begin(), cx.end(), 0.0) /
                               (double)cx.size();
    double output_ac_energy = 0.0;
    for (float value : cx) {
        const double centered = (double)value - output_mean;
        output_ac_energy += centered * centered;
    }
    const double butter_ac_ratio = output_ac_energy /
        std::max(1e-12, input_ac_energy);

    destroy_butterworth_workspace(butter_workspace);

    CUDA_CHECK(cudaFree(d_tmp2));
    CUDA_CHECK(cudaFree(d_tmp1));
    CUDA_CHECK(cudaFree(d_norm));
    CUDA_CHECK(cudaFree(d_cty));
    CUDA_CHECK(cudaFree(d_cx));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_x));

    // 小型 Host CSR 检查：3x3x1 网格中直径 1.1 mm 的圆柱只保留中心列 4。
    int64_t roi_active = 0;
    const std::vector<uint8_t> roi_test_mask =
        build_centered_cylinder_roi_mask(3, 3, 1, 1.0f,
                                         1.1f, 1.1f, roi_active);
    HostCSR64 roi_test_csr;
    roi_test_csr.num_rows = 2;
    roi_test_csr.num_cols = 9;
    roi_test_csr.nnz = 7;
    roi_test_csr.row_ptr = {0, 4, 7};
    roi_test_csr.col_idx = {0, 4, 8, 3, 4, 5, 6};
    roi_test_csr.values = {1.0f, 2.0f, 3.0f, 4.0f,
                           5.0f, 6.0f, 7.0f};
    const int64_t roi_removed =
        prune_host_csr_to_roi(roi_test_csr, roi_test_mask);
    const bool roi_csr_pass =
        roi_active == 1 && roi_removed == 5 && roi_test_csr.nnz == 2 &&
        roi_test_csr.row_ptr == std::vector<int64_t>({0, 1, 2}) &&
        roi_test_csr.col_idx == std::vector<int32_t>({4, 4}) &&
        roi_test_csr.values == std::vector<float>({2.0f, 5.0f});

    // 直接经过拼接后包装层，验证 double<->float 转换、按需 GPU 缓冲和输出几何。
    Volume stitch_filter_test;
    stitch_filter_test.nx = 9;
    stitch_filter_test.ny = 7;
    stitch_filter_test.nz = 5;
    stitch_filter_test.voxel_mm = 1.0;
    stitch_filter_test.data.assign(9u * 7u * 5u, 0.0);
    stitch_filter_test.data[stitch_filter_test.data.size() / 2] = 1.0;
    const Volume stitch_smooth_test = smooth_stitched_volume(
        stitch_filter_test, 1.2f, 3.0f);
    const Volume stitch_butter_test = butterworth_stitched_volume(
        stitch_filter_test, 3.0f, 8);
    const VolumeDiagnostics stitch_smooth_diagnostics =
        diagnose_volume(stitch_smooth_test);
    const VolumeDiagnostics stitch_butter_diagnostics =
        diagnose_volume(stitch_butter_test);
    const bool stitch_filter_wrapper_pass =
        stitch_smooth_test.data.size() == stitch_filter_test.data.size() &&
        stitch_butter_test.data.size() == stitch_filter_test.data.size() &&
        stitch_smooth_diagnostics.maximum > 0.0 &&
        stitch_smooth_diagnostics.sum > 0.0 &&
        std::isfinite(stitch_butter_diagnostics.sum) &&
        stitch_butter_diagnostics.maximum > 0.0;

    std::printf("[self-test] strict CASToR PSF adjoint error = %.9e\n",
                strict_castor_adjoint_error);
    std::printf("[self-test] fixed CASToR PSF adjoint error  = %.9e\n",
                fixed_castor_adjoint_error);
    std::printf("[self-test] CASToR PSF forward match error  = %.9e\n",
                castor_forward_match_error);
    std::printf("[self-test] OMEGA PSF constant max error    = %.9e\n",
                omega_constant_error);
    std::printf("[self-test] OMEGA ratio max error           = %.9e\n",
                omega_ratio_error);
    std::printf("[self-test] CASToR ratio max error/count    = %.9e / %llu\n",
                castor_ratio_error, castor_threshold_count);
    std::printf("[self-test] OMEGA MLEM max error            = %.9e\n",
                omega_mlem_error);
    std::printf("[self-test] CASToR MLEM max error           = %.9e\n",
                castor_mlem_error);
    std::printf("[self-test] OMEGA OSL max error/warnings    = %.9e / %llu\n",
                omega_osl_error, omega_osl.second);
    std::printf("[self-test] CASToR OSL max error/warnings   = %.9e / %llu\n",
                castor_osl_error, castor_osl.second);
    std::printf("[self-test] OSL diagnostic counts          = %s\n",
                osl_diagnostic_counts_pass ? "PASS" : "FAIL");
    std::printf("[self-test] OSL diagnostic range error     = %.9e\n",
                osl_diagnostic_range_error);
    std::printf("[self-test] OSL omega-floor/castor-keep    = %llu / %llu\n",
                omega_osl_floor_actions, castor_osl_keep_actions);
    std::printf("[self-test] TV constant max error    = %.9e\n", tv_constant_error);
    std::printf("[self-test] TV CPU/GPU max error     = %.9e\n", tv_reference_error);
    std::printf("[self-test] OMEGA MRP max error       = %.9e\n", omega_mrp_error);
    std::printf("[self-test] CASToR MRP max error      = %.9e\n", castor_mrp_error);
    std::printf("[self-test] CASToR sphere strict err  = %.9e\n",
                castor_sphere_strict_error);
    std::printf("[self-test] Butterworth H(0) error    = %.9e\n", butter_dc_error);
    std::printf("[self-test] Butterworth H(fc) error   = %.9e\n", butter_cutoff_error);
    std::printf("[self-test] Butterworth Nyquist gain  = %.9e\n", butter_nyquist_gain);
    std::printf("[self-test] Butterworth wrap ratio    = %.9e\n", butter_wrap_ratio);
    std::printf("[self-test] Butterworth linearity err = %.9e\n", butter_linearity_error);
    std::printf("[self-test] Butterworth AC ratio      = %.9e\n", butter_ac_ratio);
    std::printf("[self-test] ROI CSR prune             = %s\n",
                roi_csr_pass ? "PASS" : "FAIL");
    std::printf("[self-test] stitched filter wrappers = %s\n",
                stitch_filter_wrapper_pass ? "PASS" : "FAIL");
    const bool passed =
        fixed_castor_adjoint_error < 5e-6 &&
        fixed_castor_adjoint_error <= strict_castor_adjoint_error + 1e-12 &&
        castor_forward_match_error < 1e-7 && omega_constant_error < 5e-6 &&
        omega_ratio_error < 2e-5 && castor_ratio_error < 2e-6 &&
        castor_threshold_count == 2 && omega_mlem_error < 2e-5 &&
        castor_mlem_error < 2e-6 && omega_osl_error < 2e-4 &&
        castor_osl_error < 2e-6 && omega_osl.second == 2 &&
        castor_osl.second == 1 && osl_diagnostic_counts_pass &&
        osl_diagnostic_range_error < 2e-5 &&
        omega_osl_floor_actions == 2 && castor_osl_keep_actions == 1 &&
        tv_constant_error < 1e-6 &&
        tv_reference_error < 2e-5 && omega_mrp_error < 2e-5 &&
        castor_mrp_error < 2e-5 && castor_sphere_strict_error < 1e-6 &&
        butter_dc_error < 1e-7 && butter_cutoff_error < 1e-6 &&
        butter_nyquist_gain < 0.01 && butter_wrap_ratio < 0.1 &&
        butter_linearity_error < 2e-5 && butter_ac_ratio < 0.5 &&
        roi_csr_pass && stitch_filter_wrapper_pass;
    std::printf("[self-test] %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

// 主流程：发现床位、参数校验、重建、mask 裁剪、增益拟合和拼接。
int main(int argc, char** argv) {
    try {
        const BedMergeOptions options = parse_options(argc, argv);
        if (options.self_test_algorithms) return run_algorithm_self_tests();
        const std::vector<BedInput> beds = discover_beds(options.proj_dir);
        const double voxel_mm = read_metadata_value(options.matrix_dir / "metadata.txt", "voxel_size_mm");
        const int matrix_nz = static_cast<int>(read_metadata_value(options.matrix_dir / "metadata.txt", "Nz"));
        const int matrix_nx = static_cast<int>(read_metadata_value(options.matrix_dir / "metadata.txt", "Nx"));
        const int matrix_ny = static_cast<int>(read_metadata_value(options.matrix_dir / "metadata.txt", "Ny"));
        const int step_layers = static_cast<int>(std::llround(options.bed_step_mm / voxel_mm));
        const double nyquist_cycles_per_cm = 5.0 / voxel_mm;
        const bool stitch_uses_butterworth =
            options.stitch_filter == "butterworth" ||
            options.stitch_filter == "both";
        if (stitch_uses_butterworth) {
            for (const auto& variant : options.stitch_butter_variants) {
                if (variant.first > nyquist_cycles_per_cm) {
                    std::ostringstream message;
                    message << "stitch Butterworth cutoff " << variant.first
                            << " exceeds voxel Nyquist " << nyquist_cycles_per_cm
                            << " cycles/cm (voxel=" << voxel_mm << " mm)";
                    fail(message.str());
                }
            }
        }
        if (options.mask_nx > matrix_nx || options.mask_ny > matrix_ny ||
            options.mask_nz > matrix_nz) {
            fail("mask size must not exceed system-matrix dimensions");
        }
        const double geometry_tolerance_mm = 1e-6 * std::max(1.0, voxel_mm);
        // The ROI prunes full system-matrix columns during reconstruction.  The
        // mask is applied only after reconstruction to crop saved single-bed
        // images, so neither spatial extent is allowed to constrain the other.
        if (options.gain_roi_diameter_mm > options.roi_diameter_mm + geometry_tolerance_mm) {
            fail("--gain-roi-diameter-mm must not exceed --roi-diameter-mm");
        }
        if (beds.size() > 1 &&
            (step_layers <= 0 || options.mask_nz < step_layers)) {
            fail("multi-bed mask Nz must be at least the bed-step in voxel units");
        }
        if (options.dry_run) {
            std::printf("输出目录: %s\n", options.out.string().c_str());
            std::printf("独立床位 iteration-major 重建与拼接: "
                        "%zu beds, mask=%dx%dx%d, step=%d layers, overlap=%d layers\n",
                        beds.size(), options.mask_nx, options.mask_ny, options.mask_nz,
                        step_layers, options.mask_nz - step_layers);
            std::printf("算法: %s; implementation=%s; conv=%s/%s; prior=%s; "
                         "beta=%.6g; butterworth=%s\n",
                         algorithm_name(options.algorithm),
                         reference_implementation_name(options.implementation),
                         options.conv_mode.c_str(),
                         conv_implementation_name(options.conv_implementation),
                         options.map_type == MAP_TV ? "tv" :
                         (options.map_type == MAP_MRP ? "mrp" : "none"),
                         options.map_beta,
                         options.butter_enable ? "on" : "off");
            std::printf("数值保护: OMEGA epsilon=%.9g; CASToR threshold=%.9g; "
                         "CASToR update=[%.6g,%.6g]\n",
                         (double)options.omega_epsilon,
                         (double)options.castor_threshold,
                        (double)options.castor_min_update,
                        (double)options.castor_max_update);
            std::printf("ROI: centered cylinder diameter=%.6g mm height=%.6g mm; "
                        "CSR columns outside ROI are pruned\n",
                        options.roi_diameter_mm, options.roi_height_mm);
            std::printf("轴向重建ROI: %s; 拼接输入mask=%d层\n",
                        beds.size() == 1 ? "single-bed fixed 72 layers" :
                        "multi-bed local matrix 220 layers, then global crop",
                        options.mask_nz);
            std::printf("保存: every %d iterations -> reconN; recon_final disabled\n",
                        options.save_interval);
            std::printf("拼接后滤波: %s; smooth=%.6g mm/%g sigma; "
                        "butterworth variants=",
                        options.stitch_filter.c_str(),
                        (double)options.stitch_smooth_fwhm_mm,
                        (double)options.stitch_smooth_sigmas);
            for (size_t i = 0; i < options.stitch_butter_variants.size(); ++i) {
                const auto& variant = options.stitch_butter_variants[i];
                std::printf("%s%.6g cycles/cm,o%d",
                            i == 0 ? "" : "; ",
                            (double)variant.first, variant.second);
            }
            std::printf("\n");
            return 0;
        }

        // raw_root 暂存完整矩阵尺寸，single_root 只保存用户指定的 mask 范围。
        const fs::path raw_root = options.out / "single_bed_raw";
        const fs::path single_root = options.out / "single_bed";
        make_parent_dirs(raw_root);
        make_parent_dirs(single_root);
        std::printf("[bedmerge] reconstruct %zu beds, then center-crop mask %dx%dx%d\n",
                    beds.size(), options.mask_nx, options.mask_ny, options.mask_nz);
        const int status = run_reconstruction(options, beds, raw_root);
        if (status != 0) fail("iteration-major CUDA reconstruction returned nonzero status");

        std::vector<int> saved_iterations;
        for (int iteration = options.save_interval;
             iteration <= options.iterations;
             iteration += options.save_interval) {
            saved_iterations.push_back(iteration);
        }
        std::vector<double> report_gains;
        Volume report_output;
        int report_mask_layers = 0;
        int report_step_layers = 0;
        const fs::path stitch_dir = options.out / "stitched";
        for (int iteration : saved_iterations) {
            const std::string stem = "recon" + std::to_string(iteration);
            std::vector<Volume> volumes;
            volumes.reserve(beds.size());
            for (const BedInput& bed : beds) {
                const fs::path bed_dir = raw_root /
                    ("Bed" + std::to_string(bed.number));
                const Volume full = read_nifti(bed_dir, stem);
                Volume masked = crop_center_mask(
                    full, options.mask_nx, options.mask_ny, options.mask_nz);
                write_nifti(single_root /
                                ("Bed" + std::to_string(bed.number)),
                            masked, stem.c_str());
                volumes.push_back(std::move(masked));
            }
            for (const Volume& volume : volumes) {
                if (volume.nx != volumes[0].nx || volume.ny != volumes[0].ny ||
                    volume.nz != volumes[0].nz ||
                    std::abs(volume.voxel_mm - volumes[0].voxel_mm) > 1e-6) {
                    fail("single-bed output geometries are inconsistent at " + stem);
                }
            }
            const int actual_step_layers = static_cast<int>(
                std::llround(options.bed_step_mm / volumes[0].voxel_mm));
            if (volumes.size() > 1 && volumes[0].nz < actual_step_layers) {
                fail("multi-bed mask Nz must be at least the bed-step in reconstructed voxel units");
            }
            std::vector<double> gains = fit_gains(
                volumes, actual_step_layers,
                volumes[0].voxel_mm, options.gain_roi_diameter_mm);
            Volume stitched = stitch_volumes(volumes, gains,
                                             options.bed_step_mm);
            write_nifti(stitch_dir, stitched, stem.c_str());
            const VolumeDiagnostics unfiltered_diagnostics =
                diagnose_volume(stitched);
            if (options.stitch_filter == "smooth" ||
                options.stitch_filter == "both") {
                Volume smoothed = smooth_stitched_volume(
                    stitched, options.stitch_smooth_fwhm_mm,
                    options.stitch_smooth_sigmas);
                const VolumeDiagnostics smoothed_diagnostics =
                    diagnose_volume(smoothed);
                print_stitch_filter_diagnostics(
                    "smooth", unfiltered_diagnostics, smoothed_diagnostics);
                const fs::path smooth_dir = options.out /
                    ("stitched_smooth_fwhm" + decimal_path_token(
                        options.stitch_smooth_fwhm_mm));
                write_nifti(smooth_dir, smoothed, stem.c_str());
            }
            if (stitch_uses_butterworth) {
                for (const auto& variant : options.stitch_butter_variants) {
                    Volume filtered = butterworth_stitched_volume(
                        stitched, variant.first, variant.second);
                    const VolumeDiagnostics filtered_diagnostics =
                        diagnose_volume(filtered);
                    const std::string label =
                        "butterworth_fc" + decimal_path_token(variant.first) +
                        "_o" + std::to_string(variant.second);
                    print_stitch_filter_diagnostics(
                        label.c_str(), unfiltered_diagnostics,
                        filtered_diagnostics);
                    const fs::path butter_dir = options.out /
                        ("stitched_butter_fc" + decimal_path_token(variant.first) +
                         "cpcm_o" + std::to_string(variant.second));
                    write_nifti(butter_dir, filtered, stem.c_str());
                }
            }
            report_gains = std::move(gains);
            report_output = std::move(stitched);
            report_mask_layers = volumes[0].nz;
            report_step_layers = actual_step_layers;
            std::printf("[bedmerge] saved cropped and stitched %s\n", stem.c_str());
        }
        if (!options.keep_raw) {
            // 完整矩阵结果仅作为裁剪阶段的中间文件，默认不作为正式输出保留。
            std::error_code remove_error;
            fs::remove_all(raw_root, remove_error);
            if (remove_error) {
                fail("cannot remove temporary full-size bed volumes: " +
                     raw_root.string() + ": " + remove_error.message());
            }
        }
        write_report(options.out / "bedmerge_report.json", options, beds,
                     report_gains, saved_iterations, report_output,
                     report_mask_layers, report_step_layers);
        std::printf("[bedmerge] complete: %s\n", options.out.string().c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "bedmerge error: %s\n", error.what());
        return 1;
    }
}

