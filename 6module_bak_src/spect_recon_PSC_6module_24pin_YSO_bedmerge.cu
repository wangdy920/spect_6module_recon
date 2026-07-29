/*
 * SPECT 多床位单床重建与拼接程序。
 *
 * 流程：逐床位执行完整系统矩阵重建，按中心 mask 裁剪 x/y/z，
 * 在相邻床位重叠区估计标量增益，最后使用余弦平方权重平滑拼接。
 * 体素线性索引固定为 z*(Nx*Ny)+y*Nx+x，其中 x 为最快变化维。
 *
 * 文件结构：
 * 1. 系统矩阵、投影数据和 CUDA MLEM 重建引擎；
 * 2. 单床位 mask 裁剪、增益估计和多床位拼接；
 * 3. 命令行入口及 NIfTI-1 成对文件输出。
 * 
  编译：
  nvcc -O3 -std=c++17 -arch=sm_89 \
  recon/spect_recon_PSC_6module_24pin_YSO_bedmerge.cu \
  -o recon/process/spect_recon_PSC_6module_24pin_YSO_bedmerge_sm89 \
  -lcusparse -lcufft

  usage:
  ./spect_recon_PSC_6module_24pin_YSO_bedmerge \
  --matrix-dir  MCmatrix_PSC_48h\
  --proj-dir  ep_168\
  --out recon_ep168_mask100x100x72 \
  --mask-size 100x100x72 \
  --bed-step-mm 16.8 \
  --norm-factor MCmatrix_PSC_48h/norm_factor.bin \
  --roi-diameter-mm 32.4 \
  --roi-height-mm 48 \
  --iters 50 \
  --gpus 2 

  ./spect_recon_PSC_6module_24pin_YSO_bedmerge \
  --matrix-dir /data/SPECT/MCmatrix_PSC \
  --proj-dir /data/SPECT/simulation_preprocess \
  --out /data/SPECT/recon_simulation_mask \
  --mask-size 100x100x72 \
  --bed-step-mm 16.8 \
  --data-mode simulation \
  --no-detector-mask \
  --no-norm-factor
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
    MAP_HUBER,
    MAP_RELATIVE,
    MAP_LOGCOSH,
    MAP_HYPERBOLIC
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
    float roi_height_mm = 21.6f;
    float roi_center_mm[3] = {0.0f, 0.0f, 0.0f};
    std::string data_mode = "measured";
    bool detector_mask_enable = true;

    // 迭代次数、保存周期和 GPU 数量。
    bool check_sens_zsym = false;
    bool save_sens_edge_png = false;
    bool save_raw1 = true;
    float sens_zsym_tol = 1e-4f;

    int num_iters = 50;
    int save_interval = 5;
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
    float map_huber_threshold = 0.01f;
    float map_relative_gamma = 2.0f;
    float map_relative_epsilon = 1e-6f;

    // 正值约束和 Butterworth 频域滤波。
    bool positivity = false;
    bool butter_enable = false;
    float butter_cutoff = 0.45f;
    int butter_order = 8;
    std::string butter_mode = "final";
    int butter_period = 5;
    int butter_last_n = 10;

    // 投影角度抽样步长。
    int recon_angle_step = 1;

    // 多床位投影名称、床位步进及显式床位 z 偏移。
    std::string bed_projection_name = "projection.bin";
    float bed_step_mm = 0.0f;
    bool bed_step_set = false;
    std::vector<float> bed_offsets_mm;

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
    gg.origin_z_mm = min_center - 0.5f * (float)(meta.Nz - 1) * meta.voxel_size;

    float max_shift = 0.0f;
    for (BedInfo& bed : beds) {
        float shift = (bed.center_mm - min_center) / meta.voxel_size;
        if (shift < -1e-5f) {
            fprintf(stderr, "Internal bed geometry error: negative shift %.8g\n", shift);
            exit(1);
        }

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
        max_shift = std::max(max_shift, bed.shift_vox);
    }

    gg.global_Nz = meta.Nz + (int)ceilf(max_shift - 1e-6f);
    gg.global_voxels = (int64_t)meta.Nx * meta.Ny * gg.global_Nz;
    if (gg.global_Nz < meta.Nz || gg.global_voxels <= 0) {
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

    fread(&m.use_45sym, 4, 1, f);
    fread(&m.sym_code, 4, 1, f);
    fread(&m.symmetry_order, 4, 1, f);
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

static std::vector<float> to_float_vector(const std::vector<double>& in) {
    std::vector<float> out(in.size());
    for (size_t i = 0; i < in.size(); i++) out[i] = (float)in[i];
    return out;
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
    double min_g = HUGE_VAL;
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

    double vmin = HUGE_VAL;
    double vmax = -HUGE_VAL;
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
    float z = origin_z_mm + (float)iz * voxel_mm;

    float r = 0.5f * diameter_mm;
    bool inside_xy = (x*x + y*y <= r*r);
    bool inside_z = false;
    float half_h = 0.5f * valid_height_mm;
    for (int b = 0; b < num_beds; ++b) {
        if (fabsf(z - bed_centers_mm[b]) <= half_h) {
            inside_z = true;
            break;
        }
    }
    if (!(inside_xy && inside_z)) a[idx] = 0.0f;
}

static int64_t count_multibed_roi_voxels(int Nx, int Ny, int global_Nz,
                                         float voxel_mm, float origin_z_mm,
                                         float diameter_mm, float valid_height_mm,
                                         float center_x_mm, float center_y_mm,
                                         const std::vector<BedInfo>& beds)
{
    double r = 0.5 * (double)diameter_mm;
    double r2 = r * r;
    double half_h = 0.5 * (double)valid_height_mm;
    int64_t cnt = 0;
    for (int iz = 0; iz < global_Nz; ++iz) {
        double z = (double)origin_z_mm + (double)iz * voxel_mm;
        bool inside_z = false;
        for (const BedInfo& bed : beds) {
            if (fabs(z - (double)bed.center_mm) <= half_h) {
                inside_z = true;
                break;
            }
        }
        if (!inside_z) continue;
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
    double half_h = 0.5 * (double)valid_height_mm;
    for (int iz = 0; iz < global_Nz; ++iz) {
        double z = (double)origin_z_mm + (double)iz * voxel_mm;
        bool inside_z = false;
        for (const BedInfo& bed : beds) {
            if (fabs(z - (double)bed.center_mm) <= half_h) {
                inside_z = true;
                break;
            }
        }
        for (int iy = 0; iy < Ny; ++iy) {
            double y = ((double)iy - 0.5 * (double)(Ny - 1)) * voxel_mm - center_y_mm;
            for (int ix = 0; ix < Nx; ++ix) {
                double x = ((double)ix - 0.5 * (double)(Nx - 1)) * voxel_mm - center_x_mm;
                if (!(inside_z && x*x + y*y <= r2)) {
                    int64_t idx = (int64_t)iz * Nx * Ny + (int64_t)iy * Nx + ix;
                    a[idx] = 0.0f;
                }
            }
        }
    }
}

__global__ void k_apply_cylinder_roi(float* a,
                                     int Nx, int Ny, int Nz,
                                     float voxel_mm,
                                     float diameter_mm, float height_mm,
                                     float center_x_mm, float center_y_mm, float center_z_mm,
                                     int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int iz = (int)(idx / ((int64_t)Nx * Ny));
    int rem = (int)(idx - (int64_t)iz * Nx * Ny);
    int iy = rem / Nx;
    int ix = rem - iy * Nx;

    float x = ((float)ix - 0.5f * (float)(Nx - 1)) * voxel_mm - center_x_mm;
    float y = ((float)iy - 0.5f * (float)(Ny - 1)) * voxel_mm - center_y_mm;
    float z = ((float)iz - 0.5f * (float)(Nz - 1)) * voxel_mm - center_z_mm;

    float r = 0.5f * diameter_mm;
    float h = 0.5f * height_mm;
    bool inside = (x*x + y*y <= r*r) && (fabsf(z) <= h);
    if (!inside) a[idx] = 0.0f;
}

static int64_t count_cylinder_roi_voxels(int Nx, int Ny, int Nz,
                                         float voxel_mm,
                                         float diameter_mm, float height_mm,
                                         const float center_mm[3])
{
    double r = 0.5 * (double)diameter_mm;
    double h = 0.5 * (double)height_mm;
    double r2 = r * r;
    int64_t cnt = 0;
    for (int iz = 0; iz < Nz; iz++) {
        double z = ((double)iz - 0.5 * (double)(Nz - 1)) * voxel_mm - center_mm[2];
        if (fabs(z) > h) continue;
        for (int iy = 0; iy < Ny; iy++) {
            double y = ((double)iy - 0.5 * (double)(Ny - 1)) * voxel_mm - center_mm[1];
            for (int ix = 0; ix < Nx; ix++) {
                double x = ((double)ix - 0.5 * (double)(Nx - 1)) * voxel_mm - center_mm[0];
                if (x*x + y*y <= r2) cnt++;
            }
        }
    }
    return cnt;
}

static void apply_cylinder_roi_host(float* a,
                                    int Nx, int Ny, int Nz,
                                    float voxel_mm,
                                    float diameter_mm, float height_mm,
                                    const float center_mm[3])
{
    double r = 0.5 * (double)diameter_mm;
    double h = 0.5 * (double)height_mm;
    double r2 = r * r;
    for (int iz = 0; iz < Nz; iz++) {
        double z = ((double)iz - 0.5 * (double)(Nz - 1)) * voxel_mm - center_mm[2];
        for (int iy = 0; iy < Ny; iy++) {
            double y = ((double)iy - 0.5 * (double)(Ny - 1)) * voxel_mm - center_mm[1];
            for (int ix = 0; ix < Nx; ix++) {
                double x = ((double)ix - 0.5 * (double)(Nx - 1)) * voxel_mm - center_mm[0];
                bool inside = (x*x + y*y <= r2) && (fabs(z) <= h);
                if (!inside) {
                    int64_t idx = (int64_t)iz * Nx * Ny + (int64_t)iy * Nx + ix;
                    a[idx] = 0.0f;
                }
            }
        }
    }
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
    if (!isfinite(g) || !isfinite(p) || !isfinite(yy) || g <= 0.0f || p < 0.0f) {
        ratio[r] = 0.0f;
        return;
    }

    float pred = g * yy;
    if (!isfinite(pred) || pred < 1e-10f) pred = 1e-10f;
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

__global__ void k_div_norm(float* a, const float* norm, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float d = norm[i];
        if (d < 1e-12f) d = 1e-12f;
        a[i] /= d;
    }
}

static int castor_round_host(float x) {
    int h = (int)floorf(x);
    if ((x - h) > 0.5f) h++;
    if (h < 1) h = 1;
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

static void gpu_conv3d_renorm(float* d_in, float* d_out, float* d_tmp,
                              const float* d_norm,
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
}

__global__ void k_real_to_complex(const float* r, cufftComplex* c, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { c[i].x = r[i]; c[i].y = 0.0f; }
}

__global__ void k_complex_to_real(const cufftComplex* c, float* r, int64_t n, float invn) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = c[i].x * invn;
        r[i] = (v > 0.0f) ? v : 0.0f;
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

static std::vector<float> build_butterworth_filter(int Nx, int Ny, int Nz,
                                                   float cutoff, int order)
{
    std::vector<float> h((size_t)Nx*Ny*Nz);
    float invsqrt3 = 1.0f / sqrtf(3.0f);
    for (int z = 0; z < Nz; z++) {
        int wz = (z + Nz/2) % Nz - Nz/2;
        for (int y = 0; y < Ny; y++) {
            int wy = (y + Ny/2) % Ny - Ny/2;
            for (int x = 0; x < Nx; x++) {
                int wx = (x + Nx/2) % Nx - Nx/2;
                float u = (float)wx / (Nx / 2.0f);
                float v = (float)wy / (Ny / 2.0f);
                float w = (float)wz / (Nz / 2.0f);
                float D = sqrtf(u*u + v*v + w*w) * invsqrt3;
                float r = D / cutoff;
                float rpow = 1.0f;
                for (int k = 0; k < 2*order; k++) rpow *= r;
                h[(int64_t)z*Nx*Ny + y*Nx + x] = 1.0f / (1.0f + rpow);
            }
        }
    }
    return h;
}

static bool should_apply_filter(int iter, int total_iters, const std::string& mode,
                                int period, int last_n)
{
    if (mode == "every") return true;
    if (mode == "final") return iter == total_iters;
    if (mode == "periodic") return (period > 0 && iter % period == 0) || iter == total_iters;
    if (mode == "last_n") return iter > total_iters - last_n;
    return false;
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
    float tv_sum = delta * delta;

    if (map_type == MAP_TV) {
        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0 && dz == 0) continue;
            if (!neigh26 && abs(dx) + abs(dy) + abs(dz) != 1) continue;
            int x = ix + dx, y = iy + dy, z = iz + dz;
            if (x < 0 || x >= Nx || y < 0 || y >= Ny || z < 0 || z >= Nz) continue;
            float w = rsqrtf((float)(dx*dx + dy*dy + dz*dz));
            float d = fi - f[(int64_t)z*Nx*Ny + y*Nx + x];
            tv_sum += w * d * d;
        }
    }
    float tv_den = sqrtf(tv_sum);
    if (tv_den < 1e-12f) tv_den = 1e-12f;

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
        } else if (map_type == MAP_TV) {
            g += w * d / tv_den;
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
              int bed_count, const std::vector<float>& G_full)
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
    printf("  --roi-height-mm F         valid axial FOV per bed in mm [21.6]; multi-bed uses interval union\n");
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
    printf("  --save-every N            save interval [5]\n");
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
    printf("  --butter-cutoff F         Butterworth cutoff, 0=off [off]\n");
    printf("  --butter-order N          Butterworth order [8]\n");
    printf("  --butter-mode every|final|periodic|last_n [final]\n");
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
        else if (a == "--butter-cutoff") {
            c.butter_cutoff = (float)atof(need("--butter-cutoff"));
            c.butter_enable = c.butter_cutoff > 0.0f;
        }
        else if (a == "--butter-order") c.butter_order = atoi(need("--butter-order"));
        else if (a == "--butter-mode") c.butter_mode = need("--butter-mode");
        else if (a == "--butter-period") c.butter_period = atoi(need("--butter-period"));
        else if (a == "--butter-last-n") c.butter_last_n = atoi(need("--butter-last-n"));
        else if (a == "--help" || a == "-h") { print_usage(); exit(0); }
        else { fprintf(stderr, "Unknown option: %s\n", a.c_str()); print_usage(); exit(1); }
    }
    if (c.matrix_dir.empty() || c.proj_path.empty()) { print_usage(); exit(1); }
    if (c.num_gpus < 1) c.num_gpus = 1;
    if (c.angle_batch < 0) c.angle_batch = 0;
    if (c.gpu_slab_cache < 0) c.gpu_slab_cache = 0;
    if (c.recon_angle_step < 1) c.recon_angle_step = 1;
    if (c.roi_diameter_mm <= 0.0f || c.roi_height_mm <= 0.0f) {
        fprintf(stderr, "Invalid cylinder ROI size: diameter=%.6g height=%.6g\n",
                c.roi_diameter_mm, c.roi_height_mm);
        exit(1);
    }
    if (c.sens_zsym_tol <= 0.0f || !std::isfinite(c.sens_zsym_tol)) c.sens_zsym_tol = 1e-4f;
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

static void broadcast_image_from_gpu0(float* d_src0, std::vector<GPUWorker>& workers, int64_t n) {
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
static int bedmerge_reconstruct_main(int argc, char** argv) {
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

    cufftHandle fft_plan = 0;
    cufftComplex* d_fft = nullptr;
    float* d_butter = nullptr;
    if (cfg.butter_enable) {
        std::vector<float> h_butter = build_butterworth_filter(
            gg.Nx, gg.Ny, gg.global_Nz,
            cfg.butter_cutoff, cfg.butter_order);
        CUDA_CHECK(cudaMalloc(&d_butter, (size_t)global_N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_butter, h_butter.data(),
                              (size_t)global_N * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_fft,
                              (size_t)global_N * sizeof(cufftComplex)));
        CUFFT_CHECK(cufftPlan3d(&fft_plan, gg.global_Nz, gg.Ny, gg.Nx,
                                CUFFT_C2C));
        printf("[8] Global Butterworth on: cutoff=%.3f order=%d mode=%s\n",
               cfg.butter_cutoff, cfg.butter_order,
               cfg.butter_mode.c_str());
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
        gpu_conv3d_renorm(d_global_sens, d_f_eff, d_conv_tmp1,
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
            gpu_conv3d_renorm(d_f, d_f_eff, d_conv_tmp1,
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
                                    cfg.recon_angle_step);
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
                                cfg.recon_angle_step);
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
            gpu_conv3d_renorm(d_global_bp, d_f_eff, d_conv_tmp1,
                              d_psf_norm,
                              gg.Nx, gg.Ny, gg.global_Nz,
                              psf_kx, psf_ky, psf_kz);
            CUDA_CHECK(cudaMemcpy(d_global_bp, d_f_eff,
                                  (size_t)global_N * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }

        float* map_ptr = nullptr;
        if (cfg.map_type != MAP_NONE && cfg.map_beta > 0.0f) {
            k_map_gradient<<<nb_global, BLK>>>(
                d_f, d_map_grad,
                gg.Nx, gg.Ny, gg.global_Nz,
                (int)cfg.map_type, cfg.map_neigh26,
                cfg.map_delta, cfg.map_huber_threshold,
                cfg.map_relative_gamma,
                cfg.map_relative_epsilon);
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
                gpu_conv3d_renorm(d_f, d_f_eff, d_conv_tmp1,
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
                                cfg.butter_period,
                                cfg.butter_last_n)) {
            k_real_to_complex<<<nb_global, BLK>>>(d_f, d_fft, global_N);
            CUFFT_CHECK(cufftExecC2C(fft_plan, d_fft, d_fft,
                                     CUFFT_FORWARD));
            k_butter_mul<<<nb_global, BLK>>>(d_fft, d_butter, global_N);
            CUFFT_CHECK(cufftExecC2C(fft_plan, d_fft, d_fft,
                                     CUFFT_INVERSE));
            k_complex_to_real<<<nb_global, BLK>>>(
                d_fft, d_f, global_N, 1.0f / (float)global_N);
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

        // Save only the requested interval and the final iteration. The
        // outer bedmerge wrapper can crop and stitch these checkpoints.
        bool do_save = (cfg.save_interval > 0 &&
                        iter % cfg.save_interval == 0) ||
                       (iter == cfg.num_iters);
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
            snprintf(fn, sizeof(fn), "%s/recon_iter%03d",
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

    CUDA_CHECK(cudaSetDevice(0));
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
    save_volume_outputs(cfg.output_dir + "/recon_final",
                        h_out.data(), gg.Nx, gg.Ny, gg.global_Nz,
                        gg.voxel_mm, gg.origin_z_mm,
                        cfg.save_raw1);

    if (fft_plan) cufftDestroy(fft_plan);
    if (d_fft) cudaFree(d_fft);
    if (d_butter) cudaFree(d_butter);

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
    double roi_height_mm = 21.6;
    int iterations = 50;
    int save_interval = 0; // 0=final only; positive value saves every N iterations.
    int gpus = 1;
    int angle_batch = 0;
    int gpu_slab_cache = 6;
    std::string data_mode = "measured";
    bool no_norm_factor = false;
    fs::path norm_factor;
    std::string norm_factor_mode = "projection";
    bool no_detector_mask = false;
    // 默认删除重建后用于裁剪的完整尺寸中间结果，只保留 mask 和拼接结果。
    bool keep_raw = false;
    bool dry_run = false;
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
    std::printf(
        "Usage: spect_recon_PSC_6module_24pin_YSO_bedmerge "
        "--proj-dir DIR --out DIR [options]\n\n"
        "The input directory must contain Bed1.bin and optionally Bed2.bin, ... .\n"
        "Each bed is reconstructed with the CUDA engine; stitching is CPU-side.\n\n"
        "Options:\n"
        "  --matrix-dir DIR          system matrix directory [MCmatrix_PSC]\n"
        "  --bed-step-mm F           bed-center spacing [16.8]\n"
        "  --mask-size Nx,Ny,Nz      centered output mask in voxels [100,100,72]\n"
        "  --gain-roi-diameter-mm F ROI diameter for scalar gain fitting [30]\n"
        "  --roi-diameter-mm F       reconstruction cylindrical ROI diameter [30]\n"
        "  --roi-height-mm F         reconstruction valid height per bed [21.6]\n"
        "  --iters N                  MLEM iterations [50]\n"
        "  --save-every N             save single-bed and stitched images every N iterations; 0=final only [0]\n"
        "  --gpus N                  CUDA GPUs [1]\n"
        "  --angle-batch N           compact-angle batch [0; resident CSR for bedmerge]\n"
        "  --gpu-slab-cache N        resident angle slabs [6]\n"
        "  --data-mode measured|simulation [measured]\n"
        "  --no-detector-mask        disable measured-data bad-detector mask\n"
        "  --keep-raw                keep full matrix-size intermediate bed volumes\n"
        "  --no-norm-factor          force g_i=1 and disable factor auto-search\n"
        "  --norm-factor FILE        explicit normalization factor\n"
        "  --norm-factor-mode MODE   auto|matrix|projection [projection]\n"
        "  --dry-run                 validate and print the CUDA configuration only\n"
        "  --help                    show this help\n");
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
        } else if (arg == "--iters") {
            options.iterations = parse_int(need_value(argc, argv, i, "--iters"), "--iters");
        } else if (arg == "--save-every") {
            options.save_interval = parse_nonnegative_int(
                need_value(argc, argv, i, "--save-every"), "--save-every");
        } else if (arg == "--gpus") {
            options.gpus = parse_int(need_value(argc, argv, i, "--gpus"), "--gpus");
        } else if (arg == "--angle-batch") {
            options.angle_batch = parse_int(need_value(argc, argv, i, "--angle-batch"), "--angle-batch");
        } else if (arg == "--gpu-slab-cache") {
            options.gpu_slab_cache = parse_int(
                need_value(argc, argv, i, "--gpu-slab-cache"), "--gpu-slab-cache");
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
        } else {
            fail("unknown option: " + arg);
        }
    }
    if (options.proj_dir.empty() || options.out.empty()) {
        fail("--proj-dir and --out are required");
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
    // Some sysmatrix generators keep the authoritative geometry only in
    // metadata.bin.  The CUDA reader already uses that binary record; mirror
    // the same values here so the crop/stitch wrapper remains compatible.
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

// 读取底层引擎生成的 NIfTI-1 成对 hdr/img，并检查尺寸、体素大小和 NaN/Inf。
static Volume read_nifti(const fs::path& directory,
                         const std::string& stem = "recon_final") {
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

static std::vector<int> discover_saved_iterations(const fs::path& directory) {
    std::regex pattern("^recon_iter([0-9]+)\\.hdr$");
    std::vector<int> iterations;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        std::smatch match;
        const std::string name = entry.path().filename().string();
        if (!std::regex_match(name, match, pattern)) continue;
        const int iteration = parse_int(match[1].str().c_str(), "saved iteration");
        if (!fs::is_regular_file(directory / ("recon_iter" + match[1].str() + ".img"))) {
            fail("saved iteration is missing its .img companion: " + name);
        }
        iterations.push_back(iteration);
    }
    std::sort(iterations.begin(), iterations.end());
    iterations.erase(std::unique(iterations.begin(), iterations.end()), iterations.end());
    return iterations;
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
                         const Volume& output, int mask_layers, int step_layers,
                         const std::vector<int>& saved_iterations) {
    std::ofstream report(path);
    report << "{\n"
           << "  \"method\": \"single_bed_reconstruction_center_mask_then_cosine_stitch\",\n"
           << "  \"projection_directory\": \"" << json_path(options.proj_dir) << "\",\n"
           << "  \"matrix_directory\": \"" << json_path(options.matrix_dir) << "\",\n"
           << "  \"bed_step_mm\": " << options.bed_step_mm << ",\n"
           << "  \"save_interval\": " << options.save_interval << ",\n"
           << "  \"saved_iterations\": [";
    for (size_t i = 0; i < saved_iterations.size(); ++i) {
        report << saved_iterations[i]
               << (i + 1 == saved_iterations.size() ? "" : ", ");
    }
    report << "],\n"
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
           << "  \"beds\": [\n";
    for (size_t i = 0; i < beds.size(); ++i) {
        report << "    {\"name\": \"Bed" << beds[i].number
               << "\", \"input\": \"" << json_path(beds[i].path)
               << "\", \"gain\": " << gains[i] << "}"
               << (i + 1 == beds.size() ? "\n" : ",\n");
    }
    report << "  ],\n"
           << "  \"outputs\": ["
           << (options.keep_raw ? "\"single_bed_raw/BedN/recon_final.img\", " : "")
           << "\"single_bed/BedN/recon_final.img\", "
           << "\"single_bed/BedN/recon_iterNNN.hdr/.img\", "
           << "\"stitched/recon_stitched_minimal.hdr\", \"stitched/recon_stitched_minimal.img\", "
           << "\"stitched/recon_stitched_iterNNN.hdr/.img\"]\n"
           << "}\n";
}

// 逐床调用 CUDA 引擎；完整尺寸结果先写入 raw_root，随后由主流程统一裁剪。
// 逐床调用底层 CUDA 引擎；完整尺寸结果只作为裁剪阶段的中间数据。
static int run_reconstruction(const BedMergeOptions& options,
                              const std::vector<BedInput>& beds,
                              const fs::path& raw_root) {
    for (const BedInput& bed : beds) {
        const fs::path bed_out = raw_root / ("Bed" + std::to_string(bed.number));
        std::vector<std::string> arguments = {
            "bedmerge-reconstruction", "--matrix-dir", options.matrix_dir.string(),
            "--proj", bed.path.string(), "--out", bed_out.string(),
            "--iters", std::to_string(options.iterations),
            "--save-every", std::to_string(options.save_interval),
            "--gpus", std::to_string(options.gpus),
            "--angle-batch", std::to_string(options.angle_batch),
            "--gpu-slab-cache", std::to_string(options.gpu_slab_cache),
            "--slab-prefetch-early", "--no-raw1",
            "--data-mode", options.data_mode,
            "--roi-diameter-mm", std::to_string(options.roi_diameter_mm),
            "--roi-height-mm", std::to_string(options.roi_height_mm),
        };
        if (options.no_detector_mask || options.data_mode == "simulation") {
            arguments.push_back("--no-detector-mask");
        }
        if (options.no_norm_factor) {
            arguments.push_back("--no-norm-factor");
        } else if (!options.norm_factor.empty()) {
            arguments.push_back("--norm-factor");
            arguments.push_back(options.norm_factor.string());
            arguments.push_back("--norm-factor-mode");
            arguments.push_back(options.norm_factor_mode);
        }
        std::vector<char*> argv;
        argv.reserve(arguments.size());
        for (std::string& argument : arguments) argv.push_back(argument.data());
        const int status = bedmerge_reconstruct_main(static_cast<int>(argv.size()), argv.data());
        if (status != 0) return status;
    }
    return 0;
}

// 主流程：发现床位、参数校验、重建、mask 裁剪、增益拟合和拼接。
int main(int argc, char** argv) {
    try {
        const BedMergeOptions options = parse_options(argc, argv);
        const std::vector<BedInput> beds = discover_beds(options.proj_dir);
        const double voxel_mm = read_metadata_value(options.matrix_dir / "metadata.txt", "voxel_size_mm");
        const int matrix_nz = static_cast<int>(read_metadata_value(options.matrix_dir / "metadata.txt", "Nz"));
        const int matrix_nx = static_cast<int>(read_metadata_value(options.matrix_dir / "metadata.txt", "Nx"));
        const int matrix_ny = static_cast<int>(read_metadata_value(options.matrix_dir / "metadata.txt", "Ny"));
        const int step_layers = static_cast<int>(std::llround(options.bed_step_mm / voxel_mm));
        if (options.mask_nx > matrix_nx || options.mask_ny > matrix_ny ||
            options.mask_nz > matrix_nz) {
            fail("mask size must not exceed system-matrix dimensions");
        }
        if (beds.size() > 1 && (step_layers <= 0 || options.mask_nz <= step_layers)) {
            fail("multi-bed mask Nz must be greater than bed-step in voxel units");
        }
        if (options.dry_run) {
            std::printf("single-bed reconstruction and mask stitch: %zu beds, mask=%dx%dx%d, step=%d layers, overlap=%d layers\n",
                        beds.size(), options.mask_nx, options.mask_ny, options.mask_nz,
                        step_layers, options.mask_nz - step_layers);
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
        if (status != 0) fail("single-bed CUDA reconstruction returned nonzero status");

        std::vector<Volume> volumes;
        volumes.reserve(beds.size());
        const fs::path first_raw_bed = raw_root / ("Bed" + std::to_string(beds.front().number));
        std::vector<int> saved_iterations;
        if (options.save_interval > 0) {
            saved_iterations = discover_saved_iterations(first_raw_bed);
        }
        for (const BedInput& bed : beds) {
            const fs::path raw_bed_dir = raw_root / ("Bed" + std::to_string(bed.number));
            const fs::path single_bed_dir = single_root / ("Bed" + std::to_string(bed.number));
            const Volume full = read_nifti(raw_bed_dir);
            Volume masked = crop_center_mask(full, options.mask_nx,
                                             options.mask_ny, options.mask_nz);
            write_nifti(single_bed_dir, masked, "recon_final");
            volumes.push_back(std::move(masked));

            if (options.save_interval > 0) {
                const std::vector<int> bed_iterations = discover_saved_iterations(raw_bed_dir);
                if (bed_iterations != saved_iterations) {
                    fail("saved iteration sets differ between beds; cannot stitch intermediate outputs");
                }
                for (int iteration : saved_iterations) {
                    char stem_buffer[32] = {};
                    std::snprintf(stem_buffer, sizeof(stem_buffer), "recon_iter%03d", iteration);
                    const Volume full_iteration = read_nifti(raw_bed_dir, stem_buffer);
                    Volume masked_iteration = crop_center_mask(
                        full_iteration, options.mask_nx, options.mask_ny, options.mask_nz);
                    write_nifti(single_bed_dir, masked_iteration, stem_buffer);
                }
            }
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
        for (const Volume& volume : volumes) {
            if (volume.nx != volumes[0].nx || volume.ny != volumes[0].ny ||
                volume.nz != volumes[0].nz || std::abs(volume.voxel_mm - volumes[0].voxel_mm) > 1e-6) {
                fail("single-bed output geometries are inconsistent");
            }
        }
        const int actual_step_layers = static_cast<int>(std::llround(options.bed_step_mm / volumes[0].voxel_mm));
        if (volumes.size() > 1 && volumes[0].nz <= actual_step_layers) {
            fail("multi-bed mask Nz must be greater than bed-step in reconstructed voxel units");
        }
        const std::vector<double> gains = fit_gains(
            volumes, actual_step_layers,
            volumes[0].voxel_mm, options.gain_roi_diameter_mm);
        const Volume stitched = stitch_volumes(
            volumes, gains, options.bed_step_mm);
        const fs::path stitch_dir = options.out / "stitched";
        write_nifti(stitch_dir, stitched);
        for (int iteration : saved_iterations) {
            std::vector<Volume> iteration_volumes;
            iteration_volumes.reserve(beds.size());
            char stem_buffer[32] = {};
            std::snprintf(stem_buffer, sizeof(stem_buffer), "recon_iter%03d", iteration);
            for (const BedInput& bed : beds) {
                iteration_volumes.push_back(read_nifti(
                    single_root / ("Bed" + std::to_string(bed.number)), stem_buffer));
            }
            for (const Volume& volume : iteration_volumes) {
                if (volume.nx != volumes[0].nx || volume.ny != volumes[0].ny ||
                    volume.nz != volumes[0].nz ||
                    std::abs(volume.voxel_mm - volumes[0].voxel_mm) > 1e-6) {
                    fail("intermediate single-bed output geometries are inconsistent");
                }
            }
            const std::vector<double> iteration_gains = fit_gains(
                iteration_volumes, actual_step_layers,
                iteration_volumes[0].voxel_mm, options.gain_roi_diameter_mm);
            const Volume stitched_iteration = stitch_volumes(
                iteration_volumes, iteration_gains, options.bed_step_mm);
            std::string stitched_stem = "recon_stitched_iter" +
                std::string(stem_buffer + std::strlen("recon_iter"));
            write_nifti(stitch_dir, stitched_iteration, stitched_stem.c_str());
        }
        write_report(options.out / "bedmerge_report.json", options, beds, gains,
                     stitched, volumes[0].nz, actual_step_layers, saved_iterations);
        // 所有床位均已完成后才释放驻留的系统矩阵，确保床位循环期间只加载一次。
        g_bedmerge_matrix_cache.clear();
        std::printf("[bedmerge] complete: %s\n", options.out.string().c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "bedmerge error: %s\n", error.what());
        return 1;
    }
}
