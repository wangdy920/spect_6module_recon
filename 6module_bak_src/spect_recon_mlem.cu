/*
 * ============================================================================
 *  SPECT MLEM/MAP-EM Reconstruction Engine — v4 Compact-Batched CUDA
 * ============================================================================
 *
 *  设计目标
 *  --------
 *  这个版本专门针对“探测器较多时 V1 完整展开 CSR 过大、V3 逐角度小 SpMV
 *  调度开销过大”的问题。
 *
 *  V4 的核心思想：
 *    1) 直接读取系统矩阵生成脚本输出的 compact chunk_*.bin 文件；
 *    2) 不再把 90°/45°/Z 对称展开成 full CSR；
 *    3) 但也不再像 V3 那样每个角度单独做小 SpMV；
 *    4) 而是把每张 GPU 负责的 compact 角度范围合并成一个较大的 CSR，
 *       每个 symmetry / z-mirror 只做一次大 SpMV。
 *
 *  对应算子
 *  --------
 *    Forward : y = Wc * T_s,z(f)
 *    Ratio   : ratio_compact(row) = P_full(full_angle, full_det_z, det_x) / y(row)
 *    Backward: bp += T_s,z^T( Wc^T * ratio_compact )
 *
 *  相比 V1:
 *    · 不展开 full CSR，显存约减少 symmetry_factor 倍；
 *    · 避免探测器多时 expanded CSR 带宽压力过大。
 *
 *  相比 V3:
 *    · 从 “2 × compact_angles × sym × z” 次小 SpMV
 *      降为 “2 × sym × z” 次大 SpMV；
 *    · cuSPARSE descriptor / buffer 常驻；
 *    · chunk 常驻 GPU；
 *    · 图像旋转从每角度一次降为每 symmetry/z 一次；
 *    · 多 GPU 图像广播和反投影归约优先使用 cudaMemcpyPeer。
 *
 *  支持的系统矩阵文件
 *  ------------------
 *  来自 spect_sysmat_v730_roi_coo_annotated.cu 的输出目录：
 *    metadata.bin
 *    metadata.txt              可选，仅用于人工检查
 *    chunk_000.bin, ...
 *
 *  chunk 二进制格式：
 *    int32 rt                  本 chunk 行数 = angle_count * compact_rays_per_angle
 *    int32 angle_offset         compact angle 起点，0-based
 *    int32 angle_count          compact angle 数
 *    for row in 0..rt-1:
 *        int32 n
 *        int32 col[n]           0-based voxel index
 *        float weight[n]
 *
 *  投影文件格式
 *  ------------
 *    int32 num_angles
 *    int32 rays_per_angle       = det_nx * det_nz
 *    double data[num_angles * rays_per_angle]
 *
 *  编译示例
 *  --------
 *    nvcc -O3 -std=c++14 -arch=sm_86 \
 *      spect_recon_mlem_v4_compact_batched.cu -o spect_recon_v4 \
 *      -lcusparse -lcufft -lm
 *
 *  运行示例
 *  --------
 *    ./spect_recon_v4 \
 *      --matrix-dir /path/to/matrix_dir \
 *      --proj /path/to/projection.bin \
 *      --out recon_v4_out \
 *      --iters 50 --gpus 2 \
 *      --conv-mode psf --psf-fwhm 1.5,1.5,1.5 --psf-sigmas 3 \
 *      --butter-cutoff 0.45 --butter-order 8 --butter-mode every
 *
 *  重要说明
 *  --------
 *  1) 45° 对称的 full angle -> base angle 使用 v7.3 CUDA 系统矩阵的规则：
 *       compact_angles = full_angles / 8 + 1
 *     即基础角度覆盖 [0°, 45°] 的闭区间。
 *  2) 如果你后续在系统矩阵生成脚本中额外输出 symmetry_info.bin 中的
 *     angle_to_base_k / angle_sym_type，本文件可进一步改为直接读取查表。
 *     当前版本根据 full_num_angles 和 compact_num_angles 在 GPU 端实时计算。
 *  3) MAP 的 GPU 实现覆盖 quadratic / TV / huber / relative_difference /
 *     logcosh / hyperbolic。MRP 中值滤波 MAP 没有在此 CUDA 版中实现。
 * ============================================================================
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cufft.h>

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
#include <thread>
#include <numeric>
#include <sstream>

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

/* ============================================================================
 *  枚举和配置
 * ============================================================================ */

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

struct ReconConfig {
    std::string matrix_dir;
    std::string proj_path;
    std::string output_dir = "recon_v4_output";

    int num_iters = 50;
    int save_interval = 5;
    int num_gpus = 1;

    /* symmetry: auto / none / zonly / 180 / z180 / 90 / 45 / 45z */
    std::string sym_override = "auto";

    /* CASToR 风格卷积模式：none / psf / post / sieve */
    std::string conv_mode = "none";
    float psf_fwhm_mm[3] = {1.5f, 1.5f, 1.5f};
    float psf_nb_sigmas = 3.0f;

    /* post/sieve 更新后卷积。CUDA 版使用 separable Gaussian。 */
    std::string post_schedule = "constant";    /* constant / linear / exponential */
    float post_fwhm_mm = 0.45f;
    float post_fwhm_start_mm = 0.60f;
    float post_fwhm_end_mm = 0.15f;
    int post_start_iter = 1;
    int post_apply_every = 1;

    /* MAP-EM */
    MapType map_type = MAP_NONE;
    float map_beta = 0.0f;
    int map_neigh26 = 1;        /* 1=3d_26, 0=3d_6 */
    float map_delta = 1e-6f;    /* TV/logcosh/hyperbolic */
    float map_huber_threshold = 0.01f;
    float map_relative_gamma = 2.0f;
    float map_relative_epsilon = 1e-6f;

    /* 独立后处理 */
    bool positivity = true;
    bool butter_enable = false;
    float butter_cutoff = 0.45f;
    int butter_order = 8;
    std::string butter_mode = "final";          /* every / final / periodic / last_n */
    int butter_period = 5;
    int butter_last_n = 10;

    /* 如果投影角度已经抽样，可通过 angle-step 与 MATLAB 的 recon_angle_step 对齐。
       注意：当前 compact-batched 路径默认推荐 full angle 全部使用；抽样时会在 ratio kernel 中
       自动跳过未选 full angle。 */
    int recon_angle_step = 1;
};

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

    /* v7.3 ROI-COO 新增字段；老文件没有时为默认值。 */
    int use_45sym = 0;
    int sym_code = 0;
    int symmetry_order = 1;
    int roi_radius = 0;
    int roi_side = 0;
};

struct ChunkInfo {
    std::string path;
    int num_rows = 0;
    int angle_offset = 0;
    int angle_count = 0;
};

struct ChunkCSR {
    int num_rows = 0;
    int angle_offset = 0;
    int angle_count = 0;
    std::vector<int64_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<float> values;
};

struct HostCSR64 {
    int64_t num_rows = 0;
    int64_t num_cols = 0;
    int64_t nnz = 0;
    int start_ck = 0;
    int end_ck = 0;
    std::vector<int64_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<float> values;
};

/* ============================================================================
 *  简单工具函数
 * ============================================================================ */

static bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}
static bool ends_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(s.size()-p.size(), p.size(), p) == 0;
}

static void make_dir(const std::string& d) {
    if (!d.empty()) MKDIR(d.c_str());
}

static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep)) out.push_back(item);
    return out;
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

    /* v7.3 的 metadata 有 use45/sym_code；老 metadata 只有 use90/use_zsym。 */
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

/* ============================================================================
 *  metadata/chunk/projection 读取
 * ============================================================================ */

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

    /* 新字段可选：老文件读不到也正常。 */
    fread(&m.use_45sym, 4, 1, f);
    fread(&m.sym_code, 4, 1, f);
    fread(&m.symmetry_order, 4, 1, f);
    fread(&m.roi_radius, 4, 1, f);
    fread(&m.roi_side, 4, 1, f);
    fclose(f);

    if (m.det_nz <= 0) {
        /* 极老 metadata 没有 det_nz 时，退回估计值。 */
        m.det_nz = m.use_zsym ? (2 * m.compact_det_z) : m.compact_det_z;
    }
    if (m.compact_rpa <= 0) m.compact_rpa = m.det_nx * m.compact_det_z;
    if (m.total_voxels <= 0) m.total_voxels = m.Nx * m.Ny * m.Nz;
    return m;
}

static std::vector<ChunkInfo> scan_chunks(const std::string& dir) {
    std::vector<std::string> paths;
#ifdef _WIN32
    std::string pattern = dir + "\\chunk_*.bin";
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Cannot open matrix dir or no chunks: %s\n", dir.c_str());
        exit(1);
    }
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            paths.push_back(dir + "/" + fd.cFileName);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR* d = opendir(dir.c_str());
    if (!d) { fprintf(stderr, "Cannot open matrix dir: %s\n", dir.c_str()); exit(1); }
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        std::string name(ent->d_name);
        if (starts_with(name, "chunk_") && ends_with(name, ".bin")) {
            paths.push_back(dir + "/" + name);
        }
    }
    closedir(d);
#endif
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) { fprintf(stderr, "No chunk_*.bin found in %s\n", dir.c_str()); exit(1); }

    std::vector<ChunkInfo> infos;
    for (const auto& p : paths) {
        FILE* f = fopen(p.c_str(), "rb");
        if (!f) continue;
        ChunkInfo ci;
        ci.path = p;
        IO_CHECK(fread(&ci.num_rows, 4, 1, f), 1);
        IO_CHECK(fread(&ci.angle_offset, 4, 1, f), 1);
        IO_CHECK(fread(&ci.angle_count, 4, 1, f), 1);
        fclose(f);
        infos.push_back(ci);
    }
    std::sort(infos.begin(), infos.end(), [](const ChunkInfo& a, const ChunkInfo& b){
        return a.angle_offset < b.angle_offset;
    });
    return infos;
}

static ChunkCSR load_chunk_csr(const std::string& path) {
    ChunkCSR csr;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open chunk: %s\n", path.c_str()); exit(1); }

    IO_CHECK(fread(&csr.num_rows, 4, 1, f), 1);
    IO_CHECK(fread(&csr.angle_offset, 4, 1, f), 1);
    IO_CHECK(fread(&csr.angle_count, 4, 1, f), 1);

    long payload_pos = ftell(f);
    csr.row_ptr.resize((size_t)csr.num_rows + 1, 0);
    int64_t total_nnz = 0;
    for (int r = 0; r < csr.num_rows; r++) {
        int32_t n = 0;
        IO_CHECK(fread(&n, 4, 1, f), 1);
        csr.row_ptr[r] = total_nnz;
        total_nnz += (int64_t)n;
        if (n > 0) fseek(f, (long)n * 8L, SEEK_CUR);
    }
    csr.row_ptr[csr.num_rows] = total_nnz;
    csr.col_idx.resize((size_t)total_nnz);
    csr.values.resize((size_t)total_nnz);

    fseek(f, payload_pos, SEEK_SET);
    for (int r = 0; r < csr.num_rows; r++) {
        int32_t n = 0;
        IO_CHECK(fread(&n, 4, 1, f), 1);
        int64_t off = csr.row_ptr[r];
        if (n > 0) {
            IO_CHECK(fread(csr.col_idx.data() + off, 4, n, f), n);
            IO_CHECK(fread(csr.values.data()  + off, 4, n, f), n);
        }
    }
    fclose(f);
    return csr;
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

    std::vector<std::vector<int32_t>> row_cols;
    std::vector<std::vector<float>> row_vals;
    row_cols.resize((size_t)out.num_rows);
    row_vals.resize((size_t)out.num_rows);

    for (const auto& ci : chunks) {
        int ch_start = ci.angle_offset;
        int ch_end = ci.angle_offset + ci.angle_count;
        int a0 = std::max(start_ck, ch_start);
        int a1 = std::min(end_ck, ch_end);
        if (a0 >= a1) continue;

        ChunkCSR ch = load_chunk_csr(ci.path);
        for (int ck = a0; ck < a1; ck++) {
            int local_angle_in_chunk = ck - ch.angle_offset;
            int chunk_row0 = local_angle_in_chunk * compact_rpa;
            int worker_row0 = (ck - start_ck) * compact_rpa;
            for (int rr = 0; rr < compact_rpa; rr++) {
                int cr = chunk_row0 + rr;
                int wr = worker_row0 + rr;
                int64_t ps = ch.row_ptr[cr];
                int64_t pe = ch.row_ptr[cr + 1];
                int64_t n = pe - ps;
                if (n <= 0) continue;
                row_cols[wr].insert(row_cols[wr].end(), ch.col_idx.begin() + ps, ch.col_idx.begin() + pe);
                row_vals[wr].insert(row_vals[wr].end(), ch.values.begin() + ps, ch.values.begin() + pe);
            }
        }
    }

    int64_t nnz = 0;
    for (int64_t r = 0; r < out.num_rows; r++) {
        out.row_ptr[(size_t)r] = nnz;
        nnz += (int64_t)row_cols[(size_t)r].size();
    }
    out.row_ptr[(size_t)out.num_rows] = nnz;
    out.nnz = nnz;
    out.col_idx.resize((size_t)nnz);
    out.values.resize((size_t)nnz);
    for (int64_t r = 0; r < out.num_rows; r++) {
        int64_t ps = out.row_ptr[(size_t)r];
        int64_t n = (int64_t)row_cols[(size_t)r].size();
        if (n > 0) {
            memcpy(out.col_idx.data() + ps, row_cols[(size_t)r].data(), (size_t)n * sizeof(int32_t));
            memcpy(out.values.data()  + ps, row_vals[(size_t)r].data(), (size_t)n * sizeof(float));
        }
    }
    return out;
}

static std::vector<float> read_projection_bin(const std::string& path, int* na, int* rpa) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open projection: %s\n", path.c_str()); exit(1); }
    IO_CHECK(fread(na, 4, 1, f), 1);
    IO_CHECK(fread(rpa, 4, 1, f), 1);
    int64_t total = (int64_t)(*na) * (*rpa);
    std::vector<double> tmp((size_t)total);
    IO_CHECK(fread(tmp.data(), sizeof(double), (size_t)total, f), (size_t)total);
    fclose(f);
    std::vector<float> out((size_t)total);
    for (int64_t i = 0; i < total; i++) out[(size_t)i] = (float)tmp[(size_t)i];
    return out;
}

static void save_volume_raw(const std::string& path, const float* data,
                            int Nx, int Ny, int Nz, float voxel_size)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); return; }
    int magic = 0x52415731; /* RAW1 */
    IO_CHECK(fwrite(&Nx, 4, 1, f), 1);
    IO_CHECK(fwrite(&Ny, 4, 1, f), 1);
    IO_CHECK(fwrite(&Nz, 4, 1, f), 1);
    IO_CHECK(fwrite(&magic, 4, 1, f), 1);
    IO_CHECK(fwrite(&voxel_size, 4, 1, f), 1);
    IO_CHECK(fwrite(data, sizeof(float), (size_t)Nx*Ny*Nz, f), (size_t)Nx*Ny*Nz);
    fclose(f);
}

/* ============================================================================
 *  CUDA kernels：通用逐元素操作
 * ============================================================================ */

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

__global__ void k_mlem_update(float* f, const float* bp, const float* sens,
                              const float* map_grad, float beta,
                              int64_t n)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float denom = sens[i];
        if (map_grad && beta > 0.0f) denom += beta * map_grad[i];
        if (denom < 1e-12f) denom = 1e-12f;
        float v = f[i] * bp[i] / denom;
        f[i] = (v > 0.0f) ? v : 0.0f;
    }
}

/* ============================================================================
 *  CUDA kernels：对称坐标变换
 * ============================================================================
 *
 *  注意：这里的变换作用于图像坐标。kern_transform_fwd 做的是：
 *      fout[p] = fin[g(p)]
 *  这样 Wc * fout 等价于使用变换后的图像做 compact 投影。
 *
 *  反投影时使用同一个 g 将 bp_rot 累加回原图像：
 *      bp_orig[g(p)] += bp_rot[p]
 *  对这里的 D4/旋转/翻转来说 g 是双射；只要同一个 kernel 内每个 p 的 g(p)
 *  唯一，就不需要 atomicAdd。不同 symmetry/z 的 kernel 串行发射，因此也不会并发写。
 */

__device__ __forceinline__ void map_xy_90(int ix, int iy, int Nx, int Ny,
                                          int sym, int& ox, int& oy)
{
    /* sym: 0=I, 1=CCW90, 2=180, 3=CW90 */
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
    /*
     * D4 八种变换，对应 45°基础扇区的 8 个 symmetry sector：
     *   0: identity
     *   1: reflect x=y
     *   2: rotate 90° CCW
     *   3: flip X
     *   4: rotate 180°
     *   5: reflect anti-diagonal
     *   6: rotate 270° CCW
     *   7: flip Y
     *
     * 这里要求 Nx==Ny。若体素横向不是正方形，45°/90°旋转对称本身就不严格成立。
     */
    switch (sym) {
        case 1: ox = iy;       oy = ix;       break;
        case 2: ox = N-1-iy;   oy = ix;       break;
        case 3: ox = N-1-ix;   oy = iy;       break;
        case 4: ox = N-1-ix;   oy = N-1-iy;   break;
        case 5: ox = N-1-iy;   oy = N-1-ix;   break;
        case 6: ox = iy;       oy = N-1-ix;   break;
        case 7: ox = ix;       oy = N-1-iy;   break;
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

/* ============================================================================
 *  CUDA kernels：full projection 到 compact rows 的映射
 * ============================================================================ */

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
        /* compact_num_angles = Q + 1, base angle covers 0..45 deg inclusive. */
        if ((sym_id & 1) == 0) {
            /* sectors 0,2,4,6: local angle grows from 0 to <45, ck=0..Q-1 */
            if (ck >= Q) { *valid = 0; return 0; }
            return sym_id * Q + ck;
        } else {
            /* sectors 1,3,5,7: local angle folds back from 45 to >0, ck=Q..1 */
            if (ck <= 0) { *valid = 0; return 0; }
            return sym_id * Q + (Q - ck);
        }
    }
    return ck;
}

__global__ void k_calc_ratio_batched(const float* __restrict__ y,
                                     float* __restrict__ ratio,
                                     const float* __restrict__ P_full,
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
        /* 奇数 det_nz 时，compact 中心层不能在 mirror 分支重复使用。 */
        if (iz_full < compact_det_z) {
            ratio[r] = 0.0f;
            return;
        }
    }
    if (iz_full < 0 || iz_full >= det_nz) {
        ratio[r] = 0.0f;
        return;
    }

    float q = y[r];
    if (q < 1e-10f) q = 1e-10f;
    int64_t full_idx = (int64_t)fa * full_rpa + iz_full * det_nx + ix;
    ratio[r] = P_full[full_idx] / q;
}

__global__ void k_make_unit_ratio_batched(float* __restrict__ ratio,
                                          int local_rows,
                                          int worker_start_ck,
                                          int compact_rpa,
                                          int det_nx,
                                          int compact_det_z,
                                          int det_nz,
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
    int izc = rr / det_nx;

    int valid_angle = 1;
    int fa = full_angle_from_compact(ck, sym_mode, sym_id, compact_num_angles,
                                     full_num_angles, &valid_angle);
    if (!valid_angle || fa < 0 || fa >= full_num_angles) { ratio[r] = 0.0f; return; }
    if (recon_angle_step > 1 && (fa % recon_angle_step) != 0) { ratio[r] = 0.0f; return; }

    if (z_mirror) {
        int iz_full = det_nz - 1 - izc;
        if (iz_full < compact_det_z || iz_full >= det_nz) { ratio[r] = 0.0f; return; }
    }
    ratio[r] = 1.0f;
}

/* ============================================================================
 *  CUDA kernels：PSF / post Gaussian 卷积，CASToR erf 积分核 + 边界归一化
 * ============================================================================ */

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

/* ============================================================================
 *  CUDA kernels：Butterworth FFT 后处理
 * ============================================================================ */

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

/* ============================================================================
 *  CUDA kernels：MAP 梯度
 * ============================================================================ */

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

    /* TV 需要先计算 denom。为了简单和鲁棒，这里两遍循环。 */
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

/* ============================================================================
 *  GPU Worker：每张卡持有一个 compact CSR 子矩阵和持久 cuSPARSE 对象
 * ============================================================================ */

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
    int32_t* d_row = nullptr;
    int32_t* d_col = nullptr;
    float* d_val = nullptr;

    float* d_img_in = nullptr;
    float* d_img_rot = nullptr;
    float* d_y = nullptr;
    float* d_ratio = nullptr;
    float* d_bp_rot = nullptr;
    float* d_bp_part = nullptr;
    float* d_sens_part = nullptr;
    float* d_P_full = nullptr;

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t mat = nullptr;
    cusparseDnVecDescr_t vec_img = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    cusparseDnVecDescr_t vec_ratio = nullptr;
    cusparseDnVecDescr_t vec_bp = nullptr;
    void* d_spmv_buf = nullptr;
    size_t spmv_buf_size = 0;

    void init(int device_id, const HostCSR64& hcsr, const MatrixMeta& meta,
              const std::vector<float>& P_full)
    {
        dev = device_id;
        start_ck = hcsr.start_ck;
        end_ck = hcsr.end_ck;
        local_rows = (int)hcsr.num_rows;
        nnz = hcsr.nnz;
        if (hcsr.num_rows > 2147483647LL || hcsr.nnz > 2147483647LL) {
            fprintf(stderr,
                    "Local compact CSR is too large for 32-bit cuSPARSE indices: rows=%" PRId64 " nnz=%" PRId64 "\n",
                    hcsr.num_rows, hcsr.nnz);
            exit(1);
        }
        Nx = meta.Nx; Ny = meta.Ny; Nz = meta.Nz;
        N2 = meta.total_voxels;
        compact_rpa = meta.compact_rpa;
        det_nx = meta.det_nx;
        det_nz = meta.det_nz;
        compact_det_z = meta.compact_det_z;

        CUDA_CHECK(cudaSetDevice(dev));
        CUSPARSE_CHECK(cusparseCreate(&handle));

        std::vector<int32_t> row32((size_t)local_rows + 1);
        for (int i = 0; i <= local_rows; i++) row32[(size_t)i] = (int32_t)hcsr.row_ptr[(size_t)i];

        CUDA_CHECK(cudaMalloc(&d_row, ((size_t)local_rows + 1) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_col, (size_t)nnz * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_val, (size_t)nnz * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_row, row32.data(), ((size_t)local_rows + 1)*sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col, hcsr.col_idx.data(), (size_t)nnz*sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_val, hcsr.values.data(), (size_t)nnz*sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_img_in,    (size_t)N2*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_img_rot,   (size_t)N2*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bp_rot,    (size_t)N2*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bp_part,   (size_t)N2*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sens_part, (size_t)N2*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y,         (size_t)local_rows*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ratio,     (size_t)local_rows*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_P_full,    P_full.size()*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_P_full, P_full.data(), P_full.size()*sizeof(float), cudaMemcpyHostToDevice));

        CUSPARSE_CHECK(cusparseCreateCsr(&mat,
            local_rows, N2, nnz,
            d_row, d_col, d_val,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_img,   N2,         d_img_rot, CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y,     local_rows, d_y,       CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_ratio, local_rows, d_ratio,   CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_bp,    N2,         d_bp_rot,  CUDA_R_32F));

        float alpha = 1.0f, beta = 0.0f;
        size_t b1 = 0, b2 = 0;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat, vec_img, &beta, vec_y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &b1));
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, mat, vec_ratio, &beta, vec_bp, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &b2));
        spmv_buf_size = std::max(b1, b2);
        if (spmv_buf_size > 0) CUDA_CHECK(cudaMalloc(&d_spmv_buf, spmv_buf_size));

        printf("  GPU%d compact angles [%d,%d), rows=%d, nnz=%.2fM\n",
               dev, start_ck, end_ck, local_rows, (double)nnz/1e6);
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

    void destroy() {
        CUDA_CHECK(cudaSetDevice(dev));
        if (vec_bp)    cusparseDestroyDnVec(vec_bp);
        if (vec_ratio) cusparseDestroyDnVec(vec_ratio);
        if (vec_y)     cusparseDestroyDnVec(vec_y);
        if (vec_img)   cusparseDestroyDnVec(vec_img);
        if (mat)       cusparseDestroySpMat(mat);
        if (handle)    cusparseDestroy(handle);
        if (d_spmv_buf) cudaFree(d_spmv_buf);
        if (d_P_full) cudaFree(d_P_full);
        if (d_ratio) cudaFree(d_ratio);
        if (d_y) cudaFree(d_y);
        if (d_sens_part) cudaFree(d_sens_part);
        if (d_bp_part) cudaFree(d_bp_part);
        if (d_bp_rot) cudaFree(d_bp_rot);
        if (d_img_rot) cudaFree(d_img_rot);
        if (d_img_in) cudaFree(d_img_in);
        if (d_val) cudaFree(d_val);
        if (d_col) cudaFree(d_col);
        if (d_row) cudaFree(d_row);
    }
};

/* ============================================================================
 *  参数解析
 * ============================================================================ */

static void print_usage() {
    printf("Usage:\n");
    printf("  spect_recon_mlem_v1_compact_batched --matrix-dir DIR --proj projection.bin [options]\n\n");
    printf("Required:\n");
    printf("  --matrix-dir DIR          directory containing metadata.bin and chunk_*.bin\n");
    printf("  --proj FILE               projection binary: int32 na, int32 rpa, double data[]\n\n");
    printf("Options:\n");
    printf("  --out DIR                 output directory [recon_v4_output]\n");
    printf("  --iters N                 iterations [50]\n");
    printf("  --gpus N                  number of GPUs [1]\n");
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
        else if (a == "--out") c.output_dir = need("--out");
        else if (a == "--iters") c.num_iters = atoi(need("--iters"));
        else if (a == "--gpus") c.num_gpus = atoi(need("--gpus"));
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
    if (c.recon_angle_step < 1) c.recon_angle_step = 1;
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

/* ============================================================================
 *  多 GPU peer access 与广播/归约
 * ============================================================================ */

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

/* ============================================================================
 *  主程序
 * ============================================================================ */

int main(int argc, char** argv) {
    ReconConfig cfg = parse_args(argc, argv);
    make_dir(cfg.output_dir);

    printf("\n==============================================================\n");
    printf("  SPECT MLEM/MAP-EM v1 Matrix Compact-Batched CUDA\n");
    printf("==============================================================\n");

    MatrixMeta meta = read_metadata(cfg.matrix_dir + "/metadata.bin");
    SymMode sym_mode = parse_sym_mode(cfg.sym_override, meta);
    bool use_z_branches = (mode_uses_z(sym_mode) || meta.use_zsym != 0) && (meta.compact_det_z < meta.det_nz);
    int sym_count = xy_sym_count(sym_mode);
    int z_count = use_z_branches ? 2 : 1;
    int64_t N2 = meta.total_voxels;
    int full_rpa = meta.det_nx * meta.det_nz;

    if ((sym_mode == SYM_45 || sym_mode == SYM_45Z || sym_mode == SYM_90) && meta.Nx != meta.Ny) {
        fprintf(stderr, "45/90 symmetry requires Nx==Ny, got Nx=%d Ny=%d\n", meta.Nx, meta.Ny);
        exit(1);
    }
    if ((sym_mode == SYM_45 || sym_mode == SYM_45Z) && meta.compact_num_angles != meta.full_num_angles/8 + 1) {
        fprintf(stderr, "Warning: 45 symmetry expects compact_angles=full/8+1, got compact=%d full=%d\n",
                meta.compact_num_angles, meta.full_num_angles);
    }

    printf("[1] Metadata\n");
    printf("  image:       %d x %d x %d = %d voxels, voxel=%.4f mm\n",
           meta.Nx, meta.Ny, meta.Nz, meta.total_voxels, meta.voxel_size);
    printf("  detector:    det_nx=%d det_nz=%d compact_det_z=%d\n",
           meta.det_nx, meta.det_nz, meta.compact_det_z);
    printf("  angles:      full=%d compact=%d\n", meta.full_num_angles, meta.compact_num_angles);
    printf("  rows:        compact_rpa=%d total_rows=%d\n", meta.compact_rpa, meta.total_rows);
    printf("  symmetry:    %s, metadata_zsym=%d, xy branches=%d, z branches=%d\n", sym_name(sym_mode), meta.use_zsym, sym_count, z_count);

    int proj_na = 0, proj_rpa = 0;
    std::vector<float> P_full = read_projection_bin(cfg.proj_path, &proj_na, &proj_rpa);
    if (proj_na != meta.full_num_angles || proj_rpa != full_rpa) {
        fprintf(stderr, "Projection shape mismatch: proj=(%d,%d), expected=(%d,%d)\n",
                proj_na, proj_rpa, meta.full_num_angles, full_rpa);
        exit(1);
    }
    double p_sum = 0.0;
    for (float v : P_full) p_sum += v;
    printf("[2] Projection: angles=%d full_rpa=%d counts=%.6e\n", proj_na, proj_rpa, p_sum);

    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0) { fprintf(stderr, "No CUDA device found\n"); exit(1); }
    cfg.num_gpus = std::min(cfg.num_gpus, dev_count);
    enable_peer_access(cfg.num_gpus);

    std::vector<ChunkInfo> chunks = scan_chunks(cfg.matrix_dir);
    printf("[3] Found %zu chunks\n", chunks.size());

    printf("[4] Build compact CSR per GPU\n");
    std::vector<HostCSR64> hcsrs(cfg.num_gpus);
    for (int g = 0; g < cfg.num_gpus; g++) {
        int start_ck = g * meta.compact_num_angles / cfg.num_gpus;
        int end_ck   = (g + 1) * meta.compact_num_angles / cfg.num_gpus;
        auto t0 = std::chrono::high_resolution_clock::now();
        hcsrs[g] = build_worker_csr(chunks, start_ck, end_ck, meta.compact_rpa, meta.total_voxels);
        double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        printf("  host CSR GPU%d angles [%d,%d): rows=%" PRId64 " nnz=%.2fM, %.2fs\n",
               g, start_ck, end_ck, hcsrs[g].num_rows, (double)hcsrs[g].nnz/1e6, sec);
    }

    printf("[5] Upload CSR and allocate GPU buffers\n");
    std::vector<GPUWorker> workers(cfg.num_gpus);
    for (int g = 0; g < cfg.num_gpus; g++) {
        workers[g].init(g, hcsrs[g], meta, P_full);
    }
    hcsrs.clear(); hcsrs.shrink_to_fit();

    /* GPU0 全局图像、灵敏度、反投影和临时缓存 */
    CUDA_CHECK(cudaSetDevice(0));
    int nb_n2 = (int)((N2 + BLK - 1) / BLK);
    float *d_f = nullptr, *d_f_eff = nullptr, *d_bp = nullptr, *d_sens = nullptr;
    float *d_tmp0 = nullptr, *d_conv_tmp1 = nullptr, *d_conv_tmp2 = nullptr;
    float *d_psf_norm = nullptr, *d_post_norm = nullptr, *d_map_grad = nullptr;
    CUDA_CHECK(cudaMalloc(&d_f, (size_t)N2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_f_eff, (size_t)N2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bp, (size_t)N2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sens, (size_t)N2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp0, (size_t)N2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_tmp1, (size_t)N2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_tmp2, (size_t)N2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_psf_norm, (size_t)N2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_post_norm, (size_t)N2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_map_grad, (size_t)N2*sizeof(float)));
    k_fill<<<nb_n2, BLK>>>(d_f, 1.0f, N2);

    bool use_psf = (cfg.conv_mode == "psf" || cfg.conv_mode == "sieve");
    bool use_post = (cfg.conv_mode == "post" || cfg.conv_mode == "sieve");
    if (cfg.conv_mode == "none") { use_psf = false; use_post = false; }

    std::vector<float> psf_kx, psf_ky, psf_kz;
    if (use_psf) {
        psf_kx = build_castor_erf_1d(cfg.psf_fwhm_mm[0], meta.voxel_size, cfg.psf_nb_sigmas);
        psf_ky = build_castor_erf_1d(cfg.psf_fwhm_mm[1], meta.voxel_size, cfg.psf_nb_sigmas);
        psf_kz = build_castor_erf_1d(cfg.psf_fwhm_mm[2], meta.voxel_size, cfg.psf_nb_sigmas);
        build_conv_norm(d_psf_norm, d_conv_tmp1, d_conv_tmp2, meta.Nx, meta.Ny, meta.Nz, psf_kx, psf_ky, psf_kz);
        printf("[6] PSF on: FWHM=[%.3f %.3f %.3f] mm, kernel=%zu x %zu x %zu\n",
               cfg.psf_fwhm_mm[0], cfg.psf_fwhm_mm[1], cfg.psf_fwhm_mm[2],
               psf_kx.size(), psf_ky.size(), psf_kz.size());
    } else {
        k_fill<<<nb_n2, BLK>>>(d_psf_norm, 1.0f, N2);
        printf("[6] PSF off\n");
    }

    /* Butterworth 准备 */
    cufftHandle fft_plan = 0;
    cufftComplex* d_fft = nullptr;
    float* d_butter = nullptr;
    if (cfg.butter_enable) {
        auto h_butter = build_butterworth_filter(meta.Nx, meta.Ny, meta.Nz,
                                                 cfg.butter_cutoff, cfg.butter_order);
        CUDA_CHECK(cudaMalloc(&d_butter, (size_t)N2*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_butter, h_butter.data(), (size_t)N2*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_fft, (size_t)N2*sizeof(cufftComplex)));
        CUFFT_CHECK(cufftPlan3d(&fft_plan, meta.Nz, meta.Ny, meta.Nx, CUFFT_C2C));
        printf("[7] Butterworth on: cutoff=%.3f order=%d mode=%s\n",
               cfg.butter_cutoff, cfg.butter_order, cfg.butter_mode.c_str());
    } else {
        printf("[7] Butterworth off\n");
    }

    printf("[8] Compute sensitivity\n");
    {
        std::vector<std::thread> threads;
        for (int g = 0; g < cfg.num_gpus; g++) {
            threads.emplace_back([&, g]() {
                GPUWorker& w = workers[g];
                CUDA_CHECK(cudaSetDevice(w.dev));
                int nb_img = (int)(((int64_t)w.N2 + BLK - 1) / BLK);
                int nb_rows = (w.local_rows + BLK - 1) / BLK;
                CUDA_CHECK(cudaMemset(w.d_sens_part, 0, (size_t)w.N2*sizeof(float)));
                for (int s = 0; s < sym_count; s++) {
                    for (int zm = 0; zm < z_count; zm++) {
                        k_make_unit_ratio_batched<<<nb_rows, BLK>>>(w.d_ratio, w.local_rows,
                            w.start_ck, w.compact_rpa, w.det_nx, w.compact_det_z, w.det_nz,
                            (int)sym_mode, s, zm, meta.compact_num_angles, meta.full_num_angles,
                            cfg.recon_angle_step);
                        w.spmv_backward();
                        k_transform_bwd_add<<<nb_img, BLK>>>(w.d_bp_rot, w.d_sens_part,
                            w.Nx, w.Ny, w.Nz, (int)sym_mode, s, zm);
                    }
                }
                CUDA_CHECK(cudaDeviceSynchronize());
            });
        }
        for (auto& t : threads) t.join();
        reduce_parts_to_gpu0(d_sens, d_tmp0, workers, &GPUWorker::d_sens_part, N2);
        if (use_psf) {
            gpu_conv3d_renorm(d_sens, d_f_eff, d_conv_tmp1, d_psf_norm,
                              meta.Nx, meta.Ny, meta.Nz, psf_kx, psf_ky, psf_kz);
            CUDA_CHECK(cudaMemcpy(d_sens, d_f_eff, (size_t)N2*sizeof(float), cudaMemcpyDeviceToDevice));
        }
        k_clamp_min<<<nb_n2, BLK>>>(d_sens, 1e-12f, N2);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> h_sens((size_t)N2);
        CUDA_CHECK(cudaMemcpy(h_sens.data(), d_sens, (size_t)N2*sizeof(float), cudaMemcpyDeviceToHost));
        save_volume_raw(cfg.output_dir + "/sensitivity.raw", h_sens.data(), meta.Nx, meta.Ny, meta.Nz, meta.voxel_size);
    }

    printf("[9] Iterate: iters=%d, conv=%s, MAP=%s beta=%.3e\n",
           cfg.num_iters, cfg.conv_mode.c_str(),
           cfg.map_type == MAP_NONE ? "none" : "on", cfg.map_beta);

    std::vector<float> h_out((size_t)N2);
    auto total_t0 = std::chrono::high_resolution_clock::now();

    for (int iter = 1; iter <= cfg.num_iters; iter++) {
        auto iter_t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaSetDevice(0));

        /* Step 1: CASToR psf/sieve 模式：前向投影前 H(f)。 */
        float* d_current = d_f;
        if (use_psf) {
            gpu_conv3d_renorm(d_f, d_f_eff, d_conv_tmp1, d_psf_norm,
                              meta.Nx, meta.Ny, meta.Nz, psf_kx, psf_ky, psf_kz);
            d_current = d_f_eff;
        }

        /* Step 2: 将当前图像广播到所有 GPU。 */
        broadcast_image_from_gpu0(d_current, workers, N2);

        /* Step 3: compact-batched forward -> ratio -> backward。 */
        {
            std::vector<std::thread> threads;
            for (int g = 0; g < cfg.num_gpus; g++) {
                threads.emplace_back([&, g]() {
                    GPUWorker& w = workers[g];
                    CUDA_CHECK(cudaSetDevice(w.dev));
                    int nb_img = (int)(((int64_t)w.N2 + BLK - 1) / BLK);
                    int nb_rows = (w.local_rows + BLK - 1) / BLK;
                    CUDA_CHECK(cudaMemset(w.d_bp_part, 0, (size_t)w.N2*sizeof(float)));

                    for (int s = 0; s < sym_count; s++) {
                        for (int zm = 0; zm < z_count; zm++) {
                            /* 3.1 图像变换到 compact 矩阵坐标系。 */
                            k_transform_fwd<<<nb_img, BLK>>>(w.d_img_in, w.d_img_rot,
                                w.Nx, w.Ny, w.Nz, (int)sym_mode, s, zm);

                            /* 3.2 一次大 SpMV 覆盖本 GPU 所有 compact rows。 */
                            w.spmv_forward();

                            /* 3.3 直接在 GPU 端从 full 投影取值，形成 compact ratio。 */
                            k_calc_ratio_batched<<<nb_rows, BLK>>>(w.d_y, w.d_ratio, w.d_P_full,
                                w.local_rows, w.start_ck, w.compact_rpa,
                                w.det_nx, w.compact_det_z, w.det_nz, full_rpa,
                                (int)sym_mode, s, zm, meta.compact_num_angles,
                                meta.full_num_angles, cfg.recon_angle_step);

                            /* 3.4 反投影并旋回原图像坐标。 */
                            w.spmv_backward();
                            k_transform_bwd_add<<<nb_img, BLK>>>(w.d_bp_rot, w.d_bp_part,
                                w.Nx, w.Ny, w.Nz, (int)sym_mode, s, zm);
                        }
                    }
                    CUDA_CHECK(cudaDeviceSynchronize());
                });
            }
            for (auto& t : threads) t.join();
        }

        /* Step 4: 多 GPU 反投影归约到 GPU0。 */
        reduce_parts_to_gpu0(d_bp, d_tmp0, workers, &GPUWorker::d_bp_part, N2);

        /* Step 5: CASToR psf/sieve 模式：反投影后 H(W^T ratio)。 */
        if (use_psf) {
            gpu_conv3d_renorm(d_bp, d_f_eff, d_conv_tmp1, d_psf_norm,
                              meta.Nx, meta.Ny, meta.Nz, psf_kx, psf_ky, psf_kz);
            CUDA_CHECK(cudaMemcpy(d_bp, d_f_eff, (size_t)N2*sizeof(float), cudaMemcpyDeviceToDevice));
        }

        /* Step 6: MAP 梯度。 */
        float* map_ptr = nullptr;
        if (cfg.map_type != MAP_NONE && cfg.map_beta > 0.0f) {
            k_map_gradient<<<nb_n2, BLK>>>(d_f, d_map_grad,
                meta.Nx, meta.Ny, meta.Nz,
                (int)cfg.map_type, cfg.map_neigh26,
                cfg.map_delta, cfg.map_huber_threshold,
                cfg.map_relative_gamma, cfg.map_relative_epsilon);
            map_ptr = d_map_grad;
        }

        /* Step 7: MLEM/MAP-EM 更新 + 非负约束。 */
        k_mlem_update<<<nb_n2, BLK>>>(d_f, d_bp, d_sens, map_ptr, cfg.map_beta, N2);
        if (cfg.positivity) k_clamp_pos<<<nb_n2, BLK>>>(d_f, N2);

        /* Step 8: post/sieve 的更新后卷积。 */
        if (use_post && iter >= cfg.post_start_iter &&
            cfg.post_apply_every > 0 && ((iter - cfg.post_start_iter) % cfg.post_apply_every == 0))
        {
            float fw = scheduled_post_fwhm(cfg, iter);
            if (fw > 0.01f) {
                std::vector<float> pkx = build_castor_erf_1d(fw, meta.voxel_size, cfg.psf_nb_sigmas);
                std::vector<float> pky = build_castor_erf_1d(fw, meta.voxel_size, cfg.psf_nb_sigmas);
                std::vector<float> pkz = build_castor_erf_1d(fw, meta.voxel_size, cfg.psf_nb_sigmas);
                build_conv_norm(d_post_norm, d_conv_tmp1, d_conv_tmp2, meta.Nx, meta.Ny, meta.Nz, pkx, pky, pkz);
                gpu_conv3d_renorm(d_f, d_f_eff, d_conv_tmp1, d_post_norm,
                                  meta.Nx, meta.Ny, meta.Nz, pkx, pky, pkz);
                CUDA_CHECK(cudaMemcpy(d_f, d_f_eff, (size_t)N2*sizeof(float), cudaMemcpyDeviceToDevice));
                if (cfg.positivity) k_clamp_pos<<<nb_n2, BLK>>>(d_f, N2);
            }
        }

        /* Step 9: 独立 Butterworth 后处理。 */
        if (cfg.butter_enable && should_apply_filter(iter, cfg.num_iters, cfg.butter_mode,
                                                     cfg.butter_period, cfg.butter_last_n))
        {
            k_real_to_complex<<<nb_n2, BLK>>>(d_f, d_fft, N2);
            CUFFT_CHECK(cufftExecC2C(fft_plan, d_fft, d_fft, CUFFT_FORWARD));
            k_butter_mul<<<nb_n2, BLK>>>(d_fft, d_butter, N2);
            CUFFT_CHECK(cufftExecC2C(fft_plan, d_fft, d_fft, CUFFT_INVERSE));
            k_complex_to_real<<<nb_n2, BLK>>>(d_fft, d_f, N2, 1.0f/(float)N2);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        double iter_sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - iter_t0).count();

        bool do_save = (iter <= 3) || (cfg.save_interval > 0 && iter % cfg.save_interval == 0) || (iter == cfg.num_iters);
        if (do_save) {
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_f, (size_t)N2*sizeof(float), cudaMemcpyDeviceToHost));
            char fn[1024];
            snprintf(fn, sizeof(fn), "%s/recon_iter%03d.raw", cfg.output_dir.c_str(), iter);
            save_volume_raw(fn, h_out.data(), meta.Nx, meta.Ny, meta.Nz, meta.voxel_size);
            printf("  Iter %3d/%d | saved | %.3fs\n", iter, cfg.num_iters, iter_sec);
        } else {
            printf("  Iter %3d/%d | %.3fs\n", iter, cfg.num_iters, iter_sec);
        }
    }

    double total_sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - total_t0).count();
    printf("\nDone. Total %.2f min, avg %.3f s/iter\n", total_sec/60.0, total_sec/std::max(1, cfg.num_iters));

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_f, (size_t)N2*sizeof(float), cudaMemcpyDeviceToHost));
    save_volume_raw(cfg.output_dir + "/recon_final.raw", h_out.data(), meta.Nx, meta.Ny, meta.Nz, meta.voxel_size);

    if (fft_plan) cufftDestroy(fft_plan);
    if (d_fft) cudaFree(d_fft);
    if (d_butter) cudaFree(d_butter);

    cudaFree(d_map_grad);
    cudaFree(d_post_norm);
    cudaFree(d_psf_norm);
    cudaFree(d_conv_tmp2);
    cudaFree(d_conv_tmp1);
    cudaFree(d_tmp0);
    cudaFree(d_sens);
    cudaFree(d_bp);
    cudaFree(d_f_eff);
    cudaFree(d_f);

    for (auto& w : workers) w.destroy();
    return 0;
}
