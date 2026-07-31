# 6module_bak_src — 重建原型与系统矩阵生成端

本目录**不参与**主线 `SPECT_recon` CMake 构建，仅保存唯一的重建原型、唯一的系统矩阵生成端与投影预处理工具。

## 文件说明

| 文件 | 定位 | 用途 |
| --- | --- | --- |
| `spect_recon_PSC_6module_24pin_YSO_bedmerge.cu` | **唯一重建原型** | 多床位单床重建与拼接参考实现（编译/用法照录文件头注释，见下） |
| `spect_sysmat_PSC_6module_24pin_YSO_v4.cu` | **唯一系统矩阵生成端** | 输出 `metadata.bin`(88B, SYM_90Z) + `chunk_*.csrb`；后续响应变体的基线 |
| `mat_convert_bin.m` | 工具 | MAT → 投影 BIN（函数 `mat_convert_bin(...)`；示例见 `run_mat_convert_bin_example.m`） |
| `run_mat_convert_bin_example.m` | 示例 | `mat_convert_bin` 调用示例（含 legacy 硬编码路径，按需修改） |

## 重建原型：spect_recon_PSC_6module_24pin_YSO_bedmerge.cu

照录文件头注释：

```text
SPECT 多床位单床重建与拼接程序。

流程：逐床位执行完整系统矩阵重建，按中心 mask 裁剪 x/y/z，
在相邻床位重叠区估计标量增益，最后使用余弦平方权重平滑拼接。
体素线性索引固定为 z*(Nx*Ny)+y*Nx+x，其中 x 为最快变化维。

编译：
nvcc -O3 -std=c++17 -arch=sm_89 \
  recon/spect_recon_PSC_6module_24pin_YSO_bedmerge.cu \
  -o recon/process/spect_recon_PSC_6module_24pin_YSO_bedmerge_sm89 \
  -lcusparse -lcufft

usage:
./spect_recon_PSC_6module_24pin_YSO_bedmerge \
  --matrix-dir  MCmatrix_PSC_48h \
  --proj-dir  ep_168 \
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
```

## 系统矩阵生成端：spect_sysmat_PSC_6module_24pin_YSO_v4.cu

编译（Windows，按文件头注释口径，文件名以实际为准）：

```powershell
nvcc -O3 -std=c++14 -arch=sm_86 6module_bak_src\spect_sysmat_PSC_6module_24pin_YSO_v4.cu -o spect_sysmat_v4.exe
```

默认 `coll_modules=6`；输出目录含 `metadata.bin`（88B，SYM_90Z）、`metadata.txt`（文本版，便于人工检查）与 `chunk_000.csrb` …。

## 与主线的关系

- **唯一生产链路**：v4 生成端输出 `metadata.bin + chunk_*.csrb` → 主线 `SPECT_recon -chunkCSRDir <dir>` 直接读取（含 88B/96B metadata、SYM_90Z compact-symmetric 布局）。
- 详见 `doc/Format_Reader_Matrix_CN.md`（格式说明）与 `doc/Capability_Matrix_CN.md`（能力边界）。
