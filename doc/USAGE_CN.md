# SPECT_recon 使用说明（中文）

> 更新日期：2026-06-10  
> 版本：v2.0 (CUDA 融合方案 P1-P6 完成后)

本工具用于三维 SPECT 重建，支持 MLEM/OSEM、BSREM-II、ADMM 三类主线优化算法，以及基于 cuSPARSE + NVRTC 的 GPU 加速路径。

## 可执行文件

- **Windows 构建**: 运行 `build.bat` → 安装到 `out/install/Release/bin/SPECT_recon.exe`；构建/安装产物会在 exe 同级生成 `config/`，通常不需要传 `-config`
- **CUDA scaffold 构建**: `scripts/build_cuda_scaffold.ps1 -Build` → `build_cuda_scaffold/bin/SPECT_recon.exe`
- **Release 构建**: CMake preset `x64-release` → Ninja → `release/install/bin/SPECT_recon.exe`

推荐 RC 发布包结构如下，`config/` 与可执行文件和 DLL 放在同一个 `bin/` 目录下：

```text
SPECT_recon_<version>/
  bin/
    SPECT_recon.exe
    *.dll
    config/
      spect_config.ini
      algo_mlem.conf
      algo_bsrem.conf
      algo_admm.conf
    matrix/
      <matrix_name>/
        metadata.bin
        chunk_000.csrb
        chunk_001.csrb
        chunk_002.csrb
        chunk_003.csrb
        manifest.json
  README.md
  PARAMETERS_CN.md
  RELEASE_NOTES.md
  SHA256SUMS.txt
```

未显式传入 `-config` 时，程序按顺序搜索：exe 所在目录的 `config/spect_config.ini`、exe 所在目录的 `spect_config.ini`、exe 上一级的 `config/spect_config.ini`、当前工作目录的 `config/spect_config.ini`、当前工作目录的 `spect_config.ini`。

Compact chunk 系统矩阵目录可以作为可选资源放在 `bin/matrix/<matrix_name>/`。
默认识别优先级固定为：

1. 命令行 `-chunkCSRDir <dir>`。
2. `bin/config/spect_config.ini` 中 `[SystemMatrix] default_chunk_csr_dir`。
3. `bin/matrix/` 下仅有一个包含 `metadata.bin` 的矩阵目录时自动使用。
4. 如果存在多个候选目录且没有显式指定，程序直接报错并列出候选项。

---

## 基本用法

```bash
SPECT_recon.exe -df <proj.mat> -loadCSR <sysmat.csrb> -mapCSR on \
  -dim Nx,Ny,Nz -vox Vx,Vy,Vz -ang 72 -detN 40,40 \
  -it 50:5 -th 16 -fout recon_output/run1
```

---

## 优化算法

### MLEM/OSEM（默认）

经典 EM 算法。`-subsets > 1` 时为 OSEM。

```bash
SPECT_recon.exe ... -it 50:5 -subsets 1
```

### BSREM-II

带松弛步长的加性更新算法（Ahn & Fessler 2003）。

```bash
SPECT_recon.exe ... -alg BSREM -bsrem 1.0,0.1 -reg RDP,0.01,1e-4,2.0
```

BSREM 扩展参数:
- `-bsremCurv on|off` — 启用曲率分母 D=S+β·Dreg
- `-bsremCurvFreeze on|off` — 冻结曲率图
- `-bsremCurvWarmup <n>` — 冻结前预热迭代数（默认 2）
- `-bsremKappa on|off` — 启用 kappa 权重图 β_j=β·κ_j
- `-bsremKappaClip <min,max>` — kappa 裁剪范围（默认 0.1,10）
- `-bsremSoftMask on|off` — 启用 soft-mask 惩罚增强

### ADMM（P5/P6 新增）

交替方向乘子法（Alternating Direction Method of Multipliers），将带正则化的重建问题拆分为三个交替子问题。

```bash
# ADMM + RDP 正则化
SPECT_recon.exe ... -alg admm -admmRho 1.0 -reg RDP,0.01,1e-4,2.0 -it 50

# ADMM + TV 正则化（Chambolle-Pock 内循环）
SPECT_recon.exe ... -alg admm -admmRho 1.0 -reg TV,0.01,1e-6 \
  -admmTvInner 30 -admmAdaptive on -it 50

# ADMM + TV，固定高 ρ（更强正则化效果）
SPECT_recon.exe ... -alg admm -admmRho 10.0 -reg TV,0.01,1e-6 \
  -admmAdaptive off -it 20
```

ADMM 参数:
- `-alg admm` — 选择 ADMM 算法
- `-admmRho <value>` — ADMM 惩罚参数 ρ（默认 1.0，推荐 ≥ 1.0）
- `-admmTvInner <n>` — TV 近端算子 Chambolle-Pock 内循环迭代数（默认 20）
- `-admmAdaptive on|off` — 自适应 ρ 调整（默认 on；基于 Boyd 2011 / CASToR ADMMLim）

---

## 正则化

`-reg <type,beta,...>`

| 类型 | 格式 | 说明 |
|------|------|------|
| `None` | `-reg None` | 无正则化 |
| `TVIsotropic` | `-reg TVIsotropic,0.01` 或 `-reg TV,0.01,1e-6` | 各向同性 TV |
| `TVHuber` | `-reg TVHuber,0.01,0.1` | Huber 化 TV |
| `Tikhonov` | `-reg Tikhonov,0.01` | L2 正则 |
| `MRP` | `-reg MRP,0.01` | 中位根先验 |
| `NLM` | `-reg NLM,0.01,0.1,3,1` | 非局部均值 (β,h,search,patch) |
| `Bowsher` | `-reg Bowsher,0.01,13` | Bowsher 先验 (需 `-prior`) |
| `PLS` | `-reg PLS,0.01,1.0` | 并行水平集 (需 `-prior`) |
| `RDP` | `-reg RDP,0.01,1e-4,2.0` | 相对差分惩罚 (β,γ,δ) |
| `LogCosh` | `-reg LogCosh,0.01,0.1` | Log-Cosh 先验 (β,δ) |

---

## GPU 加速（CUDA）

### 后端选择

- `-sparseBackend <auto|cpu|ref|cuda>` — SpMV 后端（默认 auto，有 GPU 时自动选 cuda）
- `-butterBackend <auto|fftw|cufft>` — Butterworth 滤波后端

### GPU 加速覆盖

| 组件 | 技术 | 加速比 |
|------|------|--------|
| SpMV 前/后向投影 | cuSPARSE float32 | ~10x |
| Butterworth 滤波 | cuFFT R2C/C2R float | ~15x |
| PSF 卷积 | NVRTC float kernel | GPU 端执行 |
| RDP/TV 梯度 | NVRTC kernel | GPU 端执行 |
| TV proximal (ADMM) | NVRTC Chambolle-Pock | GPU 端执行 |

---

## 数据输入

### 投影数据

- `-df <file>` — 投影数据 MAT 文件（单床位，必需）
- `-dfList <list>` — 多床位投影列表（逗号/分号分隔，与 `-df` 二选一）；路径中含 `BedN` 时按床位数字排序，最多 5 个床位，允许只传 1 个床位

### 系统矩阵

四种加载方式（按推荐排序）：

1. **compact chunk 发布包自动识别**（当前 6-module compact 后端推荐）：
   ```bash
   # bin/matrix/ 下只有一个 <matrix_name>/metadata.bin 时可省略矩阵参数
   SPECT_recon.exe -df proj.bin -o recon_output -it 50:5
   ```

   多个矩阵目录时，使用下列任一方式指定：
   ```ini
   [SystemMatrix]
   default_chunk_csr_dir = matrix/<matrix_name>
   ```

   ```bash
   -chunkCSRDir bin/matrix/<matrix_name>
   ```

2. **CSR 内存映射**（标准 CSR，推荐生产使用）：
   ```bash
   -loadCSR sysmat.csrb -mapCSR on
   ```

3. **CSR 直接加载**：
   ```bash
   -loadCSR sysmat.csrb
   ```

4. **从 MAT 构建**（首次运行或调试）：
   ```bash
   -indexMat idx.mat -weightMat wgt.mat -saveCSR sysmat.csrb
   ```

### CSR 性能选项

- `-Aprune <eps>` — 剪枝阈值（默认 1e-10）
- `-Astream on|off` — 流式构建（默认 off，内存不足时开启）
- `-ApruneMask on|off` — 掩码剪枝（默认 on）

---

## 滤波与 PSF

- `-butter <cutoff,order>` — Butterworth 低通滤波（如 `-butter 0.35,2`）
- `-smooth <sigma>` — 高斯平滑
- `-joe <n>` — Joseph-like 平滑通道数（默认 0=关闭）
- `-psf <sigma>` — PSF sigma（mm），启用 PSF 卷积
- `-psfMode <mode>` — PSF 模式: `Operator`（默认）、`PostSmooth`、`None`
- `-psfProj on|off` — 投影域 PSF（默认 off，仅支持 MLEM + subsets=1）

---

## 掩码

- `-cyl <diam,height>` — 圆柱掩码（mm）
- `-cylc <cx,cy,cz>` — 圆柱中心（体素索引，1-based）
- `-maskFromAtt <threshold>` — 从衰减图自动生成身体掩码

---

## 衰减校正

- `-attMap <file>` — 衰减图 MAT 文件
- `-ctac` — 启用迭代 CTAC
- `-bProj <file>` — 背景/散射投影项
- `-bProjList <list>` — 多床位背景投影列表
- `-prior <file>` — 解剖先验图像

---

## 输出控制

- `-fout <path>` — 输出路径前缀（目录 + 文件名前缀）
- `-it <iters[:save]>` — 迭代次数与保存间隔（如 50:5 每 5 次保存）
- `-cropEnd <top[,end]>` — 最终输出 Z 轴上下端裁剪，在 `zbedflip/zoutflip` 之后执行
- `-zbedflip <0|1>` — 输出时对 `-dfList` 中每个床位段分别做 Z 轴翻转，默认 1
- `-zoutflip <0|1>` — 每床翻转之后，对最终整体输出体再做 Z 轴翻转，默认 0
- `-vb <0|1|2>` — 日志级别（0=安静，1=INFO 默认，2=DEBUG）

输出格式：MHD/RAW 体数据 + 自动日志 `<outputName>_YYYYMMDD_HHMMSS_mmm.log`

---

## 多床位重建

```bash
SPECT_recon.exe -dfList "bed1.mat;bed2.mat;bed3.mat;bed4.mat" \
  -loadCSR sysmat.csrb -mapCSR on -dim 100,100,100 -vox 0.3,0.3,0.3 \
  -ang 72 -detN 40,40 -it 50:5 -th 16 -fout recon_out/multibed
```

- `-dim` 为**单床位**维度；输出 Z = Nz × beds - sum(overlap)
- `-dfList` 默认按路径中的 `BedN` 数字排序；无 `BedN` 的路径保持原相对顺序并排在已编号床位之后
- `-bProjList` 使用同样的 `BedN` 排序规则，排序后必须与床位数量一致
- `-bedOverlap <b1,b2,b3,b4,b5>` — 固定 `Bed1..Bed5` 的床位交叉裁剪表；`b1` 匹配 `Bed1-Bed2`，`b2` 匹配 `Bed2-Bed3`，`b3` 匹配 `Bed3-Bed4`，`b4` 匹配 `Bed4-Bed5`，`b5` 作为固定五床表占位保留。实际只传 `Bed2,Bed3,Bed4` 时使用 `b2,b3`。只传一个值时复制到所有固定槽位
- `-cropEnd <top,end>` — 只负责最终输出体的上下两端裁剪，不负责床位交叉处裁剪
- `-zbedflip 1` 先对每个床位段内部做 Z 翻转；`-zoutflip 1` 再对整体输出做 Z 翻转；随后才执行 `-cropEnd top,end`
- 支持 `-mapCSR on` + `-ctac`（动态衰减，CSR 只读）

---

## 其他选项

- `-dim <x,y,z>` — 重建体素维度
- `-vox <x,y,z>` — 体素尺寸（mm）
- `-ang <n>` — 角度数
- `-detN <nz[,nx]>` — 探测器尺寸
- `-th <threads>` — OpenMP 线程数
- `-subsets <n>` — OSEM 子集数（1=MLEM）
- `-config <file>` — 加载 INI 配置文件；未指定时优先搜索 exe 同级 `config/spect_config.ini`

---

## 编译

### Windows（推荐）

```bat
build.bat
```

可执行文件: `out/install/Release/bin/SPECT_recon.exe`

### CMake Preset（Ninja）

```powershell
cmake --preset x64-release
cmake --build --preset x64-release -j 16
cmake --install --preset x64-release
```

### CUDA Scaffold 构建

```powershell
scripts/build_cuda_scaffold.ps1 -Build
```

可执行文件: `build_cuda_scaffold/bin/SPECT_recon.exe`

包含 CUDA 加速（cuSPARSE/cuFFT/NVRTC）+ scaffold 测试

### Linux / macOS

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/SPECT_recon ...
```

---

## 测试

```bash
# 单元测试 (194 项)
build_cuda_scaffold/bin/spect_tests.exe

# CUDA scaffold 测试 (25 项)
build_cuda_scaffold/bin/spect_cuda_scaffold_tests.exe

# 四床位回归
scripts/run_4bed_smoke.ps1
```

---

## 典型工作流

1. **首次运行** — 从 MAT 构建 CSR 并缓存:
   ```bash
   SPECT_recon.exe -df proj.mat -indexMat idx.mat -weightMat wgt.mat \
     -dim 200,200,200 -vox 0.4,0.4,0.4 -ang 72 -detN 40,40 \
     -it 50:5 -th 16 -saveCSR sysmat.csrb -fout run1
   ```

2. **后续运行** — 映射缓存，GPU 加速:
   ```bash
   SPECT_recon.exe -df proj.mat -loadCSR sysmat.csrb -mapCSR on \
     -dim 200,200,200 -vox 0.4,0.4,0.4 -ang 72 -detN 40,40 \
     -it 50:5 -th 16 -sparseBackend cuda -butterBackend cufft \
     -butter 0.35,2 -reg RDP,0.01,1e-4,2.0 -fout run2
   ```

3. **ADMM-TV 重建**:
   ```bash
   SPECT_recon.exe -df proj.mat -loadCSR sysmat.csrb -mapCSR on \
     -dim 200,200,200 -vox 0.4,0.4,0.4 -ang 72 -detN 40,40 \
     -alg admm -admmRho 1.0 -reg TV,0.01,1e-6 \
     -admmTvInner 30 -admmAdaptive on -it 50:5 -fout run3
   ```
