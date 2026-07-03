# SPECT-recon 代码优化清单（2026-07-03 审查）

审查范围：`src/`、`include/`、`tools/`、`scripts/`、`tests/`、`6module_bak_src/`、`CMakeLists.txt`、`doc/`、根目录工程文件。

背景约定：

- 系统矩阵标准格式为 `test_data\6module\260601`（`metadata.bin` 88B + `chunk_*.csrb`，`-chunkCSRDir` 入口）。
- 当前保留对 `test_data\6module\260525` 的兼容：小批量对称压缩单文件 CSRB（`matrix_6module.csrb` + 76B metadata，`-loadCSR -compactMeta` 入口）与旧 row-wise `chunk_*.bin`（仅离线转换工具读取）。
- 项目已完成 3 模块 → 6 模块升级；`6module_bak_src/` 存放系统矩阵生成 CUDA 源（v2/v4）与独立 MLEM 参考实现。
- 根目录当前**不是 git 仓库**，与 GitHub 已脱钩 4~5 个月。

严重度定义：**高** = 有正确性/数据风险或严重阻碍维护；**中** = 明确技术债，应排期；**低** = 择机清理。

---

## 0. P0：工程基线（先做，不动代码逻辑）

| # | 问题 | 位置 | 严重度 | 建议 |
|---|------|------|--------|------|
| 0.1 | **已修改（2026-07-03）** | 已完成 | 方案：`git init` + 首次 baseline 提交（191 文件）；remote `https://github.com/wangdy920/spect_6module_recon.git`，分支 `main` 已推送 |
| 0.2 | **已修改（2026-07-03）** | 已完成 | 方案：`.gitignore` 补 `__pycache__/`、`*.py[cod]`、`test_data/`、`.agent/`；现有 pycache 目录已不存在 |
| 0.3 | **已修改（2026-07-03）** | 已完成 | 方案：删除根目录 `cmake_test_discovery_*.json`（2 个）与 `compare_refactor.py` |
| 0.4 | **已修改（2026-07-03）** | 已完成 | 方案：`README.md` 统一文档目录为 `doc/`，删除 `docs/`/`refactor_docs/`/`matlab_ref/`/`original_source_backup/` 死链；Status 更新为 260601 chunk CSRB、v28、117.6s @ 50 iter、227/227 tests |
| 0.5 | **已修改（2026-07-03）** | 已完成 | 方案：`.agent/` 纳入 `.gitignore`（工作区笔记不入库）；关键结论已在 `doc/` 正式文档中维护 |

## 1. 主程序与架构（`src/main/`、`include/cli/`、`include/app/`）

| # | 问题 | 位置 | 严重度 | 建议 |
|---|------|------|--------|------|
| 1.1 | `MainProgram.cpp` 约 2678 行，是事实上的 God Object：数据加载、compact/standard 双重建链路、缓存、profiling、输出全部在内 | `src/main/MainProgram.cpp` | **高** | 按职责拆为 `CompactReconLoop`、`StandardReconLoop`、`ProfilingReporter` 等编译单元 |
| 1.2 | `runProgram()` 约 1630 行（L1117~L2747）；内部 lambda `runReconstruction` 约 1166 行；compact CUDA 迭代分支约 684 行 | 同上 | **高** | lambda 提升为命名类；compact 分支抽成 `CompactSymmetricReconstructor`，与 standard 路径对称 |
| 1.3 | compact 路径内独立复制了一套 BSREM 更新环（alpha/beta/curv/kappa/updateFactor），与 `BSREMReconstructor.cpp` L592~632 重复，是第三套 BSREM 实现 | `MainProgram.cpp` L1965~2054 | **高** | 提取共享 `applyBsremUpdate()`，或让 compact 路径走 `IOptimizer` |
| 1.4 | `ArgumentParser.h` 约 1788 行，几乎全部 inline 实现；`registerHandlers()` 614 行、`applyConfig()` 246 行、`validateArgs()` 204 行 | `include/cli/ArgumentParser.h` | **高** | 实现移至 `.cpp`；handler 表驱动注册；校验按 IO/几何/算法分组 |
| 1.5 | `ArgumentParser` 内直接 `std::exit(0/1)`（`-dumpConf`、部分校验失败），绕过 main 清理与 RAII | `ArgumentParser.h` L445、L1064、L1197、L1416 | **高** | 改为返回错误码/enum，由 `runProgram` 统一 exit |
| 1.6 | 双轨接口技术债：`IReconstructor`/`ReconstructorFactory`/`IReconRunner` 只有头文件无实现且主路径不用；`ReconContext`+Adapter 已完善但 MainProgram 仍直接用 `ReconArgs`/`ReconOpts`，床位/overlap 逻辑算两遍 | `include/interfaces/IReconstructor.h`、`include/app/ReconRunner.h`、`ReconContext.h` | **高** | 二选一：接入主路径，或标记 deprecated 并从 `SPECT.h` 移出。不要维持两套并行演进 |
| 1.7 | OSEM/BSREM 之间重复：`isOrderedFullSubset`、`canUseCudaFusedFullSubset*` 两份拷贝；forward AVX2+衰减+多床 offset 逻辑至少 6 处重复 | `src/app/OSEMReconstructor.cpp` L31~93 等、`BSREMReconstructor.cpp` L32~94 等 | **高** | 提取 `app/SubsetUtils.h` 与统一的 `SubsetForwardKernel`/`SubsetBackprojectKernel` |
| 1.8 | 多床位上限 5 在至少 4 处硬编码（ReconSetup、ArgumentParser、MainProgram、CompactSymmetricProjectorCUDA） | 多处 | 中 | 定义 `constexpr int kMaxBeds = 5` 单点常量 |
| 1.9 | 硬编码默认值有静默错误风险：`numAngles=72`、`detectorChannels=40`（3 模块时代遗留）、无 hint 时默认 200³ 体素 | `include/cli/ReconArgs.h` L36~38、`src/main/ReconSetup.cpp` L174~184 | 中 | 6 模块数据一律以 metadata/BIN header 为准；缺失时 fail-fast 而非静默回退 |
| 1.10 | 床位路径解析（`sortPathsByBedOrder`/`extractBedNumberFromPath`）在 CLI 与 ReconSetup 两处重复 | `ArgumentParser.h` L332~365、`ReconSetup.cpp` L50~95 | 中 | 统一到 `app/BedPathUtils` |
| 1.11 | CUDA chunk stage timing 日志块两处近乎相同（各约 80 行） | `MainProgram.cpp` L2173~2251、L2636~2711 | 中 | 提取 `logCudaChunkTimingSnapshot()` |
| 1.12 | 全局状态：`g_osemStageTiming`/`g_bsremStageTiming` 非线程安全；`gVerbosity` 全局变量 | `OSEMReconstructor.cpp` L29、`BSREMReconstructor.cpp` L30、`RunLogger.cpp` | 中/低 | timing 经 `StepParams` 传出；verbosity 注入上下文 |
| 1.13 | 裸指针生命周期靠约定：`ReconOpts` 的 `mask/b_proj/w_proj`；`args.reg.anatomical_image = &sharedAnatomical` 有悬挂风险 | `include/app/ReconConfig.h`、`MainProgram.cpp` L1287 | 中 | `std::span`/`shared_ptr` + 所有权文档 |
| 1.14 | 死兼容层：`LegacyCompat.h`（内含与 `cli/` 重复的旧 `ConfigParser`/`ReconArgs` 定义，有 ODR 隐患）与 `ReconCompat.h` 均无 src 引用；`RegularizerCompat.h` 仍被生产代码依赖但标注"将移除" | `include/compat/LegacyCompat.h`、`include/app/ReconCompat.h`、`include/regularizer/RegularizerCompat.h` | 中 | 前两者直接删除；`RegularizerCompat` 制定迁移里程碑（MainProgram 改用 `IRegularizer` 工厂） |
| 1.15 | 错误处理不一致：校验失败混用 `return false+cerr`、`throw`、`exit`；CUDA/compact 失败有时 throw 有时 warn 后继续 | `ArgumentParser.h` vs `ReconSetup.cpp` vs `MainProgram.cpp` | 中 | 定义退出码表（0/1/2）与"可 warn 降级 / 必须 abort"边界清单 |
| 1.16 | `65535.0` 衰减 uint16 缩放因子散落 10+ 处 | OSEM/BSREM/ProjectionOps/ISparseOps/SparseOpsCUDA | 低 | `constexpr double kAttenUint16Scale = 65535.0` |

## 2. 系统矩阵 IO 与格式（`src/io/`、`include/io/`、`include/core/`、`tools/`）

| # | 问题 | 位置 | 严重度 | 建议 |
|---|------|------|--------|------|
| 2.1 | 两套互不兼容的 `CSRHeader`/`CSRView` 定义并存：`include/core/`（32B 头、int64 维、double 值）vs `include/io/CSRTypes.h`（24B 头、uint32 维、float 值，真实 CSRB v1）；`core/CSRHeader.h` 注释与真实格式矛盾，测试测的还是 32B 头 | `include/core/CSRHeader.h`、`include/core/CSRView.h` vs `include/io/CSRTypes.h` | **高** | 废弃 `core/` 旧定义（或明确命名空间隔离 + 标注遗留草案），测试改对齐 24B 真实头 |
| 2.2 | `metadata.bin` 无 magic/版本/长度校验；76B（260525）与 88B（260601）靠"尾部字段读取失败即忽略"隐式区分，`(void)readScalar` 静默吞错 | `src/io/CompactSymmetryMetadata.cpp` L102~137 | **高** | 增加版本/expected_size 字段；按文件实际大小决定必读字段，读取失败报错 |
| 2.3 | 转换工具与 README 严重不符：`260601/CompactChunkCSRB_Reconstruction_README_CN.md` 称工具"只接受 `chunk_*.csrb` 并展开对称"，实现只读旧 `chunk_*.bin` 且不展开 | `tools/convert_6module_sysmat_to_csrb.cpp` L66~75、L365~378 vs 格式 README L119~133 | **高** | 以代码为准改文档，或按 README 重写工具；二者必居其一 |
| 2.4 | 单 chunk ~5GB 默认走 `loadChunkDataCopied` 全量读入（mmap 阈值默认 `uint64_t::max`，即 opt-in 关闭——这是 2026-06-02 benchmark 的有意决策，但 README 未记录该环境变量）；mmap 失败静默 `catch(...)` 回退无日志 | `src/io/CompactChunkCSR.cpp` L79~88、L446~454 | 中 | 文档化 `SPECT_COMPACT_CHUNK_MMAP_MIN_BYTES` 与默认拷贝策略的依据；回退时 logWarn |
| 2.5 | 260525 单文件路径：3.7GB `matrix_6module.csrb` 在不加 `-mapCSR` 时默认全量 vector 加载 | `include/io/CSRBuilder.h` L177~184 | **高** | compact-symmetric + 大文件默认启用 `CSRMap`，或强制要求 `-mapCSR` |
| 2.6 | **已修改（2026-07-03）** | 已完成 | 方案：`buildCSR_from_mat` 与 stream 版均改 uint64 累加，写回 uint32 前校验 `nnz <= UINT32_MAX` 超限抛错（转换工具侧未动，若按 §2a 弃用则无需处理） |
| 2.7 | **已修改（2026-07-03）** | 已完成 | 方案：两遍 `Mat_VarReadDataLinear` 均检查返回值，失败时释放 matvar/mat 句柄并抛错（消息含变量名与文件路径） |
| 2.8 | **已修改（2026-07-03）** | 已完成 | 方案：`CSRMap::open` 改用 `std::filesystem::path(path).wstring()` 构造宽字符路径，与 `CompactChunkCSR.cpp` 口径一致 |
| 2.9 | `validateChunkColIdxRange` 每次冷加载对全部 nnz（单 chunk 可达 6e8）线性扫描 | `CompactChunkCSR.cpp` L372~382 | 中 | 抽样校验或用环境变量控制 strict 模式 |
| 2.10 | **已修改（2026-07-03）** | 已完成 | 方案：`loadCSRBinary` 与 `CSRMap::open` 均校验 `version == 1`，未知版本抛错（消息含实际值与路径） |
| 2.11 | **已修改（2026-07-03）** | 已完成 | 方案：脚本增加 `--profile 260525\|260601`（默认 260525 保持向后兼容）+ `--rows/--angles/--detz` 等显式覆盖；260601 profile 实测校准为 90 compact 角 × 48 × compact z=24 = 103680 行、90z 模式；260601 无单文件 CSRB 故跳过 matrix nnz 校验（已注明 TODO） |
| 2.12 | 运行时对称模式实际仅验收 SYM_45Z/SYM_90Z，格式 README 声称支持 none/zonly/180/... | `CompactSymmetryMetadata.cpp` L254~258 vs 格式 README | 中 | README 标注当前验收范围，或补齐其余模式 |

### 2a. 260525 格式正式弃用时的删除清单（备查）

确认无人再用 `matrix_6module.csrb` 的 `-loadCSR` compact 入口后，可删除/简化：

- **整体删除**：`tools/convert_6module_sysmat_to_csrb.cpp` 及 CMake target（唯一读 `chunk_*.bin` 的代码）；`scripts/run_260601_chunk_comparison.ps1` 的 `psc_260525`/`logo_260525` 对照段。
- **简化**：`MainProgram.cpp` 中 `-loadCSR` 的 compact-symmetric 分支（standard `-loadCSR` 保留）；`validateCompactSymmetricInputs()`（chunk 路径已有独立校验）；`CompactSymmetricProjectorCUDA.cpp` 非 chunk compact CSR 路径；metadata 76B 尾部可选读逻辑（统一 88B 后）。
- **保留**：`CompactChunkCSR.*`、`CompactSymmetryMetadata.*`（几何/映射核心）、`CSRBuilder.h` 的 `loadCSRBinary`/`CSRMap`（standard 重建仍需）。

## 3. CUDA 后端（`src/cuda/`、`include/cuda/`、`include/sparse/`、`include/interfaces/`）

| # | 问题 | 位置 | 严重度 | 建议 |
|---|------|------|--------|------|
| 3.1 | `SparseOpsCUDA.cpp` 5311 行单体：NVRTC 内核源码字符串 ~240 行 + `SparseOpsCUDAState` 状态机 ~2200 行 + CSR 构建 ~600 行 + 多层 host 缓存 + stub 全在一个文件 | `src/cuda/SparseOpsCUDA.cpp` | **高** | 拆为 `SparseOpsCUDA_impl / CudaChunkStreaming / CudaProjDomainKernels / CudaCsrBuilder / stub` 等 5~7 个编译单元 |
| 3.2 | `ISparseOps.h` 910 行接口头，~600 行是 private 模板默认实现，18 个虚方法、CSRView/io::CSRView 成对重载 | `include/interfaces/ISparseOps.h` | **高** | 默认实现下沉 `.cpp`；接口拆为 `ISpmvCore` + subset/atten 组合 |
| 3.3 | 基础 SpMV 路径全同步（H2D→默认流→D2H 无 stream）；全局 `mutex_` 序列化所有 CUDA 操作（20+ 处 lock）；`executeChunkedWholeMatrixSpMV` 每 chunk 一次全流同步做 D2H | `SparseOpsCUDA.cpp` L2453~2474、L3881~3886 等 | **高** | 基础路径改 async + computeStream；锁只保护 cache 更新；批量 async D2H。注意：fused full-subset 路径已是双缓冲较优实现，可作模板 |
| 3.4 | `VolumeOpsCUDA` 所有 kernel 用默认流、每 pass 后 `cuCtxSynchronize`；`ButterworthCUDA` cuFFT 未绑定 stream | `src/cuda/VolumeOpsCUDA.cpp` L754 等、`ButterworthCUDA.cpp` L287~381 | 中 | 专用 stream + 仅最终 D2H 前同步；`cufftSetStream` |
| 3.5 | NVRTC 运行时编译：三个模块各自一份 create/compile/log/load 样板；部署需带 nvrtc DLL；首次调用有编译延迟 | `SparseOpsCUDA.cpp`、`VolumeOpsCUDA.cpp`、`ButterworthCUDA.cpp` | 中 | 共用 `CudaNvrtcLoader`；生产构建改 `nvcc -fatbin` 预编译，NVRTC 仅作未知 arch fallback |
| 3.6 | 从未调用 `cudaSetDevice`，多卡环境永远用 device 0（`CudaConfig.h` 的 `setDevice` 是死代码） | 全 CUDA 模块 | 中 | 启动读 `SPECT_CUDA_DEVICE` 环境变量/CLI |
| 3.7 | 死 scaffold：`SparseOpsGPU.h`/`CudaProjector.h` 声明的 `DeviceCSR`/`SparseOpsGPU` 无实现、CMake 未编译、且 `name() override` 对应的虚函数不存在 | `include/cuda/SparseOpsGPU.h`、`CudaProjector.h` | 中 | 删除（生产路径是 `getSparseOpsCUDA()`） |
| 3.8 | 错误处理正面：`CUDA_CHECK`/`SPECT_CUSPARSE_CHECK` 等统一抛异常。但析构中 `cudaFree` 不清错、`cudaHostRegister` 失败静默、GPU PSF 路径 `catch(...)` 无信息降级 CPU | `SparseOpsCUDA.cpp` L1254+、`CompactSymmetricProjectorCUDA.cpp` L159~197、L2905~2912 | 中 | 析构 noexcept+清错；降级路径必须记录原因（PSF 静默降级曾与"全零 numerator"事故同域，见 `doc/Full_Project_Physics_Model_Fix_Log_20260606.md` 节点 5） |
| 3.9 | `CSRView`(double) 路径 `convertCSR` 全量拷到 host 再上传（双倍内存）；`io::CSRView` 已有直传。`uploadDouble` 每次同步转换 | `SparseOpsCUDA.cpp` L1818~1841、L2949~2954 | 中 | 统一走 io::CSRView 直传口径；基础路径补 pinned+async |
| 3.10 | 硬编码资源参数：blockSize=256、prepared host cache 24GiB 上限（注释假设 64GB 主机）、device cache reserve 6GiB | `SparseOpsCUDA.cpp` L148~164、L3586 等 | 低 | 集中 `CudaLaunchConfig`；预算按 `cudaMemGetInfo`/物理内存比例；把 `SPECT_CUDA_*`、`SPECT_COMPACT_*` 全部环境变量写进 README（目前仅格式 README 记录了 2 个） |
| 3.11 | 三份重复的 `checkNvrtc`/`checkCuDriver`/`checkCufft`；`SparseOpsBackend.h` 中 `CompactDevice*` 结构 ifdef 双份定义 | 三个 cuda cpp、`include/sparse/SparseOpsBackend.h` | 低 | 合并到 `include/cuda/CudaDriverHelpers.h`；结构体无条件定义一次 |

## 4. 测试缺口（`tests/`）

| # | 问题 | 严重度 | 建议 |
|---|------|--------|------|
| 4.1 | 无针对 `test_data/6module/260601` 真实 chunk 目录的自动化测试：260601 只出现在 CLI 字符串和手工 benchmark 记录中，chunk 相关 gtest 全部是 toy 数据 | **高** | 增加可选 gtest（存在 `test_data/6module/260601` 时启用）：metadata 88B 解析、chunk header/行数公式（`103680 = 90*48*24`）、1-chunk 读入、90Z 映射覆盖 829440 full rows |
| 4.2 | 无 BIN 投影 + chunk 矩阵的小规模端到端 golden 测试（现有 golden 依赖标准 CSRB） | **高** | 用 toy 90Z metadata + 合成 chunk 做 1-iter MLEM golden，锁定 compact 主路径 |
| 4.3 | `tools/convert_6module_sysmat_to_csrb.cpp` 无 gtest 回归 | 中 | 若按 2a 弃用则不补；保留则加最小 2 行 chunk 转换回归 |
| 4.4 | `test_interfaces.cpp` 84KB 巨型测试文件，IO/parser/compact/multi-bed 混在一起 | 中 | 按域拆分 `test_compact_symmetry.cpp`、`test_parser.cpp` 等 |
| 4.5 | `test_cuda_scaffold.cpp` 的 ChunkWarmup 用合成小矩阵，无真实几何 CompactChunk 覆盖 | 中 | 与 4.1 合并处理 |

## 5. `6module_bak_src/` 与系统矩阵生成端

| # | 问题 | 严重度 | 建议 |
|---|------|--------|------|
| 5.1 | 三种 chunk 格式并存且互不兼容：v2 输出 `chunk_*.bin`（row-wise）、v4 输出 `chunk_*.csrb`、独立 `spect_recon_mlem.cu` 只读 `.bin`、主线只读 `.csrb`。v4 矩阵无法直接喂给独立 MLEM 参考实现 | **高** | 文档化格式矩阵；canonical 定为 v4 `.csrb` + 主线 `-chunkCSRDir`；v2 与独立 MLEM 标注 legacy/reference-only |
| 5.2 | **已修改（2026-07-03）** | 已完成 | 方案：v4 生成端 `coll_modules` 默认值 1→6，与 v2/硬件一致；启动打印几何/旋转摘要（voxel/pixel/det_y/rotation_sign/sym）；`coll_modules != 6` 时输出显式 WARNING，杜绝静默生成单模块矩阵 |
| 5.3 | v2/v4 约 55~65% fork 重复：针孔 CAD 坐标（4 针孔×4 层，约 130 行）两份 100% 相同；准直器构建/射线追踪 kernel 大量拷贝。物理常数（`mu_collimator=3.202`、`mu_detector=0.1368`）、准直器盒尺寸、晶体 `pixel_size=1.55`/`gap_size=0.16` 均硬编码在 `main()` | 中 | 若 v4 已验收为唯一生成端：v2 归档删除；几何参数外置为配置文件（与重建端 metadata 同源），CAD 坐标提为共享头/数据文件 |
| 5.4 | `spect_recon_mlem.cu`（1607 行）与主线功能重复约 70%，逻辑已迁入 `CompactSymmetricProjectorCUDA.cpp`，但作为历史对齐参考仍有价值 | 中 | 标注 deprecated/reference-only，不再并行维护；README 说明其唯一用途是 legacy 数值对齐 |
| 5.5 | 输出数据文件混在源码目录：`recon_iter050.img/.raw/.hdr`、`sensitivity.raw`（约 11MB，是 2026-05-23 legacy 对齐的参考数据） | 低 | 移到 `test_data/share/` 或 `test_data/legacy_ref/`（注意 `test_data/share/` 已有一份同名参考，先核对是否重复） |
| 5.6 | `mat_convert_bin.m` 硬编码输入文件名，换数据集要改源码 | 低 | 函数化 + 参数 |
| 5.7 | `6module_bak_src` 不在 CMake 内，编译方式无记录 | 中 | 加可选 CMake target 或在 README 写明 nvcc 编译命令与 v2/v4 参数示例 |

## 6. 物理模型未完成项（承接 `doc/Full_Project_Physics_Model_Review_20260606.md`，节点 1~9 已修复）

| # | 未完成项 | 严重度 | 状态 |
|---|----------|--------|------|
| 6.1 | **已修改（2026-07-03）** | 已完成（待物理 phantom 定量验证） | 方案：新增 `computeAttenFactorsLineIntegral`——角度相关平行射线 `exp(-∫mu ds)`（含体素间距、双线性采样、半步长 midpoint 积分），`-ctac` 默认启用（`-ctacModel line-integral`），`-ctacRotSign` 对齐生成端 `--voxel-rotation-sign`；旧近似保留为 `-ctacModel per-voxel`；slab phantom gtest 锁定角度依赖/间距缩放/旋转方向。compact 路径 CTAC 已完成（§B.4，2026-07-03）。剩余：针孔汇聚射线 Siddon 积分 |
| 6.2 | BSREM/RDP 的 beta/curvature 在真实 6 模块矩阵下未校准，beta 不可迁移；2026-06-06 Derenzo 扫描显示当前参数明显偏平滑 | 中 | 待做 beta sweep + likelihood/contrast/CV 记录 |
| 6.3 | **已修改（2026-07-03）** | 已完成（防护部分） | 方案：BSREM fused gate 与 `b_proj` 同口径拒绝 `w_proj`/`w_img`；`BSREM_step` 收到二者直接抛错，消除静默忽略风险。`w_proj`/`w_img` 数学语义定义与 CLI 接入仍按 review 第 11/15 项搁置 |
| 6.4 | projection-domain PSF 路径模型与 Operator PSF 不同（`A^T G^T(p/(G A f))`），且限制 subsets=1/单床，legacy OSL 正则写法仍在 | 低 | 保持隔离，文档写明模型差异 |
| 6.5 | MLEM legacy penalized path 仍用 `denom += beta*grad/f`（OSL），与 BSREM additive gradient 不一致 | 中 | 生产惩罚重建统一走 BSREM-style，OSL 显式命名为 legacy |
| 6.6 | **已修改（2026-07-03，审计新发现）** | 已完成 | 问题：标准 CSR 路径 Operator PSF 下 sensitivity 只算 `A^T 1`，缺 `G^T`（compact 路径正确），numerator/denominator 模型不自洽导致 PSF 重建定量偏差。方案：`MainProgram` 标准路径全局 denom（含 atten 分支）与 BSREM subsetDenom 计算后均补 `apply_psf_backward`，与 compact 路径口径一致 |

## 7. 文档与配置

| # | 问题 | 严重度 | 建议 |
|---|------|--------|------|
| 7.1 | 缺一份"当前能力矩阵"：standard vs compact 两条路径对 MLEM/OSEM/BSREM/ADMM、subsets、PSF、CTAC、bProj、multi-bed、CPU/CUDA 的支持情况散落在 5+ 份文档和 CLI 校验代码中，且部分文档（CompactSymmetricBackend_Design v4 的 Phase 1 范围）已过期 | **高** | 新建 `doc/Capability_Matrix_CN.md` 单点维护，按"公式已实现 / 物理已验证 / 真实矩阵 smoke 已通过"三列区分 |
| 7.2 | 运行时环境变量（`SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES`、`SPECT_COMPACT_CHUNK_MMAP_MIN_BYTES`、`SPECT_CUDA_SPMV_ALG`、`SPECT_CUDA_DEVICE` 若实现等）没有统一清单 | 中 | README 增加环境变量表 |
| 7.3 | **已修改（2026-07-03）** | 已完成 | 方案：`doc/MultiBed_Improvements_Summary.md` CTAC 行改为"一阶平行射线 line-integral 已实现（`-ctacModel`/`-ctacRotSign`）"，注明针孔汇聚射线未建模、per-voxel 旧近似仅供对比 |
| 7.4 | **已修改（2026-07-03）** | 已完成 | 方案：性能示例改为精确路径 `-chunkCSRDir test_data\6module\260601` |
| 7.5 | `CMakeLists.txt` 762 行单体（无硬编码绝对路径，依赖 vcpkg/find_package，整体健康） | 低 | 可拆 `cmake/*.cmake`，优先级低 |

---

## 建议执行顺序

1. **第 0 步（半天）**：§0 全部——git 基线、ignore、死链 README。这是所有后续工作的前提。
2. **第 1 批（正确性，1~2 周）**：2.1 双 CSRHeader、2.2 metadata 版本化、2.7 MAT 返回值、2.8 宽字符路径、2.3 工具/文档对齐、5.2 v4 默认模块数、4.1/4.2 260601 真实格式测试。
3. **第 2 批（结构，2~4 周）**：1.1~1.6 MainProgram/ArgumentParser 拆分与双轨接口收敛、3.1~3.2 SparseOpsCUDA/ISparseOps 拆分、1.7 OSEM/BSREM 内核合并、1.14 死兼容层删除。
4. **第 3 批（清理与文档）**：§2a 260525 退场（确认后）、5.3~5.5 bak_src 归档、7.1 能力矩阵、7.2 环境变量表、剩余低优先级项。

> 注意：§3 中的性能类改动（3.3/3.4/3.9）必须沿用既有验收口径——每次改动跑 `test_data/share` 50-iter 回归（对 `recon_iter050.img`，`rmse_over_reference_mean` ~1e-5 量级）+ 空闲系统 benchmark；`.agent/CONTINUITY.md` 记录了大量已证伪的优化方向（resident cache、mmap 默认化、SpMM 合并、subchunk、device-ratio 默认化），不要重复尝试。

---

# 后续改进方向（代码优化完成之后）

以下不属于"清理现有代码"，而是需要立项的功能/算法/工程方向，按建议优先级排列。

## A. 工程与交付

- **A.1 CI 流水线**：git 恢复后建立最小 CI——构建 + `spect_tests`（CPU 部分）+ 格式/parser 测试；CUDA 测试在自托管 runner 上跑。当前 227+ 个测试全靠手工触发，是回归风险的根源。
- **A.2 统一几何真相源（`DetectorGeometry`）**：把模块数、针孔 CAD 坐标、晶体尺寸、角度数从"生成端 `.cu` 硬编码 + 重建端 metadata 隐式假设"升级为一份共享几何描述文件（生成端读它生成矩阵并写入 metadata，重建端校验一致性，metadata 增加 `moduleCount` 字段）。这是 3→6 模块升级留下的最大结构性欠账，也是未来任何几何变更（更多模块/不同针孔布局）的前提。
- **A.3 Linux 支持收敛**：内存探测、host cache 预算、pinned 注册等目前 Windows-only（非 Windows 走 stub）；若有集群/服务器部署需求，补 `/proc/meminfo` 路径并在 Linux 上复跑 benchmark。
- **A.4 发布自动化**：RC 打包目前是手工脚本+手工 SHA256；可做成 CMake install + CPack 一键出包。

## B. 物理模型与图像质量

- **B.1 真实 line-integral CTAC**：**第一阶段已完成（2026-07-03，见 §6.1）**——角度相关平行射线 `exp(-∫mu ds)` 已实现并默认启用，slab phantom gtest 锁定角度/路径长度/旋转方向语义。剩余工作：(1) 物理 phantom（均匀圆柱 + 偏心热源）定量验证与 `-ctacRotSign` 方向确认；(2) 如需更高精度，沿针孔系统矩阵真实汇聚射线做 Siddon 积分（当前为平行射线近似，Z 向水平）。
- **B.2 散射校正**：当前只支持外部 additive `-bProj`；下一步可评估 DEW/TEW 能窗散射估计接入，作为 `b_i` 的内部生成器。
- **B.3 RDP/BSREM 参数标定与 Derenzo 目标函数**：建立 beta/curvature 在真实矩阵下的标定流程（likelihood + central contrast + background CV + edge/ring penalty score），替代当前肉眼挑参；detZ selfnorm vs raw MLEM 深迭代的取舍需要带 ring-penalized 指标重新评估。
- **B.4 compact 路径能力扩展**：**compact + CTAC 已完成（2026-07-03）**——方案为设备端"值烘焙"（value baking）：NVRTC kernel `compact_bake_atten_values` 在 SpMV 前把 chunk 矩阵值乘以 `att[fullAngle(row), R_branch^{-1}(col)]`（衰减表在未旋转床坐标系，按分支符号映射回原坐标，与 `transformBackwardAdd` 同一映射），复用现有 chunk/分支批处理与常驻缓存，烘焙结果按 (chunk, branch, 表指针) 缓存供 forward/backward 共用；`-chunkCSRDir + -ctac` 已放开（单床，fused kernel 自动禁用），denom cache manifest 记录 CTAC 参数防止脏缓存。验证：gtest 对 4 个 Sym90 分支 vs CPU 参考一致 + 单位表还原 plain SpMV；share 数据零 mu 回归 vs 基线 rel_l2=4.7e-8；水圆柱 mu 回归方向正确（sensitivity -9%，中心/环比 1.12→1.29）。剩余：OSEM subsets（角度子集与 90Z 折叠的交互需要设计）、ADMM/TV、compact CTAC 多床支持（设备衰减表缓存目前单槽）。
- **B.5 不确定度/收敛诊断**：迭代过程记录 log-likelihood 曲线与更新幅度，给用户提供停止判据，替代固定 50 iter 经验值。

## C. 性能（在 v28 基线 117.6s @ 50 iter 之上）

- **C.1 减少矩阵上传次数仍是唯一被 profile 支持的主线**：`cuda.compact_matrix_upload` 占比最高，交替 chunk 顺序（v28）已拿走顺序复用收益。下一步的合理候选是**量化/压缩矩阵值**（float16/int8 值 + scale，上传量减半，需严格数值验收）或**双 GPU/更大显存设备上的全常驻**（8GiB 卡已证伪，但 16GiB+ 卡上 4 chunk 全常驻可消掉全部重复上传——值得在目标部署硬件上重测）。
- **C.2 自定义稀疏格式 microbenchmark**：CONTINUITY 中多次提到但未做的 bounded 实验——针对本矩阵的行长分布设计 sliced-ELL/BSR 类布局，与 cuSPARSE CSR 对比 SpMV/SpMVT。先 microbenchmark 再决定是否接入。
- **C.3 多床位调度进一步融合**：chunk-outer/bed-inner 已实现；B>1 时 sensitivity 与 numerator 的 chunk 遍历还可以合并（同一次上传服务两个阶段），对多床全身扫描是纯上传次数收益。
- **C.4 double→float 主链路评估**：图像侧仍是 double（旋转/ratio/累加），矩阵已是 float32。评估图像链路 float32 化的精度损失（用现有 1e-7 级 compare 口径），潜在收益是 host 内存减半 + D2H/H2D 减半。

## D. 可用性

- **D.1 CLI 面收敛第二轮**：延续 OSMAPOSL/DD-PSF 退场的思路，审计剩余低使用率参数（`-joe`、`-Astream`、多种 reg 类型中从未在生产用过的），减小测试矩阵。
- **D.2 图形化/脚本化 QA 报告**：把 `qa_compact_cylinder.py`、`compare_recon_metrics.py` 的输出整合为单页 HTML/Markdown 报告（三视图 + 指标表 + pass/fail），作为每次重建的标准附件。
- **D.3 配置 profile 化**：为常见场景（PSC 单床、全身 4 床、Derenzo 高分辨）提供预设 profile 配置文件，减少长命令行。

---

## 审查依据

- 本轮四路并行代码审查（架构/IO/CUDA/工程卫生），2026-07-03。
- `doc/Full_Project_Physics_Model_Review_20260606.md` 与 `doc/Full_Project_Physics_Model_Fix_Log_20260606.md`（节点 1~9 已修复，未修复项收入 §6）。
- `.agent/CONTINUITY.md`（性能实验证伪记录、v28 主线决策、验收口径）。
- `test_data/6module/260601/CompactChunkCSRB_Reconstruction_README_CN.md`（格式与缓存策略）。
