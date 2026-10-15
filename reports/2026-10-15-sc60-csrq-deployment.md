# SC60 CSRQ 部署：流式重叠、主存回退与真实数据签收

日期：2026-10-15。源码基线 `826e3f7`（含同日他人提交的 CLI 空值修复）；冻结对照 EXE 为 `2dc6ba7`（`recon_out/csrq_deploy_20261015/baseline_2dc6ba7/`）。目标：SC60 设备端用 CSRQ（18.3 GiB）替代 62.5 GiB CSRB，在 16 GB 显存、主存可能紧张的条件下可直接部署。原始矩阵、投影、历史证据与既有设备包均未修改。

## 1. 结论

- SC60 设备包默认改为 CSRQ（`multipinhole/config/scanners/SC60-MPH3.ini`），CSRB 保留为权威格式和一行配置即可回退的路径；PSC 不变。
- 本机（RTX 4070 Ti SUPER 16 GB、64 GB、NVMe）线性管 8 床、OSEM 10 子集、PSF 1.5：CSRQ 每轮 2.72 s → **1.79 s**；同输入 CSRB 为 36.2 s/轮。
- 主存不足时不再失败：锁页预算外的分块每轮从文件读回。完全不锁页仍可运行，4.2–4.3 s/轮。
- 数值：真实数据 50 轮，CSRQ 与 CSRB 全体积 relL2 **7.4e-7**，最大绝对差为最大值的 2.6e-6（门限 1e-5）。CSRQ 量化本身有损，不能称逐位无损；在本数据的整个视野内与 CSRB 工程等价。
- RTX 5060 Ti 未实测。按显存带宽（448 vs 672 GB/s）与 PCIe（5.0 x8）推算约 2.5–3.5 s/轮，属于估计，须以现场短重建计时为准。

## 2. 定位：上传与计算没有重叠

Nsight Systems 2025.6.3 采集冻结基线 3×10 子集（8 床，分母命中），取第 2–3 轮 `compact_mlem_update` 之间的区间：

| 区间 5.340 s | 秒 |
|---|---:|
| kernel 并集 | 3.601 |
| H2D 并集 | 1.692 |
| kernel ∩ H2D | **0.030** |

逐事件序列显示：每个子集只有约一半分块含选中角度（每块 5 个 compact 角，子集步长 10），但 `csrqForEachChunkLocked` 仍让未选中块占用一个槽位。下一个选中块因此总落在上一个选中块正在使用的槽，上传必须等它的 8 床计算完毕。kernel 中 backward 占 59%，forward 29%，逐块打包/对称回写约 10%。

## 3. 修改

1. **流式遍历只访问含子集行的分块**，槽位按访问序号交替；当前块 kernel 先提交，再排下一块上传（读文件时可与 GPU 重叠）。驻留矩阵与 N1 路径行为不变。
2. **无 CTAC 时每床输入/反投目标在子集遍历内常驻**：每床只打包一次、对称回写一次（显存不足 1 GiB 余量时打印 NOTE 并回退逐块路径）。各块反投在 fp32 内跨块累加后进入 double 累加器——唯一的数值变化来源。
3. **主存锁页预算** `SPECT_CSRQ_HOST_PIN_BYTES`（auto 同 CSRB host cache 规则）。超预算或 `cudaMallocHost` 被拒的分块只锁页行指针/行最大值；run/值按所选角度区间从原文件读回，经两块 256 MiB 锁页中转上传；≥32 MiB 的区间分 4 段并发读取。文件大小或修改时间变化即报错。预热仍完整读取并校验每块一次。日志 `[NOTE] CSRQ matrix ... pinned host k/n chunks ...; ... re-read from file` 与 `[MATRIX_IO] scope=run csrq_file_reread_bytes=`。
4. 读文件异常时先排空上传流和默认流再抛出；释放时同步后再释放中转缓冲。

未改：矩阵格式、kernel 算术、CTAC/灵敏度路径、packed backward 默认（仍关）、公开 CLI。

## 4. 配对计时（本机，分母命中，5×10，稳态均值）

驱动 `recon_out/csrq_deploy_20261015/run_arm.py` 在每臂开始前等待其他会话的 `SPECT_recon` 结束（本机同时有他人实验，EXE 本身也拒绝共用 GPU），并记录运行期间是否出现他人进程（均未出现）。系统文件缓存未清空。

| 臂 | 顺序 | s/轮 | 5 轮 core s |
|---|---|---:|---:|
| 基线 2dc6ba7 | A1 / A2 | 2.720 / 2.716 | 13.602 / 13.581 |
| 候选 v1 | B1 / B2 | 1.795 / 1.794 | 9.027 / 8.999 |
| 全不锁页 v1（单线程读） | a / b | 8.407 / 8.353 | 42.06 / 41.78 |
| 全不锁页 v2（4 段并发读） | a / b | 4.301 / 4.201 | 21.46 / 21.25 |
| 锁页 6 GiB：v1 / v2 | 单次 | 6.208 / 3.240 | 31.03 / 16.19 |
| 全锁页 v2 | 单次 | 1.831 | 9.169 |

core 降幅 33.6%（两对一致）；并发读取使全读回模式降 49.5%。全读回 5 轮共读回 98.4 GB（19.7 GB/轮）。

## 5. 真实数据数值签收

输入：`E:/Xiang_Project/SPECT/6Module/experiments/sc60/linxinguan_2168_20261014/pre/Bed1..8.bin`（冻结副本在 `inputs/linxinguan8/`）；SC60 设备包配置（12 mm 步进、PSF 1.5 sieve/castor、圆柱 36 mm、输出 -60°），OSEM 50×10，每 10 轮保存。参考为冻结基线 EXE + 原生 CSRB（自动 host cache，36.2 s/轮）。区域：全体积；半径 ≤14 mm；再叠加轴向灵敏度剖面 ≥10%/50% 的层（输出旋转保持圆柱居中，故按半径与轴向构造）。

| it50 | 全体积 relL2 | r≤14 mm 且轴向≥50% | max\|Δ\|/max |
|---|---:|---:|---:|
| 基线 CSRQ vs CSRB | 5.04e-7 | 4.19e-7 | 2.54e-6 |
| 候选 CSRQ vs CSRB | 7.41e-7 | 6.83e-7 | 2.57e-6 |
| 候选 vs 基线 CSRQ | 6.26e-7 | 6.25e-7 | 1.32e-6 |
| 候选 全读回 vs 全锁页 | 5.43e-8 | 5.38e-8 | 1.48e-7 |

差异随迭代缓慢增长（候选 vs CSRB：it10 5.1e-7 → it50 7.4e-7），最大差位于轴向末端低灵敏度层（轴向剖面 3%）。2026-10-11 合成 12 床全体积 0.0926 的失败未在本真实数据出现；该合成例的根因仍未单独定位，不能据本次结果关闭。

CSRB 自身浮点次序噪声基线（同 EXE、同矩阵，`SPECT_CSRB_SCATTER_BACKWARD=0` 对 auto 选中的 scatter 反投）：it10/30/50 全体积 relL2 3.8e-8/4.4e-8/4.8e-8，max|Δ|/max 1.1e-7。CSRQ 与 CSRB 的差异约为该噪声的 10–15 倍，来自 u16 量化与不同 kernel，量级仍远低于 1e-5。

## 6. 回归

- 回归：`spect_tests` 361 项，357 通过，4 项既有条件跳过，0 失败（`full_suite_conv.xml/log`）。新增：子集×{驻留、流式锁页、流式全读回(64 字节中转)}×packed、3 块中 1 块锁页 4 KiB 中转、驱动拒绝锁页后归还已锁页分块、文件区间读与改动拒绝、80 MB 并发区间读。
- 首次全套在他人 CSRB 运行占满主存时失败 1 项：测试依赖自动锁页预算，可用内存 <8 GiB 时预算为 0，属预期策略；测试改为显式设定预算后通过。
- 原生观测模型脚本见第 9 节。

## 7. 部署说明与剩余风险

- 设备包：`E:/Xiang_Project/SPECT/6Module/release/devices_20261015/SC60-MPH3/SPECT_Core.zip`（及 PSC 同 EXE 包），由 6Module 分支 `sc60-csrq-deploy-20261015`（基于 `eb0ea45`，不含当天他人预处理实验提交）组装；`verify_deployment.py` 允许 CSRQ 目录的 `conversion_validation.json`，并按分块头与该记录核对文件大小（防截断，非内容哈希）。详见第 9 节。
- `conversion_validation.json` 的 status 文本仍写着 9-28 时的“runtime unsupported”，已过时；保留原文件。
- 主存：流式全锁页需约 18.3 GiB 不可换页内存。启动时可用内存不足 26 GiB 左右时自动部分读回；矩阵须在 SSD/NVMe 上，HDD 上读回会很慢。
- 未覆盖：RTX 5060 Ti/sm_120 实测；CSRQ 多床 CTAC（联合多床本身尚不支持 CTAC）；双窗 CSRQ 真实数据；packed backward（sm_120 原子 float4）未启用。

## 8. 交叉审查与收敛（code-optimization-convergence v2.0）

独立代码审查确认流/事件顺序、锁页缓冲生命周期、文件偏移、每床暂存和并发读取无误，指出 1 项问题并已修复：

- **P1 驱动拒绝锁页时回退可能失败**：原实现在所有分块锁页之后才申请两块 256 MiB 中转缓冲，且后续分块的行指针仍用会抛错的分配；剩余可锁页内存不足时重建中止，与“回退读文件”的承诺矛盾。修复：首个读回分块出现时立即保留中转缓冲；行指针、中转缓冲的锁页失败时依次归还最近锁页的分块载荷（该块改为读文件）后重试，无可归还时才明确报错。新增测试钩子（存活锁页数上限）模拟驱动拒绝，3 块中 2 块被归还、全部读回，数值与 CSRB 参考一致。

按“删除复杂度、保留必要回退”的原则，对本次 SC60 生产路径（CSRQ）做的收敛：

| 项 | 证据 | 处理 |
|---|---|---|
| P1 `SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES` 对 CSRQ 开启可分页主机缓存 | CSRQ 迭代从不读该缓存；旧设备说明曾要求设置此变量，设置后 18 GB 可分页副本与 18 GB 锁页副本并存 | CSRQ 忽略此变量 |
| P2 CSRQ 三个遍历函数逐（块×床）CUDA event 计时 | 每轮约 2900 次 event elapsed 查询，结果只进入 `-vb 2` 的 `cuda.csrq_fwd/bwd`；同一调用已有包含同步的墙钟计时 | 删除计时器向量、扩容函数、每床回写计时器、`maxAngles` 及两组 API 计时参数；保留调用计数供测试确认走 CSRQ 内核 |
| P2 CSRQ 预热两条 NOTE | 第一条（流式原因）与第二条（结果）重复 | 合并为一条：槽大小、原因、锁页 k/n 块、读回量；去掉 `scratch_bytes` |
| P2 CSRQ 运行打印 CSRB 专用 `[MATRIX_IO] scope=iterations` | CSRQ 下恒为 0 | 仅 CSRB 打印；CSRQ 只在有读回时打印读回行 |

保留（有依据的必要实现）：显存不足时的逐块路径回退（每床暂存放不下时）；`SPECT_CSRQ_BACKWARD_PACKED` 实验路径——它是为 sm_120 float4 原子准备的，设备 RTX 5060 Ti 正是 sm_120，现场一行环境变量即可试验，本机 sm_89 无法评估，删除会丢失唯一可在设备上低成本验证的反投优化路线；CSRB 的 locality 采样（`sortedHint` 仍决定 CSRB packed-forward 分块，不是纯诊断）。

验证：CSRQ 18 项及全套 361 项（357 通过、4 项原有条件跳过）；Compute Sanitizer memcheck 对流式子集/读回/拒绝锁页用例 0 错误。收敛版本的配对计时与 50 轮真实数据结果见第 9 节。

## 9. 补记：收敛版本与可部署包

| 检查 | 结果 |
|---|---|
| 收敛前后配对（v2 / v3，ABBA，5×10） | 1.719/1.711 vs 1.747/1.700 s/轮，无可测差异（收敛以简化为目的） |
| v3 真实数据 50×10 | 1.709 s/轮，core 85.4 s（冻结基线 CSRQ 139.5 s、2.79 s/轮）；vs CSRB relL2 7.41e-7，vs v1 5.4e-8 |
| 原生观测模型（无/有背景） | 64 组 + 2 组 operator PSF+ROI 闭合全部 PASS |
| 设备包端到端（解包 + 矩阵硬链接 + `verify_deployment` + 包内 `run_6module_recon.py`） | 校验 PASS；单条 NOTE `streamed through 2 x 1073 MiB device slots ..., pinned host 18/18 chunks`；1.73 s/轮，含冷分母 106 s；与直接运行 relL2 1.8e-7，与 CSRB 7.4e-7 |

可部署包（重建 `20261015+4e89651`，由独立干净 worktree 构建；6Module 工具 `3f64148`，分支 `sc60-csrq-deploy-20261015`，基于 `eb0ea45`，未合入 main）：

- `E:/Xiang_Project/SPECT/6Module/release/devices_20261015/SC60-MPH3/SPECT_Core.zip`
- `E:/Xiang_Project/SPECT/6Module/release/devices_20261015/PSC30-MPH6/SPECT_Core.zip`（同一 EXE，PSC 配置不变）

本地 CSRQ 矩阵目录中有百度网盘上传临时文件 `chunk_012.csrq.baiduyun.uploading.cfg`，校验器与包装器都会正确拒绝含该文件的目录；设备端下载的目录不应包含它。

本机运行期间他人会话也在使用 GPU（EXE 自带独占锁）；所有计时臂均在确认无其他重建进程后启动，运行中未出现他人进程。

## 复现

```powershell
cd recon_out/csrq_deploy_20261015
python run_arm.py <新标签> --exe <EXE> [--matrix csrb] --iterations 50:10 --save-every 10 --cache <缓存名> [--env K=V]
python compare.py <测试臂> <参考臂> --out analysis/<名>.json
```

`sequence.py`、`sequence2.py` 为本次实际执行顺序；`baseline_2dc6ba7/`、`candidate_v1/`、`candidate_v2/` 为冻结 EXE 与补丁，`runs/*/summary.json` 记录命令、逐轮时间、`[PERF]`/`[MATRIX_IO]` 与他人进程情况，`analysis/*.json` 为比较结果，`runs/base_nsys/timeline.nsys-rep` 为基线时间线。
