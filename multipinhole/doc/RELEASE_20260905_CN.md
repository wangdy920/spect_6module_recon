# SPECT_recon 2026-09-05 本地候选版本

本包用于本轮代码收敛后的发布前验证。CSRB 保持生产路径；CSRQ、双能窗联合重建、流式驻留与 packed backward 仍是实验能力。矩阵与采集数据不随包分发，运行时指定已经部署的 CSRB 目录。

## 本轮变更

- 第二轮审查（基线 `e5d376a`）：进一步删除无调用的全矩阵／子集／行区间转换与缓存残留，CSRB 预取收为独立单流；投影／背景指针表只建一次，主机分子跨迭代复用并避免重复清零，梯度与 κ 图按需分配。代码及构建定义净减少 744 行。驻留路径省去 `8 × 全局体素数` 字节的主机梯度缓冲，当前单床约 24.17 MiB；这是分配量推导，不是进程峰值或端到端加速实测。
- 移除已在 8 月拒绝的 Phase-B 转置变体及 `SPECT_COMPACT_T_SPMM` 环境变量，连同其主机转置缓存、页注册、设备半块槽、上传流和 warmup API。旧实验源码可从 `a3239d7` 检出，原结论保留在 Phase-B 报告。
- 保持默认 packed-K 正向、CSR_ALG2 反投、矩阵数值、重建默认配置和输出处理顺序。源码、头文件与两个实验驱动脚本合计净减少 629 行。
- 同步 CTAC 因子分解、分母缓存、BSREM 设备更新、88/92 字节 metadata 和单床输出的文档。默认单床为 `step+10` 层（当前 66），`-singleBedStepAligned 0` 保留 72 层兼容输出；坏探测器默认表仍包含 68。

本轮继承的 9 月 4 日优化包括：CSRB/CSRQ CTAC 因子分解及切片双缓冲、设备端 BSREM 更新、双窗共享衰减表、CSRQ 显存不足时降为流式并重试。完整历史见 `bin/SPECT_recon.exe -v`。

## 已完成验证

开发构建清理前后均完整构建并运行 CTest：283 通过、4 跳过、0 失败。跳过的是显式启用的五数据集输出滤波测试，以及依赖本机缺失 `260601` 目录的 3 项 IO 测试。既有 LNK4098 静态/动态 CRT 链接告警未在本轮更改。

| 工作负载 | 数值结果 | 门限 |
|---|---|---|
| CSRB Derenzo，50 次迭代 | 对 e5d376a RC relL2 4.9185e-8 | 1e-7 |
| CSRB 主机 MLEM，it1–3 | 最大 relL2 2.9718e-8 | 1e-7 |
| CSRB 主机 BSREM／驻留 BSREM κ off，it1–5 | 最大 relL2 1.1113e-8／1.0469e-8 | 早期结构回归 1e-6 |
| CSRQ 双窗，仅第二窗带合成背景，it1–3 | 最大 relL2 3.1931e-8；仅接口回归 | 1e-7 |
| CSRB CTAC，MDP 两床真实 μ 图，it5/10/15/20 | 对 09-04 因子分解输出最大 relL2 5.4068e-8 | 1e-5 |
| CSRQ CTAC，同两床与保存点 | 对 09-04 P3 输出最大 relL2 6.2355e-8 | 1e-5 |

配置、投影、μ 图在各次回放前后均核对 SHA-256。上表为第二轮源码候选构建回归；当前包验证收据为本机 `recon_out/convergence_buffers_20260905/release_verification.json`，包含最终包内 exe 的 it50 复验。第一轮 `e5d376a` RC 及其 `recon_out/release_convergence_20260905/release_verification.json` 保留为基线。包内 `SHA256SUMS.txt` 可用于交付完整性核验。

## 未关闭的门与使用边界

- 冻结的 `Sampling_Approach` 五床/50 次迭代性能门：本机目录缺失；历史 126.261 s 仍是保护记录，120 s 目标未关闭。本轮 120×120×220 回归不能替代该门。
- RTX 5060 Ti 的 packed backward 性能待现场验证；`SPECT_CSRQ_BACKWARD_PACKED` 默认关闭。本机 sm_89 的 scalar fallback 更慢，不能据此承诺 sm_120 加速。
- CSRQ 尚未获生产采纳。Lu-177 双窗真实 CT 验证仍需对应厂商重建 CT 与正确的逐窗输入；水模型的能量指数不能证明骨材料的精确缩放。
- BSREM 较长迭代存在累加噪声放大；原设计的逐迭代 1e-12、严格 RDP 有限差分和双窗 BSREM penalty-once 专项不能由现有早期迭代或单元测试结果替代。

## 版本与启动

本地候选目录：`SPECT_recon_rc_20260905_buffers`；第一轮 `SPECT_recon_rc_20260905_convergence` 保留为基线。源码身份以 `bin/SPECT_recon.exe --version` 的提交号为准；正式构建不得带 `-dirty`。

```powershell
.\bin\SPECT_recon.exe --version
.\bin\SPECT_recon.exe -df <Bed1.bin> -chunkCSRDir <CSRB目录> -it 50:50 -fout <输出目录>\recon
```

默认配置位于 `bin/config/`。本包不等于上述未关闭验证门已经通过，也不改变 CSRQ 的实验地位。
