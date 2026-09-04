# 实验记录：CSRQ 切片 tile backward（P3 后继）——在 sm_89 上否决

日期：2026-09-04。范围：实验路径（CSRQ）。结论：**不采用**；代码已从源码移除，内核源码存档于 `attachments/2026-09-04-csrq_backward_slice_k8.cu.txt`。

## 动机

CSRQ backward 的散射内核（每个 nnz 对 branch-major `[8][nvox]` 做 8 次全局原子加，每遍 2 个分支以保持 L2 驻留）在 120×120×220 Tc99m 矿阵上为 150 ms/遍，且已被证明卡在 L2 原子吞吐（≈22 G sector-atomic/s）。`tools/analyze_csrq_footprint_groups.py` 实测：按 (angle, z 切片) 分组，一个角度的 1152 行落在一张切片上的 22.4 万 nnz 只覆盖约 7.8 千个体素，全局原子加可减少 28.8×。设想：block 处理 (angle, z)，在 shared memory 的 fp32 整切片 tile（120×121×4 B = 58 KB，奇数行跨度消除 bank 冲突）上累加，4 个 xy 对称分支旋转后落在同一切片，z 镜像分支第二遍跑，最后每体素一次 double 原子加。

## 实现要点（已验证数值正确）

- 预热时每 chunk 建 (row, z) 索引 `zRunBegin/zNnzOff`（rows × (nz+1) × 2 × u32，183 MB/矿阵；流式矿阵存 pinned 副本随 chunk 上传）。
- 处理跨 y/z 回绕的 run：把上一切片末尾可能回绕进来的 run 纳入，按线性列重算 (y, x)，越界跳过，保证精确分区。
- 两种线程组织：(A) 批处理 128 个 run → 二分查段 → 段内偏移回溯 → 展开表 → 逐 nnz；(B) warp-per-row：每 warp 顺序处理行，32 个 run 字一批做 warp 前缀和，逐 run 由 32 lane 覆盖 nnz，ratio 与旋转仿射系数常驻寄存器。
- 单元测试（toy、32×32×2 多 tile、3 chunk 流式、双窗）全部与 CSRB 参考一致（relL2 ≤ 1e-5）；真实矿阵 it20 与散射路径 relL2 9.8e-7。

## 测量（RTX 4070 Ti SUPER，Tc99m CSRQ，Derenzo 单床，每遍全矿阵 backward）

| 变体 | backward ms/遍 | 说明 |
|---|---|---|
| 散射内核，2 分支/遍（现行） | 150 | 基线 |
| 切片 tile，批处理版 (A) | 309 | |
| (A) 把 smem 原子加换成普通写（仅计时） | 265 | 原子加只占 44 ms，簿记占主导 |
| 切片 tile，warp-per-row 版 (B) | 299 | 簿记极少，仍慢 2× |

SASS 显示 shared memory 的 `atomicAdd(float)` 在 sm_86/89 编译为 `ATOMS.CAST.SPIN`（CAS 自旋循环，无原生指令）。(B) 的瓶颈按 warp 指令数估算：run 平均 15 个 nnz、lane 利用率 <50%，每个 run 的固定开销（3 次 shuffle、起点除法、循环控制）加 4 个带分支的 CAS 循环 ≈ 200 条 warp 指令 × 2.2 亿 run × 2 遍 ≈ 130 ms 的纯发射时间，再加停顿到 300 ms；58 KB tile 使每 SM 仅驻留 1 个 block（16 warp），延迟难以隐藏。理论上再做双 run 打包、半切片 tile 提高占用率，最多接近 150 ms 打平，没有收益。

## 结论与后续

- 在 sm_86/89 上，**减少全局原子加数量的收益被共享内存 CAS 原子加和 warp 利用率抵消**，切片 tile 方案否决；保留合并散射内核（150 ms/遍）为 CSRQ backward。
- 仍可能有效但本机无法验证的方向：sm_120（Blackwell，现场 RTX 5060 Ti）支持向量全局原子加 `red.global.v4.f32`，可让现有散射内核每个 nnz 发 2 次 v4 原子而非 8 次标量原子，用 `__CUDA_ARCH__ >= 900` 分支实例化即可；需在目标机上测。
- P3 家族（footprint 进 shared memory）至此两种形态都已用数据否决，不再作为计划项；若未来换到有原生 fp32 shared atomic 的架构可重新评估，实现见附件。
