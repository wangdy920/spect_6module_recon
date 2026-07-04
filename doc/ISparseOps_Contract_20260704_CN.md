# ISparseOps 接口契约（PR19，§3.2）

- 版本：2026-07-04
- 契约作者/签核：Fable 5（高级架构审查者，按计划 §5 要求契约先行签核）
- 基线：codex/pr18-sparseops-split @ 97f77de（PR17→PR18 完成后状态,ISparseOps.h 831 行）
- 状态：**已签核**——实现代理不得偏离本契约；如发现契约与代码事实冲突,停止并上报,不得自行裁决。

## 1. 目标与不变量

把 831 行的 `include/interfaces/ISparseOps.h` 拆为**纯接口 + 下沉实现**,并给出两级接口视图。

**硬性不变量（违反任何一条即回滚）：**
1. 能力守恒：三个后端(`SparseOpsCPU`、`SparseOpsRef`、CUDA facade)在拆分前后对每个方法的分派结果**完全不变**(见 §4 能力矩阵)。
2. 数值零漂移：下沉的模板实现逐 token 移动,不改任何公式、循环顺序、并行策略、`eps`、`kAttenUint16Scale` 用法。
3. ABI 面向调用方不变：`opts.sparseOps->xxx(...)` 所有现存调用点(OSEMReconstructor、BSREMReconstructor、ProjectionOps、ReconLoopHelpers、StandardReconLoop)**一行都不用改**。
4. 虚调用粒度仍为"整次 SpMV",严禁引入元素级虚调用。
5. `SPECT_DETERMINISTIC`、`_OPENMP`、AVX2 宏守卫行为不变。

## 2. 接口分层（概念视图,不引入新抽象类）

**裁决:不做 `ISpmvCore` 独立基类拆分。** 理由:三个后端全部同时实现 core + subset 两组能力(见 §4),拆成两个抽象类不会减少任何实现、只会引入 `dynamic_cast` 或双继承复杂度,违背"迁移后能力守恒"的最小改动原则。分层仅作为头文件内的**分区注释与排序约定**:

```
ISparseOps (唯一抽象类,不变)
├── [Core 区] backend / spmv / spmvT / computeNormAT / spmvScaled
│     · CSRView(double val) 与 io::CSRView(float val) 各一组重载
└── [Subset/Atten 扩展区] spmvSubset / spmvTSubset /
      computeNormAT_Atten / computeSubsetSensitivity
      · 同样双重载;默认实现委托到下沉的 *Impl 模板
```

## 3. 方法签名与归属总表

| # | 方法 | 重载 | 纯虚? | 默认实现位置(拆分后) | 覆盖者 |
|---|------|------|-------|---------------------|--------|
| 1 | `BackendType backend() const` | — | 纯虚 | — | CPU/Ref/CUDA |
| 2 | `void spmv(const CSRView&, const double*, double*, int) const` | CSRView | 纯虚 | — | CPU/Ref/CUDA(CUDA 为 fail-fast throw) |
| 3 | `void spmvT(const CSRView&, ...) const` | CSRView | 纯虚 | — | 同上 |
| 4 | `void computeNormAT(const CSRView&, double*, int) const` | CSRView | 纯虚 | — | 同上 |
| 5 | `void spmvScaled(const CSRView&, ..., double alpha, double beta, int) const` | CSRView | 虚+默认 | 头文件保留或下沉(§5.3) | CPU/Ref |
| 6 | `void spmv(const io::CSRView&, ...) const` | io::CSRView | 虚+默认 | 同上 | CPU/Ref/CUDA |
| 7 | `void spmvT(const io::CSRView&, ...) const` | io::CSRView | 虚+默认 | 同上 | CPU/Ref/CUDA |
| 8 | `void computeNormAT(const io::CSRView&, double*, int) const` | io::CSRView | 虚+默认 | 同上 | CPU/Ref/CUDA |
| 9 | `void spmvScaled(const io::CSRView&, ...) const` | io::CSRView | 虚+默认 | 同上 | CPU/Ref |
| 10 | `void spmvSubset(const CSRView&, const double*, const std::vector<int>&, const SparseSubsetContext&, double*, int) const` | CSRView | 虚+默认→`spmvSubsetImpl` | 同上 | —(全走默认) |
| 11 | `void spmvSubset(const io::CSRView&, ...) const` | io::CSRView | 虚+默认→同 Impl | 同上 | CUDA |
| 12 | `void spmvTSubset(const CSRView&, ...) const` | CSRView | 虚+默认→`spmvTSubsetImpl` | 同上 | — |
| 13 | `void spmvTSubset(const io::CSRView&, ...) const` | io::CSRView | 虚+默认→同 Impl | 同上 | CUDA |
| 14 | `void computeNormAT_Atten(const CSRView&, const std::vector<std::vector<uint16_t>>&, int, double*, int) const` | CSRView | 虚+默认→`computeNormAT_AttenImpl` | 同上 | — |
| 15 | `void computeNormAT_Atten(const io::CSRView&, ...) const` | io::CSRView | 虚+默认→同 Impl | 同上 | CUDA |
| 16 | `void computeSubsetSensitivity(const CSRView&, const std::vector<int>&, size_t, const SparseSubsetContext&, double*, int) const` | CSRView | 虚+默认→`computeSubsetSensitivityImpl` | 同上 | — |
| 17 | `void computeSubsetSensitivity(const io::CSRView&, ...) const` | io::CSRView | 虚+默认→同 Impl | 同上 | CUDA |

注:签名以基线代码为准,上表省略参数名;实现代理**必须**从基线逐字复制签名,不得"顺手"调整参数名/默认值/const。

## 4. 能力矩阵（拆分前后必须逐格相等）

| 方法 | SparseOpsCPU | SparseOpsRef | CUDA facade | CUDA stub(无 CUDA 编译) |
|------|--------------|--------------|-------------|--------------------------|
| backend | override | override | override(CUDA) | override(CUDA) |
| spmv/spmvT/computeNormAT (CSRView) | override(AVX2/OMP) | override(标量) | override(fail-fast throw,double CSR 不上 GPU) | override(throw) |
| spmvScaled (CSRView) | override | override(标量) | 默认 | 默认 |
| spmv/spmvT/computeNormAT (io::CSRView) | override | override(标量) | override(GPU) | 默认 |
| spmvScaled (io::CSRView) | override | override(标量) | 默认 | 默认 |
| spmvSubset/spmvTSubset (CSRView) | 默认(Impl) | 默认(Impl) | 默认(Impl) | 默认(Impl) |
| spmvSubset/spmvTSubset (io::CSRView) | 默认(Impl) | 默认(Impl) | override(GPU) | 默认(Impl) |
| computeNormAT_Atten (CSRView) | 默认(Impl) | 默认(Impl) | 默认(Impl) | 默认(Impl) |
| computeNormAT_Atten (io::CSRView) | 默认(Impl) | 默认(Impl) | override(GPU) | 默认(Impl) |
| computeSubsetSensitivity (CSRView) | 默认(Impl) | 默认(Impl) | 默认(Impl) | 默认(Impl) |
| computeSubsetSensitivity (io::CSRView) | 默认(Impl) | 默认(Impl) | override(GPU) | 默认(Impl) |

实现代理在动代码**之前**必须先用 git grep 验证此矩阵与基线一致;发现出入→停止上报。

## 5. 下沉方案（唯一批准形态）

**形态:非模板自由函数 + 显式双实例化,不用 pImpl。** 理由:`*Impl` 模板只有两个实例化点(`CSRView`、`io::CSRView`),显式实例化即可全部离开头文件;pImpl 会改对象布局,风险不对称。

1. 新建 `include/interfaces/detail/SparseOpsDefaultsDecl.h`(或等价路径):声明
   ```cpp
   namespace spect::sparse_detail {
     template <typename MatrixType>
     void spmvSubsetImpl(const MatrixType&, const double*, const std::vector<int>&,
                         const SparseSubsetContext&, double*, int);
     // spmvTSubsetImpl / computeNormAT_AttenImpl / computeSubsetSensitivityImpl 同理
   }
   ```
2. 新建 `src/sparse/SparseOpsDefaults.cpp`:
   - 逐 token 迁入四个 `*Impl` 模板体 + `sparseops_effective_threads` / `sparseops_should_use_threadlocal_accum`(后两者保留在类 protected 区的**声明**,定义移到 .cpp 或以 free function 形式共存——二选一,但 SubsetUtils.h/PR13 未来合并时依赖 `should_use_threadlocal_accum` 的全局版本,不得删除任何现有符号)。
   - 文件尾部显式实例化 8 个符号(4 模板 × 2 矩阵类型)。
   - `<immintrin.h>`、AVX2 宏、`_OPENMP` 守卫随实现体一起搬走;头文件里不再包含 `<immintrin.h>`。
3. `ISparseOps.h` 中虚方法默认体保留一行委托:`sparse_detail::spmvSubsetImpl(A, x, subset_rows, ctx, y, threads);`。
   - 表 #5–#9 的小型默认体(spmvScaled 缩放逻辑、io::CSRView 标量 spmv/spmvT/computeNormAT)**允许留在头文件**(体量小、无 AVX2 依赖);若代理选择一并下沉,须同样逐 token 且在报告中说明。
4. `SparseOpsDefaults.cpp` 加入 CPU 侧库目标(与 SparseOpsCPU 所在目标一致,CMakeLists.txt 参照现有 src/sparse/ 或 src/app 源列表位置);CUDA 构建同样链接该文件(默认实现是 CUDA facade 的 CSRView-subset 回退路径)。
5. 头文件保留:`SparseSubsetContext` 定义、类注释、分区注释(§2)。目标:拆分后 ISparseOps.h ≤ 300 行。

## 6. 禁止事项(针对本 PR)

- 禁止改任何调用点文件;禁止改三个后端的 override 集合。
- 禁止把 `SparseSubsetContext` 挪出本头文件。
- 禁止引入新虚方法、新枚举、新配置项。
- 禁止对 *Impl 做"顺手优化"(合并循环、调 schedule、换 atomic 策略等)。
- 禁止修改 include 顺序以外的任何现有 include 集合(新文件除外)。

## 7. 验收门禁

- T:CPU spect_tests 恰好 257 passed / 3 skipped;CUDA scaffold 恰好 25 passed / 2 skipped。
- S(smoke)与 R50:数据恢复后由主管补跑(计划既定延后项)。
- 结构:ISparseOps.h ≤ 300 行;`git grep spmvSubsetImpl include/interfaces/ISparseOps.h` 只出现在委托行。
- 审查:Fable 5 按 §6.1 五检查项复核 diff。
