# SPECT_recon 整体流程（数据加载与重建）

本文梳理两大阶段：数据加载（含系统矩阵构建）与迭代重建；并给出与需求对齐的可选模块说明。

## 一、数据加载与系统矩阵构建

- 解析参数与线程：解析到 ReconArgs；`-th N` → 在进入重建前 `omp_set_num_threads(N)`。
- 投影加载与重排：
  - 单床位：`readMat(proj.mat, "matrix")` → `reorderProjection(...)`；
  - 多床位：`-dfList <f1,f2...>`，依次读取并重排，在内存中拼接为长向量；投影射线总数需等于 `rows * beds`。
- 背景/散射投影（可选）：
  - 单床位：`-bProj <file>`；
  - 多床位：`-bProjList <f1,f2...>`，与投影一一对应加载拼接。
- 虚拟床位堆叠（Multi-bed）：若输入多个床位，系统矩阵仅加载单床位（CSR/CSRView），在投影/反投影计算中通过隐式偏移（Implicit Offset）实现多床位逻辑（节省 N 倍系统矩阵内存）。注意：多床位+映射模式（`-mapCSR on`）下支持 `-ctac`（动态衰减因子，不修改只读 CSR；会带来额外内存/计算开销）。
- 圆柱掩码（可选）：`-cyl <diam,height>`，可配合 `-cylc <cx,cy,cz>`；打印覆盖比例并导出一次 `*_mask.mhd/.raw`；掩码用于迭代内裁剪与 CSR 构建期列剪枝（`-ApruneMask on`）。
- 身体掩码（可选）：`-maskFromAtt <th>` 从 `-attMap` / `-prior` 生成二值身体掩码，并在每次迭代更新后将掩码外体素强制置零。
- CSR 获取（3 选 1）：
  1) `-loadCSR <file>`：若配合 `-mapCSR on` → 以内存映射读取（CSRView），否则读入内存（CSR）；
  2) `-Astream on`（默认）：`buildCSR_from_mat_stream(...)` 两遍扫描、按列读；|w|≤`-Aprune`（默认 1e-10）剪枝；可 `-saveCSR` 持久化；
  3) `-Astream off`：`readMat` → `buildCSR_from_mat(...)`；
- CSR 布局与精度：`row_ptr(uint32)`、`col_idx(uint32)`、`val(float)`；计算统一转为 double（与 MATLAB 行为一致）。

## 二、迭代重建（MLEM/OSEM/BSREM‑II）

- 算法选择：默认 MLEM；`-subsets > 1` 即 OSEM；BSREM‑II 可在配置中启用（`algorithm=BSREM`），也可用命令行 `-alg BSREM -bsrem <alpha0[,gamma]>` 直接选择并覆盖配置。
- 配置：`opts.threads = args.numThreads`；`opts.n_subsets = args.numSubsets`；按参数准备 PSF/平滑/Butterworth；`opts.mask = maskPtr`。
- 归一化与初始化：`denom = computeNormAT(A)`（CSR/CSRView）；`image` 初始化为 1，准备正则梯度与缓存。
- 迭代主循环（it = 1..numIters）：
  - 子集：`make_simple_subsets(rows, n_subsets)`（映射/非映射分别取行数）；
  - 每子集：计算正则梯度 → `compute_reg_gradient(...)`；调用 `OSEM_step`（CSR 或 CSRView 重载）；
  - 迭代后：可选 3D 高斯平滑与非负钳制；或 3D Butterworth 与非负钳制；
  - 保存：按保存间隔导出体数据（保存时对掩码外体素置零）。
- 结束：保存最终体数据并打印摘要。

## 三、与需求对齐的可选模块

1) 三线性 / B‑spline 基（Joseph‑like B）

- 目的：在归一化与迭代内引入平滑基函数，降低采样非理想导致的锯齿/噪声。
- 操作：
  - 分母：`back_proj0 = B(A' * 1)`；
  - 迭代内：`F ← B(F)` 与 `back_proj ← B(back_proj)`；
  - 实现：B 用一维可分核 `[0.25, 0.5, 0.25]` 在 x/y/z 方向各做两遍卷积（两次一维近似 B‑spline/三线性）。
- 对接位置：`computeNormAT` 之后对 `denom` 应用；`OSEM_step` 后对 `back_proj`、更新前/后对 `image` 应用；可复用 Filters 的可分卷积框架（在现有 3D 高斯接口旁新增 B‑spline 核）。
- 状态：已纳入流程建议；代码侧可按上述钩子启用。

1) PSF（投影域逐角度 2D 卷积，可选）

- 目的：按角度逐帧对投影做 2D 卷积（`detNy × detNz × numAngles`），核对称、前/后向一致。
- 操作：前向将该角度切片 reshape 为 `detNz×detNy` 做 2D 卷积后回写；反向对比值 `R` 做同核 2D 卷积后回投。
- 对接位置：在前向/反向路径加入 per‑angle 2D 卷积（可参考 matlab_ref 的实现）。
- 状态：当前默认提供图像域 3D 可分 PSF（Operator 模式）；投影域 PSF 仅在 MLEM 且 `-subsets 1` 时启用；如需严格投影域 2D 卷积，可按此接口扩展。

1) 正则化（OSL‑MAP‑EM，乘性/除性）

- TV / Tikhonov / MRP：以当前迭代图像 `F_k` 计算梯度/拉普拉斯/中值先验项，并在 OSL 更新中以除性/乘性方式融合。
- NLM：非局部均值先验，利用 patch 相似性抑制噪声并保持纹理。
- Bowsher / PLS：基于解剖先验图像的正则化（需 `-prior` 或 `-attMap` 提供先验）。
- 对接位置：使用 `compute_reg_gradient` 的结果在 OSL 更新时以乘性/除性方式融合，并对更新结果做非负/限幅处理。

1) 后处理（迭代外）

- 3D 高斯 或 3D Butterworth（二选一更稳健）；保存前可叠加圆柱掩膜置零；已在流程中支持并可通过参数选择。

## 四、线程与性能要点

- 线程：`-th N` → `omp_set_num_threads(N)`，并在并行段使用 `num_threads(N)`；并行度受 OpenMP 链接与系统可用核心约束。
- 大矩阵推荐：首次 `-Astream on -Aprune 1e-10 -ApruneMask on -saveCSR <file>`；后续 `-loadCSR <file> -mapCSR on` 零拷贝加载与更低内存占用。
