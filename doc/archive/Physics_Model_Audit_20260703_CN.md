# 物理模型原理审计（PSF / RDP‑BSREM / 散射 bProj）— 2026‑07‑03

范围：应 B.2/B.3「先从原理核查、暂无参考数据」的要求，对 PSF 算子、MLEM/OSEM+RDP/BSREM 正则化、散射背景 `bProj` 三块做源码级审计。所有结论均对照源码逐行核实（非仅探查报告转述）。compact+CTAC（B.4）另见 `doc/Compact_CTAC_Design_20260703_CN.md`。

结论速览：

| 项 | 结论 | 动作 |
|---|------|------|
| PSF 前后投影一致性 | **发现并已修复一处模型不自洽**（标准路径 denominator 缺 `G^T`） | 已修（清单 6.6） |
| PSF 空间不变假设 | 数学实现正确；物理上是针孔系统的近似，需点源数据定级 | 待数据 |
| RDP 梯度/能量公式 | 公式正确（Nuyts 2002 形式 + 数值稳定项）；`delta` 命名与教科书 RDP 不同义 | 文档澄清 |
| RDP 边界条件 | 体积边界跳过邻居（Neumann 型）、mask 可选跳过——合理 | 无 |
| BSREM 超参数 | 结构正确；curvature/kappa 是启发式预条件而非解析曲率，β 不可迁移 | 待 beta sweep |
| MLEM OSL 惩罚路径 | subsets>1 时惩罚权重被放大约 n_subsets 倍（分子子集/分母全局不匹配） | 保持 legacy 定位（清单 6.5） |
| bProj 数学语义 | `p/(Af+b)`、灵敏度不含 b——正确 | 无 |
| bProj 输入防护 | 只查尺寸，不查负值/单位 | 建议小修（可派发） |
| 散射估计本体 | 仓库内无 DEW/TEW，b 全靠外部——原理上可行，需协议文档 | 待数据 |

---

## 1. PSF 算子审计

### 1.1 实现现状（已核实）

- 模型：`P = A · G`，`G` 为三轴可分离 1D 高斯（`filters::conv3d_separable_inplace`），3σ 截断，镜像边界（`conv1d_{x,y,z}_mirror`，`src/filters/Filters.cpp`）。
- 前投影：`apply_psf_forward` 先模糊图像再 spmv；反投影：`apply_psf_backward` 对 `A^T r` 再卷积。两者用同一核——因高斯核对称，`G^T = G`（镜像边界下的自伴性在边缘有次阶偏差，可接受）。
- CUDA 卷积失败时降级 CPU；有"输入非零但输出全零"防护（历史事故回归）。

### 1.2 发现的问题与修复（2026‑07‑03）

**标准 CSR 路径的 sensitivity 只算 `A^T 1`，缺 `G^T`。**

MLEM 不动点要求分子分母来自同一前向模型：`f ← f · (G^T A^T r) / (G^T A^T 1)`。修复前标准路径分母是 `A^T 1`，分子却是 `G^T A^T r`；compact 路径（`MainProgram` compact sensitivity 分支）是正确的 `G^T A^T 1`。后果：开 PSF 时标准路径存在系统性定量偏差（对均匀区近似整体缩放，但边缘/高频区域偏差不可忽略），且 standard 与 compact 两条路径结果不可对齐。

修复：`MainProgram` 标准路径全局 denom（含 `computeNormAT_Atten` 分支）与 BSREM 非全子集 `subsetDenom` 计算后统一补 `apply_psf_backward`（该函数在非 Operator 模式下为 no‑op，不影响无 PSF 运行）。BSREM 梯度 `A^T r − A_s^T 1` 两项现同为 `G^T`‑滤波量纲，梯度一致性成立。

### 1.3 空间不变假设的物理定位（原理审查，待数据）

针孔系统的分辨率核在物理上随源—针孔距离变化（放大率 `m = f/d` 依赖位置），image‑domain 空间不变高斯是一阶近似。但注意两点：

1. 系统矩阵本身（v4 生成器）已含针孔几何响应（有限孔径立体角），Operator PSF 只补偿探测器本征模糊 + 矩阵未建模的残余；这部分在 FOV 中心区域近似不变是合理的。
2. 判定近似是否够用的实验（有数据后执行）：点源置于 FOV 中心/半径 1/2/边缘处各重建一次，量 FWHM（xy 与 z 分开）。若边缘/中心 FWHM 比 > ~1.3，值得引入径向分段核或投影域 PSF；否则维持现状。

其余注意项：3σ 截断硬编码（对大 FWHM 略有能量损失，<0.3%，可忽略）；PSF 与 RDP 同时开启时无 sieve，高迭代数下可能边缘振铃——beta sweep 时一并观察。

## 2. MLEM/OSEM + RDP / BSREM 审计（B.3 原理部分）

### 2.1 RDP 公式（已核实，`src/regularizer/Regularizers.cpp` `grad_RDP`/能量项）

实现的势函数为

ψ(f_j, f_k) = (f_j − f_k)² / (f_j + f_k + γ|f_j − f_k| + 2ε)，ε = max(delta, 1e‑6)

梯度 `term_num/denom²` 与该式的解析导数一致（已手工验证：∂ψ/∂f_j = [2d·S − d²(1+γ·sign d)]/S²，d=f_j−f_k，S=分母）。6 邻域，无距离加权；边界处跳过体外邻居；`skip_outside_mask` 可选跳过 mask 外邻居。分母另有 1e‑9 下限保护。

评估：

- 公式形式即 Nuyts et al. 2002 的 RDP，γ 控制边缘保持——正确。
- **命名问题**：CLI/结构体里的 `delta` 实际是数值稳定项 ε（防 f_j=f_k=0 时 0/0），不是 Huber‑δ 那种转折参数。RDP 的"转折"由 γ 控制。文档与帮助文本应写明，避免用户拿 Huber 直觉调 `delta`。
- 6 邻域无距离加权：当前体素各向同性时无影响；若未来 z 间距不同于 xy，需按 1/‖Δx‖ 加权（记入待办，非当前问题）。
- 能量函数与梯度来自同一 ψ（已核对 `compute_reg_energy` RDP 分支同用 2ε 稳定项）——likelihood 监控与梯度自洽。

### 2.2 BSREM 更新（已核实，`src/app/BSREMReconstructor.cpp`）

f ← f + α_k · (f/S_glob) · [A_s^T r − A_s^T 1 − (β/n_sub)·κ·∇R]，α_k = α0/(1+γ_bsrem·k)

相对更新因子截断到 [−0.9, +1.0]，f 下限 eps=1e‑12，mask 外清零；可选 curvature 分母 `S + β·curv`。

- 结构上是 BSREM‑II 型（relaxed OS + 惩罚梯度），β 按 1/n_subsets 缩放——与子集化似然梯度量纲匹配，正确。
- `curv = |∇R|/max(f, curvEps)` 与 `κ = normalize(1/√S)` 均为启发式预条件，不是 RDP 解析曲率（解析式 ∂²ψ/∂f_j² 可写出但未实现）。原理上不影响不动点（乘性/加性预条件不改变梯度零点），影响收敛速度与稳定性——这正是 β 不可迁移的来源之一，留待真实数据 sweep。
- `w_proj`/`w_img` 语义未定义问题已在 2026‑07‑03 加防护（fused gate 拒绝 + `BSREM_step` 抛错，清单 6.3）。

### 2.3 OSEM/MLEM 的 OSL 惩罚路径（已核实，`OSEM_step` + `MLEMOptimizer`）

`denom_eff = S_full + β·∇R/f`（One‑Step‑Late），其中分子是**子集**反投影、S_full 是**全局** sensitivity、β 为**未缩放**的 `args.reg.beta`。

两个原理性偏差：

1. 分子子集/分母全局：这是"全局归一化 OSEM"变体，收敛慢 n_sub 倍但稳定；与经典 OSEM（子集 sensitivity）不同，文档需写明。
2. 惩罚权重：分子期望值 ≈ 全局的 1/n_sub，而 β 项不缩放 → 有效惩罚强度放大 ≈ n_sub 倍，且 OSL 本身在 β 大时不保证非负/收敛。

结论维持清单 6.5：生产惩罚重建统一走 BSREM；OSL 路径标记 legacy（subsets=1 时行为正确）。不做行为变更（避免静默改变既有结果）。

### 2.4 B.3 校准协议（数据到位后执行）

1. 固定 α0/γ_bsrem（默认），扫 β ∈ {0.01, 0.03, 0.1, 0.3, 1, 3} × γ ∈ {2, 5, 10}（RDP γ）。
2. 每组记录：惩罚似然曲线（已有能量函数）、Derenzo 杆对比度、均匀区 CV、收敛迭代数。
3. 选型标准：对比度‑CV 曲线拐点；输出 `doc/BSREM_Calibration_<date>.md` 固化推荐值与适用计数率范围。
4. kappa/curvature 开关各做一组消融，确认预条件对收敛速度的实际贡献。

## 3. 散射背景 bProj 审计（B.2 原理部分）

### 3.1 数学语义（已核实）

- MLEM/OSEM/BSREM（非 fused）：ratio = `p_i/(Σ_j a_ij f_j + b_i)`，`b` 进分母不进 sensitivity——与含背景 Poisson 模型 `y ~ Poisson(Af + b)` 的 MLEM 一致，正确。
- fused CUDA 路径不支持 b：gate 已拒绝（与 w_proj/w_img 同口径），回落非 fused，无静默错误。
- compact 路径支持 b（禁用 device‑ratio 优化）；`b=0` 与不传 b 数值等价（float 精度内）。

### 3.2 输入防护缺口

加载仅校验尺寸/几何一致，不校验：非负性（TEW 估计噪声可产生负值）、非有限值、单位量纲（b 必须是与 p 同采集时长的期望计数/bin）。建议：加载时 clamp 负值到 0 并告警、拒绝 NaN/Inf、README 写明单位约定。此为小修，可派发廉价模型。

### 3.3 散射估计原理定位（无数据阶段的结论）

- 仓库内无 DEW/TEW/MC 估计器，`-bProj` 是唯一注入口——架构上干净（估计与重建解耦），保持。
- 有数据后的推荐路径：TEW（主窗 ± 相邻窗梯形权重）→ 投影域高斯平滑（散射窗计数低、噪声大，不平滑会注入噪声）→ clamp ≥0 → 存 `b_proj`。小动物针孔 SPECT 散射分数低（Tc‑99m 典型 <10–15%），预期对定量影响小于 CTAC，与"CTAC 优先、散射次之"的排期一致。

## 4. 已完成动作汇总

| 动作 | 位置 |
|---|---|
| 标准路径 denom/subsetDenom 补 `G^T`（PSF 一致性修复） | `src/main/MainProgram.cpp`（清单 6.6） |
| BSREM w_proj/w_img 防护 | `src/app/BSREMReconstructor.cpp`（清单 6.3） |
| 全量单测 241/241 通过（CUDA 构建） | `out/build/ctac-agent` |

待派发小修：bProj 负值/NaN 防护；`-h` 帮助与 README 中 RDP `delta`=稳定项的措辞澄清；ProjDomain PSF 与 bProj 组合的显式拒绝。
