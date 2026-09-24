# SPECT_recon 2026-09-21 Release

Windows x64 / CUDA 版本。默认标准 OSEM，50 个完整迭代、10 个角度子集、无正则，仅保存最后一轮。内部算法名称为 MLEM；使用多个角度子集即 OSEM，不是 BSREM 或 POSEM。

## 默认参数

| 项目 | Tc99m / 通用默认 | Lu177 默认 |
|---|---|---|
| 配方 | 04_standard_osem50.conf | 06_standard_joint_osem50.conf |
| 迭代 | 50:10 | 50:10 |
| 正则 | None，beta=0 | None，beta=0 |
| 观测 | 单窗 141 keV | 独立 208 / 113 keV，顺序不变 |
| 矩阵 | 原生 CSRB | 两份原生 CSRB |
| 能窗权重 | 1 | 0.1036,0.0617，沿用而非新标定 |
| PSF | sieve / CASToR，1.3 mm | sieve / CASToR，1.3 mm |
| 输出 | primary + 独立 Butterworth 副本；Gaussian 关闭 | 同左 |

Lu 默认从 BSREM+RDP 切换为标准 OSEM，是本次用户明确批准的配置变化。原 `05_mouse_joint_rdp30.conf` 保留为显式实验选项，仍绑定 CSRQ；不会被默认加载。矩阵和投影必须保持核素/能窗对应，不能将合并投影伪装成两个独立能窗。Norm 仍关闭，需独立匹配的校正文件才能启用。

## 本次兼容能力

- SC60 SMMD v3 / ROT90 原生 CSRB；旧 PSC ROT90_Z 路径保留。
- CSRB 子集分段读取、原生双窗加权 q/S、共享主机缓存预算及双窗 I/O 合计。
- 完整矩阵网格多床输出：显式正步进、整数体素层对齐、有效掩膜融合。SC60 尚无实采多床和已标定步进，不使用 PSC 的 16.8 mm 作为 SC60 默认。
- CSRQ 仍实验性；不支持混合 CSRB/CSRQ 双窗。不开放 N>1 + CTAC，也不声称支持未定义的新对称格式。

## 使用

程序和 DLL 在 `bin/`，配置在 `bin/config/`。本包不含矩阵、投影、重建输出或旧灵敏度缓存。按配方的准确 matrix_id 部署矩阵，或显式传入 `-chunkCSRDir` / `-chunkCSRDir2`；缺失矩阵会明确失败，不自动替换。

```powershell
# 默认 Tc：50 轮，N=10
.\bin\SPECT_recon.exe -isotope Tc99m -chunkCSRDir "D:\matrix\Tc141" -df "D:\scan\Bed1.bin" -o recon_tc
# 默认 Lu：保留独立 208/113 双窗模型，标准 OSEM
.\bin\SPECT_recon.exe -isotope Lu177 -chunkCSRDir "D:\matrix\Lu208" -chunkCSRDir2 "D:\matrix\Lu113" -df "D:\scan\Bed1_208.bin" -df2 "D:\scan\Bed1_113.bin" -o recon_lu
# SC60 必须显式选择系统配方；这里覆盖为 50 轮、最终保存
.\bin\SPECT_recon.exe -algoConf bin/config/profiles/sc60_osem1000.conf -chunkCSRDir "D:\matrix\SC60" -df "D:\scan\Bed1.bin" -it 50:10 -save-every 0 -o recon_sc60
```

SC60 专用配方本身是此前验证所用的 1000:10、PSF1.5 mm、每50轮保存、完整网格及36×96 mm支持区。上述短跑覆盖不改变配方文件。32.8×12 mm ROI 派生矩阵须另外指定 `-cyl 32.8,12`，不可把派生支持区与原始全矩阵混为一谈。

## 构建与验证范围

来自当前 `635dec6` 基础上的未提交工作树，版本保留 `-dirty`，未伪装成干净提交。CUDA helper fatbin 包含 sm_86 / sm_89 / sm_120；仅支持 CUDA 环境，不带 NVRTC 热路径或 CPU 重建回退。包内带 CUDA/MSVC 运行库，仍须安装兼容 NVIDIA 驱动；现场 GPU 需另做匹配验证。

本次源构建全套测试：319 项，315 通过，4 项既有外部数据/显式开启条件跳过。首轮唯一失败为 CT 测试 INI 的旧 Lu 默认，已同步后复测通过。之前同数值实现完成 SC60 ROI1000/N10/PSF1.5、20检查点、迭代读取0，以及 PSC 五床配对回归；本次版本说明更新不代表重新运行这些长测试。完整迭代 wall time 与旧 `iteration_core_s` 阶段计时不是同一指标。

发行包保留已有原始矩阵、参考图像和历史发布包。本地打包不等于已替换外部采集系统或完成核素定量标定。
