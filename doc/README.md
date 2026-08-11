# Documentation Index

只维护下表的当前合同。过时设计稿、执行计划、历史 release 说明和对象/数据集特定实验文档已删除；历史过程查 git，当前行为以代码、配置和五组数据决策记录为准。

## Maintained

| File | Purpose |
|---|---|
| `../reports/2026-08-10-prototype-sieve-castor-fwhm1p3-convergence.html` | 当前 `sieve + CASToR + FWHM 1.3 mm` 原型对齐、五组数据指标与 canonical 输出决策 |
| `USAGE_CN.md` | 公开 CLI 与固定多床工作流 |
| `PARAMETERS_CN.template.md` | 发布包参数说明模板 |
| `README_RC.template.md` | 发布包 README 模板 |
| `Capability_Matrix_CN.md` | 能力矩阵与耐久决策 |
| `Format_Reader_Matrix_CN.md` | v4 chunk CSRB 矩阵格式 |
| `CTAC_AutoMu_Validation_CN.md` | CTAC / auto-mu 调试与 gate |
| `Recon_Interface_Migration_Field_CN.md` | 现场 ThreeModalities 接口迁移 |

## Update rule

- CLI 或默认配置变化：同步 `USAGE_CN.md` 和 `PARAMETERS_CN.template.md`。
- 能力边界或数值政策变化：同步 `Capability_Matrix_CN.md` 与 canonical prototype report。
- 现场接口变化：同步 `Recon_Interface_Migration_Field_CN.md`。
- 不恢复旧计划或一次性实验实现；需要历史过程时查 git。
