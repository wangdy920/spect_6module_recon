# Documentation Index

只维护下表文档。过时设计稿、PR 执行日志、旧 release 说明与 `doc/archive/` 已删除；行为以代码与当前 release 为准。

## Maintained

| File | Purpose |
|---|---|
| `RELEASE_20260710_CN.md` | 当前 release 边界、包布局、签收 |
| `USAGE_CN.md` | 公开 CLI |
| `CLI_CHANGES_CN.md` | 废弃/改名参数速查 |
| `PARAMETERS_CN.template.md` | 发布包参数说明模板 |
| `README_RC.template.md` | 发布包 README 模板 |
| `Capability_Matrix_CN.md` | 能力矩阵 + 耐久决策 |
| `Format_Reader_Matrix_CN.md` | 6module 格式 × 读者 |
| `Legacy_MLEM_B1B2_Parity_CN.md` | B1/260601 回归；B2 reference-only |
| `CTAC_AutoMu_Validation_CN.md` | CTAC / auto-mu 调试与主 gate |
| `Recon_Interface_Migration_Field_CN.md` | 现场 ThreeModalities 重建接口迁移 |

## How to decide

- 改 CLI / 默认配置 → 更新 `USAGE_CN.md`、`CLI_CHANGES_CN.md`，必要时改 template。
- 改能力边界 → 更新 `Capability_Matrix_CN.md`，并在 `RELEASE_*.md` 记一笔。
- 现场集成 → `Recon_Interface_Migration_Field_CN.md`。
- 不要恢复已删 archive；需要历史证据时查 git。
