# Release 2026-07-10

Release date: 2026-07-10

## 版本要点

- **`-v` / `--version`**：打印构建版本字符串（`YYYYMMDD+githash[-dirty]`）并退出。旧 `-v` verbosity 别名已移除，日志级别只用 `-vb`。
- **启动日志**：重建开始时 INFO 输出 `=== SPECT_recon <version> ===` 与完整 `Command line:`；`Run configuration` 标题同样带版本号，并列出 `command:` 行。
- 其余边界继承 `RELEASE_20260708_CN.md`（CUDA-only、260601 chunk 默认、无 compact multi-bed CTAC 等）。

## 验证

- 重新 `build_release.bat` + `scripts/package_rc.ps1` 生成 RC。
- `SPECT_recon.exe -v` 应打印非空版本行。
- 任意短重建日志开头应含版本 banner 与完整命令行。
