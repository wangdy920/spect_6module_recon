# utils/ 工具脚本

本目录包含与重建流程配套的辅助脚本（不参与 C++ 编译）。

> 重建流程的对外脚本入口（含对 `utils/run_l_curve.py` 的封装）见 [`scripts/README.md`](../scripts/README.md)。本目录只放底层工具与 MATLAB 实验。

## Python

### `run_l_curve.py`

用于批量扫描不同正则化权重 `beta`，并从程序输出中的 `[METRIC]` 行提取：
- `Beta`
- `Likelihood`
- `Penalty`

推荐用 `--base_arg` 逐 token 传参（避免 Windows 路径/空格问题）；也可以使用：
- `--base_args_file`：每行一个参数 token
- `--base_args`：简单空格分割（不推荐）

> 3D 体绘制可视化脚本 `render_volume_3d.py` 位于 [`scripts/`](../scripts/)，不在本目录。

## MATLAB

- `generate_psf_phantom.m`
- `create_psf_map_example.m`
- `calc_psf_map_from_sources.m`

用于 PSF/phantom 相关实验与数据生成。

