# Release 2026-07-15

Release date: 2026-07-15

## 版本边界

- 多床位默认采用 `bed_overlap=16,16,16,0,0`、`overlap_in_recon=0`、`crop_method=sigmoid`，80 层单床对应 stride 64；最终输出使用归一化 sigmoid `k=12` 融合。
- 默认开启对象感知接缝后处理：mouse 保持 no-op；规则仿体采用轮廓约束的锚点插值；zimu 采用拓扑锁定、`sigma=0.3 mm`、强度 `0.5` 的局部修正。处理仅限内部接缝，不裁剪整段首尾。
- `seam_half_width`、`phantom_close_strength`、`zimu_close_strength` 和 `object_profile` 可在 `[MultiBed]` 配置；默认开启圆形 XY FOV `15.0 mm`，边缘为 4 voxel taper。
- compact 260601 支持多床位 CTAC；line-integral factor 固定由 CUDA 按 16-angle batch 生成，不保留 CPU factor fallback。
- PSF 新增 `z_blur`，默认 `true`；公开 regularizer 收敛为 TV/RDP，移除未维护的重复实现。
- 输出 MHD 保留方向/原点元数据；新增 projection residual 诊断和对象感知接缝后处理的 C++/Python 数值一致性 gate。

## 验证合同

- C++ 接缝后处理与冻结 Python 参考在 capillary、双管、mouse、zimu 四类数据上达到 float64 rounding 一致：support mismatch 为 0，最大绝对误差不超过 `2.709e-14`。
- 最终 DICOM 回读确认四组数据均为 `108x108x400 @ 0.3 mm`，InstanceNumber 连续 `1..400`；本 release 不写入未提供的放射性药物剂量字段。
- 发布前必须通过 `spect_tests`、`spect_cuda_scaffold_tests`、Python self-check/parity，并从干净提交重新执行 `build_release.bat` 与 `scripts/package_rc.ps1`。

## 部署默认值

部署时以包内 `config/spect_config.ini` 为准。sigmoid 是最终输出拼接，不在迭代中应用，因此必须保持 `overlap_in_recon=0`；若需要联合重建 overlap，只能显式切换到 `crop_method=direct` 并重新验证。
