# Release 2026-07-21

Release date: 2026-07-21

## 发布默认

- Mouse 数据默认保持 `overlap_in_recon=0`、输出 `crop_method=sigmoid`、`mouse_seam_strength=0` 的 default 配置。
- norm W2 探测器效率权重默认关闭：`[MultiBed] detz_row_weights_file` 为空。只有显式传入 `-detZRowWeightsFile <file>` 时才启用，不会隐式影响 Mouse 重建。
- W2 仍保留为诊断/候选方法，不作为生产默认；C3 也不改变默认配置。
- `bad_dector` 默认空列表。若现场确认硬件探测器异常，可使用全局编号或详细 hardware ID；当前确认的 `link-4-FPGA-2-dector-4` 对应全局编号 `68`。

## 关键验证

- `spect_tests`：327/327 通过。
- PSC 及两组 Mouse 的 detector 68 屏蔽配对重建均成功，使用 `MLEM20 + Operator PSF 1.5 mm`。
- 屏蔽 detector 68 后，PSC/Mouse1/Mouse2 的总量变化分别约为 `-0.002%`、`+0.075%`、`+0.069%`；没有观察到明显视觉退化。
- Mouse DICOM 对比已经按统一 coronal 切面、统一 `0..50 PCNT` 显示窗口重新渲染；DICOM 回读几何一致，输出目录见发布验证记录。

## Mouse 阅图备注

来自同事的初步判断：在 Mouse 2 中，多出的异常信号主要位于肝部；由于本次药物目标为骨骼显像，该信号可能是错误信号。该判断应由有资深阅图经验的同事进一步确认，当前不将其作为已证实的生物学事实或算法缺陷结论。

现阶段仍确认：

- C3/W2 尚未证明能够修复床位缝隙处左侧腿部的连续性。
- W2 会改变 Mouse1/Mouse2 的局部对比度和组织分布，因此默认关闭，避免在组织真实性未确认前影响生产结果。
- 后续重点应是同一绝对窗宽下的资深阅图复核，以及必要时用改变床位步进/覆盖的采集验证信号是否随床位边界移动。

## 配置检查

发布包内 `bin/config/spect_config.ini` 应显示：

```ini
[MultiBed]
detz_row_weights_file =
```

使用 `-dumpConf` 检查合并配置时，`detz_row_weights_file` 应为空或 `(none)`。Mouse 默认命令不应包含 `-detZRowWeightsFile`。
