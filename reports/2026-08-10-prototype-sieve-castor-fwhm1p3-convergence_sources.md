# 本地证据源登记

| ID | 本地来源 | 证据角色 |
|---|---|---|
| S1 | `.agent/CONSTRAINTS.md` | 用户冻结的原型权威、五数据集、几何方向与接受合同 |
| S2 | `6module_bak_src/spect_recon_PSC_6module_24pin_YSO_bedmerge.cu` | 受保护 `sieve + CASToR + FWHM 1.3 mm` 原型源码 |
| S3 | `recon_out/prototype_sieve_castor_fwhm1p3_final_20260811/input_preflight.json` | 42 个输入、矩阵、配置、二进制和原型产物的大小与 SHA-256 预检 |
| S4 | `recon_out/prototype_sieve_castor_fwhm1p3_final_20260811/experiment.json` | 最终五数据集合同、逐项结果与采纳决定 |
| S5 | `recon_out/prototype_sieve_castor_fwhm1p3_final_20260811/run_record.json` | 五次正式运行的完整参数、退出码与耗时 |
| S6 | `recon_out/prototype_sieve_castor_fwhm1p3_final_20260811/current_production_parity_metrics.json` | 五数据集数值、对象、方向与输出合同主证书 |
| S7 | `recon_out/prototype_sieve_castor_fwhm1p3_final_20260811/output_certificate.json` | 21 个正式输出/清单/指标/图像/测试产物的 SHA-256 证书 |
| S8 | `recon_out/prototype_sieve_castor_fwhm1p3_final_20260811/spect_tests.json` | 正式 Release 测试二进制的 228 项机器可读结果 |
| S9 | `recon_out/prototype_detector68_mapping_20260811/experiment.json` | detector 68 与首次 PSF 前 ROI 状态的根因实验总记录 |
| S10 | `recon_out/prototype_detector68_mapping_20260811/none_one_iter_compare.json` | detector 68 修复后的无 PSF 一迭代结构证书 |
| S11 | `recon_out/prototype_detector68_mapping_20260811/sieve_one_iter_initial_mask_compare.json` | 两项根因修复后的 sieve 一迭代结构证书 |
| S12 | `recon_out/direct_weighted_sensitivity_20260811/experiment.json` | 直接 `A^T w` 候选失败并回退的记录 |
| S13 | `recon_out/numeric_policy_20260810/experiment.json` | 低前向比值与 MLEM 更新下限候选的一迭代拒绝记录 |
| S14 | `recon_out/mouse_prototype_numeric_policy_20260810/experiment.json` | 同一数值候选在低计数 mouse 50 次终点的拒绝记录 |
| S15 | `recon_out/mouse_gain_window_orientation_20260810/experiment.json` | 反向存储增益读窗单独修复的结果及 blend 极性后续 |
| S16 | `reports/assets/2026-08-11-prototype-sieve-castor-fwhm1p3-direct-slices.png` | 原型选层、共享 P99.5 窗宽、无插值、无候选缩放的直接切片 QA |

本登记只列本次报告实际使用的本地证据；受保护原型是实现一致性基线，不被表述为独立物理真值。
