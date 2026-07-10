# SPECT_recon CLI 变更速查

本文只记录容易误用的废弃入口和替代写法；完整参数以
`PARAMETERS_CN.md` 和 `docs/SPECT_recon_help.txt` 为准。

## 必须改的调用

| 旧写法 | 当前写法 |
| --- | --- |
| `-alg mlem|bsrem|admm` | `-opti mlem|bsrem|admm` |
| `-sparseBackend cpu|cuda` / `-spBackend ...` | 删除该参数；稀疏后端固定 CUDA |
| `-v <0\|1\|2>`（旧 verbosity 别名） | 改用 `-vb <0\|1\|2>`；**`-v` / `--version` 现为输出版本号** |
| `-butterBackend fftw|cufft|auto` | 删除该参数；Butterworth 后端固定 CUDA/cuFFT |
| `-psf <sigma>` | `-psf <fwhm_mm[,N]>`，默认 `1.5,4` |
| `-psfProj` | 删除；公开 PSF 只保留 image-domain Operator |
| `-psfMode PostSmooth|ProjDomain|DistanceDependent` | `-psfMode None|Operator` |
| `-ctacModel per-voxel` | `-ctacModel line-integral` 或省略 |
| `-prior ...` / anatomical prior 参数 | 删除；公开正则只保留 TV/RDP |
| `-reg NLM|Tikhonov|LogCosh|MRP|Bowsher|PLS...` | `-reg TV,...` 或 `-reg RDP,...` |

## 仍可用但不推荐

| 参数 | 当前建议 |
| --- | --- |
| `-loadCSR -compactMeta` | 260525 legacy 兼容入口；260601 生产用 `-chunkCSRDir` |
| `-mapCSR on` | 仅 single-file CSRB 大文件需要；chunk CSRB 按目录读取 |
| `-conv <params>` | legacy 卷积入口；常规平滑用 `-smooth` 或 `-butter` |
| `-joe <n>` | 默认 `0`；除非复现实验，不建议启用 |

## 推荐最小命令

```powershell
.\bin\SPECT_recon.exe `
  -df D:\data\PSC_6module_24pin_10MBq_600s_test_1.bin `
  -chunkCSRDir .\bin\matrix\260601 `
  -opti mlem `
  -it 50:5 `
  -o recon_output
```

如果发布包内 `bin/matrix/` 只有一个包含 `metadata.bin` 的矩阵目录，
`-chunkCSRDir` 也可以省略。
