# 通用包双系统重建调用审查

日期：2026-09-24。范围：`multipinhole` 当前适配代码、文档和本地构建，以及
`E:/Xiang_Project/SPECT/6Module/release/SPECT_Universal_260924` 的调用合同。
原始数据、系统矩阵、LUT和历史发布包不修改；不追加算法调参或图像质量实验。
验收为构建/测试通过、实际wrapper合成重建完成、默认与矩阵绑定一致，再创建本地提交，不推送。

## 审查结果与修正

| 问题 | 修正及验证 |
|---|---|
| 通用打包器删除06并强制Lu默认05，回退到BSREM/RDP30 | 保留06；Tc04/Lu06均为MLEM/OSEM50:10、无正则；检查真实EXE的生效配置 |
| Lu joint按模型硬编码CSRQ，不能部署当前原生CSRB默认 | 通用包使用208/113原生CSRB；05仅改变算法，不改变通用包矩阵绑定；独立RC历史05不改 |
| 208同时用于merged/joint时，以矩阵ID去重会丢失一个角色 | 资源清单保留两个角色，磁盘路径仍共用，增加回归断言 |
| SC60仅切SN后仍继承PSC裁剪、坏道68和默认ROI | wrapper要求matrix_grid_output=true、空PSC坏道表、显式ROI或关闭圆柱mask；不自动推断物理参数 |
| 显式recipe加外部--exe可能读取外部EXE旁全局配置 | 全局配置始终来自当前通用包；合成重建用外部EXE验证 |
| 旧测试只到-dumpConf，不能证明输出几何正确 | 扩展原test_universal_release.py，支持--reconstruct，实际执行三个入口并检查输出 |
| 根README仍宣称双窗仅CSRQ、Lu默认RDP、Bed1为最大Z | 对齐原生CSRB、标准OSEM默认及manifest的Bed1=min_physical_z软件坐标合同 |

共用profile不带矩阵ID；`Runtime.scanner_name`选择SN资源和JSON矩阵绑定；
矩阵metadata决定网格、体素、探头和角度，部署端负责床距、mask及输出方向。
SC60仍没有Lu177资源，也没有实采多床步进；不借用PSC资源或16.8 mm假设。

## 已完成检查

- `cmake --build build/csrb-native-20260921 --config Release --parallel 6`通过。
- `ctest --test-dir build/csrb-native-20260921 -C Release --output-on-failure`：319项，315通过、4既有跳过、0失败。
- 6Module的`test_release_tooling.py`、`test_run_6module_recon.py`、`test_run_6module_preprocess_wrapper.py`、`test_install_release_overlay.py`通过。
- 通用包合成测试：PSC Tc与Lu输出100×100×66；SC60 Tc输出90×90×240；均为1完整迭代、N10、有限/非负/非零。
- 配置测试确认标准默认50:10/None，独立208/113窗、原权重及SN绑定；缺矩阵、跨SN、错误SMMD、SC60缺少部署配置/床距/核素资源均拒绝。
- 发布校验在初步修复副本通过，0失败；4个预期警告是未随包部署的矩阵。测试中的故意错误注入日志带FAIL，不等于测试失败。

首轮构建是增量检查，数值源码未再改变，EXE仍为原`20260921+635dec6-dirty`，SHA256
`1d48b48f5ccf1a3b1173b02f6b7032533a6c429fc1b400cf8756677e480ccb94`。
不能把这个结果冒充提交后的新构建。提交后构建和最终包身份在下节补充。

## 复现

```powershell
# D:/Projects/SPECT-recon
cmake --build build/csrb-native-20260921 --config Release --target SPECT_recon spect_tests --parallel 6
ctest --test-dir build/csrb-native-20260921 -C Release --output-on-failure
# E:/Xiang_Project/SPECT/6Module
python tools/test_release_tooling.py
python tools/test_run_6module_recon.py
python tools/test_universal_release.py --release-root release/SPECT_Universal_260924_audit --reconstruct
```

合成夹具使用真实探头/图像尺寸、稀疏中心体素和旋转/Z对称映射，不是实际系统矩阵。
它证明调用、数据流、GPU执行和输出尺寸；不证明实采方向、定量标定或影像质量。
既有SC60真实ROI长跑、PSC回归及Lu时间线不重跑，见
[适配记录](2026-09-18-spect-matrix-adaptation-implementation-plan.md)及
[已完成时间线报告](../recon_out/lu177_tumor_timeline_20260923/REPORT.md)。

## 最终交付

提交后重新编译、最终修复包和身份校验待本轮完成后填写。
原`SPECT_Universal_260924`及D盘9月21日封存包保持不变；修复包使用独立目录。
