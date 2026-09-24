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
SC60没有Lu177资源，不借用PSC资源或16.8 mm假设。后续文档复核纠正了此前“没有实采”的描述：
6Module已有首批三床采集记录21mm；当前0.4mm矩阵不支持52.5层的非整数步进，21.2mm仅为试验近似。

## 已完成检查

- `cmake --build build/csrb-native-20260921 --config Release --parallel 6`通过。
- `ctest --test-dir build/csrb-native-20260921 -C Release --output-on-failure`：319项，315通过、4既有跳过、0失败。
- 6Module的`test_release_tooling.py`、`test_run_6module_recon.py`、`test_run_6module_preprocess_wrapper.py`、`test_install_release_overlay.py`通过。
- 通用包合成测试：PSC Tc与Lu输出100×100×66；SC60 Tc输出90×90×240；均为1完整迭代、N10、有限/非负/非零。
- 配置测试确认标准默认50:10/None，独立208/113窗、原权重及SN绑定；缺矩阵、跨SN、错误SMMD、SC60缺少部署配置/床距/核素资源均拒绝。
- 最终`SPECT_Universal_260924_r1`发布校验通过，68项manifest散列一致，0失败；4个预期警告是未随包部署的矩阵。测试中的故意错误注入日志带FAIL，不等于测试失败。

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
python tools/test_universal_release.py --release-root release/SPECT_Universal_260924_r1 --reconstruct
python tools/verify_deployment.py --release-root release/SPECT_Universal_260924_r1
```

合成夹具使用真实探头/图像尺寸、稀疏中心体素和旋转/Z对称映射，不是实际系统矩阵。
它证明调用、数据流、GPU执行和输出尺寸；不证明实采方向、定量标定或影像质量。
既有SC60真实ROI长跑、PSC回归及Lu时间线不重跑，见
[适配记录](2026-09-18-spect-matrix-adaptation-implementation-plan.md)及
[已完成时间线报告](../recon_out/lu177_tumor_timeline_20260923/REPORT.md)。

## 最终交付

重建源码提交 `00c18b4` 已创建，执行 `cmake -S . -B build/csrb-native-20260921` 后
`--clean-first --target SPECT_recon spect_tests` 全量重编译通过。工具链为VS18 2026/CUDA13.2.51，
三份嵌入fatbin均含sm86/89/120；重新运行CTest仍为315通过、4既有跳过。
保留既有OpenMP collapse、浮点转换、静态运行库链接警告，未为整理工作树扩大数值修改范围。

EXE版本 `20260923+00c18b4`（本机时区日期），SHA256
`02df6dc7801c0183f636d7a005d7ea55dc4fc83a3a36ed729441fa595cdafc9a`。
独立通用修复包为 `E:/Xiang_Project/SPECT/6Module/release/SPECT_Universal_260924_r1`。
默认PSC，Tc04/Lu06均为标准OSEM50:10；SC60仍须按设备资源说明手动设置几何，矩阵不随包分发。

6Module原生预处理构建也通过。EXE SHA256
`56f5ba60a601182d8ec69091f7cd05b9d01118a1ed180a3b199ade393009f113`；
与历史5948f0c6版本进行三个既有三角度输入的投影回归：PSC Tc、PSC Lu双窗、SC60共5份BIN
逐文件SHA一致，输入hash未变。不将三角度检查扩大为所有原始采集的重验。

真实SC60模拟投影经通用wrapper、32.8×12 mm ROI派生矩阵、N10/PSF1.5一轮重建完成，
对既有同参数it1参考的relL2为 **5.1682991e-8**，通过事先沿用的1e-5容差。
最终体积90×90×240、0.4 mm；此项只证明该检查点数值回归。
运行前后的目录、配方和输入身份分别记录，未修改原始矩阵或投影。

证据集中于 `recon_out/universal_integration_20260924/`：`verify_real.py`、`PASS.json`、
`preprocess_checks.json`、三组预处理日志、SC60实际命令/配置/输出；
完整CTest日志位于上一级 `universal_integration_20260924_ctest.log`。
6Module原生预处理、SN资源迁移、打包器和wrapper修改已整体核对并提交为`849d002`；
测试缓存隔离修正为`c27b2bd`，测试将缓存写入临时目录，不污染被测发布包。
最终包实际重建测试通过，日志为`universal_final.log`；最终包不含测试缓存或gtest DLL。
这些是6Module仓库提交，不是重建EXE来源；后续报告归档提交不改变上述EXE身份。
原`SPECT_Universal_260924`的manifest散列再次只读核对通过；D盘9月21日封存包未修改。
修复包使用独立目录，未生成新ZIP，未推送远端。

既有`reports/2026-09-23-lu177-tumor-timeline.md`及两组assets原样归档，
19个图片引用均存在，4份CSV可严格UTF-8读取，行数分别为2400、40、8、16。
此处仅确认附件完整性，不代表重新执行计数分析或将其科学结论纳入本轮验证。

## SPECT_Core文档与联合工作入口

用户要求后续完整包固定命名`SPECT_Core`，版本移入`version.txt`，分别管理preprocess/recon。
交付：`E:/Xiang_Project/SPECT/6Module/release/SPECT_Core`及同级`SPECT_Core.zip`。
preprocess版本为`SPECT_preprocess 20260924-sn-resources`，recon为`20260923+00c18b4`；
EXE散列沿用上节。集成源码提交`6d171c2`，version.txt单独记录tooling_revision及组装时间。
打包器移除--release-name，增加--output-parent；遇已有SPECT_Core目录或ZIP拒绝覆盖，历史包不删除。

包内Markdown由14份缩为3份：README_CN、RECON_CONFIG_AND_CALL_GUIDE_CN、DEVICE_RESOURCES_CN。
两份指南直接从源码doc复制；历史/实验说明只保留仓库，矩阵目录不再生成重复README。
算法模板6份及其值未变，重建全局INI只修正过时CSRQ注释。源仓库README更新为主要维护入口；本地WIKI仍不入Git。

验证：test_release_tooling（含三文档引用、版本分区、禁止覆盖目录/ZIP）与overlay测试通过；
最终包verify_deployment为0失败/4未安装矩阵警告，69项manifest散列通过。
test_universal_release --reconstruct在最终SPECT_Core上完成PSC Tc、PSC Lu、SC60 Tc三路径，
同时解析文档中的SC60单床配置及Lu100:10/保存10/PSF1.5示例，确认实际生效值。
444份EXE/DLL/MAT/BIN/CONF和两个wrapper与r1逐文件相同，全局参数解析值相同。
ZIP540项、389752020字节，CRC及每个文件与交付目录一致，SHA256：
`b139c350b0512e6a3a99cb48711d74d84ed25dfaf7f993e242db0755bc0a3ef6`。
证据：`recon_out/universal_integration_20260924/core_delivery.json`及同目录`verify_core_delivery.py`；后者为本次交付验证脚本，不是生产入口。

统一开发入口`D:/Projects/SPECT-workspace`内preprocess/recon为NTFS目录联接，Git解析仍分别返回E盘/D盘原仓库；
SPECT.code-workspace以原路径打开双根，未迁移源码、Git或构建缓存。
scripts/setup_spect_workspace.ps1及test_setup_spect_workspace.ps1覆盖重复执行、目标校验、冲突拒绝；检查通过。
联合行为规范仅维护根目录DEVELOPMENT.md，定义实验分支、主线采用、构建、打包、部署、文档同步与临时证据边界。
本轮不自动将6Module发布分支合并main，不推送远端，不重跑影像质量实验。
