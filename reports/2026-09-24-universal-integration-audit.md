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

### 后续文档拆分

按用户最新要求，将上位机说明拆为 `PREPROCESS_CONFIG_AND_CALL_GUIDE_CN.md` 与
`RECON_CONFIG_AND_CALL_GUIDE_CN.md`，分别包含调用示例、修改场景及无需修改的默认项。
加上入口README和设备指南，当前维护4份交付说明；上节3份为此前封包状态。
源码、打包清单、校验器、README及当前 `release/SPECT_Core/docs` 已同步；原ZIP未重封，仍是上节历史版本。

验证：`test_release_tooling.py` 通过4份文档和相对链接检查；`test_universal_release.py --release-root release/SPECT_Core`
通过原生配置解析，并新增Lu标准/mouse双床测试，确认208/113列表按物理床号排序，任务CONF的100:10、保存10、PSF1.5生效。
`verify_deployment.py` 为0失败、4个既有未装矩阵警告，69项manifest散列通过。
与封存ZIP逐文件比对，仅README和两份原有指南发生变化，另新增预处理指南；运行程序、配置、资源和version.txt未变。
本次没有重新处理ENG或运行数值迭代，不将配置/映射检查等同于实采图像验证。

### 文档驱动的黑盒调用审阅（2026-09-23 至 2026-09-24）

用户指定3个GPT 5.6 Luna子代理，分别负责预处理、重建、上位机衔接；仅阅读两份调用指南、发布README、设备指南及程序帮助，不查看源码或已有测试实现。允许执行交付入口，原始输入/资源/发布配置只读，派生样本和隔离部署均写入 `E:/Xiang_Project/SPECT/6Module/experiments/doc_cli_blackbox/`。单次调用上限60秒，无数值长重建。

结果与证据：

- [预处理报告](E:/Xiang_Project/SPECT/6Module/experiments/doc_cli_blackbox/preprocess/results.md)：Lu Bed1/Bed2各两角真实裁切样本预处理完成，6个BIN均为360×2304、6635528字节，run completed、两床ok；并发/保留中间文件、禁用衰变校正的短调用完成。将Lu输入指定为Tc仅验证调用，不作Tc科学证据。
- [重建报告](E:/Xiang_Project/SPECT/6Module/experiments/doc_cli_blackbox/recon/results.md)：原包缺矩阵，首轮被部署检查阻断；由主代理提供正确外置矩阵层级后，隔离实例的Lu标准、mouse、文档生成的100:10/保存10/PSF1.5、真实裁切双床共4条dry-run退出0，208/113床列表及矩阵路径正确。缺113 BIN负向退出1。Tc因缺可用CSRB未通过正向预检，未用CSRQ替代。
- [衔接报告](E:/Xiang_Project/SPECT/6Module/experiments/doc_cli_blackbox/integration/results.md)：覆盖非发布工作目录、空格路径、SN断言、空输入、合并窗对照；成功dry-run返回的日志路径实际未创建，已据此收紧文案。

不是完全无人协助的首轮通过：重建代理对外置矩阵层级及本机E卷不支持目录链接遇阻，主代理在D卷建立 `recon_out/doc_cli_blackbox_matrices/Lu177` 到真实矩阵根的junction后才完成正向预检。该映射只用于测试，无矩阵内容变更。

已补正文档：上位机须按采集计划核对Angle和ENG完整性，不能用360视角或completed/ok证明完整采集；异常退出时running状态不可消费；核素来自采集记录；独立SC60安装实例及绝对路径调用；矩阵root/isotope/ID示例；dry-run矩阵前置及预定日志路径语义；可运行的任务CONF生成示例。预处理和衔接代理完成修订后只读复核，重建代理运行新增配置生成示例并完成预检。

主代理复核实际JSON、失败日志及报告文件，四份交付文档10个本地引用有效，源码指南与交付副本一致；封存ZIP中的程序/配置/资源逐文件未变。本轮只更新文档，不修程序的失败状态行为，不编译、不改组件版本、不重封ZIP；未运行重建迭代或验证图像质量。

### SN矩阵设置及low/high输入协议实现（2026-09-24）

本节为上述文档审阅之后、用户明确批准的代码修改，不改变上位机两个wrapper的调用方式。
范围：输出命名、输入选择、按SN矩阵路径、构建/打包/校验及对应说明；不修改投影计数算法、OSEM/BSREM算子、既有窗权重、原始矩阵或历史发布包。验收为同计数BIN逐字节一致、同输入命名迁移重建relL2<=1e-5，以及实际入口/异常分支检查；不运行新的长迭代画质或性能实验。

- 预处理默认保留每床合并BIN，并输出 `_high.bin` / `_low.bin`；Lu对应208/113 keV。summary使用high/low键并记录能量，物理床号不变。
- 真值文件为 `multipinhole/config/scanners/{PSC30-MPH6,SC60-MPH3}.ini`，构建复制到EXE旁config/scanners，发布保留同一路径。每核素含input_mode、chunk_format、sysmat_high、sysmat_low；路径直接指向矩阵目录。通用包核素JSON不再重复矩阵路径，旧包JSON仍可读取。
- Lu默认prefer_split：全部床位分窗齐全则joint；全部无分窗则merged，使用high矩阵，不加载low或使用双窗权重。缺一窗、已声明文件丢失、同一文件充当两窗、截断/头部错误、床间混用均拒绝。空矩阵槽不决定输入模型；选中joint但low为空仍报错。旧数字命名与Lu-177别名兼容。显式merged可使用已有合并BIN；分窗归一化不能自动迁移到merged。
- 原生可选内部scannerConf入口和SystemMatrix.scanner_config用于解析SN设置，现有调用保持兼容；wrapper自动传入，无需上位机新增参数。记录SN、实际模型、选择理由、矩阵路径/设置散列。外置EXE或自定义全局INI不改变当前安装实例的矩阵绑定。
- 打包器复制权威SN模板并校验与打包资产参数一致，不向外置矩阵目录写占位文件；外部路径在部署时配置。补齐CMake原有ct_spect_registration.ini遗漏复制。两份调用指南、设备指南、README及DEVELOPMENT已同步，交付文档仍为4份。

验证结果（RTX4070Ti SUPER，驱动616.92；重建EXE `20260924+d890065-dirty`，SHA256 `a7ad484bf33305b2a279a87ada259705d70055e2cca870c400474391291ca499`）：

| 检查 | 结果/证据 |
|---|---|
| 原生Release EXE及spect_tests编译 | 成功；既有LNK4098警告仍在 |
| 原生全套测试 | 316通过、4既有跳过、0失败；`recon_out/scanner_contract_20260924/native_tests.log` |
| Python包装器、预处理、打包合同和py_compile | 通过；含完整分窗、全合并、缺窗、截断、混合床负例 |
| PSC Tc、PSC Lu、SC60 Tc一轮N10 CUDA入口 | 全部通过，非负/有限/非零；尺寸分别100x100x66、100x100x66、90x90x240 |
| high/low与旧208/113同计数短重建 | relL2=0；仅命名兼容证据，不是joint与merged等价性 |
| Lu仅合并BIN、low矩阵空值 | 真正一轮重建成功、输出有效；实际选择merged，无第二矩阵 |
| 两床各两角真实ENG裁切样本重新预处理 | 6/6合并与分窗BIN与既有结果SHA256一致；`preprocess_parity.json` |
| 最终候选完整部署检查 | 0失败、4个未安装矩阵警告；`verify_final.log`；两个EXE帮助/PW自检通过 |

入口集成证据：`recon_out/scanner_contract_20260924/final_universal_test.log`，由
`python E:/Xiang_Project/SPECT/6Module/tools/test_universal_release.py --release-root D:/Projects/SPECT-recon/recon_out/scanner_contract_20260924/final/SPECT_Core --reconstruct` 重现。
日志中FAIL标签来自刻意损坏散列/SMMD的负例，最后status=passed且进程退出0。真实裁切输入和原输出位于E仓库experiments/doc_cli_blackbox/preprocess；本次派生结果仅写D仓库recon_out/scanner_contract_20260924/preprocess。

最终候选：`D:/Projects/SPECT-recon/recon_out/scanner_contract_20260924/final/SPECT_Core`，由正式打包脚本加显式allow-dirty-recon组装；version.txt仍独立标记两组件。preprocess EXE未重新编译，本次预处理改动在Python入口；recon重新编译。中间目录SPECT_Core/verified是调试候选，不作为最终交付。E仓库release/SPECT_Core及原ZIP没有覆盖，没有创建新ZIP、推送或声称生产/定量验收；本轮未提交，先前Git身份缺失仍待提供。用户删除的两个根pipeline配置未恢复或纳入变更。

### 固定交付入口更新与wrapper效率核对（2026-09-24）

用户要求继续更新正式SPECT_Core，并给出两个仓库远端。上节“未覆盖/未提交”是候选阶段状态。本次发布采用提交、干净重建、独立打包/验证、归档旧包、替换固定入口的顺序；原始数据、算法参数和已归档包保持不变。DEVELOPMENT已明确候选与当前交付入口的区别。

Git现场核对：D仓库作者为Codex/codex@local；E仓库缺作者设置，但三个最近提交均使用相同身份，沿用已有身份做命令级本地提交，不冒充GitHub账户作者。两个origin分别为wangdy920/spect_6module_recon.git和wangdy920/spect_6module_preprocess.git。GCM账户列表包含wangdy920；全局空credential.helper屏蔽了系统manager。只在读取远端的命令中启用manager后，两仓库ls-remote均成功；未改全局配置、未输出凭据、未推送。远端可读不是写权限证明。

效率审查为只读生产代码、短样本测量，预算两分钟，不变更接口或数值算法。证据及重现脚本在 `recon_out/core_publish_20260924/profile_wrappers.py` / `wrapper_profile.json`：

| 工作负载 | 重复与耗时 | 支持的判断 |
|---|---|---|
| PSC Tc合成矩阵wrapper dry-run | 5次，墙钟中位0.123876秒，范围约0.12–0.17秒 | 当前不是值得大改的启动开销 |
| Lu真实裁切样本，两床各两角预处理 | 3次，中位1.289955秒，范围约1.28–1.36秒 | 包含原生进程、Python导入、I/O和聚合，不能全计为wrapper额外负担 |
| 预处理回归 | 3次各6个BIN，共18/18逐字节等同旧结果 | 测量未改变计数输出 |

cProfile代表性重建样本中native配置预览调用约0.0227秒；预处理代表性样本13次subprocess.run累计约0.5649秒、导入累计约0.29秒，均为嵌套累计时间，不能直接相加为独立阶段。正式重建源码会先预览、再dumpConf落日志，存在一次重复解析，但本次短样本仅显示几十毫秒量级，未修改。此次不做常驻Python服务、跨任务配置缓存或C++重写；没有大采集规模的profile，不能声称预处理全规模无瓶颈。若后续实测显示明显开销，优先核对原生进程粒度及中间I/O，保留计数守恒/分窗校准与逐字节门。

正式入口更新已完成：

- 本地代码提交：重建 `85056c8`，预处理/集成 `c5cb409`；无推送。重建工作树提交后重新配置、编译和安装，正式EXE版本 `20260924+85056c8`（无dirty），SHA256 `32225aa0f69357311712874f7699fd375903d15fb004d183ee52c662c9e13438`。预处理EXE版本/散列不变。E仓库仅保留用户原有两个根pipeline配置删除，未纳入提交，因此package.tooling_revision如实带dirty；交付tools/文档/模板来自已提交内容。
- 当前目录与ZIP为 `E:/Xiang_Project/SPECT/6Module/release/SPECT_Core` 和 `SPECT_Core.zip`，同步SHA256旁车。ZIP为389762297字节，SHA256 `5f5d281993490c845bfc4195107cbc79630f2d3ca69117dd94c6272ca879757f`；499个文件CRC与目录逐文件SHA检查通过。
- 旧目录、ZIP及旁车归档至 `release/history/20260924_before_85056c8/`，502个旧文件散列保留，旧ZIP散列仍为 `b139c350b0512e6a3a99cb48711d74d84ed25dfaf7f993e242db0755bc0a3ef6`。归档中transfer_copy为搬运调试副本，不是当前发布入口。
- 搬运后第一次复核失败：Codex PID23660持有旧docs目录句柄，exFAT留下待删除目录，incoming被嵌套。只读句柄检查定位后，由用户关闭文件面板释放；未强制关闭句柄或结束进程。修正目录并重新执行全部发布路径检查，不能将中途失败记为首次成功。DEVELOPMENT与搬运脚本补上“归档后确认旧路径已消失”的门。
- 干净构建CTest：320项中316通过、4既有跳过、0失败。新组装位置及最终E盘位置均通过三设备/核素短CUDA入口、旧/新命名relL2=0、空low的merged检查；最终部署校验0失败/4未装矩阵警告。证据：`recon_out/core_publish_20260924/{ctest.log,universal_installed.log,verify_installed.json,zip_installed.json,promotion.json}`。
- 用户随后明确授权全局凭据修复：移除了用户级空 `credential.helper`，继承系统级manager；无需重录账号。未使用命令级helper覆盖的两个普通 `git ls-remote origin HEAD` 均通过，禁止交互提示状态下也成功。没有更改账户令牌或全局Git作者，没有推送。

当前无运行中的重建/预处理任务。本次发布只改变已验证的配置/输入路由与说明，不构成新的定量或画质结论。
