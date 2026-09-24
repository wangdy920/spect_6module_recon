# SPECT 联合开发与发布行为

这是预处理与重建两个仓库的唯一联合工作流说明，面向开发维护者，不随终端使用包分发。
发布包只保留四份说明：README入口页、面向上位机的预处理调用指南和重建调用指南、设备部署与适配指南。两份调用指南分别说明调用示例、修改场景及无需修改的默认项。
本文件定义何时调用已有脚本、同步哪些记录；不新增实验平台或自动合并工具。

## 1. 一个工作入口，两个原位仓库

| 入口 | 实际目录 | 责任 |
|---|---|---|
| `D:/Projects/SPECT-workspace/preprocess` | `E:/Xiang_Project/SPECT/6Module` | 原生预处理、SN/LUT、两阶段Python编排、通用包打包和部署 |
| `D:/Projects/SPECT-workspace/recon` | `D:/Projects/SPECT-recon` | multipinhole重建、矩阵读者、算法配置、CMake/单测、独立重建构建 |

入口是NTFS目录联接，不复制或迁移源码，也不合并Git历史；通过联接编辑就是编辑原文件。
根目录不创建第三个Git仓库。打开 `D:/Projects/SPECT-workspace/SPECT.code-workspace` 可同时查看两个仓库的改动。
工作区文件使用原始绝对路径，避免编辑器重复监视联接；大数据/实验/构建目录不做文件监视。
文件浏览可用联接；CMake缓存、MSBuild、Git worktree、打包均使用真实原路径，不把虚拟入口当新源码根重配缓存。

重建入口：
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:/Projects/SPECT-recon/scripts/setup_spect_workspace.ps1
```
脚本可重复运行；已有正确联接保持不变，遇同名普通目录、错误目标或被用户修改的工作区文件就报错，不覆盖。
换机器时显式传 `-WorkspaceRoot/-PreprocessRoot/-ReconRoot`。这里只建立源码入口，不给发布包、原始数据或矩阵创建联接。
备份/清理工具不要递归跟随此统一目录；不要把“删除快捷入口”变成递归删除源码。需要移除入口时只移除联接本身。

## 2. 行为与必做项

| 行为 | 执行入口 | 必须更新/保留 | 完成标准 |
|---|---|---|---|
| 日常联调 | 两个wrapper；重建可用`--exe`指向候选构建 | 本次输出目录的run JSON、effective config、日志 | 预处理完成后才重建，SN/核素/矩阵/几何一致 |
| 开实验分支 | Git `codex/<topic>`；有隔离需要才建worktree | 复用该主题报告/实验记录，写问题、基线、输入、允许改动、判据、停止条件 | 不改原始数据、冻结基线、默认发布行为 |
| 修正与验证 | 受影响单测、实际EXE短运行；数值改动加同输入回归 | 代码/配置及其唯一说明，记录执行命令和证据 | 单测/编译不冒充数值或质量验证 |
| 合并主线 | 每仓库独立审查、提交、Git merge/cherry-pick | 相关使用指南、格式协议/CHANGELOG（仅确实变化时）、实验结论 | 用户批准采用，主线测试通过，两个仓库对应提交可追溯 |
| 构建 | 预处理MSBuild；重建CMake/CTest | 构建日志、版本及EXE散列 | 不以旧EXE验证新源码；不能把dirty构建叫正式干净版本 |
| 通用发布 | preprocess/tools/package_6module_release.py | 四份使用文档、SN资源/JSON、共用模板、version.txt、manifest | 固定包名SPECT_Core，新输出父目录、无原始矩阵、校验及入口短测通过 |
| 部署 | verify_deployment.py；必要时install_release_overlay.py | 现场SN/资源根/床距/坏道/ROI/方向，校验报告、短运行记录 | 本SN矩阵齐备；现场改动明确，不忽略二进制异常 |
| 收尾归档 | Git diff/status，更新原主题结论 | 关键证据和复现入口，CONTINUITY只留跨会话路由 | 不复制另一份长报告，不顺带删除未核对的实验/用户文件 |

没有统一自动“实验准入/合并”脚本，Git和下面已有检查就是入口；不得虚构已经自动执行的门。
执行完成、科学/任务质量、用户采用是三件事。默认参数、资源归属、矩阵协议或数值结果变化须明确说明并获采用授权。
文档/打包清单调整不自动触发长迭代或整套质量实验；数值、线程或I/O机制变化按影响范围选择回归。

## 3. 实验与合并规则

开始前在各受影响仓库运行 `git status --short`、`git branch --show-current`、`git diff`，读当地 `.agent/CONSTRAINTS.md`。
跨会话恢复才读 `.agent/CONTINUITY.md`；当前两个仓库没有 `.agent/branches` 注册表，不额外造第二份状态台账。
实验脚本/输出归属：预处理放 `experiments/<topic>/`，重建放 `recon_out/<topic>/`，关键结论进入已有 `reports/` 或实验README。
记录源码提交（dirty时附patch）、实际EXE散列、参数、输入身份、硬件和随机种子（如有）。配对比较保持同输入和口径。
仅为调参不复制生产pipeline；用独立CONF。跨仓库实验可用相同分支主题名，但每仓库仍独立提交。

```powershell
# 仅在选定的真实仓库中执行，分支名改为本次主题。
git switch -c codex/topic
git diff --check
# 审查、测试通过后，只stage本次所有权范围的文件，再本地commit。
```

合并之前对照基线查看差异、数值/接口证据、未完成项。没有新采用决定时实验仍是实验，不改生产默认。
目标通常为各仓库main，但必须先核对当前分支和实际基线；不能把名含release的分支当成已合并主线。
2026-09-24此轮之前：recon在main，preprocess在release-260923-scanner-profiles；此说明不授权自动合并后者。
若只需交付当前已批准分支，记录两仓库提交，不伪称都来自main。远端push另需明确要求。
提交后发生源码变化须重新构建/验证受影响产物；仅后续说明归档提交不改变既有EXE来源版本。

## 4. 检查与构建入口

预处理仓库的常规合同检查：
```powershell
Set-Location E:/Xiang_Project/SPECT/6Module
python tools/test_run_6module_preprocess_wrapper.py
python tools/test_run_6module_recon.py
python tools/test_release_tooling.py
python tools/test_install_release_overlay.py
```
只改说明/打包清单时至少跑test_release_tooling和包校验。测试故意注入错误时会打印FAIL，判定以脚本退出码和断言为准。
原生预处理或编号/LUT变化另做已冻结输入的BIN回归；历史PW默认采用不是要求未来一切变更都字节等同。

当前本机构建命令（在相应真实仓库执行，工具链位置换机器需重配）：
```powershell
# preprocess，原生代码变化时
& 'D:/Program Files/VisualStudio/MSBuild/Current/Bin/MSBuild.exe' SPECT_2__1/PreDataProject.vcxproj /t:Build /m:4 /p:Configuration=Release /p:Platform=x64 '/p:CUDA_PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2' /v:minimal
```
```powershell
# recon，复用当前经核对的构建目录；新分支另设独立build目录
Set-Location D:/Projects/SPECT-recon
cmake -S . -B build/csrb-native-20260921
cmake --build build/csrb-native-20260921 --config Release --target SPECT_recon spect_tests --parallel 6
ctest --test-dir build/csrb-native-20260921 -C Release --output-on-failure
# 输出到新staging目录，不能覆盖历史封存包
cmake --install build/csrb-native-20260921 --config Release --prefix D:/Projects/SPECT-recon/staging/SPECT_recon_candidate
```
上述第一条CMake命令复用现有cache（VS18/CUDA13.2、tests=ON、sm86/89/120），不是全新机器通用初始化命令。
`build_release.bat`是现有标准纯发布构建入口（tests=OFF），会重置build/Release运行输出及staging/SPECT_recon_rc_dev；
只能在这些派生目录没有需要保留的任务时执行。它不是CTest替代品，也不触碰冻结dist包。
独立重建RC用 `scripts/package_rc.ps1`；面向同伴的完整链路包用下一节通用打包器，不需要每次额外生成独立RC ZIP。

`tools/run_220_mlem_perf_gate.py`只用于已冻结Lu208五床MLEM50性能/同输入门，不是所有改动的必跑项。
`scripts/run_260601_regression.ps1`是有对应fixture时的历史canonical回归，不取代SC60或Lu joint检查。
数据特定的脚本和路径先查对应主题记录，不能仅因为文件存在就无条件运行。

## 5. 打包发布与部署

两仓库验证并提交后选择输出父目录。整体包固定命名SPECT_Core，不在文件夹/ZIP名中写版本；不再提供--release-name。
version.txt分为[preprocess]、[recon]，独立记录EXE --version及SHA256；[package]记录组装时间和集成脚本Git提交。
程序未重编的一方版本保持不变，文档/脚本/配置变化由tooling_revision和manifest体现。不得手改版本文件冒充新二进制。
正式打包输入必须是指定的已验证重建staging目录，不能自动搜最近EXE。
仅文档修改可复用已验证二进制，但必须记录其原编译提交；本轮含原生SN配置解析修改，必须重新编译后交付。
在预处理仓库执行（重建输入按本次身份替换，路径必须真实存在）：
```powershell
python tools/package_6module_release.py --recon-root D:/Projects/SPECT-recon/staging/SPECT_recon_20260924_clean
python tools/verify_deployment.py --release-root release/SPECT_Core
python tools/test_universal_release.py --release-root release/SPECT_Core --reconstruct
```
每条成功才继续。打包器负责复制二进制/依赖/模板/两设备资源、生成README及manifest，不编译EXE，也不包括矩阵和原始数据。
六份算法CONF从指定重建输入复制；通用包会移除设备matrix_id，按重建config/scanners/<SN>.ini的核素节绑定，05/06都用原生CSRB。SN模板源在recon/multipinhole/config/scanners；预处理核素JSON只维护资源。构建负责复制SN模板，打包检查路径与资产清单一致；外部矩阵路径在部署时配置。
不要把独立RC的历史05 CSRQ绑定当成通用包05。CSRQ仍实验性。
test_universal_release用合成矩阵验证PSC Tc、PSC Lu、SC60 Tc的实际一轮GPU入口/几何，不证明实采质量。
默认完整矩阵未安装时verify的4个WARN允许用于“包完整性”交付；现场须加require-matrix。
二进制/资源发生变化时按其影响补真实短样本回归；只有文档变化不重跑数小时影像实验。
需要ZIP时在新的打包操作加--zip，校验ZIP CRC/散列并记录；现有脚本在其可选--audit之前写ZIP，--audit不是所有上述门的替代。
已有SPECT_Core目录或SPECT_Core.zip会被拒绝覆盖，不自动删除或替换。
后续构建用--output-parent指定新的内部归档父目录（例如release/history/20260925_01），其下仍生成SPECT_Core；
这个父目录仅是本地归档位置，不作为产品名称或组件版本。对外目录/ZIP仍固定名称，以version.txt辨别版本。
用户要求更新当前交付时：先在独立目录组装并通过上述校验，再把现有release/SPECT_Core、同名ZIP及散列文件归档到release/history/<时间>/，最后将新包放回release/SPECT_Core及SPECT_Core.zip。不能只留下候选而让固定交付入口继续指向旧包；历史包保留用于回退。替换前确认没有进程使用该目录，并核对归档前后散列及新目录/ZIP一致性。
E盘为exFAT；替换前关闭指向发布目录的Codex文件预览/文件面板。原生程序退出不等于目录句柄已释放；归档后必须确认旧路径实际消失，否则停止，不能把incoming目录嵌套进残留目录。
源码、staging和已发release是不同层，不手工改包后忘记回写源模板。

部署在新目录安装同一通用包，按设备指南设置当前SN、SN INI矩阵路径及几何，再运行：
```powershell
python tools/verify_deployment.py --release-root D:/SPECT_install --require-matrix --allow-config-overrides
```
有意的配置变化可WARN，二进制/DLL/资源错误不可略过。无现场配置变更时不必allow-config-overrides。
若使用overlay脚本，先--dry-run，备份目标配置；它只覆盖同名文件，保留未列出文件，不清旧docs/profiles，也不自动迁移现场值。
原始矩阵、原始采集、冻结结果和历史发布包不属于清理范围。部署后按SN跑预处理→dry-run→短重建；核对输出和失败处理后再接入完整采集。

## 6. 主要维护点与同步责任

| 变更类型 | 唯一代码/配置主人 | 同步文档与检查 |
|---|---|---|
| 预处理参数/产物/状态 | preprocess/tools/run_6module_preprocess.py、SPECT_2__1原生代码及INI | 完整链路指南；wrapper测试，原生变化加投影回归 |
| 设备协议、编号、LUT、矩阵绑定 | preprocess/SPECT_2__1/scanners、设备资源；recon/multipinhole/config/scanners矩阵路径 | 设备指南；资源身份、矩阵/跨SN负例、实际设备样本 |
| 算法、配置键/默认、矩阵格式 | recon/multipinhole源码与config/profiles | 对应Format_Reader_Matrix_CN.md或参数模板；实际影响使用接口时同步完整链路指南；测试/构建 |
| wrapper或优先级 | preprocess/tools/run_6module_recon.py及recon配置解析 | 完整链路指南；两SN/核素dump和入口检查 |
| 发布内容/依赖/文档数量 | preprocess/tools/package_6module_release.py、verify_deployment.py | README索引、四份包文档、test_release_tooling及最终包验证 |
| 联合工作流程 | recon/DEVELOPMENT.md、setup_spect_workspace.ps1 | 两仓库入口索引；不再复制第二份行为说明 |
| 已批准数值/实验结果 | 该主题原始报告/实验README | 结论、边界、证据/复现命令；阶段结束才同步CONTINUITY路由 |

主要长期维护脚本：两个wrapper、read_preprocess_ini.py、package_6module_release.py、verify_deployment.py、install_release_overlay.py及对应test_*.py；
重建的CMake/build_release.bat、scripts/package_rc.ps1、stage_dev_package.ps1和数值单测；setup_spect_workspace.ps1只负责本地入口。
主要文档：本行为说明；preprocess两份调用指南、设备指南和已跟踪README索引；recon README、multipinhole/doc/README、矩阵格式协议和参数源模板。
preprocess/WIKI.md当前被.gitignore排除，仅为本地历史索引，不作为必须跨机器同步的规范。
CHANGELOG只记用户可见行为/格式/发布变化；本地文档排版不必为了触发版本重编EXE。

临时/证据类：experiments/<topic>/、recon_out/<topic>/的runner/config/图像、tmp、候选build/staging、带日期的专用分析脚本。
它们不随包分发，不成为生产入口；其中支撑结论的配置/哈希/关键输出仍须保留，不能把“临时”理解为都可删除。
已有顶层tools中的日期实验脚本仍按历史证据使用，不趁本轮迁移重命名；可复用能力确有需要时才提升为维护脚本。
旧说明如RELEASE_260720、PW_COG_INTEGRATION_PLAN、旧预处理指南保留仓库参考，不再并列作为当前交付规范。

## 7. 当前已知边界

默认Tc/Lu标准OSEM50:N10；Lu默认prefer_split，high=208/low=113原生CSRB；仅全床无分窗时选择合并BIN及high矩阵，缺一窗/损坏/床间混用拒绝。空矩阵槽不触发输入切换。SC60没有Lu资源。
SC60已有首批三床采集报告21mm；当前0.4mm矩阵下为52.5层，整数层限制仍在，21.2mm是试验近似不是正式机械参数。
早期“尚无实采”的记录是当时状态，不能覆盖现有SC60_ADAPTATION_CN.md及scanner.ini的首批实采证据。
SC60 draft引导LUT/峰锚、绝对角度/CT对齐及定量边界见当前设备指南；没有实采质量证据时不因短测通过而扩大结论。
