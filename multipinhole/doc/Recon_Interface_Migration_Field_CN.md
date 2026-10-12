# SPECT重建接口迁移说明（2026-09-14）

适用于当前源码和已更新的6Module部署；2026-09-13等冻结RC维持其原合同。上位机继续使用原两个wrapper调用，不需要自行实现方案合并、矩阵搜索或Lu分窗配对。

## 最小调用

从6Module安装根执行：

```powershell
python .\tools\run_6module_preprocess.py --input-root "D:\data\case\SPECT" --output-root "D:\data\case\preprocess" --isotope Lu177
python .\tools\run_6module_recon.py --preprocess-root "D:\data\case\preprocess" --output-root "D:\data\case\recon"
# 可选显式完整方案；未传参数继续继承配置
python .\tools\run_6module_recon.py --preprocess-root "D:\data\case\preprocess" --output-root "D:\data\case\recon03" --config .\bin\recon\config\profiles\03_mouse_rdp100.conf
```

独立重建RC的EXE为 `bin/SPECT_recon.exe`，6Module安装的EXE为 `bin/recon/SPECT_recon.exe`；配置均在各自EXE旁的 `config/`。预处理仍使用release `Spect_preprocess.exe`、资源及wrapper，不以其他ENG转换脚本替代。

## 需要调整的旧假设

| 旧行为/假设 | 当前合同 |
|---|---|
| `-it N:S`中S为保存频率 | `-it A:B`为迭代数:子集数；保存用 `-save-every N`，默认0仅末次 |
| compact固定一个子集 | OSEM/BSREM共用角度子集；CSRB及实验CSRQ均有N10和子集CTAC路径，调用参数不变；CSRQ N>1 CTAC的K4/K8、驻留/流式、双床双窗GPU数值回归已通过，仍实验、未发布 |
| 每次显式拼算法、PSF和几何四件套 | 普通wrapper只传输入输出，采集核素从summary继承；其余由方案、global及metadata解析 |
| `-opti bsrem`自动加载RDP | 已选方案不随优化器变化；明确选RDP/TV方案 |
| true只保护PSF/Filtering等四节 | true保护配置中所有已填写重建字段，包括算法、迭代、矩阵和norm |
| 一个 `260601`适用于所有数据 | 方案精确绑定核素/模型/matrix_id，不能替代本批Tc的141keV SCALE2矩阵 |
| Lu只有 `BedN.bin` | 新预处理同时保留合并BIN和独立208/113 BIN及summary身份 |
| 每核素一份norm必需 | 默认关闭；joint需匹配208/113的两份校正文件，不能套用合并norm |
| 默认输出smooth和butter | Gaussian关闭；最终primary和独立输出Butterworth |
| 看到mask即完成 | 需退出0、run JSON completed、最终体数据及manifest验证 |

## 方案与CLI优先级

全局 `default_config_Tc99m/default_config_Lu177` 分别选04 OSEM50×10与05 joint RDP30×10。不传配置用核素默认；显式 `--config recipe.conf`替换默认方案，不继承另一份默认方案；方案＋`--`后CLI按开关合并。

- false：显式CLI > 所选方案 > 全局；源码默认。
- true：所选方案 > 全局 > CLI，仅限已配置的重建字段；当前E260903部署保留此值。
- 输入输出、设备、线程及日志仍由调用方指定。PSF/Mask/Filtering/Regularization未在CLI传入时沿用配置。

上位机若继续传 `-- -it 40:10`，true模式下方案的迭代数优先；需要CLI覆盖时明确在选定配置中使用false。不要为适配而去掉PSF或改变Mask。每次正式wrapper日志记录实际argv合并后的配置及来源哈希。

## 输入输出合同

BIN保持8字节int32头＋float64数据，标准360×2304。Lu双窗共用固定标定，先得到合并与208，113=合并−208，检查有限非负与计数守恒。旧合并数据没有分窗身份时只能选03合并方案或重跑预处理；不能用复制/改名伪造第二窗。

床号保持真实连续Bed1..Bed5，步进16.8mm，`z_bed_flip=1`；投影identity和metadata几何必须匹配，不复用旧108×108×80占位参数。输出几何从MHD读取，Mask/裁剪/拼接不由上位机重算。每床预处理之后一次多床重建，停止原按床重建后另跑Stitcher的主线。

DICOM入口为最终 `_itN.mhd/.raw`，可按明确显示合同选择 `_butter`并记录来源。无额外Gaussian输出；primary仍含既定PSF/sieve保存处理。不能把 `-it A:B`的B当保存间隔去挑文件。CT/DICOM坐标配准属于后续处理，不能为目测对齐随意反转床序。

## 最小现场检查

```powershell
python .\tools\run_6module_recon.py --preprocess-root "D:\data\case\preprocess" --output-root "D:\data\case\check" --dry-run
.\bin\recon\SPECT_recon.exe -isotope Lu177 -dumpConf
```

核对核素、方案、优先级、迭代/子集、实际矩阵ID、两窗顺序、norm、PSF和输出滤波。dry-run只检查装配，目标GPU仍需一例正式运行核对最终体数据。新构建的help已同步完整配置保护与joint子集说明；旧EXE仍应以实际dump核对。

完整CLI与五方案参数见 [使用说明](USAGE_CN.md)、[参数说明](PARAMETERS_CN.template.md)。6Module安装包中 `docs/RECON_CONFIG_AND_CALL_GUIDE_CN.md` 和 `docs/DEPLOY_OVERLAY_CN.md` 是wrapper及部署入口。
