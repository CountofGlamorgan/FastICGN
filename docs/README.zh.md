# FastICGN（中文说明）

**基于 [OpenCorr](https://www.opencorr.org/) 的 GPU 加速二维数字图像相关（2D-DIC）。**

英文主文档是仓库根目录的 [`README.md`](../README.md)，本文件只做中文导览，
不重复细节；两者冲突时以英文版为准。

## 这是什么

FastICGN 用数字图像相关（DIC）从图像序列里测全场位移：FFT 互相关给初值，
再用逆向组合高斯-牛顿（IC-GN，一阶形函数）迭代细化。参考图、梯度和每个 POI
的 Hessian 常驻显存并跨帧复用，所以一段长高速录像是**一个流式任务**，
而不是 N 对独立图像。

数值定义来自上游 OpenCorr（源码 vendored 在 `src/`，见 [`NOTICE.md`](../NOTICE.md)）；
FastICGN 加的是 CUDA IC-GN kernel、batched cuFFT 的 FFTCC、带校验且可断点续跑的
多帧输出，以及 Python / agent 管线。两者都是 MPL-2.0。

它**不是** GUI、不是立体/3D-DIC、也不做应变后处理：输出是位移场和形函数梯度场，
分析在下游。

## 最快的验证方式

```bash
bash smoke/selfcheck.sh
```

自带合成散斑数据，一条命令跑通：三帧序列（raw 输出）→ 用发布的 NumPy reader
读回 → known-truth cache regression → 与 CPU 参考 CLI 对比（需要已构建
`opencorr_cli_cpu`，即 `cpu-only` 或 `cuda-fftw` preset，否则该项 `SKIP`）→
越界拒绝规则断言。逐项打印 PASS/FAIL/SKIP 和证据目录。CPU-only 构建下 GPU 相关项
标 `SKIP`，不算失败。

## 构建与运行

见英文 README 的 Requirements / Build / Quickstart 三节。四个 CMake preset：
`cuda-release`（生产配置）、`cuda-tests`、`cuda-fftw`、`cpu-only`。

两种使用方式：

```bash
# 单对图像
./build/cuda-release/opencorr_cli_cuda_mvp ref.tif tar.tif --subset-size 15 1 out.csv

# 整个序列（Python 管线，全场模式）
python3 pipeline/dic_pipeline.py --image-dir frames/ --output-dir results/
```

单对图像命令里位置参数依次是 `<参考图> <目标图> <subset_radius> <grid_space> <输出>`，
`--subset-size 15` 顶替半径，`1` 是网格步长。管线默认只认文件名以 6 位帧号结尾、
扩展名为 jpg/jpeg/tif/tiff/png/bmp 的图像，其他命名用 `--image-pattern`。图像默认按
8-bit 读，16-bit 文件会被降成 8-bit（binary 会在 stderr 报一条 warning、写明该传什么参数，但
照样算完、exit 0）：12/16-bit 数据先加 `--dry-run` 看
`input_recommendation`，再显式给 `--input-depth`（12-bit 还要 `--input-align`）。

## 机器接口（脚本与 agent）

通用用法见根目录 [`README.md`](../README.md)；下面是供脚本与 agent 使用的机器接口。

- `python3 pipeline/dic_pipeline.py --describe` 原样输出机器契约
  [`pipeline/interface_contract.json`](../pipeline/interface_contract.json)：参数、单位、
  默认值、输出 schema（含 COD 公式）、错误 / 警告码、退出码、已知局限。
- `--dry-run`：不写盘、不启动 GPU，stdout 输出 JSON 计划（完整 binary argv、测线、位深
  探针、`--output-dir` 现状、耗时区间）；退出码 0 = 可运行，2 = 看 `errors[]`。
- `--json-summary PATH`：严格 JSON，`verdict` 为 `ok` / `warn` / `fail`（只由 warn / fail
  级告警决定），逐事件给 `status`、`valid_ratio`、`zncc_median`、`warnings`（沿用旧含义：只放
  warn / fail 条目、旧文本）与 `warning_details`（全部条目带级别）、实际几何和 COD 统计。顶层
  `warnings` 为空不代表没有 warning。
- `--strict`：正式运行前做 dry-run 的全部校验（不通过 exit 2），有失败事件 exit 3；不加时
  campaign 有事件失败仍 exit 0。`--max-frames N` 只算前 N 个目标帧。
- raw manifest 的 `engine` 段记录构建 hash、kernel、FFT 后端、越界策略（`oob_policy`）、
  整帧 fallback 策略（`frame_fallback_policy`）、设备。
- binary 自身：`--help` 列出每个选项及默认值（exit 0）；`--version` 给 `git_hash`、
  `build_date`、`version`（语义版本）、`git_dirty`、`backend`、`fftw`；每次计算结束（成功或失败）
  在 stdout 打一行 JSON 状态（`status`、`exit_code`、`error`、帧数、`output_dir`、耗时、
  `warnings`），进度、warning、error 走 stderr，`--quiet` 只去掉每帧的进度行；
  `--verify-only <输出目录>` 不计算、不写盘，核对多帧输出目录是否完整、每个文件是否与 ledger
  一致，stdout 输出一行 JSON 结论，只有完整时 exit 0（见
  [`docs/output-integrity.md`](output-integrity.md)，英文）。
- 结果可不可信怎么判断：[`docs/quality-checks.md`](quality-checks.md)（英文）。

## subset：radius 还是 size

这里踩过坑，单独说一遍：

- binary 的**位置参数**是 subset **radius**（半径，像素）。
- pipeline 的 `--subset` 和 binary 的 `--subset-size` 是**奇数全尺寸 size**。
- 换算 `radius = (size - 1) / 2`，所以 `--subset-size 15` = radius 7。
- 把 15 当 radius 传进去，实际算的是 31 像素子区——曾造成约 100 μm 的假失败。
- binary 拒绝偶数 `--subset-size`，也拒绝同时给位置参数和 `--subset-size`。

## 输出格式

CSV 或 raw（每个 POI 36 字节）+ JSON manifest + 隐藏的完整性 ledger。
坐标方向、单位、有效性判据（`zncc >= 0`）、梯度列含义、`--output-roi/--output-step`
语义，以及 NumPy reader 的用法，全部写在
[`docs/output-format.md`](output-format.md)。

要点：`x` 是列（向右增），`y` 是行（**向下**增），单位像素；`u/v` 是相对固定参考帧
的位移，单位像素；`ux/uy/vx/vy` 是形函数梯度，单位 1/px；`zncc < 0` 的行是状态行，
其 `u/v` 是残留诊断值，**必须排除**。

manifest 的 `engine` 段记录真正跑的是什么：kernel 模式、FFT 后端、
skip/fallback/temporal 策略（`frame_fallback_policy` 记录整帧 fallback 如何与
种子结果逐点合并）、durability、build 身份、CUDA 设备与驱动版本。

## 实验室专用部分

下面这些是为实验室破裂实验写的，不用它们也能用 DIC 引擎：

- **COD 后处理**：`dic_pipeline.py` 给了 `--interface-y` 或 `--line-y` 就进 COD 模式，
  只算两条测线并聚合成界面滑移量 `cod_um`：界面两侧的相对位移（两条测线水平位移 `u` 之差，
  L0 侧 − L1 侧），不是张开量；"COD"是沿用下来的名字。默认 `--cod-method naive`（两测线直接做差，
  2026-10-05 起）；`--cod-method fillboundary` 可选（沿 `∂u/∂y` 外推到两线中点，
  2026-07-09 至 2026-10-05 曾是默认）。fillboundary 遇到缺 `uy` 列的帧**直接报
  `cod_uy_column_missing`**，不再静默退化成 naive。
- **MCP / agent 接口**：`pipeline/mcp_server.py`，带写入沙箱、读白名单和审计日志，
  不含任何数值实现；只支持 COD 模式，plan 与 run 一律带 `--strict` 调 pipeline；campaign 可用
  `events`、`event_interface_y`（对应 `--events`、`--event-interface-y`）。
- **部署路径**：`FASTICGN_GPU_BIN` 等环境变量 → 可选
  `~/.config/fasticgn/config.toml`（模板 `pipeline/config.example.toml`）→ 仓库
  自身的 `build/` 树 → 历史 `/home/chonglang` 路径（仍可用，但会打 deprecation 警告）。
  细节见 [`pipeline/DIC_PIPELINE_GUIDE.md`](../pipeline/DIC_PIPELINE_GUIDE.md)
  和 [`docs/lab/simulacrum.md`](lab/simulacrum.md)。

## 许可证

MPL-2.0，版权 Chonglang Wang。FFTW 是 GPL-2.0-or-later 且可选——生产 CUDA 构建
不链接它。第三方清单以 [`NOTICE.md`](../NOTICE.md) 为准。
引用方式见 [`CITATION.cff`](../CITATION.cff)。
