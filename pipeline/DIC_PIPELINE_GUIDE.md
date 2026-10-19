# FastICGN pipeline — 使用指南

> 图像序列 → 全场位移场（通用），或 → COD CSV（实验室后处理）

> **路径说明（lab-specific examples）。** 文中出现的绝对路径、主机名和数据集
> （`/home/chonglang/...`、`/data/lab-jobs/...`）都是原作者实验室那套部署的实例，
> **不是通用要求**。真正的解析规则见下面的「路径解析」一节，配置模板见
> [`config.example.toml`](config.example.toml)；COD 模式本身也是实验室专用后处理。

## 两种模式

- **不给 `--interface-y` / `--line-y`** → **全场位移模式**（通用入口）：
  binary 算整幅 ROI，frame CSV 或 raw 留在 `<output-dir>/<event>/`，不做聚合，
  `--cal` 也不需要。这是给仓库之外的用户的默认路径。
- **给了 `--interface-y` 或 `--line-y`** → **COD 模式**（实验室专用）：
  binary 只算两条测线，pipeline 再聚合成 COD CSV，`--cal` 必填。这里的 COD（列 `cod_um`）是
  **滑移量**：界面两侧的相对位移（两条测线水平位移 `u` 之差，L0 侧 − L1 侧），不是张开量。
  下文没有特别说明的地方讲的都是 COD 模式。

## 路径解析（2026-09-17 起可配置）

`--gpu-bin` 的缺省值不再硬编码。解析顺序：

1. 命令行 `--gpu-bin`
2. 环境变量 `FASTICGN_GPU_BIN`
3. 配置文件 `$FASTICGN_CONFIG` 或 `~/.config/fasticgn/config.toml` 的
   `[paths].gpu_bin`（模板见 `pipeline/config.example.toml`）
4. 仓库自身的 `build/{cuda-release,cuda-tests,cuda-fftw,cpu-only,}` 树
5. 实验室旧路径 `/home/chonglang/opencorr-gpu-mvp-build/opencorr_cli_cuda_mvp`
   —— **文件存在时仍然可用**，但会在 stderr 打一条 deprecation 警告，并在
   `--json-summary` 的 `params.gpu_bin_source` 里记成 `legacy_lab_path`。

同一套顺序也适用于 `FASTICGN_CPU_BIN`、`FASTICGN_HOME`、
`FASTICGN_READ_ALLOWLIST`（MCP 读白名单，默认 `/data`、`/media` 加仓库根）、
`FASTICGN_JOB_ROOT`（MCP 写根，默认 `~/dic_jobs`）。
`run_dic.sh` 另外认 `FASTICGN_PIPELINE` / `FASTICGN_PYTHON` / `FASTICGN_VENV`。

Python 依赖：`pipeline/requirements.txt`（numpy、Pillow），MCP 另加
`pipeline/requirements-mcp.txt`。旧的 `/tmp/phase5-cod-venv` sys.path 注入已删除。

## 文件位置

- **脚本真身（git 管理，2026-07-10 起）**: `simulacrum:/home/chonglang/opencorr-gpu-mvp-src/pipeline/dic_pipeline.py`；build 目录内是指向它的 symlink——**改脚本请改 git 仓库并 commit**
- **Wrapper**: `/home/chonglang/opencorr-gpu-mvp-build/run_dic.sh`（推荐，自动激活 venv）
- **GPU binary**: `/home/chonglang/opencorr-gpu-mvp-build/opencorr_cli_cuda_mvp`。当前版本以 `--version` 为准（2026-10-04 实测 `git_hash=638ad1845d01`、`build_date=2026-10-03T14:54:01Z`）；同目录的 `opencorr_cli_cuda_mvp_v11_hibit` 等是历史 binary，不是生产入口。本机 `~/.config/fasticgn/config.toml` 把 pipeline 和 MCP 的默认 binary 指向它，所以下文"路径解析"第 5 级的 deprecation 提示在本机不会出现
- **Python venv**: `/home/chonglang/opencorr-gpu-mvp-build/.venv/`（numpy 等依赖）
- **服务器**: Simulacrum（2× RTX 6000 Ada, CUDA 13.2）

> ⚠️ **v5_csv_opt 及更早 binary 已弃用**：v5 在低 SNR / 较大位移数据上 60–90% POI 落 `zncc=-5` 失败（`final_conv` 是 per-thread 寄存器，非 tid 0 线程在 iter 1 后误 break）。v9 已修复但只有 5 列输出——默认的 naive 聚合（2026-10-05 起）不需要 `uy`，5 列可以直接聚合，summary 记 `cod_mode: "naive"`；**显式 `--cod-method fillboundary` 喂 5 列会直接报错** `cod_uy_column_missing`（2026-09-17 起；此前是补零静默退化成 naive）。COD 要 fill-boundary 口径必须用 9 列 binary（v10 及之后）。

## 快速开始

### 1. 准备数据

将高速相机照片按事件放在子文件夹中：

```
PHOTO/
  1/          ← 事件1
    000001.jpg
    000002.jpg
    ...        （601 或 801 张）
  2/          ← 事件2
    000001.jpg
    ...
```

图片要求：
- 灰度或彩色均可（自动转灰度）
- 文件名包含 6 位数字帧号（如 `000001.jpg`、`15_C001H001S0001000001.jpg`）
- 每个子文件夹的第一张图自动作为参考帧

### 2. 运行

```bash
# 推荐：用 wrapper（自动激活 venv）。它只在登录 shell 的 PATH 里（~/.local/bin），
# `ssh simulacrum '<命令>'` 这类非登录 shell 找不到，所以一律写绝对路径
/home/chonglang/opencorr-gpu-mvp-build/run_dic.sh \
  --photo-dir /path/to/PHOTO/ \
  --output-dir /path/to/RESULTS/ \
  --subset 15 \
  --step 1 \
  --interface-y 131 --gauge 16 \
  --cal 0.5641 \
  --roi-cols 72,958
```

这是 config1 30 kN event01–20 的惯例参数：界面 y=131，测线取界面 ±8 px（123/139）。30 kN 的界面行
按段登记：event21–25 的界面在 133（整个试样在 event20 → 21 之间相对相机平移约 2 px），要用
`--interface-y 133 --gauge 16`（125/141）。一次 campaign 跑两段：`--interface-y 131` 给默认段，
`--event-interface-y 'event2[1-5]=133'` 给后段（2026-10-05 起；命令与实测见
[`docs/lab/simulacrum.md`](../docs/lab/simulacrum.md) 的 campaign 一节）；只跑部分事件用 `--events`。`--gauge` 要显式写，
不写就是默认 35（`--interface-y 133` 配默认 gauge 是 116/150）。别的数据集先查该文的"数据集惯例参数"，
换一批数据先量界面行。

### 3. 获取结果

```
RESULTS/
  1.csv       ← 事件1的 COD 数据
  2.csv       ← 事件2的 COD 数据
  ...
```

CSV 格式：
```
frame,x_mm,cod_um
2,5.0769,-3.5993
2,5.6410,-2.4083
...
```

- `frame`: 帧号，取自图像文件名（默认是扩展名前的 6 位数字）；参考帧没有行
- `x_mm`: `x_px × cal`，原点是图像第 0 列，不是 ROI 起点
- `cod_um`: **滑移量**（μm）：界面两侧的相对位移，即两条测线水平位移 `u` 之差，`L0 侧 − L1 侧`；
  不是张开量（2026-10-05 定）。列名 `cod_um`、参数 `--cod-method` 和"COD"这个缩写是沿用下来的名字，不改。
  默认 naive（2026-10-05 起）直接相减；`--cod-method fillboundary` 先把各线 `u` 沿各自 `∂u/∂y`
  外推到两线中点再相减。用的是哪种看 summary 的 `events[].cod_mode`。L0 是 `--line-y` 的第一个值或
  `interface_y − gauge//2`。正负号只由公式决定：`u_L0 > u_L1` 为正，两条测线的顺序反过来符号就反过来
  （完整定义见契约 `outputs.cod_csv`）

---

## 参数说明

### 主要参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--photo-dir` | 总照片目录（含事件子文件夹） | `/media/disk/PHOTO/` |
| `--output-dir` | 输出目录 | `/media/disk/RESULTS/` |
| `--cal` | 像素标定值 (mm/px)，**COD 模式必填**，全场模式不用 | `0.5641` |

COD 模式的测线位置（三选一；都不给就是全场模式）：

| 方式 | 参数 | 说明 |
|------|------|------|
| 给中线 | `--interface-y 133` | 断层面 y 像素值，按默认 gauge 35 在上下各偏移 17 px 取测线（得到 116,150） |
| 中线 + 自定义间距（实验室 30 kN 惯例） | `--interface-y 131 --gauge 16` | 手动控制两测线间距（像素）；这组得到 123,139，即界面 ±8 px。界面行要按事件所在段给：30 kN event21–25 的界面在 133，用 `--interface-y 133 --gauge 16`（125,141）；给错段会让一条测线的子区跨进过渡带 |
| 直接指定测线 | `--line-y 123,139` | 直接给出 L0 和 L1 的 y 值；外推目标是两线中点 |
| 逐事件界面行（只限 campaign） | `--interface-y 131 --event-interface-y 'event2[1-5]=133'` | 匹配 `PATTERN` 的事件改用界面行 `Y`，其余用 `--interface-y`；可重复；`--gauge` 全局；不能与 `--line-y` 同给 |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--subset` | 15 | 子集大小（像素）。**注意是 size 不是 radius**，内部换算 `radius=(size-1)//2`，即 `--subset 15` = radius 7。binary 侧现在也有 `--subset-size 15`（等价 radius 7，与位置参数二选一，输出逐字节相同） |
| `--step` | 1 | 网格步长（像素） |
| `--output-format` | `csv` | 全场模式的 frame 产物格式 `csv/raw`；COD 模式必须 csv（聚合读 CSV），传 raw 会被拒 |
| `--image-pattern` | —（六位帧号规则） | 图像发现正则，对文件名做忽略大小写 `search`。第一个捕获组是纯数字就当帧号，否则按排序从 1 编号 |
| `--image-order` | `frame-number` | 排序策略；`name` = 按文件名排序 |
| `--gauge` | 35 | 派生测线的请求间距（像素）；实现取 `half=gauge//2`，故奇数会向下成偶数实际间距（默认实际 34 px） |
| `--cod-method` | `naive` | COD 聚合方式。`naive`=直接 `u_L0−u_L1`（默认：2026-10-05 起，2026-07-09 以前也是）；`fillboundary`=用 ∂u/∂y（frame CSV 按 0 起算的第 5 列、即人读第 6 列 `uy`）把两测线的 u 线性外推到两线中点再做差（Eq.3，对齐 paper §2.3，消除测线离界面 half 像素的系统欠估；可选，2026-07-09 至 2026-10-05 曾是默认）。改回 naive 的原因：测线干净时 fill-boundary 只高几个百分点，但单点噪声是 naive 的 2–3.5 倍，界面行错 2 px 时偏低约 22%（naive 约 6%），见 `docs/quality-checks.md` 第 6 节。fillboundary 下 CSV 无 uy 列时**报错** `cod_uy_column_missing`，不再自动退化 |
| `--conv-threshold` | 0.001 | ICGN 收敛阈值，**恒显式透传**给 binary（钉死语义、不依赖 binary 自身默认；0.001=CPU 对齐值，老 binary 默认 0.05 会留 ~4μm COD plateau） |
| `--input-depth` | —（binary 默认 8） | 输入位深透传 `8/12/16/auto`。12-bit 左移数据（highspeed2）**必须显式** `--input-depth 12 --input-align left`，别用 auto（会误判 16） |
| `--input-align` | — | 12-bit 在 16-bit 容器内的对齐 `right/left/auto`；`left`=存储值÷16 |
| `--input-normalize` | — | 灰度归一化 `none/to-8/to-16/unit`（ZNCC 线性不变，不影响 DIC 结果，仅改浮点范围） |
| `--roi-cols` | 全图 | COD 聚合的水平 ROI，格式 `x1,x2`（含两端；反序会让 COD 为空） |
| `--max-frames` | 全部 | 只算参考帧之后的前 N 个目标帧（N≥1，试跑用）；dry-run、真跑、摘要一致 |
| `--events` | 全部 | 只限 `--photo-dir`：只跑部分事件。逗号分隔的事件名或 glob（fnmatch，区分大小写），如 `'event0*,event1*,event20'`、`'event2[1-5]'`。每一项都必须匹配到至少一个含图片的事件子目录，否则读图之前报错（`event_not_found` / `event_pattern_no_match` / `no_events_selected` 等），**不加 `--strict` 也 exit 2**，不跑 binary、不写输出（给了 `--json-summary` 只写失败摘要，同 `--strict` 预检拒绝）。筛选条件、选中数与被筛掉的事件名进 dry-run 与摘要顶层的 `event_selection`；被筛掉的事件不规划、不运行、不报 |
| `--event-interface-y` | — | 只限 `--photo-dir`，可重复，`PATTERN=Y`：事件名匹配 `PATTERN`（事件名或 glob，一条一个）的事件用界面行 `Y`，其余用 `--interface-y`（必须给）；测线 `Y∓gauge//2`。下列情况读图之前报错、**任何模式都 exit 2**：格式不对或 `Y` 不是非负整数（`invalid_event_interface_y`）、与 `--line-y` 同给、缺 `--interface-y`、某条不匹配任何选中的事件（包括只匹配被 `--events` 筛掉的事件）、一个事件命中两条不同的 `Y`（值相同的重复不算冲突）。每个事件实际用的界面行、来源（`interface_y` / `event_interface_y`）和命中的覆盖见 dry-run 与摘要的 `events[].geometry` |
| `--keep-frames` | 否 | 保留 COD 模式的中间 frame CSV（全场模式的帧就是产物，总是保留；binary 或聚合失败时也保留）。COD 为空时和 638ad18 一样照常删除，好让改参后往同一目录重跑不被 binary 拒绝 |
| `--strict` | 否 | 严格模式：先做 dry-run 的全部校验（不通过 exit 2，什么都不写），另外拒绝 16-bit 文件按 8-bit 读（`input_depth_truncates_container`，报错里给出探针建议的参数）；campaign 的预检是全有或全无（一个事件不过，整批都不跑）；预检的 warning 写进摘要顶层 `warning_details`；全零 COD、目标帧号重复按失败处理，COD 为空时保留帧；有失败事件 exit 3。不加时退出码保持旧行为 |
| `--saturation-warn-fraction` | 0.05 | =255 像素比例超过它，`input_8bit_max_255` 才记为 warn，否则 info。COD 模式看各测线带的最大占比（测线带 = 该测线参与 COD 聚合的子区像素：行 L±r，列为 `--roi-cols` 两端各扩 r，没给 ROI 时为 binary 在该测线上的 POI 范围各扩 r，裁剪到图像内）；全场模式看参考帧整帧。启发式、可调，见下文"结构化运行摘要" |
| `--json-summary` | 否 | 写入结构化运行摘要 JSON；含全量参数、binary provenance、事件质量统计、几何与 COD 统计、`warnings`（只放 warn/fail 条目，沿用旧文本）、`warning_details`（全部条目带级别）与 verdict |
| `--dry-run` | 否 | 只读校验并向 stdout 输出 JSON 计划（含输出目录现状与耗时区间）；不创建输出目录、不启动 GPU；0=可跑、2=有错误；`warnings` 要逐条看，`notes` 只是说明 |
| `--describe` | 否 | 独立使用；原样输出机器可读 `interface_contract.json` 后退出 |

`--conv-threshold` 写 `nan`/`inf` 会在解析参数时直接报错（exit 2）；`--cal` 只在 COD 模式检查（`nan`/`inf` 报错 exit 2），全场模式用不到它、也不检查。选项可以像 638ad18 一样写成无歧义的缩写（`--dry`、`--desc`、`--keep`），有歧义的（`--d`）报错。
偶数 `--subset`、`--cal` ≤ 0、`--gauge` ≤ 1、两条测线相同、`--roi-cols` 反序这类 dry-run 会拒绝的参数，不加 `--strict` 时照旧运行，但会打印并在摘要顶层记一条 `argument_warning`。

### 单事件模式

如果只处理一个事件，用 `--image-dir` 代替 `--photo-dir`：

```bash
/home/chonglang/opencorr-gpu-mvp-build/run_dic.sh \
  --image-dir /path/to/PHOTO/1/ \
  --output-dir /path/to/RESULTS/ \
  --interface-y 131 --gauge 16 \
  --cal 0.5641
```

输出为 `RESULTS/cod_result.csv`。

### 结构化运行摘要

正常运行建议加 `--json-summary "$OUT/summary.json"`。摘要 schema 版本为 1，包含：

- `params`：argparse 全量参数；固定输入模式给出 `input_*_effective`，`auto` 则诚实
  标为 `runtime_per_image`
- `binary`：由生产 binary `--version` 读取的路径、git hash、build date
- `events`：目标帧数、每帧 POI 行数、pooled valid ratio、有效 ZNCC 中位数、COD
  路径、参考帧实际输入模式与 warnings（`auto` 的目标帧仍可能逐帧不同）
- `global`：wall time、成功/失败事件数
- `verdict`：`ok`、`warn` 或 `fail`

`valid_ratio` 与 binary QA 同口径：全部 frame 行中
`finite(zncc) && zncc >= 0` 的 pooled 比例。它不是逐帧比例的简单平均；统计位移
`u,v` 时还应额外过滤非有限值。

每条 warning 都有级别（`warning_details` 里的 `severity`）：`info` 只记录，`warn`
让 verdict 变 `warn`，`fail` 让事件失败。摘要里两个列表分工：`warnings` 沿用旧含义，只放
warn/fail 条目，旧 code 保持原文（如 `input_8bit_max_255: possible saturation_or_high_bit_truncation`、
裸的 `gpu_cli_failed`），按固定文本把关的旧脚本照常可用；`warning_details` 列出全部条目（含 info）、
级别和完整细节（百分比、退出码、`Error:` 行）。verdict 按 `warning_details` 的级别算。常见的有 valid ratio <0.95、ZNCC 中位数 <0.9、
事件帧数偏离 campaign 唯一众数（warn），以及 8-bit 首帧有像素等于 255 的
`input_8bit_max_255`：判据占比不超过 `--saturation-warn-fraction`（默认 0.05）时是 info，
超过才是 warn。COD 模式的判据是各测线带（该测线参与 COD 聚合的子区像素）=255 占比的最大值，
全场模式是参考帧整帧占比（`input_saturated_fraction`）；16-bit 文件按 8-bit 读时总是 warn
（可能是高位截断）。判据、阈值、整帧占比和每条测线带的行列范围、占比、含 255 的子区比例、
255 超过 10% 的子区比例都在 `events[].input_saturation`（`warning_details` 的 detail 里也有）；
dry-run 对每个事件给出同一个对象，预告的级别与真跑一致。2026-10-04 起 COD 模式改看测线带：
实验室参考帧在第 0–108 行有 1,400 个以上的 255 像素（第 0–40 行每帧都是 796 个），远离测线、
不进任何子区，单这些就占整帧约 0.73%；而按 30 kN 的测线几何量，config1 40 kN 与 lowrough 20 kN
的下测线带占比是整帧的 2.2–2.4 倍（实测例子见 `docs/quality-checks.md` 第 5 节）。默认值 0.05 是启发式，不是按相关误差标定的上限。COD 跑完会自检（`cod_stats`）：COD CSV 只有
表头是 `cod_empty`（事件失败；帧 CSV 和以前一样删掉，`--keep-frames`/`--strict` 才保留），测线在帧里没有行是 `cod_line_missing`，
全为 0 是 `cod_all_zero`（warn，`--strict` 下失败）。失败事件的 `cod_csv` 为 null，同一路径
上留着的旧文件记为 `stale_cod_csv`（不移动、不删除，文件名不变：只看退出码再按文件名读 CSV 的脚本会读到旧结果，
要看 `events[].status`）。binary 的完整 stderr 存在帧目录旁边的
`<输出目录>/<帧目录名>.binary.log`。完整词表、级别和处置见契约
`outputs.json_summary.warning_vocabulary`；顶层 `warnings` 为空不代表没有 warning。
不传该 flag 时不做额外统计或 version 探针，但 campaign 的控制流也不同：只有 1 张图的
事件被静默跳过、单个事件抛异常会中止整个 campaign；传了则记为失败事件、其余事件继续。

### 机器可读接口契约

`pipeline/interface_contract.json` 是参数、单位、默认值、枚举、输出 schema、退出码、
坑清单与已知局限的单一事实源。`run_dic.sh --describe` 原样输出该文件，可直接接
`python3 -m json.tool`。Pipeline argparse defaults/choices、binary
`parseCommonOption` flags、9 列顺序和 [`docs/lab/simulacrum.md`](../docs/lab/simulacrum.md) 的生成坑清单都有自动一致性测试。

C++ binary 的 `--help`：`--version` 输出里带 `version=` 行的构建（2026-10-05 起）会在 stdout
打印 Usage 和每个选项的默认值、exit 0；更早的构建没有 `--help` 分支（2026-10-04 在生产 binary
`638ad18` 上实测如此），会报 `Error: unknown option: --help`、打印 Usage 后 exit 1。同一批新构建在每次
计算结束时（成功或失败）往 stdout 打一行 JSON 状态（pipeline 照旧丢弃 binary 的 stdout，只留 stderr 日志），
并支持 `--quiet` 和只读核验 `--verify-only <输出目录>`，见契约 `binary.process_output` 与
`binary.modes`。binary 审计以源码 `printUsage`/`parseCommonOption` 与 `--version` 为准。普通 pipeline CSV 模式不写 binary manifest，摘要中的 git hash/build date
来自只读 `--version` 探针。

### Dry-run：先校验再占 GPU

把正式命令原样加 `--dry-run`，stdout 会得到 JSON：路径与事件/帧数、每个事件首帧
分辨率和位深探针、参数错误、将执行的完整 binary argv（相对路径已转成绝对路径）、
`--output-dir` 和各事件帧目录的现状（`output_state`：已有 COD CSV 会被替换、已有
ledger 时 binary 只在参数/输入/binary 都不变时续跑、没有 ledger 的旧帧会被 binary
拒绝、路径是文件）、重复帧号，以及按模式标定的耗时（`runtime_s` 和
`runtime_s_range`，2026-10-03 用当时的生产 binary 实测标定；设超时用区间上端）与磁盘
估计。探针会识别“uint16 容器 + 低 4 位恒零 + max>4095”为 12-bit left，并明确记录
`auto` 会误判为 16 的局限。

dry-run 退出码 0 表示计划通过，2 表示 `errors` 非空。不要直接 pipe 给 formatter
后再读取 `$?`，那会得到 formatter 的退出码；先把 JSON 写到临时文件并保存 wrapper
退出码。dry-run 不创建 `--output-dir`，也不会执行 binary 或写 `--json-summary`。
它会比较已有 ledger 记录的目标列表（例如 `--max-frames` 把已跑完的列表缩短会报
`frame_dir_targets_changed`），但不比较 ledger 里的 binary / 输入指纹，也不检查 `--gpu-device`
是否存在，所以计划通过后真跑仍可能被 binary 拒绝（resume rejected、invalid device ordinal）。
`warnings` 只放 warn 级（要逐条看），`notes` 放 info 级（`gpu_bin_outside_checkout`、
`max_frames_applied` 等），按配置文件部署时 `warnings` 仍为空。

### FastMCP stdio（agent 工具调用）

依赖固定在 `pipeline/requirements-mcp.txt`（`fastmcp==3.4.4` 加 `requirements.txt`）。
生产 venv 已经装好（2026-10-03：fastmcp 3.4.4、numpy 2.4.4、Pillow 12.2.0），**不要再对它
执行 `pip install -r`**：`requirements.txt` 写的是 `Pillow>=9.0,<12`，照装会把生产 venv 的
Pillow 降级。新环境才按 requirements 安装。

stdio server 启动命令：

```bash
/home/chonglang/opencorr-gpu-mvp-build/.venv/bin/python \
  /home/chonglang/opencorr-gpu-mvp-src/pipeline/mcp_server.py
```

客户端通常把该命令放在 `ssh -T simulacrum ...` 后运行；实测可用的 `.mcp.json` 见
[`docs/lab/simulacrum.md`](../docs/lab/simulacrum.md)。服务恰好提供 9 个工具：契约/坑清单、图像探针、dry-run 计划、
单事件/campaign 运行、已有 frame CSV 的 COD 重聚合、summary/manifest 查询，以及取消作业的
`dic_job_cancel`（第 9 个，2026-10-05 起）。
运行结果复用本脚本的结构化摘要，不另造数值口径。`params` 仍包在一层 `{"params": {...}}` 里，
但 tools/list 已给出每个键的类型、单位、枚举、范围、默认值和说明（契约 `mcp.run_params`
同步记录），server 还在 initialize 时下发调用顺序与常见坑；不必先读整份契约。
`dic_describe` 不带参数时只返回"跑之前要知道"的部分（2026-10-05 起，约 15 KB，单测卡在 20 480 B 以内；此前是分节索引加
整节 `mcp` 和 `exit_codes`，约 33 KB）：分节索引、`mcp.run_params`、`mcp.guardrails`、工具名等；
其余部分列在 `mcp_view.on_demand` 里，按节名取：`section="mcp.results"`（返回体说明）、
`"mcp.errors"`、`"mcp.job_record"`、`"mcp.manifest_query"`、`"mcp.known_issues"` 等，
`section="mcp"` 取整节，`section="exit_codes"` 取退出码；取整份契约要显式传 `section="all"`（2026-10-05 实测约 140 KB）。

**MCP 只支持 COD 模式**：运行类工具必须给 `cal` 和 `interface_y`/`line_y`，不能跑全场模式，
也不接受 `output_format`、`image_pattern`、`image_order`。`max_frames` 对应本脚本的
`--max-frames`（只算每个事件前 N 个目标帧，试跑用）。

**campaign 的事件筛选与逐事件界面行**（2026-10-05 起，只用于 `dic_plan` 与 `dic_run_campaign`，
即给 `photo_dir` 时）：`events` 是字符串数组，每项一个事件名或 glob，对应本脚本的 `--events`
（按逗号拼接，所以一项里不能有逗号，glob 的字符类写 `[12]` 不写 `[1,2]`）；`event_interface_y` 是
`{"pattern": ..., "interface_y": ...}` 对象数组，每项对应一条 `--event-interface-y PATTERN=Y`，
必须同时给 `interface_y`（其余事件用它），不能和 `line_y` 同给。规则全部沿用本脚本：MCP 只查 JSON
形状（不是数组、空数组、项不是字符串、对象缺键或多键、`interface_y` 不是整数 → `invalid_events` /
`invalid_event_interface_y`，带 `field`、`expected`、`hint`；缺 `interface_y` →
`event_interface_y_requires_interface_y`），匹配不到、事件不存在、覆盖冲突、覆盖不匹配任何选中事件、
与 `line_y` 同给、负的界面行等由 dry-run 报，`dic_plan` 里是 `valid: false` 加 `errors`，运行工具是
`plan_invalid`，什么都不写。30 kN 一次跑两段：

```json
{"params": {"photo_dir": "/data/lab-jobs/bimaterial/config1-bimaterial/30kN/photos",
            "output_dir": "/home/chonglang/dic_jobs/<新目录>", "subset": 15, "step": 1, "gauge": 16,
            "cal": 0.5641, "roi_cols": [72, 958], "interface_y": 131,
            "event_interface_y": [{"pattern": "event2[1-5]", "interface_y": 133}]}}
```

返回里有本脚本的 `event_selection` 与每个事件 `geometry` 的 `interface_y`、`interface_y_source`；
`mcp.cod_definition.per_event_lines` 按测线分组列出事件（每组最多列 10 个名字），`mcp.brief` 写出
`events filter selected N of M` 和各组测线；超过 10 个事件的精简记录多带 `line_ys`、`interface_y`、
`interface_y_source`；`event_selection` 里的名单超过 10 个只留前 10 个（计数不变，
`mcp.event_selection_view` 说怎么取全文）。不给这两个键时，发给本脚本的 argv 和返回字段与以前相同。

**MCP 一律带 `--strict`**（server 固定，不能关）：`dic_plan` 与运行工具的判定一致，全零 COD 也算
事件失败。事件失败时本脚本 exit 3，运行工具照常返回摘要（`verdict` 为 `fail`、`ok` 为 false、
`mcp.job.exit_code` 为 3）；本脚本自己的预检拒绝（exit 2，只在 plan 之后输入或输出目录变了时
出现）返回与 plan 校验同形的 `plan_invalid`。campaign 的预检全有或全无：一个事件不过，plan 就无效、
run 一个事件都不跑。`mcp.brief` 一行给出 verdict、失败事件的名字、实际测线、COD 行数与峰值和带级别的
warning。超过 10 个事件的 plan / campaign 结果每个事件只给一行精简记录（`mcp.events_view` 说明怎么取
全文，并给出失败事件及每个的一行原因）；plan / run 的每个回复（含 `plan_invalid` 等拒绝）都不超过
48 KiB，省略的内容在 `omitted` 里给出计数和取全文的方式；完整摘要在输出目录的 `summary.json`，用
`dic_query_manifest` 取。

server 强制执行以下边界：读限 `/data`、`/media`、仓库根（本机
`/home/chonglang/opencorr-gpu-mvp-src`）、`/home/chonglang/DIC*` 和 job root；写限 job root
子目录（job root 默认 `~/dic_jobs`，可由 `FASTICGN_JOB_ROOT` 或配置文件改）；路径先 resolve，因此 symlink 不能越界；
路径中的 C0/DEL 控制字符也会拒绝，不能注入 tar-list 记录。
subset 为奇数 SIZE 5–51、step 1–20、cal `(0,10]`、gauge 4–200、timeout
最多 3600 s、campaign 最多 120 个事件；`line_y` 两个互异非负行号（顺序决定 COD 符号），
`roi_cols` 非负且有序，`gpu_device` 不超过本机可见 GPU 数；`output_dir` 不能落在另一个 run 目录
（含其 `_frames_*`）里（`nested_output_dir`）。生产 binary 固定且每块 GPU 有
非阻塞锁；相同 canonical 输出目录也有独立非阻塞锁。两把锁由作业进程（job runner +
pipeline）持有：客户端关闭会话、server 被 SIGTERM 后作业照常跑完，锁一直占到作业结束，
新会话得到带占用者信息的 `gpu_busy`。作业状态写在输出目录的 `.mcp-job.json`，pipeline
输出在 `.mcp-pipeline.*.log`。timeout 由作业进程执行，终止整个进程组；pipeline 单独被杀时作业进程
也会清掉它留下的 binary。作业进程本身被杀时，server 给作业另起一个作业进程接着盯（起不来才自己等，
锁照旧占着），记为 `runner_lost`（退出码未知，看 verdict）；什么都没留下也没有摘要时报 `job_lost`。
非空旧 run 目录即使 `overwrite=true` 也拒绝，避免 pipeline 的 resume 行为混入陈旧 frame。

**长作业**（2026-10-05 起）：run 工具默认仍阻塞到结束。`params` 里加 `"wait": false` 则作业起来后立刻
返回 `job_id`、`output_dir`、`job_record`、`cancel_token`；之后用
`dic_query_manifest(run_dir=output_dir, documents=[".mcp-job.json"])` 看 `state` 和 `progress`
（作业进程每 3 s 写一次：已完成事件数 / 帧数、当前事件），`state` 为 `exited` 后读 `summary.json`；
`dic_job_cancel(output_dir)` 取消（向作业进程的进程组发 SIGTERM，记录写成 `cancelled`，锁释放）。
别的会话取消要带 `cancel_token`，否则 `job_not_owned`；重复取消、取消已结束的作业都返回
`action: "none"`。细节见 [`docs/lab/simulacrum.md`](../docs/lab/simulacrum.md) 的 MCP 一节和契约
`mcp.jobs`（`dic_describe(section="mcp.jobs")`）。
拒绝一律返回 `ok: false` + `code`/`error`，并尽量带 `field`、`expected`、`hint`、`retryable`。
`dic_query_manifest` 的 `max_bytes`（默认 16 KB）是硬上限，整个回复都算在内（大数组折叠为计数和统计，
并列出 status 不是 ok 的条目，可按 `fields` 取片段）；
`dic_aggregate_cod` 测线不在数据里时报 `no_cod_rows` 并列出可用行号，输出只能是新的 `.csv`。
每次调用写 job root 下 `mcp_audit.log` 的 JSONL 审计记录；middleware 也记录被 MCP
schema 直接拒绝、未进入工具函数的畸形调用。

---

## 使用示例

### 基本用法

```bash
/home/chonglang/opencorr-gpu-mvp-build/run_dic.sh \
  --photo-dir /media/disk/PHOTO/ \
  --output-dir /media/disk/RESULTS/ \
  --interface-y 131 --gauge 16 \
  --cal 0.5641
```

### 自定义 subset 和 ROI

```bash
/home/chonglang/opencorr-gpu-mvp-build/run_dic.sh \
  --photo-dir /media/disk/PHOTO/ \
  --output-dir /media/disk/RESULTS/ \
  --interface-y 133 \
  --cal 0.5641 \
  --subset 21 \
  --roi-cols 50,1000
```

这里不传 `--gauge`（默认 35，测线 116/150）：`--subset 21` 是 radius 10，测线离界面至少要
11 px，界面 ±8 px 的惯例在这个 subset 下会让子区跨界面。

### 自定义测线位置

```bash
/home/chonglang/opencorr-gpu-mvp-build/run_dic.sh \
  --photo-dir /media/disk/PHOTO/ \
  --output-dir /media/disk/RESULTS/ \
  --line-y 110,160 \
  --cal 0.5641
```

### 12-bit 数据（highspeed2 左移存储）

```bash
/home/chonglang/opencorr-gpu-mvp-build/run_dic.sh \
  --image-dir "/data/highspeed2/5w-jingzhi1_iX 726样机_1_.../" \
  --output-dir /tmp/hs2_cod/ \
  --line-y 96,128 \
  --cal 1.0 \
  --input-depth 12 --input-align left
```

- highspeed2 TIFF 标称 16-bit 实为 **12-bit 左移 4 位**，必须显式 `12 left`（`auto` 会把左移数据误判成 16；u/v/zncc 因 ZNCC 线性不变性不受影响，但灰度值与 manifest 元数据口径是错的）。
- 带空格的原生路径可直接用（2026-07-09 起；v10 及更早 binary 会静默错切）。
- hs2 样机数据未做空间标定：`--cal 1.0` 时输出 `cod_um` 列数值 = px×1000，仅作相对量用。测线 96,128 与 [`docs/lab/simulacrum.md`](../docs/lab/simulacrum.md) 的 canonical 命令一致。

### 只跑部分事件、逐事件界面行（campaign）

```bash
/home/chonglang/opencorr-gpu-mvp-build/run_dic.sh \
  --photo-dir /data/lab-jobs/bimaterial/config1-bimaterial/30kN/photos \
  --output-dir "$OUT" \
  --subset 15 --step 1 \
  --interface-y 131 --event-interface-y 'event2[1-5]=133' --gauge 16 \
  --cal 0.5641 --roi-cols 72,958 \
  --strict --json-summary "$OUT/summary.json"
```

event01–20 用 131（测线 123/139），event21–25 用 133（125/141）。只跑后段：`--events 'event2[1-5]' --interface-y 133`。
pattern 要加引号，免得 shell 先展开 `*`、`[...]`。先 `--dry-run` 看每个事件的 `geometry.interface_y` /
`interface_y_source` 和顶层 `event_selection`。

---

## 如何确定参数

推荐使用 **DIC Setup Tool**（Windows 桌面工具，不在本仓库里）交互式确定以下参数：

### interface-y（断层面位置）

在 DIC Setup Tool 中点击 **Interface** 模式，点击断层面（通常是明显的水平暗线）即可获取 y 像素坐标。

也可手动：用图片查看器打开照片，读取断层面的 y 像素坐标。

### cal（像素标定值）

在 DIC Setup Tool 中点击 **Calibration** 模式，点击试样上两条已知间距的划线，输入实际物理距离(mm)，工具自动计算 `cal = 实际距离 / 像素距离`。

也可手动：用已知长度的标尺拍照，计算 `实际长度(mm) / 像素长度(px)`。

### roi-cols（感兴趣区域）

在 DIC Setup Tool 中点击 **ROI** 模式，点击左右边界即可获取 `x1,x2`。此参数可选。

### gauge（测线间距）

代码默认 35 像素（测线离界面 ±17 px）。实验室 config1 30 kN 的惯例是 `--gauge 16`（界面
±8 px，2026-10-05 定；界面行按段：event01–20 为 131、event21–25 为 133，一次 campaign 用
`--event-interface-y` 给后段），要显式传；其他数据集按 [`docs/lab/simulacrum.md`](../docs/lab/simulacrum.md)
登记的参数，没登记的先问。

测线间距的选择原则：
- 测线必须在断层面两侧的**均匀变形区**内：子区（radius r）不能跨进界面过渡带，即测线离界面
  至少 r + 1 px（30 kN/event10、`--subset 15` 时过渡带是 y=124…138，最近的干净测线是 123/139）
- 间距 > 断层过程区宽度 + subset 大小
- 用 `--cod-method fillboundary` 时，各线 `u` 外推到中点，单点噪声随外推距离放大：25 组实验室
  static-ref 上，±8 px 时噪声底约为 naive 的 2.0 倍，±17.5 px 时约 3.5 倍（2026-10-05）；默认 naive 不外推
- 间距过大会降低 COD 的空间分辨率

---

## 性能参考

历史记录（2026-05-03，v9_finalconv_fix，Simulacrum 单 RTX 6000 Ada；当时只计 binary 本身）。
2026-10-03 在生产 binary `638ad18` 上、主机空闲时实测端到端（含 Python 启动、聚合、写 CSV
和 `--json-summary` 统计）：event10 COD 800 帧 4.7–6.8 s，全场 CSV 30 帧 4.2 s，全场 raw
90 帧 1.4 s。dry-run 的 `estimates` 按这组实测分模式标定（`calibration` 字段写明日期和
binary），给出 `runtime_s` 和 `runtime_s_range`；冷缓存、主机繁忙时可能慢 3 倍以上，设超时
用区间上端。下表是更早的记录：

| 数据规模 | 耗时 | 备注 |
|----------|------|------|
| 单事件 1201 帧（1036×186 图片）| ~1.4 秒 | 多帧 + skip-fftcc，单 line（L0,L1） |
| 10 个事件 | ~15–20 秒 | 含 Python 启动 + COD 聚合 |
| 83 个事件（10/20/30 kN，旧批处理脚本剔除 2 个之后） | ~2–3 分钟 | 同上；含 40 kN 为 115 个事件。数据目录里实有 85 个，现行 campaign 不剔除、会处理全部 85 个（见 `docs/lab/simulacrum.md` 坑 9） |

注：
- pipeline 默认走「多帧 + skip-fftcc」路径——参考帧只算一次 FFTCC，后续帧用上一帧位移做 ICGN warm start，CUDA context 全程驻留。
- vs CPU `opencorr_cli`（同一事件 1201 帧）|Δu| 中位 1.85e-4 px ≈ 0.10 μm（≈ subset-15 静态噪声底 σ_u≈0.0035 px 的 **~5%**；2026-07-08 与 paper §4.2/Table 2 统一重锚，旧口径"12.6 μm 的 0.7%"分母来自区分度弱的 zero-strain 试验，弃用），99.8% (x,y) 重合。
- 旧 v5_csv_opt 的「~3 秒/800 帧」记录已作废，那时的 ICGN 收敛 bug 让大量 POI 早退，runtime 不可比。

---

## 常见问题

**Q: 报错 "GPU binary 不存在"**
A: 确认 Simulacrum 上 `/home/chonglang/opencorr-gpu-mvp-build/opencorr_cli_cuda_mvp` 存在且可执行。

**Q: 报错 "No module named 'numpy'"**
A: 用 wrapper 脚本 `run_dic.sh` 运行（自动激活 venv）。或手动：`source /home/chonglang/opencorr-gpu-mvp-build/.venv/bin/activate` 后再运行。

**Q: 输出 CSV 只有表头、行数很少，或 `cod_um` 全是 0**
A: 非有限的 COD 点会被省略，不会写成 NaN。只有表头通常是测线落在图外或 binary margin 内
（binary 跳过该线，摘要里是 `binary_warning` + `cod_line_missing`）或 `--roi-cols` 写反了，
摘要记 `cod_empty`、事件状态 `fail`、控制台打 ❌（不加 `--strict` 时单事件退出码仍是 0）；
全 0 通常是 `--gauge` 0/1 让两条线重合，记 `cod_all_zero`。先用同样参数跑 `--dry-run`，
它会报这些错误；加 `--strict` 则正式运行前就拒绝。

**Q: 想看中间的位移场数据**
A: 加 `--keep-frames`，中间文件保留在 `_frames_<事件名>/` 目录中，每帧一个 CSV（当前 GPU 为 `x,y,u,v,zncc,uy,vy,ux,vx`）。

**Q: 中间 CSV 里 `zncc < 0` 的 `u,v` 能用吗？**
A: 不能当有效位移用。`zncc < 0` 是状态/失败行（例如边界、未收敛或 NaN 保护），`u,v` 只是初值、上一帧种子或失败时最后迭代状态。COD/位移聚合必须过滤 `zncc >= 0`，必要时再加更严格的 `zncc` 质量阈值。

**Q: 照片目录或文件名带空格？**
A: 2026-07-09 起支持（tar-list 按最后一段空白切分，tab 优先）。v10 及更早 binary 的旧 parser 会静默错切路径——遇到"invalid tar"或输出错位先查 binary 版本。

**Q: 跑 supp 数据时 60–90% POI 输出 `zncc=-5`？**
A: 那是 v5_csv_opt（及更早）的 ICGN 收敛 bug——`final_conv` 是 per-thread 寄存器，非 tid 0 线程读到自己的零值副本后在 iter 1 后误 break。已在 v9_finalconv_fix（2026-05-03）修复；当前默认 `opencorr_cli_cuda_mvp` 也包含该修复。

## CLI 失败与续跑（2B）

CLI 失败时，pipeline 保留 frame 目录和 tar-list，不再删除已有结果；binary 的完整
stderr 在 `<输出目录>/<帧目录名>.binary.log`，摘要的 `gpu_cli_failed` 带退出码和第一条
`Error:` 行，`remediation` 给出一句处置建议（例如 resume rejected → 换新的 `--output-dir`）。
多帧 CLI 在 `.opencorr-state/` 记录输入、参数、binary 身份及输出完整性；
必须把隐藏目录与 frame 文件一起保存。重试使用相同 binary、输入、参数，
目标列表只能保持不变或在原前缀末尾追加。未完成任务会从第一帧重放计算，
验证并保留原文件，只补缺失文件；不是从最后一帧直接接算。
损坏文件、陌生旧输出和不匹配配置会被拒绝，需保留原目录另起新目录重算。
成功后的 `--keep-frames` 清理规则不变。完整契约见
[CLI output integrity](../docs/output-integrity.md)。
