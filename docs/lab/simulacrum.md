# Simulacrum 实验室部署手册（lab-specific）

> **实验室专用。** 本文件只描述原作者实验室那台机器（Simulacrum）上的部署、数据集
> 和 agent 操作约定，**不是 FastICGN 的通用用法**；主机名、绝对路径和数据目录只对那套
> 部署有效。通用入口是仓库根 [`README.md`](../../README.md)；结果可信度检查见
> [`docs/quality-checks.md`](../quality-checks.md)。
>
> 文中带日期的实测值（hash、行数、耗时）是当天的记录，不是验收标准；现场以
> `--version` 和本次运行的输出为准。

## 部署

- **源码仓库**：`/home/chonglang/opencorr-gpu-mvp-src`（git，`master`）。
- **生产目录**：`/home/chonglang/opencorr-gpu-mvp-build/`
  - `opencorr_cli_cuda_mvp`：生产 binary，不在 git 工作树里。2026-10-03 22:56 上产
    （master `638ad18`，整帧 fallback 逐点合并），2026-10-04 实测 `--version` 为
    `git_hash=638ad1845d01`、`build_date=2026-10-03T14:54:01Z`，sha256
    `547434faaa00e569f81a555e6e291b9d8f98ddae85a9a622ba6c16c1c1af73b0`。产出它的耐久构建树是
    `/home/chonglang/opencorr-gpu-mvp-build-20261003-fallback/`。
  - 上一版的回滚备份是同目录的
    `opencorr_cli_cuda_mvp.pre-fallback-merge-20261003T225622+0800-1ca731b37472.bak`
    （`git_hash=1ca731b37472`，sha256 `862dc198…`）。
  - 同目录的其他 `*.bak`、`opencorr_cli_cuda_mvp_v1`…`_v11_hibit`、`*_working` 等是历史
    binary，只用于回滚和对照。除 `*.pre-stage2c-*`、`*.pre-stage2d-*`、`*.pre-stage2e-*`、
    `*.pre-fallback-merge-*` 四个备份外，它们都早于输出 ledger（`.opencorr-state/`），不认它，
    **不得**用来续跑已有输出目录。
  - `run_dic.sh`、`dic_pipeline.py`、`DIC_PIPELINE_GUIDE.md` 是指向源码
    仓库 master 工作树的 symlink：**master 上 `pipeline/` 的任何改动立即进生产**，
    开发一律在分支 / worktree 上做。
  - `.venv/`：生产 Python 3.12 venv（2026-10-03：numpy 2.4.4、Pillow 12.2.0、
    fastmcp 3.4.4）。只读使用，不要往里装包——`pipeline/requirements.txt` 写的是
    `Pillow<12`，照装会把 Pillow 降级。
- **`~/.config/fasticgn/config.toml`**（2026-09-23 写入）：
  `[paths] gpu_bin = "/home/chonglang/opencorr-gpu-mvp-build/opencorr_cli_cuda_mvp"`。
  它让本机上的 pipeline 和 MCP 默认用生产 binary（dry-run 的
  `resolved_params.gpu_bin_source` 显示 `config:/home/chonglang/.config/fasticgn/config.toml`），
  包括从别的 checkout 跑 pipeline 的时候——要用某个构建就显式传 `--gpu-bin`（相对路径
  会按当前目录转成绝对路径）。这时 dry-run、控制台和摘要都会带一条 info 级的
  `gpu_bin_outside_checkout`，属正常情况、不影响 verdict。pipeline 单测已与它隔离，本机
  真实 HOME 下也全过。
- **PATH**：`run_dic.sh` 只在登录 shell 的 PATH 里（`~/.local/bin/run_dic.sh` symlink）。
  `ssh simulacrum '<命令>'` 是非登录 shell，找不到它：**一律写绝对路径**
  `/home/chonglang/opencorr-gpu-mvp-build/run_dic.sh`。
- CPU 原版 `/home/chonglang/opencorr` 与 `/data` 只读。GPU：2 块 RTX 6000 Ada（CUDA 13.2）。

## 两种模式

- **全场模式**（不给 `--interface-y` / `--line-y`）：每个目标帧的整幅位移场写到
  `$OUT/<事件名>/frame_NNNNNN.csv`（`--output-format raw` 则为 `.bin` + manifest），
  不做聚合，不需要 `--cal`。
- **COD 模式**（给了其中之一）：binary 只算两条测线，pipeline 聚合成
  `$OUT/cod_result.csv`（单事件）或 `$OUT/<事件名>.csv`（campaign），`--cal` 必填。
  `cod_um` 是**滑移量**：界面两侧的相对位移，即两条测线水平位移 `u` 之差（L0 侧 − L1 侧），不是张开量
  （2026-10-05 定；列名、`--cod-method`、"COD"都是沿用的名字）；正负号只由公式决定，`u_L0 > u_L1` 为正。
  列定义（`x_mm` 原点、COD 公式与正负号、帧号）见契约 `outputs.cod_csv`。

下面四条 canonical 命令都是 COD 模式，都带 `--strict`：参数或输出目录有问题时正式运行前
就以 exit 2 拒绝（什么都不写、不占 GPU），任何事件失败（含 COD 为空、全零）时以 exit 3
结束，所以退出码 0 就是"全部事件 ok"。不加 `--strict` 的旧写法照样能跑，但 campaign 有
事件失败时仍 exit 0。

## Canonical commands

每次用新的输出目录，避免混入旧结果。带日期的实测值见每条命令下方；先读结果判读见下文
"如何验证一次运行"。

### 8-bit 单事件

```bash
OUT="$HOME/dic_jobs/fasticgn-8bit-$(date +%Y%m%d-%H%M%S)"
/home/chonglang/opencorr-gpu-mvp-build/run_dic.sh \
  --image-dir /data/lab-jobs/bimaterial/config1-bimaterial/30kN/photos/event10 \
  --output-dir "$OUT" \
  --subset 15 --step 1 \
  --interface-y 131 --gauge 16 --cal 0.5641 --roi-cols 72,958 \
  --strict --json-summary "$OUT/summary.json"
```

**测线惯例：界面 ±8 px（2026-10-05 定），界面行按段登记。** radius 7 时受界面影响的是界面上下
各 7 行（30 kN/event10：y = 124…138，中心 131），不跨界面的最近测线是界面 ±8 px。30 kN 的界面行
分两段（2026-10-05 逐事件实测，按段登记）：

- **event01–20：界面 131**（实测 131–131.5）→ `--interface-y 131 --gauge 16`，测线 123 / 139（上面的命令）。
- **event21–25：界面 133**（实测 133–133.5）→ `--interface-y 133 --gauge 16`，测线 125 / 141。
  event24 末段没有可分辨的破裂，界面行量不出，按所在段的 133 处理。

分段的原因：整个试样在 event20 → 21 之间（2026-03-25 20:59:34–21:01:00）相对相机画面平移了一次，
上下两板的散斑一起向下约 2 px、向右约 3 px，界面随之从 131 移到 133–133.5；不是 ±8 px 惯例的问题。
用错段的界面行，有一条测线会落进过渡带：event21 / 22 / 23 / 25 按 131 跑，末 10 帧的 naive COD
是平台跳变的 93–94%，fill-boundary 只有 77–79%（按 133 为 103–105%）；反过来 event10 按 133 跑，
fill-boundary 掉到 81%。

**COD 口径：默认 naive（2026-10-05 起）。** 上面的命令不带 `--cod-method`，算的是两条测线 `u` 的直接差分。
`--cod-method fillboundary`（2026-07-09 至 2026-10-05 的默认）可选，见下文"可选：fill-boundary 口径"。
改回的原因：测线干净时 fill-boundary 只比 naive 高几个百分点，但单点噪声随外推距离放大——同一批
25 组 static-ref、现生产 binary 实测，123/139（外推 8 px）的 COD 噪声底 naive 2.52 μm、fill-boundary
5.06 μm，约 2.0 倍；116/151（外推 17.5 px）为 2.77 / 9.65 μm，约 3.5 倍——而且如上所述，界面行给错时
偏得多得多。三个坑：

- **不传 `--gauge` 时默认仍是 35**（`half = 35 // 2 = 17`），`--interface-y 133` 得到测线
  116 / 150、外推目标 133——这是 2026-10-05 以前的 canonical 几何，离界面远、单点噪声大。
- **`--interface-y` 要和事件所在的段对上。** dry-run 和 campaign 都不知道界面在哪：`131 + 16`
  用在 event21–25、`133 + 16` 用在 event01–20 都照样全 `ok`、不报警。一次 campaign 跑两段用
  `--event-interface-y`（2026-10-05 起），见下文 campaign 一节。
- **换一批数据先量界面行。** 别的载荷、别的录制，甚至同一载荷的后段事件，界面行都可能变（20 kN
  在 event26 → 27 之间同样平移过一次）；用一对测线前先按下文"数据集惯例参数"的量法确认它们在
  过渡带外。

2026-10-05 用默认改为 naive 之后的 pipeline（分支 `serf/cod-default-naive-20261005`）和
生产 binary（`git_hash=638ad1845d01`，`--gpu-bin` 显式给出）按上面命令实跑：exit 0；800 个目标帧，
测线 123/139；`events[].cod_mode` 为 `naive`，`geometry.extrapolation_target_y` /
`extrapolation_px` 为 `null`；`cod_result.csv` 709,600 行（800 帧 × ROI 内 887 个 x 点，SHA-256
`6a788fe9…fc8f`，与改动前显式 `--cod-method naive` 的输出逐字节相同）；`verdict` 为 `ok`：事件级只有
一条 info 级的 `input_8bit_max_255`（两条测线带 0.03% / 0.81% 像素为 255，低于默认阈值 5%；整帧
1.30%）；valid_ratio 1.0，zncc_median 0.998；端到端约 6 s。`cod_stats`：COD 最大值 340.3 μm 在
frame 801、x=120 px，最小值 −23.9 μm，|COD| 的 p95 为 311.0 μm；naive 口径下 `naive_at_max_um` /
`naive_at_min_um` 为 `null`（最大、最小值本身就是 naive 值）。经生产 wrapper、binary 走 config 解析时，
顶层还有一条 info 级的 `gpu_bin_outside_checkout`（属正常，顶层 `warnings` 仍是 `[]`）。

2026-10-05 以前（默认 fill-boundary 时）这条命令的输出是 `9188d1e8…61c5`（COD 最大值 368.9 μm 在
frame 625、x=261 px，同一点 naive 336.6 μm，p95 327.3 μm）；现在要显式 `--cod-method fillboundary`
才得到它，见下文。默认 gauge（`--interface-y 133`，测线 116/150）下的 fill-boundary 历史记录：
2026-10-03/04 实跑、2026-10-05 同法复跑一致——709,600 行，SHA-256 `89d3b67f…4b58`；两条带
0.02% / 1.73%；COD 最大值 432.2 μm（frame 761、x=758 px，同一点 naive 301.9 μm），|COD| p95
339.8 μm（p95 为 10-05 复跑读数）。

### 12-bit highspeed2 单事件

```bash
OUT="$HOME/dic_jobs/fasticgn-12bit-$(date +%Y%m%d-%H%M%S)"
/home/chonglang/opencorr-gpu-mvp-build/run_dic.sh \
  --image-dir "/data/highspeed2/5w-jingzhi1_iX 726样机_1_D260708_T185341" \
  --output-dir "$OUT" \
  --subset 15 --step 1 \
  --line-y 96,128 --cal 1.0 \
  --input-depth 12 --input-align left \
  --strict --json-summary "$OUT/summary.json"
```

2026-10-04 同上方式实跑（与 10-03 记录一致）：exit 0；401 张 TIFF（带空格和中文的原生路径可直接
用）、400 个目标帧；812,000 行（400 × 2030）；`verdict` 为 `ok`；参考帧实际按 12-bit、left 读入；
valid_ratio 1.0，zncc_median 0.9995；端到端约 5 s。该采集未做空间标定：`--cal 1.0` 时
`cod_um / 1000` 是像素量，不能报成 μm。

### campaign（30 kN 两段一条命令）

一条命令跑完 30 kN 的 25 个事件：`--interface-y 131` 是默认界面行（event01–20），
`--event-interface-y 'event2[1-5]=133'` 把 event21–25 改成 133（event24 没有可分辨的破裂，按所在段取 133）。
`--gauge 16` 对所有事件都一样，所以测线是 123/139 与 125/141。

```bash
OUT="$HOME/dic_jobs/fasticgn-campaign-$(date +%Y%m%d-%H%M%S)"
/home/chonglang/opencorr-gpu-mvp-build/run_dic.sh \
  --photo-dir /data/lab-jobs/bimaterial/config1-bimaterial/30kN/photos \
  --output-dir "$OUT" \
  --subset 15 --step 1 \
  --interface-y 131 --event-interface-y 'event2[1-5]=133' --gauge 16 \
  --cal 0.5641 --roi-cols 72,958 \
  --strict --json-summary "$OUT/summary.json"
echo "campaign exit $?"
```

输出是 `$OUT/eventNN.csv`。命令不带 `--cod-method`，是 naive（2026-10-05 起的默认）；要 fill-boundary 就加
`--cod-method fillboundary`。pattern 要加引号，免得 shell 先展开。先 `--dry-run` 看每个事件的
`events[].geometry.interface_y` / `interface_y_source` / `line_ys` 和顶层 `event_selection`；summary 里是同样的字段。
只跑一部分事件用 `--events`（逗号分隔的事件名或 glob），例如只跑后段：
`--events 'event2[1-5]' --interface-y 133`。写了不存在的事件名、pattern 一个都没匹配到、某条
`--event-interface-y` 不匹配任何选中的事件、一个事件命中两个不同的界面行、与 `--line-y` 同给，都在读图之前
报错（加不加 `--strict` 都是 exit 2），不跑 binary、不写输出；给了 `--json-summary` 时只写一份失败摘要
（`verdict` `fail`，与 `--strict` 预检拒绝时相同，所以摘要放在输出目录里时这个目录会被建出来）。段的划分仍要人按"数据集惯例参数"维护：pipeline 不知道界面在哪，
`131` 写给 event21–25 也照样全 `ok`。

2026-10-05 用分支 `serf/event-select-20261005` 的 pipeline 和生产 binary（`git_hash=638ad1845d01`，`--gpu-bin`
显式给出）按上面命令实跑：dry-run exit 0（估计 114 s，区间 57–457 s），每个事件的 `binary_command` 与实跑的 argv
相同；实跑 exit 0、约 141 s，`verdict` `ok`，25/25 事件 `ok`。`geometry`：event01–20 界面 131、测线 123/139、
来源 `interface_y`；event21–25 界面 133、测线 125/141、来源 `event_interface_y`（`event2[1-5]=133`）。每个事件
709,600 行、valid_ratio 1.0，只有 info 级的 `input_8bit_max_255`。25 个 CSV 与同日用生产入口按旧做法（符号链接
分段、两段各跑一次）跑出的逐字节相同。naive 读数：event10 最大值 340.3 μm、|COD| p95 311.0 μm（与上文单事件相同），
event22 最大值 320.8 μm、p95 296.5 μm。`--events event22` 加 `--cod-method fillboundary`：SHA-256 `c53faf59…76ee`，
与下文 fill-boundary 一节的 event22 相同。只跑后段（`--events 'event2[1-5]' --interface-y 133`）：5/5 `ok`、约 30 s，
CSV 与上面的 event21–25 逐字节相同，`event_selection` 记选中 5 个、筛掉 20 个。
带 `--strict` 时任何事件失败都以 exit 3 结束；不带时 campaign 有事件失败仍 exit 0，必须逐个看 `events[].status`。

2026-10-05 以前没有这两个参数，按段 campaign 是为每段建一个只含该段事件符号链接的目录、分段各跑一次；
那样跑出的 CSV 与上面一条命令的逐字节相同（见上）。上午默认还是 fill-boundary 时，按段 campaign 实跑过一次
（生产 binary `git_hash=638ad1845d01`）：131 段 20/20、133 段 5/5 `ok`，每事件 709,600 行；131 段的 20 个 CSV
与同日不分段（全部按 131）的 campaign 逐字节相同。

### 可选：fill-boundary 口径

```bash
OUT="$HOME/dic_jobs/fasticgn-fillboundary-$(date +%Y%m%d-%H%M%S)"
/home/chonglang/opencorr-gpu-mvp-build/run_dic.sh \
  --image-dir /data/lab-jobs/bimaterial/config1-bimaterial/30kN/photos/event10 \
  --output-dir "$OUT" \
  --subset 15 --step 1 \
  --interface-y 131 --gauge 16 --cal 0.5641 --roi-cols 72,958 \
  --cod-method fillboundary \
  --strict --json-summary "$OUT/summary.json"
```

默认是 `naive`（2026-10-05 起；2026-07-09 以前也是）。`fillboundary` 沿各测线的 `uy` 把 `u` 外推到两线
中点再相减，2026-07-09 至 2026-10-05 是默认：复现这段时间里没写 `--cod-method` 的结果要显式选它，
测线也要与旧结果一致（见下文"数据集惯例参数"里的历史产物）；拿不准时看那次 summary 的
`events[].cod_mode`。用它报极值时并列同一点的 naive 值（`cod_stats.naive_at_max_um`，只在
fillboundary 下有值）和 p95。

2026-10-05 用默认改为 naive 之后的 pipeline 和生产 binary（`git_hash=638ad1845d01`）实跑：
exit 0，`verdict` 为 `ok`，`cod_mode` 为 `fillboundary`，测线 123/139、外推目标 131、`extrapolation_px`
[8, −8]；709,600 行，SHA-256 `9188d1e8…61c5`（与改动前不带 `--cod-method` 的输出逐字节相同）；COD
最大值 368.9 μm（frame 625、x=261 px，同一点 naive 336.6 μm），|COD| p95 327.3 μm。event22 按段用
`--interface-y 133 --gauge 16`（测线 125/141）：fill-boundary SHA-256 `c53faf59…76ee`（与 2026-10-05
上午按段 campaign 的 `event22.csv` 逐字节相同），最大值 348.2 μm（同一点 naive 307.8 μm），p95
306.7 μm；同一事件不带 `--cod-method`（naive）SHA-256 `b18dc5da…7108`，最大值 320.8 μm，p95 296.5 μm。
都是 exit 0、`ok`、valid_ratio 1.0。

## 数据集惯例参数

只登记有证据的条目；没有登记的一律先问 Boss，不要从别的事件推断。

| 数据集 | 惯例参数 | 证据 |
| --- | --- | --- |
| config1-bimaterial 30 kN（`/data/lab-jobs/bimaterial/config1-bimaterial/30kN/photos/event*`） | 界面行按段：**event01–20** `--subset 15 --step 1 --interface-y 131 --gauge 16 --cal 0.5641 --roi-cols 72,958`（测线 123/139，外推目标 y=131）；**event21–25** 同上但 `--interface-y 133`（测线 125/141，外推目标 133）。都是界面 ±8 px；`--gauge 16` 必须显式写，不写就是默认 35 | canonical 命令；界面行逐事件实测（2026-10-05）：event01–20 为 131–131.5，event21 为 133、event22 / 23 / 25 为 133.5，event24 没有可分辨的破裂、按所在段取 133；分段原因是整个试样在 event20 → 21 之间相对相机平移约 2 px（两板散斑互相关，2026-10-05）；按段 campaign 2026-10-05 实跑（一条命令：`--interface-y 131 --event-interface-y 'event2[1-5]=133'`，见 campaign 一节）。2026-10-05 以前登记的是 `--interface-y 133`、默认 gauge（测线 116/150，外推目标 133） |
| highspeed2（`/data/highspeed2/`，iX 726 样机） | `--input-depth 12 --input-align left --line-y 96,128 --cal 1.0` | canonical 命令（`5w-jingzhi1_…T185341` 采集）；未做空间标定 |
| config1-bimaterial 10 kN（`…/config1-bimaterial/10kN/photos/event*`，30 个事件） | **未登记，需问 Boss**。实测界面行（2026-10-05）：众数 131，范围 130.5–131（29 个事件）；event05 无法判定 | 逐事件实测（量法见表下）；按 30 kN 的 123/139 算 29/29 干净 |
| config1-bimaterial 20 kN（`…/20kN/photos/event*`，30 个事件） | **未登记，需问 Boss**（cal、ROI 没有登记，没有 canonical 命令）。实测界面行按段（2026-10-05）：event01–26 约 131（130.5–131.5），event27–30 约 133（133–133.5） | 逐事件实测；分段原因与 30 kN 相同：整个试样在 event26 → 27 之间（2026-03-25 18:31:01–18:32:42）相对相机平移约 2 px（两板散斑互相关）。参数登记后可以与 30 kN 同样写成一条 campaign（`--interface-y 131 --event-interface-y 'event2[7-9]=133' --event-interface-y 'event30=133'` 这类写法），这里不给命令。按段取界面行时 event27–30 的 125/141 在过渡带外；event18 / 22 的 139 / 123 擦着过渡带边（偏离为峰值的约 3%） |
| config1-bimaterial 40 kN（`…/40kN/run{1,2,3}/photos/event*`，13 + 13 + 6 个事件） | **未登记，需问 Boss**。实测界面行（2026-10-05）：run1 众数 130.5（130.5–131，event04 无法判定）；run2 众数 130（129–130.5）；run3 众数 130（130–130.5） | 逐事件实测；按 123/139 算 run1 9/12、run2 5/13、run3 6/6 干净（不干净的都是 123 伸进 1 行） |
| config2-symmetric lowrough-20260918（`/data/lab-jobs/bimaterial/config2-symmetric/lowrough-20260918/{10,20,30}kN/photos/event*`，30 + 30 + 19 个事件） | **未登记，需问 Boss**。实测界面行（2026-10-05）：10 kN 众数 128.5（128–128.5）；20 kN 众数 129（128.5–130，event01 无法判定）；30 kN 129.5 与 130.5 各 6 个（128–130.5，event04 无法判定） | 逐事件实测。数据目录自带 `dic_processing_config.json`（2026-09-23 在各载荷 event01 首帧手工选点）写 `gauge_px` 16、`interface_y` 130 / 131 / 131、`processing_roi_cols` 111–931 / 106–930 / 107–933、`cal_mm_per_px` ≈ 0.6098 / 0.6067 / 0.6048；按它的测线（10 kN 122/138，20 / 30 kN 123/139）实测干净的事件只有 0/30、1/29、6/18，都是上测线伸进过渡带 1–3 行 |
| 其他配置 | **未登记，需问 Boss** | — |

"实测界面行"只说明位移在哪一行跳变，不是参数登记：测线、cal、ROI 仍以登记为准，未登记的照旧先问。
量法（2026-10-05，生产 binary `git_hash=638ad1845d01`）：每个事件参考帧取 pipeline 选的第一张图、目标帧
取最后 10 帧，直调 binary 全场计算（radius 7、grid 1、`--conv-threshold 0.001`、`--no-skip-fftcc`、
`--output-format raw`、`--input-depth 8`；这些数据都是 1036×186、8-bit 的 RGB JPEG）；只用 `zncc >= 0`、
x 72–958 的点，逐行取 `u`、`uy` 的中位数再对 10 帧平均；过渡带 = 峰值行两侧 |`uy` − 背景| 超过
max(峰值的 2.5%, 3 倍背景噪声) 的连续行，界面行 = 过渡带中心（可为 .5），带外紧邻的行是"干净行"；
峰值不到噪声的 10 倍或跳变小于 0.01 px 记为无法判定。30 kN/event10 得过渡带 124…138、界面 131，与
最初按 20 帧量的结果相同。阈值取 1% 或 5% 时界面行最多变 0.5 行，30 kN 的干净 / 不干净判定不变。

`/data/lab-jobs/bimaterial/config1-bimaterial/30kN/dic-sub15-step1/<event>/` 里的历史产物
是另外两套口径，**不能直接与默认输出对比**（2026-10-03 打开 event12 的文件核实）：

- `cod_data.npz`：键 `cod_um, x_px, x_mm, frame_numbers, l0_y, l1_y, gauge_mm,
  col_spacing_mm`；测线 116/151；x 为整条线 1020 个点（没有 ROI 裁剪）。其 `cod_um` 与
  同目录 5 列 frame CSV（`x,y,u,v,zncc`，旧 binary 输出）按 naive 公式
  `(u_L0 − u_L1) × 0.5641 × 1000` 重算的结果一致（差 ≤ 2e-5 μm），即 naive 口径。
- `cod_data_v3.npz`：键另含 `interface_y=131`、`gauge_px=16`，测线 123/139；同样
  验证为 naive 口径。

按 canonical 命令跑出的输出是 naive（2026-10-05 起的默认）、测线 123/139（event21–25 按段为 125/141）、ROI 72–958；
口径与 `cod_data_v3.npz`（naive、123/139）相同，但它的 x 是整条线、不裁 ROI，对比时要先按 ROI 取列。
2026-07-09 至 2026-10-05 的默认输出是 fill-boundary，不能直接与这两个 npz 对比。
不传 `--gauge` 时 pipeline 默认仍是 gauge 35，`--interface-y 133` 得到的是测线 116/150。

## 接口契约与 dry-run

机器可读的单一事实源是 `pipeline/interface_contract.json`；`--describe` 原样输出它：

```bash
/home/chonglang/opencorr-gpu-mvp-build/run_dic.sh --describe | python3 -m json.tool
```

正式运行前，用完全相同的输入和参数先 dry-run。下面的写法保留 wrapper 自己的退出码，
不会被 JSON formatter 掩盖：

```bash
PLAN="$(mktemp "${TMPDIR:-/tmp}/fasticgn-plan.XXXXXX.json")"
set +e
/home/chonglang/opencorr-gpu-mvp-build/run_dic.sh \
  --image-dir "/data/highspeed2/5w-jingzhi1_iX 726样机_1_D260708_T185341" \
  --output-dir "$HOME/dic_jobs/plan-only" \
  --subset 15 --step 1 --line-y 96,128 --cal 1.0 \
  --input-depth 12 --input-align left --strict --dry-run >"$PLAN"
RC=$?
set -e
python3 -m json.tool "$PLAN"
test "$RC" -eq 0
```

dry-run 只读首帧，输出完整 binary argv、测线与几何（`resolved_params.geometry`；每个事件自己的在
`events[].geometry`，campaign 的事件筛选与逐事件界面行在 `event_selection`）、尺寸 /
位深探针和建议、校验错误、输出目录现状（`output_state`），以及耗时 / 磁盘估计；不创建
`--output-dir`、不写 `--json-summary`、不启动 GPU。退出码 0 = 可运行，2 = 有机器可读错误；
`--interface-y 99999` 这类越界输入必须得到 2。输出目录已有旧 COD CSV（`cod_csv_exists`）或
ledger（`frame_dir_has_ledger`）只是 warning；路径是个文件、没有 ledger 的旧帧是 error。
耗时按模式用 2026-10-03 的实测标定（800 帧 COD 估约 5.7 s，`runtime_s_range` 约
2.9–23 s）；设超时用区间上端。它不比较 ledger 里的指纹，也不检查 GPU 号是否存在。

## MCP stdio 入口

需要工具调用时，通过 SSH stdio 连接受限的 FastMCP server。项目级 `.mcp.json`（Claude Code
读当前项目根的这个文件；其他客户端照搬 `command` + `args`）：

```json
{
  "mcpServers": {
    "fasticgn": {
      "type": "stdio",
      "command": "ssh",
      "args": [
        "-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
        "simulacrum",
        "/home/chonglang/opencorr-gpu-mvp-build/.venv/bin/python",
        "/home/chonglang/opencorr-gpu-mvp-src/pipeline/mcp_server.py"
      ]
    }
  }
}
```

2026-10-03 从本地 Mac 按这份 `command` + `args` 原样起 stdio 连接：initialize 与 tools/list
成功，列出 8 个工具，约 1.5 s。等价的一行注册（未单独实测）：
`claude mcp add --scope project fasticgn -- ssh -T -o BatchMode=yes -o ServerAliveInterval=30 simulacrum /home/chonglang/opencorr-gpu-mvp-build/.venv/bin/python /home/chonglang/opencorr-gpu-mvp-src/pipeline/mcp_server.py`。
`-T` 不给协议流分配终端；ssh 默认不转发 `FASTICGN_*` 环境变量，远端以
`~/.config/fasticgn/config.toml` 为准。

- **升级后要重连**：server 进程每次 `dic_describe` 都现读磁盘上的契约，但跑的是它启动时的代码。仓库里的
  MCP 更新之后（合并、换分支），已经开着的会话会"说"新接口、"跑"旧代码：比如 `dic_describe` 说有
  `wait: false` 和 `dic_job_cancel`，它自己却回 `unknown_params` / 没有这个工具（拒绝是干净的，不会跑错东西）。
  合并或升级 MCP 之后，让已开着的会话（包括 LabAgent）重连一次。
- 9 个工具：`dic_describe`、`dic_get_pitfalls`、`dic_probe_images`、`dic_plan`、
  `dic_run_event`、`dic_run_campaign`、`dic_aggregate_cod`、`dic_query_manifest`，以及
  `dic_job_cancel`（第 9 个，2026-10-05 起，取消作业；前 8 个的顺序和默认行为不变）。
  运行工具先走同一份 dry-run 校验，再返回 pipeline 的 `--json-summary` 摘要（另加 `ok`、
  `call_id` 和 `mcp` 块）。`gpu_bin` 固定为本机解析到的 binary（即上面 config 指向的生产
  binary），agent 不能改写。
- **一律 `--strict`**（server 固定，不是参数）：`dic_plan` 与 `dic_run_*` 都带 `--strict` 调
  pipeline，plan 和 run 判定一致；全零 COD 也算事件失败。事件失败时 pipeline exit 3，run 照常
  返回摘要（`verdict` 为 `fail`、`ok` 为 false、`mcp.job.exit_code` 为 3），不是报错；pipeline
  自己的预检拒绝（exit 2，只会在 plan 通过之后输入、binary 或输出目录又变了时出现）返回与 plan
  校验同形的 `plan_invalid`（`errors` 列表，另带 `output_dir`、`job_record`）。`--strict` 下 16-bit
  图不给 `input_depth` 会被拒（`input_depth_truncates_container`，报错里有建议的参数）；campaign 的
  预检全有或全无，一个事件不过整批都不跑。
- **大 campaign 的回复**：超过 10 个事件时，plan 与 run 结果的每个事件只给一行精简记录（名称、状态、
  帧数、关键指标、warning code），`mcp.events_view` 写明怎么取全文（`dic_query_manifest` 读
  `summary.json` 的 `events[i]`），并列出失败事件和每个失败事件的一行原因（`failure_reasons`，相同的
  合并计数）；`mcp.brief` 先点名失败的事件。`dic_plan` 与 `dic_run_*` 的每个回复（含 `plan_invalid`、
  `timeout` 等拒绝）都不超过 48 KiB：放不下的先省尾部和列表尾，失败名单、原因和取全文的方式最后才省；
  省了什么、省了多少列在 `mcp.omitted`（拒绝回复在顶层 `omitted`）。`ok`、`verdict`、`mcp` 排在
  最前。10 个事件以内、没超限的返回字段不变。
- **工具面自带说明**：`dic_plan` / `dic_run_*` 仍是 `{"params": {...}}`，但 tools/list 里
  `params` 的每个键都有类型、单位、枚举、范围、默认值和一句说明（subset 是 SIZE、cal 是
  mm/px、`line_y` 是 [上, 下]、12-bit 左移数据要 `input_depth: "12"` + `input_align: "left"`、
  `max_frames: N` 只算每个事件前 N 个目标帧，试跑用），不必先读契约。initialize 时 server 下发
  instructions（probe → plan → run → 核验 + 常见坑），每个工具有标题、2–4 句说明和
  annotations。`dic_describe` 不带参数时只返回"跑之前要知道"的部分（2026-10-05 起）：分节索引
  （各节名称、字节数、一句说明）加 `mcp` 一节里的 `run_params`、`guardrails`、工具名等，
  2026-10-05 实测 15 327 B（单测卡在 20 480 B 以内；此前带整节 `mcp` 和 `exit_codes`，约 33 KB）；没带的部分列在
  `mcp_view.on_demand`（节名、字节数、一句说明），按需取：`section="mcp.results"`（返回体说明）、
  `"mcp.errors"`、`"mcp.job_record"`、`"mcp.manifest_query"`、`"mcp.known_issues"` 等；
  `section="mcp"` / `"outputs"` / `"exit_codes"` 取一整节，`section="all"`
  取整份契约（2026-10-05 实测 140 039 B，按 3.5–4 字节 / token 估算约 35k–40k token，超过 Claude Code
  默认的单次 MCP 输出上限 25k token）。
- **只支持 COD 模式**：`dic_plan` / `dic_run_*` 必须给 `cal` 和 `interface_y` 或 `line_y`；
  不能跑全场模式，不接受 `output_format`、`image_pattern`、`image_order`，只认
  "文件名以 6 位帧号结尾" 的图像（被拒时报错会直说这两点）。全场或其他命名请用上面的 CLI。
- **campaign 筛选与逐事件界面行**（2026-10-05 起）：`dic_plan` / `dic_run_campaign` 接受
  `events`（字符串数组，每项一个事件名或 glob，等于 CLI 的 `--events`）和 `event_interface_y`
  （`[{"pattern": "event2[1-5]", "interface_y": 133}]`，每项等于一条 `--event-interface-y`，要同时给
  `interface_y`，不能配 `line_y`）。上文 30 kN 两段 campaign 经 MCP 就是 `"interface_y": 131,
  "event_interface_y": [{"pattern": "event2[1-5]", "interface_y": 133}], "gauge": 16`。规则与 CLI 相同：
  形状不对返回 `invalid_events` / `invalid_event_interface_y`（带 `field`），缺 `interface_y` 返回
  `event_interface_y_requires_interface_y`，其余（无匹配、事件不存在、冲突、覆盖落空、配 `line_y`）
  是 plan 的 `errors` / 运行工具的 `plan_invalid`，不写任何输出。返回带 `event_selection`、
  每个事件 `geometry` 的界面行与来源、`mcp.cod_definition.per_event_lines`（按测线分组），`mcp.brief`
  写出各组测线；名单超过 10 个只列前 10 个（`mcp.event_selection_view`）。不给这两个键时 argv 与返回
  字段不变。`dic_run_event` 不发布这两个键。
- **坐标**：`line_y` 两个互异的非负行号，顺序原样保留：COD = u(L0) − u(L1)，逆序会翻转符号
  （结果里有 `line_y_descending` 警告和 `cod_definition`）；`roi_cols` 要 0 ≤ x_min ≤ x_max；
  没有负数或 `-1` 表示"全宽"的写法。`gpu_device` 按本机可见 GPU 数（2 块：0–1）校验。
- 读白名单：`/data`、`/media`、仓库根 `/home/chonglang/opencorr-gpu-mvp-src`、
  `/home/chonglang/DIC*` 和 job root；写只限 job root 的子目录。job root 本机是 `~/dic_jobs`
  （`FASTICGN_JOB_ROOT` 或配置文件 `[paths].job_root` 可改，plan 的 `mcp.job_root` 显示实际值）。
  symlink 越界、路径里的控制字符和非 UTF-8 的事件目录名、图片名都会被拒绝；`photo_dir` 顶层的每个
  名字都查控制字符（杂项文件也查，与 638ad18 相同），杂项文件的名字不要求 UTF-8（如 GBK 名的说明文件
  照常接受）。已有输出默认拒绝；`overwrite=true` 也只允许空目录。
- **返回与报错**：拒绝时 `ok: false` + `code` / `error`，并尽量带 `field`、`got`、`expected`、
  `hint`、`retryable`；`gpu_busy` / `output_busy` 带占用者（输出目录、开始时间、其会话是否已
  关闭）。run 结果的 `mcp` 块有作业状态、`cod_definition` 和一行 `brief`（verdict、实际测线
  行号、COD 行数与峰值、每条 warning 带级别，如 `input_8bit_max_255(info)`），失败时还有
  stdout/stderr 尾部和 `error_lines`（binary 的 `Error:` 行在这里；完整 stderr 在
  `events[].binary_stderr_log`）。摘要里的 `cod_csv` / `field_dir` 在磁盘上不存在时（不保留帧时
  帧目录总是已删，`field_dir_exists` 为 false），MCP 返回里置 null 并列进 `mcp.nulled_paths`。
- **作业与锁**：run 工具在输出目录写 `.mcp-job.json`（状态、pid、退出码）和
  `.mcp-pipeline.*.log`；GPU / 输出锁由作业进程持有。客户端关会话（关 stdin → 约 2 s →
  SIGTERM）后作业照常跑完、锁一直占着，新会话会得到 `gpu_busy`；作业结束锁即释放。
  结果事后用 `dic_query_manifest(output_dir)` 看（`.mcp-job.json` 也在里面）。超时由作业进程
  自己执行，超时或失败的输出目录不能再用。pipeline 单独被杀时，作业进程先清掉留下的 binary 再放锁；
  作业进程本身被杀时，server 给这个作业另起一个作业进程接着盯（2026-10-05 起；记录里多出
  `runner_lost_at`、`runner_replaced_at`，超时和进度照旧），起不来才由 server 自己等；作业结束记为
  `runner_lost`（退出码未知）或 `job_lost`（什么都没留下），此后 server 再退出也不影响收尾。
  `output_dir` 不能放在另一个 run 目录或 `_frames_*` 里（`nested_output_dir`）。
- **长作业：不等待、看进度、取消**（2026-10-05 起）。run 工具默认仍然阻塞到作业结束、返回摘要，
  与以前完全一样。在 `params` 里加控制键 `"wait": false`，工具在作业起来后立刻返回一张"作业单"：
  `job_id`（就是这次调用的 `call_id`）、`output_dir`、`job_record`（`.mcp-job.json` 的路径）、
  `summary_json`、`cancel_token`，以及 `mcp.poll` / `mcp.result` / `mcp.cancel` 三段现成的下一步调用。
  启动之前的拒绝（`plan_invalid`、`gpu_busy`、`output_exists` 等）和阻塞调用一样当场返回。
  - **看进度**：`dic_query_manifest(run_dir=<output_dir>, documents=[".mcp-job.json"],
    fields=["state", "progress"])`（作业单的 `mcp.poll` 里就是这一条，连同建议的间隔 `poll_after_s`：
    单事件 5 s，长 campaign 最多 30 s）。带 `fields` 的一次回复约 0.7 KB，不带则返回整条记录（约 3 KB，
    含 `exit_code`、`verdict`、`cancel_reason` 等结束后才有的字段）。
    `state` 为 `starting` / `running` 表示还在跑；作业进程每 3 s 重写一次 `progress`（没变化也写，
    `updated_at` 就是心跳）：`events_done` / `events_total`、`frames_done` / `frames_total`、
    `current_event` 及其帧数、`elapsed_s`。进度是作业进程数输出目录里的文件得来的（事件的帧目录、
    COD CSV、binary 日志；帧数是当前事件的 `frame_<n>.csv` 个数），不是 pipeline 报的：事件做聚合
    的那一小段进度不动。阻塞调用的作业记录里同样有 `progress`，别的会话可以看。
  - **结束**：`state` 变成 `exited` 就是 pipeline 结束了，`exit_code` 0 = 每个事件都 ok（`--strict`），
    3 = 有事件失败；记录里抄了摘要的 `verdict`、`events_ok`、`events_failed`。结果读
    `summary.json`（`dic_query_manifest(run_dir, documents=["summary.json"])`）：内容就是阻塞调用返回的
    那份摘要，但没有 `ok`、`mcp.brief` 和大 campaign 的精简视图（那些只在阻塞调用的回复里）。
    其余终态：`timed_out`、`cancelled`、`spawn_failed`、`runner_lost`、`job_lost`。
    记录写成 `exited` 之后约 30 ms GPU 锁才释放（2026-10-05 实测 25–30 ms）；看到 `exited` 立刻提交下一个作业
    不用自己等：run 工具发现占用者的记录已是终态时原地等锁，最多 1 s，等不到才报 `gpu_busy`。
  - **取消**：`dic_job_cancel(output_dir[, job_id][, cancel_token])` 向作业进程的进程组发 SIGTERM；
    作业进程停掉 pipeline 和 binary（SIGTERM，5 s 后 SIGKILL）、把记录写成 `cancelled`（带
    `cancel_reason`）后退出，锁随之释放；工具等这一切完成（最多 20 s，再不行自己杀并记 `cancelled`）
    才返回。返回里的 `action` 说明发生了什么：`cancelled`（这次调用停掉了作业）、`none`（作业早已结束，
    或者刚好自己结束了；作业结束之后再取消也是它，不发任何信号。两个取消同时到、作业还在被停的过程中，
    两边都回 `cancelled`）、`settled`（记录写着在跑、其实什么进程都没有了：
    把这条陈旧记录收尾成 `runner_lost` / `job_lost`）。正被阻塞调用等着的作业被取消时，那个调用返回
    `job_cancelled`。取消后的输出目录留着半成品，不能再用。作业进程刚被外力杀掉、server 还在接管的那一瞬间
    取消也一样：没有作业进程可通知时直接停作业的进程组，记录仍写 `cancelled`。发任何信号之前都在 `/proc`
    里核对身份：进程的启动时间要与记录里的 `runner_start_ticks` / `job_start_ticks` 相同，命令行里要有这条
    作业记录（作业进程）或恰好这个输出目录（pipeline、binary；`…/run1` 不会认成 `…/run10`）。终态一旦
    写下就不再被覆盖。
  - **谁能取消**：启动作业的那个会话只给 `output_dir` 就行；别的会话必须带那次 `wait: false` 返回的
    `cancel_token`，否则 `job_not_owned`、什么都不停。`job_id` 不算凭据（作业记录和 `gpu_busy` 的占用者
    信息里都看得到）；token 只返回一次，记录里只存它的 sha256，审计日志里只记哈希前缀（键名含 `token` 的
    参数在任何层级都这样处理，被 schema 拒掉的畸形调用和 server 的 stderr 告警也一样）。阻塞调用起的
    作业没有 token，只有它自己的会话能取消。**别取消不是自己起的作业**：碰到 `gpu_busy` 就等，或换一块
    GPU。`output_dir` 里没有作业记录是 `job_not_found`，`job_id` 对不上是 `job_id_mismatch`。
  - **陈旧记录**：作业进程和 server 同一刻被杀时，没人给记录收尾，`state` 会一直是 `running`，
    但 `progress.updated_at` 不再前进，时间也会超过 `deadline_at`；等这个作业的进程都没了，任何会话调
    `dic_job_cancel(output_dir)` 都能把它收尾（`action: settled`）。
  - 例（30 kN 两段 campaign，不等待）：`{"params": {"photo_dir": ".../30kN/photos", "output_dir":
    "<job root>/<新目录>", "cal": 0.5641, "subset": 15, "step": 1, "interface_y": 131, "gauge": 16,
    "roi_cols": [72, 958], "event_interface_y": [{"pattern": "event2[1-5]", "interface_y": 133}],
    "timeout": 1800, "wait": false}}`。
- `dic_query_manifest` 的 `max_bytes`（默认 16 KB，≤ 64 KB）是整个回复的硬上限：超过 16 项的
  数组折叠成计数 + 统计，并点名 status 不是 ok 的条目；`fields`（如 `["events[0].warnings",
  "frames[0:20]"]`，每条 ≤ 256 字符）、`documents`、`detail` 取指定部分；放不下的列在 `omitted`。
  指向白名单外的 JSON symlink 不读、不报大小。
- `dic_aggregate_cod`：测线不在 frame 数据里时返回 `no_cod_rows` 和可用行号，不写文件；
  输出必须是 `.csv`，不能写进 frames 目录，不能替换 run 目录里已有的文件（`cod_result.csv`、
  `summary.json` 等）。
- 每次调用向 job root 下的 `mcp_audit.log` 追加 started / final 两条 JSONL 审计记录（包括只读的
  `dic_plan`；server 收到 SIGTERM 时给在途调用补 final：`detached` / `server_terminated`；参数超
  8 KiB 只记摘要）；写权限受限的 agent 用 CLI 而不是 MCP。
- 已知限制：`wait: false` 的作业靠轮询，没有推送通知；它结束后只有 `summary.json`，没有 `ok` /
  `mcp.brief`；`cancel_token` 丢了又换了会话就取消不了，只能等它结束、超时或由人处理；锁只管 MCP
  作业，看不到直接跑的 CLI；`.mcp-control/` 下的锁文件不清理。server、作业进程、pipeline 的 python
  三者同一刻被 SIGKILL（例如一句 `pkill -9 python`）时，binary 会留下继续跑而锁已经空了，第二个作业
  能上同一块 GPU（作业模型之前就是这样）：带这个作业 token 的 `dic_job_cancel` 能停掉它，没有 token 就得人来杀。

## 如何验证一次运行

```bash
/home/chonglang/opencorr-gpu-mvp-build/opencorr_cli_cuda_mvp --version
test -s "$OUT/cod_result.csv"
head -n 2 "$OUT/cod_result.csv"

python3 - <<'PY' "$OUT/summary.json"
import json, sys
def reject(value):
    raise ValueError(f"non-standard JSON constant: {value}")
s = json.loads(open(sys.argv[1], encoding="utf-8").read(), parse_constant=reject)
assert s["schema_version"] == 1
levels = lambda d: [(w["code"], w["severity"]) for w in d["warning_details"]]
print("verdict:", s["verdict"], "| top level:", levels(s))
print("binary:", s["binary"])
print("global:", s["global"])
for e in s["events"]:
    print(f"event {e['name']}: status={e['status']} frames={e['frames']} "
          f"valid_ratio={e['valid_ratio']} zncc_median={e['zncc_median']} "
          f"warnings={levels(e)}")
    print("  geometry:", e["geometry"])
    print("  cod_stats:", e["cod_stats"])
assert s["verdict"] in {"ok", "warn"}, s["verdict"]
assert s["global"]["events_failed"] == 0, s["global"]
for e in s["events"]:
    assert e["status"] == "ok", e["name"]
    assert e["valid_ratio"] >= 0.95, e["name"]
    assert e["zncc_median"] >= 0.9, e["name"]
    c = e["cod_stats"]
    assert c["rows"] and not c["all_zero"], e["name"]
    assert c["frames_with_points"] == c["frames"], e["name"]
PY

python3 - <<'PY' "$OUT/cod_result.csv"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
frames = {r["frame"] for r in rows}
xs = {r["x_mm"] for r in rows}
print(f"rows={len(rows)} frames={len(frames)} x_points={len(xs)} "
      f"frames*x_points={len(frames) * len(xs)}")
assert rows, "COD CSV has a header but no data rows"
assert any(float(r["cod_um"]) != 0.0 for r in rows), "COD is identically zero"
PY
```

**看 `warning_details` 的级别**：`warnings` 沿用旧含义，只放 warn / fail 条目、旧 code 保持原文；
`warning_details` 列出全部条目和级别。`info` 不影响 verdict、也不进 `warnings`（例如 30 kN/event10 的
`input_8bit_max_255`：COD 模式按测线带判，canonical 测线 123/139 的两条带 0.03% / 0.81% 像素为 255，
低于默认阈值 5%，记为 info；服务器上的生产 binary 走 config 解析时的 `gpu_bin_outside_checkout` 也是
info，所以这里顶层 `warnings` 是 `[]`）；`warn` 要逐条读原因。`input_8bit_max_255` 的判据、阈值和
每条测线带的数字在 `events[].input_saturation`（dry-run 也有）；按 30 kN 的 canonical 测线几何
（123/139，2026-10-05 dry-run）量，config1 40 kN run3/event04 的下测线带 5.94%、lowrough-20260918
20 kN/event16 为 6.68% 像素为 255，都是 warn；按默认 gauge 的 116/150 量则约 7% / 8%
（`docs/quality-checks.md` 第 5 节）。顶层只装 binary / provenance / 参数告警，事件级原因在 `events[]`。每个 code 的含义、
级别与处置见契约 `outputs.json_summary.warning_vocabulary`。还要核对：

- COD CSV 应覆盖全部目标帧（`cod_stats.frames_with_points` 等于 `frames`），行数约为
  `目标帧数 × ROI 内 x 点数`，且不全为 0。pipeline 已经自检这些（`cod_empty` 失败、
  `cod_all_zero` 警告），脚本里的断言和对 CSV 的复核是第二道保险。campaign 的 COD 文件是
  `$OUT/<事件名>.csv`。
- 需要审计位移场或估噪声时加 `--keep-frames`，在 `_frames_<事件名>/` 的 9 列 frame CSV
  里过滤 `finite(u,v,zncc) && zncc >= 0` 后再统计。
- `params` 同时保存 argparse 原值和 `input_*_effective`；用 `auto` 时该字段标为
  `runtime_per_image`，`events[].input_reference` 只代表参考帧。
- 报单点极值（如 COD 最大值）前看 [`docs/quality-checks.md`](../quality-checks.md) 第 6 节：
  默认 naive 的极值就是 naive 值；显式 `--cod-method fillboundary` 时单点值对噪声敏感，应并列同一点的
  naive 值（`cod_stats.naive_at_max_um`）或报分位数。

## 操作红线

适用于只跑 DIC 的操作型 agent：

- 不重编、不替换、不修改生产 binary、`run_dic.sh` / pipeline 脚本和
  `~/.config/fasticgn/config.toml`；不往生产 venv 装包。
- 输出只写新目录（`~/dic_jobs/<新目录>` 或自己的 scratch），不覆盖、不续跑旧目录；
  `/data` 与 `/home/chonglang/opencorr` 只读；不删除任何东西。
- GPU 作业前先看 `nvidia-smi`。
- 契约、dry-run、摘要和实际文件互相矛盾时停下来报告，不要猜。带日期的记录值（本文的
  hash、行数、耗时）与现场不同不算矛盾。
- 改代码、尤其是任何可能改变输出字节的数值改动，按根 [`README.md`](../../README.md) 中
  Numerical parity 的要求执行；替换生产 binary 是另一步，需要 Boss 再次明确批准。改脚本时同一提交
  更新对应文档。
- 验收命令必须真实运行并保留原始输出；区分 verified fact 与 inference。

## 求助规则

契约、dry-run、摘要和实际文件互相矛盾时，停下来报告：

1. 完整命令和退出码；
2. wrapper `--help` 与 `opencorr_cli_cuda_mvp --version` 的输出；
3. JSON 摘要（若已生成）；
4. 相关路径、首帧尺寸 / 位深，以及你无法消歧的具体问题。

## 必须掌握的坑

以下由 `pipeline/sync_agent_docs.py` 从契约 `pitfalls`（通用，`dic_get_pitfalls` 返回的就是它）和 `lab_notes`（2026-10-19 起单列，只对本实验室成立）生成，不要手改：

<!-- BEGIN GENERATED: interface_contract.json#pitfalls -->
1. **subset 有两套语义。** Pipeline 的 `--subset 15` 是 SIZE，内部换算 `radius=(size-1)//2=7`；binary 的位置参数是 RADIUS。把 15 直接传给 binary 曾造成约 100 μm 的假失败。binary 现在也接受 `--subset-size 15`（= radius 7，输出逐字节相同），与位置参数二选一。
2. **收敛阈值有版本史。** 2026-07-05 起的构建 CLI 默认 0.001；Pipeline 恒显式传 `--conv-threshold 0.001`。若绕过 pipeline 调用更老 binary，必须自己传 0.001；老默认 0.05 会留下约 4 μm plateau（`--help` 会打印当前构建的默认值）。
3. **`zncc < 0` 是状态行。** 其中的 `u,v` 可能是初值、种子或失败残留，不能当有效位移。所有统计和聚合至少过滤 `zncc >= 0`。
4. **12-bit 左对齐数据要显式传位深。** 有的相机把 12-bit 样本左移 4 位存进 uint16 容器（低 4 位恒零）：`--input-depth auto` 会把它当 16-bit（只有存储最大值 ≤ 4095 才推断成 12），`auto` 不得与显式 `--input-align` 同用，而默认的 `--input-depth 8` 会把任何 16-bit 文件降成 8 bit 且照样 exit 0。先跑 dry-run、按 `events[].input_recommendation` 传参；左对齐 12-bit 数据显式传 `--input-depth 12 --input-align left`，不要用 `auto`。
5. **COD 默认口径的沿革：现在是 naive。** 默认口径 2026-07-09 以前是 `naive`（两测线 `u` 直接差分），2026-07-09 至 2026-10-05 是 `fillboundary`（各测线 `u` 沿 `uy` 外推到两线中点再相减），2026-10-05 起改回 `naive`，`fillboundary` 仍可用 `--cod-method fillboundary` 选（2026-10-05 拍板）。原因：测线干净时 fill-boundary 只比 naive 高几个百分点（默认 gauge 35、两个 800 帧实验室事件的逐帧空间中位数约 5–7%；界面 ±8 px、界面行给对时末 10 帧平台约 1.7–3.4%），但每条测线的 `uy` 噪声被外推距离放大：25 组实验室 static-ref 上，测线距界面 ±8 px 时 fill-boundary 噪声底约为 naive 的 2.0 倍，±17.5 px 时约 3.5 倍；那两个 800 帧事件上逐点比值（只计 naive 绝对值 > 50 μm 的点）1%–99% 分位约 0.8–1.3，事件最大值是同一点 naive 的 1.27 和 1.43 倍。测线进了界面过渡带时 fill-boundary 偏得也多得多：界面行错 2 px 的四个实验室事件，末 10 帧 fill-boundary 只有平台跳变的 77–79%，naive 为 93–94%。本实验室各数据集的测线惯例见 `lab_notes`（cod-lab-line-conventions）。做法：不需要 fill-boundary 语义就不传 `--cod-method`（即 naive）；2026-07-09 至 2026-10-05 之间没传 `--cod-method` 的结果是 fill-boundary，复现要显式 `--cod-method fillboundary`，以 summary 的 `events[].cod_mode` 为准。用 fill-boundary 时报极值要并列同一点的 naive 值（`cod_stats.naive_at_max_um`，只在 fillboundary 下有值）和 |COD| 的 p95（`cod_stats.abs_p95_um`），或用中位数/分位数（见 `docs/quality-checks.md` 第 6 节）。`--gauge` 要显式写，不写就是 35。换一批数据先量界面行，两条测线都要在过渡带外：dry-run 和 campaign 全 `ok` 都查不出这一点。界面行不同的事件可以在一次 campaign 里用 `--interface-y` 给默认行、`--event-interface-y PATTERN=行` 给其余事件（2026-10-05 起；此前一次 campaign 只能给一个界面行，要分组跑）。
6. **v9 及更早只有 5 列。** 缺少 `uy` 的 CSV 让 fill-boundary 数学上等同 naive；2026-09-17 起显式 `--cod-method fillboundary` 遇到这种 CSV 直接报 `cod_uy_column_missing`，不再静默降级。默认的 `naive`（2026-10-05 起）不需要 `uy`，5 列可以直接聚合（summary 记 `cod_mode: naive`）；要 fill-boundary 就用 9 列 binary（v10+）。

以下是契约 `lab_notes`（只对本实验室成立，不在 `dic_get_pitfalls` 里；MCP 用 `dic_describe(section='lab_notes')` 取）：

7. **highspeed2 是 12-bit 左移。** TIFF 容器为 uint16，但低 4 位恒零；正确参数是 `--input-depth 12 --input-align left`。`auto` 会误判成 16，而且 `auto` 不得与显式 `--input-align` 同用。对这批数据不要用 `auto`。
8. **30 kN 的测线惯例与分段界面行。** 实验室惯例是界面 ±8 px，界面行按段登记（2026-10-05 逐事件实测末 10 帧）：event01–20 界面 131，`--interface-y 131 --gauge 16`（测线 123/139）；event21–25 界面 133，`--interface-y 133 --gauge 16`（测线 125/141；event24 没有可分辨的破裂，按段取 133）。原因是整个试样在 event20 → 21 之间相对相机向下平移约 2 px（20 kN 同样在 event26 → 27 之间，界面约 131 → 约 133）。用错段的界面行会有一条测线落进过渡带（fill-boundary：event21/22/23/25 按 131 为 77–79%、按 133 为 103–105%；event10 按 131 为 103%、按 133 为 81%）。`--interface-y 133` 不带 `--gauge` 是默认 35（测线 116/150，2026-10-05 以前的几何）。其他 config1 载荷和 lowrough-20260918 实测在 128–133.5（见"数据集惯例参数"表）。一次 campaign 用 `--interface-y 131 --event-interface-y 'event2[1-5]=133'` 覆盖两段（2026-10-05 起；此前只能按段分开跑）。实验室 COD 不传 `--cod-method`（即 naive），`--gauge 16` 要显式写；通用规则见 pitfall cod-method-generation。
9. **不要混用两套 regime。** GPU 生产常用 size 15、step 1（config1 1036×186 全幅 171,024 个 POI）；LabAgent/绘图常用 size 21、step 5（同一全幅 6,699 个 POI；LabAgent 自己的 ROI 未登记）。binary 全幅网格：x 从 m 起每 step 取一点、到 宽 − m 前止，y 同理，m = radius + 2（radius 7 为 1018 × 168，radius 10、step 5 为 203 × 33）。比较结果前先核对 regime。
10. **带空格路径需要新 binary。** `9b22758` 起支持（随 v1.1.0 发布）；更老版本的 tar-list parser 会错切。
11. **事件数口径。** `config1-bimaterial` 的 10/20/30 kN 数据目录里共 85 个事件（30 + 30 + 25），40 kN 另有 32 个（run1–run3）。旧批处理脚本 `batch_bimaterial_altbin.py` 把 `10kN/event05`、`30kN/event24` 当作历史坏事件剔除，所以历史统计是 83 个（含 40 kN 为 115 个）；剔除原因没有记录；2026-10-05 实测这两个事件末 10 帧在界面两侧没有可分辨的位移跳变（行 111–122 对 140–151 只差 0.0003 / 0.0005 px），界面行无法判定。现行 pipeline 不做剔除，campaign 会处理全部 85 个。引用时写明是哪个口径和载荷范围。
<!-- END GENERATED: interface_contract.json#pitfalls -->
