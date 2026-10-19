#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
FastICGN pipeline：图像序列 → 全场位移场（通用入口）或 COD CSV（实验室后处理）

两种模式由测线参数决定：
  * 不给 --interface-y / --line-y  → **全场位移模式**（通用）：跑整幅 ROI，
    frame CSV 或 raw 留在 <output-dir>/<event>/，不做任何聚合。
  * 给了 --interface-y 或 --line-y → **COD 模式**（实验室专用后处理）：
    binary 只算两条测线，再按 fill-boundary / naive 聚合成 COD CSV。

全场模式（通用）：
  python3 dic_pipeline.py \\
    --image-dir /path/to/frames/ \\
    --output-dir /path/to/results/

COD 批量模式（实验室）：
  python3 dic_pipeline.py \\
    --photo-dir /media/disk/PHOTO/ \\
    --output-dir /media/disk/RESULTS/ \\
    --interface-y 131 \\
    --gauge 16 \\
    --cal 0.5641

COD 批量、只跑部分事件、逐事件界面行（campaign 专用）：
  python3 dic_pipeline.py \\
    --photo-dir /media/disk/PHOTO/ \\
    --output-dir /media/disk/RESULTS/ \\
    --events 'event0*,event1*,event2[0-5]' \\
    --interface-y 131 \\
    --event-interface-y 'event2[1-5]=133' \\
    --gauge 16 \\
    --cal 0.5641

COD 单事件模式：
  python3 dic_pipeline.py \\
    --image-dir /media/disk/PHOTO/1/ \\
    --output-dir /media/disk/RESULTS/ \\
    --interface-y 131 \\
    --gauge 16 \\
    --cal 0.5641

可选参数：
  --subset 15          子集 SIZE（默认 15，奇数；radius=(size-1)//2）
  --step 1             网格步长（默认 1）
  --output-format csv  全场产物格式 csv|raw（默认 csv；raw 仅全场模式）
  --image-pattern RE   图像发现正则（默认「文件名以六位帧号结尾」）
  --image-order X      frame-number（默认）| name
  --roi-cols 50,1000   水平 ROI 限制（像素，可选；COD 聚合用）
  --gauge 16           两条测线的请求间距（像素，默认 35，不会自动计算；实际间距
                       2*(gauge//2)，默认即 34）。上面 30 kN 示例的界面 ±8 px 惯例
                       （测线 123/139）要显式传 --gauge 16，不传就是 35
  --events LIST        campaign 只跑部分事件：逗号分隔的事件名或 glob（区分大小写）；
                       每一项都必须匹配到事件，否则读图前报错（exit 2），什么都不写
  --event-interface-y PATTERN=Y
                       campaign 逐事件界面行（可重复）：匹配 PATTERN 的事件用 Y，其余用
                       --interface-y；--gauge 仍全局。不能与 --line-y 同给；一个事件命中
                       两个不同的 Y、或某条覆盖不匹配任何选中的事件都报错（exit 2）
  --cod-method X       COD 聚合方式：naive（默认，两测线 u 直接差分）| fillboundary
                       （可选，Eq.3 梯度外推到两线中点；2026-07-09 至 2026-10-05 曾是默认）
  --gpu-device 0       GPU 编号（默认 0）
  --gpu-bin PATH       CLI binary；缺省按 FASTICGN_GPU_BIN → 配置文件 → 仓库
                       build/ 树 → 实验室旧路径（deprecated）解析
  --keep-frames        保留 COD 模式的中间 frame CSV（默认删除）
  --max-frames N       只算参考帧之后的前 N 个目标帧（试跑用）
  --strict             先按 dry-run 全部校验、可疑结果按失败处理、有失败事件即非零退出
  --saturation-warn-fraction F
                       =255 像素比例超过 F 才把 input_8bit_max_255 记为 warn（默认 0.05；
                       COD 模式看测线带里参与计算的子区，全场模式看参考帧整帧）
  --json-summary PATH  写结构化运行摘要（可选）
  --dry-run            只读校验并向 stdout 输出 JSON 计划（不启动 GPU）
  --describe           原样输出机器可读接口契约后退出

输出：
  全场模式:   <output-dir>/<event>/frame_000002.csv|bin, ... + manifest
  COD 批量:   <output-dir>/1.csv, <output-dir>/2.csv, ...
  COD 单事件: <output-dir>/cod_result.csv
  binary 日志: <output-dir>/<frame 目录名>.binary.log（binary 的完整 stderr；
               放在帧目录旁边而不是里面，因为 binary 拒绝帧目录里的陌生文件）
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

DEFAULT_IMAGE_PATTERN = r"(\d{6})\.(jpg|jpeg|tif|tiff|png|bmp)$"
IMAGE_RE = re.compile(DEFAULT_IMAGE_PATTERN, re.IGNORECASE)
IMAGE_ORDER_CHOICES = ("frame-number", "name")

# 默认测线间距（像素），基于 bimaterial 实验经验值
DEFAULT_GAUGE = 35

REPO_ROOT = Path(__file__).resolve().parent.parent

# 数据文件（契约与配置模板）。源码树/部署目录里它们就在模块旁边；
# `pip install .` 只装模块，数据文件由 pyproject 的 [tool.setuptools.data-files]
# 放到 <sys.prefix>/share/fasticgn/，所以这里按同样顺序找。
DATA_SHARE_SUBDIR = Path("share") / "fasticgn"


def data_file_path(name: str) -> Path:
    """找一个随包数据文件：模块旁边 → <sys.prefix>/share/fasticgn → 安装基址。"""
    beside = Path(__file__).resolve().with_name(name)
    candidates = [beside]
    for prefix in (sys.prefix, sys.base_prefix):
        candidates.append(Path(prefix) / DATA_SHARE_SUBDIR / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return beside  # 让调用方拿到最自然的路径去报错


INTERFACE_CONTRACT_PATH = data_file_path("interface_contract.json")

# ── 路径解析（review-2026-09-16 §4 / 09-02 审阅 L4）────────────────────────
# 解析顺序固定为：命令行参数 → FASTICGN_* 环境变量 → 配置文件
# (FASTICGN_CONFIG 或 ~/.config/fasticgn/config.toml) → 仓库自身的 build/ 树
# → 实验室旧路径（仅当文件存在，并附 deprecation 提示）。
ENV_PREFIX = "FASTICGN_"
ENV_GPU_BIN = ENV_PREFIX + "GPU_BIN"
ENV_CPU_BIN = ENV_PREFIX + "CPU_BIN"
ENV_HOME = ENV_PREFIX + "HOME"
ENV_CONFIG = ENV_PREFIX + "CONFIG"
ENV_READ_ALLOWLIST = ENV_PREFIX + "READ_ALLOWLIST"
ENV_JOB_ROOT = ENV_PREFIX + "JOB_ROOT"

USER_CONFIG_PATH = Path.home() / ".config" / "fasticgn" / "config.toml"
EXAMPLE_CONFIG_PATH = data_file_path("config.example.toml")

GPU_BINARY_NAME = "opencorr_cli_cuda_mvp"
CPU_BINARY_NAME = "opencorr_cli_cpu"

# Build trees searched inside a plain clone (or under FASTICGN_HOME), in order.
BUILD_SUBDIRS = (
    "build/cuda-release", "build/cuda-tests", "build/cuda-fftw",
    "build/cpu-only", "build",
)

# Legacy lab deployment. Used only as the last resort, only when the file is
# really there, and always with a machine-readable deprecation note.
LEGACY_LAB_HOME = "/home/chonglang"
LEGACY_LAB_BUILD_DIR = "/home/chonglang/opencorr-gpu-mvp-build"
# Kept so that existing importers (mcp_server, lab scripts) keep working.
DEFAULT_GPU_BIN = f"{LEGACY_LAB_BUILD_DIR}/{GPU_BINARY_NAME}"

# Read roots the lab MCP server has always allowed; a configured allowlist
# replaces them (the repository root and the job root are always allowed).
DEFAULT_READ_ROOTS = ("/data", "/media")

# ── Exit codes ───────────────────────────────────────────────────────────────
# 0/1/2 keep their historical meaning. 3 exists only with --strict: the run
# finished, but at least one event has status "fail" (without --strict a failed
# single event exits 1 and a campaign exits 0, as before).
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_INVALID = 2
EXIT_STRICT_EVENT_FAILED = 3

# ── Warning severity (review-2026-10-03 F11/F12) ────────────────────────────
# Every summary warning string starts with a stable code (text before ': ').
# The verdict is decided by severity only: any "fail" -> fail, otherwise any
# "warn" -> warn, otherwise ok; "info" never changes the verdict.
# input_8bit_max_255 is info or warn per event (input_saturation_severity);
# the table holds the level used when nothing more specific is known.
SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_FAIL = "fail"
SEVERITY_RANK = {SEVERITY_INFO: 0, SEVERITY_WARN: 1, SEVERITY_FAIL: 2}
WARNING_SEVERITY = {
    # event level: the event did not produce a usable result
    "insufficient_images": SEVERITY_FAIL,
    "gpu_cli_failed": SEVERITY_FAIL,
    "cod_uy_column_missing": SEVERITY_FAIL,
    "cod_aggregation_failed": SEVERITY_FAIL,
    "cod_empty": SEVERITY_FAIL,
    "cod_line_missing": SEVERITY_FAIL,
    "pipeline_exception": SEVERITY_FAIL,
    # event level: a result exists but needs a look
    "cod_all_zero": SEVERITY_WARN,
    "cod_frames_without_points": SEVERITY_WARN,
    "binary_warning": SEVERITY_WARN,
    "stale_cod_csv": SEVERITY_WARN,
    "cod_csv_replaced": SEVERITY_WARN,
    "duplicate_frame_number": SEVERITY_WARN,
    "duplicate_reference_frame_number": SEVERITY_WARN,
    "valid_ratio_below_0.95": SEVERITY_WARN,
    "zncc_median_below_0.9": SEVERITY_WARN,
    "frame_stats_unavailable": SEVERITY_WARN,
    "zncc_median_unavailable": SEVERITY_WARN,
    "input_8bit_max_255": SEVERITY_WARN,
    "frame_csv_read_errors": SEVERITY_WARN,
    "preexisting_frame_csv_detected": SEVERITY_WARN,
    "frame_count_differs_from_mode": SEVERITY_WARN,
    "effective_input_reference_unavailable": SEVERITY_WARN,
    # event level: context only
    "resumed_verified": SEVERITY_INFO,
    "max_frames_applied": SEVERITY_INFO,
    "frame_fallback_used": SEVERITY_INFO,
    "binary_log_unavailable": SEVERITY_INFO,
    # top level
    "pipeline_error": SEVERITY_FAIL,
    "argument_warning": SEVERITY_WARN,
    "binary_version_probe_failed": SEVERITY_WARN,
    "binary_git_hash_unavailable": SEVERITY_WARN,
    "binary_build_date_unavailable": SEVERITY_WARN,
    "gpu_bin_outside_checkout": SEVERITY_INFO,
}
# --strict turns these warn-level codes into failures of their event.
STRICT_FAIL_CODES = frozenset({"cod_all_zero", "duplicate_frame_number"})

# "638ad18", here and in every comment and docstring below, is the pipeline and
# binary of 2026-10-03 (a commit of the author's private development history,
# not resolvable in the public repository): the compatibility baseline whose
# summary fields, warning wording and option abbreviations this file keeps.
#
# The summary's warnings lists keep the 638ad18 meaning (review 2026-10-04 R2):
# only warn and fail entries, and for codes 638ad18 already emitted exactly its
# text. Everything, info entries and the full detail included, is in
# warning_details. These two codes carried a different text in 638ad18.
LEGACY_WARNING_TEXT = {
    "input_8bit_max_255": "input_8bit_max_255: possible saturation_or_high_bit_truncation",
    "gpu_cli_failed": "gpu_cli_failed",
}

# Dry-run issues that --strict does not copy into the run summary: they only
# concern the dry-run itself, or the run reports them again on its own.
PRECHECK_NOT_RECORDED = frozenset({
    "json_summary_ignored", "gpu_bin_outside_checkout", "max_frames_applied",
    "duplicate_reference_frame_number",
})

# Heuristic, tunable with --saturation-warn-fraction: the fraction of pixels at
# 255 (8-bit effective input) above which input_8bit_max_255 is a warning rather
# than information. COD mode grades the pixels the binary actually correlates:
# for each measurement line, the band of its subsets that enter the COD
# aggregation (saturation_report), and the largest band fraction decides. Full-
# field mode grades the whole reference frame. Until 2026-10-04 COD mode graded
# the whole frame too (default 0.02). But every lab reference frame surveyed on
# 2026-10-04 (159 events, 8 datasets) has 1,400 or more pixels at 255 in rows
# 0-108, far above the measurement lines; they alone make about 0.73 % of the
# frame. In the bright datasets the band of one line holds more than twice the
# frame fraction (docs/quality-checks.md section 5). A reference read from a
# container deeper than 8 bits stays a warning whatever the fraction, because
# then a pixel at 255 hints at high-bit truncation, not at saturation.
DEFAULT_SATURATION_WARN_FRACTION = 0.05
# subsets_over_10pct counts the subsets with more than this share of pixels at 255.
SUBSET_SATURATION_CUT = 0.10

# Measurement lines exist only in COD mode; the binary skips a line inside its
# subset margin with a "Warning:" line on stderr.
BINARY_LOG_SUFFIX = ".binary.log"
MAX_BINARY_WARNINGS_IN_SUMMARY = 5


def config_file_path() -> Path | None:
    """FASTICGN_CONFIG 优先；否则用 ~/.config/fasticgn/config.toml（存在才用）。"""
    override = os.environ.get(ENV_CONFIG)
    if override:
        return Path(override).expanduser()
    if USER_CONFIG_PATH.is_file():
        return USER_CONFIG_PATH
    return None


def load_config(notes: list[str] | None = None) -> dict:
    """读取可选 TOML 配置的 [paths] 段；任何问题都降级为 note，不抛异常。"""
    path = config_file_path()
    if path is None:
        return {}
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11
        if notes is not None:
            notes.append(f"config_file_ignored: tomllib unavailable: {path}")
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, ValueError) as exc:
        if notes is not None:
            notes.append(f"config_file_unreadable: {path}: {exc}")
        return {}
    paths = data.get("paths") if isinstance(data, dict) else None
    return paths if isinstance(paths, dict) else {}


def resolve_setting(key: str, argument: object = None,
                    notes: list[str] | None = None) -> tuple[object, str]:
    """参数 → 环境变量 → 配置文件；返回 (值, 来源)，都没有时返回 (None, "unset")。"""
    if argument:
        return argument, "argument"
    env_name = ENV_PREFIX + key.upper()
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value, f"environment:{env_name}"
    value = load_config(notes).get(key)
    if value:
        return value, f"config:{config_file_path()}"
    return None, "unset"


def binary_search_roots() -> list[Path]:
    """二进制搜索根：FASTICGN_HOME（若设置）在前，然后是仓库自身。"""
    roots: list[Path] = []
    home, _ = resolve_setting("home")
    if isinstance(home, str) and home:
        roots.append(Path(home).expanduser())
    roots.append(REPO_ROOT)
    return roots


def binary_candidates(name: str) -> list[Path]:
    candidates: list[Path] = []
    for root in binary_search_roots():
        candidates.append(root / name)
        candidates.extend(root / sub / name for sub in BUILD_SUBDIRS)
    return candidates


def resolve_binary(name: str, key: str, argument: str | None = None,
                   notes: list[str] | None = None) -> tuple[str, str]:
    """解析一个 CLI binary 路径。返回 (路径, 来源)；来源 "unresolved" 表示没找到。"""
    value, source = resolve_setting(key, argument, notes)
    if isinstance(value, str) and value:
        # Absolute, so that "./x" cannot turn into a bare name that subprocess
        # then looks up on PATH (review-2026-10-03 S1-4). Only the current
        # directory is prepended: no normpath, which would fold "link/.." textually
        # and point past a symlinked directory (review 2026-10-04 R5).
        path = Path(os.path.expanduser(value))
        if not path.is_absolute():
            path = Path(os.getcwd()) / path
        return str(path), source
    for candidate in binary_candidates(name):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate), "repository_build"
    legacy = Path(LEGACY_LAB_BUILD_DIR) / name
    if legacy.is_file():
        if notes is not None:
            notes.append(
                f"deprecated_lab_default_path: {name} resolved to {legacy} from the "
                f"hard-coded lab path; set {ENV_PREFIX}{key.upper()} or pass an "
                "explicit path instead"
            )
        return str(legacy), "legacy_lab_path"
    return str(legacy), "unresolved"


def resolve_gpu_bin(argument: str | None = None,
                    notes: list[str] | None = None) -> tuple[str, str]:
    return resolve_binary(GPU_BINARY_NAME, "gpu_bin", argument, notes)


def resolve_cpu_bin(argument: str | None = None,
                    notes: list[str] | None = None) -> tuple[str, str]:
    return resolve_binary(CPU_BINARY_NAME, "cpu_bin", argument, notes)


def binary_search_description(name: str) -> str:
    """把实际搜索顺序写进错误信息，免得用户猜。"""
    tried = [str(path) for path in binary_candidates(name)]
    tried.append(str(Path(LEGACY_LAB_BUILD_DIR) / name) + " (deprecated lab path)")
    return (
        f"no {name} found. Resolution order: explicit argument, "
        f"${ENV_PREFIX}{'GPU_BIN' if name == GPU_BINARY_NAME else 'CPU_BIN'}, "
        f"{config_file_path() or USER_CONFIG_PATH} [paths], then: " + ", ".join(tried)
    )


def binary_is_from_checkout(path: str, source: str) -> bool:
    """True when the binary was passed explicitly or lives in this checkout's build trees."""
    if source == "argument":
        return True
    if source != "repository_build":
        return False
    parent = os.path.dirname(os.path.abspath(path))
    checkout_dirs = [REPO_ROOT] + [REPO_ROOT / sub for sub in BUILD_SUBDIRS]
    return any(parent == os.path.abspath(directory) for directory in checkout_dirs)


def binary_source_note(path: str, source: str) -> str | None:
    """Info-level note when the binary did not come from --gpu-bin or this checkout."""
    if binary_is_from_checkout(path, source):
        return None
    return (
        f"the GPU binary {path} was resolved from {source}, not from --gpu-bin or this "
        f"checkout's build/ trees ({REPO_ROOT}); pass --gpu-bin to choose the build explicitly"
    )


def finite_float(text: str) -> float:
    """argparse type: a finite float (NaN and Infinity are rejected at parse time)."""
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be finite, got {text!r}")
    return value


def positive_int(text: str) -> int:
    """argparse type: an integer >= 1."""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def unit_fraction(text: str) -> float:
    """argparse type: a finite fraction in [0, 1]."""
    value = finite_float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be within [0, 1], got {text!r}")
    return value


def resolve_read_allowlist(notes: list[str] | None = None) -> tuple[list[Path], str]:
    """MCP 读白名单：配置值替换 /data,/media 默认；仓库根始终允许。"""
    value, source = resolve_setting("read_allowlist", None, notes)
    entries: list[str]
    if isinstance(value, str) and value:
        entries = [item for item in value.split(os.pathsep) if item.strip()]
    elif isinstance(value, (list, tuple)) and value:
        entries = [str(item) for item in value if str(item).strip()]
    else:
        entries, source = list(DEFAULT_READ_ROOTS), "default"
    roots = [Path(entry).expanduser() for entry in entries]
    if REPO_ROOT not in roots:
        roots.append(REPO_ROOT)
    return roots, source


class PipelineArgumentParser(argparse.ArgumentParser):
    """Normal CLI errors stay argparse-compatible; dry-run errors become stdout JSON."""

    machine_readable_errors = False

    def error(self, message: str) -> None:
        if self.machine_readable_errors:
            payload = minimal_dry_run("argument_parse_error", message)
            print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
            raise SystemExit(2)
        super().error(message)


def minimal_dry_run(code: str, message: str) -> dict:
    """The dry-run JSON when no plan could be built (argument_parse_error, dry_run_exception).

    It has every field of contract outputs.dry_run.required_top_level_fields,
    notes included, so a caller can read plan["notes"] whatever happened
    (review 2026-10-04 recheck N1); full plans add full_plan_additional_fields.
    """
    return {
        "schema_version": 1,
        "dry_run": True,
        "valid": False,
        "exit_code": 2,
        "gpu_started": False,
        "filesystem_writes": False,
        "errors": [{"code": code, "message": message}],
        "warnings": [],
        "notes": [],
        "events": [],
    }


def requested_long_options(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    """Long options named on a command line, abbreviations resolved as argparse does.

    argparse accepts any unambiguous prefix (``--dry`` for ``--dry-run``,
    ``--desc`` for ``--describe``), as 638ad18 did. main() must recognise those
    before parsing: ``--describe`` is answered without the required options, and
    a dry-run must report even a parse error as JSON (review 2026-10-04 R3).
    """
    options = [option for action in parser._actions for option in action.option_strings
               if option.startswith("--")]
    found: set[str] = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("--") or len(token) < 3:
            continue
        name = token.split("=", 1)[0]
        if name in options:
            found.add(name)
            continue
        matches = [option for option in options if option.startswith(name)]
        if len(matches) == 1:
            found.add(matches[0])
    return found

# numpy/Pillow 由 pipeline/requirements.txt 管理；不再向 sys.path 注入外部 venv。


def compile_image_pattern(pattern: str | None) -> re.Pattern[str]:
    """--image-pattern 的编译；None = 默认的“文件名以六位帧号结尾”规则。"""
    return IMAGE_RE if not pattern else re.compile(pattern, re.IGNORECASE)


def find_images(image_dir: str, pattern: str | None = None,
                order: str = "frame-number") -> list[tuple[int, str]]:
    """扫描目录中的图片，返回 (帧号, 路径) 列表。

    默认行为与历史一致：文件名以六位数字 + 已知扩展名结尾，帧号取那六位数字，
    按帧号排序。``pattern``/``order`` 让非实验室命名也能接入：
      * ``pattern``  —— 任意 Python 正则（对文件名做 ``search``，忽略大小写）。
        若第一个捕获组是纯数字就当帧号，否则按 ``order`` 排序后用 1 起序号。
      * ``order``    —— ``frame-number``（默认，按帧号）或 ``name``（按文件名）。
    """
    if order not in IMAGE_ORDER_CHOICES:
        raise ValueError(f"--image-order must be one of {', '.join(IMAGE_ORDER_CHOICES)}")
    compiled = compile_image_pattern(pattern)
    matches: list[tuple[str, int | None]] = []
    for name in sorted(os.listdir(image_dir)):
        match = compiled.search(name)
        if not match:
            continue
        frame_no: int | None = None
        if match.groups() and match.group(1) is not None and match.group(1).isdigit():
            frame_no = int(match.group(1))
        matches.append((name, frame_no))

    if order == "frame-number" and all(frame_no is not None for _, frame_no in matches):
        matches.sort(key=lambda item: (item[1], item[0]))
    else:
        matches.sort(key=lambda item: item[0])

    images: list[tuple[int, str]] = []
    for index, (name, frame_no) in enumerate(matches, start=1):
        number = frame_no if frame_no is not None else index
        images.append((number, os.path.join(image_dir, name)))
    return images


def find_images_for(args: argparse.Namespace, image_dir: str | Path) -> list[tuple[int, str]]:
    return find_images(str(image_dir), getattr(args, "image_pattern", None),
                       getattr(args, "image_order", "frame-number"))


def discover_events(photo_dir: Path, minimum_images: int = 2,
                    pattern: str | None = None,
                    order: str = "frame-number") -> list[tuple[str, Path]]:
    """发现 photo_dir 下所有含图片的子文件夹。"""
    events: list[tuple[str, Path]] = []
    for entry in sorted(photo_dir.iterdir()):
        if entry.is_dir():
            # 检查是否有图片
            imgs = find_images(str(entry), pattern, order)
            if len(imgs) >= minimum_images:
                events.append((entry.name, entry))
    return events


def compute_line_ys(interface_y: int, gauge: int) -> tuple[int, int]:
    """根据断层面 y 值和测线间距计算两条测线的 y 坐标。"""
    half = gauge // 2
    l0_y = interface_y - half
    l1_y = interface_y + half
    return l0_y, l1_y


def parse_roi_cols(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("--roi-cols 格式应为 x1,x2")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError("--roi-cols 必须包含两个整数") from exc


def is_cod_mode(args: argparse.Namespace) -> bool:
    """有测线参数 = COD 后处理模式；没有 = 通用全场位移模式。"""
    return bool(args.line_y) or args.interface_y is not None


def field_mode_name(args: argparse.Namespace) -> str:
    return "cod" if is_cod_mode(args) else "full_field"


def resolve_line_ys(args: argparse.Namespace) -> tuple[int, int] | None:
    """返回两条测线 y；全场模式（无 --line-y/--interface-y）返回 None。"""
    if args.line_y:
        parts = args.line_y.split(",")
        if len(parts) != 2:
            raise ValueError("--line-y 格式应为 L0,L1")
        try:
            return int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError("--line-y 必须包含两个整数") from exc
    if args.interface_y is not None:
        return compute_line_ys(args.interface_y, args.gauge)
    return None


# ── Campaign event selection (--events) and per-event interface rows
# (--event-interface-y), 2026-10-05. Both are checked before any image is read
# or anything is written; every error below stops the run with exit 2, with or
# without --strict (they are new options, so no historical behaviour to keep).
EVENT_GLOB_CHARS = frozenset("*?[")
EVENT_SELECTION_CODES = frozenset({
    "invalid_events", "events_requires_photo_dir", "event_not_found",
    "event_pattern_no_match", "no_events_selected",
    "invalid_event_interface_y", "event_interface_y_requires_photo_dir",
    "event_interface_y_requires_interface_y", "event_interface_y_with_line_y",
    "event_interface_y_no_match", "event_interface_y_conflict",
})


def is_event_glob(pattern: str) -> bool:
    return any(char in pattern for char in EVENT_GLOB_CHARS)


def event_matches(name: str, pattern: str) -> bool:
    """fnmatch, case-sensitive on every platform: a literal name matches only itself."""
    return fnmatch.fnmatchcase(name, pattern)


def parse_event_patterns(text: str) -> list[str]:
    items = [item.strip() for item in text.split(",")]
    if any(not item for item in items):
        raise ValueError(
            f"--events must be a comma-separated list of event names or glob patterns "
            f"without empty items; got {text!r}"
        )
    return items


def parse_event_interface_y(text: str) -> tuple[str, int]:
    """``PATTERN=Y`` → (PATTERN, Y); one pattern per option, Y a non-negative integer row."""
    pattern, separator, value = text.rpartition("=")
    pattern = pattern.strip()
    if not separator or not pattern:
        raise ValueError(
            f"--event-interface-y must be PATTERN=Y, for example 'event2[1-5]=133'; got {text!r}"
        )
    if "," in pattern:
        raise ValueError(
            f"--event-interface-y {text!r}: one event name or glob per option (no commas); "
            "repeat the option for more patterns"
        )
    try:
        row = int(value.strip())
    except ValueError:
        raise ValueError(
            f"--event-interface-y {text!r}: the interface row must be an integer, got {value!r}"
        ) from None
    if row < 0:
        raise ValueError(
            f"--event-interface-y {text!r}: the interface row must be >= 0 (a reference-image row)"
        )
    return pattern, row


def campaign_event_names(args: argparse.Namespace) -> list[str] | None:
    """The campaign's events as the dry-run counts them: subdirectories with a recognized image.

    Lists directories only (no image is opened). None without a readable --photo-dir
    or with an invalid --image-pattern (those are reported by their own checks).
    """
    if not getattr(args, "photo_dir", None):
        return None
    root = Path(args.photo_dir)
    if not root.is_dir():
        return None
    try:
        compile_image_pattern(args.image_pattern)
    except re.error:
        return None
    return [entry.name for entry in sorted(root.iterdir())
            if entry.is_dir() and find_images_for(args, entry)]


def _name_preview(names: list[str], limit: int = 8) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f", ... ({len(names)} in all)" if len(names) > limit else "")


def plan_event_selection(args: argparse.Namespace,
                         event_names: list[str] | None) -> dict:
    """Resolve --events and --event-interface-y against the campaign's event names.

    Shared by validate_args (dry-run, --strict) and main (normal execution), so
    both see the same errors. ``interface`` maps each selected event to the
    interface row it will use and where that row came from. With neither option
    the selection is every event and every event uses --interface-y, which is
    exactly the behaviour without these options.
    """
    errors: list[dict] = []
    events_arg = getattr(args, "events", None)
    override_args = list(getattr(args, "event_interface_y", None) or [])
    photo_dir = getattr(args, "photo_dir", None)
    patterns: list[str] | None = None
    if events_arg is not None:
        try:
            patterns = parse_event_patterns(events_arg)
        except ValueError as exc:
            errors.append(_issue("invalid_events", str(exc)))
        if not photo_dir:
            errors.append(_issue(
                "events_requires_photo_dir",
                "--events selects events of a campaign and needs --photo-dir; with --image-dir "
                "the directory itself is the one event",
            ))
    overrides: list[dict] = []
    if override_args:
        if not photo_dir:
            errors.append(_issue(
                "event_interface_y_requires_photo_dir",
                "--event-interface-y sets interface rows per campaign event and needs --photo-dir; "
                "for one event pass its row as --interface-y",
            ))
        if args.line_y:
            errors.append(_issue(
                "event_interface_y_with_line_y",
                "--event-interface-y cannot be combined with --line-y: --line-y fixes both lines "
                "for every event, --event-interface-y moves them per event (use --interface-y "
                "and --gauge instead)",
            ))
        elif args.interface_y is None:
            errors.append(_issue(
                "event_interface_y_requires_interface_y",
                "--event-interface-y overrides --interface-y for the events it matches; give "
                "--interface-y as the row of every other event (it also selects COD mode)",
            ))
        for text in override_args:
            try:
                pattern, row = parse_event_interface_y(text)
            except ValueError as exc:
                errors.append(_issue("invalid_event_interface_y", str(exc)))
                continue
            overrides.append({"argument": text, "pattern": pattern, "interface_y": row,
                              "events": None})

    selection = {
        "errors": errors,
        "events_argument": events_arg,
        "patterns": patterns,
        "overrides": overrides,
        "event_names": event_names,
        "selected": None,
        "excluded": None,
        "interface": {},
    }
    if event_names is None:
        return selection

    selected = list(event_names)
    if patterns is not None:
        root = Path(photo_dir)
        for pattern in patterns:
            if any(event_matches(name, pattern) for name in event_names):
                continue
            if is_event_glob(pattern):
                errors.append(_issue(
                    "event_pattern_no_match",
                    f"--events pattern {pattern!r} matches none of the {len(event_names)} events "
                    f"under {root} ({_name_preview(event_names)})",
                ))
            else:
                where = (" (the directory exists but has no recognized image)"
                         if (root / pattern).is_dir() else "")
                errors.append(_issue(
                    "event_not_found",
                    f"--events names {pattern!r}, which is not an event under {root}{where}; "
                    f"events: {_name_preview(event_names)}",
                ))
        selected = [name for name in event_names
                    if any(event_matches(name, pattern) for pattern in patterns)]
        if not selected:
            errors.append(_issue(
                "no_events_selected",
                f"--events {events_arg!r} selects none of the {len(event_names)} events under "
                f"{root}; nothing would run",
            ))
    excluded = [name for name in event_names if name not in selected]
    selection["selected"] = selected
    selection["excluded"] = excluded

    if args.line_y or args.interface_y is None:
        return selection
    for override in overrides:
        override["events"] = [name for name in selected if event_matches(name, override["pattern"])]
        if override["events"]:
            continue
        only_excluded = [name for name in excluded if event_matches(name, override["pattern"])]
        if only_excluded:
            detail = (f"it matches only events that --events leaves out "
                      f"({_name_preview(only_excluded)})")
        else:
            detail = f"it matches no event under {photo_dir}"
        errors.append(_issue(
            "event_interface_y_no_match",
            f"--event-interface-y {override['argument']!r} applies to no selected event: {detail}",
        ))
    for name in selected:
        hits = [override for override in overrides if name in (override["events"] or [])]
        rows = sorted({override["interface_y"] for override in hits})
        if len(rows) > 1:
            errors.append(_issue(
                "event_interface_y_conflict",
                f"event {name} matches --event-interface-y "
                + ", ".join(repr(override["argument"]) for override in hits)
                + f", which give different interface rows {rows}; make the patterns disjoint",
                name,
            ))
            continue
        if hits:
            selection["interface"][name] = {
                "interface_y": rows[0], "source": "event_interface_y",
                "overrides": [override["argument"] for override in hits],
            }
        else:
            selection["interface"][name] = {
                "interface_y": args.interface_y, "source": "interface_y", "overrides": [],
            }
    return selection


def event_interface(selection: dict | None, args: argparse.Namespace,
                    event_name: str) -> dict | None:
    """The interface row an event uses and its source; None when no --interface-y line is used."""
    if args.line_y or args.interface_y is None:
        return None
    if selection is not None and event_name in selection["interface"]:
        return selection["interface"][event_name]
    return {"interface_y": args.interface_y, "source": "interface_y", "overrides": []}


def event_line_ys(args: argparse.Namespace, interface: dict | None,
                  default: tuple[int, int] | None) -> tuple[int, int] | None:
    """An event's [L0, L1]: from its interface row, else the run-wide lines (``default``)."""
    if interface is None:
        return default
    return compute_line_ys(interface["interface_y"], args.gauge)


def event_selection_public(selection: dict | None, args: argparse.Namespace) -> dict | None:
    """``event_selection`` of the dry-run and the summary: null for a single event."""
    if selection is None or not getattr(args, "photo_dir", None):
        return None
    names = selection["event_names"]
    selected = selection["selected"]
    excluded = selection["excluded"]
    return {
        "events_argument": selection["events_argument"],
        "patterns": selection["patterns"],
        "events_discovered": len(names) if names is not None else None,
        "events_selected": len(selected) if selected is not None else None,
        "events_excluded": len(excluded) if excluded is not None else None,
        "excluded_events": excluded,
        "interface_y_default": (
            args.interface_y if not args.line_y and args.interface_y is not None else None
        ),
        "interface_y_overrides": [
            {key: override[key] for key in ("argument", "pattern", "interface_y", "events")}
            for override in selection["overrides"]
        ],
    }


# The binary's POI grid (opencorr_cli_cuda_mvp.cpp buildPoiQueue): no POI closer
# than subset radius + 2 px to an image edge, x = margin, margin + step, ... on
# every computed row; a --line row inside that margin is skipped with a warning.
def binary_poi_margin(subset_radius: int) -> int:
    return subset_radius + 2


def binary_poi_columns(width: int, subset_radius: int, step: int) -> list[int]:
    """POI x coordinates the binary computes on each row (and on each --line row)."""
    margin = binary_poi_margin(subset_radius)
    return list(range(margin, max(margin, width - margin), max(step, 1)))


def binary_computes_line(y: int, height: int, subset_radius: int) -> bool:
    """Whether the binary computes a --line row (False: skipped as too close to an edge)."""
    margin = binary_poi_margin(subset_radius)
    return margin <= y < height - margin


def roi_poi_columns(columns: list[int], roi_cols: tuple[int, int] | None) -> list[int]:
    """The POI columns COD aggregation keeps: those inside --roi-cols, both ends included."""
    if roi_cols is None:
        return list(columns)
    return [x for x in columns if roi_cols[0] <= x <= roi_cols[1]]


def build_dic_command(gpu_bin: str, ref_path: str, tar_list_path: Path,
                      frame_dir: Path, subset_radius: int, step: int,
                      l0_y: int | None, l1_y: int | None, gpu_device: int,
                      extra_cli: list[str] | None = None) -> list[str]:
    """构造真实执行与 dry-run 共用的完整 binary argv。

    `--line` 只在 COD 模式传（l0_y/l1_y 都给出时）。全场模式不传，binary 就
    输出整幅 ROI 的位移场 —— review-2026-09-16 §4 要求的通用入口。
    """
    cmd = [
        gpu_bin,
        "--multi-frame",
        "--ref", ref_path,
        "--tar-list", str(tar_list_path),
        "--output-dir", str(frame_dir),
        str(subset_radius),
        str(step),
        "--gpu-device", str(gpu_device),
    ]
    if l0_y is not None and l1_y is not None:
        cmd += ["--line", f"{l0_y},{l1_y}"]
    if extra_cli:
        cmd += extra_cli
    return cmd


def invoke_dic(gpu_bin: str, ref_path: str, tar_entries: list[tuple[int, str]],
               frame_dir: Path, subset_radius: int, step: int,
               l0_y: int | None, l1_y: int | None, gpu_device: int,
               extra_cli: list[str] | None = None,
               log_path: Path | None = None) -> dict:
    """调用 GPU CLI，并把它的完整 stderr 流式写进 ``log_path``（若给出）。

    输出名恒写 ``frame_NNNNNN.csv``：`--output-format raw` 时 binary 自己把
    扩展名换成 ``.bin``，这是 CLI 的既有约定。日志不放进 ``frame_dir``：
    binary 续跑时拒绝帧目录里它没写过的文件（"untracked entry"）。

    stderr 在 binary 运行时直接写进日志文件（第一行是命令，结束后追加退出码），
    所以 pipeline 被 kill 或超时后日志仍在（review 2026-10-04 R9）；结束后再从
    文件读回同一段文本做解析。日志文件打不开时退回内存捕获，``log_error`` 说明原因。
    stdout 与以前一样被捕获后丢弃。
    """
    # 生成 tar_list.txt
    tar_list_path = frame_dir / "tar_list.txt"
    with open(tar_list_path, "w") as f:
        for frame_no, tar_path in tar_entries:
            f.write(f"{tar_path} frame_{frame_no:06d}.csv\n")

    cmd = build_dic_command(
        gpu_bin, ref_path, tar_list_path, frame_dir, subset_radius, step,
        l0_y, l1_y, gpu_device, extra_cli,
    )

    outcome = {
        "ok": False,
        "returncode": None,
        "stderr": "",
        "input_reference": None,
        "log_path": None,
        "log_error": None,
    }
    log_handle = None
    if log_path is not None:
        try:
            log_handle = open(log_path, "wb")
            log_handle.write(("# command: " + json.dumps(cmd, ensure_ascii=False) + "\n").encode("utf-8"))
            log_handle.flush()
        except OSError as exc:
            outcome["log_error"] = str(exc)
            if log_handle is not None:
                log_handle.close()
                log_handle = None
    if log_handle is None:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stderr_bytes = result.stderr or b""
    else:
        with log_handle:
            stderr_start = log_handle.tell()
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log_handle)
            log_handle.flush()
            with open(log_path, "rb") as reread:
                reread.seek(stderr_start)
                stderr_bytes = reread.read()
            suffix = b"" if not stderr_bytes or stderr_bytes.endswith(b"\n") else b"\n"
            log_handle.seek(0, os.SEEK_END)
            log_handle.write(suffix + f"# exit code: {result.returncode}\n".encode("utf-8"))
        outcome["log_path"] = log_path
    # Same text the former text=True capture produced (universal newlines),
    # except that undecodable bytes no longer raise.
    stderr = stderr_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    outcome.update({"ok": result.returncode == 0, "returncode": result.returncode, "stderr": stderr})

    if result.returncode != 0:
        print(f"    ❌ GPU CLI 失败 (exit {result.returncode})")
        first_error = first_binary_error(stderr)
        if first_error:
            print(f"    {first_error}")
        if stderr:
            # 再打印最后几行（binary 常在 Error 之后打印 usage）
            lines = stderr.strip().split("\n")
            for line in lines[-3:]:
                if line != first_error:
                    print(f"    {line}")
        if outcome["log_path"] is not None:
            print(f"    完整 stderr: {outcome['log_path']}")
        return outcome
    tar_list_path.unlink(missing_ok=True)
    outcome["input_reference"] = parse_reference_input_summary(stderr)
    return outcome


def run_dic(gpu_bin: str, ref_path: str, tar_entries: list[tuple[int, str]],
            frame_dir: Path, subset_radius: int, step: int,
            l0_y: int | None, l1_y: int | None, gpu_device: int,
            extra_cli: list[str] | None = None) -> tuple[bool, dict | None]:
    """兼容旧调用方：只返回 (成功与否, 参考帧输入模式)。新代码用 invoke_dic。"""
    outcome = invoke_dic(
        gpu_bin, ref_path, tar_entries, frame_dir, subset_radius, step,
        l0_y, l1_y, gpu_device, extra_cli,
    )
    return outcome["ok"], outcome["input_reference"]


def first_binary_error(stderr: str) -> str | None:
    """binary stderr 里第一条 ``Error:`` 行（它在 usage 之前，最后几行常常看不到它）。"""
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("Error:"):
            return stripped
    return None


def binary_warning_lines(stderr: str) -> list[str]:
    """binary 成功时 stderr 里的 ``Warning:`` 行（去重、保序，去掉前缀）。"""
    seen: list[str] = []
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("Warning:"):
            text = stripped[len("Warning:"):].strip()
            if text not in seen:
                seen.append(text)
    return seen


# (stderr substring, remediation). The first match wins; {frame_dir} is filled in.
BINARY_REMEDIATIONS = (
    ("resume rejected: run configuration/reference/binary changed",
     "{frame_dir} holds an earlier run made with other parameters, other inputs or another "
     "binary, and the binary refuses to mix them; use a new --output-dir (or move that "
     "directory aside)"),
    ("resume rejected: targets changed/reordered",
     "{frame_dir} holds an earlier run with a different target list (only an identical "
     "prefix plus appended frames can resume); use a new --output-dir"),
    ("duplicate output",
     "two images map to the same frame number and therefore the same output name; check "
     "--image-pattern and remove duplicate files such as macOS '._*' copies"),
    ("untracked entry in output directory",
     "{frame_dir} contains files the binary did not write; use a new --output-dir"),
    ("has no verified run record",
     "{frame_dir} holds frames without a run ledger (an older binary or a copied run); "
     "use a new --output-dir"),
    ("run state has no valid contract",
     "{frame_dir}/.opencorr-state has no valid run contract; use a new --output-dir"),
    ("already locked",
     "another process is writing {frame_dir}; wait for it to finish or use a new --output-dir"),
    ("invalid device ordinal",
     "--gpu-device names no CUDA device (compare with nvidia-smi); the binary may already "
     "have recorded that device in {frame_dir}/.opencorr-state, so rerun with a new "
     "--output-dir"),
    ("no POIs generated",
     "no measurement point lies inside the image and the binary's subset margin; check "
     "--interface-y/--line-y/--gauge against the image height with --dry-run"),
    ("--repair",
     "a tracked frame in {frame_dir} failed verification; use a new --output-dir, or call "
     "the binary directly with --repair (the pipeline does not pass it)"),
    ("non-UTF-8",
     "a path or output name is not valid UTF-8; rename it"),
    ("failed to load target image",
     "a target image could not be read; check that file"),
    ("image sizes must match",
     "every image of an event must have the reference image's size"),
    ("out of memory",
     "the GPU ran out of memory; free the device or choose another --gpu-device"),
)


def binary_failure_remediation(stderr: str, frame_dir: Path) -> str:
    """把 binary 的常见拒绝翻译成一句可操作的建议。

    只看 ``Error:`` 行：参数错误后 binary 还会打印 usage，其中也出现 ``--repair``。
    """
    error_text = "\n".join(
        line.strip() for line in stderr.splitlines() if line.strip().startswith("Error:")
    )
    for needle, advice in BINARY_REMEDIATIONS:
        if needle in error_text:
            return advice.format(frame_dir=frame_dir)
    return (
        "read the first 'Error:' line and the full binary stderr log; rerun the dry-run "
        "binary_command directly to reproduce"
    )


# "Key: <integer>" totals the multi-frame binary prints at the end of a run.
BINARY_TOTAL_KEYS = {
    "Frames with skipped FFTCC": "frames_with_skipped_fftcc",
    "Frames with fallback FFTCC": "frames_with_fallback_fftcc",
    "Frames with POI-level fallback": "frames_with_poi_fallback",
    "POIs submitted to POI-level fallback": "pois_submitted_to_poi_fallback",
    "POIs recovered by POI-level fallback": "pois_recovered_by_poi_fallback",
    "Frames with temporal extrapolation seed": "frames_with_temporal_seed",
    "POIs seeded by temporal extrapolation": "pois_seeded_by_temporal_extrapolation",
}


def parse_binary_totals(stderr: str) -> dict | None:
    """读 binary 收尾时打印的帧级计数（CSV 模式没有 manifest，这是唯一来源）。"""
    totals: dict = {}
    for line in stderr.splitlines():
        stripped = line.strip()
        match = re.match(r"Processed frames: (\d+), skipped: (\d+)$", stripped)
        if match:
            totals["processed_frames"] = int(match.group(1))
            totals["skipped_frames"] = int(match.group(2))
            continue
        match = re.match(r"Resume: (\d+) verified files", stripped)
        if match:
            totals["resume_verified_files"] = int(match.group(1))
            continue
        if stripped.startswith("Verified complete run"):
            totals["verified_complete_run"] = True
            continue
        key, sep, value = stripped.partition(": ")
        if sep and key in BINARY_TOTAL_KEYS and value.strip().isdigit():
            totals[BINARY_TOTAL_KEYS[key]] = int(value.strip())
    return totals or None


def build_extra_cli(args) -> list[str]:
    """由 argparse 结果构造透传给 GPU CLI 的附加参数。

    conv-threshold 恒显式传，把收敛语义钉死在 pipeline 层，不依赖 binary
    自身默认（老 binary 默认 0.05，warm-start 下会留 ~4um COD plateau 系统差）。
    输入位深三参数仅在用户显式指定时透传（缺省 = binary 默认 8-bit legacy 行为）。
    """
    extra = ["--conv-threshold", str(args.conv_threshold)]
    if getattr(args, "output_format", "csv") == "raw":
        extra += ["--output-format", "raw"]
    if args.input_depth:
        extra += ["--input-depth", args.input_depth]
    if args.input_align:
        extra += ["--input-align", args.input_align]
    if args.input_normalize:
        extra += ["--input-normalize", args.input_normalize]
    return extra


def summary_params(args: argparse.Namespace) -> dict:
    """返回 argparse 全量回显，并显式区分 requested/effective 输入模式。

    非有限 float（argparse 已拒绝，只可能来自程序化调用）写成字符串，
    保证摘要仍是严格 JSON。
    """
    params = {
        key: (repr(value) if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in vars(args).items()
    }
    requested_depth = args.input_depth or "8"
    if requested_depth == "auto":
        effective_depth = "runtime_per_image"
    else:
        effective_depth = requested_depth
    if requested_depth == "12":
        effective_align = (
            args.input_align
            if args.input_align not in {None, "auto"}
            else "runtime_per_image"
        )
    else:
        effective_align = None
    params.update({
        "input_depth_effective": effective_depth,
        "input_align_effective": effective_align,
        "input_normalize_effective": args.input_normalize or "none",
        "field_mode": field_mode_name(args),
        "gpu_bin_source": getattr(args, "gpu_bin_source", "unset"),
    })
    return params


def parse_reference_input_summary(stderr: str) -> dict | None:
    """解析 binary 对 reference 首帧打印的实际输入模式。"""
    number = r"[-+0-9.eE]+"
    pattern = re.compile(
        rf"Reference input: source_depth=(?P<source_depth>[^,]+), "
        rf"source_range=\[(?P<source_min>{number}),(?P<source_max>{number})\], "
        rf"effective_depth=(?P<effective_depth>[^,]+), "
        rf"effective_align=(?P<effective_align>[^,]+), "
        rf"output_depth=(?P<output_depth>[^,]+), "
        rf"output_range=\[(?P<output_min>{number}),(?P<output_max>{number})\], "
        rf"scale=(?P<scale>{number})"
    )
    match = pattern.search(stderr)
    if not match:
        return None
    info = match.groupdict()
    for key in ("source_min", "source_max", "output_min", "output_max", "scale"):
        info[key] = float(info[key])
    info["scope"] = "reference_frame"
    return info


def probe_binary_version(gpu_bin: str) -> tuple[dict, list[str]]:
    """通过只读 --version 探针获取 binary provenance。"""
    binary = {
        "path": str(Path(gpu_bin).resolve()),
        "git_hash": None,
        "build_date": None,
    }
    warnings: list[str] = []
    try:
        result = subprocess.run(
            [gpu_bin, "--version"], capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        warnings.append(f"binary_version_probe_failed: {exc}")
        return binary, warnings

    if result.returncode != 0:
        warnings.append(f"binary_version_probe_failed: exit {result.returncode}")
        return binary, warnings

    for line in (result.stdout + "\n" + result.stderr).splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key in {"git_hash", "build_date"}:
            binary[key] = value or None
    if not binary["git_hash"]:
        warnings.append("binary_git_hash_unavailable")
    if not binary["build_date"]:
        warnings.append("binary_build_date_unavailable")
    return binary, warnings


def saturation_geometry(args: argparse.Namespace, subset_radius: int,
                        line_ys: tuple[int, int] | None,
                        roi_cols: tuple[int, int] | None) -> dict:
    """The geometry the saturation probe grades, from the run's own effective values.

    The dry-run and the run pass what they hand to the binary and to aggregate_cod:
    ``line_ys`` from resolve_line_ys, ``subset_radius`` = (subset - 1) // 2,
    ``roi_cols`` from parse_roi_cols. COD mode grades the measurement-line bands,
    full-field mode the whole frame.
    """
    return {
        "cod": is_cod_mode(args),
        "line_ys": list(line_ys) if line_ys is not None else None,
        "subset_radius": subset_radius,
        "step": args.step,
        "roi_cols": roi_cols,
    }


def line_band(y: int, width: int, height: int, subset_radius: int, step: int,
              roi_cols: tuple[int, int] | None) -> dict | None:
    """The band of line y: the pixels of its subsets that enter the COD aggregation.

    Rows [y - r, y + r]; columns [x0 - r, x1 + r] for --roi-cols x0,x1, otherwise the
    binary's first and last POI x on the line +/- r; both clipped to the image.
    ``poi_columns`` are the centres of those subsets (the binary's POI columns that
    aggregate_cod keeps). None when no subset of the line enters the aggregation: the
    binary skips a line inside its margin, and --roi-cols may hold no POI column.
    """
    if not binary_computes_line(y, height, subset_radius):
        return None
    columns = roi_poi_columns(binary_poi_columns(width, subset_radius, step), roi_cols)
    if not columns:
        return None
    r = subset_radius
    left, right = (roi_cols[0] - r, roi_cols[1] + r) if roi_cols is not None else (
        columns[0] - r, columns[-1] + r)
    return {
        "rows": [max(0, y - r), min(height - 1, y + r)],
        "cols": [max(0, left), min(width - 1, right)],
        "poi_columns": columns,
    }


def measure_line_band(saturated, label: str, y: int, subset_radius: int, step: int,
                      roi_cols: tuple[int, int] | None) -> dict:
    """Saturation of one measurement line's band and of each of its subsets.

    ``saturated`` is the reference frame's boolean mask of pixels at 255. Fractions
    are None (and ``subsets`` 0) when no subset of the line enters the aggregation.
    """
    import numpy as np

    height, width = saturated.shape
    entry = {
        "label": label, "y": y, "band_rows": None, "band_cols": None,
        "band_fraction": None, "subsets": 0,
        "subsets_with_saturation": None, "subsets_over_10pct": None,
    }
    band = line_band(y, width, height, subset_radius, step, roi_cols)
    if band is None:
        return entry
    (top, bottom), (left, right) = band["rows"], band["cols"]
    region = saturated[top:bottom + 1, left:right + 1]
    # Pixels at 255 per subset: per-column counts over the subset rows, summed over
    # each centre's 2r + 1 columns (the binary margin keeps every subset inside).
    r = subset_radius
    cumulative = np.concatenate(([0], np.cumsum(np.count_nonzero(saturated[y - r:y + r + 1, :], axis=0))))
    centres = np.asarray(band["poi_columns"], dtype=np.int64)
    per_subset = cumulative[centres + r + 1] - cumulative[centres - r]
    area = (2 * r + 1) ** 2
    entry.update({
        "band_rows": [top, bottom],
        "band_cols": [left, right],
        "band_fraction": float(np.count_nonzero(region) / region.size),
        "subsets": int(centres.size),
        "subsets_with_saturation": float(np.count_nonzero(per_subset > 0) / centres.size),
        "subsets_over_10pct": float(
            np.count_nonzero(per_subset > SUBSET_SATURATION_CUT * area) / centres.size),
    })
    return entry


def saturation_report(saturated, container_bit_depth: int | None, pixel_max: int | None,
                      geometry: dict | None, threshold: float) -> dict:
    """events[].input_saturation: what input_8bit_max_255 is graded on, and its level.

    COD mode (basis line_bands): one entry per measurement line, value = the largest
    band fraction (None when no line has a band). Full-field mode (basis frame):
    value = the whole frame's fraction. severity is None when no pixel is 255 (then
    the warning is not raised).
    """
    import numpy as np

    frame_fraction = float(np.count_nonzero(saturated) / saturated.size)
    cod = bool(geometry and geometry.get("cod"))
    lines: list[dict] = []
    value: float | None = frame_fraction
    if cod:
        for label, y in zip(("L0", "L1"), geometry.get("line_ys") or ()):
            lines.append(measure_line_band(
                saturated, label, int(y), geometry["subset_radius"], geometry["step"],
                geometry.get("roi_cols"),
            ))
        measured = [line["band_fraction"] for line in lines if line["band_fraction"] is not None]
        value = max(measured) if measured else None
    return {
        "basis": "line_bands" if cod else "frame",
        "value": value,
        "threshold": threshold,
        "frame_fraction": frame_fraction,
        "severity": (input_saturation_severity(value, container_bit_depth, threshold)
                     if pixel_max == 255 else None),
        "container_bit_depth": container_bit_depth,
        "lines": lines,
    }


def probe_input_saturation(image_path: str, effective_depth: str,
                           geometry: dict | None = None,
                           threshold: float = DEFAULT_SATURATION_WARN_FRACTION) -> dict:
    """首帧最大值与 255 像素比例；8-bit legacy 对 uint16 使用高 8 位口径。

    返回 ``{"pixel_max", "saturated_fraction", "container_bit_depth",
    "input_saturation"}``，读不了时全为 None。``saturated_fraction``（整帧）与
    ``input_saturation``（saturation_report；``geometry`` 为 saturation_geometry，None
    即全场口径）只在有效位深为 8 时计算（=255 的像素，Pillow 灰度转换，与 binary 的
    OpenCV 转换可能差 1 个灰阶）。dry-run 与真跑都调用它，所以两边的结果相同。
    """
    empty = {"pixel_max": None, "saturated_fraction": None, "container_bit_depth": None,
             "input_saturation": None}
    probe = dict(empty)
    try:
        import numpy as np
        from PIL import Image

        with Image.open(image_path) as image:
            if len(image.getbands()) > 1:
                image = image.convert("L")
            pixels = np.asarray(image)
        if pixels.size == 0:
            return probe
        probe["container_bit_depth"] = int(pixels.dtype.itemsize * 8)
        if effective_depth == "8" and pixels.dtype.itemsize > 1:
            pixels = pixels.astype(np.uint64) >> 8
        probe["pixel_max"] = int(np.max(pixels))
        if effective_depth == "8":
            saturated = pixels == 255
            probe["saturated_fraction"] = float(np.count_nonzero(saturated) / pixels.size)
    except Exception:
        return dict(empty)
    if probe["saturated_fraction"] is not None:
        try:
            probe["input_saturation"] = saturation_report(
                saturated, probe["container_bit_depth"], probe["pixel_max"], geometry, threshold,
            )
        except Exception:
            # A heuristic must never fail the run: without the report the
            # warning, if any, is a warn with the 638ad18 text.
            probe["input_saturation"] = None
    return probe


def planned_effective_depth(input_depth: str | None, container_bit_depth: int | None) -> str | None:
    """The effective depth the binary will report for the reference frame (dry-run forecast).

    A fixed --input-depth is used as given; auto reads an 8-bit source as 8-bit and a
    deeper one as 12 or 16 by its values (oc_image.cpp inferAutoDepth), hence None.
    """
    requested = input_depth or "8"
    if requested != "auto":
        return requested
    if isinstance(container_bit_depth, int) and container_bit_depth <= 8:
        return "8"
    return None


def probe_input_pixel_max(image_path: str, effective_depth: str) -> int | None:
    """读取首帧最大值；8-bit legacy 对 uint16 使用高 8 位口径。"""
    return probe_input_saturation(image_path, effective_depth)["pixel_max"]


def input_saturation_severity(value: float | None,
                              container_bit_depth: int | None,
                              threshold: float = DEFAULT_SATURATION_WARN_FRACTION) -> str:
    """input_8bit_max_255 的级别：判据值（COD 为测线带最大占比，全场为整帧占比）不超过阈值
    且容器就是 8-bit 时为 info；判据未知（None）或容器更深时为 warn。"""
    if container_bit_depth is not None and container_bit_depth > 8:
        return SEVERITY_WARN  # a deeper container read as 8-bit: possible truncation
    if value is None:
        return SEVERITY_WARN
    return SEVERITY_WARN if value > threshold else SEVERITY_INFO


def describe_saturation_bands(report: dict) -> str:
    """The measurement-line bands of a line_bands report as one readable clause."""
    parts = []
    for line in report.get("lines") or []:
        if line["band_fraction"] is None:
            parts.append(f"{line['label']} y={line['y']}: no subset enters the COD")
            continue
        (top, bottom), (left, right) = line["band_rows"], line["band_cols"]
        parts.append(
            f"{line['label']} y={line['y']} rows {top}-{bottom} cols {left}-{right}: "
            f"{line['band_fraction']:.4%} at 255, subsets with a pixel at 255 "
            f"{line['subsets_with_saturation']:.1%}, with more than "
            f"{SUBSET_SATURATION_CUT:.0%} at 255 {line['subsets_over_10pct']:.1%}"
        )
    return ("measurement-line bands (the pixels of the subsets the COD uses): "
            + ("; ".join(parts) if parts else "none"))


def saturation_warning_detail(report: dict | None, container_bit_depth: int | None) -> str:
    """The warning_details detail of input_8bit_max_255: the numbers behind its level."""
    deep = container_bit_depth is not None and container_bit_depth > 8
    if report is None:
        if deep:
            return (f"the reference file is a {container_bit_depth}-bit container read as 8-bit; "
                    "possible high-bit truncation (check --input-depth)")
        return "possible saturation_or_high_bit_truncation"
    frame = f"{report['frame_fraction']:.4%}"
    bands = describe_saturation_bands(report) if report["basis"] == "line_bands" else ""
    if deep:
        # The level is warn whatever the fraction (input_saturation_severity):
        # say so, instead of quoting a threshold that did not decide it (R13).
        return (f"the reference file is a {container_bit_depth}-bit container read as 8-bit; "
                f"{frame} of its pixels are 255" + (f"; {bands}" if bands else "")
                + "; possible high-bit truncation (check --input-depth)")
    if report["basis"] == "frame":
        return (f"{frame} of reference pixels are 255 (warn above {report['threshold']:.2%}); "
                "possible saturation or high-bit truncation")
    if report["value"] is None:
        return (f"no measurement line has a subset in the COD aggregation, so no band was graded "
                f"({bands}); whole frame: {frame} of reference pixels are 255; possible "
                "saturation or high-bit truncation")
    return (f"largest measurement-line band {report['value']:.4%} at 255 (warn above "
            f"{report['threshold']:.2%}); {bands}; whole frame: {frame} of reference pixels "
            "are 255; possible saturation or high-bit truncation")


def unique_mode(values: list[int]) -> int | None:
    """返回唯一众数；并列时返回 None，避免伪造多数口径。"""
    if not values:
        return None
    frequencies = Counter(values)
    top = max(frequencies.values())
    winners = [value for value, count in frequencies.items() if count == top]
    return winners[0] if len(winners) == 1 else None


def collect_frame_stats(frame_dir: Path, output_format: str = "csv") -> dict:
    """纯聚合 frame 产物的质量统计，不重算任何 DIC 量。"""
    import numpy as np

    if output_format == "raw":
        return collect_raw_frame_stats(frame_dir)

    frame_files = sorted(
        f for f in frame_dir.iterdir()
        if f.name.startswith("frame_") and f.suffix == ".csv"
    )
    row_counts: list[int] = []
    valid_count = 0
    total_count = 0
    valid_zncc: list = []
    read_errors = 0

    for frame_file in frame_files:
        try:
            data = np.loadtxt(frame_file, delimiter=",", skiprows=1, dtype=np.float64)
            data = np.atleast_2d(data)
            if data.shape[1] < 5:
                raise ValueError("frame CSV has fewer than 5 columns")
        except Exception:
            read_errors += 1
            continue

        rows = int(data.shape[0])
        row_counts.append(rows)
        total_count += rows
        # 与 binary FrameQa / raw manifest 同口径：有效性只由 finite zncc>=0 定义。
        # 若要统计位移本身，调用方仍须额外过滤 finite(u,v)。
        valid = np.isfinite(data[:, 4]) & (data[:, 4] >= 0)
        valid_count += int(np.count_nonzero(valid))
        if np.any(valid):
            valid_zncc.append(data[valid, 4])

    poi_per_frame = unique_mode(row_counts)

    zncc_median = None
    if valid_zncc:
        zncc_median = float(np.median(np.concatenate(valid_zncc)))

    return {
        "frames": len(frame_files),
        "poi_per_frame": poi_per_frame,
        "poi_per_frame_min": min(row_counts) if row_counts else None,
        "poi_per_frame_max": max(row_counts) if row_counts else None,
        "valid_ratio": (valid_count / total_count) if total_count else None,
        "zncc_median": zncc_median,
        "frame_read_errors": read_errors,
    }


def collect_raw_frame_stats(frame_dir: Path) -> dict:
    """raw (.bin) 全场产物的同口径统计，走 fasticgn_io 的公开 reader。"""
    import numpy as np

    import fasticgn_io

    frame_files = sorted(
        f for f in frame_dir.iterdir()
        if f.name.startswith("frame_") and f.suffix == ".bin"
    )
    row_counts: list[int] = []
    valid_count = 0
    total_count = 0
    valid_zncc: list = []
    read_errors = 0

    for frame_file in frame_files:
        try:
            records = fasticgn_io.read_raw_frame(frame_file)
        except Exception:
            read_errors += 1
            continue
        rows = int(records.shape[0])
        row_counts.append(rows)
        total_count += rows
        valid = fasticgn_io.valid_mask(records)
        valid_count += int(np.count_nonzero(valid))
        if np.any(valid):
            valid_zncc.append(records["zncc"][valid].astype(np.float64))

    zncc_median = None
    if valid_zncc:
        zncc_median = float(np.median(np.concatenate(valid_zncc)))
    return {
        "frames": len(frame_files),
        "poi_per_frame": unique_mode(row_counts),
        "poi_per_frame_min": min(row_counts) if row_counts else None,
        "poi_per_frame_max": max(row_counts) if row_counts else None,
        "valid_ratio": (valid_count / total_count) if total_count else None,
        "zncc_median": zncc_median,
        "frame_read_errors": read_errors,
    }


def event_warnings(stats: dict, effective_depth: str, pixel_max: int | None,
                   stale_frames: bool = False,
                   saturated_fraction: float | None = None,
                   saturation_threshold: float = DEFAULT_SATURATION_WARN_FRACTION,
                   container_bit_depth: int | None = None,
                   saturation: dict | None = None) -> list[str]:
    """生成 M3 起步质量告警；字符串前缀是稳定的机器可读 code。

    级别不在这里定：见 WARNING_SEVERITY 与 input_saturation_severity。这里的文本
    是 warning_details 的 detail；summary 的 warnings 列表对 638ad18 已有的 code
    用当时的原文（LEGACY_WARNING_TEXT）。``saturation`` 是 probe_input_saturation
    的 input_saturation（判据与测线带）；只给 ``saturated_fraction`` 的旧调用按整帧描述。
    """
    warnings: list[str] = []
    valid_ratio = stats.get("valid_ratio")
    zncc_median = stats.get("zncc_median")
    if valid_ratio is None:
        warnings.append("frame_stats_unavailable")
    elif valid_ratio < 0.95:
        warnings.append(f"valid_ratio_below_0.95: {valid_ratio:.6f}")
    if zncc_median is None:
        warnings.append("zncc_median_unavailable")
    elif zncc_median < 0.9:
        warnings.append(f"zncc_median_below_0.9: {zncc_median:.6f}")
    if effective_depth == "8" and pixel_max == 255:
        if saturation is None and saturated_fraction is not None:
            # Callers of the earlier signature: the whole frame, as before.
            saturation = {"basis": "frame", "value": saturated_fraction,
                          "threshold": saturation_threshold,
                          "frame_fraction": saturated_fraction, "lines": []}
        warnings.append(
            "input_8bit_max_255: " + saturation_warning_detail(saturation, container_bit_depth)
        )
    if stats.get("frame_read_errors", 0):
        warnings.append(f"frame_csv_read_errors: {stats['frame_read_errors']}")
    if stale_frames:
        warnings.append("preexisting_frame_csv_detected: output may include stale frames")
    return warnings


def write_json_summary(path: Path, summary: dict) -> None:
    """原子写 UTF-8 JSON；禁止 NaN/Infinity 污染机器接口。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as tmp:
            tmp_name = tmp.name
            json.dump(summary, tmp, ensure_ascii=False, indent=2, allow_nan=False)
            tmp.write("\n")
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)


def warning_code(text: str) -> str:
    """Stable code of a warning string: the text before the first ':'."""
    return text.split(":", 1)[0].strip()


def add_warning(result: dict, code: str, detail: str | None = None,
                severity: str | None = None) -> str:
    """Append ``code[: detail]`` to result["warnings"]; ``severity`` overrides the table."""
    text = code if detail is None else f"{code}: {detail}"
    result.setdefault("warnings", []).append(text)
    if severity is not None:
        result.setdefault("_severity", {})[text] = severity
    return text


def warning_severity(text: str, overrides: dict | None = None, strict: bool = False) -> str:
    code = warning_code(text)
    level = (overrides or {}).get(text) or WARNING_SEVERITY.get(code, SEVERITY_WARN)
    if strict and code in STRICT_FAIL_CODES:
        level = SEVERITY_FAIL
    return level


def warning_details(warnings: list[str], overrides: dict | None = None,
                    strict: bool = False) -> list[dict]:
    """Every entry of a run's internal warning list: code, severity and full detail, same order."""
    details = []
    for text in warnings:
        _, sep, detail = text.partition(":")
        details.append({
            "code": warning_code(text),
            "severity": warning_severity(text, overrides, strict),
            "detail": detail.strip() if sep else None,
        })
    return details


def public_warnings(warnings: list[str], overrides: dict | None = None,
                    strict: bool = False) -> list[str]:
    """The summary's ``warnings`` list: warn and fail entries only, in 638ad18 wording.

    Codes 638ad18 already emitted keep its exact text (LEGACY_WARNING_TEXT for
    the two whose text changed since); newer codes keep ``code: detail``. Info
    entries and the extra detail exist only in ``warning_details``
    (review 2026-10-04 R2).
    """
    public = []
    for text in warnings:
        if warning_severity(text, overrides, strict) == SEVERITY_INFO:
            continue
        public.append(LEGACY_WARNING_TEXT.get(warning_code(text), text))
    return public


def precheck_record(plan: dict) -> list[dict]:
    """The --strict pre-check's warnings and notes, for the summary's top-level warning_details."""
    entries = []
    for item in list(plan.get("warnings") or []) + list(plan.get("notes") or []):
        if item.get("code") in PRECHECK_NOT_RECORDED:
            continue
        entry = {
            "code": item.get("code"),
            "severity": item.get("severity") or SEVERITY_WARN,
            "detail": item.get("message"),
            "stage": "strict_precheck",
        }
        if "event" in item:
            entry["event"] = item["event"]
        entries.append(entry)
    return entries


def precheck_public(entries: list[dict]) -> list[str]:
    """Warn/fail pre-check entries as ``code: message [event X]`` strings for ``warnings``."""
    return [
        f"{entry['code']}: {entry['detail']}" + (f" [event {entry['event']}]" if "event" in entry else "")
        for entry in entries if entry["severity"] != SEVERITY_INFO
    ]


def finalize_event(event: dict, strict: bool = False) -> dict:
    """Decide status from severity (idempotent); fill cod_csv only for a good result."""
    levels = [
        warning_severity(text, event.get("_severity"), strict)
        for text in event.get("warnings", [])
    ]
    failed = bool(event.get("_hard_fail")) or not event.get("ok", False) or SEVERITY_FAIL in levels
    if "_hard_fail" not in event:
        # First call: remember whether the event failed on its own, before any
        # severity rule, so that a later call can only add failures.
        event["_hard_fail"] = not event.get("ok", False)
    event["ok"] = not failed
    event["status"] = "fail" if failed else "ok"
    written = event.get("_cod_written")
    if written is not None:
        event["cod_csv"] = written if not failed else None
    return event


def build_run_summary(args: argparse.Namespace, binary: dict, binary_warnings: list[str],
                      events: list[dict], wall_time_s: float,
                      precheck: list[dict] | None = None,
                      event_selection: dict | None = None) -> dict:
    """汇总 campaign/single 结果，并追加跨事件帧数众数告警。

    verdict 只看 warning_details 的级别：任一 fail 级（或失败事件）→ fail；否则
    任一 warn 级 → warn；否则 ok。info 级只记录、不影响 verdict。``warnings``
    列表只放 warn/fail 条目（638ad18 的措辞），见 public_warnings。``precheck``
    是 --strict 预检的条目，进顶层 warning_details（warn 级也进顶层 warnings）。
    ``event_selection``（event_selection_public）是 campaign 的事件筛选与逐事件界面行，
    单事件为 null（2026-10-05 追加）。
    """
    strict = bool(getattr(args, "strict", False))
    for event in events:
        finalize_event(event, strict)
    public_events: list[dict] = []
    frame_counts = [
        event["frames"] for event in events
        if event.get("ok") and isinstance(event.get("frames"), int)
    ]
    frame_mode = unique_mode(frame_counts)

    levels: list[str] = []
    for event in events:
        public = {
            key: value for key, value in event.items()
            if key != "ok" and not key.startswith("_")
        }
        internal = list(public.get("warnings", []))
        if event.get("ok") and frame_mode is not None and event.get("frames") != frame_mode:
            internal.append(
                f"frame_count_differs_from_mode: {event.get('frames')} != {frame_mode}"
            )
        public["warnings"] = public_warnings(internal, event.get("_severity"), strict)
        public["warning_details"] = warning_details(internal, event.get("_severity"), strict)
        levels.extend(item["severity"] for item in public["warning_details"])
        public_events.append(public)

    precheck = list(precheck or [])
    top_details = warning_details(list(binary_warnings), None, strict) + precheck
    levels.extend(item["severity"] for item in top_details)
    events_ok = sum(bool(event.get("ok")) for event in events)
    events_failed = len(events) - events_ok
    if events_failed or SEVERITY_FAIL in levels:
        verdict = "fail"
    elif SEVERITY_WARN in levels:
        verdict = "warn"
    else:
        verdict = "ok"
    return {
        "schema_version": 1,
        "params": summary_params(args),
        "binary": binary,
        "events": public_events,
        "global": {
            "wall_time_s": round(float(wall_time_s), 6),
            "events_ok": events_ok,
            "events_failed": events_failed,
        },
        "warnings": public_warnings(list(binary_warnings), None, strict) + precheck_public(precheck),
        "warning_details": top_details,
        "verdict": verdict,
        "event_selection": event_selection,
    }


def build_failure_summary(args: argparse.Namespace, message: str,
                          wall_time_s: float = 0.0,
                          extra_messages: list[str] | None = None,
                          precheck: list[dict] | None = None) -> dict:
    """为解析参数后的 preflight/runtime 异常保留机器可读失败结果。"""
    warnings = [f"pipeline_error: {message}"]
    warnings += [f"pipeline_error: {item}" for item in (extra_messages or [])]
    precheck = list(precheck or [])
    return {
        "schema_version": 1,
        "params": summary_params(args),
        "binary": {
            "path": str(Path(args.gpu_bin).resolve()),
            "git_hash": None,
            "build_date": None,
        },
        "events": [],
        "global": {
            "wall_time_s": round(float(wall_time_s), 6),
            "events_ok": 0,
            "events_failed": 1,
        },
        "warnings": warnings + precheck_public(precheck),
        "warning_details": warning_details(warnings) + precheck,
        "verdict": "fail",
    }


def event_frame_dir(output_dir: Path, event_name: str,
                    args: argparse.Namespace) -> Path:
    """COD 模式沿用隐藏中间目录 `_frames_<event>`；全场模式直接用 `<event>/`。"""
    if is_cod_mode(args):
        return output_dir / f"_frames_{event_name}"
    return output_dir / event_name


def event_output_csv(output_dir: Path, event_name: str,
                     args: argparse.Namespace) -> Path | None:
    """COD CSV 路径；全场模式没有聚合产物，返回 None。"""
    if not is_cod_mode(args):
        return None
    return output_dir / (f"{event_name}.csv" if args.photo_dir else "cod_result.csv")


def event_binary_log(output_dir: Path, event_name: str,
                     args: argparse.Namespace) -> Path:
    """binary 完整 stderr 的日志路径：帧目录旁边（不在里面，binary 会拒绝陌生文件）。"""
    frame_dir = event_frame_dir(output_dir, event_name, args)
    return output_dir / f"{frame_dir.name}{BINARY_LOG_SUFFIX}"


def effective_geometry(args: argparse.Namespace, subset_radius: int,
                       line_ys: tuple[int, int] | None,
                       roi_cols: tuple[int, int] | None,
                       interface: dict | None = None) -> dict:
    """实际生效的几何：与 aggregate_cod 用的同一套 L0/L1、y0 = (L0+L1)/2。

    ``interface``（event_interface 的结果）给出该事件的界面行与来源；不给时按
    --interface-y。interface_y* 三个字段是 2026-10-05 的追加字段。
    """
    geometry = {
        "subset_size": args.subset,
        "subset_radius": subset_radius,
        "step": args.step,
        "line_ys": None,
        "line_source": None,
        "line_separation_px": None,
        "gauge_mm": None,
        "extrapolation_target_y": None,
        "extrapolation_px": None,
        "roi_cols": list(roi_cols) if roi_cols is not None else None,
        "interface_y": None,
        "interface_y_source": None,
        "interface_y_overrides": None,
    }
    if line_ys is None:
        return geometry
    l0_y, l1_y = line_ys
    geometry["line_ys"] = [l0_y, l1_y]
    geometry["line_source"] = "line_y" if args.line_y else "interface_y"
    if not args.line_y:
        if interface is None:
            interface = event_interface(None, args, "")
        if interface is not None:
            geometry["interface_y"] = interface["interface_y"]
            geometry["interface_y_source"] = interface["source"]
            geometry["interface_y_overrides"] = list(interface["overrides"])
    geometry["line_separation_px"] = abs(l1_y - l0_y)
    cal = getattr(args, "cal", None)
    if isinstance(cal, (int, float)) and math.isfinite(cal):
        geometry["gauge_mm"] = abs(l1_y - l0_y) * cal
    if getattr(args, "cod_method", "naive") == "fillboundary":
        y0 = 0.5 * (l0_y + l1_y)
        geometry["extrapolation_target_y"] = y0
        geometry["extrapolation_px"] = [y0 - l0_y, y0 - l1_y]
    return geometry


def file_signature(path: Path | None) -> tuple | None:
    """(inode, mtime_ns, size) of an existing regular file, else None."""
    if path is None:
        return None
    try:
        info = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    return (info.st_ino, info.st_mtime_ns, info.st_size)


def describe_mtime(path: Path) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(path.stat().st_mtime))
    except OSError:
        return "unknown time"


def make_event_result(event_name: str, output_csv: Path | None,
                      args: argparse.Namespace,
                      field_dir: Path | None = None) -> dict:
    """事件结果骨架。COD 模式填 cod_mode；全场模式不聚合。

    ``cod_csv`` 只在本次运行确实写出、且事件状态为 ok 时才填（见
    finalize_event）。``field_dir`` 与 638ad18 相同，总是这次运行的帧目录（COD 模式
    聚合后通常已删除）；目录是否还在看新增的 ``field_dir_exists``。
    """
    cod = is_cod_mode(args)
    return {
        "ok": False,
        "status": "fail",
        "name": event_name,
        "field_mode": field_mode_name(args),
        "frames": 0,
        "poi_per_frame": None,
        "poi_per_frame_min": None,
        "poi_per_frame_max": None,
        "valid_ratio": None,
        "zncc_median": None,
        "cod_csv": None,
        "cod_method": args.cod_method if cod else None,
        # review-2026-09-16 §4: the aggregation actually used must be machine
        # readable, so a naive fallback can never masquerade as fill-boundary.
        "cod_mode": args.cod_method if cod else None,
        "field_dir": str(field_dir.resolve()) if field_dir is not None else None,
        "field_dir_exists": field_dir is not None and field_dir.is_dir(),
        "output_format": getattr(args, "output_format", "csv"),
        "input_pixel_max": None,
        "input_reference": None,
        "warnings": [],
        # Additive fields (review-2026-10-03); see interface_contract.json.
        "input_saturated_fraction": None,
        # Additive (2026-10-04): what input_8bit_max_255 is graded on.
        "input_saturation": None,
        "target_frames_available": None,
        "target_frames_selected": None,
        "geometry": None,
        "cod_stats": None,
        "binary_exit_code": None,
        "binary_stderr_log": None,
        "binary_totals": None,
        "remediation": None,
    }


def probe_image(image_path: str) -> dict:
    """只读首帧尺寸/像素探针；Pillow 保留 uint16 TIFF 原始载荷。"""
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
        mode = image.mode
        pixels = np.asarray(image)
    if pixels.size == 0:
        raise ValueError("image has no pixels")

    unsigned = pixels.astype(np.uint64, copy=False)
    bit_depth = int(pixels.dtype.itemsize * 8)
    low_4_bits_all_zero = bool(
        bit_depth > 8 and np.all(np.bitwise_and(unsigned, 0xF) == 0)
    )
    return {
        "path": str(Path(image_path).resolve()),
        "width": int(width),
        "height": int(height),
        "pillow_mode": mode,
        "dtype": str(pixels.dtype),
        "container_bit_depth": bit_depth,
        "pixel_min": int(np.min(pixels)),
        "pixel_max": int(np.max(pixels)),
        "low_4_bits_all_zero": low_4_bits_all_zero,
    }


def recommend_input_mode(image_probe: dict) -> dict:
    """按 image_input_depth_probe 的可观察量给出诚实的输入模式建议。"""
    bit_depth = image_probe["container_bit_depth"]
    pixel_max = image_probe["pixel_max"]
    low4 = image_probe["low_4_bits_all_zero"]
    limitations: list[str] = []

    if bit_depth <= 8:
        depth, align = "8", None
        reason = "8-bit container"
    elif pixel_max <= 4095:
        depth, align = "12", "right"
        reason = "uint16 container with observed max <= 4095"
        if low4:
            limitations.append(
                "dark left-shifted 12-bit data can also have max <= 4095; "
                "this probe cannot disambiguate it from right-aligned 12-bit"
            )
    elif low4:
        depth, align = "12", "left"
        reason = "uint16 payload has all low 4 bits zero and max > 4095"
    else:
        depth, align = "16", None
        reason = "uint16 payload uses values above 4095 without a low-4-bit zero pattern"

    suggested_args = ["--input-depth", depth]
    if align:
        suggested_args += ["--input-align", align]
    return {
        "input_depth": depth,
        "input_align": align,
        "suggested_args": suggested_args,
        "reason": reason,
        "limitations": limitations,
        "auto_limitation": (
            "--input-depth auto classifies left-shifted 12-bit data with max > 4095 "
            "as 16-bit; such data (uint16, low 4 bits always zero) needs explicit "
            "--input-depth 12 --input-align left"
        ),
    }


class CodAggregationError(Exception):
    """COD 聚合的机器可读拒绝；``code`` 进 summary warnings。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _issue(code: str, message: str, event: str | None = None,
           severity: str | None = None) -> dict:
    item = {"code": code, "message": message}
    if event is not None:
        item["event"] = event
    if severity is not None:
        item["severity"] = severity
    return item


def validate_args(args: argparse.Namespace, strict: bool = False) -> dict:
    """不读图像的参数校验：dry-run、--strict 和正常执行共用（F4/F6/F14）。

    返回 ``{"errors", "warnings", "roi_cols", "line_ys", "selection"}``。errors 的 code 与
    dry-run 一致；正常执行默认只把它们降级成 ``argument_warning``，--strict
    下按 dry-run 的退出码 2 拒绝。例外：EVENT_SELECTION_CODES（--events、
    --event-interface-y）在任何模式下都以退出码 2 拒绝。
    """
    errors: list[dict] = []
    warnings: list[dict] = []
    cod_mode = is_cod_mode(args)

    source = getattr(args, "gpu_bin_source", None)
    if source == "unresolved":
        errors.append(_issue(
            "gpu_binary_unresolved", binary_search_description(GPU_BINARY_NAME),
        ))
    elif strict and source == "legacy_lab_path":
        errors.append(_issue(
            "gpu_bin_legacy_lab_path",
            f"--strict refuses the deprecated hard-coded lab path {args.gpu_bin}; pass "
            f"--gpu-bin or set {ENV_GPU_BIN}",
        ))

    if args.subset < 5 or args.subset > 51 or args.subset % 2 == 0:
        errors.append(_issue(
            "invalid_subset_size",
            f"--subset must be an odd SIZE in [5, 51]; got {args.subset}, which runs as "
            f"radius {(args.subset - 1) // 2}",
        ))
    if args.step <= 0:
        errors.append(_issue("invalid_step", f"--step must be > 0; got {args.step}"))
    if cod_mode and not args.line_y and args.gauge <= 0:
        errors.append(_issue("invalid_gauge", f"--gauge must be > 0; got {args.gauge}"))
    if cod_mode:
        if args.cal is None:
            errors.append(_issue(
                "missing_cal", "--cal is required in COD mode (--interface-y/--line-y)",
            ))
        elif not math.isfinite(args.cal) or args.cal <= 0:
            errors.append(_issue(
                "invalid_cal",
                f"--cal must be finite and > 0; got {args.cal} (a negative value flips the "
                "sign of cod_um and x_mm, zero makes every COD 0)",
            ))
        if args.output_format != "csv":
            errors.append(_issue(
                "cod_requires_csv_frames",
                "COD aggregation reads frame CSV; --output-format raw is full-field only",
            ))
    # Full-field mode never uses --cal, so its value is not checked there
    # (review 2026-10-04 R3; 638ad18 ran such a command).
    try:
        compile_image_pattern(args.image_pattern)
    except re.error as exc:
        errors.append(_issue("invalid_image_pattern", f"--image-pattern is not a valid regex: {exc}"))
    if not math.isfinite(args.conv_threshold) or args.conv_threshold <= 0:
        errors.append(_issue(
            "invalid_conv_threshold",
            f"--conv-threshold must be finite and > 0; got {args.conv_threshold}",
        ))
    if args.gpu_device < 0:
        errors.append(_issue("invalid_gpu_device", f"--gpu-device must be >= 0; got {args.gpu_device}"))
    if args.input_align is not None and args.input_depth != "12":
        errors.append(_issue(
            "invalid_input_alignment",
            "--input-align is only valid with --input-depth 12; auto + explicit align is rejected",
        ))
    if not Path(args.gpu_bin).is_file():
        errors.append(_issue("gpu_binary_missing", f"GPU binary does not exist: {args.gpu_bin}"))
    elif not os.access(args.gpu_bin, os.X_OK):
        errors.append(_issue("gpu_binary_not_executable", f"GPU binary is not executable: {args.gpu_bin}"))

    try:
        roi_cols = parse_roi_cols(args.roi_cols)
    except ValueError as exc:
        roi_cols = None
        errors.append(_issue("invalid_roi_cols", str(exc)))
    if roi_cols is not None and (roi_cols[0] > roi_cols[1] or roi_cols[0] < 0):
        errors.append(_issue(
            "roi_cols_out_of_bounds",
            f"ROI [{roi_cols[0]}, {roi_cols[1]}] must be ordered (x1 <= x2) and non-negative; "
            "a reversed ROI selects no column, so the COD CSV would be empty",
        ))
    try:
        line_ys: tuple[int, int] | None = resolve_line_ys(args)
    except ValueError as exc:
        line_ys = None
        errors.append(_issue("invalid_line_y", str(exc)))
    if line_ys is not None and line_ys[0] == line_ys[1]:
        errors.append(_issue(
            "line_y_not_distinct",
            f"L0 and L1 must be distinct; both are {line_ys[0]} (with --interface-y this is "
            "--gauge 0 or 1), so every COD would be 0",
        ))
    if args.line_y and args.interface_y is not None:
        warnings.append(_issue(
            "line_y_overrides_interface_y",
            "--line-y overrides --interface-y and --gauge; the fill-boundary target is the "
            "midpoint of the two --line-y values",
            severity=SEVERITY_INFO,
        ))
    # --events / --event-interface-y: lists the campaign's directories, opens no image.
    selection = plan_event_selection(args, campaign_event_names(args))
    errors.extend(selection["errors"])
    return {"errors": errors, "warnings": warnings, "roi_cols": roi_cols, "line_ys": line_ys,
            "selection": selection}


# Codes normal execution has always enforced itself (exit 1), so the default
# mode keeps them as hard errors with their historical messages; every other
# validate_args error is reported as an argument_warning unless --strict.
LEGACY_ENFORCED_CODES = frozenset({
    "gpu_binary_unresolved", "missing_cal", "cod_requires_csv_frames",
    "invalid_image_pattern", "invalid_roi_cols", "invalid_line_y",
    "gpu_binary_missing", "gpu_binary_not_executable",
})


def describe_output_dir(path: Path) -> dict:
    """--output-dir 的现状：是否存在、是否目录、能否写入（不存在时看最近的已存在祖先）。"""
    exists = path.exists()
    is_directory = path.is_dir()
    nearest = path
    if not exists:
        nearest = path.parent
        while not nearest.exists() and nearest != nearest.parent:
            nearest = nearest.parent
    writable = nearest.is_dir() and os.access(nearest, os.W_OK | os.X_OK)
    return {
        "path": str(path.resolve()),
        "exists": exists,
        "is_directory": is_directory,
        "writable": bool(writable) if (is_directory or not exists) else False,
        "nearest_existing_ancestor": None if exists else str(nearest.resolve()),
    }


def ledger_target_files(frame_dir: Path) -> list[str] | None:
    """Frame file names an existing ledger contract lists, in order (None if unreadable)."""
    try:
        contract = json.loads((frame_dir / ".opencorr-state" / "contract.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    targets = contract.get("targets") if isinstance(contract, dict) else None
    if not isinstance(targets, list):
        return None
    names = [target.get("file") if isinstance(target, dict) else None for target in targets]
    return names if all(isinstance(name, str) for name in names) else None


def describe_event_output(frame_dir: Path, output_csv: Path | None, log_path: Path,
                          planned_files: list[str] | None = None) -> dict:
    """事件输出位置的现状，以及 binary 面对它时会怎么做（只看文件名，不比内容摘要）。

    有 ledger 时再比较目标文件名：binary 只接受与原列表相同、或在原列表末尾
    追加的目标列表（review 2026-10-04 R16：--max-frames 缩短已有运行会被拒绝）。
    """
    state = {
        "frame_dir": str(frame_dir.resolve()),
        "frame_dir_exists": frame_dir.exists(),
        "frame_dir_is_directory": frame_dir.is_dir(),
        "frame_dir_entries": None,
        "ledger_contract": False,
        "ledger_complete_record": False,
        "untracked_entries": 0,
        "binary_expectation": "fresh",
        "cod_csv_exists": output_csv.exists() if output_csv is not None else None,
        "binary_stderr_log_exists": log_path.exists(),
    }
    if not frame_dir.is_dir():
        if frame_dir.exists():
            state["binary_expectation"] = "fail_not_a_directory"
        return state
    entries = sorted(frame_dir.iterdir())
    state["frame_dir_entries"] = len(entries)
    ledger = frame_dir / ".opencorr-state"
    state["ledger_contract"] = (ledger / "contract.json").is_file()
    state["ledger_complete_record"] = (ledger / "complete.json").is_file()
    untracked = [
        entry for entry in entries
        if entry.name != "tar_list.txt"
        and not entry.name.startswith(".opencorr-tmp-")
        and not (entry == ledger and entry.is_dir() and not any(entry.iterdir()))
    ]
    if state["ledger_contract"]:
        state["binary_expectation"] = "resume_if_unchanged"
        recorded = ledger_target_files(frame_dir)
        if recorded is not None and planned_files is not None:
            state["ledger_targets"] = len(recorded)
            if planned_files[:len(recorded)] != recorded:
                state["binary_expectation"] = "reject_targets_changed"
    elif untracked:
        state["untracked_entries"] = len(untracked)
        state["binary_expectation"] = "reject_untracked"
    return state


def duplicate_frame_numbers(images: list[tuple[int, str]]) -> list[int]:
    counts = Counter(number for number, _ in images)
    return sorted(number for number, count in counts.items() if count > 1)


def select_targets(images: list[tuple[int, str]],
                   max_frames: int | None) -> tuple[list[tuple[int, str]], int]:
    """参考帧之后的目标帧；--max-frames 只保留前 N 个。返回 (选中的, 原有数)。"""
    ref_no = images[0][0]
    targets = [(frame_no, path) for frame_no, path in images if frame_no != ref_no]
    available = len(targets)
    if max_frames is not None:
        targets = targets[:max_frames]
    return targets, available


# ── Dry-run runtime / disk estimates ────────────────────────────────────────
# Calibrated on the lab host (2 x RTX 6000 Ada, idle, warm page cache) on
# 2026-10-03 with the binary git_hash 638ad1845d01: pipeline end to end (Python
# start-up, binary, frame parsing, aggregation, CSV writing, and the
# --json-summary statistics as a separate per-row term), event10 (1036 x 186,
# subset 15, step 1). runtime ~= startup_s + rows * per_row_s, rows = target
# frames x POI per frame (2036 for two COD lines, 171024 full field). Measured:
# COD 100 / 800 frames 0.97-1.05 / 4.73-6.77 s; full-field CSV 10 / 30 frames
# 1.78-1.80 / 4.21-4.22 s (2.25 s without --json-summary); raw 30 / 90 frames
# 0.84-0.91 / 1.35-1.37 s (0.94 s without). A cold page cache on a busy host made
# one 800-frame COD run take 18.1 s, 3.2x the warm time (review 2026-10-03),
# hence the upper range factor of 4.
ESTIMATE_CALIBRATION = {
    "date": "2026-10-03",
    "binary_git_hash": "638ad1845d01",
    "host": "lab host (2 x RTX 6000 Ada), idle, warm page cache",
    "range_factors": [0.5, 4.0],
    "modes": {
        "cod": {"startup_s": 0.34, "per_row_s": 2.8e-6, "summary_per_row_s": 0.5e-6},
        "full_field_csv": {"startup_s": 0.57, "per_row_s": 3.3e-7, "summary_per_row_s": 3.8e-7},
        "full_field_raw": {"startup_s": 0.63, "per_row_s": 2.0e-8, "summary_per_row_s": 2.7e-8},
    },
    "bytes_per_frame_row": {"cod": 86, "full_field_csv": 68, "full_field_raw": 36},
    "bytes_per_cod_row": 21.2,
}


def estimate_mode(cod_mode: bool, output_format: str) -> str:
    if cod_mode:
        return "cod"
    return "full_field_raw" if output_format == "raw" else "full_field_csv"


def build_estimates(mode: str, total_targets: int, total_rows: int,
                    with_summary: bool, peak_bytes: int, retained_bytes: int) -> dict:
    model = ESTIMATE_CALIBRATION["modes"][mode]
    per_row = model["per_row_s"] + (model["summary_per_row_s"] if with_summary else 0.0)
    runtime = model["startup_s"] + total_rows * per_row if total_targets else 0.0
    low, high = ESTIMATE_CALIBRATION["range_factors"]
    return {
        "rough": True,
        "basis": (
            f"per-mode linear model ({mode}): {model['startup_s']} s start-up + "
            f"{per_row:.3g} s per frame row (target frames x POI per frame), measured "
            f"end to end on {ESTIMATE_CALIBRATION['date']} with binary "
            f"{ESTIMATE_CALIBRATION['binary_git_hash']} ({ESTIMATE_CALIBRATION['host']}); "
            "a cold page cache or a busy host can take several times longer"
        ),
        "mode": mode,
        "target_frames": total_targets,
        "frame_rows": total_rows,
        "runtime_s": round(runtime, 3),
        "runtime_s_range": [round(runtime * low, 3), round(runtime * high, 3)],
        "calibration": ESTIMATE_CALIBRATION,
        "peak_output_bytes": peak_bytes,
        "retained_output_bytes": retained_bytes,
    }


def build_dry_run(args: argparse.Namespace) -> dict:
    """构造不写磁盘、不启动 GPU 的校验与执行计划。"""
    errors: list[dict] = []
    warnings: list[dict] = []
    output_dir = Path(args.output_dir)
    subset_radius = (args.subset - 1) // 2
    cod_mode = is_cod_mode(args)
    strict = bool(getattr(args, "strict", False))
    max_frames = getattr(args, "max_frames", None)

    for note in getattr(args, "path_resolution_notes", []) or []:
        warnings.append(_issue("path_resolution", note, severity=SEVERITY_WARN))
    source = getattr(args, "gpu_bin_source", None)
    if source not in (None, "unresolved"):
        note = binary_source_note(args.gpu_bin, source)
        if note:
            warnings.append(_issue("gpu_bin_outside_checkout", note, severity=SEVERITY_INFO))

    checks = validate_args(args, strict=strict)
    errors.extend(checks["errors"])
    warnings.extend(checks["warnings"])
    roi_cols = checks["roi_cols"]
    line_ys = checks["line_ys"]
    selection = checks["selection"]
    roi_ordered = roi_cols is not None and 0 <= roi_cols[0] <= roi_cols[1]
    selected_names = (
        set(selection["selected"])
        if selection["patterns"] is not None and selection["selected"] is not None else None
    )

    if args.json_summary:
        warnings.append(_issue(
            "json_summary_ignored",
            "--dry-run already writes JSON to stdout; --json-summary is not written",
            severity=SEVERITY_INFO,
        ))

    output_state = describe_output_dir(output_dir)
    if output_state["exists"] and not output_state["is_directory"]:
        errors.append(_issue(
            "output_dir_not_directory", f"--output-dir exists and is not a directory: {output_dir}",
        ))
    elif not output_state["writable"]:
        errors.append(_issue(
            "output_dir_not_writable",
            f"--output-dir cannot be written or created: {output_dir}",
        ))

    mode = "campaign" if args.photo_dir else "single_event"
    source_root = Path(args.photo_dir or args.image_dir)
    source_summary = {
        "mode": mode,
        "path": str(source_root.resolve()),
        "event_subdirectories": None,
        "events_with_images": 0,
    }
    event_sources: list[tuple[str, Path, list[tuple[int, str]]]] = []

    if not source_root.is_dir():
        errors.append(_issue("input_directory_missing", f"input directory does not exist: {source_root}"))
    elif args.photo_dir:
        subdirectories = sorted(entry for entry in source_root.iterdir() if entry.is_dir())
        source_summary["event_subdirectories"] = len(subdirectories)
        discovered = 0
        for entry in subdirectories:
            # --events: only the selected events are planned; a subdirectory the
            # selection leaves out is not reported (2026-10-05).
            if selection["patterns"] is not None and not any(
                    event_matches(entry.name, pattern) for pattern in selection["patterns"]):
                if find_images_for(args, entry):
                    discovered += 1
                continue
            images = find_images_for(args, entry)
            if images:
                discovered += 1
                if selected_names is None or entry.name in selected_names:
                    event_sources.append((entry.name, entry, images))
            else:
                warnings.append(_issue(
                    "event_without_images", "event subdirectory contains no recognized images",
                    entry.name, severity=SEVERITY_WARN,
                ))
        if not discovered:
            errors.append(_issue(
                "no_event_images", "campaign contains no event subdirectory with recognized images",
            ))
    else:
        source_summary["event_subdirectories"] = 0
        images = find_images_for(args, source_root) if source_root.is_dir() else []
        if images:
            event_sources.append((source_root.name, source_root, images))
        elif source_root.is_dir():
            errors.append(_issue("no_images", "single-event directory contains no recognized images"))
    # Events found under the input, before --events (event_selection has the selected count).
    source_summary["events_with_images"] = discovered if args.photo_dir and source_root.is_dir() \
        else len(event_sources)

    event_plans: list[dict] = []
    total_targets = 0
    total_rows = 0
    estimated_peak_bytes = 0
    estimated_retained_bytes = 0
    frame_row_bytes = ESTIMATE_CALIBRATION["bytes_per_frame_row"][
        estimate_mode(cod_mode, args.output_format)
    ]

    for event_name, image_dir, images in event_sources:
        # This event's interface row and lines (--event-interface-y may move them).
        interface = event_interface(selection, args, event_name)
        ev_line_ys = event_line_ys(args, interface, line_ys)
        if len(images) < 2:
            errors.append(_issue(
                "insufficient_images", f"event needs at least 2 images, found {len(images)}", event_name,
            ))

        ref_no, ref_path = images[0]
        duplicates = duplicate_frame_numbers(images)
        target_duplicates = [number for number in duplicates if number != ref_no]
        if target_duplicates:
            errors.append(_issue(
                "duplicate_frame_number",
                f"{len(target_duplicates)} target frame number(s) occur more than once (first: "
                f"{target_duplicates[:5]}); duplicate targets make the binary reject the output "
                "names. Check --image-pattern and files such as macOS '._*' copies",
                event_name,
            ))
        if ref_no in duplicates:
            # The run succeeds: the extra images with the reference number are
            # never handed to the binary (review 2026-10-04 R4).
            same = [Path(path).name for number, path in images if number == ref_no]
            warnings.append(_issue(
                "duplicate_reference_frame_number",
                f"{len(same)} images carry the reference frame number {ref_no} ({same[:3]}); the "
                f"first in name order, {same[0]}, is the reference and the others are not used. "
                "Check --image-pattern and files such as macOS '._*' copies",
                event_name, severity=SEVERITY_WARN,
            ))
        target_entries, available_targets = select_targets(images, max_frames)
        if max_frames is not None and available_targets > len(target_entries):
            warnings.append(_issue(
                "max_frames_applied",
                f"--max-frames {max_frames}: only the first {len(target_entries)} of "
                f"{available_targets} target frames are planned",
                event_name, severity=SEVERITY_INFO,
            ))
        total_targets += len(target_entries)
        probe = None
        recommendation = None
        try:
            probe = probe_image(ref_path)
            recommendation = recommend_input_mode(probe)
        except Exception as exc:
            errors.append(_issue("image_probe_failed", f"cannot probe first frame: {exc}", event_name))
        # The run's input_saturation, from the same probe and geometry (its level
        # is the one input_8bit_max_255 will get when the run reports it).
        input_saturation = None
        if probe is not None and planned_effective_depth(
                args.input_depth, probe.get("container_bit_depth")) == "8":
            input_saturation = probe_input_saturation(
                ref_path, "8", saturation_geometry(args, subset_radius, ev_line_ys, roi_cols),
                getattr(args, "saturation_warn_fraction", DEFAULT_SATURATION_WARN_FRACTION),
            )["input_saturation"]

        if probe is not None:
            width, height = probe["width"], probe["height"]
            margin = binary_poi_margin(subset_radius)
            if width <= 2 * margin:
                errors.append(_issue(
                    "image_too_narrow_for_subset",
                    f"image width {width} leaves no POI after binary margin {margin}", event_name,
                ))
            if interface is not None and not 0 <= interface["interface_y"] < height:
                origin = (
                    f"--interface-y {interface['interface_y']}"
                    if interface["source"] == "interface_y"
                    else f"--event-interface-y {', '.join(interface['overrides'])}"
                )
                errors.append(_issue(
                    "interface_y_out_of_bounds",
                    f"{origin} is outside image height {height}", event_name,
                ))
            if ev_line_ys is not None:
                for label, y in zip(("L0", "L1"), ev_line_ys):
                    if not 0 <= y < height:
                        errors.append(_issue(
                            "line_y_out_of_bounds", f"{label}={y} is outside image height {height}", event_name,
                        ))
                    elif not binary_computes_line(y, height, subset_radius):
                        errors.append(_issue(
                            "line_y_inside_binary_margin",
                            f"{label}={y} violates binary margin {margin} for subset radius {subset_radius}",
                            event_name,
                        ))
            if roi_ordered and roi_cols[1] >= width:
                errors.append(_issue(
                    "roi_cols_out_of_bounds",
                    f"ROI [{roi_cols[0]}, {roi_cols[1]}] must lie inside image width {width}",
                    event_name,
                ))

            if recommendation is not None:
                requested_depth = args.input_depth or "8"
                requested_align = args.input_align
                container = probe.get("container_bit_depth")
                if strict and requested_depth == "8" and isinstance(container, int) and container > 8:
                    # The project's first pitfall: OpenCV converts the deeper file
                    # to 8 bits before the binary sees it, silently (review
                    # 2026-10-04 R6a). Only --strict refuses; MCP always uses it.
                    suggestion = " ".join(recommendation["suggested_args"])
                    mcp_form = ", ".join(
                        f"{key}='{recommendation[key]}'" for key in ("input_depth", "input_align")
                        if recommendation[key] is not None
                    )
                    errors.append(_issue(
                        "input_depth_truncates_container",
                        f"the first frame is a {container}-bit container ({probe.get('dtype')}, max "
                        f"{probe.get('pixel_max')}) but would be read as 8-bit"
                        + ("" if args.input_depth else " (the default --input-depth 8)")
                        + f", which keeps only its high bits; the probe recommends {suggestion} "
                        f"({recommendation['reason']}; MCP: {mcp_form}). --strict refuses this; "
                        "without --strict it runs with a warning",
                        event_name,
                    ))
                if (
                    requested_depth != recommendation["input_depth"]
                    or (
                        recommendation["input_align"] is not None
                        and requested_align != recommendation["input_align"]
                    )
                ):
                    warnings.append(_issue(
                        "input_mode_differs_from_probe",
                        "requested input mode differs from first-frame probe recommendation: "
                        + " ".join(recommendation["suggested_args"]),
                        event_name, severity=SEVERITY_WARN,
                    ))

            x_positions = binary_poi_columns(width, subset_radius, args.step)
            y_rows = (
                len(ev_line_ys) if ev_line_ys is not None
                else len(range(margin, max(margin, height - margin), max(args.step, 1)))
            )
            poi_rows = len(x_positions) * y_rows
            cod_x_positions = roi_poi_columns(x_positions, roi_cols)
            if roi_ordered and not cod_x_positions:
                errors.append(_issue(
                    "roi_has_no_poi_columns",
                    "ROI is inside the image but contains no planned POI x coordinate",
                    event_name,
                ))
            total_rows += len(target_entries) * poi_rows
            frame_bytes = len(target_entries) * (29 + poi_rows * frame_row_bytes)
            cod_bytes = (
                int(22 + len(target_entries) * len(cod_x_positions)
                    * ESTIMATE_CALIBRATION["bytes_per_cod_row"])
                if cod_mode else 0
            )
            estimated_peak_bytes += frame_bytes + cod_bytes
            retained_frames = args.keep_frames or not cod_mode
            estimated_retained_bytes += cod_bytes + (frame_bytes if retained_frames else 0)

        frame_dir = event_frame_dir(output_dir, event_name, args)
        tar_list_path = frame_dir / "tar_list.txt"
        command = build_dic_command(
            args.gpu_bin, ref_path, tar_list_path, frame_dir,
            subset_radius, args.step,
            ev_line_ys[0] if ev_line_ys is not None else None,
            ev_line_ys[1] if ev_line_ys is not None else None,
            args.gpu_device, build_extra_cli(args),
        ) if (ev_line_ys is not None or not cod_mode) else None
        output_csv = event_output_csv(output_dir, event_name, args)
        log_path = event_binary_log(output_dir, event_name, args)
        frame_suffix = ".bin" if args.output_format == "raw" else ".csv"
        planned_files = [f"frame_{number:06d}{frame_suffix}" for number, _ in target_entries]
        event_state = describe_event_output(frame_dir, output_csv, log_path, planned_files)
        if event_state["cod_csv_exists"]:
            warnings.append(_issue(
                "cod_csv_exists",
                f"{output_csv} already exists and would be replaced by this run",
                event_name, severity=SEVERITY_WARN,
            ))
        if event_state["binary_expectation"] == "fail_not_a_directory":
            errors.append(_issue(
                "frame_dir_not_directory",
                f"{frame_dir} exists and is not a directory, so the run cannot create its frames",
                event_name,
            ))
        elif event_state["binary_expectation"] == "resume_if_unchanged":
            warnings.append(_issue(
                "frame_dir_has_ledger",
                f"{frame_dir} holds an earlier run's ledger: the binary resumes it only if the "
                "binary, the inputs, the parameters and the target list prefix are unchanged, "
                "and rejects the run otherwise (resume rejected)",
                event_name, severity=SEVERITY_WARN,
            ))
        elif event_state["binary_expectation"] == "reject_targets_changed":
            errors.append(_issue(
                "frame_dir_targets_changed",
                f"{frame_dir} holds an earlier run of {event_state.get('ledger_targets')} target "
                f"frame(s), and the planned {len(planned_files)} target(s) do not start with that "
                "list (for example --max-frames shortened it): the binary resumes only an identical "
                "list or one with frames appended (resume rejected: targets changed/reordered). "
                "Use a new --output-dir",
                event_name,
            ))
        elif event_state["binary_expectation"] == "reject_untracked":
            errors.append(_issue(
                "frame_dir_untracked",
                f"{frame_dir} holds {event_state['untracked_entries']} entr(y/ies) without a run "
                "ledger; the binary rejects such a directory (untracked entry / no verified run "
                "record). Use a new --output-dir",
                event_name,
            ))
        event_plans.append({
            "name": event_name,
            "image_dir": str(image_dir.resolve()),
            "images": len(images),
            "reference_frame": ref_no,
            "target_frames": len(target_entries),
            "target_frames_available": available_targets,
            "target_frame_range": (
                [target_entries[0][0], target_entries[-1][0]] if target_entries else None
            ),
            "first_frame": probe,
            "input_recommendation": recommendation,
            "input_saturation": input_saturation,
            "tar_list_path": str(tar_list_path.resolve()),
            "tar_entries": len(target_entries),
            "binary_command": command,
            "field_dir": str(frame_dir.resolve()) if not cod_mode else None,
            "cod_csv": str(output_csv.resolve()) if output_csv is not None else None,
            "binary_stderr_log": str(log_path.resolve()),
            "output_state": event_state,
            # Additive (2026-10-05): the event's own geometry, as the summary's events[].geometry.
            "geometry": effective_geometry(args, subset_radius, ev_line_ys, roi_cols, interface),
        })

    valid = not errors
    # warnings keeps what needs a look (warn level); info-level context goes to
    # notes, so a deployment configured through the config file still gets an
    # empty warnings list, as with 638ad18 (review 2026-10-04 R2).
    notes = [item for item in warnings if item.get("severity") == SEVERITY_INFO]
    warnings = [item for item in warnings if item.get("severity") != SEVERITY_INFO]
    return {
        "schema_version": 1,
        "dry_run": True,
        "valid": valid,
        "exit_code": 0 if valid else 2,
        "gpu_started": False,
        "filesystem_writes": False,
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
        "input": source_summary,
        "output_state": output_state,
        "event_selection": event_selection_public(selection, args),
        "resolved_params": {
            **summary_params(args),
            "subset_radius": subset_radius,
            "line_ys": list(line_ys) if line_ys is not None else None,
            "roi_cols_parsed": list(roi_cols) if roi_cols is not None else None,
            "geometry": effective_geometry(args, subset_radius, line_ys, roi_cols),
        },
        "events": event_plans,
        "estimates": build_estimates(
            estimate_mode(cod_mode, args.output_format), total_targets, total_rows,
            bool(args.json_summary), estimated_peak_bytes, estimated_retained_bytes,
        ),
        "known_limitations": [
            "first-frame probes do not prove that every frame has the same dimensions or bit layout",
            "dark left-shifted 12-bit data with max <= 4095 is ambiguous and may look right-aligned",
            "--input-depth auto misclassifies left-shifted 12-bit data with max > 4095 as 16-bit",
            "runtime and disk estimates are rough, not measured promises",
            "output_state compares file names only: whether an existing ledger resumes or is "
            "rejected depends on binary and input fingerprints that the dry-run does not compute",
        ],
    }


def aggregate_cod(frame_dir: Path, l0_y: int, l1_y: int, cal: float,
                  roi_cols: tuple[int, int] | None,
                  cod_method: str = "naive",
                  diagnostics: dict | None = None) -> tuple:
    """从 frame CSV 提取 COD 数据。

    ``diagnostics``（可选 dict）只读地收集结果自检要用的量：每条测线在帧数据里
    的行数、读不了的帧数、x 网格（像素）以及两条测线的 u 矩阵（给 naive 对照值）。
    它不参与、也不改变任何计算。

    cod_method:
      "naive"（默认，2026-10-05 起；2026-07-09 以前也是）— 直接 (u_L0 − u_L1) 不外推。
      "fillboundary"（可选；2026-07-09 至 2026-10-05 是默认）— Rubino2019 /
          paper §2.3 Eq(3) 梯度外推。
          先用每条测线的 ∂u/∂y（CSV 第 5 列 uy）把 u 线性外推到界面 y0，
          再做差：COD = [u_L0 + uy_L0·(y0−y_L0)] − [u_L1 + uy_L1·(y0−y_L1)]。
          消除"测线离界面 half 像素、子区不骑界面"带来的系统欠估，但单点噪声
          随外推距离放大，界面行给错时偏差也更大（docs/quality-checks.md §6）。

    review-2026-09-16 §4：CSV 无 uy 列（旧 5 列格式）时 **不再静默补零**，
    因为那等价于把 fillboundary 偷偷降级成 naive。fillboundary 下缺列直接抛
    ``CodAggregationError``（code ``cod_uy_column_missing``），调用方必须改用
    9 列输出，或显式 ``--cod-method naive``。
    """
    import numpy as np

    frame_files = sorted(
        [(int(f.stem.split("_")[1]), f) for f in frame_dir.iterdir()
         if f.name.startswith("frame_") and f.suffix == ".csv"],
        key=lambda t: t[0],
    )
    if not frame_files:
        return None, None, None, None

    # 从第一帧获取 x 网格
    first_data = np.atleast_2d(
        np.loadtxt(frame_files[0][1], delimiter=",", skiprows=1, dtype=np.float64)
    )
    all_xs = np.sort(np.unique(first_data[:, 0].astype(int)))

    # 应用 ROI 列限制
    if roi_cols:
        all_xs = all_xs[(all_xs >= roi_cols[0]) & (all_xs <= roi_cols[1])]

    n_cols = len(all_xs)
    x_to_idx = {int(x): i for i, x in enumerate(all_xs)}

    n_frames = len(frame_files)
    u_l0_all = np.full((n_frames, n_cols), np.nan, dtype=np.float64)
    u_l1_all = np.full((n_frames, n_cols), np.nan, dtype=np.float64)
    # ∂u/∂y 梯度（fillboundary 外推用）
    uy_l0_all = np.zeros((n_frames, n_cols), dtype=np.float64)
    uy_l1_all = np.zeros((n_frames, n_cols), dtype=np.float64)
    frame_numbers = []
    line_rows_l0 = 0
    line_rows_l1 = 0
    unreadable_frames = 0

    for fi, (frame_no, fpath) in enumerate(frame_files):
        frame_numbers.append(frame_no)
        try:
            data = np.loadtxt(fpath, delimiter=",", skiprows=1, dtype=np.float64)
        except Exception:
            unreadable_frames += 1
            continue
        data = np.atleast_2d(data)

        # 第 5 列 = uy (∂u/∂y)，来自 9 列输出。缺列时不补零，直接报错。
        if data.shape[1] <= 5:
            if cod_method == "fillboundary":
                raise CodAggregationError(
                    "cod_uy_column_missing",
                    f"{fpath} has {data.shape[1]} columns; fill-boundary COD needs the "
                    "9-column frame CSV (x,y,u,v,zncc,uy,vy,ux,vx). Re-run the CLI so it "
                    "emits uy, or pass --cod-method naive to accept the un-extrapolated "
                    "difference explicitly.",
                )
            uy_col = np.zeros(data.shape[0], dtype=np.float64)
        else:
            uy_col = data[:, 5]

        y_col = data[:, 1].astype(int)
        x_col = data[:, 0].astype(int)
        u_col = data[:, 2]
        zncc_col = data[:, 4]
        line_rows_l0 += int(np.count_nonzero(y_col == l0_y))
        line_rows_l1 += int(np.count_nonzero(y_col == l1_y))

        mask_l0 = (y_col == l0_y) & (zncc_col >= 0)
        for x, u, uy in zip(x_col[mask_l0], u_col[mask_l0], uy_col[mask_l0]):
            idx = x_to_idx.get(int(x))
            if idx is not None:
                u_l0_all[fi, idx] = u
                uy_l0_all[fi, idx] = uy

        mask_l1 = (y_col == l1_y) & (zncc_col >= 0)
        for x, u, uy in zip(x_col[mask_l1], u_col[mask_l1], uy_col[mask_l1]):
            idx = x_to_idx.get(int(x))
            if idx is not None:
                u_l1_all[fi, idx] = u
                uy_l1_all[fi, idx] = uy

    if cod_method == "fillboundary":
        # 界面 y0 取两测线中点；两侧 u 沿 ∂u/∂y 外推到界面再做差 (Eq.3)
        y0 = 0.5 * (l0_y + l1_y)
        d_l0 = y0 - l0_y   # = +half（L0 在界面上方，y 更小）
        d_l1 = y0 - l1_y   # = −half（L1 在界面下方，y 更大）
        u_l0_i = u_l0_all + uy_l0_all * d_l0
        u_l1_i = u_l1_all + uy_l1_all * d_l1
        cod_um = (u_l0_i - u_l1_i) * cal * 1000.0
    else:
        cod_um = (u_l0_all - u_l1_all) * cal * 1000.0

    x_mm = all_xs.astype(np.float64) * cal
    gauge_mm = abs(l1_y - l0_y) * cal
    frame_numbers = np.array(frame_numbers, dtype=int)

    if diagnostics is not None:
        diagnostics.update({
            "line_rows": {"L0": line_rows_l0, "L1": line_rows_l1},
            "unreadable_frames": unreadable_frames,
            "x_px": all_xs,
            "u_l0": u_l0_all,
            "u_l1": u_l1_all,
        })

    return cod_um, x_mm, frame_numbers, gauge_mm


def write_cod_csv(output_path: Path, cod_um, x_mm, frame_numbers):
    """写出最终 COD CSV。"""
    import numpy as np

    with open(output_path, "w") as f:
        f.write("frame,x_mm,cod_um\n")
        for fi, frame_no in enumerate(frame_numbers):
            for xi in range(len(x_mm)):
                val = cod_um[fi, xi]
                if np.isfinite(val):
                    f.write(f"{frame_no},{x_mm[xi]:.4f},{val:.4f}\n")


def publish_cod_csv(output_path: Path, cod_um, x_mm, frame_numbers) -> None:
    """原子发布 COD CSV：同目录临时文件写完再 rename（字节与 write_cod_csv 相同）。

    临时文件用 0666 & ~umask 创建，权限与原先直接 open('w') 一致。失败时目标
    路径保持原样（旧文件不会被半截文件替换）。
    """
    directory = output_path.parent
    for attempt in range(100):
        temporary = directory / f".{output_path.name}.{os.getpid()}.{attempt}.tmp"
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue
        os.close(fd)
        break
    else:  # pragma: no cover - 100 stale temporaries with our pid
        raise FileExistsError(f"cannot create a temporary file next to {output_path}")
    try:
        write_cod_csv(temporary, cod_um, x_mm, frame_numbers)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def cod_statistics(cod_um, x_mm, frame_numbers, cal: float, diagnostics: dict,
                   cod_method: str) -> dict:
    """COD 级统计（只读已算好的数组，不重算 COD）：行数、覆盖、极值及其位置。

    数值按 CSV 的 4 位小数取整；``all_zero`` 指写出的每个 cod_um 都是 0.0000。
    fillboundary 时 ``naive_at_max_um`` / ``naive_at_min_um`` 是同一 (frame, x)
    的 naive 差分 1000·cal·(u_L0 − u_L1)。
    """
    import numpy as np

    finite = np.isfinite(cod_um)
    values = cod_um[finite]
    frames_total = int(len(frame_numbers))
    frames_with_points = int(np.count_nonzero(finite.any(axis=1))) if cod_um.size else 0
    line_rows = diagnostics.get("line_rows", {"L0": None, "L1": None})
    stats = {
        "rows": int(values.size),
        "frames": frames_total,
        "frames_with_points": frames_with_points,
        "frames_without_points": frames_total - frames_with_points,
        "x_points": int(len(x_mm)),
        "line_rows": dict(line_rows),
        "line_rows_present": {
            label: (count > 0 if isinstance(count, int) else None)
            for label, count in line_rows.items()
        },
        "all_zero": bool(values.size) and bool(np.all(np.abs(values) < 5e-5)),
        "max_um": None, "max_frame": None, "max_x_px": None, "max_x_mm": None,
        "min_um": None, "min_frame": None, "min_x_px": None, "min_x_mm": None,
        "abs_p95_um": None,
        "naive_at_max_um": None,
        "naive_at_min_um": None,
        # Frame CSVs aggregate_cod could not parse; their columns stay empty.
        "unreadable_frames": diagnostics.get("unreadable_frames"),
    }
    if not values.size:
        return stats
    x_px = diagnostics.get("x_px")
    u_l0 = diagnostics.get("u_l0")
    u_l1 = diagnostics.get("u_l1")
    for label, flat in (
        ("max", int(np.argmax(np.where(finite, cod_um, -np.inf)))),
        ("min", int(np.argmin(np.where(finite, cod_um, np.inf)))),
    ):
        fi, xi = np.unravel_index(flat, cod_um.shape)
        stats[f"{label}_um"] = round(float(cod_um[fi, xi]), 4)
        stats[f"{label}_frame"] = int(frame_numbers[fi])
        stats[f"{label}_x_px"] = int(x_px[xi]) if x_px is not None else None
        stats[f"{label}_x_mm"] = round(float(x_mm[xi]), 4)
        if cod_method == "fillboundary" and u_l0 is not None and u_l1 is not None:
            naive = (u_l0[fi, xi] - u_l1[fi, xi]) * cal * 1000.0
            stats[f"naive_at_{label}_um"] = (
                round(float(naive), 4) if np.isfinite(naive) else None
            )
    stats["abs_p95_um"] = round(float(np.percentile(np.abs(values), 95)), 4)
    return stats


def process_single_event(event_name: str, image_dir: Path, output_dir: Path,
                         args: argparse.Namespace, subset_radius: int,
                         l0_y: int | None, l1_y: int | None,
                         roi_cols: tuple | None,
                         interface: dict | None = None) -> dict:
    """处理单个事件。

    ``interface`` 是该事件的界面行与来源（event_interface），只进 summary 的
    geometry；测线本身由 l0_y / l1_y 给出。

    COD 模式：DIC → fill-boundary/naive 聚合 → 一个 COD CSV。
    全场模式（无 --line-y/--interface-y）：只跑 DIC，把整幅位移场的 frame
    CSV/raw 留在 `<output-dir>/<event>/`，不做任何聚合。

    返回的事件 dict 里 ``_legacy_ok`` 记录旧版的成功判据（默认模式的退出码仍按它），
    ``ok``/``status`` 是如实的结果（见 finalize_event）。
    """
    cod_mode = is_cod_mode(args)
    strict = bool(getattr(args, "strict", False))
    output_csv = event_output_csv(output_dir, event_name, args)
    frame_dir = event_frame_dir(output_dir, event_name, args)
    log_path = event_binary_log(output_dir, event_name, args)
    images = find_images_for(args, image_dir)
    result = make_event_result(event_name, output_csv, args, frame_dir)
    result["_legacy_ok"] = False
    line_ys = (l0_y, l1_y) if l0_y is not None and l1_y is not None else None
    result["geometry"] = effective_geometry(args, subset_radius, line_ys, roi_cols, interface)
    csv_before = file_signature(output_csv) if cod_mode else None

    def finish() -> dict:
        # field_dir keeps its 638ad18 value (the run's frame directory, deleted or
        # not); whether it still exists is field_dir_exists (review 2026-10-04).
        result["field_dir"] = str(frame_dir.resolve())
        result["field_dir_exists"] = frame_dir.is_dir()
        if cod_mode and output_csv is not None and result.get("_cod_written") is None:
            if output_csv.exists():
                add_warning(
                    result, "stale_cod_csv",
                    f"{output_csv} was not written by this run (last modified "
                    f"{describe_mtime(output_csv)}); it belongs to an earlier run",
                )
        finalize_event(result, strict)
        return result

    if len(images) < 2:
        print(f"  [{event_name}] ⚠️ 图片不足 ({len(images)} 张)，跳过")
        result["warnings"].append(f"insufficient_images: {len(images)}")
        return finish()

    ref_no, ref_path = images[0]
    tar_entries, available_targets = select_targets(images, getattr(args, "max_frames", None))
    result["target_frames_available"] = available_targets
    result["target_frames_selected"] = len(tar_entries)
    duplicates = duplicate_frame_numbers(images)

    print(f"  [{event_name}] {len(tar_entries)} 帧", end="")
    if len(tar_entries) < available_targets:
        add_warning(
            result, "max_frames_applied",
            f"first {len(tar_entries)} of {available_targets} target frames "
            f"(--max-frames {args.max_frames})",
        )
        print(f"（--max-frames：共 {available_targets} 帧）", end="")
    target_duplicates = [number for number in duplicates if number != ref_no]
    if target_duplicates:
        add_warning(
            result, "duplicate_frame_number",
            f"{len(target_duplicates)} target frame number(s) occur more than once (first: "
            f"{target_duplicates[:5]}); check --image-pattern",
        )
    if ref_no in duplicates:
        same = [Path(path).name for number, path in images if number == ref_no]
        add_warning(
            result, "duplicate_reference_frame_number",
            f"{len(same)} images carry the reference frame number {ref_no}; {same[0]} is the "
            "reference and the others were not used; check --image-pattern",
        )

    stale_frames = frame_dir.is_dir() and any(frame_dir.glob("frame_*.*"))
    frame_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Step 1: DIC
    run = invoke_dic(
        args.gpu_bin, ref_path, tar_entries, frame_dir,
        subset_radius, args.step, l0_y, l1_y, args.gpu_device,
        build_extra_cli(args), log_path,
    )
    result["binary_exit_code"] = run["returncode"]
    result["binary_stderr_log"] = str(run["log_path"].resolve()) if run["log_path"] else None
    result["binary_totals"] = parse_binary_totals(run["stderr"])
    if run["log_error"]:
        add_warning(result, "binary_log_unavailable",
                    f"cannot write {log_path}: {run['log_error']}; the stderr was kept in memory only")
    if not run["ok"]:
        # Preserve partial results and integrity records for diagnosis/recovery.
        # In particular, a CLI refusal of stale/corrupt data must never delete it.
        print(f" → ❌ DIC 失败，保留中间目录: {frame_dir}")
        first_error = first_binary_error(run["stderr"])
        detail = f"exit={run['returncode']}"
        if first_error:
            detail += f": {first_error}"
        add_warning(result, "gpu_cli_failed", detail)
        result["remediation"] = binary_failure_remediation(run["stderr"], frame_dir)
        print(f"    建议: {result['remediation']}")
        return finish()
    input_reference = run["input_reference"]
    result["input_reference"] = input_reference
    totals = result["binary_totals"] or {}
    resumed_verified = stale_frames and (
        totals.get("verified_complete_run") or totals.get("resume_verified_files") is not None
    )
    for text in binary_warning_lines(run["stderr"])[:MAX_BINARY_WARNINGS_IN_SUMMARY]:
        add_warning(result, "binary_warning", text)
    extra_warnings = len(binary_warning_lines(run["stderr"])) - MAX_BINARY_WARNINGS_IN_SUMMARY
    if extra_warnings > 0:
        add_warning(
            result, "binary_warning",
            f"{extra_warnings} more distinct warning(s) in {result['binary_stderr_log']}",
        )
    if resumed_verified:
        add_warning(
            result, "resumed_verified",
            "the frame directory already held frames of a matching earlier run; the binary "
            "verified them against its ledger instead of recomputing or mixing them",
        )
    fallback_frames = totals.get("frames_with_fallback_fftcc")
    if fallback_frames:
        add_warning(
            result, "frame_fallback_used",
            f"{fallback_frames} frame(s) needed the whole-frame FFTCC fallback (see the binary "
            "stderr log); treat valid POIs with zncc < 0.8 in those frames as suspect",
        )

    # Step 2: COD 聚合（只有 COD 模式才做）
    cod_error: CodAggregationError | None = None
    cod_um = x_mm = frame_numbers = gauge_mm = None
    cod_diagnostics: dict = {}
    if cod_mode:
        try:
            cod_um, x_mm, frame_numbers, gauge_mm = aggregate_cod(
                frame_dir, l0_y, l1_y, args.cal, roi_cols, args.cod_method,
                diagnostics=cod_diagnostics,
            )
        except CodAggregationError as exc:
            cod_error = exc

    if args.json_summary:
        stats = collect_frame_stats(frame_dir, getattr(args, "output_format", "csv"))
        for key in (
            "frames", "poi_per_frame", "poi_per_frame_min", "poi_per_frame_max",
            "valid_ratio", "zncc_median",
        ):
            result[key] = stats[key]
        effective_depth = (
            input_reference.get("effective_depth")
            if input_reference is not None
            else (args.input_depth or "8")
        )
        threshold = getattr(args, "saturation_warn_fraction", DEFAULT_SATURATION_WARN_FRACTION)
        saturation = probe_input_saturation(
            ref_path, effective_depth,
            saturation_geometry(args, subset_radius, line_ys, roi_cols), threshold,
        )
        report = saturation["input_saturation"]
        result["input_pixel_max"] = saturation["pixel_max"]
        result["input_saturated_fraction"] = saturation["saturated_fraction"]
        result["input_saturation"] = report
        for text in event_warnings(
            stats, effective_depth, result["input_pixel_max"],
            stale_frames and not resumed_verified,
            container_bit_depth=saturation["container_bit_depth"], saturation=report,
        ):
            severity = None
            if warning_code(text) == "input_8bit_max_255":
                severity = (report["severity"] if report is not None else input_saturation_severity(
                    None, saturation["container_bit_depth"], threshold))
            add_warning(result, warning_code(text), text.partition(": ")[2] or None, severity)
        if input_reference is None:
            result["warnings"].append("effective_input_reference_unavailable")

    if cod_error is not None:
        # 缺 uy 列是硬失败：保留 frame 目录，让调用方能改参数重跑。
        print(f" → ❌ COD 聚合被拒: {cod_error.code}")
        print(f"    {cod_error}")
        result["warnings"].append(f"{cod_error.code}: {cod_error}")
        return finish()

    cod_stats = None
    if cod_mode and cod_um is not None:
        cod_stats = cod_statistics(
            cod_um, x_mm, frame_numbers, args.cal, cod_diagnostics, args.cod_method,
        )
        result["cod_stats"] = cod_stats
    cod_empty = cod_stats is not None and cod_stats["rows"] == 0

    # 清理中间文件（只有 COD 模式才有“中间”文件可清；全场模式产物就是它们）。
    # 默认模式与 638ad18 一样删掉，COD 为空也删：留着的帧目录带着 ledger，改了
    # 参数往同一 --output-dir 重跑时 binary 会拒绝（review 2026-10-04 R1）。
    # --keep-frames 总是保留；--strict 在 COD 为空时原地保留，便于诊断。
    if cod_mode and not args.keep_frames and not (strict and cod_empty):
        shutil.rmtree(frame_dir, ignore_errors=True)

    elapsed = time.time() - t0
    if not cod_mode:
        frames = sorted(frame_dir.glob("frame_*.*"))
        payloads = [f for f in frames if f.suffix in {".csv", ".bin"}]
        result["_legacy_ok"] = True
        result["ok"] = True
        result["status"] = "ok"
        finish()
        print(f" → {event_mark(result, strict)} {elapsed:.1f}s, {len(payloads)} 个全场帧 → {frame_dir}")
        print_event_warnings(result, strict)
        return result

    if cod_um is None:
        print(" → ❌ COD 聚合失败")
        result["warnings"].append("cod_aggregation_failed")
        return finish()

    # Step 3: 写 CSV（临时文件 + 原子 rename）
    publish_cod_csv(output_csv, cod_um, x_mm, frame_numbers)
    result["_cod_written"] = str(output_csv.resolve())
    result["_legacy_ok"] = True
    result["ok"] = True
    if csv_before is not None:
        add_warning(
            result, "cod_csv_replaced",
            f"replaced an existing {output_csv.name} from an earlier run; use a new "
            "--output-dir per run",
        )
    elapsed = time.time() - t0

    labels = {"L0": l0_y, "L1": l1_y}
    missing = [
        f"{label} (y={labels[label]})"
        for label, present in cod_stats["line_rows_present"].items() if present is False
    ]
    if missing:
        add_warning(
            result, "cod_line_missing",
            f"{' and '.join(missing)} has no row in any frame CSV; the binary skips a line "
            "inside its subset margin (see binary_warning). Check --interface-y/--gauge/--line-y "
            "with --dry-run",
        )
    if cod_empty:
        if cod_stats["x_points"] == 0:
            reason = f"--roi-cols {args.roi_cols} selects no POI column"
        elif missing:
            reason = f"{' and '.join(missing)} has no row"
        else:
            reason = "no (frame, x) point has a valid row (finite zncc >= 0) on both lines"
        add_warning(
            result, "cod_empty",
            f"{output_csv.name} has a header and no data row: {reason}; "
            + (f"the frame CSVs were kept in {frame_dir}" if frame_dir.is_dir() else
               "the frame CSVs were deleted as usual (--keep-frames or --strict keep them)"),
        )
    elif cod_stats["all_zero"]:
        add_warning(
            result, "cod_all_zero",
            f"all {cod_stats['rows']} written cod_um values are 0.0000 (both lines on one row, "
            "--cal 0, or no motion at 4-decimal precision)",
        )
    if cod_stats["frames_without_points"] and not cod_empty:
        add_warning(
            result, "cod_frames_without_points",
            f"{cod_stats['frames_without_points']} of {cod_stats['frames']} target frames have "
            "no COD point",
        )
    finish()

    import numpy as np
    max_cod = float(np.nanmax(np.abs(cod_um))) if np.any(np.isfinite(cod_um)) else 0.0
    size_kb = output_csv.stat().st_size / 1024
    mark = event_mark(result, strict)
    if result["status"] != "ok":
        print(f" → {mark} {elapsed:.1f}s, {cod_stats['rows']} 行 → {output_csv.name}")
        print_event_warnings(result, strict)
        return result
    print(f" → {mark} {elapsed:.1f}s, maxCOD={max_cod:.0f}μm, {size_kb:.0f}KB → {output_csv.name}")
    if cod_stats["rows"]:
        extreme = "max" if abs(cod_stats["max_um"]) >= abs(cod_stats["min_um"]) else "min"
        naive = cod_stats[f"naive_at_{extreme}_um"]
        print(
            f"    |COD| 最大的点: {cod_stats[f'{extreme}_um']}μm @ frame {cod_stats[f'{extreme}_frame']}, "
            f"x={cod_stats[f'{extreme}_x_px']}px"
            + (f"（同一点 naive {naive}μm）" if naive is not None else "")
            + f"；{cod_stats['rows']} 行，{cod_stats['frames_with_points']}/{cod_stats['frames']} 帧有点"
        )
    print_event_warnings(result, strict)
    return result


def event_mark(event: dict, strict: bool = False) -> str:
    """控制台标记：失败 ❌；有 warn 级告警 ⚠️；否则 ✅（info 不影响）。"""
    if event.get("status") != "ok":
        return "❌"
    levels = {
        warning_severity(text, event.get("_severity"), strict)
        for text in event.get("warnings", [])
    }
    return "⚠️" if SEVERITY_WARN in levels else "✅"


def print_event_warnings(event: dict, strict: bool = False) -> None:
    """把 warn / fail 级原因逐条打印出来（info 级只进摘要）。"""
    for text in event.get("warnings", []):
        level = warning_severity(text, event.get("_severity"), strict)
        if level == SEVERITY_FAIL:
            print(f"    ❌ {text}")
        elif level == SEVERITY_WARN:
            print(f"    ⚠️ {text}")


def create_parser() -> argparse.ArgumentParser:
    parser = PipelineArgumentParser(
        description="FastICGN pipeline：图像序列 → 全场位移（默认）或 COD CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Unambiguous abbreviations (--dry, --desc, --keep) work as in 638ad18;
        # main() resolves them with requested_long_options (review 2026-10-04 R3).
    )

    # 输入方式（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--photo-dir",
                             help="总照片目录（自动遍历子文件夹，每个子文件夹为一个事件）")
    input_group.add_argument("--image-dir",
                             help="单个事件的图片目录")

    # 输出
    parser.add_argument("--output-dir", required=True,
                        help="输出目录（每次运行用一个新目录；已有的 COD CSV 会被替换）")

    # 必选参数
    parser.add_argument("--interface-y", type=int, default=None,
                        help="界面 y 像素值（参考图的行号，向下增大；与 --gauge 配合）；"
                             "给了它或 --line-y 才进 COD 模式。测线 L0=Y-gauge//2、L1=Y+gauge//2")
    parser.add_argument("--line-y", type=str, default=None,
                        help="直接指定两条测线的 y 值，格式: L0,L1（覆盖 --interface-y 和 --gauge）。"
                             "--cod-method fillboundary 时外推到两线中点；COD = L0 侧 − L1 侧，顺序决定正负号")
    parser.add_argument("--events", type=str, default=None, metavar="LIST",
                        help="campaign 只跑部分事件（只配 --photo-dir）：逗号分隔的事件名或 glob"
                             "（fnmatch，区分大小写），如 'event0*,event1*,event20' 或 'event2[1-5]'。"
                             "每一项都必须匹配到至少一个事件，否则读图之前报错（exit 2），不写任何输出；"
                             "筛选条件与被筛掉的事件进 dry-run / summary 的 event_selection")
    parser.add_argument("--event-interface-y", action="append", default=None,
                        metavar="PATTERN=Y",
                        help="campaign 逐事件界面行（可重复，只配 --photo-dir 与 --interface-y）："
                             "事件名匹配 PATTERN（事件名或 glob）的事件用界面行 Y，其余用 --interface-y；"
                             "测线 Y∓gauge//2，--gauge 仍全局。与 --line-y 同给、某条不匹配任何选中的"
                             "事件、一个事件命中两个不同的 Y 都报错（exit 2）。每个事件用的界面行与来源"
                             "见 events[].geometry")
    parser.add_argument("--cal", type=float, default=None,
                        help="像素标定值 (mm/px，有限且 >0)。COD 模式必填，NaN/inf 在 COD 模式下"
                             "直接报错；全场模式不用、也不检查")

    # DIC 参数
    parser.add_argument("--subset", type=int, default=15,
                        help="子集 SIZE（默认 15，奇数，5–51）。注意是 size 不是 radius，"
                             "内部换算 radius=(size-1)//2")
    parser.add_argument("--step", type=int, default=1,
                        help="网格步长（默认 1）")
    parser.add_argument("--output-format", choices=["csv", "raw"], default="csv",
                        help="全场模式的 frame 产物格式（默认 csv）；raw 只在全场模式可用")
    parser.add_argument("--image-pattern", type=str, default=None,
                        help="图像发现正则（默认 "
                             r"'(\d{6})\.(jpg|jpeg|tif|tiff|png|bmp)$'"
                             "）；第一个捕获组是数字就当帧号")
    parser.add_argument("--image-order", choices=list(IMAGE_ORDER_CHOICES),
                        default="frame-number",
                        help="图像排序策略（默认 frame-number；name = 按文件名，帧号用 1 起序号）")
    parser.add_argument("--max-frames", type=positive_int, default=None,
                        help="只算参考帧之后的前 N 个目标帧（>=1；试跑用，默认全部）")
    parser.add_argument("--roi-cols", type=str, default=None,
                        help="COD 聚合的水平 ROI，格式: x1,x2（像素，含两端；不减少 binary 计算的点）")
    parser.add_argument("--gauge", type=int, default=DEFAULT_GAUGE,
                        help=f"两条测线的请求间距（像素，默认 {DEFAULT_GAUGE}）；"
                             "实际间距 2*(gauge//2)，默认即 34")
    parser.add_argument("--cod-method", choices=["fillboundary", "naive"],
                        default="naive",
                        help="COD（列 cod_um）是滑移量：界面两侧的相对位移 u_L0−u_L1，不是张开量。"
                             "聚合方式：naive=直接 u_L0−u_L1（默认，2026-10-05 起）；"
                             "fillboundary=沿各测线 ∂u/∂y 线性外推到两线中点再相减（可选，"
                             "单点噪声更大、对界面行偏差更敏感；2026-07-09 至 2026-10-05 曾是默认）")
    parser.add_argument("--conv-threshold", type=finite_float, default=0.001,
                        help="ICGN 收敛阈值，恒显式透传给 GPU CLI（默认 0.001=CPU 对齐值；"
                             "钉死语义，不依赖 binary 自身默认）")

    # 输入位深（透传给 GPU CLI；缺省=binary 默认 8-bit legacy 行为）
    parser.add_argument("--input-depth", choices=["8", "12", "16", "auto"],
                        default=None,
                        help="输入图像位深（透传）。左移存储的 12-bit 数据（uint16 容器、低 4 位恒零、"
                             "max>4095）必须显式 --input-depth 12 --input-align left（auto 会误判成 16）")
    parser.add_argument("--input-align", choices=["right", "left", "auto"],
                        default=None,
                        help="12-bit 数据在 16-bit 容器内的对齐（透传）：left=存储值÷16")
    parser.add_argument("--input-normalize", choices=["none", "to-8", "to-16", "unit"],
                        default=None,
                        help="灰度归一化（透传；ZNCC 线性不变，不影响 DIC 结果，仅改浮点范围）")

    # GPU 参数
    parser.add_argument("--gpu-device", type=int, default=0,
                        help="GPU 编号（默认 0）")
    parser.add_argument("--gpu-bin", type=str, default=None,
                        help=f"GPU CLI binary 路径（相对路径按当前目录转成绝对路径）。缺省解析顺序："
                             f"${ENV_GPU_BIN} → {USER_CONFIG_PATH} 的 [paths].gpu_bin → "
                             "仓库 build/ 树 → 实验室旧路径（deprecated）")

    # 其他
    parser.add_argument("--keep-frames", action="store_true",
                        help="保留 COD 模式的中间 frame CSV（默认删除）；全场模式的帧就是产物，总是保留")
    parser.add_argument("--strict", action="store_true",
                        help="严格模式：先做 dry-run 的全部校验（不通过 exit 2）；全零 COD、重复帧号"
                             "按失败处理；有失败事件时 exit 3（默认模式下单事件失败 exit 1、"
                             "campaign exit 0）")
    parser.add_argument("--saturation-warn-fraction", type=unit_fraction,
                        default=DEFAULT_SATURATION_WARN_FRACTION,
                        help="=255 像素比例超过它时 input_8bit_max_255 记为 warn，否则 info。COD 模式"
                             "看各测线带（该测线参与 COD 聚合的子区像素）的最大占比，全场模式看参考帧"
                             f"整帧（启发式、可调，默认 {DEFAULT_SATURATION_WARN_FRACTION}）")
    parser.add_argument("--json-summary", type=str, default=None,
                        help="运行结束写入结构化 JSON 摘要（schema_version=1）")
    parser.add_argument("--dry-run", action="store_true",
                        help="不写磁盘、不启动 GPU；stdout 输出 JSON 校验与执行计划")
    parser.add_argument("--describe", action="store_true",
                        help="stdout 原样输出 pipeline/interface_contract.json 后退出")

    return parser


def apply_path_resolution(args: argparse.Namespace) -> argparse.Namespace:
    """就地把 args.gpu_bin 解析成具体路径，并记录来源与 deprecation 提示。"""
    notes: list[str] = []
    resolved, source = resolve_gpu_bin(getattr(args, "gpu_bin", None), notes)
    args.gpu_bin = resolved
    args.gpu_bin_source = source
    args.path_resolution_notes = notes
    return args


def exception_event(event_name: str, output_dir: Path, args: argparse.Namespace,
                    exc: Exception) -> dict:
    """--json-summary 下事件内异常的失败记录（不再指向可能残留的旧 COD CSV）。"""
    output_csv = event_output_csv(output_dir, event_name, args)
    result = make_event_result(
        event_name, output_csv, args, event_frame_dir(output_dir, event_name, args),
    )
    result["_legacy_ok"] = False
    result["warnings"].append(f"pipeline_exception: {exc}")
    if output_csv is not None and output_csv.exists():
        add_warning(
            result, "stale_cod_csv",
            f"{output_csv} exists (last modified {describe_mtime(output_csv)}) but this run "
            "did not finish the event, so it may belong to an earlier run",
        )
    finalize_event(result, bool(getattr(args, "strict", False)))
    return result


def first_failure_reason(event: dict, strict: bool) -> str:
    for text in event.get("warnings", []):
        if warning_severity(text, event.get("_severity"), strict) == SEVERITY_FAIL:
            return text
    return "failed"


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = create_parser()
    requested = requested_long_options(parser, raw_argv)
    if "--describe" in requested:
        if len(raw_argv) != 1:
            print("错误: --describe 不接受其他参数", file=sys.stderr)
            return 2
        try:
            sys.stdout.write(INTERFACE_CONTRACT_PATH.read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"错误: 无法读取接口契约: {exc}", file=sys.stderr)
            return 1
        return 0
    if isinstance(parser, PipelineArgumentParser):
        parser.machine_readable_errors = "--dry-run" in requested
    args = parser.parse_args(raw_argv)
    apply_path_resolution(args)
    t_total = time.time()

    if args.dry_run:
        try:
            plan = build_dry_run(args)
        except Exception as exc:
            plan = minimal_dry_run("dry_run_exception", str(exc))
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return int(plan["exit_code"])

    # --strict pre-check warnings and notes, recorded in the summary (R6b).
    precheck_entries: list[dict] = []
    # --events / --event-interface-y, for the summary (set below for a campaign).
    selection: dict | None = None

    def fail(message: str, code: int = EXIT_FAILURE,
             extra_messages: list[str] | None = None) -> int:
        print(f"错误: {message}", file=sys.stderr)
        for item in extra_messages or []:
            print(f"错误: {item}", file=sys.stderr)
        if args.json_summary:
            try:
                summary = build_failure_summary(
                    args, message, time.time() - t_total, extra_messages, precheck_entries,
                )
                write_json_summary(Path(args.json_summary), summary)
            except Exception as exc:
                print(f"错误: 无法写失败摘要: {exc}", file=sys.stderr)
        return code

    def write_summary(events: list[dict], binary: dict, top_warnings: list[str],
                      elapsed: float) -> bool:
        """写摘要；失败时降级写一份失败摘要，而不是只留 traceback。"""
        try:
            summary = build_run_summary(args, binary, top_warnings, events, elapsed,
                                        precheck_entries,
                                        event_selection_public(selection, args))
            write_json_summary(Path(args.json_summary), summary)
            return True
        except Exception as exc:
            print(f"错误: 无法写运行摘要: {exc}", file=sys.stderr)
            try:
                fallback = build_failure_summary(
                    args, f"summary_write_failed: {exc}", time.time() - t_total,
                )
                write_json_summary(Path(args.json_summary), fallback)
                print("  已改写一份失败摘要（verdict=fail）", file=sys.stderr)
            except Exception as exc2:
                print(f"错误: 失败摘要也写不出: {exc2}", file=sys.stderr)
            return False

    if is_cod_mode(args) and args.cal is not None and not math.isfinite(args.cal):
        # COD mode multiplies by --cal; full-field mode never reads it (R3).
        parser.error(f"argument --cal: must be finite in COD mode, got {args.cal!r}")

    strict = bool(args.strict)
    for note in args.path_resolution_notes:
        print(f"警告: {note}", file=sys.stderr)

    if strict:
        # --strict: the complete dry-run validation (image-aware checks, output
        # directory state, duplicate frame numbers) must pass before any write.
        # A campaign is checked as a whole: one refused event refuses them all.
        try:
            plan = build_dry_run(args)
        except Exception as exc:
            plan = minimal_dry_run("dry_run_exception", str(exc))
        precheck_entries.extend(precheck_record(plan))
        if not plan["valid"]:
            messages = [
                f"{item['code']}: {item['message']}"
                + (f" [event {item['event']}]" if "event" in item else "")
                for item in plan["errors"]
            ]
            return fail(
                "--strict: dry-run validation failed, nothing was run", EXIT_INVALID, messages,
            )
        for item in plan.get("warnings", []):
            if item.get("severity") == SEVERITY_WARN:
                where = f" [event {item['event']}]" if "event" in item else ""
                print(f"警告: {item['code']}: {item['message']}{where}", file=sys.stderr)

    # --events / --event-interface-y: refused with exit 2 in every mode, before any
    # image is read or anything is written (with --strict the dry-run above already
    # refused them). Lists the campaign's directories only.
    selection = plan_event_selection(args, campaign_event_names(args))
    if selection["errors"]:
        return fail(
            "--events / --event-interface-y are invalid, nothing was run", EXIT_INVALID,
            [f"{item['code']}: {item['message']}"
             + (f" [event {item['event']}]" if "event" in item else "")
             for item in selection["errors"]],
        )

    # 参数处理（默认模式保持历史检查顺序与退出码 1）
    cod_mode = is_cod_mode(args)
    subset_radius = (args.subset - 1) // 2
    if args.gpu_bin_source == "unresolved":
        return fail(binary_search_description(GPU_BINARY_NAME))
    if cod_mode and args.cal is None:
        return fail("COD 模式（--interface-y/--line-y）必须给 --cal")
    if cod_mode and args.output_format != "csv":
        return fail("COD 聚合读 frame CSV；--output-format raw 只用于全场模式")
    try:
        compile_image_pattern(args.image_pattern)
    except re.error as exc:
        return fail(f"--image-pattern 不是合法正则: {exc}")
    try:
        roi_cols = parse_roi_cols(args.roi_cols)
    except ValueError as exc:
        return fail(str(exc))

    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return fail(f"无法创建输出目录 {output_dir}: {exc}")

    if not Path(args.gpu_bin).is_file():
        return fail(f"GPU binary 不存在: {args.gpu_bin}")
    if not os.access(args.gpu_bin, os.X_OK):
        return fail(f"GPU binary 不可执行: {args.gpu_bin}")

    # 计算测线位置（全场模式为 None）
    try:
        line_ys = resolve_line_ys(args)
    except ValueError as exc:
        return fail(str(exc))
    l0_y, l1_y = line_ys if line_ys is not None else (None, None)

    # 默认模式：dry-run 会拒绝、而正常执行历来放行的参数，如实告警但照旧运行。
    argument_warnings: list[str] = []
    if not strict:
        for item in validate_args(args)["errors"]:
            if item["code"] not in LEGACY_ENFORCED_CODES | EVENT_SELECTION_CODES:
                argument_warnings.append(f"argument_warning: {item['code']}: {item['message']}")
    source_note = binary_source_note(args.gpu_bin, args.gpu_bin_source)

    print("=" * 60)
    print("FastICGN pipeline — " + ("COD 模式" if cod_mode else "全场位移模式")
          + ("（--strict）" if strict else ""))
    print("=" * 60)
    print(f"  subset={args.subset} (size, radius={subset_radius}), step={args.step}")
    if cod_mode:
        gauge_mm = abs(l1_y - l0_y) * args.cal
        print(f"  cal={args.cal} mm/px")
        print(f"  断层面 Y={args.interface_y}, 测线 L0=Y{l0_y}, L1=Y{l1_y}, gauge={gauge_mm:.1f}mm")
        print(f"  COD 方式: {args.cod_method}"
              + ("（Eq.3 梯度外推到界面）" if args.cod_method == "fillboundary" else "（naive 直接差分）"))
        for override in selection["overrides"]:
            o_l0, o_l1 = compute_line_ys(override["interface_y"], args.gauge)
            print(f"  逐事件界面行 {override['argument']}: Y={override['interface_y']}, "
                  f"测线 L0=Y{o_l0}, L1=Y{o_l1} → {', '.join(override['events'] or [])}")
    else:
        print(f"  全场输出: --output-format {args.output_format}（不做 COD 聚合）")
    print(f"  binary: {args.gpu_bin} (source={args.gpu_bin_source})")
    if source_note:
        print(f"  ℹ️ {source_note}")
    print(f"  conv-threshold: {args.conv_threshold}（显式透传）")
    if args.input_depth or args.input_align or args.input_normalize:
        print(f"  输入位深: depth={args.input_depth or '8(默认)'}"
              f" align={args.input_align or '-'}"
              f" normalize={args.input_normalize or '-'}")
    if roi_cols:
        print(f"  ROI 列范围: {roi_cols[0]} - {roi_cols[1]}")
    if args.max_frames is not None:
        print(f"  --max-frames {args.max_frames}：每个事件只算前 {args.max_frames} 个目标帧")
    print(f"  GPU device: {args.gpu_device}")
    for text in argument_warnings:
        print(f"  ⚠️ {text}")
        print(f"警告: {text}", file=sys.stderr)
    print()

    binary = {"path": str(Path(args.gpu_bin).resolve()), "git_hash": None, "build_date": None}
    binary_warnings: list[str] = []
    if args.json_summary:
        binary, binary_warnings = probe_binary_version(args.gpu_bin)
    top_warnings = list(binary_warnings) + argument_warnings
    if source_note:
        top_warnings.append(f"gpu_bin_outside_checkout: {source_note}")
    event_results: list[dict] = []

    if args.photo_dir:
        # ── 批量模式 ──
        photo_dir = Path(args.photo_dir)
        if not photo_dir.is_dir():
            return fail(f"目录不存在: {photo_dir}")

        # 摘要模式必须把仅 1 帧的事件计为失败；旧模式保持原来的静默忽略行为。
        events = discover_events(
            photo_dir, 1 if args.json_summary else 2,
            args.image_pattern, args.image_order,
        )
        if not events:
            return fail(f"{photo_dir} 下没有找到含图片的子文件夹")
        if selection["patterns"] is not None:
            chosen = set(selection["selected"] or [])
            print(f"--events {selection['events_argument']}: 选中 {len(chosen)} 个，"
                  f"筛掉 {len(selection['excluded'] or [])} 个")
            events = [(name, path) for name, path in events if name in chosen]
            if not events:
                return fail(f"--events {selection['events_argument']} 没有选中可运行的事件")

        print(f"发现 {len(events)} 个事件:")
        for name, path in events:
            n_imgs = len(find_images_for(args, path))
            print(f"  {name}/ — {n_imgs} 张图片")
        print()

        for name, path in events:
            interface = event_interface(selection, args, name)
            ev_l0, ev_l1 = event_line_ys(args, interface, line_ys) or (None, None)
            try:
                event_result = process_single_event(
                    name, path, output_dir, args, subset_radius, ev_l0, ev_l1, roi_cols,
                    interface=interface,
                )
            except Exception as exc:
                if not args.json_summary:
                    raise
                print(f"  [{name}] → ❌ pipeline 异常: {exc}")
                event_result = exception_event(name, output_dir, args, exc)
            event_results.append(event_result)
        success = sum(bool(event["ok"]) for event in event_results)
        failed = len(event_results) - success

        elapsed = time.time() - t_total
        print(f"\n{'=' * 60}")
        if failed == 0:
            print(f"✅ 全部完成！{success} 成功, {failed} 失败, 总耗时 {elapsed:.1f}s")
        else:
            print(f"❌ 运行结束，但有失败事件：{success} 成功, {failed} 失败, 总耗时 {elapsed:.1f}s")
            for event in event_results:
                if not event["ok"]:
                    print(f"   ❌ {event['name']}: {first_failure_reason(event, strict)}")
            if not strict:
                print("   （默认模式下 campaign 仍 exit 0；加 --strict 则 exit 3）")
        print(f"   输出目录: {output_dir}")
        print(f"{'=' * 60}")
        if args.json_summary and not write_summary(event_results, binary, top_warnings, elapsed):
            return EXIT_FAILURE
        if strict and failed:
            return EXIT_STRICT_EVENT_FAILED

    else:
        # ── 单事件模式 ──
        image_dir = Path(args.image_dir)
        if not image_dir.is_dir():
            return fail(f"目录不存在: {image_dir}")

        event_name = image_dir.name
        output_csv = event_output_csv(output_dir, event_name, args)

        try:
            event_result = process_single_event(
                event_name, image_dir, output_dir, args,
                subset_radius, l0_y, l1_y, roi_cols,
            )
        except Exception as exc:
            if not args.json_summary:
                raise
            print(f"  [{event_name}] → ❌ pipeline 异常: {exc}")
            event_result = exception_event(event_name, output_dir, args, exc)
        event_results.append(event_result)
        elapsed = time.time() - t_total
        if args.json_summary and not write_summary(event_results, binary, top_warnings, elapsed):
            return EXIT_FAILURE

        if event_result["ok"]:
            target = output_csv if output_csv is not None else event_frame_dir(
                output_dir, event_name, args)
            warn_count = sum(
                warning_severity(text, event_result.get("_severity"), strict) == SEVERITY_WARN
                for text in event_result["warnings"]
            ) + sum(warning_severity(text) == SEVERITY_WARN for text in top_warnings) + sum(
                entry["severity"] == SEVERITY_WARN for entry in precheck_entries)
            if warn_count:
                print(f"\n⚠️ 完成（{warn_count} 条警告）！耗时 {elapsed:.1f}s → {target}")
            else:
                print(f"\n✅ 完成！耗时 {elapsed:.1f}s → {target}")
        else:
            print(f"\n❌ 事件失败：{first_failure_reason(event_result, strict)}")
            if event_result.get("_legacy_ok"):
                print("   （默认模式下退出码仍为 0；加 --strict 则 exit 3）")
        if strict:
            return EXIT_OK if event_result["ok"] else EXIT_STRICT_EVENT_FAILED
        if not event_result.get("_legacy_ok"):
            return EXIT_FAILURE

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
