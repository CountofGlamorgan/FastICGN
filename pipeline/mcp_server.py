#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Guarded FastMCP stdio facade for the FastICGN pipeline.

The server intentionally contains no numerical implementation.  It validates
agent input, delegates planning/runs to ``dic_pipeline.py``, and reuses its COD
aggregation for retained frame CSV files.

Agent-facing surface (interface revision 3; revision 2 introduced the typed
surface below, revision 3 the trimmed dic_describe default, the wait:false job
model and dic_job_cancel): every tool has a typed,
documented input schema; plan/run tools still take a single ``params``
object.  Rejections are structured (``ok``/``code``/``error`` plus ``field``,
``got``, ``expected``, ``hint``, ``retryable``).  Runs go through a small job
runner that holds the GPU/output locks for exactly as long as the job lives,
so a closed MCP session can never let a second job onto a busy GPU.  Plans and
runs always pass ``--strict`` to the pipeline: exit 3 (an event failed) still
returns the summary, exit 2 (the pipeline's own check refused) is plan_invalid.

Job model (2026-10-05): run tools block until the job ends by default, as
before.  With the control key ``wait: false`` they return as soon as the job
has started (job_id, output_dir, job record path); the job record carries the
progress the runner writes every few seconds and is read with
dic_query_manifest, and dic_job_cancel stops a job by sending SIGTERM to its
runner's process group.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import fcntl
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Callable

import dic_pipeline as pipeline

try:
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError
    from fastmcp.server.middleware import Middleware
except ImportError:  # Unit tests for the guardrails do not require FastMCP.
    FastMCP = None  # type: ignore[assignment]
    ToolError = None  # type: ignore[assignment]
    Middleware = None  # type: ignore[assignment]

try:  # pydantic ships with fastmcp; only the published schemas need it.
    from pydantic import ConfigDict, Field, WithJsonSchema, create_model
except ImportError:  # pragma: no cover - exercised only without fastmcp
    ConfigDict = Field = WithJsonSchema = create_model = None  # type: ignore[assignment]


PIPELINE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = PIPELINE_DIR / "dic_pipeline.py"
# Same lookup dic_pipeline.py uses, so a pip-installed server finds the
# contract in <sys.prefix>/share/fasticgn/ instead of failing at import.
CONTRACT_PATH = pipeline.INTERFACE_CONTRACT_PATH

# Path configuration is shared with dic_pipeline.py: FASTICGN_* environment
# variables, then the optional ~/.config/fasticgn/config.toml, then the
# repository's own build trees, and only then the deprecated lab defaults.
# The server itself still fixes the binary for agents; it is the operator, not
# the agent, who chooses it.
PATH_NOTES: list[str] = []
_GPU_BIN_PATH, GPU_BIN_SOURCE = pipeline.resolve_gpu_bin(None, PATH_NOTES)
GPU_BIN = Path(_GPU_BIN_PATH)
_JOB_ROOT_SETTING, JOB_ROOT_SOURCE = pipeline.resolve_setting("job_root", None, PATH_NOTES)
JOB_ROOT = Path(_JOB_ROOT_SETTING or (Path.home() / "dic_jobs")).expanduser()
if not _JOB_ROOT_SETTING:
    JOB_ROOT_SOURCE = "default:~/dic_jobs"
AUDIT_LOG_NAME = "mcp_audit.log"
CONTROL_DIR_NAME = ".mcp-control"

READ_ROOTS, READ_ALLOWLIST_SOURCE = (lambda pair: (tuple(pair[0]), pair[1]))(
    pipeline.resolve_read_allowlist(PATH_NOTES)
)
# Legacy lab convention: <lab home>/DIC* stayed readable. The comment used to
# say 'only while the hard-coded home really exists' without checking, so the
# rule applied on every host. It is now really conditional: on a machine
# without that directory DIC_HOME is None and the allowlist is exactly
# READ_ROOTS + the job root.
_LEGACY_DIC_HOME = Path(pipeline.LEGACY_LAB_HOME)
DIC_HOME: Path | None = _LEGACY_DIC_HOME if _LEGACY_DIC_HOME.is_dir() else None
MAX_CAMPAIGN_EVENTS = 120
MAX_TIMEOUT_SECONDS = 3600.0
DEFAULT_TIMEOUT_SECONDS = 600.0
MAX_MANIFEST_BYTES = 10 * 1024 * 1024

# Revision of the agent-facing MCP surface (schemas, envelopes, job records).
# Revision 1 was the 2026-07-10 server with untyped params.
MCP_INTERFACE_REVISION = 3
JOB_RECORD_NAME = ".mcp-job.json"
JOB_STDOUT_NAME = ".mcp-pipeline.stdout.log"
JOB_STDERR_NAME = ".mcp-pipeline.stderr.log"
JOB_RECORD_SCHEMA = "fasticgn.mcp_job/1"
# The server waits this long past the job timeout before it stops the job
# runner itself; the runner normally enforces the timeout on its own.
RUNNER_GRACE_SECONDS = 20.0
# Job model (2026-10-05). The runner rewrites the record's progress block this
# often (also when nothing changed: updated_at is its heartbeat); a wait=false
# reply suggests polling every JOB_POLL_AFTER_SECONDS; dic_job_cancel waits up to
# CANCEL_WAIT_SECONDS for the runner to stop the job (the runner gives the
# pipeline 5 s after SIGTERM, then SIGKILL) before it stops the job itself.
PROGRESS_INTERVAL_SECONDS = 3.0
JOB_POLL_AFTER_SECONDS = 5
JOB_POLL_AFTER_SECONDS_MAX = 30
# What a poll needs from the job record. Both are in the record from the start,
# so a poll with these fields is complete and small (about 0.7 KB instead of
# about 3 KB for the whole record). The outcome fields (exit_code, verdict, ...)
# keep appearing only once they are known, as before.
JOB_POLL_FIELDS = ("state", "progress")
CANCEL_WAIT_SECONDS = 20.0
JOB_ACTIVE_STATES = frozenset({"starting", "running"})
# A job's record says exited a moment before its locks are free: the runner writes the record
# and exits, and the server thread that waited for it leaves the locks some tens of
# milliseconds later (measured 25-30 ms). A run tool that finds the GPU held by a job whose
# record is already final waits for the lock in place, up to this long, before it answers
# gpu_busy; dic_job_cancel waits as long as LOCK_RELEASE_WAIT_SECONDS for the locks of the job
# it stopped before it says that they are released (review 2026-10-05 F4).
ENDED_HOLDER_WAIT_SECONDS = 1.0
LOCK_RELEASE_WAIT_SECONDS = 3.0
PROGRESS_EVENT_TOKEN = "@@EVENT@@"
TAIL_LINES = 40
TAIL_BYTES = 4000
MAX_AUDIT_PARAM_BYTES = 8192
DEFAULT_QUERY_MAX_BYTES = 16384
MIN_QUERY_MAX_BYTES = 1024
MAX_QUERY_MAX_BYTES = 65536
QUERY_DETAILS = ("auto", "compact", "full")
QUERY_FIELD_MAX_CHARS = 256
# Plan and run replies (review 2026-10-04 F1): with more events than this, each
# event is one compact record; every reply stays within MAX_REPLY_BYTES of
# compact JSON (structuredContent; the wire carries it twice, as text too).
COMPACT_EVENTS_THRESHOLD = 10
MAX_REPLY_BYTES = 48 * 1024
# The limit covers the whole reply: every plan and run result and every
# rejection from those tools (plan_invalid, timeout, pipeline_failed, ...), the
# omitted notes and the call_id that _dispatch adds last (recheck R2/R3).
CALL_ID_RESERVE = len(',"call_id":""') + 32 + 8
REPLY_CAPPED_TOOLS = frozenset({"dic_plan", "dic_run_event", "dic_run_campaign"})
FAILURE_REASON_CHARS = 240
# Observed on the lab host (RTX 6000 Ada, 2026-10-03): 5-26 s per 800-frame
# 8-bit COD event end to end. The pipeline's own estimate is GPU-only.
TIMEOUT_HEURISTIC_BASE_S = 60.0
TIMEOUT_HEURISTIC_PER_FRAME_S = 0.04
IMAGE_NAME_RULE = (
    "file names must end in a 6-digit frame number followed by .jpg, .jpeg, "
    ".tif, .tiff, .png or .bmp (case-insensitive), e.g. 000123.tif or "
    "C001H001S0001000123.jpg; the lowest frame number is the reference"
)
COD_ONLY_NOTE = (
    "MCP runs COD mode only. Full-field displacement output, --output-format raw, "
    "--image-pattern and --image-order are available from the pipeline CLI "
    "(pipeline/dic_pipeline.py) but not through MCP."
)

PIPELINE_PARAM_KEYS = {
    "photo_dir", "image_dir", "output_dir", "interface_y", "line_y", "cal",
    "subset", "step", "roi_cols", "gauge", "cod_method", "conv_threshold",
    "input_depth", "input_align", "input_normalize", "gpu_device", "keep_frames",
    "max_frames", "events", "event_interface_y",
}
# Campaign-only keys (2026-10-05): pipeline --events and --event-interface-y.
# Published for dic_plan and dic_run_campaign only; with image_dir the
# pipeline's own check refuses them (events_requires_photo_dir,
# event_interface_y_requires_photo_dir). MCP checks only the JSON shape; every
# rule about the values (matching, conflicts, line_y, rows >= 0, commas in a
# pattern) is the pipeline's, reached through its dry-run.
CAMPAIGN_ONLY_KEYS = frozenset({"events", "event_interface_y"})
# Server behaviour, not a parameter: plans and runs always use the pipeline's
# --strict mode, so a plan and its run judge the inputs the same way, an event
# failure is a non-zero exit (3) and the pipeline re-checks everything before
# it writes (exit 2, returned as plan_invalid).
PIPELINE_STRICT_FLAG = "--strict"
PIPELINE_EXIT_STRICT_REFUSED = pipeline.EXIT_INVALID
PIPELINE_EXIT_EVENT_FAILED = pipeline.EXIT_STRICT_EVENT_FAILED
CONTROL_PARAM_KEYS = {"timeout", "overwrite", "wait"}
CLI_ONLY_KEYS = {"output_format", "image_pattern", "image_order"}
OPERATOR_KEYS = {"gpu_bin", "json_summary", "dry_run", "describe", "strict"}
PARAM_SYNONYMS = {
    "subset_size": "subset", "radius": "subset", "subset_radius": "subset",
    "calibration": "cal", "mm_per_px": "cal", "scale": "cal",
    "line_ys": "line_y", "lines": "line_y", "roi": "roi_cols", "roi_x": "roi_cols",
    "output": "output_dir", "out_dir": "output_dir", "outdir": "output_dir",
    "image_path": "image_dir", "images": "image_dir", "input_dir": "image_dir",
    "device": "gpu_device", "gpu": "gpu_device", "depth": "input_depth",
    "method": "cod_method",
}
PARAMS_TOOLS = ("dic_plan", "dic_run_event", "dic_run_campaign")
_INPROCESS_LOCKS: dict[str, threading.Lock] = {}
_INPROCESS_LOCKS_GUARD = threading.Lock()
# Tool calls currently inside _dispatch, and the subset that owns a running
# job. The SIGTERM/SIGHUP handler uses them to close the audit trail and to
# mark detached jobs before the server exits.
_INFLIGHT: dict[str, dict[str, Any]] = {}
_ACTIVE_JOBS: dict[str, dict[str, Any]] = {}
_REGISTRY_GUARD = threading.Lock()
_CALL_CONTEXT = threading.local()


def _utc(timestamp: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def _read_server_version() -> str:
    """Package version from the source tree's pyproject.toml, else metadata."""

    pyproject = pipeline.REPO_ROOT / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
        section = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
        if section:
            match = re.search(r'^version\s*=\s*"([^"]+)"', section.group(1), re.M)
            if match:
                return match.group(1)
    except OSError:
        pass
    try:
        from importlib.metadata import version

        return version("fasticgn-pipeline")
    except Exception:
        return "unknown"


SERVER_VERSION = _read_server_version()


def _detect_gpu_devices() -> tuple[int | None, str]:
    """Number of CUDA devices this server can address, without loading CUDA."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
        if not entries or entries[0].startswith("-"):
            return 0, "CUDA_VISIBLE_DEVICES"
        return len(entries), "CUDA_VISIBLE_DEVICES"
    try:
        return sum(1 for _ in Path("/proc/driver/nvidia/gpus").iterdir()), "/proc/driver/nvidia/gpus"
    except OSError:
        return None, "unknown"


GPU_DEVICE_COUNT, GPU_DEVICE_COUNT_SOURCE = _detect_gpu_devices()


class InterfaceError(Exception):
    """Expected agent-facing rejection with a stable machine-readable code.

    ``field``/``got``/``expected``/``hint``/``retryable`` and ``details`` are
    optional and only make the error actionable; ``code`` and the message
    (returned as ``error``) are the stable part.
    """

    def __init__(self, code: str, message: str, *, field: str | None = None,
                 got: Any = None, expected: Any = None, hint: str | None = None,
                 retryable: bool = False, **details: Any):
        super().__init__(message)
        self.code = code
        self.field = field
        self.got = got
        self.expected = expected
        self.hint = hint
        self.retryable = retryable
        self.details = details

    def result(self, call_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": False, "code": self.code, "error": str(self), "call_id": call_id}
        if self.field is not None:
            payload["field"] = self.field
        if self.got is not None:
            payload["got"] = _bounded_echo(self.got)
        if self.expected is not None:
            payload["expected"] = self.expected
        if self.hint:
            payload["hint"] = self.hint
        payload["retryable"] = self.retryable
        for key, value in self.details.items():
            payload.setdefault(key, value)
        return payload


def _bounded_echo(value: Any, limit: int = 200) -> Any:
    """Echo a rejected value back without letting it bloat the reply."""

    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"... [{len(value)} chars]"
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    if len(text) <= limit:
        return value
    return text[:limit] + f"... [{len(text)} chars]"


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _printable_path(value: str) -> str:
    """A path with undecodable bytes (surrogate-escaped by os.listdir) as \\xNN escapes."""

    return os.fsencode(value).decode("utf-8", "backslashreplace")


def _reject_unsafe_path_characters(value: str, field: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        # A name that is not valid UTF-8 used to fail only when the reply was
        # serialised (isError "surrogates not allowed"; review 2026-10-04 F13).
        shown = _printable_path(value)
        raise InterfaceError(
            "unsafe_path_characters", f"{field} is not valid UTF-8: {shown}",
            field=field, got=shown, hint="rename the file or directory to a UTF-8 name",
        ) from None
    _reject_control_characters(value, field)


def _reject_control_characters(value: str, field: str) -> None:
    characters = sorted({ord(character) for character in value if ord(character) < 32 or ord(character) == 127})
    if characters:
        escaped = ", ".join(f"U+{code:04X}" for code in characters)
        raise InterfaceError(
            "unsafe_path_characters",
            f"{field} contains control characters ({escaped}); C0/DEL are forbidden in MCP paths",
            field=field, hint="remove tabs, newlines and other control characters from the path",
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _absolute_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise InterfaceError(
            "invalid_path", f"{field} must be a non-empty absolute path",
            field=field, got=value, expected="absolute path string",
        )
    if len(value) > 4096:
        raise InterfaceError(
            "invalid_path", f"{field} exceeds the 4096-character path limit",
            field=field, expected="absolute path of at most 4096 characters",
        )
    _reject_unsafe_path_characters(value, field)
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise InterfaceError(
            "invalid_path", f"{field} must be absolute: {value!r}",
            field=field, got=value, expected="absolute path (starting with /)",
        )
    return path


def _make_private_dir(path: Path) -> None:
    """Create ``path`` as 0700 if it is missing; never touch existing modes.

    Earlier revisions re-applied chmod 0700 on every call, which silently
    revoked group access an operator had granted on a shared job root.
    """

    if path.is_dir():
        return
    try:
        path.mkdir(mode=0o700, parents=True)
    except FileExistsError:
        return
    os.chmod(path, 0o700)


def _ensure_job_root() -> Path:
    root = JOB_ROOT.expanduser()
    if root.is_symlink():
        raise InterfaceError("unsafe_job_root", f"write root must not be a symlink: {root}")
    try:
        _make_private_dir(root)
        return root.resolve(strict=True)
    except OSError as exc:
        raise InterfaceError("job_root_unavailable", f"cannot initialize write root {root}: {exc}") from exc


def _allowlist_description(job_root: Path) -> str:
    roots = [str(root) for root in READ_ROOTS]
    if DIC_HOME is not None:
        roots.append(f"{DIC_HOME}/DIC*")
    roots.append(str(job_root))
    return ", ".join(roots)


def _resolve_read_path(value: Any, field: str, *, kind: str | None = None) -> Path:
    path = _absolute_path(value, field)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise InterfaceError(
            "read_path_missing", f"{field} does not exist: {path}", field=field,
            hint="check the path (case matters); dic_probe_images only reads existing directories",
        ) from exc

    job_root = _ensure_job_root()
    allowed = any(_is_relative_to(resolved, root.resolve(strict=False)) for root in READ_ROOTS)
    allowed = allowed or _is_relative_to(resolved, job_root)
    if DIC_HOME is not None:
        home = DIC_HOME.resolve(strict=False)
        if _is_relative_to(resolved, home):
            relative = resolved.relative_to(home)
            allowed = allowed or bool(relative.parts and relative.parts[0].startswith("DIC"))
    if not allowed:
        raise InterfaceError(
            "read_path_denied",
            f"{field} is outside read allowlist ({_allowlist_description(job_root)}): "
            f"{resolved}. Set FASTICGN_READ_ALLOWLIST or the config file "
            "[paths].read_allowlist to change it.",
            field=field, expected=f"a path under one of: {_allowlist_description(job_root)}",
            hint="the allowlist is set by the server operator; an agent cannot change it",
        )
    if kind == "dir" and not resolved.is_dir():
        raise InterfaceError(
            "not_a_directory", f"{field} is not a directory: {resolved}", field=field,
            expected="a directory",
        )
    if kind == "file" and not resolved.is_file():
        raise InterfaceError("not_a_file", f"{field} is not a file: {resolved}", field=field)
    return resolved


def _resolve_write_path(value: Any, field: str) -> Path:
    path = _absolute_path(value, field)
    job_root = _ensure_job_root()
    try:
        resolved = path.resolve(strict=False)
    except OSError as exc:
        raise InterfaceError("invalid_write_path", f"cannot resolve {field}: {path}: {exc}", field=field) from exc
    if resolved == job_root or not _is_relative_to(resolved, job_root):
        raise InterfaceError(
            "write_path_denied", f"{field} must be a child of {job_root}: {resolved}",
            field=field, expected=f"{job_root}/<new name>",
            hint="MCP writes only below its job root; pick a new name there",
        )
    first_component = resolved.relative_to(job_root).parts[0]
    if (
        first_component in {AUDIT_LOG_NAME, CONTROL_DIR_NAME}
        or (first_component.startswith(".gpu-") and first_component.endswith(".lock"))
    ):
        raise InterfaceError(
            "reserved_write_path", f"{field} targets an MCP audit/lock control path: {resolved}",
            field=field, hint=f"use a new name under {job_root} other than {AUDIT_LOG_NAME} or {CONTROL_DIR_NAME}",
        )
    return resolved


def _sample_names(directory: Path, limit: int = 5) -> tuple[list[str], int]:
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return [], 0
    return [repr(name)[1:-1][:120] for name in names[:limit]], len(names)


def _validate_event_images(directory: Path, field: str) -> list[tuple[int, str]]:
    """Reject descendant symlinks before pipeline code scans or opens images."""

    try:
        images = pipeline.find_images(str(directory))
    except OSError as exc:
        raise InterfaceError("image_scan_failed", f"cannot scan {field} {directory}: {exc}", field=field) from exc
    for _, image_name in images:
        image = Path(image_name)
        _reject_unsafe_path_characters(str(image), field)
        if image.is_symlink():
            raise InterfaceError(
                "read_symlink_denied", f"{field} contains a symlink image, which MCP will not follow: {image}",
                field=field, hint="MCP reads only regular image files; point it at the directory holding the real files",
            )
        try:
            resolved = image.resolve(strict=True)
        except OSError as exc:
            raise InterfaceError("read_path_missing", f"image disappeared during validation: {image}", field=field) from exc
        if not resolved.is_file() or not _is_relative_to(resolved, directory):
            raise InterfaceError(
                "read_path_denied", f"image escapes validated event directory {directory}: {resolved}",
                field=field,
            )
    return images


def _validate_source_tree(source: Path, mode: str) -> None:
    if mode == "event":
        _validate_event_images(source, "image_dir")
        return
    try:
        entries = sorted(source.iterdir())
    except OSError as exc:
        raise InterfaceError("image_scan_failed", f"cannot scan photo_dir {source}: {exc}", field="photo_dir") from exc
    for entry in entries:
        # As in 638ad18, every name directly under photo_dir is checked for
        # control characters (C0/DEL), stray files included: guards are never
        # looser than the baseline. Names must also be UTF-8 only where they are
        # used: subdirectories are events (the pipeline lists every one in its
        # plan) and their images are checked below. A stray file such as a
        # GBK-named notes file is accepted, as in 638ad18; requiring UTF-8 of
        # every entry refused such campaigns (review 2026-10-04 recheck R1).
        _reject_control_characters(str(entry), "photo_dir event")
        if not entry.is_dir():
            continue
        _reject_unsafe_path_characters(str(entry), "photo_dir event")
        if entry.is_symlink():
            raise InterfaceError(
                "read_symlink_denied",
                f"photo_dir contains a symlink event directory, which MCP will not follow: {entry}",
                field="photo_dir",
            )
        resolved = entry.resolve(strict=True)
        if not _is_relative_to(resolved, source):
            raise InterfaceError("read_path_denied", f"event escapes photo_dir {source}: {resolved}", field="photo_dir")
        _validate_event_images(resolved, "photo_dir event")


_FRAME_CSV_RE = re.compile(r"^frame_(\d+)(_.*)?\.csv$")


def _validate_frame_files(directory: Path) -> int:
    count = 0
    for frame in directory.glob("frame_*.csv"):
        _reject_unsafe_path_characters(str(frame), "frame CSV")
        if frame.is_symlink():
            raise InterfaceError(
                "read_symlink_denied", f"frames_dir contains a symlink frame CSV: {frame}", field="frames_dir",
            )
        # dic_pipeline.aggregate_cod reads the frame number as the second
        # '_'-separated token of the stem; anything else used to surface as
        # an opaque internal_error ("invalid literal for int()").
        if not _FRAME_CSV_RE.match(frame.name):
            raise InterfaceError(
                "invalid_frame_name",
                f"frames_dir contains {frame.name!r}, which is not a frame CSV name",
                field="frames_dir", expected="frame_<digits>.csv (as written by the binary)",
                hint="aggregate a directory that contains only the frame CSVs of one run (<output_dir>/_frames_<event>)",
            )
        try:
            resolved = frame.resolve(strict=True)
        except OSError as exc:
            raise InterfaceError("read_path_missing", f"frame CSV disappeared: {frame}", field="frames_dir") from exc
        if not resolved.is_file() or not _is_relative_to(resolved, directory):
            raise InterfaceError("read_path_denied", f"frame CSV escapes frames_dir {directory}: {resolved}", field="frames_dir")
        count += 1
    return count


_SECRET_KEY_PART = "token"
_REDACT_MAX_DEPTH = 32


def _secret_digest(value: Any) -> str:
    """What a log keeps of a secret: 'sha256:' and 12 hex digits of a string, the JSON type of anything else."""

    if isinstance(value, str):
        return "sha256:" + hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()[:12]
    return f"<{_json_type(value)}>"


def _redact_secrets(value: Any, _depth: int = 0) -> Any:
    """A copy of ``value`` fit for a log: no cancel token in it (review 2026-10-05 F1).

    Whatever is stored under a key whose name contains 'token', at any depth
    (cancel_token, a misspelt 'token', one wrapped in 'params' as the run tools
    want their arguments), becomes its digest. The argument is not changed.
    """

    if isinstance(value, dict):
        if _depth >= _REDACT_MAX_DEPTH:
            return "<nested too deep to log>"
        return {key: (_secret_digest(item) if _SECRET_KEY_PART in str(key).lower()
                      else _redact_secrets(item, _depth + 1))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if _depth >= _REDACT_MAX_DEPTH:
            return "<nested too deep to log>"
        return [_redact_secrets(item, _depth + 1) for item in value]
    return value


def _audit_params(params: Any) -> Any:
    """Bound what one audit line can carry (huge or hostile inputs)."""

    try:
        text = json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = repr(params)
    encoded = text.encode()
    if len(encoded) <= MAX_AUDIT_PARAM_BYTES:
        return params
    return {
        "_truncated": True,
        "original_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "keys": sorted(str(key) for key in params)[:50] if isinstance(params, dict) else None,
        "preview": encoded[:1024].decode("utf-8", "ignore"),
    }


def _audit_record(tool: str, call_id: str, params: dict[str, Any], result_code: str,
                  extra: dict[str, Any] | None = None) -> None:
    root = _ensure_job_root()
    path = root / AUDIT_LOG_NAME
    record = {
        "time": _utc(),
        "call_id": call_id,
        "tool": tool,
        # Every audit line is written here (tool calls, the middleware for calls the
        # schema rejects, the signal handler, job threads): tokens go first, then the size cap.
        "params": _audit_params(_redact_secrets(params)),
        "result_code": result_code,
    }
    if extra:
        record.update(extra)
    payload = (json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                view = view[written:]
        finally:
            os.close(fd)
    except OSError as exc:
        raise InterfaceError("audit_unavailable", f"cannot append MCP audit log: {exc}") from exc


def _current_call_id() -> str:
    return getattr(_CALL_CONTEXT, "call_id", None) or uuid.uuid4().hex


def _dispatch(tool: str, params: dict[str, Any], operation: Callable[[], dict[str, Any]],
              *, envelope: bool = True) -> dict[str, Any]:
    call_id = uuid.uuid4().hex
    try:
        _audit_record(tool, call_id, params, "started")
    except InterfaceError as exc:
        return exc.result(call_id)

    with _REGISTRY_GUARD:
        _INFLIGHT[call_id] = {"tool": tool, "params": params, "started": time.time()}
    _CALL_CONTEXT.call_id = call_id
    started = time.monotonic()
    try:
        result = operation()
        if not isinstance(result, dict):
            raise TypeError(f"tool implementation returned {type(result).__name__}, expected dict")
        # A success that is not a verdict (job started, job cancelled) names its own audit code.
        audit_code = result.pop("_audit_code", None)
        if audit_code is not None:
            result_code = str(audit_code)
        elif result.get("valid") is False:
            result_code = "invalid_plan"
        else:
            result_code = str(
                result.get("code") or result.get("verdict")
                or ("ok" if result.get("ok", True) else "error")
            )
        if envelope:
            result.setdefault("call_id", call_id)
    except InterfaceError as exc:
        result = exc.result(call_id)
        if tool in REPLY_CAPPED_TOOLS:
            result = _shape_error_reply(result)
        result_code = exc.code
    except SystemExit as exc:
        # argparse and other libraries exit instead of raising.  SystemExit is
        # not an Exception subclass, so before revision 2 it escaped here and
        # killed the server's event loop (review 2026-10-03 S1-5).
        result = InterfaceError(
            "internal_error",
            f"a library called sys.exit({exc.code!r}) while handling {tool}; the call was aborted "
            "and the server keeps running",
            hint="this is a server bug; report the params from this call",
        ).result(call_id)
        result_code = "internal_error:SystemExit"
    except Exception as exc:  # Keep protocol stdout clean and return a bounded error.
        result = {
            "ok": False, "code": "internal_error", "error": str(exc)[:2000], "call_id": call_id,
            "exception": type(exc).__name__, "retryable": False,
        }
        result_code = "internal_error"
    finally:
        _CALL_CONTEXT.call_id = None
        with _REGISTRY_GUARD:
            _INFLIGHT.pop(call_id, None)

    try:
        _audit_record(tool, call_id, params, result_code,
                      extra={"duration_s": round(time.monotonic() - started, 3)})
    except InterfaceError as exc:
        return {
            "ok": False,
            "code": exc.code,
            "error": str(exc),
            "operation_result_code": result_code,
            "call_id": call_id,
            "retryable": False,
        }
    return result


def _params_dict(params: Any) -> Any:
    """Accept a plain dict (Python callers) or the published pydantic model."""

    if isinstance(params, dict):
        return params
    dump = getattr(params, "model_dump", None)
    if callable(dump):
        return dump(exclude_unset=True)
    return params


def _unknown_param_error(unknown: list[str]) -> InterfaceError:
    allowed = sorted(PIPELINE_PARAM_KEYS | CONTROL_PARAM_KEYS)
    reasons: list[str] = []
    cli_only = sorted(set(unknown) & CLI_ONLY_KEYS)
    operator = sorted(set(unknown) & OPERATOR_KEYS)
    other = sorted(set(unknown) - CLI_ONLY_KEYS - OPERATOR_KEYS)
    if cli_only:
        reasons.append(f"{', '.join(cli_only)}: {COD_ONLY_NOTE}")
    if operator:
        reasons.append(
            f"{', '.join(operator)}: fixed by the server (gpu_bin comes from the operator's "
            "FASTICGN_GPU_BIN/config; dic_plan is the dry run; dic_describe returns the contract; "
            "run tools always write summary.json; plans and runs always use --strict)"
        )
    suggestions: dict[str, str] = {}
    for key in other:
        lowered = key.lower()
        if lowered in PARAM_SYNONYMS:
            suggestions[key] = PARAM_SYNONYMS[lowered]
            continue
        close = difflib.get_close_matches(lowered, allowed, n=1, cutoff=0.75)
        if close:
            suggestions[key] = close[0]
    if other:
        for key in other:
            reasons.append(f"{key}: not a parameter"
                           + (f" (did you mean {suggestions[key]}?)" if key in suggestions else "") + ".")
    message = "unsupported parameter(s): " + ", ".join(sorted(unknown)) + ". " + " ".join(reasons)
    return InterfaceError(
        "unknown_params", message, field="params", got=sorted(unknown), expected=allowed,
        hint=("subset is the subset SIZE (odd, 15 = radius 7)" if "subset" in suggestions.values()
              else "remove the listed keys; the input schema of this tool lists every allowed key"),
        suggestions=suggestions or None,
    )


def _require_dict(params: Any) -> dict[str, Any]:
    params = _params_dict(params)
    if not isinstance(params, dict):
        raise InterfaceError(
            "invalid_params", "params must be a JSON object", field="params",
            got=_json_type(params), expected="object",
        )
    unknown = sorted(set(params) - PIPELINE_PARAM_KEYS - CONTROL_PARAM_KEYS)
    if unknown:
        raise _unknown_param_error(unknown)
    return dict(params)


def _number(value: Any, field: str, *, minimum: float | None = None,
            maximum: float | None = None, exclusive_minimum: bool = False,
            hint: str | None = None) -> float:
    bounds = []
    if minimum is not None:
        bounds.append(f"{'>' if exclusive_minimum else '>='} {minimum:g}")
    if maximum is not None:
        bounds.append(f"<= {maximum:g}")
    expected = "finite JSON number" + (f" {' and '.join(bounds)}" if bounds else "")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"{field} must be a finite number"
        if isinstance(value, str):
            message += f" (got the string {value[:40]!r}; send it as a JSON number without quotes)"
        raise InterfaceError(
            "invalid_parameter", message, field=field, got=value, expected=expected, hint=hint,
        )
    result = float(value)
    if not math.isfinite(result):
        raise InterfaceError("invalid_parameter", f"{field} must be finite", field=field, got=str(value),
                             expected=expected, hint=hint)
    if minimum is not None and (result <= minimum if exclusive_minimum else result < minimum):
        operator = ">" if exclusive_minimum else ">="
        raise InterfaceError(
            "parameter_out_of_bounds", f"{field} must be {operator} {minimum}",
            field=field, got=value, expected=expected, hint=hint,
        )
    if maximum is not None and result > maximum:
        raise InterfaceError(
            "parameter_out_of_bounds", f"{field} must be <= {maximum}",
            field=field, got=value, expected=expected, hint=hint,
        )
    return result


def _integer(value: Any, field: str, *, minimum: int | None = None,
             maximum: int | None = None, hint: str | None = None) -> int:
    bounds = []
    if minimum is not None:
        bounds.append(f">= {minimum}")
    if maximum is not None:
        bounds.append(f"<= {maximum}")
    expected = "integer" + (f" {' and '.join(bounds)}" if bounds else "")
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"{field} must be an integer"
        if isinstance(value, float) and value.is_integer():
            message += f" (got {value!r}; send {int(value)} without a decimal point)"
        elif isinstance(value, str):
            message += f" (got the string {value[:40]!r}; send a JSON integer)"
        raise InterfaceError("invalid_parameter", message, field=field, got=value, expected=expected, hint=hint)
    if minimum is not None and value < minimum:
        raise InterfaceError(
            "parameter_out_of_bounds", f"{field} must be >= {minimum}",
            field=field, got=value, expected=expected, hint=hint,
        )
    if maximum is not None and value > maximum:
        raise InterfaceError(
            "parameter_out_of_bounds", f"{field} must be <= {maximum}",
            field=field, got=value, expected=expected, hint=hint,
        )
    return value


def _pair(value: Any, field: str) -> str:
    expected = "two integers as a JSON array [a, b] or the string 'a,b'"
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise InterfaceError(
            "invalid_parameter", f"{field} must be 'a,b' or a two-integer array",
            field=field, got=value, expected=expected,
        )
    if len(parts) != 2:
        raise InterfaceError(
            "invalid_parameter", f"{field} must contain exactly two integers",
            field=field, got=value, expected=expected,
        )
    integers: list[int] = []
    for item in parts:
        if isinstance(item, bool):
            raise InterfaceError("invalid_parameter", f"{field} must contain exactly two integers",
                                 field=field, got=value, expected=expected)
        if isinstance(item, int):
            integers.append(item)
            continue
        if isinstance(item, str):
            try:
                parsed = int(item)
            except ValueError as exc:
                raise InterfaceError(
                    "invalid_parameter", f"{field} must contain exactly two integers",
                    field=field, got=value, expected=expected,
                ) from exc
            if str(parsed) != item.strip() and not (
                item.strip().startswith("+") and str(parsed) == item.strip()[1:]
            ):
                raise InterfaceError("invalid_parameter", f"{field} must contain exactly two integers",
                                     field=field, got=value, expected=expected)
            integers.append(parsed)
            continue
        raise InterfaceError("invalid_parameter", f"{field} must contain exactly two integers",
                             field=field, got=value, expected=expected)
    return f"{integers[0]},{integers[1]}"


def _checked_line_pair(value: Any, field: str = "line_y") -> tuple[int, int]:
    """Two distinct non-negative rows.  Their ORDER is kept as given: COD is
    u(first) - u(second), so the order sets the sign (no convention enforced)."""

    first, second = (int(part) for part in _pair(value, field).split(","))
    expected = "two distinct non-negative integers [L0, L1] (L0 = upper line by convention)"
    if first < 0 or second < 0:
        raise InterfaceError(
            "invalid_parameter", f"{field} values must be non-negative pixel rows",
            field=field, got=value, expected=expected,
            hint="image rows start at 0 at the top; there is no negative or 'full range' shorthand",
        )
    if first == second:
        raise InterfaceError(
            "invalid_parameter", f"{field} values must be distinct",
            field=field, got=value, expected=expected,
        )
    return first, second


def _checked_roi(value: Any, field: str = "roi_cols") -> tuple[int, int]:
    first, second = (int(part) for part in _pair(value, field).split(","))
    expected = "[x_min, x_max] with 0 <= x_min <= x_max (inclusive pixel columns)"
    if first < 0 or second < 0:
        raise InterfaceError(
            "invalid_parameter", f"{field} values must be non-negative pixel columns",
            field=field, got=value, expected=expected,
            hint="omit roi_cols for the full width; there is no -1 or negative shorthand",
        )
    if first > second:
        raise InterfaceError(
            "invalid_parameter", f"{field} must be ordered (x_min <= x_max)",
            field=field, got=value, expected=expected,
            hint=f"did you mean [{second}, {first}]?",
        )
    return first, second


_EVENTS_EXPECTED = ('JSON array of event names or glob patterns, one per item, e.g. ["event10", "event22"] '
                    'or ["event2[1-5]"]')
_EVENT_INTERFACE_Y_EXPECTED = ('JSON array of {"pattern": "<event name or glob>", "interface_y": <row>} objects, '
                               'e.g. [{"pattern": "event2[1-5]", "interface_y": 133}]')


def _checked_events(value: Any) -> str:
    """events -> the pipeline's --events value (comma-joined); only the JSON shape is checked here."""

    field = "events"
    hint = "omit events to run every event; for one event send ['event10']"
    if not isinstance(value, list):
        raise InterfaceError(
            "invalid_events", f"events must be a JSON array of strings, got {_json_type(value)}",
            field=field, got=value, expected=_EVENTS_EXPECTED, hint=hint,
        )
    if not value:
        raise InterfaceError("invalid_events", "events must list at least one event name or pattern",
                             field=field, got=value, expected=_EVENTS_EXPECTED, hint=hint)
    if len(value) > MAX_CAMPAIGN_EVENTS:
        raise InterfaceError(
            "invalid_events", f"events lists {len(value)} items; at most {MAX_CAMPAIGN_EVENTS}",
            field=field, got=len(value), expected=_EVENTS_EXPECTED, hint="use glob patterns such as 'event0*'",
        )
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise InterfaceError(
                "invalid_events", f"events[{index}] must be a string, got {_json_type(item)}",
                field=f"events[{index}]", got=item, expected=_EVENTS_EXPECTED, hint=hint,
            )
        if "," in item:
            # The pipeline's --events is comma-separated: a comma would silently
            # turn one item into two patterns.
            raise InterfaceError(
                "invalid_events",
                f"events[{index}] {item!r} contains a comma; give one event name or pattern per item "
                "(a glob class is written [12], not [1,2])",
                field=f"events[{index}]", got=item, expected=_EVENTS_EXPECTED,
            )
    return ",".join(value)


def _checked_event_interface_y(value: Any) -> list[str]:
    """event_interface_y -> the pipeline's repeated --event-interface-y PATTERN=Y values.

    Only the JSON shape and types are checked here; a negative row, a comma in
    a pattern, an empty pattern, a pattern that matches no selected event and
    conflicting rows are refused by the pipeline's dry-run with its own codes.
    """

    field = "event_interface_y"
    if not isinstance(value, list):
        raise InterfaceError(
            "invalid_event_interface_y", f"event_interface_y must be a JSON array of objects, got {_json_type(value)}",
            field=field, got=value, expected=_EVENT_INTERFACE_Y_EXPECTED,
            hint="omit event_interface_y when every event uses interface_y",
        )
    if not value:
        raise InterfaceError(
            "invalid_event_interface_y", "event_interface_y must hold at least one {pattern, interface_y} entry",
            field=field, got=value, expected=_EVENT_INTERFACE_Y_EXPECTED,
            hint="omit event_interface_y when every event uses interface_y",
        )
    if len(value) > MAX_CAMPAIGN_EVENTS:
        raise InterfaceError(
            "invalid_event_interface_y", f"event_interface_y has {len(value)} entries; at most {MAX_CAMPAIGN_EVENTS}",
            field=field, got=len(value), expected=_EVENT_INTERFACE_Y_EXPECTED,
            hint="one entry per group of events: patterns such as 'event2[1-5]' cover several events",
        )
    arguments: list[str] = []
    for index, item in enumerate(value):
        where = f"event_interface_y[{index}]"
        if not isinstance(item, dict):
            raise InterfaceError(
                "invalid_event_interface_y", f"{where} must be an object {{pattern, interface_y}}, got {_json_type(item)}",
                field=where, got=item, expected=_EVENT_INTERFACE_Y_EXPECTED,
            )
        missing = sorted({"pattern", "interface_y"} - set(item))
        extra = sorted(set(item) - {"pattern", "interface_y"})
        if missing or extra:
            problems = ([f"missing {', '.join(missing)}"] if missing else []) + \
                       ([f"unknown key(s) {', '.join(map(str, extra))}"] if extra else [])
            raise InterfaceError(
                "invalid_event_interface_y", f"{where}: {'; '.join(problems)}",
                field=where, got=item, expected=_EVENT_INTERFACE_Y_EXPECTED,
            )
        pattern, row = item["pattern"], item["interface_y"]
        if not isinstance(pattern, str):
            raise InterfaceError(
                "invalid_event_interface_y", f"{where}.pattern must be a string, got {_json_type(pattern)}",
                field=f"{where}.pattern", got=pattern, expected="an event name or glob such as 'event2[1-5]'",
            )
        if isinstance(row, bool) or not isinstance(row, int):
            message = f"{where}.interface_y must be an integer, got {_json_type(row)}"
            if isinstance(row, float) and row.is_integer():
                message += f" (send {int(row)} without a decimal point)"
            elif isinstance(row, str):
                message += f" (the string {row[:40]!r}; send a JSON integer)"
            raise InterfaceError(
                "invalid_event_interface_y", message, field=f"{where}.interface_y", got=row,
                expected="integer >= 0: the interface row in px of the matched events",
            )
        arguments.append(f"{pattern}={row}")
    return arguments


def _uses_event_selection(params: dict[str, Any]) -> bool:
    return any(key in params for key in CAMPAIGN_ONLY_KEYS)


def _reject_nested_output_dir(output: Path) -> None:
    """A run's output_dir must not lie inside another run directory or a frames directory.

    Output locks are per exact path, so a run nested in another run's tree
    (for example in its _frames_<event>) could otherwise make that run's own
    binary or --strict pre-check reject a foreign entry (review 2026-10-04 F5).
    """

    job_root = _ensure_job_root()
    parts = output.relative_to(job_root).parts
    if any(part.startswith("_frames_") for part in parts):
        raise InterfaceError(
            "nested_output_dir", f"output_dir lies inside a frames directory: {output}",
            field="output_dir", hint="choose a new directory directly under the job root",
        )
    enclosing = _enclosing_run_dir(output.parent, job_root)
    if enclosing is not None:
        raise InterfaceError(
            "nested_output_dir",
            f"output_dir lies inside the run directory {enclosing}, which belongs to another run",
            field="output_dir", hint="choose a new directory that is not inside another run's output_dir",
        )


def _normalize_pipeline_params(raw: Any, expected_mode: str | None = None) -> dict[str, Any]:
    params = _require_dict(raw)
    mcp_warnings: list[dict[str, str]] = []
    has_event = params.get("image_dir") is not None
    has_campaign = params.get("photo_dir") is not None
    if has_event == has_campaign:
        raise InterfaceError(
            "invalid_input_mode", "supply exactly one of image_dir or photo_dir",
            field="image_dir|photo_dir",
            expected="image_dir (one event directory) or photo_dir (directory of event subdirectories)",
        )
    mode = "event" if has_event else "campaign"
    if expected_mode is not None and mode != expected_mode:
        required = "image_dir" if expected_mode == "event" else "photo_dir"
        other_tool = "dic_run_campaign" if expected_mode == "event" else "dic_run_event"
        raise InterfaceError(
            "wrong_input_mode", f"this tool requires {required}", field=required,
            hint=f"use {other_tool} for {'photo_dir' if expected_mode == 'event' else 'image_dir'}",
        )

    source_key = "image_dir" if has_event else "photo_dir"
    normalized: dict[str, Any] = {
        source_key: str(_resolve_read_path(params[source_key], source_key, kind="dir")),
    }
    _validate_source_tree(Path(normalized[source_key]), mode)
    if "output_dir" not in params:
        raise InterfaceError(
            "missing_parameter", "output_dir is required", field="output_dir",
            expected=f"absolute path of a new directory under {JOB_ROOT}",
        )
    normalized["output_dir"] = str(_resolve_write_path(params["output_dir"], "output_dir"))
    _reject_nested_output_dir(Path(normalized["output_dir"]))

    if "cal" not in params:
        raise InterfaceError(
            "missing_parameter", "cal is required", field="cal",
            expected="calibration in mm per pixel, 0 < cal <= 10",
            hint="use 1.0 only for uncalibrated images (then cod_um/1000 is in pixels)",
        )
    normalized["cal"] = _number(
        params["cal"], "cal", minimum=0, maximum=10, exclusive_minimum=True,
        hint="cal is millimetres per pixel, e.g. 46.2 um/px -> 0.0462",
    )

    subset_hint = "subset is the odd subset SIZE (15 = radius 7), not the radius"
    subset = _integer(params.get("subset", 15), "subset", minimum=5, maximum=51, hint=subset_hint)
    if subset % 2 == 0:
        raise InterfaceError(
            "parameter_out_of_bounds", "subset is SIZE and must be odd in [5, 51]",
            field="subset", got=subset, expected="odd integer 5..51", hint=subset_hint,
        )
    if subset < 11:
        mcp_warnings.append({
            "code": "subset_small",
            "message": f"subset={subset} is the subset SIZE (radius {(subset - 1) // 2}); "
                       f"if you meant radius {subset}, use subset={2 * subset + 1}",
        })
    normalized["subset"] = subset
    normalized["step"] = _integer(params.get("step", 1), "step", minimum=1, maximum=20)
    normalized["gauge"] = _integer(params.get("gauge", 35), "gauge", minimum=4, maximum=200)

    if params.get("events") is not None:
        normalized["events"] = _checked_events(params["events"])
    if params.get("event_interface_y") is not None:
        normalized["event_interface_y"] = _checked_event_interface_y(params["event_interface_y"])
        if params.get("line_y") is None and params.get("interface_y") is None:
            # MCP would otherwise answer missing_parameter (interface_y|line_y);
            # the pipeline's own code and message say what the overrides need.
            probe = argparse.Namespace(
                events=None, event_interface_y=normalized["event_interface_y"],
                photo_dir=normalized.get("photo_dir"), line_y=None, interface_y=None,
            )
            issues = pipeline.plan_event_selection(probe, None)["errors"]
            required = [issue for issue in issues if issue["code"] == "event_interface_y_requires_interface_y"]
            raise InterfaceError(
                "event_interface_y_requires_interface_y",
                required[0]["message"] if required else
                "event_interface_y overrides interface_y for the events it matches; give interface_y",
                field="interface_y",
                expected="interface_y (integer row) for every event that no event_interface_y entry matches",
                hint="add interface_y (the default row); event_interface_y cannot be used with line_y",
                errors=issues,
            )

    if params.get("line_y") is not None:
        l0, l1 = _checked_line_pair(params["line_y"])
        normalized["line_y"] = f"{l0},{l1}"
        if l0 > l1:
            mcp_warnings.append({
                "code": "line_y_descending",
                "message": f"line_y=[{l0}, {l1}] puts L0 below L1. COD = u(L0) - u(L1), so its sign is "
                           f"opposite to [{l1}, {l0}] and to the interface_y convention (L0 above)",
            })
        if params.get("interface_y") is not None:
            mcp_warnings.append({
                "code": "interface_y_ignored",
                "message": "line_y overrides interface_y and gauge; interface_y is not used",
            })
        elif "gauge" in params:
            mcp_warnings.append({
                "code": "gauge_ignored", "message": "gauge is only used with interface_y; line_y overrides it",
            })
    elif params.get("interface_y") is not None:
        normalized["interface_y"] = _integer(
            params["interface_y"], "interface_y", minimum=0,
            hint="interface_y is a pixel row counted from the top of the image",
        )
    else:
        raise InterfaceError(
            "missing_parameter",
            "supply interface_y or line_y: " + COD_ONLY_NOTE,
            field="interface_y|line_y",
            expected="interface_y (integer row) or line_y [L0, L1]",
            hint="lines are at interface_y +/- gauge//2 (gauge default 35); or give both rows as line_y",
        )
    if params.get("interface_y") is not None and "interface_y" not in normalized:
        normalized["interface_y"] = _integer(params["interface_y"], "interface_y", minimum=0)
    if params.get("roi_cols") is not None:
        x0, x1 = _checked_roi(params["roi_cols"])
        normalized["roi_cols"] = f"{x0},{x1}"

    cod_method = params.get("cod_method", "naive")
    if cod_method not in {"fillboundary", "naive"}:
        raise InterfaceError(
            "invalid_parameter", "cod_method must be fillboundary or naive", field="cod_method",
            got=cod_method, expected=["fillboundary", "naive"],
        )
    normalized["cod_method"] = cod_method
    normalized["conv_threshold"] = _number(
        params.get("conv_threshold", 0.001), "conv_threshold", minimum=0, exclusive_minimum=True,
    )
    maximum_device = GPU_DEVICE_COUNT - 1 if GPU_DEVICE_COUNT else None
    normalized["gpu_device"] = _integer(
        params.get("gpu_device", 0), "gpu_device", minimum=0, maximum=maximum_device,
        hint=(f"this host exposes {GPU_DEVICE_COUNT} CUDA device(s): 0..{GPU_DEVICE_COUNT - 1}"
              if GPU_DEVICE_COUNT else None),
    )
    if not GPU_DEVICE_COUNT:
        mcp_warnings.append({
            "code": "gpu_device_unverified",
            "message": f"the server could not count CUDA devices ({GPU_DEVICE_COUNT_SOURCE}); "
                       "an invalid gpu_device only fails when the binary starts",
        })

    input_depth = params.get("input_depth")
    input_align = params.get("input_align")
    input_normalize = params.get("input_normalize")
    if input_depth is not None:
        if input_depth not in {"8", "12", "16", "auto"}:
            message = "input_depth must be 8, 12, 16, or auto"
            if isinstance(input_depth, int) and not isinstance(input_depth, bool):
                message += f" as a JSON string (got the integer {input_depth}; send \"{input_depth}\")"
            raise InterfaceError(
                "invalid_parameter", message, field="input_depth", got=input_depth,
                expected=["8", "12", "16", "auto"],
                hint="16-bit TIFFs whose low 4 bits are always 0 (12-bit left-shifted) need "
                     "input_depth='12' with input_align='left'; dic_probe_images recommends a value",
            )
        normalized["input_depth"] = input_depth
    if input_align is not None:
        if input_align not in {"right", "left", "auto"}:
            raise InterfaceError(
                "invalid_parameter", "input_align must be right, left, or auto", field="input_align",
                got=input_align, expected=["right", "left", "auto"],
            )
        if input_depth != "12":
            raise InterfaceError(
                "invalid_parameter", "input_align is legal only with input_depth='12'", field="input_align",
                got={"input_depth": input_depth, "input_align": input_align},
                expected="input_depth='12' together with input_align",
                hint="for 12-bit data left-shifted into 16-bit TIFFs use input_depth='12', input_align='left'",
            )
        normalized["input_align"] = input_align
    if input_normalize is not None:
        if input_normalize not in {"none", "to-8", "to-16", "unit"}:
            raise InterfaceError(
                "invalid_parameter", "invalid input_normalize value", field="input_normalize",
                got=input_normalize, expected=["none", "to-8", "to-16", "unit"],
            )
        normalized["input_normalize"] = input_normalize

    if params.get("max_frames") is not None:
        normalized["max_frames"] = _integer(
            params["max_frames"], "max_frames", minimum=1,
            hint="number of target frames after the reference to run (a quick trial); omit it for all frames",
        )

    keep_frames = params.get("keep_frames", False)
    overwrite = params.get("overwrite", False)
    for key, value in (("keep_frames", keep_frames), ("overwrite", overwrite)):
        if not isinstance(value, bool):
            raise InterfaceError(
                "invalid_parameter", "keep_frames and overwrite must be booleans",
                field=key, got=value, expected="true or false (JSON boolean)",
            )
    normalized["keep_frames"] = keep_frames
    normalized["overwrite"] = overwrite
    # Control key (2026-10-05): false returns once the job has started. null means "not given".
    wait = params.get("wait")
    if wait is not None and not isinstance(wait, bool):
        raise InterfaceError(
            "invalid_parameter", "wait must be a boolean", field="wait", got=wait,
            expected="true (default: block until the job ends) or false (return job_id at once)",
        )
    normalized["wait"] = True if wait is None else wait
    normalized["timeout"] = _number(
        params.get("timeout", DEFAULT_TIMEOUT_SECONDS), "timeout",
        minimum=0, maximum=MAX_TIMEOUT_SECONDS, exclusive_minimum=True,
        hint="seconds; dic_plan reports mcp.suggested_timeout_s",
    )
    normalized["mode"] = mode
    normalized["_mcp_warnings"] = mcp_warnings
    return normalized


def _pipeline_argv(params: dict[str, Any]) -> list[str]:
    """Pipeline argv in --option=value form, always with --strict.

    The attached form keeps argparse from reading a value such as '-5,150'
    as an option (review 2026-10-03 S1-5); the parsed namespace is identical
    to the old '--option value' form for every value MCP accepts. dic_plan
    parses the same argv, so the plan and the run apply the same checks.
    """

    source_key = "image_dir" if params["mode"] == "event" else "photo_dir"
    argv = [f"--{source_key.replace('_', '-')}={params[source_key]}", f"--output-dir={params['output_dir']}"]
    if "line_y" in params:
        argv.append(f"--line-y={params['line_y']}")
    else:
        argv.append(f"--interface-y={params['interface_y']}")
    # Only when given, so a request without them has exactly the old argv.
    if "events" in params:
        argv.append(f"--events={params['events']}")
    for override in params.get("event_interface_y", ()):
        argv.append(f"--event-interface-y={override}")
    argv += [
        f"--cal={params['cal']}",
        f"--subset={params['subset']}",
        f"--step={params['step']}",
        f"--gauge={params['gauge']}",
        f"--cod-method={params['cod_method']}",
        f"--conv-threshold={params['conv_threshold']}",
        f"--gpu-device={params['gpu_device']}",
        f"--gpu-bin={GPU_BIN}",
    ]
    if "roi_cols" in params:
        argv.append(f"--roi-cols={params['roi_cols']}")
    for key in ("input_depth", "input_align", "input_normalize"):
        if key in params:
            argv.append(f"--{key.replace('_', '-')}={params[key]}")
    if "max_frames" in params:
        argv.append(f"--max-frames={params['max_frames']}")
    if params["keep_frames"]:
        argv.append("--keep-frames")
    argv.append(PIPELINE_STRICT_FLAG)
    return argv


def _raise_parse_error(message: str) -> None:
    raise InterfaceError(
        "argument_parse_error", f"pipeline argument parser rejected the request: {message}",
        hint="this should not happen for values that passed MCP validation; report the params",
    )


def _suggested_timeout(target_frames: int) -> float:
    value = TIMEOUT_HEURISTIC_BASE_S + TIMEOUT_HEURISTIC_PER_FRAME_S * max(0, target_frames)
    return float(min(MAX_TIMEOUT_SECONDS, math.ceil(value)))


def _cod_definition(line_ys: Any, cod_method: str) -> dict[str, Any] | None:
    if not isinstance(line_ys, (list, tuple)) or len(line_ys) != 2:
        return None
    l0, l1 = int(line_ys[0]), int(line_ys[1])
    return {
        "L0_y": l0,
        "L1_y": l1,
        "formula": "cod_um = (u_L0 - u_L1) * cal * 1000"
                   + ("; fillboundary first extrapolates each u along du/dy to the midpoint row "
                      f"y0 = {(l0 + l1) / 2:g}" if cod_method == "fillboundary" else ""),
        "line_order": "L0 above L1 (smaller y), the interface_y convention" if l0 < l1
                      else "L0 below L1: COD sign is flipped relative to the interface_y convention",
        "sign_convention_note": "cod_um is the slip across the interface: the relative displacement of its two "
                                "sides, u on the L0 side minus u on the L1 side, not an opening. It is positive "
                                "when u_L0 > u_L1; the order of line_y alone sets the sign",
    }


def _build_plan(params: dict[str, Any]) -> dict[str, Any]:
    parser = pipeline.create_parser()
    # Never let argparse print usage and exit inside the server.
    parser.error = _raise_parse_error  # type: ignore[method-assign]
    try:
        namespace = parser.parse_args(_pipeline_argv(params) + ["--dry-run"])
    except SystemExit as exc:  # pragma: no cover - parser.error already raises
        raise InterfaceError("argument_parse_error", f"pipeline argument parser exited ({exc.code})") from None
    pipeline.apply_path_resolution(namespace)
    plan = pipeline.build_dry_run(namespace)
    for warning in params.get("_mcp_warnings", []):
        plan["warnings"].append({**warning, "source": "mcp"})
    resolved = plan.get("resolved_params") or {}
    target_frames = int((plan.get("estimates") or {}).get("target_frames") or 0)
    plan["mcp"] = {
        "fixed_gpu_binary": str(GPU_BIN),
        "gpu_binary_source": GPU_BIN_SOURCE,
        "read_allowlist": [str(root) for root in READ_ROOTS],
        "read_allowlist_source": READ_ALLOWLIST_SOURCE,
        "job_root": str(JOB_ROOT),
        "job_root_source": JOB_ROOT_SOURCE,
        "path_resolution_notes": list(PATH_NOTES),
        "audit_log_write_excluded_from_pipeline_filesystem_writes": True,
        "server_version": SERVER_VERSION,
        "interface_revision": MCP_INTERFACE_REVISION,
        "gpu_device_count": GPU_DEVICE_COUNT,
        "gpu_device_count_source": GPU_DEVICE_COUNT_SOURCE,
        "suggested_timeout_s": _suggested_timeout(target_frames),
        "suggested_timeout_basis": (
            f"heuristic {TIMEOUT_HEURISTIC_BASE_S:g} s + {TIMEOUT_HEURISTIC_PER_FRAME_S:g} s per target frame "
            "(observed 5-26 s per 800-frame 8-bit event on the lab host); estimates.runtime_s is GPU-only "
            "and optimistic"
        ),
        "cod_definition": _cod_definition(resolved.get("line_ys"), params.get("cod_method", "naive")),
    }
    cod_definition = plan["mcp"]["cod_definition"]
    if cod_definition is not None and "event_interface_y" in params:
        cod_definition["per_event_lines"] = _per_event_lines(plan.get("events"))
        cod_definition["per_event_lines_note"] = (
            "L0_y/L1_y above are the lines of interface_y (the default row); events matched by "
            "event_interface_y use the lines of their own row, listed in per_event_lines"
            + (" (fillboundary extrapolates to each event's interface row)"
               if params.get("cod_method") == "fillboundary" else "")
        )
    plan["ok"] = bool(plan.get("valid"))
    return plan


def _per_event_lines(events: Any) -> list[dict[str, Any]]:
    """Events grouped by the measurement lines they use (dry-run or summary events[].geometry).

    Each group: L0_y, L1_y, interface_y, interface_y_source ('interface_y' = the
    default row, 'event_interface_y' = an override), event_count and the first
    COMPACT_EVENTS_THRESHOLD event names (more_events counts the rest).
    """

    groups: dict[tuple[Any, ...], list[str]] = {}
    for event in events if isinstance(events, list) else []:
        geometry = event.get("geometry") if isinstance(event, dict) else None
        lines = geometry.get("line_ys") if isinstance(geometry, dict) else None
        if not (isinstance(lines, list) and len(lines) == 2):
            continue
        key = (lines[0], lines[1], geometry.get("interface_y"), geometry.get("interface_y_source"))
        groups.setdefault(key, []).append(str(event.get("name")))
    result = []
    for (l0, l1, row, source), names in groups.items():
        entry: dict[str, Any] = {"L0_y": l0, "L1_y": l1, "interface_y": row, "interface_y_source": source,
                                 "event_count": len(names), "events": names[:COMPACT_EVENTS_THRESHOLD]}
        if len(names) > COMPACT_EVENTS_THRESHOLD:
            entry["more_events"] = len(names) - COMPACT_EVENTS_THRESHOLD
        result.append(entry)
    return result


def _lines_brief(events: Any, limit: int = 5) -> str | None:
    """'lines y 123/139 (interface_y 131): event10; lines y 125/141 (event_interface_y 133): event22'."""

    parts = []
    for group in _per_event_lines(events):
        names = group["events"][:limit]
        more = group["event_count"] - len(names)
        parts.append(
            f"lines y {group['L0_y']}/{group['L1_y']} ({group['interface_y_source']} {group['interface_y']}, "
            f"{group['event_count']} event{'s' if group['event_count'] != 1 else ''}): " + ", ".join(names)
            + (f" +{more} more" if more > 0 else "")
        )
    return "; ".join(parts) or None


def _cap_event_selection(result: dict[str, Any], how_to_read: str) -> None:
    """Keep event_selection's name lists to COMPACT_EVENTS_THRESHOLD names each (counts stay exact).

    The view is described in mcp.event_selection_view; nothing happens when no
    list is longer than the threshold (always the case without events and
    event_interface_y: excluded_events and interface_y_overrides are empty).
    """

    selection = result.get("event_selection")
    if not isinstance(selection, dict):
        return
    keep = COMPACT_EVENTS_THRESHOLD
    totals: dict[str, int] = {}
    excluded = selection.get("excluded_events")
    if isinstance(excluded, list) and len(excluded) > keep:
        totals["excluded_events"] = len(excluded)
        selection["excluded_events"] = excluded[:keep]
    for index, override in enumerate(selection.get("interface_y_overrides") or []):
        names = override.get("events") if isinstance(override, dict) else None
        if isinstance(names, list) and len(names) > keep:
            totals[f"interface_y_overrides[{index}].events"] = len(names)
            override["events"] = names[:keep]
    if totals:
        result.setdefault("mcp", {})["event_selection_view"] = {
            "view": "capped", "names_kept_each": keep, "totals": totals,
            "counts": "events_discovered, events_selected and events_excluded stay exact",
            "how_to_read": how_to_read,
        }


def _check_output_state(params: dict[str, Any]) -> None:
    output = Path(params["output_dir"])
    if not output.exists():
        return
    if not output.is_dir():
        raise InterfaceError(
            "output_exists", f"output_dir exists and is not a directory: {output}",
            field="output_dir", hint="choose a new output_dir under the job root",
        )
    if not params["overwrite"]:
        raise InterfaceError(
            "output_exists", f"output_dir already exists; choose a fresh path or set overwrite=true: {output}",
            field="output_dir",
            hint="pick a new directory name; overwrite=true only allows an existing EMPTY directory",
        )
    if any(output.iterdir()):
        raise InterfaceError(
            "nonempty_output_rejected",
            "overwrite=true permits only an existing empty output directory; non-empty output is never mixed or deleted",
            field="output_dir", hint="choose a new output_dir; MCP never reuses or resumes a used directory",
        )


def _read_small_json(path: Path, limit: int = 65536) -> dict[str, Any] | None:
    try:
        with open(path, "rb") as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            return None
        value = json.loads(data.decode("utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _pid_alive(pid: Any) -> bool | None:
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _holder_summary(path: Path, *, mention_output: bool = True) -> tuple[str, dict[str, Any] | None]:
    """Describe who holds an MCP lock, from the JSON the holder wrote into it."""

    info = _read_small_json(path, 8192)
    if not info:
        return "", None
    holder = {key: info.get(key) for key in (
        "tool", "call_id", "output_dir", "gpu_device", "acquired_at", "server_pid", "runner_pid",
    ) if info.get(key) is not None}
    acquired = info.get("acquired_epoch")
    if isinstance(acquired, (int, float)):
        holder["running_for_s"] = round(max(0.0, time.time() - acquired), 1)
    server_alive = _pid_alive(info.get("server_pid"))
    if server_alive is not None:
        holder["server_alive"] = server_alive
    who = str(holder.get("tool") or "another MCP call")
    if holder.get("call_id"):
        who += f" (call {str(holder['call_id'])[:8]})"
    if mention_output and holder.get("output_dir"):
        who += f" writing {holder['output_dir']}"
    if holder.get("acquired_at"):
        who += f" since {holder['acquired_at']}"
    if "running_for_s" in holder:
        who += f" ({holder['running_for_s']} s)"
    if server_alive is False:
        who += "; its MCP session has closed, and the job keeps the lock until it exits"
    return "; held by " + who, holder


def _write_lock_holder(handle, info: dict[str, Any]) -> None:
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(info, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    except OSError:
        pass


@contextlib.contextmanager
def _file_lock(path: Path, busy_code: str, busy_message: str, holder: dict[str, Any] | None = None):
    """Non-blocking flock on ``path``; yields the open handle.

    The handle is never explicitly unlocked: closing it releases the lock only
    when no other process still holds the same open file description, so a job
    runner that inherited the descriptor keeps the lock after this server exits.
    """

    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise InterfaceError("lock_error", f"cannot open MCP lock {path}: {exc}") from exc
    handle = os.fdopen(fd, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            suffix, holder_info = _holder_summary(path, mention_output=busy_code != "output_busy")
            raise InterfaceError(
                busy_code, busy_message + suffix, retryable=True, holder=holder_info,
                hint="wait for that job to finish and retry"
                     + (", or use another gpu_device" if busy_code == "gpu_busy" else ""),
            ) from exc
        info = {"server_pid": os.getpid(), "acquired_at": _utc(), "acquired_epoch": time.time()}
        info.update(holder or {})
        _write_lock_holder(handle, info)
        yield handle
    finally:
        handle.close()


def _holder_job_ended(holder: Any) -> bool:
    """The lock is held for a job whose record already holds a final state: it is about to be released.

    ``holder`` is what the lock file says (tool, call_id, output_dir). The record is read only
    inside the job root and must be that job's (same call_id).
    """

    if not isinstance(holder, dict) or not isinstance(holder.get("output_dir"), str) or not holder.get("call_id"):
        return False
    try:
        directory = Path(holder["output_dir"]).resolve(strict=False)
        if not _is_relative_to(directory, _ensure_job_root()):
            return False
    except (OSError, ValueError, InterfaceError):
        return False
    record = _read_small_json(directory / JOB_RECORD_NAME, 1 << 20)
    return (bool(record) and record.get("call_id") == holder["call_id"]
            and record.get("state") is not None and record.get("state") not in JOB_ACTIVE_STATES)


def _lock_holder_if_held(path: Path) -> dict[str, Any] | None:
    """What the lock file says about its holder while somebody holds the lock, None when it is free.

    A probe: a shared, non-blocking flock that is dropped at once (it fails exactly while a job
    holds the exclusive lock, in this process or another).
    """

    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return _read_small_json(path, 8192) or {}
        return None
    finally:
        os.close(fd)


@contextlib.contextmanager
def _exclusive_lock(path: Path, busy_code: str, busy_message: str, holder: dict[str, Any] | None = None, *,
                    ended_holder_wait: float = 0.0):
    """Non-blocking lock that works across both threads and server processes.

    With ``ended_holder_wait`` a lock held for a job that has already recorded its end is
    waited for, that long at most; any other holder is refused at once, as ever.
    """

    deadline = time.monotonic() + ended_holder_wait
    with contextlib.ExitStack() as stack:
        while True:
            try:
                handle = stack.enter_context(_exclusive_lock_once(path, busy_code, busy_message, holder))
                break
            except InterfaceError as exc:
                if (exc.code != busy_code or time.monotonic() >= deadline
                        or not _holder_job_ended(exc.details.get("holder"))):
                    raise
                time.sleep(0.01)
        yield handle


@contextlib.contextmanager
def _exclusive_lock_once(path: Path, busy_code: str, busy_message: str, holder: dict[str, Any] | None = None):
    key = str(path)
    with _INPROCESS_LOCKS_GUARD:
        inprocess = _INPROCESS_LOCKS.setdefault(key, threading.Lock())
    if not inprocess.acquire(blocking=False):
        suffix, holder_info = _holder_summary(path, mention_output=busy_code != "output_busy")
        raise InterfaceError(
            busy_code, busy_message + suffix, retryable=True, holder=holder_info,
            hint="wait for that job to finish and retry"
                 + (", or use another gpu_device" if busy_code == "gpu_busy" else ""),
        )
    try:
        with _file_lock(path, busy_code, busy_message, holder) as handle:
            yield handle
    finally:
        inprocess.release()
        with _INPROCESS_LOCKS_GUARD:
            if not inprocess.locked() and _INPROCESS_LOCKS.get(key) is inprocess:
                _INPROCESS_LOCKS.pop(key, None)


def _control_dir() -> Path:
    root = _ensure_job_root()
    control = root / CONTROL_DIR_NAME
    if control.is_symlink():
        raise InterfaceError("lock_error", f"MCP control directory must not be a symlink: {control}")
    try:
        _make_private_dir(control)
    except OSError as exc:
        raise InterfaceError("lock_error", f"cannot initialize MCP control directory: {exc}") from exc
    return control


def _gpu_lock_path(device: int) -> Path:
    return _control_dir() / f"gpu-{device}.lock"


def _output_lock_path(output: Path) -> Path:
    digest = hashlib.sha256(os.fsencode(str(output))).hexdigest()
    return _control_dir() / f"output-{digest}.lock"


@contextlib.contextmanager
def _gpu_lock(device: int, holder: dict[str, Any] | None = None):
    with _exclusive_lock(
        _gpu_lock_path(device), "gpu_busy", f"GPU device {device} already has an MCP run in progress",
        {"gpu_device": device, **(holder or {})}, ended_holder_wait=ENDED_HOLDER_WAIT_SECONDS,
    ) as handle:
        yield handle


@contextlib.contextmanager
def _output_lock(output: Path, holder: dict[str, Any] | None = None):
    with _exclusive_lock(
        _output_lock_path(output), "output_busy", f"another MCP run is using output_dir: {output}",
        {"output_dir": str(output), **(holder or {})},
    ) as handle:
        yield handle


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


@contextlib.contextmanager
def _record_lock(path: Path, timeout: float = 2.0):
    """flock the record's directory around a read-modify-write (review 2026-10-04 F9).

    The job runner takes the same lock, so its updates and the server's
    (detach marks, runner-lost adoption) never overwrite each other. Bounded:
    after ``timeout`` the update proceeds unlocked rather than hang a signal
    handler.
    """

    try:
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        yield
        return
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    break
                time.sleep(0.01)
        yield
    finally:
        os.close(fd)


def _merge_job_record(path: Path, **fields: Any) -> None:
    with _record_lock(path):
        record = _read_small_json(path, 1 << 20) or {}
        record.update(fields)
        _write_json_atomic(path, record)


def _finalize_job_record(path: Path, **fields: Any) -> dict[str, Any]:
    """Write a final state only while the record is still starting/running; return the record.

    Several parties can end a job nobody recorded (the server that adopted it,
    dic_job_cancel from this or another session): the first final state stays.
    """

    with _record_lock(path):
        record = _read_small_json(path, 1 << 20) or {}
        if record.get("state") in JOB_ACTIVE_STATES or "state" not in record:
            record.update(fields)
            _write_json_atomic(path, record)
        return record


def _unrecorded_end(record_path: Path, summary_present: bool) -> dict[str, Any]:
    """State of a job that ended without its runner recording it.

    runner_lost when the pipeline left a summary (it ended by itself), else
    cancelled when dic_job_cancel had asked for the job to stop, else job_lost.
    """

    if summary_present:
        return {"state": "runner_lost"}
    request = (_read_small_json(record_path, 1 << 20) or {}).get("cancel_requested_by")
    if isinstance(request, dict):
        return {"state": "cancelled",
                "cancel_reason": f"dic_job_cancel (call {request.get('call_id')}); the job runner was gone, "
                                 "so the job's process group was stopped directly"}
    return {"state": "job_lost"}


def _tail(path: Path, lines: int = TAIL_LINES, limit: int = TAIL_BYTES) -> list[str]:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            data = handle.read()
    except OSError:
        return []
    text = data.decode("utf-8", "replace")
    if size > limit:
        text = text.split("\n", 1)[-1]
    return [line[:400] for line in text.splitlines()[-lines:]]


_ERROR_LINE_RE = re.compile(r"(error|exception|traceback|failed|失败|错误|❌)", re.IGNORECASE)


def _error_lines(*tails: list[str]) -> list[str]:
    found: list[str] = []
    for tail in tails:
        for line in tail:
            if _ERROR_LINE_RE.search(line) and line.strip() not in found:
                found.append(line.strip())
    return found[-10:]


# Which process a pid is, and whether a process group is a given job's, read
# from /proc (review 2026-10-05 F2). One implementation for two users: the
# server (dic_job_cancel) executes this text into _JOB_IDENTITY below, and the
# job runner, which is started from source text, carries it as part of
# _JOB_RUNNER_LIB (its second form stops a job it did not start). Standard
# library only.
_JOB_IDENTITY_SOURCE = r"""
import os
import time


def proc_stat(pid):
    # The fields of /proc/<pid>/stat after "pid (comm) ": [0] state, [2] process group,
    # [19] start time in clock ticks since boot. comm may hold spaces and parentheses,
    # so the split is after the last ") ".
    try:
        with open("/proc/%d/stat" % int(pid), "rb") as handle:
            fields = handle.read().rsplit(b") ", 1)[1].split()
        return fields if len(fields) > 19 else None
    except (OSError, IndexError, TypeError, ValueError):
        return None


def proc_start_ticks(pid):
    # A pid is reused, a pid with its start time is not: together they name one process
    # for as long as the system is up.
    fields = proc_stat(pid)
    try:
        return int(fields[19]) if fields else None
    except ValueError:
        return None


def proc_cmdline(pid):
    try:
        with open("/proc/%d/cmdline" % int(pid), "rb") as handle:
            return handle.read()
    except (OSError, TypeError, ValueError):
        return None


def proc_arguments(pid):
    # The command line of a running process. While a process is in exec (the instant after
    # it was started) it reads as empty: wait for it, up to 0.1 s. A zombie has none.
    cmdline = proc_cmdline(pid)
    tries = 0
    while cmdline == b"" and tries < 20:
        fields = proc_stat(pid)
        if fields is None or fields[0] == b"Z":
            break
        time.sleep(0.005)
        cmdline = proc_cmdline(pid)
        tries += 1
    return cmdline


def procfs_available():
    return proc_stat(os.getpid()) is not None


def argv_names_directory(cmdline, directory):
    # True when one argument of the NUL-separated command line is `directory` itself or a
    # path inside it, alone or as the value of an --option=<path>. A directory that merely
    # starts with the same characters does not count: /jobs/run1 is not /jobs/run10.
    for argument in cmdline.split(b"\0"):
        at = argument.find(directory)
        while at >= 0:
            before = argument[at - 1:at] if at else b""
            after = argument[at + len(directory):at + len(directory) + 1]
            if before in (b"", b"=") and after in (b"", b"/"):
                return True
            at = argument.find(directory, at + 1)
    return False


def job_group_alive(pgid, directory, start_ticks=None):
    # Process group `pgid` exists and is the job that writes `directory`: the pipeline is its
    # leader (argv --output-dir=<directory>, start time `start_ticks` when the job record has
    # one) and the binary a member (its frame directory lies inside). A group number that
    # another process took over is not the job. Zombies do not count. Without procfs the
    # group id is all there is to go by.
    if not isinstance(pgid, int) or isinstance(pgid, bool) or pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    if not procfs_available():
        return True
    needle = os.fsencode(str(directory))
    leader = proc_stat(pgid)
    if leader is not None:
        try:
            if start_ticks is not None and int(leader[19]) != int(start_ticks):
                return False  # the pid, and with it the group number, is another process's now
        except (TypeError, ValueError):
            pass
        if leader[0] != b"Z" and argv_names_directory(proc_arguments(pgid) or b"", needle):
            return True
    # The leader is gone, or it is not the job: look at every member (the binary).
    try:
        entries = [entry for entry in os.listdir("/proc") if entry.isdigit()]
    except OSError:
        return True
    for entry in entries:
        fields = proc_stat(entry)
        try:
            if fields is None or fields[0] == b"Z" or int(fields[2]) != pgid:
                continue
        except ValueError:
            continue
        if argv_names_directory(proc_arguments(entry) or b"", needle):
            return True
    return False
"""

_JOB_IDENTITY: dict[str, Any] = {}
exec(compile(_JOB_IDENTITY_SOURCE, "<fasticgn job identity>", "exec"), _JOB_IDENTITY)  # noqa: S102 - our own text
_proc_start_ticks: Callable[[Any], int | None] = _JOB_IDENTITY["proc_start_ticks"]
_proc_arguments: Callable[[Any], bytes | None] = _JOB_IDENTITY["proc_arguments"]
_procfs_available: Callable[[], bool] = _JOB_IDENTITY["procfs_available"]
_argv_names_directory: Callable[[bytes, bytes], bool] = _JOB_IDENTITY["argv_names_directory"]


# The job runner is started as ``python -c _JOB_RUNNER_SOURCE <spec-json>`` in
# its own session.  It inherits the lock descriptors (pass_fds), starts the
# pipeline in a second session, enforces the timeout, writes the progress
# (finished events and frames, counted from the files the pipeline leaves in
# the output directory) into the job record every few seconds, stops the job
# when it gets SIGTERM (dic_job_cancel) and writes the outcome into the job
# record.  It imports only the standard library so it starts in a few
# milliseconds and does not depend on fastmcp.  _JOB_RUNNER_LIB holds its
# functions (a unit test runs the progress counting from it), _JOB_RUNNER_MAIN
# the two ways it runs: starting a job, or watching one whose runner was lost.
# A final state in the record is never overwritten by the runner: whoever
# records the end of a job first is right (review 2026-10-05).
_JOB_RUNNER_LIB = r"""
import signal

stop_signal = None


def on_terminate(signum, frame):
    # Only a flag: the main loop stops the job and writes the record, so a
    # signal never interrupts a record update halfway.
    global stop_signal
    stop_signal = signum


# First of all, before the other imports: a SIGTERM that arrives while the
# runner is still starting is kept and acted on, not fatal.
signal.signal(signal.SIGTERM, on_terminate)

import fcntl, json, re, subprocess, sys, tempfile, time
""" + _JOB_IDENTITY_SOURCE + r"""

spec = json.loads(sys.argv[1])
record_path = spec["record"]
child = None
FRAME_RE = re.compile(r"^frame_[0-9]+[.]csv$")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def record_lock():
    # Same lock as the server's _record_lock: flock on the output directory,
    # so the two read-modify-write updates of the record never interleave.
    try:
        fd = os.open(os.path.dirname(record_path) or ".", os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return None
    deadline = time.monotonic() + 2.0
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() > deadline:
                return fd
            time.sleep(0.01)


def read_record():
    try:
        with open(record_path, encoding="utf-8") as handle:
            record = json.load(handle)
        return record if isinstance(record, dict) else {}
    except (OSError, ValueError):
        return {}


def is_final(record):
    # A state other than starting/running. A record that cannot be read is not final.
    return record.get("state") is not None and record.get("state") not in ("starting", "running")


def update(**fields):
    # Merge `fields` into the record unless it already holds a final state: then nothing
    # is written and False is returned. Somebody else recorded the end of this job
    # (a cancellation, the server, another runner); the first final state stays.
    lock = record_lock()
    try:
        record = read_record()
        if is_final(record):
            return False
        record.update(fields)
        fd, temporary = tempfile.mkstemp(prefix=".mcp-job.json.", suffix=".tmp",
                                         dir=os.path.dirname(record_path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, record_path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return True
    finally:
        if lock is not None:
            os.close(lock)


def safe_update(**fields):
    # True: written. False: the record is final already. None: it could not be written.
    try:
        return update(**fields)
    except Exception as exc:
        print("job runner: cannot update job record: %s" % exc, file=sys.stderr, flush=True)
        return None


def cancel_requested():
    # dic_job_cancel writes this before it signals anything.
    request = read_record().get("cancel_requested_by")
    return request if isinstance(request, dict) else None


def stop_job():
    if child is None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if child.poll() is not None:
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                return
        time.sleep(0.05)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


# Progress: the events in run order with their target frames, and where the
# pipeline leaves each event's frame directory, COD CSV and binary log.
plan = spec.get("progress") or {}
plan_events = [(str(name), int(frames)) for name, frames in plan.get("events") or []]
seen_frames = {}


def event_path(kind, name):
    template = plan.get(kind)
    if not template:
        return None
    return os.path.join(spec["output_dir"], template.replace(plan.get("token") or "@@EVENT@@", name))


def count_frames(directory):
    try:
        with os.scandir(directory) as entries:
            return sum(1 for entry in entries if FRAME_RE.match(entry.name))
    except (OSError, TypeError):
        return 0


def present(path, directory=False):
    return bool(path) and (os.path.isdir(path) if directory else os.path.exists(path))


def snapshot(started, ended=False):
    # Events run one after another in plan order. An event has started once
    # its frame directory, COD CSV or binary log exists; it is finished once
    # its COD CSV exists, a later event has started or the pipeline has ended.
    # The frames of an event are the most frame files ever seen in its frame
    # directory: the pipeline deletes that directory after the aggregation and
    # only then publishes the COD CSV, and the count must not fall back to 0 in
    # between (seen on the 30 kN campaign, 2026-10-05).
    progress = {"updated_at": utc(), "elapsed_s": round(time.monotonic() - started, 1),
                "events_total": len(plan_events), "events_done": 0,
                "frames_total": sum(frames for _, frames in plan_events), "frames_done": 0,
                "current_event": None}
    try:
        last = -1
        for index, (name, _) in enumerate(plan_events):
            if present(event_path("cod_csv", name)) or present(event_path("frames_dir", name), True) \
                    or present(event_path("binary_log", name)):
                last = index
        for index, (name, frames) in enumerate(plan_events[:last + 1]):
            has_csv = present(event_path("cod_csv", name))
            counted = count_frames(event_path("frames_dir", name))
            seen_frames[name] = max(seen_frames.get(name, 0), counted)
            if has_csv or index < last or ended:
                progress["events_done"] += 1
                progress["frames_done"] += frames if has_csv else min(frames, seen_frames[name])
            else:
                progress["current_event"] = name
                progress["current_event_frames_done"] = min(frames, seen_frames[name])
                progress["current_event_frames_total"] = frames
                progress["frames_done"] += min(frames, seen_frames[name])
    except Exception as exc:
        progress["error"] = str(exc)[:200]
    return progress


def summary_fields():
    # The verdict and event counts of the pipeline's summary, when it wrote one.
    try:
        with open(spec["summary"], encoding="utf-8") as handle:
            summary = json.load(handle)
        counts = summary.get("global") if isinstance(summary.get("global"), dict) else {}
        return {"verdict": summary.get("verdict"), "events_ok": counts.get("events_ok"),
                "events_failed": counts.get("events_failed")}
    except Exception:
        return {}


def stop_group(pgid, alive):
    # For a job this process did not start (the second runner): SIGTERM, 5 s, SIGKILL.
    # `alive` says whether the group is still that job (job_group_alive: start time and
    # arguments) and is asked before each signal, so a group number that another process
    # took over in the meantime is never signalled (review 2026-10-05).
    if not alive():
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.monotonic() + 5.0
    while alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    if alive():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + 5.0
    while alive() and time.monotonic() < deadline:
        time.sleep(0.05)


def finish(state, returncode, started, ended):
    # Write the outcome, unless the record already has one. `state` None: a job this
    # runner did not start has ended and nobody knows its exit code.
    summary = spec.get("summary")
    has_summary = bool(summary) and os.path.isfile(summary)
    request = cancel_requested()
    if state is None:
        # With a summary the pipeline ended by itself. Without one, a cancellation that
        # was asked for is what ended it (dic_job_cancel stopped the process group
        # itself); otherwise nothing is known.
        state = "runner_lost" if has_summary else ("cancelled" if request else "job_lost")
    elif state == "exited" and returncode is not None and returncode < 0 and request:
        state = "cancelled"  # ended by a signal after dic_job_cancel had asked for it
    outcome = {"state": state, "ended_at": utc(), "exit_code": returncode,
               "wall_s": round(time.monotonic() - started, 3), "summary_present": has_summary,
               "progress": snapshot(started, ended=ended)}
    if has_summary:
        outcome.update(summary_fields())
    if state == "cancelled":
        outcome["cancel_reason"] = (
            "dic_job_cancel (call %s)" % request.get("call_id") if request
            else "job runner received " + signal.Signals(stop_signal or signal.SIGTERM).name)
    if safe_update(**outcome) is False:
        print("job runner: the job record already holds a final state (%r); %r was not written"
              % (read_record().get("state"), state), file=sys.stderr, flush=True)
    return state
"""

_JOB_RUNNER_MAIN = r"""

interval = max(0.05, float(spec.get("progress_interval_s") or 3.0))
adopted = spec.get("adopt_pgid")
if adopted:
    # Second runner. The first one was killed while its job ran; the MCP server
    # started this one to watch the job's process group in its place, so the
    # job still gets its timeout, its progress and a final state when the
    # server exits too (review 2026-10-04 recheck R4). It is not the parent of
    # the pipeline, so the exit code stays unknown: runner_lost or job_lost,
    # or cancelled when dic_job_cancel had asked for the job to stop.
    adopted_start = spec.get("adopt_start_ticks")

    def job_alive():
        return job_group_alive(adopted, spec["output_dir"], adopted_start)

    started = time.monotonic() - float(spec.get("elapsed_s") or 0.0)
    if safe_update(runner_pid=os.getpid(), runner_start_ticks=proc_start_ticks(os.getpid()),
                   runner_replaced_at=utc(),
                   note="the first job runner ended without recording an outcome; a second runner watches the "
                        "pipeline and enforces the timeout") is False:
        sys.exit(0)  # the end of the job was recorded while this runner started
    deadline = time.monotonic() + spec["timeout_s"]
    next_progress = time.monotonic() + interval
    state = None
    while job_alive():
        if stop_signal is not None:
            state = "cancelled"
            break
        if time.monotonic() >= deadline:
            state = "timed_out"
            break
        time.sleep(0.1)
        if time.monotonic() >= next_progress:
            safe_update(progress=snapshot(started))
            next_progress = time.monotonic() + interval
    if state is not None:
        stop_group(adopted, job_alive)
    state = finish(state, None, started, ended=state is None)
    sys.exit(128 + (stop_signal or signal.SIGTERM) if state == "cancelled" else (124 if state == "timed_out" else 0))

if is_final(read_record()):
    # Somebody settled this job before its runner was up (a record that looked stale):
    # that state stays, and the pipeline is not started.
    print("job runner: the job record already holds a final state (%r); the pipeline was not started"
          % read_record().get("state"), file=sys.stderr, flush=True)
    sys.exit(75)
env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"
lock_fds = tuple(spec.get("lock_fds", ()))
with open(spec["stdout_log"], "ab", buffering=0) as out, open(spec["stderr_log"], "ab", buffering=0) as err:
    started = time.monotonic()
    try:
        child = subprocess.Popen(spec["argv"], stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                 env=env, start_new_session=True, pass_fds=lock_fds)
    except OSError as exc:
        safe_update(state="spawn_failed", ended_at=utc(), error=str(exc))
        sys.exit(70)
    if safe_update(state="running", runner_pid=os.getpid(), runner_start_ticks=proc_start_ticks(os.getpid()),
                   job_pid=child.pid, job_pgid=child.pid, job_start_ticks=proc_start_ticks(child.pid),
                   running_since=utc(), progress=snapshot(started)) is False:
        # Settled in the moment between the check above and here: stop what was just started.
        stop_job()
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        print("job runner: the job record already holds a final state (%r); the pipeline was stopped"
              % read_record().get("state"), file=sys.stderr, flush=True)
        sys.exit(75)
    timed_out = False
    cancelled = False
    deadline = time.monotonic() + spec["timeout_s"]
    next_progress = time.monotonic() + interval
    while True:
        if stop_signal is not None:
            # A job that ended by itself just before the signal is not cancelled.
            if child.poll() is None:
                cancelled = True
                stop_job()
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            stop_job()
            break
        try:
            child.wait(timeout=min(0.2, remaining))
            break
        except subprocess.TimeoutExpired:
            pass
        if time.monotonic() >= next_progress:
            safe_update(progress=snapshot(started))
            next_progress = time.monotonic() + interval
    returncode = child.returncode
    # Whatever is left of the job's process group goes too, before this runner
    # (which holds the locks) exits: a binary whose pipeline was killed alone
    # must not keep using the GPU after the lock is free (review 2026-10-04 F8).
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    state = finish("cancelled" if cancelled else ("timed_out" if timed_out else "exited"), returncode, started,
                   ended=not (cancelled or timed_out))
if state == "cancelled":
    sys.exit(128 + (stop_signal or signal.SIGTERM))
sys.exit(124 if timed_out else (0 if returncode == 0 else 1))
"""

_JOB_RUNNER_SOURCE = _JOB_RUNNER_LIB + _JOB_RUNNER_MAIN


def _stop_runner(runner: subprocess.Popen, record_path: Path) -> None:
    """Safety net when the runner itself overstays: stop runner and job groups."""

    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(runner.pid, signal.SIGTERM)
    try:
        runner.wait(timeout=15)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(runner.pid, signal.SIGKILL)
        record = _read_small_json(record_path, 1 << 20) or {}
        job_pgid = record.get("job_pgid")
        if isinstance(job_pgid, int) and job_pgid > 1:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(job_pgid, signal.SIGKILL)
        runner.wait()


def _group_alive(pgid: Any) -> bool:
    if not isinstance(pgid, int) or pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _adopt_job_of_lost_runner(runner: subprocess.Popen, record_path: Path, summary_path: Path,
                              deadline: float, respawn: Callable[[int, float], subprocess.Popen | None] | None = None
                              ) -> None:
    """The runner ended; if it recorded no outcome, take over its job (review 2026-10-04 F4).

    A runner killed from outside (OOM killer, an administrator) leaves the record
    at starting/running while the pipeline may still be running in its own
    session. The job then gets a second runner (``respawn``), which watches the
    pipeline's process group in its own session, so the timeout, the progress
    and the final state no longer depend on this server staying alive (recheck
    R4, 2026-10-05). If that runner cannot be started or is lost too, the server
    waits for the process group itself (still holding the locks) and stops it
    at the job's deadline. The exit code of a process nobody waited for is
    unknown. Final states: runner_lost (the job ended by itself; read
    summary.json), timed_out, cancelled, or job_lost (nothing left and no
    summary).
    """

    record = _read_small_json(record_path, 1 << 20) or {}
    if record.get("state") not in JOB_ACTIVE_STATES:
        return
    pgid = record.get("job_pgid")
    if _group_alive(pgid):
        _merge_job_record(
            record_path, runner_lost_at=_utc(), runner_exit=runner.returncode,
            note="the job runner ended without recording an outcome; the MCP server waits for the pipeline "
                 "and enforces the timeout",
        )
        second = None
        if respawn is not None:
            with contextlib.suppress(Exception):
                second = respawn(pgid, max(0.0, deadline - time.monotonic()))
        if second is not None:
            try:
                second.wait(timeout=max(0.0, deadline - time.monotonic()) + RUNNER_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                _stop_runner(second, record_path)
                _merge_job_record(record_path, state="timed_out", ended_at=_utc(),
                                  note="the server stopped an unresponsive job runner")
            if (_read_small_json(record_path, 1 << 20) or {}).get("state") not in JOB_ACTIVE_STATES:
                return
            # The second runner is gone without an outcome as well: go on here.
        while _group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.2)
        if _group_alive(pgid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGTERM)
            grace = time.monotonic() + 5.0
            while _group_alive(pgid) and time.monotonic() < grace:
                time.sleep(0.05)
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGKILL)
            _finalize_job_record(record_path, state="timed_out", ended_at=_utc(), exit_code=None,
                                 summary_present=summary_path.is_file(),
                                 note="the job runner was lost; the MCP server stopped the job at its timeout")
            return
    present = summary_path.is_file()
    _finalize_job_record(record_path, **_unrecorded_end(record_path, present), ended_at=_utc(),
                         runner_exit=runner.returncode, exit_code=None, summary_present=present)


def _warning_code(warning: Any) -> str:
    """Code of a summary warning: 'code: detail' strings or {"code": ...} objects."""

    if isinstance(warning, dict):
        return str(warning.get("code") or warning.get("id") or "warning")
    return str(warning).split(":", 1)[0]


# One refused dry-run check in the failure summary the pipeline writes when its
# --strict check fails: "pipeline_error: <code>: <message>[ [event <name>]]".
_STRICT_ISSUE_RE = re.compile(r"^(?P<code>[a-z][a-z0-9_.]*): (?P<message>.*?)(?: \[event (?P<event>.*)\])?$", re.S)


def _strict_refusal_errors(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """The dry-run errors behind a --strict refusal (exit 2), shaped like plan errors."""

    errors: list[dict[str, Any]] = []
    first_detail = None
    for text in summary.get("warnings") or []:
        if not isinstance(text, str):
            continue
        code, _, detail = text.partition(": ")
        if code != "pipeline_error":
            continue
        first_detail = first_detail or detail
        match = _STRICT_ISSUE_RE.match(detail)
        if match is None:
            continue
        item: dict[str, Any] = {"code": match["code"], "message": match["message"]}
        if match["event"] is not None:
            item["event"] = match["event"]
        errors.append(item)
    if not errors:
        errors.append({
            "code": "strict_check_failed",
            "message": first_detail or "the pipeline's --strict check failed without naming a check",
        })
    return errors


def _warning_labels(entry: dict[str, Any]) -> list[str]:
    """'code(severity)' from a summary's warning_details, else the bare warning codes."""

    details = entry.get("warning_details")
    if isinstance(details, list) and details:
        labels: list[str] = []
        for item in details:
            if not isinstance(item, dict):
                continue
            label = str(item.get("code") or "warning")
            if item.get("severity"):
                label += f"({item['severity']})"
            if label not in labels:
                labels.append(label)
        return labels
    return sorted({_warning_code(w) for w in entry.get("warnings") or []})


def _cod_brief(event: dict[str, Any]) -> list[str]:
    """Measurement rows actually used and the COD row count / peak, when the summary has them."""

    parts: list[str] = []
    geometry = event.get("geometry")
    lines = geometry.get("line_ys") if isinstance(geometry, dict) else None
    if isinstance(lines, list) and len(lines) == 2:
        parts.append(f"lines y {lines[0]}/{lines[1]}")
    stats = event.get("cod_stats")
    if isinstance(stats, dict) and isinstance(stats.get("rows"), int):
        text = f"COD {stats['rows']} rows"
        extremes = [value for value in (stats.get("max_um"), stats.get("min_um"))
                    if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if extremes:
            text += f", peak {max(extremes, key=abs):g} um"
        parts.append(text)
    return parts


BRIEF_EVENTS = 3
BRIEF_FAILED_NAMES = 10


def _event_line(event: dict[str, Any]) -> str:
    valid = event.get("valid_ratio")
    zncc = event.get("zncc_median")
    labels = _warning_labels(event)
    return (
        ", ".join([
            f"{event.get('name')}: {event.get('status')}",
            f"{event.get('frames')} frames",
            f"valid_ratio {valid if valid is None else round(valid, 4)}",
            f"zncc_median {zncc if zncc is None else round(zncc, 4)}",
            *_cod_brief(event),
        ])
        + (f", warnings {','.join(labels)}" if labels else "")
    )


def _event_brief(summary: dict[str, Any], selection: bool = False) -> str:
    """One line: verdict, the failed events by name, then up to three events, failed ones first.

    ``selection`` (a run with events or event_interface_y) adds how many events
    the filter selected and which events used which measurement lines.
    """

    events = [event for event in summary.get("events") or [] if isinstance(event, dict)]
    parts = [f"verdict {summary.get('verdict')}"]
    glob = summary.get("global") or {}
    if isinstance(glob, dict) and "events_ok" in glob:
        parts.append(f"{glob.get('events_ok')}/{len(events)} events ok")
    chosen = summary.get("event_selection")
    if selection and isinstance(chosen, dict):
        if chosen.get("events_argument") is not None:
            parts.append(f"events filter selected {chosen.get('events_selected')} of "
                         f"{chosen.get('events_discovered')} ({chosen.get('events_excluded')} excluded)")
        lines = _lines_brief(events)
        if lines:
            parts.append(lines)
    failed = [event for event in events if event.get("status") != "ok"]
    if failed:
        names = [str(event.get("name")) for event in failed]
        parts.append("failed: " + ", ".join(names[:BRIEF_FAILED_NAMES])
                     + (f" (+{len(names) - BRIEF_FAILED_NAMES} more)" if len(names) > BRIEF_FAILED_NAMES else ""))
    top = _warning_labels(summary)
    if top:
        parts.append(f"top-level warnings {','.join(top)}")
    shown = (failed + [event for event in events if event.get("status") == "ok"])[:BRIEF_EVENTS]
    parts.extend(_event_line(event) for event in shown)
    if len(events) > len(shown):
        parts.append(f"... {len(events) - len(shown)} more events")
    return "; ".join(parts)


def _ordered(result: dict[str, Any], first: tuple[str, ...]) -> dict[str, Any]:
    """The same dict with the keys in ``first`` moved to the front (JSON readers see them first)."""

    return {**{key: result[key] for key in first if key in result},
            **{key: value for key, value in result.items() if key not in first}}


def _compact_event(event: Any, lines: bool = False) -> Any:
    """One line per event: name, status, key numbers and the warn/fail codes (F1).

    ``lines`` (runs with event_interface_y) adds the event's line_ys,
    interface_y and interface_y_source.
    """

    if not isinstance(event, dict):
        return event
    record: dict[str, Any] = {"name": event.get("name"), "status": event.get("status"),
                              "frames": event.get("frames")}
    if lines:
        record.update(_event_lines_record(event))
    for key in ("valid_ratio", "zncc_median"):
        value = event.get(key)
        record[key] = round(value, 4) if isinstance(value, float) else value
    stats = event.get("cod_stats")
    if isinstance(stats, dict):
        record["cod_rows"] = stats.get("rows")
        extremes = [value for value in (stats.get("max_um"), stats.get("min_um"))
                    if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if extremes:
            record["cod_peak_um"] = max(extremes, key=abs)
    labels = [label for label in _warning_labels(event) if not label.endswith("(info)")]
    if labels:
        record["warnings"] = labels
    return record


def _event_lines_record(event: dict[str, Any]) -> dict[str, Any]:
    """line_ys, interface_y and interface_y_source of an event's geometry (compact views, event_interface_y)."""

    geometry = event.get("geometry") if isinstance(event.get("geometry"), dict) else {}
    return {key: geometry.get(key) for key in ("line_ys", "interface_y", "interface_y_source")}


def _compact_plan_event(event: Any, errors: list[Any], warnings: list[Any], lines: bool = False) -> Any:
    if not isinstance(event, dict):
        return event
    probe = event.get("first_frame") if isinstance(event.get("first_frame"), dict) else {}
    state = event.get("output_state") if isinstance(event.get("output_state"), dict) else {}
    name = event.get("name")
    record = {key: event.get(key) for key in (
        "name", "images", "reference_frame", "target_frames", "target_frames_available")}
    record.update({"width": probe.get("width"), "height": probe.get("height"),
                   "container_bit_depth": probe.get("container_bit_depth"),
                   "binary_expectation": state.get("binary_expectation")})
    if lines:
        record.update(_event_lines_record(event))
    for key, items in (("errors", errors), ("warnings", warnings)):
        codes = sorted({str(item.get("code")) for item in items
                        if isinstance(item, dict) and item.get("event") == name})
        if codes:
            record[key] = codes
    return record


def _reply_fits(result: dict[str, Any]) -> bool:
    """The reply, its omitted notes and the call_id _dispatch adds last fit MAX_REPLY_BYTES."""

    return _json_size(result) + CALL_ID_RESERVE <= MAX_REPLY_BYTES


Fits = Callable[..., bool]


def _fit_reply(result: dict[str, Any], reductions: list[tuple[str, Callable[[dict[str, Any], Fits], Any]]],
               holder: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply ``reductions`` in order until the reply fits MAX_REPLY_BYTES (F1, recheck R2/R3).

    Each reduction gets the reply and ``fits``; ``fits(note)`` also counts a note
    that is not recorded yet. It returns a note (what was left out, how many,
    where to read it in full) or None when it had nothing to do. The notes go to
    ``holder["omitted"]`` (mcp.omitted for plans and runs, the top level for
    rejections) and count against the limit, so adding them can no longer push
    a reply that just fitted over it.
    """

    holder = result.setdefault("mcp", {}) if holder is None else holder
    omitted: list[Any] = []

    def fits(pending: dict[str, Any] | None = None) -> bool:
        notes = omitted + ([pending] if pending else [])
        if notes:
            holder["omitted"] = notes
        try:
            return _reply_fits(result)
        finally:
            if omitted:
                holder["omitted"] = omitted
            else:
                holder.pop("omitted", None)

    for _name, reduce in reductions:
        if fits():
            break
        note = reduce(result, fits)
        if note:
            omitted.append(note)
            holder["omitted"] = omitted
    return result


def _trim_events(result: dict[str, Any], fits: Fits, is_problem: Callable[[Any], bool],
                 note: dict[str, Any]) -> dict[str, Any] | None:
    """Keep as many events as fit, the problem ones first; say how many were left out."""

    current = result.get("events")
    if not isinstance(current, list) or not current:
        return None
    ordered = [event for event in current if is_problem(event)] + [event for event in current if not is_problem(event)]
    total = len(ordered)

    def entry(count: int) -> dict[str, Any]:
        return {**note, "field": "events", "kept": count, "omitted": total - count}

    low, high = 0, total  # the reply grows with the count: keep the largest count that fits
    while low < high:
        middle = (low + high + 1) // 2
        result["events"] = ordered[:middle]
        if fits(entry(middle)):
            low = middle
        else:
            high = middle - 1
    result["events"] = ordered[:low]
    return entry(low)


def _trim_lists(result: dict[str, Any], keys: tuple[str, ...], keep: int, where: str) -> dict[str, Any] | None:
    trimmed = {}
    for key in keys:
        items = result.get(key)
        if isinstance(items, list) and len(items) > keep:
            trimmed[key] = len(items)
            result[key] = items[:keep]
    if not trimmed:
        return None
    return {"field": ", ".join(trimmed), "kept_each": keep, "total": trimmed, "where": where}


def _one_line(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _failure_line(event: dict[str, Any]) -> str:
    """Why an event failed, in one line: its first fail-level warning as 'code: detail'."""

    details = [item for item in event.get("warning_details") or [] if isinstance(item, dict)]
    chosen = (next((item for item in details if item.get("severity") == "fail"), None)
              or next((item for item in details if item.get("severity") == "warn"), None))
    if chosen is not None:
        text = str(chosen.get("code") or "warning")
        if chosen.get("detail"):
            text += f": {chosen['detail']}"
    else:
        strings = [warning for warning in event.get("warnings") or [] if isinstance(warning, str)]
        text = strings[0] if strings else f"status {event.get('status')}"
    return _one_line(text, FAILURE_REASON_CHARS)


def _failure_reasons(events: list[Any]) -> list[dict[str, Any]]:
    """The failed events grouped by that line, identical lines merged (recheck R3)."""

    groups: dict[str, list[str]] = {}
    for event in events:
        if isinstance(event, dict) and event.get("status") != "ok":
            groups.setdefault(_failure_line(event), []).append(str(event.get("name")))
    return [{"reason": reason, "count": len(names), "events": names} for reason, names in groups.items()]


def _grouped_nulled_paths(nulled: Any) -> Any:
    """mcp.nulled_paths in the compact view: one entry per field and reason, with a count (recheck R3).

    With keep_frames=false every event has the same entry (its frame directory
    was deleted); 120 of them took 26 KB of a 48 KiB reply.
    """

    if not isinstance(nulled, list):
        return nulled
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in nulled:
        if isinstance(item, dict):
            groups.setdefault((str(item.get("field")), str(item.get("reason"))), []).append(item)
    grouped = []
    for (field, reason), items in groups.items():
        names = [str(item.get("event")) for item in items]
        entry = {"field": field, "reason": reason, "count": len(items), "events": names[:COMPACT_EVENTS_THRESHOLD]}
        if len(names) > COMPACT_EVENTS_THRESHOLD:
            entry["more_events"] = len(names) - COMPACT_EVENTS_THRESHOLD
        entry["example_path"] = items[0].get("path")
        grouped.append(entry)
    return grouped


def _shape_run_reply(summary: dict[str, Any], output: Path, selection: bool = False) -> dict[str, Any]:
    """Run results: ok/verdict/mcp.brief first, compact events above the threshold, hard size cap (F1).

    Under the cap the failed events, one line of reason per failed event and the
    way to read everything in full are the last things to go (recheck R3).
    ``selection`` (events or event_interface_y given) adds each event's lines to
    the compact records.
    """

    events = summary.get("events") if isinstance(summary.get("events"), list) else []
    summary["mcp"] = _ordered(summary.get("mcp") or {}, ("brief", "job"))
    mcp = summary["mcp"]
    summary_file = str(output / "summary.json")
    how = (f"dic_query_manifest(run_dir={str(output)!r}, documents=['summary.json'], fields=['events[3]']) "
           "returns event 3 in full; fields=['events[0:5]'] a slice")
    _cap_event_selection(summary, f"dic_query_manifest(run_dir={str(output)!r}, documents=['summary.json'], "
                                  "fields=['event_selection'], detail='full')")
    # Notes name files relative to mcp.output_dir and point at events_view.how_to_read
    # instead of repeating it: at a small cap the notes themselves used to fill the reply.
    in_summary = "summary.json (mcp.summary_json; mcp.events_view.how_to_read)"

    def compact(result: dict[str, Any], fits: Fits | None = None) -> dict[str, Any] | None:
        if "events_view" in mcp:
            return None
        result["events"] = [_compact_event(event, lines=selection) for event in events]
        view: dict[str, Any] = {
            "view": "compact", "events": len(events), "threshold": COMPACT_EVENTS_THRESHOLD,
            "failed": [str(event.get("name")) for event in events
                       if isinstance(event, dict) and event.get("status") != "ok"],
        }
        reasons = _failure_reasons(events)
        if reasons:
            view["failure_reasons"] = reasons
        view.update(full_summary=summary_file, how_to_read=how)
        mcp["events_view"] = view
        if isinstance(mcp.get("nulled_paths"), list):
            mcp["nulled_paths"] = _grouped_nulled_paths(mcp["nulled_paths"])
        return {"field": "events", "view": "compact", "where": in_summary}

    if len(events) > COMPACT_EVENTS_THRESHOLD:
        compact(summary)

    def drop_tails(result: dict[str, Any], fits: Fits) -> dict[str, Any] | None:
        if "stdout_tail" not in mcp and "stderr_tail" not in mcp:
            return None
        mcp.pop("stdout_tail", None)
        mcp.pop("stderr_tail", None)
        return {"field": "mcp.stdout_tail, mcp.stderr_tail",
                "where": f"{JOB_STDOUT_NAME}, {JOB_STDERR_NAME} in mcp.output_dir"}

    def drop_params(result: dict[str, Any], fits: Fits) -> dict[str, Any] | None:
        return {"field": "params", "where": in_summary} if result.pop("params", None) is not None else None

    def minimal(result: dict[str, Any], fits: Fits) -> dict[str, Any]:
        for key in [key for key in result if key not in {"ok", "verdict", "global", "mcp"}]:
            result.pop(key)
        for key in [key for key in mcp if key not in {"brief", "job", "events_view", "output_dir", "summary_json",
                                                      "job_record", "omitted", "warnings", "error_lines", "hint"}]:
            mcp.pop(key)
        return {"field": "all but ok, verdict, global and the essential mcp fields (brief, job, events_view "
                         "with the failed events and their reasons, error_lines, hint)", "where": in_summary}

    def shorten_failure_lists(result: dict[str, Any], fits: Fits) -> dict[str, Any] | None:
        view = mcp.get("events_view") if isinstance(mcp.get("events_view"), dict) else {}
        changed = False
        for keep in (50, 20, 10, 5, 2, 1):
            failed = view.get("failed")
            if isinstance(failed, list) and len(failed) > keep:
                view.setdefault("failed_total", len(failed))
                view["failed"] = failed[:keep]
                changed = True
            reasons = view.get("failure_reasons")
            if isinstance(reasons, list):
                if len(reasons) > keep:
                    view.setdefault("failure_reasons_total", len(reasons))
                    view["failure_reasons"] = reasons = reasons[:keep]
                    changed = True
                for group in reasons:
                    if isinstance(group, dict) and isinstance(group.get("events"), list) and len(group["events"]) > keep:
                        group["events"] = group["events"][:keep]  # "count" keeps the number
                        changed = True
            for key in ("warnings", "error_lines"):
                if isinstance(mcp.get(key), list) and len(mcp[key]) > keep:
                    mcp[key] = mcp[key][:keep]
                    changed = True
            limit = {50: 2000, 20: 1000, 10: 600, 5: 400, 2: 300, 1: 200}[keep]
            if isinstance(mcp.get("brief"), str) and len(mcp["brief"]) > limit:
                mcp["brief"] = mcp["brief"][:limit] + " ..."
                changed = True
            if fits():
                break
        if not changed:
            return None
        return {"field": "events_view.failed/failure_reasons, warnings, error_lines, brief (shortened; "
                         "failed_total and count keep the numbers)", "where": in_summary}

    def last_resort(result: dict[str, Any], fits: Fits) -> dict[str, Any]:
        view = mcp.get("events_view") if isinstance(mcp.get("events_view"), dict) else {}
        failed = [str(event.get("name")) for event in events if isinstance(event, dict) and event.get("status") != "ok"]
        essentials = {
            "brief": str(mcp.get("brief") or "")[:300],
            "job": mcp.get("job"),
            "output_dir": mcp.get("output_dir"),
            "summary_json": summary_file,
            "events_view": {"failed": failed[:5], "failed_total": len(failed),
                            "failure_reasons": [{**group, "reason": _one_line(group.get("reason", ""), 160),
                                                 "events": group.get("events", [])[:2]}
                                                for group in (view.get("failure_reasons") or [])[:1]
                                                if isinstance(group, dict)],
                            "how_to_read": how},
            "error_lines": [_one_line(line, 160) for line in (mcp.get("error_lines") or [])[:2]],
        }
        mcp.clear()
        mcp.update(essentials)
        for key in [key for key in result if key not in {"ok", "verdict", "mcp"}]:
            result.pop(key)
        return {"field": "all but ok, verdict and the essential mcp fields", "where": in_summary}

    _fit_reply(summary, [
        ("compact", compact),
        ("tails", drop_tails),
        ("params", drop_params),
        ("lists", lambda result, fits: _trim_lists(result, ("warnings", "warning_details"), 20, in_summary)),
        ("events", lambda result, fits: _trim_events(
            result, fits, lambda event: isinstance(event, dict) and event.get("status") != "ok",
            {"where": in_summary})),
        ("minimal", minimal),
        ("failure_lists", shorten_failure_lists),
        ("last_resort", last_resort),
    ])
    return _ordered(summary, ("ok", "verdict", "mcp", "global"))


def _shape_plan_reply(plan: dict[str, Any], selection: bool = False) -> dict[str, Any]:
    """Plans: verdict fields first, compact events above the threshold, hard size cap (F1).

    ``selection`` (events or event_interface_y given) adds each event's lines
    to the compact records and points how_to_read at events=['<event>'].
    """

    events = plan.get("events") if isinstance(plan.get("events"), list) else []
    errors = [item for item in plan.get("errors") or [] if isinstance(item, dict)]
    warnings = [item for item in plan.get("warnings") or [] if isinstance(item, dict)]
    mcp = plan.setdefault("mcp", {})
    if selection:
        how = ("dic_plan with events=['<event>'] and the same other params returns that event's full plan "
               "(probe, output_state, binary_command, geometry)")
    else:
        how = ("dic_plan with image_dir=<photo_dir>/<event> and the same other params returns that event's full "
               "plan (probe, output_state, binary_command)")
    _cap_event_selection(plan, "event_selection.excluded_events = the discovered events minus the selected ones "
                               "(events[]); the pipeline CLI's --dry-run prints every name")

    def compact(result: dict[str, Any], fits: Fits | None = None) -> dict[str, Any] | None:
        if "events_view" in mcp:
            return None
        result["events"] = [_compact_plan_event(event, errors, warnings, lines=selection) for event in events]
        mcp["events_view"] = {
            "view": "compact", "events": len(events), "threshold": COMPACT_EVENTS_THRESHOLD,
            "with_errors": sorted({str(item.get("event")) for item in errors if item.get("event") is not None}),
            "how_to_read": how,
        }
        return {"field": "events", "view": "compact", "where": "mcp.events_view.how_to_read"}

    if len(events) > COMPACT_EVENTS_THRESHOLD:
        compact(plan)

    def drop_limitations(result: dict[str, Any], fits: Fits) -> dict[str, Any] | None:
        if result.pop("known_limitations", None) is None:
            return None
        return {"field": "known_limitations", "where": "dic_describe(section='known_limitations')"}

    def minimal(result: dict[str, Any], fits: Fits) -> dict[str, Any]:
        for key in [key for key in result if key not in {"ok", "valid", "exit_code", "errors", "mcp", "estimates"}]:
            result.pop(key)
        if isinstance(result.get("errors"), list):
            result["errors"] = result["errors"][:20]
        return {"field": "all but ok, valid, exit_code, the first 20 errors, mcp and estimates",
                "where": "mcp.events_view.how_to_read"}

    def shorten(result: dict[str, Any], fits: Fits) -> dict[str, Any] | None:
        view = mcp.get("events_view") if isinstance(mcp.get("events_view"), dict) else {}
        before = _json_size(result)
        keep = 10
        for keep in (10, 5, 2, 1):
            if isinstance(result.get("errors"), list):
                result["errors"] = [{**item, "message": _one_line(item.get("message", ""), 300)}
                                    if isinstance(item, dict) else item for item in result["errors"][:keep]]
            names = view.get("with_errors")
            if isinstance(names, list) and len(names) > keep:
                view.setdefault("with_errors_total", len(names))
                view["with_errors"] = names[:keep]
            if fits():
                break
        if _json_size(result) == before:
            return None
        return {"field": "errors, mcp.events_view.with_errors (shortened)", "kept_each": keep,
                "total": {"errors": len(errors)}, "where": "mcp.events_view.how_to_read"}

    def last_resort(result: dict[str, Any], fits: Fits) -> dict[str, Any]:
        kept = {key: result.get(key) for key in ("ok", "valid", "exit_code")}
        first = next((item for item in result.get("errors") or [] if isinstance(item, dict)), None)
        result.clear()
        result.update(kept)
        result["errors"] = [{**first, "message": _one_line(first.get("message", ""), 200)}] if first else []
        mcp.clear()
        mcp["events_view"] = {"events": len(events), "how_to_read": how}
        result["mcp"] = mcp
        return {"field": "all but ok, valid, exit_code and the first error", "total": {"errors": len(errors)},
                "where": "mcp.events_view.how_to_read"}

    _fit_reply(plan, [
        ("compact", compact),
        ("limitations", drop_limitations),
        ("lists", lambda result, fits: _trim_lists(result, ("warnings", "notes"), 30, "dic_plan on fewer events")),
        ("events", lambda result, fits: _trim_events(
            result, fits, lambda event: isinstance(event, dict) and bool(event.get("errors")),
            {"where": "mcp.events_view.how_to_read"})),
        ("errors", lambda result, fits: _trim_lists(result, ("errors",), 50, "dic_plan on fewer events")),
        ("minimal", minimal),
        ("shorten", shorten),
        ("last_resort", last_resort),
    ])
    return _ordered(plan, ("ok", "valid", "exit_code", "errors", "warnings", "notes", "mcp", "estimates"))


def _shape_error_reply(result: dict[str, Any]) -> dict[str, Any]:
    """Rejections from plan and run tools obey MAX_REPLY_BYTES too (review 2026-10-04 recheck R2).

    A plan_invalid from a run tool used to carry every plan error twice (as the
    JSON text of error and as errors): 79 KB for 120 events, 212 KB with long
    event names. Replies that fit are returned unchanged. Otherwise the tails
    go first (they are in the output directory's logs), then the tails of
    errors, warnings and error_lines (plan_invalid keeps error equal to the JSON
    text of the errors kept), then long strings; the top-level omitted says what
    was left out, how many and how to read it in full.
    """

    if _json_size(result) <= MAX_REPLY_BYTES:
        return result
    output = result.get("output_dir")
    if result.get("code") == "plan_invalid":
        how = (f"the pipeline's failure summary lists every refused check: dic_query_manifest(run_dir={output!r}, "
               "documents=['summary.json'], fields=['warnings'])" if result.get("job_record") else
               "dic_plan with image_dir=<photo_dir>/<event> and the same other params lists that event's errors "
               "in full; the pipeline CLI with --dry-run prints every error")
    elif output:
        how = f"{Path(output) / JOB_STDOUT_NAME}, {Path(output) / JOB_STDERR_NAME} and the job record"
    else:
        how = "the MCP audit log (call_id) and the same call with fewer events"

    def drop_tails(reply: dict[str, Any], fits: Fits) -> dict[str, Any] | None:
        dropped = [key for key in ("stdout_tail", "stderr_tail") if reply.pop(key, None) is not None]
        if not dropped:
            return None
        where = (f"{Path(output) / JOB_STDOUT_NAME}, {Path(output) / JOB_STDERR_NAME}" if output
                 else "not kept: no output directory")
        return {"field": ", ".join(dropped), "where": where}

    def trim_lists(reply: dict[str, Any], fits: Fits) -> dict[str, Any] | None:
        totals: dict[str, int] = {}
        note: dict[str, Any] = {}
        for keep in (50, 20, 10, 5, 2, 1):
            for key in ("errors", "warnings", "error_lines"):
                items = reply.get(key)
                if isinstance(items, list) and len(items) > keep:
                    totals.setdefault(key, len(items))
                    reply[key] = items[:keep]
            if reply.get("code") == "plan_invalid" and isinstance(reply.get("errors"), list):
                reply["error"] = json.dumps(reply["errors"], ensure_ascii=False)
            note = {"field": ", ".join(totals), "kept_each": keep, "total": dict(totals), "how_to_read": how}
            if totals and fits(note):
                break
        return note if totals else None

    def shorten_strings(reply: dict[str, Any], fits: Fits) -> dict[str, Any] | None:
        changed: list[str] = []
        for key in ("errors", "warnings"):
            items = reply.get(key)
            if isinstance(items, list):
                shortened = [{**item, "message": _one_line(item["message"], 300)}
                             if isinstance(item, dict) and isinstance(item.get("message"), str)
                             and len(item["message"]) > 300 else item for item in items]
                if shortened != items:
                    reply[key] = shortened
                    changed.append(f"{key}[].message")
        if reply.get("code") == "plan_invalid" and isinstance(reply.get("errors"), list):
            reply["error"] = json.dumps(reply["errors"], ensure_ascii=False)
        for key, limit in (("error", 2000), ("hint", 1000)):
            if isinstance(reply.get(key), str) and len(reply[key]) > limit and not (
                    key == "error" and reply.get("code") == "plan_invalid"):
                reply[key] = reply[key][:limit] + " ..."
                changed.append(key)
        return {"field": ", ".join(changed), "shortened_to_chars": 300, "how_to_read": how} if changed else None

    def minimal(reply: dict[str, Any], fits: Fits) -> dict[str, Any]:
        kept = {"ok", "code", "error", "call_id", "field", "retryable", "hint", "output_dir", "job_record", "omitted"}
        for key in [key for key in reply if key not in kept]:
            reply.pop(key)
        for key, limit in (("error", 1000), ("hint", 500)):
            if isinstance(reply.get(key), str) and len(reply[key]) > limit:
                reply[key] = reply[key][:limit] + " ..."
        return {"field": "everything but ok, code, error, field, retryable, hint and the output paths",
                "how_to_read": how}

    return _fit_reply(result, [
        ("tails", drop_tails),
        ("lists", trim_lists),
        ("strings", shorten_strings),
        ("minimal", minimal),
    ], holder=result)


def _finalize_run_result(summary: dict[str, Any], output: Path, record_path: Path,
                         record: dict[str, Any], params: dict[str, Any],
                         cod_definition: dict[str, Any] | None = None) -> dict[str, Any]:
    nulled: list[dict[str, Any]] = []
    mcp_warnings: list[dict[str, str]] = list(params.get("_mcp_warnings", []))
    cod_ok = True
    for event in summary.get("events") or []:
        if not isinstance(event, dict):
            continue
        for key in ("cod_csv", "field_dir"):
            value = event.get(key)
            if isinstance(value, str) and value and not Path(value).exists():
                event[key] = None
                reason = "not on disk"
                if key == "field_dir" and not params.get("keep_frames"):
                    reason += ": frame CSVs are deleted after COD aggregation unless keep_frames=true"
                elif key == "cod_csv":
                    reason += ": the event produced no COD CSV"
                nulled.append({"event": event.get("name"), "field": key, "path": value, "reason": reason})
        cod_csv = event.get("cod_csv")
        if event.get("status") == "ok":
            if cod_csv is None:
                cod_ok = False
            else:
                try:
                    header_only = Path(cod_csv).stat().st_size <= len("frame,x_mm,cod_um\n")
                except OSError:
                    header_only = True
                if header_only:
                    cod_ok = False
                    mcp_warnings.append({
                        "code": "cod_csv_empty",
                        "message": f"{event.get('name')}: {cod_csv} has a header but no data rows; the "
                                   "measurement lines produced no valid COD (lines outside the image or margin, "
                                   "gauge too small, or ROI without POI columns)",
                    })
    if record.get("state") == "runner_lost":
        mcp_warnings.append({
            "code": "job_runner_lost",
            "message": "the job runner was killed while the pipeline ran; the server waited for the pipeline, "
                       "which ended by itself, so its exit code is unknown: judge the run by verdict and "
                       "events[].status",
        })
    verdict = summary.get("verdict")
    failed = verdict == "fail" or record.get("exit_code") not in (0, None)
    mcp: dict[str, Any] = {
        "output_dir": str(output),
        "summary_json": str(output / "summary.json"),
        "job_record": str(record_path),
        "job": {key: record.get(key) for key in ("state", "exit_code", "wall_s") if key in record},
        "brief": _event_brief(summary, selection=_uses_event_selection(params)),
    }
    if cod_definition:
        mcp["cod_definition"] = cod_definition
    if nulled:
        mcp["nulled_paths"] = nulled
    if mcp_warnings:
        mcp["warnings"] = mcp_warnings
    if failed:
        stdout_tail = _tail(output / JOB_STDOUT_NAME)
        stderr_tail = _tail(output / JOB_STDERR_NAME)
        mcp["stdout_tail"] = stdout_tail
        mcp["stderr_tail"] = stderr_tail
        mcp["error_lines"] = _error_lines(stdout_tail, stderr_tail)
        mcp["hint"] = ("read events[].warning_details (severity fail first), events[].remediation and "
                       "mcp.error_lines; the binary's complete stderr is the file events[].binary_stderr_log; "
                       "this output_dir cannot be reused")
    summary["ok"] = bool(verdict in {"ok", "warn"} and not failed and cod_ok)
    summary["mcp"] = mcp
    return _shape_run_reply(summary, output, selection=_uses_event_selection(params))


def _progress_plan(params: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any] | None:
    """What the job runner needs to report progress.

    The events in run order with their target frames, and where the pipeline
    leaves each event's frame directory, COD CSV and binary log (relative to
    output_dir, the event name written as PROGRESS_EVENT_TOKEN), taken from the
    pipeline's own path helpers.
    """

    try:
        parser = pipeline.create_parser()
        parser.error = _raise_parse_error  # type: ignore[method-assign]
        namespace = parser.parse_args(_pipeline_argv(params))
        output = Path(params["output_dir"])
        token = PROGRESS_EVENT_TOKEN
        cod_csv = pipeline.event_output_csv(output, token, namespace)
        return {
            "token": token,
            "frames_dir": str(pipeline.event_frame_dir(output, token, namespace).relative_to(output)),
            "cod_csv": str(cod_csv.relative_to(output)) if cod_csv is not None else None,
            "binary_log": str(pipeline.event_binary_log(output, token, namespace).relative_to(output)),
            "events": [[str(event.get("name")), int(event.get("target_frames") or 0)]
                       for event in plan.get("events") or [] if isinstance(event, dict)],
        }
    except Exception:  # progress is an extra: a job never fails to start over it
        return None


def _initial_progress(plan: dict[str, Any], now: float) -> dict[str, Any]:
    """The progress block before the runner has counted anything: the totals of the plan."""

    events = [event for event in plan.get("events") or [] if isinstance(event, dict)]
    return {"updated_at": _utc(now), "elapsed_s": 0.0, "events_total": len(events), "events_done": 0,
            "frames_total": sum(int(event.get("target_frames") or 0) for event in events), "frames_done": 0,
            "current_event": None}


def _poll_interval(suggested_timeout_s: Any) -> int:
    """Seconds between polls worth suggesting: 5 for one event, up to 30 for a long campaign."""

    try:
        seconds = float(suggested_timeout_s) / 30.0
    except (TypeError, ValueError):
        return JOB_POLL_AFTER_SECONDS
    return int(min(JOB_POLL_AFTER_SECONDS_MAX, max(JOB_POLL_AFTER_SECONDS, seconds)))


def _run_pipeline_locked(params: dict[str, Any], output_handle=None,
                         background: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the job and return its result.

    ``background`` (a wait=false call) carries token_sha256 and on_started: the
    callback gets the job's paths as soon as the runner has started, and the
    calling thread goes on waiting for the job exactly as a blocking call does.
    """

    _check_output_state(params)
    plan = _build_plan(params)
    if not plan["valid"]:
        raise InterfaceError(
            "plan_invalid", json.dumps(plan["errors"], ensure_ascii=False),
            errors=plan["errors"], warnings=plan.get("warnings", []),
            hint="dic_plan with the same params shows the full plan; fix every entry in errors",
        )
    if params["mode"] == "campaign" and len(plan["events"]) > MAX_CAMPAIGN_EVENTS:
        raise InterfaceError(
            "campaign_too_large",
            f"campaign has {len(plan['events'])} events; MCP limit is {MAX_CAMPAIGN_EVENTS}",
            field="photo_dir", expected=f"at most {MAX_CAMPAIGN_EVENTS} event subdirectories",
        )

    call_id = _current_call_id()
    tool = "dic_run_event" if params["mode"] == "event" else "dic_run_campaign"
    output = Path(params["output_dir"])
    summary_path = output / "summary.json"
    record_path = output / JOB_RECORD_NAME
    holder = {"tool": tool, "call_id": call_id, "output_dir": str(output)}
    with _gpu_lock(params["gpu_device"], holder) as gpu_handle:
        try:
            output.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise InterfaceError("output_unavailable", f"cannot create output_dir {output}: {exc}",
                                 field="output_dir") from exc
        argv = [sys.executable, str(PIPELINE_PATH), *_pipeline_argv(params), f"--json-summary={summary_path}"]
        now = time.time()
        lock_fds = [gpu_handle.fileno()] + ([output_handle.fileno()] if output_handle is not None else [])
        record = {
            "schema": JOB_RECORD_SCHEMA,
            "state": "starting",
            "call_id": call_id,
            "tool": tool,
            "output_dir": str(output),
            "summary_json": str(summary_path),
            "gpu_device": params["gpu_device"],
            "server_pid": os.getpid(),
            "server_version": SERVER_VERSION,
            "created_at": _utc(now),
            "timeout_s": params["timeout"],
            "deadline_at": _utc(now + params["timeout"]),
            "wait": background is None,
            "progress_interval_s": PROGRESS_INTERVAL_SECONDS,
            # Present from the start (the plan's totals), so a poll never has to ask whether it exists yet.
            "progress": _initial_progress(plan, now),
            "stdout_log": JOB_STDOUT_NAME,
            "stderr_log": JOB_STDERR_NAME,
            "pipeline_argv": argv,
            "states": "starting -> running -> exited | timed_out | cancelled | spawn_failed | runner_lost | "
                      "job_lost. exited means the pipeline process ended (read summary.json and its verdict; "
                      "exit_code 0 = every event ok, 3 = an event failed). cancelled: dic_job_cancel or a "
                      "SIGTERM to the job runner stopped the job (cancel_reason). "
                      "server_detached_at means the MCP server exited first; the job kept its GPU/output locks "
                      "until it ended. runner_lost: the job runner was killed, the server waited for the "
                      "pipeline, which ended by itself (exit code unknown); job_lost: nothing was left running "
                      "and there is no summary. While starting/running the runner rewrites progress every "
                      "progress_interval_s seconds; a record that stays starting/running after deadline_at, or "
                      "whose progress.updated_at stopped advancing, is stale: dic_job_cancel settles it.",
        }
        if background is not None:
            record["cancel_token_sha256"] = background["token_sha256"]
        try:
            _write_json_atomic(record_path, record)
            stderr_handle = open(output / JOB_STDERR_NAME, "ab", buffering=0)
            open(output / JOB_STDOUT_NAME, "ab").close()
        except OSError as exc:
            raise InterfaceError("output_unavailable", f"cannot write the job record in {output}: {exc}",
                                 field="output_dir") from exc
        spec = {
            "record": str(record_path),
            "output_dir": str(output),
            "argv": argv,
            "timeout_s": params["timeout"],
            "stdout_log": str(output / JOB_STDOUT_NAME),
            "stderr_log": str(output / JOB_STDERR_NAME),
            "summary": str(summary_path),
            "lock_fds": lock_fds,
            "progress_interval_s": PROGRESS_INTERVAL_SECONDS,
            "progress": _progress_plan(params, plan),
        }
        try:
            runner = subprocess.Popen(
                [sys.executable, "-c", _JOB_RUNNER_SOURCE, json.dumps(spec)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr_handle,
                start_new_session=True, pass_fds=lock_fds,
            )
        except OSError as exc:
            stderr_handle.close()
            _merge_job_record(record_path, state="spawn_failed", error=str(exc), ended_at=_utc())
            raise InterfaceError("pipeline_failed", f"cannot start the job runner: {exc}") from exc
        finally:
            if not stderr_handle.closed:
                stderr_handle.close()
        _write_lock_holder(gpu_handle, {
            **holder, "gpu_device": params["gpu_device"], "server_pid": os.getpid(),
            "runner_pid": runner.pid, "acquired_at": _utc(now), "acquired_epoch": now,
        })
        if output_handle is not None:
            _write_lock_holder(output_handle, {
                **holder, "server_pid": os.getpid(), "runner_pid": runner.pid,
                "acquired_at": _utc(now), "acquired_epoch": now,
            })
        # The runner records its pid and start time too; dic_job_cancel must find it while the job is
        # still starting (the child is not reaped yet, so the pid is the runner's for certain).
        with contextlib.suppress(Exception):
            _merge_job_record(record_path, runner_pid=runner.pid, runner_start_ticks=_proc_start_ticks(runner.pid))
        with _REGISTRY_GUARD:
            _ACTIVE_JOBS[call_id] = {
                "runner": runner, "record": record_path, "output_dir": str(output), "tool": tool,
                "params": (_INFLIGHT.get(call_id) or {}).get("params", {}), "background": background is not None,
            }

        def respawn(job_pgid: int, remaining: float) -> subprocess.Popen | None:
            """A second runner for a job whose first runner was lost: it watches the job's process group."""
            adopt_spec = {**spec, "adopt_pgid": job_pgid, "timeout_s": remaining,
                          "adopt_start_ticks": (_read_small_json(record_path, 1 << 20) or {}).get("job_start_ticks"),
                          "elapsed_s": max(0.0, time.time() - now)}
            with open(output / JOB_STDERR_NAME, "ab", buffering=0) as log:
                second = subprocess.Popen(
                    [sys.executable, "-c", _JOB_RUNNER_SOURCE, json.dumps(adopt_spec)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log,
                    start_new_session=True, pass_fds=lock_fds,
                )
            with _REGISTRY_GUARD:
                if call_id in _ACTIVE_JOBS:
                    _ACTIVE_JOBS[call_id]["runner"] = second
            return second

        if background is not None:
            background["on_started"]({
                "output_dir": str(output), "record": record_path, "summary": summary_path,
                "cod_definition": (plan.get("mcp") or {}).get("cod_definition"),
                "suggested_timeout_s": (plan.get("mcp") or {}).get("suggested_timeout_s"),
            })
        try:
            try:
                runner.wait(timeout=params["timeout"] + RUNNER_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                _stop_runner(runner, record_path)
                _merge_job_record(record_path, state="timed_out", ended_at=_utc(),
                                  note="the server stopped an unresponsive job runner")
            else:
                # Still inside the lock contexts: an adopted job keeps both locks.
                _adopt_job_of_lost_runner(runner, record_path, summary_path,
                                          deadline=time.monotonic() - (time.time() - now) + params["timeout"],
                                          respawn=respawn)
        finally:
            with _REGISTRY_GUARD:
                _ACTIVE_JOBS.pop(call_id, None)

    record = _read_small_json(record_path, 1 << 20) or {}
    state = record.get("state")
    stdout_tail = _tail(output / JOB_STDOUT_NAME)
    stderr_tail = _tail(output / JOB_STDERR_NAME)
    if state == "job_lost":
        raise InterfaceError(
            "job_lost",
            f"the job runner ended ({record.get('runner_exit')}) without recording an outcome and the pipeline "
            "is gone without a summary",
            retryable=True, output_dir=str(output), job_record=str(record_path),
            stdout_tail=stdout_tail, stderr_tail=stderr_tail, error_lines=_error_lines(stdout_tail, stderr_tail),
            hint="something killed the job (for example the OOM killer); this output_dir cannot be reused: rerun "
                 "with a NEW output_dir",
        )
    if state in {"timed_out", "cancelled"}:
        frames = sum(1 for _ in output.glob("_frames_*/frame_*"))
        code = "timeout" if state == "timed_out" else "job_cancelled"
        message = (
            f"pipeline exceeded timeout={params['timeout']}s and its process group was terminated"
            if state == "timed_out" else f"the job was cancelled ({record.get('cancel_reason')})"
        )
        raise InterfaceError(
            code, message, field="timeout" if state == "timed_out" else None,
            got=params["timeout"] if state == "timed_out" else None,
            retryable=True, output_dir=str(output), job_record=str(record_path),
            partial={"frame_files": frames, "summary_json": summary_path.is_file()},
            hint="this output_dir now holds partial output and cannot be reused: rerun with a NEW output_dir"
                 + (" and a larger timeout (dic_plan reports mcp.suggested_timeout_s)"
                    if state == "timed_out" else ""),
        )

    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InterfaceError("invalid_summary", f"cannot read pipeline summary: {exc}") from exc
        if not isinstance(summary, dict):
            raise InterfaceError("invalid_summary", "pipeline summary is not a JSON object")
        if record.get("exit_code") == PIPELINE_EXIT_STRICT_REFUSED:
            # --strict re-ran the dry-run checks in the pipeline process and they
            # failed: nothing was computed. Same shape as the plan check above.
            errors = _strict_refusal_errors(summary)
            raise InterfaceError(
                "plan_invalid", json.dumps(errors, ensure_ascii=False),
                errors=errors, warnings=[],
                hint="the pipeline's own --strict check refused the run before computing anything (the "
                     "inputs, the binary or output_dir changed after the plan check); call dic_plan with the "
                     "same params, fix every entry in errors and use a NEW output_dir",
                output_dir=str(output), job_record=str(record_path),
            )
        # Exit 3 (--strict: an event failed) and the historical codes return the
        # summary: verdict fail, ok false and the failure details in the mcp block.
        return _finalize_run_result(summary, output, record_path, record, params,
                                    (plan.get("mcp") or {}).get("cod_definition"))
    raise InterfaceError(
        "pipeline_failed",
        f"pipeline exited {record.get('exit_code')} without summary; stdout={chr(10).join(stdout_tail)[-2000:]!r}; "
        f"stderr={chr(10).join(stderr_tail)[-2000:]!r}",
        output_dir=str(output), job_record=str(record_path), job_state=state,
        stdout_tail=stdout_tail, stderr_tail=stderr_tail, error_lines=_error_lines(stdout_tail, stderr_tail),
        hint="read error_lines/stderr_tail; this output_dir cannot be reused",
    )


def _run_pipeline(params: dict[str, Any]) -> dict[str, Any]:
    if not params.get("wait", True):
        return _start_background_job(params)
    output = Path(params["output_dir"])
    holder = {"call_id": _current_call_id(),
              "tool": "dic_run_event" if params["mode"] == "event" else "dic_run_campaign"}
    with _output_lock(output, holder) as output_handle:
        # Re-check only after serializing all MCP runs for this canonical output.
        return _run_pipeline_locked(params, output_handle)


def _start_background_job(params: dict[str, Any]) -> dict[str, Any]:
    """wait=false: start the job, return its handle, keep waiting for it in a thread.

    The thread runs the same code as a blocking call (output lock, state check,
    plan check, GPU lock, runner, adoption of a lost runner), so every refusal
    that happens before the runner starts reaches the caller unchanged. Once the
    runner has started the caller gets job_id, output_dir and the job record's
    path; the thread stays in the lock contexts until the job ends and then
    closes the audit trail with a job_finished entry. If the server exits first
    the job goes on detached, as a blocking call's job does.
    """

    call_id = _current_call_id()
    tool = "dic_run_event" if params["mode"] == "event" else "dic_run_campaign"
    output = Path(params["output_dir"])
    with _REGISTRY_GUARD:
        audit_params = (_INFLIGHT.get(call_id) or {}).get("params", {})
    # Returned once, kept only as a hash: whoever holds it may cancel the job from another session.
    token = secrets.token_hex(24)
    started = threading.Event()
    box: dict[str, Any] = {}

    def on_started(info: dict[str, Any]) -> None:
        box["info"] = info
        started.set()

    def worker() -> None:
        _CALL_CONTEXT.call_id = call_id
        outcome = "internal_error"
        try:
            with _output_lock(output, {"call_id": call_id, "tool": tool}) as output_handle:
                result = _run_pipeline_locked(params, output_handle, {
                    "on_started": on_started,
                    "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                })
            outcome = str(result.get("verdict") or ("ok" if result.get("ok") else "error"))
        except InterfaceError as exc:
            outcome = exc.code
            if "info" not in box:
                box["error"] = exc
        except BaseException as exc:  # reported to the caller, or audited: a thread has nowhere to raise
            outcome = f"internal_error:{type(exc).__name__}"
            if "info" not in box:
                box["error"] = exc
        finally:
            job_started = "info" in box
            started.set()
            if job_started:
                with contextlib.suppress(Exception):
                    final = _read_small_json(box["info"]["record"], 1 << 20) or {}
                    _audit_record(tool, call_id, audit_params, f"job_finished:{outcome}", extra={
                        "job_state": final.get("state"), "exit_code": final.get("exit_code"),
                        "output_dir": str(output),
                    })

    threading.Thread(target=worker, name=f"fasticgn-job-{call_id[:8]}", daemon=True).start()
    started.wait()
    if "info" not in box:
        error = box.get("error")
        if isinstance(error, BaseException):
            raise error  # the same refusal a blocking call gets (_dispatch shapes and audits it)
        raise InterfaceError("pipeline_failed", "the job did not start")

    info = box["info"]
    record = _read_small_json(info["record"], 1 << 20) or {}
    state = record.get("state", "starting")
    mcp_block: dict[str, Any] = {
        "job": {"state": state, "wait": False, "tool": tool, "gpu_device": params["gpu_device"],
                "timeout_s": params["timeout"], "deadline_at": record.get("deadline_at")},
        "poll": {
            "tool": "dic_query_manifest",
            "arguments": {"run_dir": info["output_dir"], "documents": [JOB_RECORD_NAME],
                          "fields": list(JOB_POLL_FIELDS)},
            "poll_after_s": _poll_interval(info.get("suggested_timeout_s")),
            "read": f"documents['{JOB_RECORD_NAME}']: state starting or running = still going, with progress "
                    "(events_done/events_total, frames_done/frames_total, current_event, updated_at). "
                    "exited = the pipeline ended: read the summary (mcp.result). timed_out, cancelled, "
                    "spawn_failed, runner_lost, job_lost: no usable result. Leave fields out to read the whole "
                    "record (exit_code 0 = every event ok, 3 = an event failed; verdict, events_ok, "
                    "events_failed; cancel_reason). A record still starting or running after deadline_at, or "
                    "whose progress.updated_at stopped advancing, is stale: dic_job_cancel settles it",
        },
        "result": {
            "tool": "dic_query_manifest",
            "arguments": {"run_dir": info["output_dir"], "documents": ["summary.json"]},
            "when": "state is exited",
            "read": "verdict (ok or warn = every event ran; fail = an event failed), events[].status and "
                    "events[].warning_details as after a blocking run; the COD CSVs are events[].cod_csv",
        },
        "cancel": {
            "tool": "dic_job_cancel",
            "arguments": {"output_dir": info["output_dir"], "job_id": call_id, "cancel_token": token},
            "note": "the session that started the job may leave cancel_token out; another session needs it "
                    "(it is returned only here and stored only as a hash)",
        },
    }
    if info.get("cod_definition"):
        mcp_block["cod_definition"] = info["cod_definition"]
    if params.get("_mcp_warnings"):
        mcp_block["warnings"] = list(params["_mcp_warnings"])
    return {
        "ok": True,
        "job_id": call_id,
        "state": state,
        "output_dir": info["output_dir"],
        "job_record": str(info["record"]),
        "summary_json": str(info["summary"]),
        "cancel_token": token,
        "mcp": mcp_block,
        "_audit_code": "job_started",
    }


_DETACH_MARKED: set[str] = set()


def _mark_jobs_detached(cause: str, *, only_background: bool = False,
                        audit_extra: dict[str, Any] | None = None) -> set[str]:
    """The server is leaving: say so in the record of every job it still watches, and in the audit log.

    Also for a job whose runner is gone and which this server had adopted: until
    2026-10-05 such a record got no mark and stayed at running with nobody left
    to update it (review 2026-10-04 recheck R4). Returns the call ids handled.
    """

    acquired = _REGISTRY_GUARD.acquire(timeout=1.0)
    try:
        jobs = dict(_ACTIVE_JOBS)
        inflight = set(_INFLIGHT)
    finally:
        if acquired:
            _REGISTRY_GUARD.release()
    handled: set[str] = set()
    for call_id, job in jobs.items():
        if call_id in _DETACH_MARKED or (only_background and (not job.get("background") or call_id in inflight)):
            continue
        record = _read_small_json(job["record"], 1 << 20) or {}
        if record.get("state") not in JOB_ACTIVE_STATES:
            continue
        if job["runner"].returncode is None:
            reason = f"{cause}; the job keeps running and holds its GPU/output locks until it exits"
        else:
            reason = (f"{cause} while the server was waiting for this job after its runner was lost: the "
                      "pipeline keeps its locks until it ends, but nobody updates this record any more; "
                      "read summary.json when it appears, and dic_job_cancel settles the record")
        _DETACH_MARKED.add(call_id)
        handled.add(call_id)
        with contextlib.suppress(Exception):
            _merge_job_record(job["record"], server_detached_at=_utc(), server_detach_reason=reason)
        with contextlib.suppress(Exception):
            _audit_record(job.get("tool") or "dic_run", call_id, job.get("params") or {}, "detached", extra={
                **(audit_extra or {}), "output_dir": job["output_dir"], "job_record": str(job["record"]),
            })
    return handled


def _handle_server_termination(signum: int, frame: Any) -> None:
    """SIGTERM/SIGHUP: close the audit trail, mark detached jobs, then exit.

    MCP clients end a stdio session by closing stdin, waiting ~2 s and sending
    SIGTERM to the server's process group.  Running jobs live in their own
    session and hold their locks, so they finish on their own; this handler
    only records that the server left (review 2026-10-03 F2). That covers the
    jobs of blocking calls and the wait=false jobs whose call has returned.
    """

    with contextlib.suppress(Exception):
        name = signal.Signals(signum).name
        acquired = _REGISTRY_GUARD.acquire(timeout=1.0)
        try:
            inflight = list(_INFLIGHT.items())
        finally:
            if acquired:
                _REGISTRY_GUARD.release()
        detached = _mark_jobs_detached(f"MCP server received {name}", audit_extra={"signal": name}) | _DETACH_MARKED
        for call_id, entry in inflight:
            if call_id in detached:
                continue
            with contextlib.suppress(Exception):
                _audit_record(entry["tool"], call_id, entry["params"], "server_terminated", extra={"signal": name})
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _install_signal_handlers() -> None:
    for signum in (signal.SIGTERM, signal.SIGHUP):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(signum, _handle_server_termination)


# ---------------------------------------------------------------------------
# Published parameter schemas.  Validation stays in the server code above so
# that every rejection is a structured {ok:false, code, ...} result; the
# schemas document types, units, enums, bounds and defaults for the agent.
# ---------------------------------------------------------------------------

_PAIR_SCHEMA = {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 2, "maxItems": 2}
_GPU_SCHEMA: dict[str, Any] = {"type": "integer", "minimum": 0}
if GPU_DEVICE_COUNT:
    _GPU_SCHEMA["maximum"] = GPU_DEVICE_COUNT - 1

PARAM_SPECS: dict[str, dict[str, Any]] = {
    "image_dir": {
        "schema": {"type": "string"},
        "description": "Absolute path of ONE event directory; frames are named ...NNNNNN.jpg/jpeg/tif/tiff/png/bmp "
                       "(6-digit frame number) and the lowest number is the reference image.",
    },
    "photo_dir": {
        "schema": {"type": "string"},
        "description": "Absolute path of a campaign directory; each subdirectory with frames is one event "
                       f"(at most {MAX_CAMPAIGN_EVENTS}).",
    },
    "output_dir": {
        "schema": {"type": "string"},
        "description": "Absolute path of a NEW directory under the job root (dic_plan reports mcp.job_root); a "
                       "used, failed or timed-out directory is never reused or resumed.",
    },
    "cal": {
        "schema": {"type": "number", "exclusiveMinimum": 0, "maximum": 10},
        "description": "Calibration in mm/px (millimetres per pixel; 46.2 um/px -> 0.0462); with 1.0 on "
                       "uncalibrated images cod_um/1000 is a pixel value.",
    },
    "interface_y": {
        "schema": {"type": "integer", "minimum": 0},
        "description": "Interface row in px (y down); lines at interface_y - gauge//2 (L0, upper) and "
                       "interface_y + gauge//2 (L1). Give interface_y or line_y.",
    },
    "line_y": {
        "schema": _PAIR_SCHEMA,
        "description": "[L0, L1] measurement rows in px, L0 = upper line by convention; COD = u(L0) - u(L1), so "
                       "swapping them flips the sign. Overrides interface_y and gauge.",
    },
    "events": {
        "schema": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1,
                   "maxItems": MAX_CAMPAIGN_EVENTS},
        "description": "Campaign only (photo_dir): run just these events. One event name or glob per item "
                       "(fnmatch, case-sensitive), e.g. [\"event10\", \"event22\"] or [\"event0*\", \"event20\"]; no "
                       "commas inside an item. Every item must match an event, else the plan is invalid "
                       "(event_not_found, event_pattern_no_match). Pipeline --events; omit it to run every event.",
    },
    "event_interface_y": {
        "schema": {
            "type": "array", "minItems": 1, "maxItems": MAX_CAMPAIGN_EVENTS,
            "items": {
                "type": "object", "additionalProperties": False, "required": ["pattern", "interface_y"],
                "properties": {
                    "pattern": {"type": "string", "minLength": 1,
                                "description": "event name or glob (fnmatch, case-sensitive), no commas"},
                    "interface_y": {"type": "integer", "minimum": 0,
                                    "description": "interface row in px used for the matched events"},
                },
            },
        },
        "description": "Campaign only (photo_dir), needs interface_y, not with line_y: per-event interface rows, "
                       "e.g. [{\"pattern\": \"event2[1-5]\", \"interface_y\": 133}]. Matched events use that row "
                       "(lines row -/+ gauge//2), all others interface_y. An entry that matches no selected event "
                       "or an event matched by two different rows makes the plan invalid. Pipeline "
                       "--event-interface-y PATTERN=Y (repeatable).",
    },
    "roi_cols": {
        "schema": _PAIR_SCHEMA,
        "description": "[x_min, x_max] inclusive pixel columns kept in the COD CSV (aggregation only); omit for "
                       "the full width, there is no -1 shorthand.",
    },
    "gauge": {
        "schema": {"type": "integer", "minimum": 4, "maximum": 200},
        "default": 35,
        "description": "Requested line spacing in px used with interface_y: the lines sit at interface_y -/+ "
                       "gauge//2, so the actual spacing is 2*(gauge//2) (34 for the default 35); ignored with "
                       "line_y.",
    },
    "subset": {
        "schema": {"type": "integer", "minimum": 5, "maximum": 51},
        "default": 15,
        "description": "Subset SIZE in px, odd (15 = radius 7); it is not the radius.",
    },
    "step": {
        "schema": {"type": "integer", "minimum": 1, "maximum": 20},
        "default": 1,
        "description": "POI grid step in px.",
    },
    "cod_method": {
        "schema": {"type": "string", "enum": ["fillboundary", "naive"]},
        "default": "naive",
        "description": "naive (default since 2026-10-05) is u(L0) - u(L1); fillboundary (optional, the default "
                       "from 2026-07-09 to 2026-10-05) extrapolates each line's u along du/dy to the midpoint row.",
    },
    "conv_threshold": {
        "schema": {"type": "number", "exclusiveMinimum": 0},
        "default": 0.001,
        "description": "IC-GN convergence threshold; keep 0.001 unless reproducing old runs.",
    },
    "input_depth": {
        "schema": {"type": "string", "enum": ["8", "12", "16", "auto"]},
        "description": "Bit depth as a STRING, omit for 8-bit images; 16-bit TIFFs whose low 4 bits are always 0 "
                       "(12-bit left-shifted) need '12' with input_align='left', never 'auto'.",
    },
    "input_align": {
        "schema": {"type": "string", "enum": ["right", "left", "auto"]},
        "description": "Only with input_depth='12': 'left' when the 12 bits are the high bits (low 4 bits always "
                       "0), 'right' when values are <= 4095.",
    },
    "input_normalize": {
        "schema": {"type": "string", "enum": ["none", "to-8", "to-16", "unit"]},
        "description": "Grey-level normalisation passed to the binary (ZNCC is insensitive to it); usually omit.",
    },
    "gpu_device": {
        "schema": _GPU_SCHEMA,
        "default": 0,
        "description": "CUDA device index"
                       + (f" (this host: 0..{GPU_DEVICE_COUNT - 1})" if GPU_DEVICE_COUNT else "")
                       + "; one MCP job per device, a busy device returns gpu_busy.",
    },
    "keep_frames": {
        "schema": {"type": "boolean"},
        "default": False,
        "description": "Keep the per-frame CSVs in <output_dir>/_frames_<event>/ (deleted otherwise); needed "
                       "for dic_aggregate_cod or displacement statistics.",
    },
    "max_frames": {
        "schema": {"type": "integer", "minimum": 1},
        "description": "Run only the first N target frames after the reference of each event (a quick trial, "
                       "e.g. 20); omit it for every frame. dic_plan's estimates and suggested timeout use the same N.",
    },
    "timeout": {
        "schema": {"type": "number", "exclusiveMinimum": 0, "maximum": MAX_TIMEOUT_SECONDS},
        "default": DEFAULT_TIMEOUT_SECONDS,
        "description": "Seconds before the job is killed; use dic_plan's mcp.suggested_timeout_s (runs take "
                       "about 5-30 s per 800-frame event here).",
    },
    "overwrite": {
        "schema": {"type": "boolean"},
        "default": False,
        "description": "true only permits an existing EMPTY output_dir; a non-empty directory is never reused.",
    },
    "wait": {
        "schema": {"type": "boolean"},
        "default": True,
        "description": "true (default) blocks until the job ends and returns the summary. false returns as soon "
                       "as the job has started, with job_id, output_dir, job_record and cancel_token: poll "
                       "dic_query_manifest(run_dir=output_dir, documents=['.mcp-job.json']) for state and "
                       "progress, stop the job with dic_job_cancel. Ignored by dic_plan.",
    },
}

_COMMON_KEYS = (
    "output_dir", "cal", "interface_y", "line_y", "roi_cols", "gauge", "subset", "step", "cod_method",
    "conv_threshold", "input_depth", "input_align", "input_normalize", "gpu_device", "keep_frames",
    "max_frames", "timeout", "overwrite", "wait",
)
_PARAMS_SCHEMA_DESCRIPTION = (
    "All parameters in ONE object: {\"params\": {...}}. Required: output_dir, cal, interface_y or line_y, and "
    "{source}. COD mode only; unknown keys are rejected with the list of allowed keys."
)


def _params_schema_finalizer(required: list[str], description: str) -> Callable[[dict[str, Any]], None]:
    def finalize(schema: dict[str, Any]) -> None:
        for name, prop in schema.get("properties", {}).items():
            if prop.get("default", "<absent>") is None:
                del prop["default"]
            spec = PARAM_SPECS.get(name)
            if spec and "default" in spec:
                prop["default"] = spec["default"]
        schema["additionalProperties"] = False
        schema["required"] = list(required)
        schema["description"] = description
        schema.pop("title", None)

    return finalize


def _params_model(name: str, keys: tuple[str, ...], required: list[str], source: str):
    if create_model is None:
        return dict
    fields = {
        key: (
            Annotated[Any, WithJsonSchema(dict(PARAM_SPECS[key]["schema"])),
                      Field(description=PARAM_SPECS[key]["description"])],
            None,
        )
        for key in keys
    }
    config = ConfigDict(
        extra="allow",
        json_schema_extra=_params_schema_finalizer(
            required, _PARAMS_SCHEMA_DESCRIPTION.replace("{source}", source)),
    )
    return create_model(name, __config__=config, **fields)


_CAMPAIGN_KEYS = ("events", "event_interface_y")  # CAMPAIGN_ONLY_KEYS in schema order
PlanParams = _params_model(
    "PlanParams", ("image_dir", "photo_dir") + _COMMON_KEYS + _CAMPAIGN_KEYS, ["output_dir", "cal"],
    "exactly one of image_dir (one event) or photo_dir (campaign)",
)
EventRunParams = _params_model(
    "EventRunParams", ("image_dir",) + _COMMON_KEYS, ["image_dir", "output_dir", "cal"], "image_dir",
)
CampaignRunParams = _params_model(
    "CampaignRunParams", ("photo_dir",) + _COMMON_KEYS + _CAMPAIGN_KEYS, ["photo_dir", "output_dir", "cal"],
    "photo_dir",
)


def _arg(schema: dict[str, Any], description: str, optional: bool = False):
    """Annotated type for a plain tool argument: documented, validated by us.

    ``optional`` arguments default to None, so their published type also
    admits null; otherwise the schema's "default": null contradicted its type
    (review 2026-10-04 F11).
    """

    if WithJsonSchema is None:
        return Any
    # The description goes inside the schema: for plain function arguments
    # WithJsonSchema replaces the whole property, Field(description=...) included.
    if optional:
        return Annotated[Any, WithJsonSchema({"anyOf": [schema, {"type": "null"}], "description": description})]
    return Annotated[Any, WithJsonSchema({**schema, "description": description})]


_CONTRACT_AT_IMPORT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8")) if CONTRACT_PATH.is_file() else {}
_CONTRACT_SECTIONS = tuple(_CONTRACT_AT_IMPORT.keys())
# dic_describe without a section returns an index instead of the whole contract:
# the full contract is over 120 KB of JSON (roughly 30-35k tokens), more than
# MCP clients such as Claude Code accept in one tool result by default.
DESCRIBE_INDEX = "index"
DESCRIBE_ALL = "all"
# "Give less by default" (2026-10-05). Until then the default reply carried the
# whole mcp section and exit_codes: 32 759 B against a 32 768 B test limit, so
# nothing could be added to the contract's mcp section any more. The default
# reply now holds only what a client needs BEFORE a run: the keys of the mcp
# section listed here. Every other key is fetched by name
# (dic_describe(section='mcp.<key>'), summarised in MCP_ON_DEMAND_NOTES);
# section='mcp' still returns the whole section and exit_codes is a section of
# its own. A unit test requires every key of the contract's mcp section to be in
# exactly one of the two, so a new key forces the decision, and requires the
# default reply plus the on-demand parts to add up to the whole section.
DESCRIBE_BEFORE_RUN_KEYS = (
    "implementation", "dependency", "dependency_file", "transport", "server_name", "server_version",
    "interface_revision", "supported_modes", "mode_limitation", "tool_count", "tool_names",
    "run_params", "guardrails",
)
MCP_ON_DEMAND_NOTES = {
    "interface_revision_note": "what each interface revision changed",
    "instructions": "what the server instructions sent at initialize cover",
    "numeric_core_modified": "false: the server holds no DIC numerics",
    "numeric_core_modified_semantics": "what numeric_core_modified means",
    "tools": "per tool: parameters, read_only and what it returns",
    "results": "reply bodies: envelope, ok, the run mcp block, cod_definition, compact events, size cap",
    "errors": "error shape, optional fields, every code, retryable",
    "jobs": "wait=false jobs: the handle, polling, progress, dic_job_cancel, who may cancel, stale records",
    "job_record": ".mcp-job.json: path, logs, states, fields, progress, detached jobs",
    "manifest_query": "dic_query_manifest: candidates, max_bytes, detail, condensed view, fields",
    "aggregation": "dic_aggregate_cod rules: line_y, output_path, locks, frame names",
    "known_issues": "known issues of this server",
    "error_semantics": "schema rejection (isError) versus structured ok=false results",
    "dry_run_side_effect_note": "what dic_plan writes (only the audit log)",
    "client_config": "how a client starts the server (local and over ssh)",
}
DESCRIBE_DEFAULT_MAX_BYTES = 20 * 1024
_MCP_SECTION_KEYS = tuple((_CONTRACT_AT_IMPORT.get("mcp") or {}).keys())
DESCRIBE_SUBSECTIONS = tuple(f"mcp.{key}" for key in _MCP_SECTION_KEYS)
CONTRACT_SECTION_NOTES = {
    "contract": "name of this document",
    "schema_version": "contract schema version",
    "implementation": "which programs implement the interface and how they are versioned",
    "pipeline": "pipeline CLI: input and field modes, validation scope, every parameter with type, default and meaning",
    "binary": "C++ CLI options and behaviour the pipeline relies on",
    "outputs": "every output: frame and COD CSV, manifests, run state, the --json-summary fields, warning vocabulary "
               "with severities and verdict rule, and the dry-run plan",
    "mcp": "this MCP server; the default reply carries its before-run part (run params, guardrails), the rest "
           "is listed in mcp_view.on_demand and fetched with section='mcp.<key>'",
    "exit_codes": "pipeline and binary exit codes (--strict: 2 refused before running, 3 an event failed)",
    "pitfalls": "historical traps; dic_get_pitfalls returns the same list",
    "lab_notes": "facts that hold only for the author's laboratory (its datasets, line conventions, old "
                 "binaries); not part of dic_get_pitfalls",
    "known_limitations": "known limitations of the pipeline, the binary and their outputs",
}
SectionArg = _arg(
    {"type": "string", "enum": [DESCRIBE_INDEX, DESCRIBE_ALL, *_CONTRACT_SECTIONS, *DESCRIBE_SUBSECTIONS]},
    "Omit it (or 'index') for the default reply: an index of the contract's sections (name, bytes, one-line "
    "summary) plus what a client needs before a run (mcp.run_params, mcp.guardrails, the tool names) and the "
    "list of the parts left out. 'mcp.<key>' returns one of those parts, e.g. 'mcp.results' (reply bodies), "
    "'mcp.errors', 'mcp.jobs' (wait=false, polling, cancelling), 'mcp.job_record', 'mcp.manifest_query' or "
    "'mcp.known_issues'; a section name such as 'mcp', 'outputs', 'exit_codes' or 'pitfalls' returns that whole section; 'all' returns the whole "
    "contract (over 120 KB, more than many MCP clients accept in one reply).",
    optional=True,
)
ProbeDirArg = _arg({"type": "string"}, "Absolute path of one image directory inside the read allowlist.")
FramesDirArg = _arg(
    {"type": "string"},
    "Absolute path of a retained frame directory, normally <output_dir>/_frames_<event> from a run with "
    "keep_frames=true. Files must be frame_<number>.csv with the 9 columns x,y,u,v,zncc,uy,vy,ux,vx "
    "(5-column CSVs work only with method='naive', the default).",
)
OutputPathArg = _arg(
    {"type": "string", "pattern": r"(?i)\.csv$"},
    "Absolute path of the COD CSV to write: under the job root, ending in .csv, not inside a frames "
    "directory, and never an existing file inside a run directory (pipeline products such as cod_result.csv "
    "or summary.json are never replaced).",
)
AggregateLineArg = _arg(
    _PAIR_SCHEMA,
    "[L0, L1] rows that exist in the frame CSVs (the run's resolved_params.line_ys; with interface_y they are "
    "interface_y -/+ gauge//2). L0 is the upper line by convention; swapping the values flips the COD sign.",
)
AggregateCalArg = _arg(
    {"type": "number", "exclusiveMinimum": 0, "maximum": 10},
    "Calibration in millimetres per pixel (same meaning as the run parameter cal), 0 < cal <= 10.",
)
MethodArg = _arg(
    {"type": "string", "enum": ["fillboundary", "naive"]},
    "'naive' (default since 2026-10-05) is u(L0) - u(L1); 'fillboundary' extrapolates each line's u along "
    "du/dy to the midpoint row.",
)
AggregateRoiArg = _arg(
    _PAIR_SCHEMA,
    "[x_min, x_max] inclusive pixel columns to keep, 0 <= x_min <= x_max; omit for every column present.",
    optional=True,
)
AggregateOverwriteArg = _arg(
    {"type": "boolean"},
    "Replace an existing output_path. Never allowed for files inside a run directory.",
)
RunDirArg = _arg(
    {"type": "string"},
    "Absolute path of a run output directory (or any allowlisted directory holding JSON summaries/manifests).",
)
DetailArg = _arg(
    {"type": "string", "enum": list(QUERY_DETAILS)},
    "'auto' (default): documents in full when everything fits in max_bytes, otherwise condensed; "
    "'compact': always condensed (long arrays become counts and statistics); 'full': never condensed, "
    "documents that do not fit are omitted and listed.",
)
FieldsArg = _arg(
    {"type": "array", "items": {"type": "string"}, "maxItems": 32},
    "Optional paths to extract from each document: dot-separated keys with optional [index] or [start:stop] "
    "on arrays, e.g. ['verdict', 'events[0].warnings', 'frames[0:20]']; at most 256 characters each.",
    optional=True,
)
DocumentsArg = _arg(
    {"type": "array", "items": {"type": "string"}, "maxItems": 32},
    "Optional document names to include, as listed in available_documents (e.g. ['summary.json']).",
    optional=True,
)
JobDirArg = _arg(
    {"type": "string"},
    "Absolute path of the job's output_dir (the directory under the job root a run tool was given; it holds "
    ".mcp-job.json). The output_dir is the handle of a job.",
)
JobIdArg = _arg(
    {"type": "string", "pattern": "^[0-9a-f]{32}$"},
    "Optional check: the job_id a wait=false run returned (the call_id of the run). When given it must be the "
    "id of the job in output_dir, else the call is refused with job_id_mismatch and nothing is stopped.",
    optional=True,
)
CancelTokenArg = _arg(
    {"type": "string", "minLength": 1, "maxLength": 128},
    "The cancel_token a wait=false run returned. The session that started a job cancels it without the token; "
    "any other session needs it (a job id is not enough: it is readable in the job record).",
    optional=True,
)
MaxBytesArg = _arg(
    {"type": "integer", "minimum": MIN_QUERY_MAX_BYTES, "maximum": MAX_QUERY_MAX_BYTES},
    f"Hard limit for the reply, in bytes of compact JSON (the structured content; the wire carries the reply "
    f"twice, as text and as structured content), {MIN_QUERY_MAX_BYTES}..{MAX_QUERY_MAX_BYTES}.",
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _contract_index(contract: dict[str, Any]) -> dict[str, Any]:
    """The default dic_describe reply: the section index plus the before-run part of the mcp section."""

    full_bytes = _json_size(contract)
    mcp_section = contract.get("mcp") if isinstance(contract.get("mcp"), dict) else {}
    included = [key for key in DESCRIBE_BEFORE_RUN_KEYS if key in mcp_section]
    return {
        "schema_version": contract.get("schema_version"),
        "contract": contract.get("contract"),
        "view": DESCRIBE_INDEX,
        "full_contract_bytes": full_bytes,
        "sections": [
            {"section": name, "bytes": _json_size(value),
             "summary": CONTRACT_SECTION_NOTES.get(name, "see dic_describe(section=...)")}
            for name, value in contract.items()
        ],
        # No section is included whole any more (until 2026-10-05: mcp and exit_codes).
        "included_sections": [],
        "mcp": {key: mcp_section[key] for key in included},
        "mcp_view": {
            "view": "before_run",
            "included_keys": included,
            "on_demand": [
                {"section": f"mcp.{key}", "bytes": _json_size(value),
                 "summary": MCP_ON_DEMAND_NOTES.get(key, "see dic_describe(section='mcp')")}
                for key, value in mcp_section.items() if key not in included
            ],
            "how": (f"dic_describe(section='mcp.results') returns one part; section='mcp' the whole mcp section "
                    f"({_json_size(mcp_section)} bytes)"),
        },
        "more": (f"dic_describe(section='<name>') returns one section, section='mcp.<key>' one part of the mcp "
                 f"section; section='all' returns the whole contract ({full_bytes} bytes of JSON, more than many "
                 "MCP clients accept in one reply)"),
    }


def dic_describe(section: SectionArg = None) -> dict[str, Any]:
    """Return the before-run part of the FastICGN interface contract, one section or part, or all of it."""

    params = {} if section is None else {"section": section}

    def operation() -> dict[str, Any]:
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        if section is None or section == DESCRIBE_INDEX:
            return _contract_index(contract)
        if section == DESCRIBE_ALL:
            return contract
        if isinstance(section, str) and section in contract:
            return {"schema_version": contract.get("schema_version"), "section": section,
                    section: contract[section]}
        parent, dot, key = section.partition(".") if isinstance(section, str) else ("", "", "")
        mcp_section = contract.get("mcp") if isinstance(contract.get("mcp"), dict) else {}
        if dot and parent == "mcp" and key in mcp_section:
            # The same JSON path as in the whole contract: reply["mcp"][key].
            return {"schema_version": contract.get("schema_version"), "section": section,
                    "mcp": {key: mcp_section[key]}}
        raise InterfaceError(
            "invalid_parameter", f"unknown contract section: {_bounded_echo(section)!r}",
            field="section", got=section,
            expected=[DESCRIBE_INDEX, DESCRIBE_ALL, *contract, *(f"mcp.{name}" for name in mcp_section)],
            hint="dic_describe() without arguments lists every section and every mcp.<key> part with its size",
        )

    return _dispatch("dic_describe", params, operation, envelope=False)


def dic_get_pitfalls() -> dict[str, Any]:
    """Return the machine-readable historical pitfall list."""

    def operation() -> dict[str, Any]:
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        return {"schema_version": contract["schema_version"], "pitfalls": contract["pitfalls"]}

    return _dispatch("dic_get_pitfalls", {}, operation, envelope=False)


def dic_probe_images(image_dir: ProbeDirArg) -> dict[str, Any]:
    """Probe an allowlisted event directory without starting the GPU."""

    params = {"image_dir": image_dir}

    def operation() -> dict[str, Any]:
        directory = _resolve_read_path(image_dir, "image_dir", kind="dir")
        images = _validate_event_images(directory, "image_dir")
        if not images:
            samples, total = _sample_names(directory)
            raise InterfaceError(
                "no_images", f"no recognized images in {directory}: {IMAGE_NAME_RULE}",
                field="image_dir",
                hint="other naming schemes need the pipeline CLI's --image-pattern (not available through MCP)",
                directory_entries=total, sample_names=samples,
            )
        first = _resolve_read_path(images[0][1], "first_image", kind="file")
        probe = pipeline.probe_image(str(first))
        warnings: list[str] = []
        if len(images) < 2:
            warnings.append("only one recognized image: nothing to correlate (the first image is the reference)")
        return {
            "ok": True,
            "image_dir": str(directory),
            "images": len(images),
            "reference_frame": images[0][0],
            "target_frames": max(0, len(images) - 1),
            "runnable": len(images) >= 2,
            "first_frame": probe,
            "input_recommendation": pipeline.recommend_input_mode(probe),
            "warnings": warnings,
        }

    return _dispatch("dic_probe_images", params, operation)


def dic_plan(params: PlanParams) -> dict[str, Any]:
    """Return pipeline --dry-run for exactly the params a run would use."""

    raw = _params_dict(params)

    def operation() -> dict[str, Any]:
        normalized = _normalize_pipeline_params(raw)
        _check_output_state(normalized)
        plan = _build_plan(normalized)
        if normalized["mode"] == "campaign" and len(plan["events"]) > MAX_CAMPAIGN_EVENTS:
            plan["valid"] = False
            plan["ok"] = False
            plan["exit_code"] = 2
            plan["errors"].append({
                "code": "campaign_too_large",
                "message": f"campaign has {len(plan['events'])} events; MCP limit is {MAX_CAMPAIGN_EVENTS}",
            })
        return _shape_plan_reply(plan, selection=_uses_event_selection(normalized))

    return _dispatch("dic_plan", raw, operation)


def dic_run_event(params: EventRunParams) -> dict[str, Any]:
    """Run one validated image_dir and return its summary."""

    raw = _params_dict(params)
    return _dispatch(
        "dic_run_event", raw,
        lambda: _run_pipeline(_normalize_pipeline_params(raw, expected_mode="event")),
    )


def dic_run_campaign(params: CampaignRunParams) -> dict[str, Any]:
    """Run every event under photo_dir, or those events selects (at most 120), and return the summary."""

    raw = _params_dict(params)
    return _dispatch(
        "dic_run_campaign", raw,
        lambda: _run_pipeline(_normalize_pipeline_params(raw, expected_mode="campaign")),
    )


def _is_run_dir(directory: Path) -> bool:
    return (directory / "summary.json").exists() or (directory / JOB_RECORD_NAME).exists()


def _enclosing_run_dir(path: Path, job_root: Path) -> Path | None:
    """Nearest directory at or above ``path`` (inside the job root) that is a run directory."""

    current = path
    while _is_relative_to(current, job_root) and current != job_root:
        if current.is_dir() and _is_run_dir(current):
            return current
        current = current.parent
    return None


def _check_aggregate_destination(destination: Path, overwrite: bool) -> Path | None:
    """Apply the aggregation write rules; return the run directory to lock, if any."""

    job_root = _ensure_job_root()
    if destination.suffix.lower() != ".csv":
        raise InterfaceError(
            "aggregate_output_rejected", f"output_path must end in .csv: {destination}",
            field="output_path", got=str(destination), expected="<job root>/.../<name>.csv",
        )
    relative_parts = destination.relative_to(job_root).parts
    if any(part.startswith("_frames_") for part in relative_parts[:-1]) or any(destination.parent.glob("frame_*.csv")):
        raise InterfaceError(
            "aggregate_output_rejected",
            f"output_path is inside a frames directory: {destination}",
            field="output_path", hint="write next to the run (e.g. <output_dir>/cod_<lines>.csv), never among frame CSVs",
        )
    if destination.name == "cod_result.csv" or _FRAME_CSV_RE.match(destination.name):
        raise InterfaceError(
            "aggregate_output_rejected",
            f"output_path uses a name reserved for pipeline products: {destination.name}",
            field="output_path", hint="choose another file name, e.g. cod_<L0>_<L1>_<method>.csv",
        )
    run_dir = _enclosing_run_dir(destination.parent, job_root)
    if destination.exists():
        if run_dir is not None:
            raise InterfaceError(
                "aggregate_output_rejected",
                f"output_path already exists inside run directory {run_dir}; files there are never replaced",
                field="output_path", hint="choose a new file name",
            )
        if not overwrite:
            raise InterfaceError(
                "output_exists", f"output_path exists; set overwrite=true explicitly: {destination}",
                field="output_path",
            )
        if not destination.is_file():
            raise InterfaceError(
                "output_exists", f"output_path exists and is not a file: {destination}", field="output_path",
            )
    return run_dir


def _available_rows(source: Path) -> dict[str, Any]:
    import numpy as np

    frames = sorted(source.glob("frame_*.csv"), key=lambda path: int(path.stem.split("_")[1]))
    if not frames:
        return {}
    try:
        data = np.atleast_2d(np.loadtxt(frames[0], delimiter=",", skiprows=1, dtype=np.float64))
        rows = sorted({int(value) for value in data[:, 1]}) if data.size else []
    except (OSError, ValueError, IndexError):
        return {"first_frame": frames[0].name, "available_line_ys": None}
    return {
        "first_frame": frames[0].name,
        "available_line_ys": rows[:50],
        "available_line_ys_count": len(rows),
    }


def dic_aggregate_cod(
    frames_dir: FramesDirArg,
    output_path: OutputPathArg,
    line_y: AggregateLineArg,
    cal: AggregateCalArg,
    method: MethodArg = "naive",
    roi_cols: AggregateRoiArg = None,
    overwrite: AggregateOverwriteArg = False,
) -> dict[str, Any]:
    """Re-aggregate retained frame CSV files without running DIC."""

    params = {
        "frames_dir": frames_dir, "output_path": output_path, "line_y": line_y,
        "cal": cal, "method": method, "roi_cols": roi_cols, "overwrite": overwrite,
    }

    def operation() -> dict[str, Any]:
        source = _resolve_read_path(frames_dir, "frames_dir", kind="dir")
        frame_count = _validate_frame_files(source)
        destination = _resolve_write_path(output_path, "output_path")
        if method not in {"fillboundary", "naive"}:
            raise InterfaceError(
                "invalid_parameter", "method must be fillboundary or naive", field="method",
                got=method, expected=["fillboundary", "naive"],
            )
        l0_y, l1_y = _checked_line_pair(line_y)
        calibration = _number(cal, "cal", minimum=0, maximum=10, exclusive_minimum=True,
                              hint="cal is millimetres per pixel")
        roi = None
        if roi_cols is not None:
            roi = _checked_roi(roi_cols)
        if not isinstance(overwrite, bool):
            raise InterfaceError("invalid_parameter", "overwrite must be a boolean", field="overwrite",
                                 got=overwrite, expected="true or false")
        if frame_count == 0:
            raise InterfaceError(
                "aggregation_failed", f"no frame_*.csv files found in {source}", field="frames_dir",
                hint="frame CSVs are kept only by runs with keep_frames=true, in <output_dir>/_frames_<event>/",
            )
        run_dir = _check_aggregate_destination(destination, overwrite)
        job_root = _ensure_job_root()
        source_run = _enclosing_run_dir(source, job_root) if _is_relative_to(source, job_root) else None
        lock_targets = sorted({path for path in (run_dir, source_run) if path is not None}, key=str)
        holder = {"tool": "dic_aggregate_cod", "call_id": _current_call_id()}
        with contextlib.ExitStack() as stack:
            for target in lock_targets:
                stack.enter_context(_output_lock(target, holder))
            # The destination may have appeared while we were not holding the lock.
            _check_aggregate_destination(destination, overwrite)
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                cod_um, x_mm, frame_numbers, gauge_mm = pipeline.aggregate_cod(
                    source, l0_y, l1_y, calibration, roi, method,
                )
            except pipeline.CodAggregationError as exc:
                raise InterfaceError(
                    exc.code, str(exc), field="method" if exc.code == "cod_uy_column_missing" else None,
                    hint="through MCP pass method='naive' (dic_aggregate_cod) or cod_method='naive' (run tools) "
                         "to accept the un-extrapolated difference explicitly",
                ) from exc
            except (ValueError, IndexError, OSError) as exc:
                raise InterfaceError(
                    "aggregation_failed", f"cannot read the frame CSVs in {source}: {exc}", field="frames_dir",
                    hint="frame CSVs must be the binary's comma-separated output with a header row",
                ) from exc
            if cod_um is None:
                raise InterfaceError("aggregation_failed", f"no frame_*.csv files found in {source}",
                                     field="frames_dir")
            rows = int(sum(math.isfinite(float(value)) for value in cod_um.flat))
            if rows == 0:
                available = _available_rows(source)
                raise InterfaceError(
                    "no_cod_rows",
                    f"line_y [{l0_y}, {l1_y}] produced no COD value: at least one of the rows has no valid "
                    f"(zncc >= 0) point in the frame CSVs{' inside roi_cols' if roi else ''}; nothing was written",
                    field="line_y", got=[l0_y, l1_y],
                    expected="two rows listed in available_line_ys",
                    hint="use the run's resolved_params.line_ys (with interface_y: interface_y -/+ gauge//2, "
                         "gauge default 35)",
                    **available,
                )
            fd, temporary_name = tempfile.mkstemp(prefix=".cod-", suffix=".csv", dir=destination.parent)
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                pipeline.write_cod_csv(temporary, cod_um, x_mm, frame_numbers)
                if overwrite:
                    os.replace(temporary, destination)
                else:
                    try:
                        os.link(temporary, destination)
                    except FileExistsError as exc:
                        raise InterfaceError(
                            "output_exists", f"output_path appeared during aggregation: {destination}",
                            field="output_path",
                        ) from exc
                    temporary.unlink()
            finally:
                if temporary.exists():
                    temporary.unlink()
        warnings: list[dict[str, str]] = []
        if l0_y > l1_y:
            warnings.append({
                "code": "line_y_descending",
                "message": f"line_y=[{l0_y}, {l1_y}] puts L0 below L1: every COD value has the opposite sign "
                           f"to line_y=[{l1_y}, {l0_y}]",
            })
        return {
            "ok": True,
            "output_path": str(destination),
            "method": method,
            "frames": int(len(frame_numbers)),
            "x_positions": int(len(x_mm)),
            "rows": rows,
            "gauge_mm": float(gauge_mm),
            "cod_definition": _cod_definition([l0_y, l1_y], method),
            "warnings": warnings,
        }

    return _dispatch("dic_aggregate_cod", params, operation)


# --- job cancellation -------------------------------------------------------


def _is_job_runner(pid: Any, record_path: Path, start_ticks: Any = None) -> bool:
    """``pid`` is running and is the job runner of this record, not a process that reused its pid.

    The runner's argv holds its spec, which names the record (in quotes, so no
    other record's path matches), and the record holds the runner's start time
    (``runner_start_ticks``; older records have none). A zombie has no argv and
    is not a runner. Without procfs the pid is all there is to go by.
    """

    if not _pid_alive(pid):
        return False
    if not _procfs_available():
        return True
    if isinstance(start_ticks, int) and not isinstance(start_ticks, bool) and _proc_start_ticks(pid) != start_ticks:
        return False
    cmdline = _proc_arguments(pid)  # empty for a zombie; waits out the instant of exec
    return cmdline is not None and json.dumps(str(record_path)).encode() in cmdline


def _job_group_alive(pgid: Any, directory: Path, start_ticks: Any = None) -> bool:
    """Process group ``pgid`` exists and is the job that writes ``directory`` (_JOB_IDENTITY_SOURCE).

    ``start_ticks`` is the job record's ``job_start_ticks``, the start time of
    the pipeline, which leads the group; arguments must name the directory
    exactly (review 2026-10-05 F2: a substring matched /jobs/run10 for /jobs/run1).
    """

    return bool(_JOB_IDENTITY["job_group_alive"](pgid, directory, start_ticks))


def _stop_job_group(pgid: Any, directory: Path, start_ticks: Any = None) -> None:
    """SIGTERM, five seconds, SIGKILL: what the job runner does to its job.

    Before each signal the group is checked to be the job's, so a number that
    another process took over in between is not signalled.
    """

    def alive() -> bool:
        return _job_group_alive(pgid, directory, start_ticks)

    if not alive():
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGTERM)
    grace = time.monotonic() + 5.0
    while alive() and time.monotonic() < grace:
        time.sleep(0.05)
    if alive():
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)
    gone = time.monotonic() + 2.0
    while alive() and time.monotonic() < gone:
        time.sleep(0.05)


def _cancel_token_matches(token: Any, record: dict[str, Any]) -> bool:
    expected = record.get("cancel_token_sha256")
    if not isinstance(token, str) or not isinstance(expected, str) or not expected:
        return False
    return hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(), expected)


def dic_job_cancel(
    output_dir: JobDirArg,
    job_id: JobIdArg = None,
    cancel_token: CancelTokenArg = None,
) -> dict[str, Any]:
    """Stop a running MCP job: SIGTERM to its job runner's process group."""

    # The audit log never holds the token itself: _audit_record writes its digest (_redact_secrets),
    # also for a call that the schema rejects before this function runs.
    params: dict[str, Any] = {"output_dir": output_dir}
    if job_id is not None:
        params["job_id"] = job_id
    if cancel_token is not None:
        params["cancel_token"] = cancel_token

    def operation() -> dict[str, Any]:
        if job_id is not None and not (isinstance(job_id, str) and re.fullmatch(r"[0-9a-f]{32}", job_id)):
            raise InterfaceError(
                "invalid_parameter", "job_id must be the 32 hex digits a run tool returned as job_id (call_id)",
                field="job_id", got=job_id, expected="32 lowercase hexadecimal digits",
            )
        if cancel_token is not None and not (isinstance(cancel_token, str) and 1 <= len(cancel_token) <= 128):
            raise InterfaceError(
                "invalid_parameter", "cancel_token must be the string a wait=false run returned",
                field="cancel_token", expected="string of 1 to 128 characters",
            )
        directory = _resolve_write_path(output_dir, "output_dir")
        record_path = directory / JOB_RECORD_NAME
        record = None
        if directory.is_dir() and not record_path.is_symlink():
            record = _read_small_json(record_path, 1 << 20)
        if record is None or record.get("schema") != JOB_RECORD_SCHEMA:
            raise InterfaceError(
                "job_not_found", f"there is no MCP job in {directory} (no {JOB_RECORD_NAME})",
                field="output_dir",
                hint="a job is named by the output_dir given to dic_run_event or dic_run_campaign; "
                     "check the path (a wait=false reply returns it as output_dir)",
            )
        if job_id is not None and record.get("call_id") != job_id:
            raise InterfaceError(
                "job_id_mismatch",
                f"the job in {directory} has id {record.get('call_id')}, not {job_id}; nothing was stopped",
                field="job_id", got=job_id, expected=record.get("call_id"),
                hint="pass the output_dir and job_id of one and the same wait=false reply, or leave job_id out",
            )
        call_id = _current_call_id()

        def own_job() -> dict[str, Any] | None:
            with _REGISTRY_GUARD:
                job = _ACTIVE_JOBS.get(str(record.get("call_id")))
            return job if job is not None and Path(job["record"]) == record_path else None

        own = own_job()
        # A job caught between its record and its runner (a few milliseconds) is not a stale one.
        just_started = time.monotonic() + 2.0
        while (own is None and record.get("state") == "starting" and "runner_pid" not in record
               and time.monotonic() < just_started):
            time.sleep(0.05)
            record = _read_small_json(record_path, 1 << 20) or record
            own = own_job()

        def latest() -> dict[str, Any]:
            return _read_small_json(record_path, 1 << 20) or record

        def live_runner() -> int | None:
            """pid of the runner that watches the job at this moment, None when there is none.

            Asked again every time it matters (review 2026-10-05 F3): a runner that
            was killed is replaced by a second one, another pid, a moment later, and
            a pid read earlier may by then be dead, a zombie or somebody else's.
            """

            job = own_job()
            if job is not None:
                runner = job["runner"]  # our own child: poll() also sees a zombie as ended
                return runner.pid if runner.poll() is None else None
            current = latest()
            pid = current.get("runner_pid")
            return pid if _is_job_runner(pid, record_path, current.get("runner_start_ticks")) else None

        def group_alive() -> bool:
            # Read again each time: a job cancelled while starting gets its process group later.
            current = latest()
            return _job_group_alive(current.get("job_pgid"), directory, current.get("job_start_ticks"))

        def reply(action: str, final: dict[str, Any], note: str) -> dict[str, Any]:
            result: dict[str, Any] = {
                "ok": True,
                "action": action,
                "state": final.get("state"),
                "job_id": final.get("call_id"),
                "output_dir": str(directory),
                "job_record": str(record_path),
                "job": {key: final[key] for key in (
                    "state", "tool", "gpu_device", "exit_code", "ended_at", "wall_s", "cancel_reason", "verdict",
                    "note") if key in final},
                "partial": {"frame_files": sum(1 for _ in directory.glob("_frames_*/frame_*")),
                            "summary_json": (directory / "summary.json").is_file()},
                "processes": {"runner_alive": live_runner() is not None, "job_group_alive": group_alive()},
                "note": note,
                "_audit_code": f"cancel:{action}",
            }
            if isinstance(final.get("progress"), dict):
                result["progress"] = final["progress"]
            if action == "cancelled":
                result["hint"] = ("this output_dir holds partial output and cannot be reused: rerun with a NEW "
                                  "output_dir")
            return result

        if record.get("state") not in JOB_ACTIVE_STATES:
            return reply("none", record,
                         f"the job had already ended (state {record.get('state')}); nothing was signalled")

        if own is None and live_runner() is None and not group_alive():
            # Nothing of the job is running and no server watches it: the record
            # is stale (its runner and its server are gone). Anyone may settle it.
            present = (directory / "summary.json").is_file()
            final = _finalize_job_record(
                record_path, **_unrecorded_end(record_path, present), ended_at=_utc(), exit_code=None,
                summary_present=present, settled_by=f"dic_job_cancel (call {call_id})",
                note="the job ended without its runner or server recording it; dic_job_cancel found no process "
                     "of it and settled the record",
            )
            return reply("settled", final,
                         "no process of this job was running; the stale record was settled as "
                         f"{final.get('state')}" + (" (the pipeline left a summary: read summary.json)"
                                                    if present else ""))

        if own is None and not _cancel_token_matches(cancel_token, record):
            raise InterfaceError(
                "job_not_owned",
                "this job was started by another MCP session; only that session, or a caller with the "
                "cancel_token its wait=false call returned, may cancel it"
                + ("" if record.get("cancel_token_sha256") else " (it was started with wait=true, so there is "
                                                                 "no token)"),
                field="cancel_token",
                hint="leave the job alone: it ends by itself or at deadline_at, and gpu_busy clears then; to "
                     "cancel your own job from a new session pass the cancel_token of its wait=false reply",
                job={key: record[key] for key in ("state", "tool", "gpu_device", "created_at", "deadline_at")
                     if key in record},
            )

        # Written first: whoever records the end of the job from here on (a runner, the server
        # that watches it, this call) knows that a cancellation was asked for.
        request = {"call_id": call_id, "server_pid": os.getpid(), "at": _utc()}
        _merge_job_record(record_path, cancel_requested_at=request["at"], cancel_requested_by=request)
        signalled: set[int] = set()
        group_stopped = False

        def push() -> None:
            """One step towards a stopped job.

            SIGTERM to the runner that watches the job now (once per runner): it
            leads its own process group, stops the pipeline's group (SIGTERM,
            SIGKILL after 5 s), records cancelled and exits, which releases the
            locks. Without a runner (it was killed, or the one that was signalled
            died before it acted) the job's process group is stopped directly.
            """

            nonlocal group_stopped
            pid = live_runner()
            if pid is not None:
                if pid not in signalled:
                    signalled.add(pid)
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.killpg(pid, signal.SIGTERM)
            elif not group_stopped and group_alive():
                group_stopped = True
                current = latest()
                _stop_job_group(current.get("job_pgid"), directory, current.get("job_start_ticks"))

        def settled() -> dict[str, Any] | None:
            current = _read_small_json(record_path, 1 << 20) or {}
            if current.get("state") in JOB_ACTIVE_STATES or live_runner() is not None or group_alive():
                return None
            return current

        deadline = time.monotonic() + CANCEL_WAIT_SECONDS
        while True:
            push()
            final = settled()
            if final is not None or time.monotonic() >= deadline:
                break
            if live_runner() is None and not group_alive():
                break  # nothing of the job is left: nobody is going to stop anything, the record is written below
            time.sleep(0.05)
        forced = False
        if final is None:
            # Either nothing of the job is left and its end is not recorded yet, or the wait is over:
            # then what is left is killed, the runner that watches the job now and the job's group
            # (both identified again, not from a pid read earlier). A final state that somebody
            # else writes in the meantime stays (_finalize_job_record).
            pid = live_runner()
            forced = pid is not None or group_alive()
            if pid is not None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(pid, signal.SIGKILL)
            current = latest()
            if _job_group_alive(current.get("job_pgid"), directory, current.get("job_start_ticks")):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(current["job_pgid"], signal.SIGKILL)
            gone = time.monotonic() + 3.0
            while (live_runner() is not None or group_alive()) and time.monotonic() < gone:
                time.sleep(0.05)
            present = (directory / "summary.json").is_file()
            if forced:
                outcome = {"state": "cancelled",
                           "cancel_reason": f"dic_job_cancel (call {call_id}); the job was not stopped in time, "
                                            "so what was left of it was killed"}
            else:
                # cancelled, unless the pipeline left a summary: then it had ended by itself (runner_lost).
                outcome = _unrecorded_end(record_path, present)
            final = _finalize_job_record(record_path, **outcome, ended_at=_utc(), exit_code=None,
                                         summary_present=present)
        # The server thread that waited for this job (in this process, or in the session that started
        # it) leaves the lock contexts a moment after the runner has exited: the reply says that the
        # locks are released, so wait until this job no longer holds its GPU lock.
        def still_locked() -> bool:
            with _REGISTRY_GUARD:
                if str(record.get("call_id")) in _ACTIVE_JOBS:
                    return True
            device = record.get("gpu_device")
            if not isinstance(device, int) or isinstance(device, bool):
                return False
            with contextlib.suppress(InterfaceError, OSError):
                holder = _lock_holder_if_held(_gpu_lock_path(device))
                return holder is not None and holder.get("call_id") == record.get("call_id")
            return False

        released = time.monotonic() + LOCK_RELEASE_WAIT_SECONDS
        while still_locked() and time.monotonic() < released:
            time.sleep(0.01)
        locks_released = not still_locked()
        if final.get("state") in JOB_ACTIVE_STATES:
            raise InterfaceError(
                "cancel_failed", f"the job in {directory} could not be stopped", retryable=True,
                output_dir=str(directory), job_record=str(record_path),
                hint="retry; if it persists the operator must stop the processes of this job by hand",
            )
        if final.get("state") == "cancelled":
            return reply("cancelled", final,
                         "the job was stopped" + (" (forced with SIGKILL)" if forced else "")
                         + ("; its GPU and output locks are released" if locks_released else
                            "; the server that waited for it has not released its GPU lock yet: a run that "
                            "meets gpu_busy now can be retried in a moment"))
        return reply("none", final, f"the job ended by itself (state {final.get('state')}) before the "
                                    "cancellation took effect")

    return _dispatch("dic_job_cancel", params, operation)


# --- bounded manifest query -------------------------------------------------

_CONDENSE_THRESHOLD = 16
_STRING_LIMIT = 512
_PATH_SEGMENT_RE = re.compile(r"^(?P<key>[^\[\]]*)(?P<indexes>(?:\[[^\[\]]*\])*)$")


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode())


def _summarize_list(items: list[Any], threshold: int = _CONDENSE_THRESHOLD) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": len(items)}
    if items and all(isinstance(item, dict) for item in items):
        keys: list[str] = []
        for item in items:
            for key in item:
                if key not in keys and len(keys) < 40:
                    keys.append(key)
        numeric: dict[str, Any] = {}
        boolean: dict[str, Any] = {}
        categorical: dict[str, Any] = {}
        nonempty: dict[str, int] = {}
        for key in keys:
            values = [item.get(key) for item in items]
            numbers = [float(v) for v in values
                       if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
            if numbers:
                numeric[key] = {"min": min(numbers), "max": max(numbers),
                                "mean": round(sum(numbers) / len(numbers), 6), "n": len(numbers)}
            flags = [v for v in values if isinstance(v, bool)]
            if flags:
                boolean[key] = {"true": sum(flags), "false": len(flags) - sum(flags)}
            strings = [v for v in values if isinstance(v, str)]
            if strings and len(set(strings)) <= 10:
                counts: dict[str, int] = {}
                for value in strings:
                    counts[value] = counts.get(value, 0) + 1
                categorical[key] = counts
            lists = [v for v in values if isinstance(v, list)]
            if lists:
                nonempty[key] = sum(1 for v in lists if v)
        summary.update({"item_keys": keys, "numeric": numeric})
        if boolean:
            summary["boolean"] = boolean
        if categorical:
            summary["categorical"] = categorical
        if nonempty:
            summary["nonempty_lists"] = nonempty
        # Name the items whose status is not ok (failed events of a summary), so a
        # condensed campaign still says which events failed (review 2026-10-04 F3).
        not_ok = [{"index": index, "name": item.get("name"), "status": item.get("status")}
                  for index, item in enumerate(items) if "status" in item and item.get("status") != "ok"]
        if not_ok:
            summary["not_ok"] = not_ok[:50]
            if len(not_ok) > 50:
                summary["not_ok_count"] = len(not_ok)
        summary["first"] = _condense(items[0], threshold)[0]
        summary["last"] = _condense(items[-1], threshold)[0]
    elif items and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in items):
        finite = [float(item) for item in items if math.isfinite(item)]
        if finite:
            summary.update({"min": min(finite), "max": max(finite), "mean": round(sum(finite) / len(finite), 6)})
        summary["first"] = items[:3]
    else:
        summary["first"] = [_condense(item, threshold)[0] for item in items[:3]]
    return summary


def _condense(value: Any, threshold: int = _CONDENSE_THRESHOLD) -> tuple[Any, bool]:
    """Condensed copy of a JSON value and whether anything was condensed.

    Lists longer than ``threshold`` become a summary (counts, statistics, the
    items whose status is not ok); strings longer than 512 characters are cut.
    """

    if isinstance(value, dict):
        changed = False
        out = {}
        for key, item in value.items():
            out[key], item_changed = _condense(item, threshold)
            changed = changed or item_changed
        return out, changed
    if isinstance(value, list):
        if len(value) > threshold:
            return {"_condensed_list": _summarize_list(value, threshold),
                    "_how_to_read": "pass fields=['<path>[start:stop]'] to read items"}, True
        changed = False
        out_list = []
        for item in value:
            condensed, item_changed = _condense(item, threshold)
            out_list.append(condensed)
            changed = changed or item_changed
        return out_list, changed
    if isinstance(value, str) and len(value) > _STRING_LIMIT:
        return value[:_STRING_LIMIT] + f"... [{len(value)} chars]", True
    return value, False


def _select_path(document: Any, path: str) -> Any:
    current = document
    for segment in path.split("."):
        match = _PATH_SEGMENT_RE.match(segment)
        if not match or (not match.group("key") and not match.group("indexes")):
            raise KeyError(f"bad path segment {segment!r}")
        key = match.group("key")
        if key:
            if not isinstance(current, dict) or key not in current:
                raise KeyError(f"no key {key!r}")
            current = current[key]
        for index in re.findall(r"\[([^\[\]]*)\]", match.group("indexes")):
            if not isinstance(current, list):
                raise KeyError(f"{segment!r} indexes a non-array")
            try:
                if ":" in index:
                    start_text, stop_text = index.split(":", 1)
                    start = int(start_text) if start_text.strip() else None
                    stop = int(stop_text) if stop_text.strip() else None
                    current = current[start:stop]
                else:
                    current = current[int(index)]
            except (ValueError, IndexError) as exc:
                raise KeyError(f"bad index [{index}] ({exc})") from exc
    return current


def _string_list(value: Any, field: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value) or len(value) > 32:
        raise InterfaceError("invalid_parameter", f"{field} must be an array of at most 32 strings",
                             field=field, got=value, expected="array of strings")
    if any(len(item) > QUERY_FIELD_MAX_CHARS for item in value):
        # Every entry may be echoed back (missing_fields): bound it (F2).
        raise InterfaceError("invalid_parameter",
                             f"each entry of {field} must be at most {QUERY_FIELD_MAX_CHARS} characters",
                             field=field, got=value, expected=f"strings of at most {QUERY_FIELD_MAX_CHARS} characters")
    return list(value)


def dic_query_manifest(
    run_dir: RunDirArg,
    detail: DetailArg = "auto",
    fields: FieldsArg = None,
    documents: DocumentsArg = None,
    max_bytes: MaxBytesArg = DEFAULT_QUERY_MAX_BYTES,
) -> dict[str, Any]:
    """Read JSON summaries/manifests inside an allowlisted run directory, size-bounded."""

    params: dict[str, Any] = {"run_dir": run_dir}
    for key, value, default in (("detail", detail, "auto"), ("fields", fields, None),
                                ("documents", documents, None), ("max_bytes", max_bytes, DEFAULT_QUERY_MAX_BYTES)):
        if value != default:
            params[key] = value

    def operation() -> dict[str, Any]:
        directory = _resolve_read_path(run_dir, "run_dir", kind="dir")
        if detail not in QUERY_DETAILS:
            raise InterfaceError("invalid_parameter", f"detail must be one of {', '.join(QUERY_DETAILS)}",
                                 field="detail", got=detail, expected=list(QUERY_DETAILS))
        wanted_fields = _string_list(fields, "fields")
        wanted_documents = _string_list(documents, "documents")
        budget = _integer(max_bytes, "max_bytes", minimum=MIN_QUERY_MAX_BYTES, maximum=MAX_QUERY_MAX_BYTES)
        # _dispatch adds call_id (32 hex digits) after this returns: keep room for it.
        limit = budget - len(',"call_id":""') - 32
        candidates = set(directory.glob("*.json"))
        candidates.update(directory.glob("*/*.manifest.json"))
        if not candidates:
            raise InterfaceError(
                "manifest_not_found", f"no direct JSON summary/manifest found in {directory}", field="run_dir",
                hint="run tools write summary.json and .mcp-job.json into their output_dir",
            )

        def relative(path: Path) -> str:
            return _printable_path(str(path.relative_to(directory)))

        def priority(name: str) -> tuple[int, str]:
            rank = {"summary.json": 0, JOB_RECORD_NAME: 1}.get(name, 2 if name.endswith("manifest.json") else 3)
            return (rank, name)

        # Priority order before the 32-document cut, and the documents filter
        # before it too, so summary.json can never be cut off (F14).
        by_priority = sorted(candidates, key=lambda path: priority(relative(path)))
        all_names = {relative(path) for path in by_priority}
        listed = by_priority[:32]
        selected = ([path for path in by_priority if relative(path) in wanted_documents][:32]
                    if wanted_documents is not None else listed)
        errors: list[dict[str, str]] = []

        def outside_allowlist(path: Path) -> bool:
            if not path.is_symlink():
                return False
            try:
                _resolve_read_path(str(path), "manifest", kind="file")
                return False
            except InterfaceError:
                return True

        available = []
        for candidate in listed:
            name = relative(candidate)
            if outside_allowlist(candidate):
                # Neither the target's size nor its path is reported (F10).
                available.append({"name": name, "bytes": None, "skipped": "symlink outside the read allowlist"})
                continue
            with contextlib.suppress(OSError):
                available.append({"name": name, "bytes": candidate.stat().st_size})
        loaded: dict[str, Any] = {}
        missing_fields: dict[str, list[str]] = {}
        for candidate in selected:
            name = relative(candidate)
            if outside_allowlist(candidate):
                errors.append({"path": _printable_path(str(candidate)),
                               "error": "symlink to a path outside the read allowlist; not read"})
                continue
            try:
                resolved = _resolve_read_path(str(candidate), "manifest", kind="file")
                if resolved.stat().st_size > MAX_MANIFEST_BYTES:
                    raise InterfaceError("manifest_too_large", f"manifest exceeds 10 MiB: {name}")
                document = json.loads(resolved.read_text(encoding="utf-8"))
            except (InterfaceError, OSError, ValueError) as exc:
                # ValueError covers JSON errors and undecodable bytes (F14).
                errors.append({"path": _printable_path(str(candidate)), "error": _printable_path(str(exc))[:300]})
                continue
            if wanted_fields is not None:
                chosen: dict[str, Any] = {}
                for path in wanted_fields:
                    try:
                        chosen[path] = _select_path(document, path)
                    except KeyError as exc:
                        missing_fields.setdefault(name, []).append(f"{path}: {str(exc.args[0])[:80]}")
                document = chosen
            loaded[name] = document
        if wanted_documents is not None:
            unknown = sorted(set(wanted_documents) - all_names)
            if unknown:
                errors.append({"path": ", ".join(unknown)[:300], "error": "no such document in this directory"})

        views: dict[str, str] = {}
        out_documents: dict[str, Any] = {}
        omitted: list[dict[str, Any]] = []
        full_sizes = {name: _json_size(value) for name, value in loaded.items()}
        notes: list[str] = []

        def reply() -> dict[str, Any]:
            result = {
                "ok": not errors,
                "run_dir": str(directory),
                "documents": out_documents,
                "errors": errors,
                "truncated": len(candidates) > len(listed),
                "detail": detail,
                "document_views": views,
                "available_documents": available,
                "max_bytes": budget,
                "complete": (not omitted and not errors and not missing_fields
                             and all(view == "full" for view in views.values())),
            }
            if omitted:
                result["omitted"] = omitted
                result["hint"] = ("omitted documents did not fit in max_bytes: request them with documents=[...], "
                                  "select parts with fields=[...] or raise max_bytes (<= "
                                  f"{MAX_QUERY_MAX_BYTES})")
            if missing_fields:
                result["missing_fields"] = missing_fields
            if notes:
                result["budget_notes"] = notes
            return result

        def variants(value: Any) -> list[tuple[Any, str]]:
            options: list[tuple[Any, str]] = [] if detail == "compact" else [(value, "full")]
            if detail != "full":
                for threshold in (_CONDENSE_THRESHOLD, 4, 1):
                    condensed, changed = _condense(value, threshold)
                    options.append((condensed, "condensed" if changed else "full"))
            return options

        # Highest priority first; each document takes the first view that still
        # fits, and one that fits in no view is omitted, so a large summary.json
        # no longer pushes out a small job record (F2).
        for name in sorted(loaded, key=priority):
            for value, view in variants(loaded[name]):
                out_documents[name], views[name] = value, view
                if _json_size(reply()) <= limit:
                    break
                del out_documents[name], views[name]
            else:
                entry: dict[str, Any] = {"document": name, "bytes_full": full_sizes.get(name),
                                         "reason": "exceeds max_bytes"}
                if isinstance(loaded.get(name), dict):
                    entry["top_level_keys"] = list(loaded[name])[:40]
                omitted.append(entry)

        # max_bytes is a hard limit: shrink the envelope itself if needed.
        def drop_top_level_keys() -> None:
            for entry in omitted:
                entry.pop("top_level_keys", None)

        def condense_available() -> None:
            nonlocal available
            if len(available) > 3:
                available = available[:3] + [{"name": f"... {len(available) - 3} more", "bytes": None}]

        def trim_missing() -> None:
            for name in list(missing_fields):
                if len(missing_fields[name]) > 2:
                    missing_fields[name] = missing_fields[name][:2] + [f"... {len(missing_fields[name]) - 2} more"]

        def trim_errors() -> None:
            del errors[3:]
            for item in errors:
                item["error"] = item["error"][:120]

        def drop_documents() -> None:
            while out_documents and _json_size(reply()) > limit:
                name = sorted(out_documents, key=priority)[-1]
                out_documents.pop(name)
                views.pop(name)
                omitted.append({"document": name, "bytes_full": full_sizes.get(name), "reason": "exceeds max_bytes"})

        for shrink in (drop_top_level_keys, condense_available, trim_missing, trim_errors, drop_documents):
            if _json_size(reply()) <= limit:
                break
            shrink()
            notes.append(shrink.__name__)
        result = reply()
        if _json_size(result) > limit:
            result = {"ok": not errors, "run_dir": str(directory)[:200], "documents": {}, "complete": False,
                      "max_bytes": budget, "budget_notes": ["minimal reply: even the envelope exceeded max_bytes"],
                      "hint": "raise max_bytes or narrow fields/documents"}
        return result

    return _dispatch("dic_query_manifest", params, operation)


TOOL_FUNCTIONS = (
    dic_describe,
    dic_get_pitfalls,
    dic_probe_images,
    dic_plan,
    dic_run_event,
    dic_run_campaign,
    dic_aggregate_cod,
    dic_query_manifest,
    dic_job_cancel,
)

_READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
_RUNS = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False}
TOOL_SPECS: dict[str, dict[str, Any]] = {
    "dic_describe": {
        "title": "Interface contract",
        "description": "Read the machine-readable FastICGN interface contract (interface_contract.json). Without "
                       "arguments it returns what matters before a run: an index of the sections (name, size, "
                       "one-line summary), the run params, the guardrails and the tool names, plus the list of the "
                       "parts left out (mcp_view.on_demand). Those are read by name when needed: "
                       "section='mcp.results' (reply bodies), 'mcp.errors', 'mcp.jobs', 'mcp.job_record', "
                       "'mcp.manifest_query', 'mcp.known_issues' and so on; section='mcp' / 'outputs' / "
                       "'exit_codes' / 'pitfalls' returns a whole section and section='all' the whole contract "
                       "(over 120 KB). Usually not needed before a run: the input schemas of these tools carry "
                       "the types, units and bounds.",
        "annotations": _READ_ONLY,
    },
    "dic_get_pitfalls": {
        "title": "Historical pitfalls",
        "description": "Return the pitfall list (subset size vs radius, convergence threshold, zncc status rows, "
                       "12-bit input alignment, COD method history, five-column CSVs). Read it once per session "
                       "before the first run. Notes that hold only for the author's laboratory are a separate "
                       "section: dic_describe(section='lab_notes').",
        "annotations": _READ_ONLY,
    },
    "dic_probe_images": {
        "title": "Probe an image directory",
        "description": "Inspect one image directory without the GPU: number of recognised frames, reference "
                       "frame, first-frame size and bit depth, and the recommended input_depth/input_align. Use "
                       "it before dic_plan on any new dataset. Only names ending in a 6-digit frame number "
                       "with a jpg/jpeg/tif/tiff/png/bmp extension are recognised.",
        "annotations": _READ_ONLY,
    },
    "dic_plan": {
        "title": "Dry-run plan",
        "description": "Validate params exactly as dic_run_event/dic_run_campaign would and return the "
                       "pipeline dry-run: resolved measurement lines, binary argv, per-event probes, errors and "
                       "warnings, plus mcp.suggested_timeout_s. Nothing is written and the GPU is not used. Run "
                       "only if valid=true (ok mirrors it), after reading every entry in warnings (notes are "
                       "context only). A campaign is checked as a whole: one refused event makes the whole plan "
                       "invalid, and with more than 10 events each event is one compact line. For a campaign, "
                       "events runs only some events and event_interface_y gives matched events their own "
                       "interface row; event_selection and mcp.cod_definition.per_event_lines show the result.",
        "annotations": _READ_ONLY,
    },
    "dic_run_event": {
        "title": "Run one event",
        "description": "Run GPU DIC + COD aggregation for ONE image directory (lowest frame number = "
                       "reference) and block until the job ends (typically 5-30 s per 800 frames). Needs a NEW "
                       "output_dir under the job root; writes cod_result.csv (frame,x_mm,cod_um; cod_um is the "
                       "slip across the interface, u on line L0 minus u on line L1, not an opening), summary.json "
                       "and a job record .mcp-job.json. The pipeline runs in --strict mode, so a failed event "
                       "(an empty or all-zero COD included) gives verdict fail and ok false. Check ok, verdict and "
                       "events[0].warning_details; set keep_frames=true if you will call dic_aggregate_cod. "
                       "With wait=false the call returns once the job has started (job_id, output_dir, "
                       "job_record, cancel_token) instead of blocking: poll dic_query_manifest and stop the job "
                       "with dic_job_cancel.",
        "annotations": _RUNS,
    },
    "dic_run_campaign": {
        "title": "Run a campaign",
        "description": f"Run every event subdirectory of photo_dir, or the ones events selects (at most "
                       f"{MAX_CAMPAIGN_EVENTS}), with the "
                       "same params, one after another on one GPU, and block until all end; the --strict pre-check "
                       "is all or nothing, so one refused event refuses the whole campaign (plan_invalid). Writes "
                       "<output_dir>/<event>.csv per event plus summary.json. A campaign with failed events "
                       "still returns a summary (pipeline --strict exit 3, ok false): check verdict, "
                       "mcp.brief (it names the failed events) and every events[].status. With more than 10 events "
                       "each event is one compact line; the full summary is <output_dir>/summary.json "
                       "(dic_query_manifest). events (names or globs) runs only some events; event_interface_y "
                       "([{pattern, interface_y}]) moves the lines of matched events, and mcp.brief then names "
                       "the lines each event used. For a long campaign pass wait=false: the call returns once "
                       "the job has started (job_id, output_dir, job_record, cancel_token); "
                       "dic_query_manifest(run_dir=output_dir, documents=['.mcp-job.json']) then shows state and "
                       "progress (events_done of events_total), and dic_job_cancel stops the job.",
        "annotations": _RUNS,
    },
    "dic_aggregate_cod": {
        "title": "Re-aggregate COD from kept frames",
        "description": "Recompute COD from the frame CSVs of a run made with keep_frames=true "
                       "(<output_dir>/_frames_<event>), without the GPU. line_y must be rows present in those "
                       "frames ([upper, lower]); output_path must be a new .csv under the job root and never a "
                       "pipeline product. If the rows are not in the data it returns ok=false, code "
                       "no_cod_rows and the available rows, and writes nothing.",
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False,
                        "openWorldHint": False},
    },
    "dic_query_manifest": {
        "title": "Read run summaries",
        "description": "Read the JSON documents of a run directory (summary.json, the job record .mcp-job.json, "
                       "*.manifest.json). Large documents are condensed (long arrays become counts and "
                       "statistics, naming items whose status is not ok) and the whole reply stays within "
                       "max_bytes (default 16384); use fields (e.g. "
                       "['verdict','events']) or documents to get specific parts. Anything left out is listed in "
                       "omitted.",
        "annotations": _READ_ONLY,
    },
    "dic_job_cancel": {
        "title": "Cancel a running job",
        "description": "Stop a running job, named by its output_dir: SIGTERM goes to the job runner, which stops "
                       "the pipeline and the binary (SIGKILL after 5 s), records state cancelled and releases the "
                       "GPU and output locks. The reply tells what happened in action: 'cancelled', 'none' (the "
                       "job had already ended, or was cancelled before: nothing is signalled) or 'settled' (no "
                       "process was left and the stale record was closed). A job of another MCP session is "
                       "refused (job_not_owned) unless you pass the cancel_token its wait=false call returned; "
                       "an output_dir without a job is job_not_found. Partial output stays in output_dir, which "
                       "cannot be reused.",
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True,
                        "openWorldHint": False},
    },
}

# Order matters (review 2026-10-05 F6): some MCP clients show only about the first 2000 characters
# of a server's instructions. What the server is, the workflow and the long-jobs paragraph (how to
# use wait=false, and the warning not to cancel other people's jobs) come first and end before
# character 1500; a unit test holds that and the total length (at most 3000).
SERVER_INSTRUCTIONS = (
    "FastICGN runs GPU 2-D digital image correlation (FFTCC + IC-GN) on image sequences. MCP exposes COD mode "
    "only: cod_um is the slip across the interface between two measurement lines, u on line L0 minus u on line "
    "L1, in micrometres (COD is a historical name; it is not an opening displacement). Full-field and raw "
    "output are CLI-only.\n"
    "Workflow: (1) dic_probe_images on each new image directory. (2) dic_plan with exactly the params you will "
    "run; continue only if valid=true and after reading every warning. (3) dic_run_event (one directory) or "
    "dic_run_campaign (subdirectories) with a NEW output_dir under the job root and timeout >= "
    "mcp.suggested_timeout_s (max_frames=20 makes a quick trial). (4) Check ok, verdict ('ok'|'warn'|'fail'), "
    "every events[].warning_details, valid_ratio >= 0.95, zncc_median >= 0.9 and mcp.brief; runs use --strict, "
    "so any failed event makes verdict 'fail' and ok false.\n"
    "Long jobs: a run tool blocks until the job ends. With wait=false in params it returns at once with "
    "job_id, output_dir, job_record and cancel_token; poll dic_query_manifest(run_dir=output_dir, "
    "documents=['.mcp-job.json']) for state and progress until state is 'exited', then read summary.json the "
    "same way. dic_job_cancel(output_dir) stops your own job. Never cancel a job you did not start: on "
    "gpu_busy wait, or use another gpu_device.\n"
    "Traps: subset is the odd subset SIZE (15 = radius 7), not the radius. cal is millimetres per pixel. "
    "line_y is [upper, lower]; reversing it flips the COD sign. COD is naive u(L0) - u(L1) unless cod_method="
    "'fillboundary' (the default only from 2026-07-09 to 2026-10-05). 16-bit TIFFs whose low 4 bits are always 0 "
    "(12-bit left-shifted) need input_depth='12' and input_align='left', never 'auto'; input_depth is a string. "
    "Rows with zncc < 0 are status rows, not measurements. A used, failed, cancelled or timed-out output_dir is "
    "never reused: pick a new one. gpu_busy means another job holds that GPU, possibly from a closed session. "
    "Campaigns: events=['event10', 'event2*'] runs only matching events (every item must match one); "
    "event_interface_y=[{'pattern': 'event2[1-5]', 'interface_y': 133}] gives matched events their own "
    "interface row (needs interface_y for the rest, not with line_y; check event_selection, events[].geometry "
    "and mcp.cod_definition.per_event_lines).\n"
    "Errors return ok=false with code and error, plus field, expected and hint when known; only a call that "
    "does not match a tool's schema (for example without the params wrapper) comes back as an MCP error text "
    "(isError). Campaign replies with more than 10 events list one compact line per event; the full summary "
    "is <output_dir>/summary.json. dic_describe returns the before-run part of the contract and lists the rest: "
    "read a part with section='mcp.results' (reply bodies), 'mcp.errors', 'mcp.jobs', 'mcp.job_record', "
    "'mcp.manifest_query' or 'mcp.known_issues', a whole section with section='mcp' or 'outputs', everything "
    "with section='all'."
)


class _SecretLogFilter(logging.Filter):
    """Keep tokens out of FastMCP's own log lines (review 2026-10-05 F1).

    When a call does not match a tool's schema FastMCP logs "Invalid arguments
    for tool ...: <pydantic errors>" to stderr, and each error carries the
    rejected input: for dic_job_cancel that is the cancel token. The input is
    redacted like the audit log's params, and whole when the error is about an
    argument whose name contains 'token'.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        with contextlib.suppress(Exception):
            if isinstance(record.args, tuple):
                record.args = tuple(self._clean(argument) for argument in record.args)
        return True

    @staticmethod
    def _clean(argument: Any) -> Any:
        if not isinstance(argument, (dict, list, tuple)):
            return argument
        cleaned = _redact_secrets(argument)
        for error in cleaned if isinstance(cleaned, list) else []:
            if isinstance(error, dict) and "input" in error and any(
                    _SECRET_KEY_PART in str(part).lower() for part in error.get("loc") or ()):
                error["input"] = _secret_digest(error["input"])
        return cleaned


if Middleware is not None:
    # The logger that prints the rejected arguments (fastmcp 3.4.4: fastmcp/server/server.py).
    logging.getLogger("fastmcp.server.server").addFilter(_SecretLogFilter())

    class ToolCallAuditMiddleware(Middleware):
        """Audit MCP-level rejections that happen before a tool reaches _dispatch.

        The arguments are logged as they came; _audit_record redacts tokens.
        """

        async def on_call_tool(self, context, call_next):
            message = context.message
            tool = str(getattr(message, "name", "unknown_tool"))
            arguments = getattr(message, "arguments", None)
            params = arguments if isinstance(arguments, dict) else {"arguments": arguments}
            if (
                tool in PARAMS_TOOLS and isinstance(arguments, dict) and arguments
                and "params" not in arguments
                and set(arguments) & (PIPELINE_PARAM_KEYS | CONTROL_PARAM_KEYS)
            ):
                call_id = uuid.uuid4().hex
                _audit_record(tool, call_id, params, "started")
                _audit_record(tool, call_id, params, "mcp_rejected:params_not_wrapped")
                raise ToolError(
                    f"{tool} takes ONE argument named params: call it as {{\"params\": {{...}}}} with "
                    f"these keys inside. Got top-level keys: {', '.join(sorted(map(str, arguments)))}"
                )
            try:
                return await call_next(context)
            except SystemExit as exc:
                call_id = uuid.uuid4().hex
                _audit_record(tool, call_id, params, "started")
                _audit_record(tool, call_id, params, "mcp_rejected:SystemExit")
                raise ToolError(
                    f"internal error: {tool} raised SystemExit({exc.code!r}); the call was aborted and the "
                    "server keeps running"
                ) from None
            except Exception as exc:
                call_id = uuid.uuid4().hex
                _audit_record(tool, call_id, params, "started")
                _audit_record(
                    tool, call_id, params,
                    f"mcp_rejected:{type(exc).__name__}",
                )
                raise
else:
    ToolCallAuditMiddleware = None  # type: ignore[assignment,misc]


def create_server():
    if FastMCP is None:
        raise RuntimeError(
            "fastmcp is not installed; install pipeline/requirements-mcp.txt in the production venv"
        )
    server = FastMCP("FastICGN", instructions=SERVER_INSTRUCTIONS, version=SERVER_VERSION)
    server.add_middleware(ToolCallAuditMiddleware())
    for function in TOOL_FUNCTIONS:
        spec = TOOL_SPECS[function.__name__]
        server.tool(
            function, name=function.__name__, title=spec["title"], description=spec["description"],
            annotations=spec["annotations"],
        )
    return server


mcp = create_server() if FastMCP is not None else None


def main() -> int:
    if mcp is None:
        print(
            "fastmcp is not installed; install pipeline/requirements-mcp.txt in the production venv",
            file=sys.stderr,
        )
        return 1
    _install_signal_handlers()
    try:
        mcp.run(transport="stdio", show_banner=False)
    finally:
        # The session ended without a signal (stdin closed): wait=false jobs
        # whose call has returned go on detached, and their records say so.
        with contextlib.suppress(Exception):
            _mark_jobs_detached("the MCP session ended (stdin closed)", only_background=True,
                                audit_extra={"signal": None})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
