#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Guardrail and fake-binary integration tests for the FastICGN MCP facade.

Hermetic: the module is imported with FASTICGN_CONFIG pointing at an empty
file and FASTICGN_JOB_ROOT at a throwaway directory (the environment is
restored right after the import), every test patches the job root, read
allowlist and binary, and the stdio servers started by the end-to-end tests
get their own HOME. Nothing reads ~/.config/fasticgn or writes ~/dic_jobs.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

_HERMETIC = tempfile.TemporaryDirectory(prefix="fasticgn-mcp-tests-")
_EMPTY_CONFIG = Path(_HERMETIC.name) / "empty-config.toml"
_EMPTY_CONFIG.write_text("", encoding="utf-8")
_IMPORT_ENV = {
    "FASTICGN_CONFIG": str(_EMPTY_CONFIG),
    "FASTICGN_JOB_ROOT": str(Path(_HERMETIC.name) / "import-job-root"),
}
_SAVED_ENV = {key: os.environ.get(key) for key in _IMPORT_ENV}
os.environ.update(_IMPORT_ENV)
try:
    import mcp_server as server
finally:
    for _key, _value in _SAVED_ENV.items():
        if _value is None:
            os.environ.pop(_key, None)
        else:
            os.environ[_key] = _value


CSV_HEADER = "x,y,u,v,zncc,uy,vy,ux,vx\n"
# dic_run_event publishes neither photo_dir nor the campaign-only keys (events, event_interface_y).
EVENT_RUN_KEYS = (server.PIPELINE_PARAM_KEYS | server.CONTROL_PARAM_KEYS) - {"photo_dir"} - server.CAMPAIGN_ONLY_KEYS

FAKE_BINARY = textwrap.dedent("""\
    #!/usr/bin/env python3
    import sys
    import time
    from pathlib import Path

    if sys.argv[1:] == ["--version"]:
        print("opencorr_cli_cuda_mvp")
        print("git_hash=mcpfake123")
        print("build_date=2099-01-02T03:04:05Z")
        raise SystemExit(0)
    args = sys.argv[1:]
    ref = str(Path(args[args.index("--ref") + 1]))
    if "sleep" in ref:
        time.sleep(60)
    if "slow" in ref:
        time.sleep(6)
    if "fail" in ref:
        print("Total POIs: 4", file=sys.stderr)
        print("Error: synthetic device failure", file=sys.stderr)
        raise SystemExit(1)
    print("Reference input: source_depth=CV_8U, source_range=[0,200], "
          "effective_depth=8, effective_align=auto, output_depth=CV_32F, "
          "output_range=[0,200], scale=1", file=sys.stderr)
    tar_list = Path(args[args.index("--tar-list") + 1])
    output_dir = Path(args[args.index("--output-dir") + 1])
    lines = [int(v) for v in args[args.index("--line") + 1].split(",")]
    # "zero" events: both lines move alike, so every COD value is 0.
    upper, lower = ("0.5,0,0.98,0", "0.5,0,0.98,0") if "zero" in ref else ("1,0,0.98,0.01", "0.5,0,0.98,-0.01")
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in tar_list.read_text(encoding="utf-8").splitlines():
        _, name = entry.rsplit(maxsplit=1)
        if "tick" in ref:  # a job whose progress can be watched: one frame every 0.25 s
            time.sleep(0.25)
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            handle.write("x,y,u,v,zncc,uy,vy,ux,vx\\n")
            for x in (8, 9):
                handle.write(f"{x},{lines[0]},{upper},0,0,0\\n")
                handle.write(f"{x},{lines[1]},{lower},0,0,0\\n")
""")


def _write_event(directory: Path, frames: int = 3, value: int = 128) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    pixels = np.full((32, 32), value, dtype=np.uint8)
    for frame in range(1, frames + 1):
        Image.fromarray(pixels).save(directory / f"{frame:06d}.png")
    return directory


def _write_frames(directory: Path, rows: dict[int, float], frames=(2, 3), columns: int = 9) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    header = ",".join(CSV_HEADER.strip().split(",")[:columns]) + "\n"
    for frame in frames:
        with (directory / f"frame_{frame:06d}.csv").open("w", encoding="utf-8") as handle:
            handle.write(header)
            for y, u in rows.items():
                values = [8, y, u, 0, 0.98, 0.01, 0, 0, 0][:columns]
                handle.write(",".join(str(value) for value in values) + "\n")
    return directory


def _lock_is_free(path: Path) -> bool:
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def _group_gone(pgid: int, wait: float = 3.0) -> bool:
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


class McpTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.jobs = self.root / "jobs"
        self.outside = self.root / "outside"
        self.data.mkdir()
        self.outside.mkdir()
        self.event = _write_event(self.data / "event with spaces")

        self.fake_bin = self.root / "fake_gpu.py"
        self.fake_bin.write_text(FAKE_BINARY, encoding="utf-8")
        self.fake_bin.chmod(0o755)

        patchers = [
            mock.patch.object(server, "JOB_ROOT", self.jobs),
            mock.patch.object(server, "READ_ROOTS", (self.data,)),
            mock.patch.object(server, "DIC_HOME", self.root / "home"),
            mock.patch.object(server, "GPU_BIN", self.fake_bin),
            mock.patch.dict(os.environ, {"FASTICGN_CONFIG": str(_EMPTY_CONFIG)}),
        ]
        self.patchers = patchers
        for patcher in patchers:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temp.cleanup()

    def base_params(self, output_name: str = "run") -> dict:
        return {
            "image_dir": str(self.event),
            "output_dir": str(self.jobs / output_name),
            "line_y": [10, 20],
            "cal": 1.0,
            "subset": 15,
        }

    def audit_records(self) -> list[dict]:
        path = self.jobs / server.AUDIT_LOG_NAME
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def wait_for_record(self, path: Path, condition, timeout: float = 15.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record = {}
            if condition(record):
                return record
            time.sleep(0.05)
        self.fail(f"job record never matched: {path}")

    def run_in_thread(self, params: dict) -> tuple[threading.Thread, dict]:
        result: dict = {}
        thread = threading.Thread(target=lambda: result.update(server.dic_run_event(params)))
        thread.start()
        return thread, result

    def server_env(self) -> dict[str, str]:
        """Environment for a real stdio server confined to this test's directories."""

        env = {key: value for key, value in os.environ.items() if not key.startswith("FASTICGN_")}
        home = self.root / "server-home"
        home.mkdir(exist_ok=True)
        env.update({
            "HOME": str(home),
            "FASTICGN_CONFIG": str(_EMPTY_CONFIG),
            "FASTICGN_JOB_ROOT": str(self.jobs),
            "FASTICGN_READ_ALLOWLIST": str(self.data),
            "FASTICGN_GPU_BIN": str(self.fake_bin),
        })
        return env


class McpGuardrailTests(McpTestBase):
    def test_describe_and_pitfalls_are_contract_backed_and_audited(self) -> None:
        described = server.dic_describe()
        pitfalls = server.dic_get_pitfalls()
        contract = json.loads(server.CONTRACT_PATH.read_text(encoding="utf-8"))

        # Without a section: an index plus the before-run part of the mcp
        # section, not the whole contract (over 120 KB, over MCP clients'
        # default reply limit) and, since 2026-10-05, not the whole mcp section.
        self.assertEqual(described["view"], "index")
        self.assertEqual([entry["section"] for entry in described["sections"]], list(contract))
        for entry in described["sections"]:
            self.assertEqual(entry["bytes"], server._json_size(contract[entry["section"]]))
            self.assertTrue(entry["summary"])
        self.assertEqual(described["included_sections"], [])
        self.assertNotIn("exit_codes", described)
        self.assertEqual(list(described["mcp"]), list(server.DESCRIBE_BEFORE_RUN_KEYS))
        self.assertEqual(described["mcp"], {key: contract["mcp"][key] for key in server.DESCRIBE_BEFORE_RUN_KEYS})
        self.assertEqual(described["mcp_view"]["view"], "before_run")
        self.assertEqual(described["mcp_view"]["included_keys"], list(server.DESCRIBE_BEFORE_RUN_KEYS))
        self.assertEqual(described["full_contract_bytes"], server._json_size(contract))
        self.assertLess(server._json_size(described), server.DESCRIBE_DEFAULT_MAX_BYTES)
        self.assertLessEqual(server.DESCRIBE_DEFAULT_MAX_BYTES, 20 * 1024)
        self.assertGreater(server._json_size(contract), 64 * 1024)
        self.assertEqual(pitfalls["pitfalls"], contract["pitfalls"])
        records = self.audit_records()
        self.assertEqual(len(records), 4)
        self.assertEqual([record["result_code"] for record in records], [
            "started", "ok", "started", "ok",
        ])

        self.assertEqual(server.dic_describe("all"), contract)
        self.assertEqual(server.dic_describe("index"), described)
        section = server.dic_describe("mcp")
        self.assertEqual(section["mcp"], contract["mcp"])
        self.assertEqual(section["section"], "mcp")
        self.assertEqual(server.dic_describe("exit_codes")["exit_codes"], contract["exit_codes"])
        # 2026-10-19: the lab-only notes are a section of their own, read by name
        # and not part of dic_get_pitfalls.
        self.assertNotIn("lab_notes", pitfalls)
        self.assertEqual(server.dic_describe("lab_notes")["lab_notes"], contract["lab_notes"])
        unknown = server.dic_describe("nope")
        self.assertEqual(unknown["code"], "invalid_parameter")
        self.assertIn("mcp", unknown["expected"])
        self.assertIn("all", unknown["expected"])
        self.assertIn("mcp.results", unknown["expected"])

    def test_describe_gives_less_by_default_and_the_rest_by_name(self) -> None:
        """2026-10-05: the default reply is the before-run part; every other key of the mcp section
        is read with section='mcp.<key>', and together they are the whole section (nothing is lost)."""

        contract = json.loads(server.CONTRACT_PATH.read_text(encoding="utf-8"))
        mcp_section = contract["mcp"]
        # Every key of the contract's mcp section is classified exactly once.
        before, on_demand = set(server.DESCRIBE_BEFORE_RUN_KEYS), set(server.MCP_ON_DEMAND_NOTES)
        self.assertEqual(before & on_demand, set())
        self.assertEqual(before | on_demand, set(mcp_section))
        for key in ("run_params", "guardrails", "tool_names", "interface_revision", "mode_limitation"):
            self.assertIn(key, before)
        for key in ("results", "errors", "job_record", "manifest_query", "known_issues", "tools"):
            self.assertIn(key, on_demand)

        described = server.dic_describe()
        listed = described["mcp_view"]["on_demand"]
        self.assertEqual([entry["section"] for entry in listed],
                         [f"mcp.{key}" for key in mcp_section if key not in before])
        rebuilt = dict(described["mcp"])
        for entry in listed:
            with self.subTest(section=entry["section"]):
                part = server.dic_describe(entry["section"])
                key = entry["section"].split(".", 1)[1]
                self.assertEqual(part["section"], entry["section"])
                self.assertEqual(list(part["mcp"]), [key])  # the same JSON path as in the whole contract
                self.assertEqual(part["mcp"][key], mcp_section[key])
                self.assertEqual(entry["bytes"], server._json_size(mcp_section[key]))
                self.assertEqual(entry["summary"], server.MCP_ON_DEMAND_NOTES[key])
                self.assertLess(server._json_size(part), 16 * 1024)
                rebuilt.update(part["mcp"])
        self.assertEqual(rebuilt, mcp_section)
        # A key of the default reply can be read alone too; anything else is refused.
        self.assertEqual(server.dic_describe("mcp.run_params")["mcp"], {"run_params": mcp_section["run_params"]})
        for bad in ("mcp.nope", "mcp.", "outputs.cod_csv", "pitfalls.0", "mcp.results.envelope", 7):
            with self.subTest(section=bad):
                refused = server.dic_describe(bad)
                self.assertEqual(refused["code"], "invalid_parameter")
                self.assertEqual(refused["field"], "section")
        # The default reply stays under its own cap (it was 32 759 B on 2026-10-05, 9 B below the old
        # test limit; 13 866 B right after this change) and says how to get the rest.
        self.assertLess(server._json_size(described), server.DESCRIBE_DEFAULT_MAX_BYTES)
        self.assertIn("section='mcp.results'", described["mcp_view"]["how"])
        self.assertIn("mcp.results", server.TOOL_SPECS["dic_describe"]["description"])
        self.assertIn("mcp.results", server.SERVER_INSTRUCTIONS)
        self.assertEqual(server.DESCRIBE_SUBSECTIONS, tuple(f"mcp.{key}" for key in mcp_section))

    def test_read_and_write_symlink_escapes_are_rejected_with_reason(self) -> None:
        escaped_read = self.data / "escaped"
        escaped_read.symlink_to(self.outside, target_is_directory=True)
        read_result = server.dic_probe_images(str(escaped_read))
        self.assertFalse(read_result["ok"])
        self.assertEqual(read_result["code"], "read_path_denied")
        self.assertIn("outside read allowlist", read_result["error"])

        outside_image = self.outside / "000004.png"
        Image.fromarray(np.full((32, 32), 1, dtype=np.uint8)).save(outside_image)
        linked_image = self.event / "000004.png"
        linked_image.symlink_to(outside_image)
        probe_descendant = server.dic_probe_images(str(self.event))
        self.assertFalse(probe_descendant["ok"])
        self.assertEqual(probe_descendant["code"], "read_symlink_denied")
        descendant = server.dic_plan(self.base_params("descendant-symlink"))
        self.assertFalse(descendant["ok"])
        self.assertEqual(descendant["code"], "read_symlink_denied")
        linked_image.unlink()

        self.jobs.mkdir(exist_ok=True)
        escaped_write = self.jobs / "escaped"
        escaped_write.symlink_to(self.outside, target_is_directory=True)
        params = self.base_params()
        params["output_dir"] = str(escaped_write / "run")
        write_result = server.dic_plan(params)
        self.assertFalse(write_result["ok"])
        self.assertEqual(write_result["code"], "write_path_denied")

        reserved = self.base_params("unused")
        reserved["output_dir"] = str(self.jobs / server.CONTROL_DIR_NAME / "run")
        reserved_result = server.dic_plan(reserved)
        self.assertEqual(reserved_result["code"], "reserved_write_path")

    def test_legacy_dic_home_is_only_honored_when_that_directory_exists(self) -> None:
        """DIC_HOME=None 时不得凭空放行 <lab home>/DIC*，报错也不该提它。"""
        legacy_home = self.root / "home"
        legacy_dic = legacy_home / "DIC-legacy"
        legacy_dic.mkdir(parents=True)
        (legacy_dic / "000002.png").write_bytes(b"")

        with mock.patch.object(server, "DIC_HOME", legacy_home):
            allowed = server.dic_probe_images(str(legacy_dic))
        self.assertNotEqual(allowed.get("code"), "read_path_denied")

        with mock.patch.object(server, "DIC_HOME", None):
            denied = server.dic_probe_images(str(legacy_dic))
        self.assertFalse(denied["ok"])
        self.assertEqual(denied["code"], "read_path_denied")
        self.assertIn("outside read allowlist", denied["error"])
        self.assertNotIn("/DIC*", denied["error"])
        self.assertIn(str(self.data), denied["error"])

    def test_bounds_unknown_keys_and_noninteger_pairs_are_rejected(self) -> None:
        cases = [
            ({"subset": 14}, "parameter_out_of_bounds"),
            ({"step": 21}, "parameter_out_of_bounds"),
            ({"cal": 10.1}, "parameter_out_of_bounds"),
            ({"gauge": 3}, "parameter_out_of_bounds"),
            ({"timeout": 3600.1}, "parameter_out_of_bounds"),
            ({"gpu_bin": "/bin/true"}, "unknown_params"),
            ({"line_y": [10.5, 20]}, "invalid_parameter"),
        ]
        for index, (change, expected_code) in enumerate(cases):
            with self.subTest(change=change):
                params = self.base_params(f"bounds-{index}")
                params.update(change)
                result = server.dic_plan(params)
                self.assertFalse(result["ok"])
                self.assertEqual(result["code"], expected_code)
                self.assertIn("field", result)
                self.assertFalse(result["retryable"])

    def test_negative_coordinates_return_structured_errors(self) -> None:
        """Review 2026-10-03 S1-5: these used to raise SystemExit inside argparse."""

        cases = [
            ({"line_y": [-5, 20]}, "line_y", "invalid_parameter"),
            ({"line_y": "-5,20"}, "line_y", "invalid_parameter"),
            ({"line_y": [10, 10]}, "line_y", "invalid_parameter"),
            ({"roi_cols": [-5, 20]}, "roi_cols", "invalid_parameter"),
            ({"roi_cols": [20, 5]}, "roi_cols", "invalid_parameter"),
            ({"line_y": None, "interface_y": -5}, "interface_y", "parameter_out_of_bounds"),
        ]
        for index, (change, field, code) in enumerate(cases):
            for tool in (server.dic_plan, server.dic_run_event):
                with self.subTest(change=change, tool=tool.__name__):
                    params = self.base_params(f"negative-{index}-{tool.__name__}")
                    params.update(change)
                    result = tool(params)
                    self.assertFalse(result["ok"])
                    self.assertEqual(result["code"], code)
                    self.assertEqual(result["field"], field)
                    self.assertFalse((self.jobs / f"negative-{index}-{tool.__name__}").exists())
        self.assertEqual(
            [record["result_code"] for record in self.audit_records()].count("started"),
            len(cases) * 2,
        )

    def test_pipeline_argv_attaches_every_value(self) -> None:
        params = self.base_params("argv")
        params.update({"keep_frames": True, "max_frames": 5})
        normalized = server._normalize_pipeline_params(params)
        argv = server._pipeline_argv(normalized)
        flags = [token for token in argv if token.startswith("--")]
        self.assertEqual(flags, argv)
        self.assertIn("--line-y=10,20", argv)
        self.assertIn("--max-frames=5", argv)
        # 2026-10-05: runs default to naive COD; fillboundary stays selectable.
        self.assertIn("--cod-method=naive", argv)
        self.assertIn("--cod-method=fillboundary", server._pipeline_argv(server._normalize_pipeline_params(
            {**self.base_params("argv-fb"), "cod_method": "fillboundary"})))
        self.assertEqual(argv[-1], "--strict")
        self.assertTrue(all("=" in token for token in argv if token not in {"--keep-frames", "--strict"}))

        parser = server.pipeline.create_parser()
        attached = vars(parser.parse_args(argv + ["--dry-run"]))
        separate = vars(parser.parse_args(
            [part for token in argv for part in token.split("=", 1)] + ["--dry-run"]
        ))
        self.assertEqual(attached, separate)

    def test_system_exit_inside_a_tool_is_contained(self) -> None:
        def exits() -> dict:
            raise SystemExit(2)

        result = server._dispatch("dic_plan", {"probe": True}, exits)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "internal_error")
        self.assertEqual(self.audit_records()[-1]["result_code"], "internal_error:SystemExit")

    def test_errors_are_actionable(self) -> None:
        unknown = self.base_params("unknown")
        unknown["subset_size"] = 15
        result = server.dic_plan(unknown)
        self.assertEqual(result["code"], "unknown_params")
        self.assertTrue(result["error"].startswith("unsupported parameter(s): subset_size"))
        self.assertEqual(result["suggestions"], {"subset_size": "subset"})
        self.assertEqual(set(result["expected"]), server.PIPELINE_PARAM_KEYS | server.CONTROL_PARAM_KEYS)
        self.assertNotIn("gpu_bin/json_summary", result["error"])

        for key, value in (("output_format", "raw"), ("image_pattern", "img_(\\d+)")):
            params = self.base_params(key)
            params[key] = value
            cli_only = server.dic_plan(params)
            self.assertEqual(cli_only["code"], "unknown_params")
            self.assertIn("COD mode only", cli_only["error"])

        full_field = self.base_params("full-field")
        full_field.pop("line_y")
        missing = server.dic_plan(full_field)
        self.assertEqual(missing["code"], "missing_parameter")
        self.assertIn("COD mode only", missing["error"])

        depth = self.base_params("depth")
        depth["input_depth"] = 12
        depth_result = server.dic_plan(depth)
        self.assertEqual(depth_result["code"], "invalid_parameter")
        self.assertIn('send "12"', depth_result["error"])

        cal = self.base_params("cal-string")
        cal["cal"] = "0.5"
        self.assertIn("JSON number", server.dic_plan(cal)["error"])

        bad_names = self.data / "bad names"
        bad_names.mkdir()
        Image.fromarray(np.full((32, 32), 1, dtype=np.uint8)).save(bad_names / "img_1.png")
        no_images = server.dic_probe_images(str(bad_names))
        self.assertEqual(no_images["code"], "no_images")
        self.assertIn("6-digit frame number", no_images["error"])
        self.assertEqual(no_images["sample_names"], ["img_1.png"])

        single = _write_event(self.data / "single", frames=1)
        probe = server.dic_probe_images(str(single))
        self.assertTrue(probe["ok"])
        self.assertFalse(probe["runnable"])

    def test_gpu_device_is_checked_against_visible_devices(self) -> None:
        params = self.base_params("gpu-count")
        params["gpu_device"] = 2
        with mock.patch.object(server, "GPU_DEVICE_COUNT", 2):
            rejected = server.dic_plan(params)
        self.assertEqual(rejected["code"], "parameter_out_of_bounds")
        self.assertEqual(rejected["field"], "gpu_device")
        with mock.patch.object(server, "GPU_DEVICE_COUNT", None):
            planned = server.dic_plan(params)
        self.assertTrue(planned["valid"])
        self.assertIn("gpu_device_unverified", [warning["code"] for warning in planned["warnings"]])

    def test_control_characters_are_rejected_in_user_and_discovered_paths(self) -> None:
        tab_event = self.data / "tab\tevent"
        tab_event.mkdir()
        pixels = np.full((32, 32), 1, dtype=np.uint8)
        for frame in (1, 2):
            Image.fromarray(pixels).save(tab_event / f"{frame:06d}.png")
        params = self.base_params("tab-event")
        params["image_dir"] = str(tab_event)
        direct = server.dic_plan(params)
        self.assertEqual(direct["code"], "unsafe_path_characters")

        newline_image = self.event / "000004.png\n"
        Image.fromarray(pixels).save(newline_image, format="PNG")
        discovered = server.dic_probe_images(str(self.event))
        self.assertEqual(discovered["code"], "unsafe_path_characters")
        newline_image.unlink()

        campaign = self.data / "campaign"
        injected_event = campaign / "event\tname"
        injected_event.mkdir(parents=True)
        for frame in (1, 2):
            Image.fromarray(pixels).save(injected_event / f"{frame:06d}.png")
        campaign_params = {
            "photo_dir": str(campaign),
            "output_dir": str(self.jobs / "campaign-control"),
            "line_y": [10, 20],
            "cal": 1.0,
        }
        descendant = server.dic_plan(campaign_params)
        self.assertEqual(descendant["code"], "unsafe_path_characters")

    def test_plan_is_gpu_free_and_does_not_create_output(self) -> None:
        output = self.jobs / "plan-only"
        result = server.dic_plan(self.base_params("plan-only"))

        self.assertTrue(result["valid"])
        self.assertTrue(result["ok"])
        self.assertFalse(result["gpu_started"])
        self.assertFalse(result["filesystem_writes"])
        self.assertFalse(output.exists())
        self.assertEqual(result["events"][0]["target_frames"], 2)
        command = result["events"][0]["binary_command"]
        self.assertEqual(command[0], str(self.fake_bin))
        self.assertEqual(command[-2:], ["--conv-threshold", "0.001"])
        self.assertTrue(result["mcp"]["audit_log_write_excluded_from_pipeline_filesystem_writes"])
        self.assertEqual(result["mcp"]["cod_definition"]["L0_y"], 10)
        self.assertGreaterEqual(result["mcp"]["suggested_timeout_s"], 60)

        descending = self.base_params("descending")
        descending["line_y"] = [20, 10]
        warned = server.dic_plan(descending)
        self.assertTrue(warned["valid"])
        self.assertIn("line_y_descending", [warning["code"] for warning in warned["warnings"]])

    def test_invalid_plan_has_invalid_audit_result_code(self) -> None:
        params = self.base_params("invalid-plan")
        params.pop("line_y")
        params["interface_y"] = 999

        result = server.dic_plan(params)

        self.assertFalse(result["valid"])
        self.assertFalse(result["ok"])
        self.assertEqual(self.audit_records()[-1]["result_code"], "invalid_plan")

        run = server.dic_run_event(params)
        self.assertEqual(run["code"], "plan_invalid")
        self.assertEqual(json.loads(run["error"]), run["errors"])
        self.assertIn("interface_y_out_of_bounds", [item["code"] for item in run["errors"]])

    def test_existing_output_requires_explicit_empty_directory_override(self) -> None:
        output = self.jobs / "existing"
        output.mkdir(parents=True)
        denied = server.dic_run_event(self.base_params("existing"))
        self.assertEqual(denied["code"], "output_exists")

        (output / "old.csv").write_text("stale", encoding="utf-8")
        params = self.base_params("existing")
        params["overwrite"] = True
        nonempty = server.dic_run_event(params)
        self.assertEqual(nonempty["code"], "nonempty_output_rejected")

    def test_same_output_is_serialized_before_state_check(self) -> None:
        params = self.base_params("locked-output")
        output = Path(params["output_dir"]).resolve(strict=False)

        with server._output_lock(output):
            result = server.dic_run_event(params)

        self.assertEqual(result["code"], "output_busy")
        self.assertTrue(result["retryable"])

    def test_gpu_busy_names_the_holder_in_process_and_across_processes(self) -> None:
        holder = {"tool": "dic_run_event", "call_id": "x" * 32, "output_dir": "/elsewhere"}
        with server._gpu_lock(0, holder):
            busy = server.dic_run_event(self.base_params("gpu-busy-inprocess"))
        self.assertEqual(busy["code"], "gpu_busy")
        self.assertTrue(busy["retryable"])
        self.assertEqual(busy["holder"]["output_dir"], "/elsewhere")
        self.assertFalse((self.jobs / "gpu-busy-inprocess").exists())

        lock_path = server._gpu_lock_path(0)
        script = textwrap.dedent(f"""\
            import fcntl, json, os, sys, time
            fd = os.open({str(lock_path)!r}, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.ftruncate(fd, 0)
            os.write(fd, json.dumps({{"server_pid": os.getpid(), "output_dir": "/other-process",
                                      "acquired_epoch": time.time()}}).encode())
            print("locked", flush=True)
            time.sleep(60)
        """)
        holder_process = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(holder_process.stdout.readline().strip(), "locked")
            busy = server.dic_run_event(self.base_params("gpu-busy-crossprocess"))
            self.assertEqual(busy["code"], "gpu_busy")
            self.assertEqual(busy["holder"]["output_dir"], "/other-process")
            self.assertTrue(busy["holder"]["server_alive"])
        finally:
            holder_process.terminate()
            holder_process.wait(timeout=10)
            holder_process.stdout.close()
        # The flock died with its holder: no stale lock.
        self.assertTrue(_lock_is_free(lock_path))
        after = server.dic_run_event(self.base_params("gpu-after-holder"))
        self.assertEqual(after["verdict"], "ok")

    def test_event_run_returns_m3_summary_and_matches_direct_cli(self) -> None:
        params = self.base_params("mcp-run")
        summary = server.dic_run_event(params)
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["verdict"], "ok")
        self.assertTrue(summary["ok"])
        self.assertEqual(len(summary["call_id"]), 32)
        self.assertEqual(summary["binary"]["git_hash"], "mcpfake123")
        self.assertEqual(summary["events"][0]["frames"], 2)
        self.assertEqual(summary["mcp"]["job"]["state"], "exited")
        # keep_frames=false deletes the frame directory: never hand it out as usable.
        # summary.json keeps the pipeline's (638ad18) value with field_dir_exists
        # false; the MCP result nulls the path and says so.
        self.assertIsNone(summary["events"][0]["field_dir"])
        self.assertFalse(summary["events"][0]["field_dir_exists"])
        self.assertEqual(summary["mcp"]["nulled_paths"][0]["field"], "field_dir")
        # Runs are --strict; the brief reads geometry, cod_stats and severities.
        self.assertIs(summary["params"]["strict"], True)
        self.assertIn("lines y 10/20, COD 4 rows, peak 500 um", summary["mcp"]["brief"])

        output = self.jobs / "mcp-run"
        record = json.loads((output / server.JOB_RECORD_NAME).read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "exited")
        self.assertEqual(record["exit_code"], 0)
        self.assertEqual(record["pipeline_argv"][-2], "--strict")
        self.assertTrue(record["summary_present"])
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

        direct = self.root / "direct-run"
        command = [
            sys.executable, str(server.PIPELINE_PATH),
            "--image-dir", str(self.event),
            "--output-dir", str(direct),
            "--line-y", "10,20", "--cal", "1.0", "--subset", "15",
            "--step", "1", "--gauge", "35", "--cod-method", "naive",
            "--conv-threshold", "0.001", "--gpu-device", "0",
            "--gpu-bin", str(self.fake_bin),
            "--json-summary", str(direct / "summary.json"),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            (output / "cod_result.csv").read_bytes(),
            (direct / "cod_result.csv").read_bytes(),
        )

    def test_dangling_summary_paths_are_nulled(self) -> None:
        """Defence in depth: paths a summary names that are not on disk (an older
        pipeline, or files removed after the run) are never handed out."""

        output = self.jobs / "dangling"
        output.mkdir(parents=True)
        summary = {"verdict": "ok", "global": {"events_ok": 1}, "events": [{
            "name": "e", "status": "ok", "frames": 2, "warnings": [],
            "cod_csv": str(output / "cod_result.csv"), "field_dir": str(output / "_frames_e"),
        }]}
        result = server._finalize_run_result(
            summary, output, output / server.JOB_RECORD_NAME, {"state": "exited", "exit_code": 0},
            {"keep_frames": False},
        )
        self.assertEqual([entry["field"] for entry in result["mcp"]["nulled_paths"]], ["cod_csv", "field_dir"])
        self.assertIsNone(result["events"][0]["cod_csv"])
        self.assertIsNone(result["events"][0]["field_dir"])
        self.assertFalse(result["ok"])

    def test_run_brief_accepts_string_and_structured_warnings(self) -> None:
        summary = {"verdict": "warn", "global": {"events_ok": 1}, "events": [{
            "name": "e", "status": "ok", "frames": 2, "valid_ratio": 1.0, "zncc_median": 0.99,
            "warnings": ["input_8bit_max_255: possible saturation", {"code": "valid_ratio_low", "severity": "warn"}],
        }]}
        brief = server._event_brief(summary)
        self.assertIn("warnings input_8bit_max_255,valid_ratio_low", brief)

    def test_binary_failure_reports_its_error_line(self) -> None:
        failing = _write_event(self.data / "fail-event", frames=2)
        params = self.base_params("binary-failure")
        params["image_dir"] = str(failing)
        result = server.dic_run_event(params)
        self.assertEqual(result["verdict"], "fail")
        self.assertFalse(result["ok"])
        self.assertNotIn("code", result)  # a summary, not an error envelope
        self.assertEqual(result["mcp"]["job"]["exit_code"], server.PIPELINE_EXIT_EVENT_FAILED)
        self.assertIn("Error: synthetic device failure", result["mcp"]["error_lines"])
        self.assertIn("binary_stderr_log", result["mcp"]["hint"])
        self.assertTrue(Path(result["events"][0]["binary_stderr_log"]).is_file())
        self.assertIn("gpu_cli_failed(fail)", result["mcp"]["brief"])
        self.assertIsNone(result["events"][0]["cod_csv"])
        self.assertEqual(self.audit_records()[-1]["result_code"], "fail")

    def test_strict_counts_an_all_zero_cod_as_a_failed_event(self) -> None:
        """--strict is server behaviour: the same input passes the CLI's default mode with a warning."""

        zero = _write_event(self.data / "zero-event", frames=2)
        params = self.base_params("all-zero")
        params["image_dir"] = str(zero)
        result = server.dic_run_event(params)
        self.assertEqual(result["verdict"], "fail")
        self.assertFalse(result["ok"])
        self.assertEqual(result["events"][0]["status"], "fail")
        self.assertIn({"code": "cod_all_zero", "severity": "fail"},
                      [{key: item[key] for key in ("code", "severity")}
                       for item in result["events"][0]["warning_details"]])
        self.assertEqual(result["mcp"]["job"]["exit_code"], server.PIPELINE_EXIT_EVENT_FAILED)

        direct = self.root / "zero-direct"
        completed = subprocess.run(
            [sys.executable, str(server.PIPELINE_PATH), "--image-dir", str(zero), "--output-dir", str(direct),
             "--line-y", "10,20", "--cal", "1.0", "--gpu-bin", str(self.fake_bin),
             "--json-summary", str(direct / "summary.json")],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads((direct / "summary.json").read_text(encoding="utf-8"))["verdict"], "warn")

    def test_max_frames_reaches_plan_and_run(self) -> None:
        params = self.base_params("max-frames")
        params["max_frames"] = 1
        plan = server.dic_plan(params)
        self.assertTrue(plan["valid"])
        self.assertEqual(plan["events"][0]["target_frames"], 1)
        self.assertEqual(plan["events"][0]["target_frames_available"], 2)
        self.assertEqual(plan["estimates"]["target_frames"], 1)
        self.assertIn(("max_frames_applied", "info"),
                      [(item["code"], item.get("severity")) for item in plan["notes"]])

        run = server.dic_run_event(params)
        self.assertEqual(run["verdict"], "ok")
        self.assertTrue(run["ok"])
        self.assertEqual(run["params"]["max_frames"], 1)
        self.assertEqual(run["events"][0]["frames"], 1)
        self.assertEqual(run["events"][0]["target_frames_selected"], 1)
        self.assertEqual(run["events"][0]["target_frames_available"], 2)
        self.assertIn("max_frames_applied(info)", run["mcp"]["brief"])

        for index, (value, code) in enumerate(((0, "parameter_out_of_bounds"), (-3, "parameter_out_of_bounds"),
                                               (1.5, "invalid_parameter"), ("2", "invalid_parameter"),
                                               (True, "invalid_parameter"))):
            with self.subTest(max_frames=value):
                bad = self.base_params(f"max-frames-bad-{index}")
                bad["max_frames"] = value
                for tool in (server.dic_plan, server.dic_run_event):
                    rejected = tool(bad)
                    self.assertEqual(rejected["code"], code)
                    self.assertEqual(rejected["field"], "max_frames")
                self.assertFalse((self.jobs / f"max-frames-bad-{index}").exists())

    def test_strict_refusal_after_a_passing_plan_is_plan_invalid(self) -> None:
        """Exit 2: the pipeline's own --strict check refuses what the MCP plan check let through
        (a race: inputs changed between the two). Simulated with a plan that reports valid."""

        params = self.base_params("strict-refused")
        params["line_y"] = [10, 200]  # outside the 32-px image
        real_plan = server._build_plan(server._normalize_pipeline_params(params))
        self.assertFalse(real_plan["valid"])
        real_build_plan = server._build_plan

        def stale_plan(normalized):
            plan = real_build_plan(normalized)
            plan.update(valid=True, ok=True, errors=[])
            return plan

        with mock.patch.object(server, "_build_plan", stale_plan):
            result = server.dic_run_event(params)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "plan_invalid")
        self.assertFalse(result["retryable"])
        self.assertEqual(json.loads(result["error"]), result["errors"])
        self.assertEqual(result["errors"], real_plan["errors"])
        self.assertEqual(result["warnings"], [])
        self.assertIn("NEW output_dir", result["hint"])
        output = self.jobs / "strict-refused"
        self.assertEqual(Path(result["output_dir"]), output.resolve())
        self.assertEqual(Path(result["job_record"]), output.resolve() / server.JOB_RECORD_NAME)
        record = json.loads((output / server.JOB_RECORD_NAME).read_text(encoding="utf-8"))
        self.assertEqual(record["exit_code"], server.PIPELINE_EXIT_STRICT_REFUSED)
        self.assertFalse((output / "cod_result.csv").exists())
        self.assertFalse(any(output.glob("_frames_*")))
        self.assertEqual(self.audit_records()[-1]["result_code"], "plan_invalid")
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

    def test_strict_refusal_errors_parse_the_failure_summary(self) -> None:
        summary = {"warnings": [
            "pipeline_error: --strict: dry-run validation failed, nothing was run",
            "pipeline_error: line_y_out_of_bounds: L1=200 is outside image height 32 [event e [1]]",
            "pipeline_error: gpu_binary_missing: GPU binary does not exist: /x: y",
        ]}
        self.assertEqual(server._strict_refusal_errors(summary), [
            {"code": "line_y_out_of_bounds", "message": "L1=200 is outside image height 32", "event": "e [1]"},
            {"code": "gpu_binary_missing", "message": "GPU binary does not exist: /x: y"},
        ])
        fallback = server._strict_refusal_errors({"warnings": ["pipeline_error: something else"]})
        self.assertEqual(fallback[0]["code"], "strict_check_failed")

    def test_probe_aggregate_and_query_manifest(self) -> None:
        probe = server.dic_probe_images(str(self.event))
        self.assertTrue(probe["ok"])
        self.assertEqual(probe["images"], 3)
        self.assertEqual(probe["input_recommendation"]["input_depth"], "8")

        frames = _write_frames(self.data / "retained", {10: 1, 20: 0.5})

        escaped_frame = frames / "frame_000004.csv"
        escaped_frame.symlink_to(self.outside / "outside.csv")
        (self.outside / "outside.csv").write_text(CSV_HEADER, encoding="utf-8")
        denied_frame = server.dic_aggregate_cod(
            str(frames), str(self.jobs / "aggregate" / "denied.csv"), [10, 20], 1.0,
        )
        self.assertEqual(denied_frame["code"], "read_symlink_denied")
        escaped_frame.unlink()

        reserved_output = server.dic_aggregate_cod(
            str(frames), str(self.jobs / server.AUDIT_LOG_NAME), [10, 20], 1.0,
            overwrite=True,
        )
        self.assertEqual(reserved_output["code"], "reserved_write_path")
        aggregate_path = self.jobs / "aggregate" / "cod.csv"
        aggregate = server.dic_aggregate_cod(
            str(frames), str(aggregate_path), [10, 20], 1.0,
        )
        self.assertTrue(aggregate["ok"])
        self.assertEqual(aggregate["frames"], 2)
        self.assertEqual(aggregate["rows"], 2)
        self.assertEqual(aggregate_path.read_text().splitlines()[0], "frame,x_mm,cod_um")

        manifest = self.jobs / "aggregate" / "summary.json"
        manifest.write_text(json.dumps({"verdict": "ok"}), encoding="utf-8")
        queried = server.dic_query_manifest(str(manifest.parent))
        self.assertTrue(queried["ok"])
        self.assertEqual(queried["documents"]["summary.json"]["verdict"], "ok")
        self.assertTrue(queried["complete"])

    def test_aggregation_reports_missing_rows_and_protects_pipeline_products(self) -> None:
        frames = _write_frames(self.data / "rows", {10: 1, 20: 0.5})
        missing = server.dic_aggregate_cod(str(frames), str(self.jobs / "agg" / "none.csv"), [11, 21], 1.0)
        self.assertFalse(missing["ok"])
        self.assertEqual(missing["code"], "no_cod_rows")
        self.assertEqual(missing["available_line_ys"], [10, 20])
        self.assertFalse((self.jobs / "agg" / "none.csv").exists())

        flipped = server.dic_aggregate_cod(str(frames), str(self.jobs / "agg" / "flipped.csv"), [20, 10], 1.0)
        self.assertTrue(flipped["ok"])
        self.assertEqual([warning["code"] for warning in flipped["warnings"]], ["line_y_descending"])

        run = self.jobs / "run-dir"
        run.mkdir(parents=True)
        (run / "summary.json").write_text('{"verdict": "ok"}', encoding="utf-8")
        (run / "cod_result.csv").write_text("frame,x_mm,cod_um\n", encoding="utf-8")
        (run / "kept.csv").write_text("frame,x_mm,cod_um\n", encoding="utf-8")
        run_frames = _write_frames(run / "_frames_event", {10: 1, 20: 0.5})
        rejected = {
            "summary.json": (run / "summary.json", True),
            "cod_result.csv": (run / "cod_result.csv", True),
            "existing file in run dir": (run / "kept.csv", True),
            "frames dir": (run_frames / "frame_000999.csv", False),
            "frames dir other name": (run_frames / "extra.csv", False),
        }
        for label, (target, overwrite) in rejected.items():
            with self.subTest(target=label):
                result = server.dic_aggregate_cod(str(frames), str(target), [10, 20], 1.0, overwrite=overwrite)
                self.assertFalse(result["ok"])
                self.assertEqual(result["code"], "aggregate_output_rejected")
        self.assertEqual((run / "summary.json").read_text(encoding="utf-8"), '{"verdict": "ok"}')

        new_in_run = server.dic_aggregate_cod(str(run_frames), str(run / "cod_naive.csv"), [10, 20], 1.0,
                                              method="naive")
        self.assertTrue(new_in_run["ok"])
        with server._output_lock(run.resolve()):
            busy = server.dic_aggregate_cod(str(run_frames), str(run / "cod_other.csv"), [10, 20], 1.0)
        self.assertEqual(busy["code"], "output_busy")

        five = _write_frames(self.data / "five", {10: 1, 20: 0.5}, columns=5)
        legacy = server.dic_aggregate_cod(str(five), str(self.jobs / "agg" / "five.csv"), [10, 20], 1.0,
                                          method="fillboundary")
        self.assertEqual(legacy["code"], "cod_uy_column_missing")
        self.assertIn("method='naive'", legacy["hint"])
        # 2026-10-05: the default method is naive, which needs no uy column.
        legacy_default = server.dic_aggregate_cod(str(five), str(self.jobs / "agg" / "five-naive.csv"),
                                                  [10, 20], 1.0)
        self.assertTrue(legacy_default["ok"], legacy_default)
        self.assertEqual(legacy_default["method"], "naive")
        odd = _write_frames(self.data / "odd", {10: 1, 20: 0.5})
        (odd / "frame_abc.csv").write_text(CSV_HEADER, encoding="utf-8")
        named = server.dic_aggregate_cod(str(odd), str(self.jobs / "agg" / "odd.csv"), [10, 20], 1.0)
        self.assertEqual(named["code"], "invalid_frame_name")
        string_cal = server.dic_aggregate_cod(str(frames), str(self.jobs / "agg" / "s.csv"), [10, 20], "1.0")
        self.assertEqual(string_cal["code"], "invalid_parameter")

    def test_manifest_query_is_bounded(self) -> None:
        run = self.jobs / "big-run"
        run.mkdir(parents=True)
        frames = [{"output_name": f"frame_{index:06d}.bin", "rows": 171024, "valid_ratio": 1.0 - index * 1e-6,
                   "fallback_used": index % 100 == 0, "tar_path": "/data/x" * 10} for index in range(4000)]
        (run / "manifest.opencorr-raw.json").write_text(
            json.dumps({"format": "opencorr-gpu-raw-v1", "frames": frames}), encoding="utf-8")
        (run / "summary.json").write_text(json.dumps({"verdict": "ok", "events": []}), encoding="utf-8")
        self.assertGreater((run / "manifest.opencorr-raw.json").stat().st_size, 500_000)

        default = server.dic_query_manifest(str(run))
        size = len(json.dumps(default, ensure_ascii=False, separators=(",", ":")).encode())
        self.assertLessEqual(size, server.DEFAULT_QUERY_MAX_BYTES)
        condensed = default["documents"]["manifest.opencorr-raw.json"]["frames"]["_condensed_list"]
        self.assertEqual(condensed["count"], 4000)
        self.assertEqual(condensed["boolean"]["fallback_used"]["true"], 40)
        self.assertEqual(default["document_views"]["manifest.opencorr-raw.json"], "condensed")
        self.assertEqual(default["documents"]["summary.json"], {"verdict": "ok", "events": []})
        self.assertFalse(default["complete"])

        sliced = server.dic_query_manifest(str(run), fields=["frames[1:3]", "nope"],
                                           documents=["manifest.opencorr-raw.json"])
        self.assertEqual(
            [frame["output_name"] for frame in sliced["documents"]["manifest.opencorr-raw.json"]["frames[1:3]"]],
            ["frame_000001.bin", "frame_000002.bin"],
        )
        self.assertIn("manifest.opencorr-raw.json", sliced["missing_fields"])

        full = server.dic_query_manifest(str(run), detail="full", max_bytes=4096)
        self.assertLessEqual(len(json.dumps(full, separators=(",", ":")).encode()), 4096)
        self.assertEqual([entry["document"] for entry in full["omitted"]], ["manifest.opencorr-raw.json"])
        self.assertIn("frames", full["omitted"][0]["top_level_keys"])

        too_big = server.dic_query_manifest(str(run), max_bytes=10_000_000)
        self.assertEqual(too_big["code"], "parameter_out_of_bounds")

    def test_job_root_permissions_are_set_only_on_creation(self) -> None:
        self.jobs.mkdir(mode=0o750)
        os.chmod(self.jobs, 0o750)
        server.dic_get_pitfalls()
        self.assertEqual(self.jobs.stat().st_mode & 0o777, 0o750)

        fresh = self.root / "fresh-root"
        with mock.patch.object(server, "JOB_ROOT", fresh):
            server.dic_get_pitfalls()
        self.assertEqual(fresh.stat().st_mode & 0o777, 0o700)

    def test_audit_truncates_oversized_params(self) -> None:
        params = self.base_params("huge")
        params["output_dir"] = "/" + "x" * 100_000
        server.dic_plan(params)
        record = self.audit_records()[-1]
        self.assertTrue(record["params"]["_truncated"])
        self.assertLess(len(json.dumps(record)), server.MAX_AUDIT_PARAM_BYTES)

    def test_the_audit_writer_itself_keeps_tokens_out(self) -> None:
        """Review 2026-10-05 F1: every audit line goes through _audit_record (tool calls, the middleware,
        the signal handler, job threads), so that is where a token becomes a hash prefix."""

        token = "7a11ad0be5" * 5
        digest = "sha256:" + hashlib.sha256(token.encode()).hexdigest()[:12]
        params = {"cancel_token": token, "output_dir": "/x",
                  "nested": {"X_Token": 5, "list": [{"TOKEN": [token]}, {"plain": "kept"}]}}
        server._audit_record("dic_job_cancel", "c" * 32, params, "started")
        text = (self.jobs / server.AUDIT_LOG_NAME).read_text(encoding="utf-8")
        self.assertNotIn(token, text)
        self.assertEqual(self.audit_records()[-1]["params"], {
            "cancel_token": digest, "output_dir": "/x",
            "nested": {"X_Token": "<integer>", "list": [{"TOKEN": "<array>"}, {"plain": "kept"}]}})
        self.assertEqual(params["cancel_token"], token)  # the caller's own dict is not changed
        # Oversized params: the preview (the first 1024 bytes, keys sorted) is cut after the token is gone.
        server._audit_record("dic_job_cancel", "d" * 32, {"cancel_token": token, "zz_blob": "x" * 20_000}, "started")
        record = self.audit_records()[-1]
        self.assertTrue(record["params"]["_truncated"])
        self.assertNotIn(token, (self.jobs / server.AUDIT_LOG_NAME).read_text(encoding="utf-8"))
        self.assertIn(digest, record["params"]["preview"])

    # ── review 2026-10-04 (rv-mcp) ──

    def make_campaign(self, name: str, events: int, failing: tuple[int, ...] = ()) -> Path:
        campaign = self.data / name
        for index in range(events):
            suffix = "-fail" if index in failing else ""
            _write_event(campaign / f"e{index:02d}{suffix}", frames=2)
        return campaign

    def campaign_params(self, campaign: Path, output_name: str) -> dict:
        return {"photo_dir": str(campaign), "output_dir": str(self.jobs / output_name),
                "line_y": [10, 20], "cal": 1.0, "subset": 15}

    def test_large_campaigns_reply_with_compact_events_within_the_cap(self) -> None:
        """F1/F3: ok/verdict/brief first, one compact line per event, failed events named."""
        campaign = self.make_campaign("big", 12, failing=(5,))
        plan = server.dic_plan(self.campaign_params(campaign, "big-run"))
        self.assertEqual(list(plan)[:3], ["ok", "valid", "exit_code"])
        self.assertEqual(plan["mcp"]["events_view"]["view"], "compact")
        self.assertEqual(len(plan["events"]), 12)
        self.assertNotIn("binary_command", plan["events"][0])
        self.assertLessEqual(server._json_size(plan), server.MAX_REPLY_BYTES)

        run = server.dic_run_campaign(self.campaign_params(campaign, "big-run"))
        self.assertEqual(list(run)[:3], ["ok", "verdict", "mcp"])
        self.assertEqual(list(run["mcp"])[0], "brief")
        self.assertFalse(run["ok"])
        self.assertEqual(run["verdict"], "fail")
        self.assertEqual(run["mcp"]["events_view"]["failed"], ["e05-fail"])
        self.assertIn("failed: e05-fail", run["mcp"]["brief"])
        self.assertTrue(run["mcp"]["brief"].index("e05-fail: fail") < run["mcp"]["brief"].index("e00: ok"))
        self.assertEqual(len(run["events"]), 12)
        allowed = {"name", "status", "frames", "valid_ratio", "zncc_median", "cod_rows", "cod_peak_um", "warnings"}
        self.assertTrue(all(set(event) <= allowed for event in run["events"]))
        self.assertIn("gpu_cli_failed(fail)", run["events"][5]["warnings"])
        self.assertLessEqual(server._json_size(run), server.MAX_REPLY_BYTES)
        full = json.loads((self.jobs / "big-run" / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("warning_details", full["events"][0])  # the full summary stays on disk

        small = self.make_campaign("small", 2)
        small_run = server.dic_run_campaign(self.campaign_params(small, "small-run"))
        self.assertNotIn("events_view", small_run["mcp"])
        self.assertIn("warning_details", small_run["events"][0])  # small replies keep every field

        with mock.patch.object(server, "MAX_REPLY_BYTES", 5000):
            capped = server.dic_run_campaign(self.campaign_params(campaign, "big-capped"))
            capped_plan = server.dic_plan(self.campaign_params(campaign, "big-capped-plan"))
        self.assertLessEqual(server._json_size(capped), 5000)
        self.assertLessEqual(server._json_size(capped_plan), 5000)
        self.assertTrue(capped["mcp"]["omitted"])
        self.assertEqual(capped["verdict"], "fail")
        self.assertIn("e05-fail", capped["mcp"]["brief"])

    def test_condensed_summary_names_the_events_that_failed(self) -> None:
        """F3: the condensed events list says which events are not ok."""
        run = self.jobs / "condensed"
        run.mkdir(parents=True)
        events = [{"name": f"event{index:02d}", "status": "fail" if index == 3 else "ok",
                   "warnings": ["x" * 400], "field_dir": f"/data/x/{index}" * 20} for index in range(25)]
        (run / "summary.json").write_text(json.dumps({"verdict": "fail", "events": events}), encoding="utf-8")
        result = server.dic_query_manifest(str(run))
        condensed = result["documents"]["summary.json"]["events"]["_condensed_list"]
        self.assertEqual(condensed["not_ok"], [{"index": 3, "name": "event03", "status": "fail"}])

    def test_manifest_query_max_bytes_is_a_hard_limit(self) -> None:
        """F2: the whole reply fits max_bytes; a big document no longer pushes out a small one."""
        run = self.jobs / "budget"
        run.mkdir(parents=True)
        events = [{"name": f"e{index}", "status": "ok", "text": "y" * 2000} for index in range(30)]
        (run / "summary.json").write_text(json.dumps({"verdict": "ok", "events": events}), encoding="utf-8")
        (run / server.JOB_RECORD_NAME).write_text(json.dumps({"state": "exited", "exit_code": 0}), encoding="utf-8")
        for index in range(3):
            (run / f"m{index}.manifest.json").write_text(json.dumps({"frames": list(range(100))}), encoding="utf-8")
        cases = [
            {"max_bytes": 1024},
            {"max_bytes": 1024, "fields": [f"nope{index}." + "z" * 240 for index in range(32)]},
            {"max_bytes": 4096, "detail": "full"},
            {"max_bytes": 2048, "documents": ["summary.json", "missing.json"]},
        ]
        for case in cases:
            with self.subTest(case={key: (value if key != "fields" else "32 long fields")
                                    for key, value in case.items()}):
                result = server.dic_query_manifest(str(run), **case)
                self.assertNotIn("code", result)
                self.assertLessEqual(server._json_size(result), case["max_bytes"])
        full = server.dic_query_manifest(str(run), detail="full", max_bytes=4096)
        self.assertIn(server.JOB_RECORD_NAME, full["documents"])  # the small one fits
        self.assertIn("summary.json", [entry["document"] for entry in full["omitted"]])
        too_long = server.dic_query_manifest(str(run), fields=["a" * 300])
        self.assertEqual(too_long["code"], "invalid_parameter")
        self.assertEqual(too_long["field"], "fields")

    def test_manifest_query_survives_odd_directories(self) -> None:
        """F10: a symlink out of the allowlist reveals nothing; F14: summary.json is never cut
        off and an undecodable JSON file is an error entry, not internal_error."""
        run = self.data / "odd-run"
        run.mkdir()
        (run / "summary.json").write_text('{"verdict": "ok"}', encoding="utf-8")
        for index in range(40):
            (run / f"a{index:02d}.json").write_text("{}", encoding="utf-8")
        (run / "zz-binary.json").write_bytes(b"\xff\xfe\x00{")
        secret = self.outside / "secret.json"
        secret.write_text('{"secret": "' + "s" * 2000 + '"}', encoding="utf-8")
        (run / "0leak.json").symlink_to(secret)  # sorts before a00.json: among the first 32 listed

        default = server.dic_query_manifest(str(run))
        self.assertEqual(default["available_documents"][0]["name"], "summary.json")
        self.assertEqual(default["documents"]["summary.json"], {"verdict": "ok"})
        self.assertTrue(default["truncated"])
        leak = next(entry for entry in default["available_documents"] if entry["name"] == "0leak.json")
        self.assertEqual(leak, {"name": "0leak.json", "bytes": None, "skipped": "symlink outside the read allowlist"})
        self.assertNotIn(str(secret), json.dumps(default))
        self.assertFalse(default["ok"])
        self.assertFalse(default["complete"])

        only = server.dic_query_manifest(str(run), documents=["summary.json"])
        self.assertEqual(only["documents"], {"summary.json": {"verdict": "ok"}})
        self.assertTrue(only["complete"])
        binary = server.dic_query_manifest(str(run), documents=["zz-binary.json"])
        self.assertNotEqual(binary.get("code"), "internal_error")
        self.assertFalse(binary["ok"])
        self.assertEqual(len(binary["errors"]), 1)

    def test_output_dir_inside_another_run_is_rejected(self) -> None:
        """F5: neither inside another run directory nor inside a frames directory."""
        running = self.jobs / "run-a"
        running.mkdir(parents=True)
        (running / server.JOB_RECORD_NAME).write_text('{"state": "running"}', encoding="utf-8")
        for index, output in enumerate((running / "nested", self.jobs / "x" / "_frames_e" / "rerun")):
            params = self.base_params("unused")
            params["output_dir"] = str(output)
            for tool in (server.dic_plan, server.dic_run_event):
                with self.subTest(output=str(output), tool=tool.__name__):
                    result = tool(params)
                    self.assertEqual(result["code"], "nested_output_dir")
                    self.assertEqual(result["field"], "output_dir")
            self.assertFalse(output.exists())

    def test_ok_requires_a_zero_exit_code(self) -> None:
        """F7 (mutant M4): a non-zero exit code makes ok false whatever the verdict says."""
        output = self.jobs / "exit-code"
        output.mkdir(parents=True)
        cod = output / "cod_result.csv"
        cod.write_text("frame,x_mm,cod_um\n2,8.0000,1.0000\n", encoding="utf-8")
        summary = {"verdict": "ok", "global": {"events_ok": 1}, "events": [{
            "name": "e", "status": "ok", "frames": 2, "warnings": [], "cod_csv": str(cod), "field_dir": None}]}
        good = server._finalize_run_result(json.loads(json.dumps(summary)), output, output / server.JOB_RECORD_NAME,
                                           {"state": "exited", "exit_code": 0}, {"keep_frames": False})
        self.assertTrue(good["ok"])
        bad = server._finalize_run_result(json.loads(json.dumps(summary)), output, output / server.JOB_RECORD_NAME,
                                          {"state": "exited", "exit_code": 1}, {"keep_frames": False})
        self.assertFalse(bad["ok"])

    def test_aggregation_waits_for_a_running_job_that_has_no_summary_yet(self) -> None:
        """F7 (mutant M8): a run directory with only .mcp-job.json is a run directory too."""
        running = self.jobs / "still-running"
        frames = _write_frames(running / "_frames_e", {10: 1, 20: 0.5})
        (running / server.JOB_RECORD_NAME).write_text('{"state": "running"}', encoding="utf-8")
        with server._output_lock(running.resolve()):
            into = server.dic_aggregate_cod(str(frames), str(running / "cod_other.csv"), [10, 20], 1.0)
            out_of = server.dic_aggregate_cod(str(frames), str(self.jobs / "agg" / "copy.csv"), [10, 20], 1.0)
        self.assertEqual(into["code"], "output_busy")
        self.assertEqual(out_of["code"], "output_busy")
        self.assertFalse((running / "cod_other.csv").exists())

    def test_a_killed_job_runner_does_not_turn_a_running_job_into_a_failure(self) -> None:
        """F4: the server adopts the job of a killed runner, waits for it and reports it."""
        slow = _write_event(self.data / "slow-adopt", frames=2)
        params = self.base_params("adopted")
        params["image_dir"] = str(slow)
        record_path = self.jobs / "adopted" / server.JOB_RECORD_NAME
        thread, result = self.run_in_thread(params)
        record = self.wait_for_record(record_path, lambda r: r.get("state") == "running" and r.get("runner_pid"))
        os.kill(record["runner_pid"], signal.SIGKILL)
        thread.join(timeout=60)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["verdict"], "ok", result)
        self.assertEqual(result["mcp"]["job"]["state"], "runner_lost")
        self.assertIn("job_runner_lost", [warning["code"] for warning in result["mcp"]["warnings"]])
        final = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(final["state"], "runner_lost")
        self.assertIsNone(final["exit_code"])
        self.assertIn("runner_lost_at", final)
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

        # Runner and job both gone, no summary: job_lost, not "pipeline exited None".
        sleepy = _write_event(self.data / "sleep-lost", frames=2)
        params = self.base_params("lost")
        params["image_dir"] = str(sleepy)
        record_path = self.jobs / "lost" / server.JOB_RECORD_NAME
        thread, result = self.run_in_thread(params)
        record = self.wait_for_record(record_path, lambda r: r.get("state") == "running" and r.get("job_pgid"))
        os.kill(record["runner_pid"], signal.SIGKILL)
        os.killpg(record["job_pgid"], signal.SIGKILL)
        thread.join(timeout=60)
        self.assertEqual(result["code"], "job_lost")
        self.assertTrue(result["retryable"])
        self.assertEqual(json.loads(record_path.read_text(encoding="utf-8"))["state"], "job_lost")
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

    def test_a_killed_pipeline_does_not_leave_its_binary_running(self) -> None:
        """F8: the runner clears the job's process group before it releases the locks."""
        sleepy = _write_event(self.data / "sleep-orphan", frames=2)
        params = self.base_params("orphan")
        params["image_dir"] = str(sleepy)
        record_path = self.jobs / "orphan" / server.JOB_RECORD_NAME
        thread, result = self.run_in_thread(params)
        record = self.wait_for_record(record_path, lambda r: r.get("state") == "running" and r.get("job_pid"))
        # Wait until the binary (a child of the pipeline, same process group) exists.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            members = subprocess.run(["pgrep", "-g", str(record["job_pgid"])], capture_output=True, text=True)
            if len(members.stdout.split()) >= 2:
                break
            time.sleep(0.05)
        os.kill(record["job_pid"], signal.SIGKILL)  # the pipeline alone
        thread.join(timeout=60)
        self.assertTrue(_group_gone(record["job_pgid"], wait=5.0), "the binary outlived its pipeline")
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))
        self.assertFalse(result["ok"])

    def test_names_that_are_not_utf8_are_structured_errors(self) -> None:
        """F13: used to fail while serialising the reply (isError)."""
        event = _write_event(self.data / "latin1", frames=2)
        with open(os.fsencode(str(event)) + b"/\xe9t\xe9_000003.png", "wb") as handle:
            handle.write((event / "000001.png").read_bytes())
        result = server.dic_probe_images(str(event))
        self.assertEqual(result["code"], "unsafe_path_characters")
        self.assertIn("\\xe9", result["error"])
        json.dumps(result)  # serialisable

    def test_lock_holder_text_reads_as_a_sentence(self) -> None:
        """F11: no more 'held by since ...'."""
        with server._gpu_lock(0, {"tool": "dic_run_event", "call_id": "f" * 32}):
            busy = server.dic_run_event(self.base_params("holder-text"))
        self.assertEqual(busy["code"], "gpu_busy")
        self.assertNotIn("held by since", busy["error"])
        self.assertIn("held by dic_run_event (call ffffffff)", busy["error"])

    def test_timeout_terminates_pipeline_process_group(self) -> None:
        sleepy = _write_event(self.data / "sleep-event", frames=2)
        params = self.base_params("timeout")
        params.update({"image_dir": str(sleepy), "timeout": 0.5})

        result = server.dic_run_event(params)

        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "timeout")
        self.assertIn("process group was terminated", result["error"])
        self.assertTrue(result["retryable"])
        self.assertIn("NEW output_dir", result["hint"])
        record = json.loads((self.jobs / "timeout" / server.JOB_RECORD_NAME).read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "timed_out")
        self.assertTrue(_group_gone(record["job_pgid"]), "pipeline/binary processes survived the timeout")
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

    # ── review 2026-10-04 recheck (rv-mcp R1-R3, R5) ──

    def test_a_stray_file_with_a_control_character_refuses_the_campaign_as_in_638ad18(self) -> None:
        """Guards never looser than 638ad18: every name directly under photo_dir is checked for C0/DEL."""
        campaign = self.make_campaign("stray-tab", 2)
        (campaign / "tab\tnotes.txt").write_text("not read", encoding="utf-8")
        params = self.campaign_params(campaign, "stray-tab-run")
        for tool in (server.dic_plan, server.dic_run_campaign):
            with self.subTest(tool=tool.__name__):
                refused = tool(params)
                self.assertEqual(refused["code"], "unsafe_path_characters")
                self.assertEqual(refused["error"], "photo_dir event contains control characters (U+0009); "
                                                   "C0/DEL are forbidden in MCP paths")
        self.assertFalse((self.jobs / "stray-tab-run").exists())

    def test_a_stray_file_with_a_non_utf8_name_does_not_refuse_a_campaign(self) -> None:
        """R1: a stray file needs no UTF-8 name (638ad18 ran this campaign); events are processed as usual."""
        campaign = self.make_campaign("stray", 2)
        try:
            with open(os.fsencode(str(campaign)) + b"/\xca\xc2\xbc\xfe-notes.txt", "wb") as handle:
                handle.write(b"notes")
        except OSError as exc:  # a file system that insists on UTF-8 names
            self.skipTest(f"cannot create a non-UTF-8 file name here: {exc}")
        params = self.campaign_params(campaign, "stray-run")
        plan = server.dic_plan(params)
        self.assertTrue(plan["valid"], plan)
        run = server.dic_run_campaign(params)
        self.assertTrue(run["ok"], run)
        self.assertEqual([(event["name"], event["status"]) for event in run["events"]], [("e00", "ok"), ("e01", "ok")])
        for name in ("e00", "e01"):
            rows = (self.jobs / "stray-run" / f"{name}.csv").read_text(encoding="utf-8").splitlines()
            self.assertGreater(len(rows), 1, f"{name}.csv has no COD row")
        json.dumps(run)  # serialisable

        # A directory with such a name is an event (the plan lists every
        # subdirectory) and is still refused with a readable name (F13).
        bad = os.fsencode(str(campaign)) + b"/\xca\xc2-event"
        os.mkdir(bad)
        for frame in (1, 2):
            with open(bad + f"/{frame:06d}.png".encode(), "wb") as handle:
                handle.write((campaign / "e00" / "000001.png").read_bytes())
        refused = server.dic_plan(self.campaign_params(campaign, "stray-run-2"))
        self.assertEqual(refused["code"], "unsafe_path_characters")
        self.assertIn("\\xca\\xc2-event", refused["error"])

    def test_run_rejections_obey_the_reply_cap(self) -> None:
        """R2: plan_invalid from a run tool (both kinds) and other rejections stay within MAX_REPLY_BYTES."""
        campaign = self.make_campaign("bad-rows", 12)
        params = self.campaign_params(campaign, "bad-rows-run")
        params["line_y"] = [10, 200]  # outside the 32-px images: one error per event
        uncapped = server.dic_run_campaign(params)
        self.assertEqual(uncapped["code"], "plan_invalid")
        self.assertNotIn("omitted", uncapped)  # small replies are unchanged
        self.assertEqual(json.loads(uncapped["error"]), uncapped["errors"])
        total = len(uncapped["errors"])
        self.assertGreaterEqual(total, 12)

        with mock.patch.object(server, "MAX_REPLY_BYTES", 2500):
            capped = server.dic_run_campaign(params)
        self.assertEqual(capped["code"], "plan_invalid")
        self.assertLessEqual(server._json_size(capped), 2500)
        self.assertEqual(json.loads(capped["error"]), capped["errors"])
        self.assertLess(len(capped["errors"]), total)
        note = next(item for item in capped["omitted"] if "errors" in item.get("total", {}))
        self.assertEqual(note["total"]["errors"], total)
        self.assertIn("dic_plan with image_dir=<photo_dir>/<event>", note["how_to_read"])
        self.assertFalse((self.jobs / "bad-rows-run").exists())

        # The pipeline's own --strict refusal (exit 2 after a passing plan check).
        real_build_plan = server._build_plan

        def stale_plan(normalized):
            plan = real_build_plan(normalized)
            plan.update(valid=True, ok=True, errors=[])
            return plan

        with mock.patch.object(server, "_build_plan", stale_plan), \
                mock.patch.object(server, "MAX_REPLY_BYTES", 2500):
            refused = server.dic_run_campaign(self.campaign_params(campaign, "bad-rows-strict") | {
                "line_y": [10, 200]})
        self.assertEqual(refused["code"], "plan_invalid")
        self.assertLessEqual(server._json_size(refused), 2500)
        self.assertIn("job_record", refused)
        note = next(item for item in refused["omitted"] if "errors" in item.get("total", {}))
        self.assertIn("summary.json", note["how_to_read"])

        # Any other rejection of a run tool, here with oversized tails.
        def failing() -> dict:
            raise server.InterfaceError(
                "pipeline_failed", "pipeline exited 1 without summary", output_dir=str(self.jobs / "x"),
                job_record=str(self.jobs / "x" / server.JOB_RECORD_NAME),
                stdout_tail=["s" * 400] * 40, stderr_tail=["e" * 400] * 40,
                error_lines=[f"Error {index}: " + "x" * 380 for index in range(10)])

        with mock.patch.object(server, "MAX_REPLY_BYTES", 4000):
            failed = server._dispatch("dic_run_event", {}, failing)
        self.assertEqual(failed["code"], "pipeline_failed")
        self.assertLessEqual(server._json_size(failed), 4000)
        self.assertNotIn("stdout_tail", failed)
        self.assertIn(server.JOB_STDOUT_NAME, failed["omitted"][0]["where"])
        small = server._dispatch("dic_run_event", {}, lambda: (_ for _ in ()).throw(
            server.InterfaceError("gpu_busy", "busy", retryable=True)))
        self.assertNotIn("omitted", small)

    def test_a_campaign_whose_events_all_fail_keeps_the_names_and_reasons(self) -> None:
        """R3: under the cap the failed events, one reason line each and the way to read them survive."""
        campaign = self.data / "all-zero"
        for index in range(12):
            _write_event(campaign / f"zero{index:02d}", frames=2)
        params = self.campaign_params(campaign, "all-zero-run")
        run = server.dic_run_campaign(params)
        self.assertFalse(run["ok"])
        self.assertEqual(run["verdict"], "fail")
        view = run["mcp"]["events_view"]
        self.assertEqual(len(view["failed"]), 12)
        self.assertEqual(len(run["events"]), 12)  # every compact record fits at the real cap
        self.assertEqual([group["count"] for group in view["failure_reasons"]], [12])
        self.assertTrue(view["failure_reasons"][0]["reason"].startswith("cod_all_zero: "))
        nulled = run["mcp"]["nulled_paths"]
        self.assertEqual(len(nulled), 1)  # 12 identical entries merged
        self.assertEqual((nulled[0]["field"], nulled[0]["count"]), ("field_dir", 12))

        with mock.patch.object(server, "MAX_REPLY_BYTES", 3000):
            capped = server.dic_run_campaign(self.campaign_params(campaign, "all-zero-capped"))
        self.assertLessEqual(server._json_size(capped), 3000)
        self.assertEqual(capped["verdict"], "fail")
        view = capped["mcp"]["events_view"]
        self.assertEqual(view.get("failed_total", len(view["failed"])), 12)
        self.assertEqual(view["failure_reasons"][0]["count"], 12)
        self.assertTrue(view["failure_reasons"][0]["reason"].startswith("cod_all_zero"))
        self.assertIn("summary.json", view["how_to_read"])
        self.assertIn("error_lines", capped["mcp"])
        self.assertTrue(capped["mcp"]["omitted"])

    def test_an_adopted_job_is_stopped_at_its_deadline(self) -> None:
        """R5 (mutant N2): after its runner is killed, the server still enforces the job's timeout."""
        sleepy = _write_event(self.data / "sleep-deadline", frames=2)
        params = self.base_params("adopted-deadline")
        params.update({"image_dir": str(sleepy), "timeout": 3})
        record_path = self.jobs / "adopted-deadline" / server.JOB_RECORD_NAME
        started = time.monotonic()
        thread, result = self.run_in_thread(params)
        record = self.wait_for_record(record_path, lambda r: r.get("state") == "running" and r.get("job_pgid"))
        try:
            os.kill(record["runner_pid"], signal.SIGKILL)
            thread.join(timeout=25)  # the fake binary sleeps 60 s
            self.assertFalse(thread.is_alive(), "the adopted job ran past its timeout")
        finally:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(record["job_pgid"], signal.SIGKILL)
            thread.join(timeout=30)
        self.assertEqual(result["code"], "timeout", result)
        self.assertLess(time.monotonic() - started, 20)
        final = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(final["state"], "timed_out")
        self.assertIn("runner_lost_at", final)
        self.assertTrue(_group_gone(record["job_pgid"]))
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

    def test_job_record_updates_wait_for_the_record_lock(self) -> None:
        """R5 (mutant N10): the server's read-modify-write of .mcp-job.json takes the record lock."""
        run = self.jobs / "record-lock"
        run.mkdir(parents=True)
        record_path = run / server.JOB_RECORD_NAME
        record_path.write_text('{"state": "running"}', encoding="utf-8")
        held = threading.Event()

        def hold_the_lock() -> None:
            fd = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                held.set()
                time.sleep(0.6)
            finally:
                os.close(fd)

        holder = threading.Thread(target=hold_the_lock)
        holder.start()
        self.assertTrue(held.wait(5))
        started = time.monotonic()
        server._merge_job_record(record_path, server_detached_at="now")
        waited = time.monotonic() - started
        holder.join(5)
        self.assertGreaterEqual(waited, 0.4, "the record was rewritten while another writer held its lock")
        self.assertEqual(json.loads(record_path.read_text(encoding="utf-8")),
                         {"state": "running", "server_detached_at": "now"})


class McpEventSelectionTests(McpTestBase):
    """2026-10-05: events (pipeline --events) and event_interface_y (--event-interface-y)."""

    def make_events(self, name: str, count: int) -> Path:
        campaign = self.data / name
        for index in range(1, count + 1):
            _write_event(campaign / f"event{index:02d}", frames=2)
        return campaign

    def params(self, campaign: Path, output_name: str, **extra) -> dict:
        # interface_y 15, gauge 10 -> lines 10/20; an override row 16 -> lines 11/21.
        return {"photo_dir": str(campaign), "output_dir": str(self.jobs / output_name),
                "interface_y": 15, "gauge": 10, "cal": 1.0, "subset": 15, **extra}

    def test_argv_without_the_new_keys_is_unchanged(self) -> None:
        campaign = self.make_events("argv-same", 2)
        normalized = server._normalize_pipeline_params(self.params(campaign, "argv-same-run"))
        self.assertEqual(server._pipeline_argv(normalized), [
            f"--photo-dir={campaign}", f"--output-dir={self.jobs / 'argv-same-run'}", "--interface-y=15",
            "--cal=1.0", "--subset=15", "--step=1", "--gauge=10", "--cod-method=naive",
            "--conv-threshold=0.001", "--gpu-device=0", f"--gpu-bin={self.fake_bin}", "--strict",
        ])
        self.assertFalse(server._uses_event_selection(normalized))
        # null means "not given", as for the other optional keys.
        nulls = server._normalize_pipeline_params(
            self.params(campaign, "argv-null", events=None, event_interface_y=None))
        self.assertEqual(server._pipeline_argv(nulls),
                         [token.replace("argv-same-run", "argv-null") for token in server._pipeline_argv(normalized)])

    def test_new_keys_become_pipeline_options(self) -> None:
        campaign = self.make_events("argv-new", 2)
        normalized = server._normalize_pipeline_params(self.params(
            campaign, "argv-new-run", events=["event01", "event0*"],
            event_interface_y=[{"pattern": "event02", "interface_y": 16}, {"pattern": "e*1", "interface_y": 15}]))
        argv = server._pipeline_argv(normalized)
        self.assertIn("--events=event01,event0*", argv)
        self.assertEqual([token for token in argv if token.startswith("--event-interface-y")],
                         ["--event-interface-y=event02=16", "--event-interface-y=e*1=15"])
        parser = server.pipeline.create_parser()
        parsed = parser.parse_args(argv + ["--dry-run"])
        self.assertEqual(parsed.events, "event01,event0*")
        self.assertEqual(parsed.event_interface_y, ["event02=16", "e*1=15"])
        separate = vars(parser.parse_args([part for token in argv for part in token.split("=", 1)] + ["--dry-run"]))
        self.assertEqual(vars(parsed), separate)

    def test_plan_reports_the_selection_and_the_lines_of_each_event(self) -> None:
        campaign = self.make_events("plan", 4)
        plan = server.dic_plan(self.params(campaign, "plan-run", events=["event01", "event03"],
                                           event_interface_y=[{"pattern": "event03", "interface_y": 16}]))
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertEqual([event["name"] for event in plan["events"]], ["event01", "event03"])
        lines = {event["name"]: (event["geometry"]["line_ys"], event["geometry"]["interface_y_source"])
                 for event in plan["events"]}
        self.assertEqual(lines, {"event01": ([10, 20], "interface_y"), "event03": ([11, 21], "event_interface_y")})
        self.assertIn("--line 11,21", " ".join(plan["events"][1]["binary_command"]))
        selection = plan["event_selection"]
        self.assertEqual((selection["events_discovered"], selection["events_selected"], selection["events_excluded"]),
                         (4, 2, 2))
        self.assertEqual(selection["excluded_events"], ["event02", "event04"])
        self.assertEqual(selection["interface_y_overrides"][0]["events"], ["event03"])
        definition = plan["mcp"]["cod_definition"]
        self.assertEqual((definition["L0_y"], definition["L1_y"]), (10, 20))
        self.assertEqual(definition["per_event_lines"], [
            {"L0_y": 10, "L1_y": 20, "interface_y": 15, "interface_y_source": "interface_y", "event_count": 1,
             "events": ["event01"]},
            {"L0_y": 11, "L1_y": 21, "interface_y": 16, "interface_y_source": "event_interface_y", "event_count": 1,
             "events": ["event03"]},
        ])
        self.assertNotIn("event_selection_view", plan["mcp"])
        # events alone: no per_event_lines (every event uses interface_y).
        only = server.dic_plan(self.params(campaign, "plan-only", events=["event0[12]"]))
        self.assertTrue(only["valid"], only["errors"])
        self.assertEqual(only["event_selection"]["events_selected"], 2)
        self.assertNotIn("per_event_lines", only["mcp"]["cod_definition"])

    def test_every_refusal_is_structured_with_the_pipeline_code(self) -> None:
        campaign = self.make_events("errors", 3)
        override = [{"pattern": "event03", "interface_y": 16}]
        plan_cases = {
            "pattern_no_match": ({"events": ["event9*"]}, "event_pattern_no_match"),
            "name_not_found": ({"events": ["event01", "event26"]}, "event_not_found"),
            "conflict": ({"event_interface_y": [{"pattern": "event0*", "interface_y": 16},
                                                {"pattern": "event03", "interface_y": 17}]},
                         "event_interface_y_conflict"),
            "override_no_match": ({"event_interface_y": [{"pattern": "event3[0-9]", "interface_y": 16}]},
                                  "event_interface_y_no_match"),
            "override_excluded": ({"events": ["event01"], "event_interface_y": override},
                                  "event_interface_y_no_match"),
            "with_line_y": ({"line_y": [10, 20], "event_interface_y": override}, "event_interface_y_with_line_y"),
            "negative_row": ({"event_interface_y": [{"pattern": "event03", "interface_y": -1}]},
                             "invalid_event_interface_y"),
            "comma_in_pattern": ({"event_interface_y": [{"pattern": "event0[1,2]", "interface_y": 16}]},
                                 "invalid_event_interface_y"),
            "empty_item": ({"events": ["event01", ""]}, "invalid_events"),
        }
        for name, (extra, code) in plan_cases.items():
            with self.subTest(case=name):
                plan = server.dic_plan(self.params(campaign, f"err-{name}", **extra))
                self.assertFalse(plan["ok"])
                self.assertFalse(plan["valid"])
                self.assertIn(code, [item["code"] for item in plan["errors"]], plan["errors"])
                run = server.dic_run_campaign(self.params(campaign, f"err-{name}", **extra))
                self.assertEqual(run["code"], "plan_invalid")
                self.assertIn(code, [item["code"] for item in run["errors"]])
                self.assertFalse((self.jobs / f"err-{name}").exists())
        with_image_dir = server.dic_plan({**self.params(campaign, "err-image-dir", events=["event01"]),
                                          "photo_dir": None, "image_dir": str(campaign / "event01")})
        self.assertIn("events_requires_photo_dir", [item["code"] for item in with_image_dir["errors"]])

        shape_cases = {
            "missing_interface_y": ({"interface_y": None, "event_interface_y": override},
                                    "event_interface_y_requires_interface_y", "interface_y"),
            "events_string": ({"events": "event01,event02"}, "invalid_events", "events"),
            "events_empty": ({"events": []}, "invalid_events", "events"),
            "events_item_type": ({"events": ["event01", 2]}, "invalid_events", "events[1]"),
            "events_comma": ({"events": ["event01,event02"]}, "invalid_events", "events[0]"),
            "override_not_array": ({"event_interface_y": {"event03": 16}}, "invalid_event_interface_y",
                                   "event_interface_y"),
            "override_string": ({"event_interface_y": ["event03=16"]}, "invalid_event_interface_y",
                                "event_interface_y[0]"),
            "override_missing_key": ({"event_interface_y": [{"pattern": "event03"}]}, "invalid_event_interface_y",
                                     "event_interface_y[0]"),
            "override_extra_key": ({"event_interface_y": [{"pattern": "event03", "interface_y": 16, "y": 1}]},
                                   "invalid_event_interface_y", "event_interface_y[0]"),
            "row_string": ({"event_interface_y": [{"pattern": "event03", "interface_y": "133"}]},
                           "invalid_event_interface_y", "event_interface_y[0].interface_y"),
            "row_float": ({"event_interface_y": [{"pattern": "event03", "interface_y": 16.0}]},
                          "invalid_event_interface_y", "event_interface_y[0].interface_y"),
            "row_bool": ({"event_interface_y": [{"pattern": "event03", "interface_y": True}]},
                         "invalid_event_interface_y", "event_interface_y[0].interface_y"),
            "pattern_type": ({"event_interface_y": [{"pattern": 3, "interface_y": 16}]},
                             "invalid_event_interface_y", "event_interface_y[0].pattern"),
        }
        for name, (extra, code, field) in shape_cases.items():
            with self.subTest(case=name):
                for tool in (server.dic_plan, server.dic_run_campaign):
                    result = tool(self.params(campaign, f"shape-{name}", **extra))
                    self.assertFalse(result["ok"])
                    self.assertEqual(result["code"], code, result)
                    self.assertEqual(result["field"], field)
                    self.assertTrue(result["expected"])
                self.assertFalse((self.jobs / f"shape-{name}").exists())
        records = [entry for entry in self.audit_records() if entry["result_code"] != "started"]
        self.assertTrue(all(entry["result_code"] != "internal_error" for entry in records))

    def test_run_names_the_lines_of_each_event(self) -> None:
        campaign = self.make_events("run", 3)
        run = server.dic_run_campaign(self.params(
            campaign, "run-out", events=["event01", "event03"],
            event_interface_y=[{"pattern": "event03", "interface_y": 16}]))
        self.assertTrue(run["ok"], run["mcp"])
        self.assertEqual([event["name"] for event in run["events"]], ["event01", "event03"])
        self.assertEqual([event["geometry"]["line_ys"] for event in run["events"]], [[10, 20], [11, 21]])
        self.assertEqual(run["event_selection"]["events_excluded"], 1)
        brief = run["mcp"]["brief"]
        self.assertIn("events filter selected 2 of 3 (1 excluded)", brief)
        self.assertIn("lines y 10/20 (interface_y 15, 1 event): event01", brief)
        self.assertIn("lines y 11/21 (event_interface_y 16, 1 event): event03", brief)
        self.assertEqual([group["events"] for group in run["mcp"]["cod_definition"]["per_event_lines"]],
                         [["event01"], ["event03"]])
        self.assertEqual(sorted(path.name for path in (self.jobs / "run-out").glob("*.csv")),
                         ["event01.csv", "event03.csv"])
        # The fake binary writes the --line rows it was given; event03's COD came from rows 11/21.
        command = json.loads(Path(run["events"][1]["binary_stderr_log"]).read_text(encoding="utf-8")
                             .splitlines()[0].removeprefix("# command: "))
        self.assertEqual(command[command.index("--line") + 1], "11,21")

    def test_more_than_ten_events_stay_compact_and_bounded(self) -> None:
        campaign = self.make_events("big", 24)
        extra = {"events": ["event0*", "event10", "event11", "event12"],
                 "event_interface_y": [{"pattern": "event1*", "interface_y": 16}]}
        plan = server.dic_plan(self.params(campaign, "big-run", **extra))
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertEqual(plan["mcp"]["events_view"]["view"], "compact")
        self.assertEqual(len(plan["events"]), 12)  # event01-09, event10-12
        self.assertEqual(plan["events"][-1]["line_ys"], [11, 21])
        self.assertEqual(plan["events"][0]["interface_y_source"], "interface_y")
        self.assertIn("events=['<event>']", plan["mcp"]["events_view"]["how_to_read"])
        selection = plan["event_selection"]
        self.assertEqual(selection["events_excluded"], 12)
        self.assertEqual(len(selection["excluded_events"]), server.COMPACT_EVENTS_THRESHOLD)
        self.assertEqual(plan["mcp"]["event_selection_view"]["totals"], {"excluded_events": 12})
        self.assertLessEqual(server._json_size(plan), server.MAX_REPLY_BYTES)

        run = server.dic_run_campaign(self.params(campaign, "big-run", **extra))
        self.assertTrue(run["ok"], run["mcp"])
        self.assertEqual(run["mcp"]["events_view"]["view"], "compact")
        self.assertEqual({tuple(event["line_ys"]) for event in run["events"]}, {(10, 20), (11, 21)})
        self.assertEqual(len(run["event_selection"]["excluded_events"]), server.COMPACT_EVENTS_THRESHOLD)
        self.assertIn("event_selection", run["mcp"]["event_selection_view"]["how_to_read"])
        self.assertIn("lines y 11/21 (event_interface_y 16, 3 events): event10, event11, event12", run["mcp"]["brief"])
        self.assertIn("lines y 10/20 (interface_y 15, 9 events): event01, event02, event03, event04, event05 +4 more",
                      run["mcp"]["brief"])
        self.assertLessEqual(server._json_size(run), server.MAX_REPLY_BYTES)
        full = json.loads((self.jobs / "big-run" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(full["event_selection"]["excluded_events"]), 12)  # summary.json keeps every name

        # Without the new keys a large campaign reply is shaped exactly as before.
        plain = server.dic_run_campaign(self.params(campaign, "big-plain", max_frames=1))
        allowed = {"name", "status", "frames", "valid_ratio", "zncc_median", "cod_rows", "cod_peak_um", "warnings"}
        self.assertTrue(all(set(event) <= allowed for event in plain["events"]))
        self.assertNotIn("event_selection_view", plain["mcp"])
        self.assertNotIn("per_event_lines", plain["mcp"]["cod_definition"])
        self.assertNotIn("events filter", plain["mcp"]["brief"])


class McpJobModelTests(McpTestBase):
    """2026-10-05: wait=false jobs, progress in the job record, dic_job_cancel (review 2026-10-03 #14)."""

    def setUp(self) -> None:
        super().setUp()
        interval = mock.patch.object(server, "PROGRESS_INTERVAL_SECONDS", 0.1)
        interval.start()
        self.patchers.append(interval)

    def job_params(self, event_name: str, output_name: str, frames: int = 3, **extra) -> dict:
        params = self.base_params(output_name)
        params["image_dir"] = str(_write_event(self.data / event_name, frames=frames))
        params.update(extra)
        return params

    def record_path(self, output_name: str) -> Path:
        return self.jobs / output_name / server.JOB_RECORD_NAME

    def poll(self, output_name: str) -> dict:
        """The job record as a client reads it: dic_query_manifest on the output directory."""
        reply = server.dic_query_manifest(str(self.jobs / output_name), documents=[server.JOB_RECORD_NAME])
        return reply.get("documents", {}).get(server.JOB_RECORD_NAME, {})

    def wait_until_idle(self, timeout: float = 15.0) -> None:
        """Until the server threads that watch jobs have left their lock contexts."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with server._REGISTRY_GUARD:
                busy = bool(server._ACTIVE_JOBS)
            if not busy and not server._INPROCESS_LOCKS:
                return
            time.sleep(0.02)
        self.fail("a job thread is still running")

    def wait_for_binary(self, record: dict, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            members = subprocess.run(["pgrep", "-g", str(record["job_pgid"])], capture_output=True, text=True)
            if len(members.stdout.split()) >= 2:
                return
            time.sleep(0.05)
        self.fail("the fake binary never started")

    def audit_codes(self, call_id: str) -> list[str]:
        return [entry["result_code"] for entry in self.audit_records() if entry["call_id"] == call_id]

    def assert_fields_listed(self, record: dict) -> None:
        """Every key of a job record is in the contract's mcp.job_record.fields (review 2026-10-05 F7)."""
        contract = json.loads(server.CONTRACT_PATH.read_text(encoding="utf-8"))
        listed = {entry.split(" ", 1)[0] for entry in contract["mcp"]["job_record"]["fields"]}
        self.assertFalse(set(record) - listed, f"job record fields missing from the contract: {set(record) - listed}")

    def test_wait_false_returns_a_handle_and_the_job_ends_like_a_blocking_run(self) -> None:
        params = self.job_params("tick-handle", "bg", frames=6, wait=False)
        handle = server.dic_run_event(params)
        self.assertTrue(handle["ok"], handle)
        self.assertEqual(handle["job_id"], handle["call_id"])
        # The call returned while the job (5 frames x 0.25 s) was still going.
        self.assertIn(handle["state"], {"starting", "running"})
        self.assertIn(self.poll("bg").get("state"), {"starting", "running"})
        self.assertEqual(Path(handle["output_dir"]), (self.jobs / "bg").resolve())
        self.assertEqual(Path(handle["job_record"]), (self.jobs / "bg").resolve() / server.JOB_RECORD_NAME)
        self.assertEqual(Path(handle["summary_json"]).name, "summary.json")
        self.assertRegex(handle["cancel_token"], r"^[0-9a-f]{48}$")
        for key in ("verdict", "events", "global", "code"):
            self.assertNotIn(key, handle)  # a handle, not a summary and not an error
        poll = handle["mcp"]["poll"]
        self.assertEqual(poll["tool"], "dic_query_manifest")
        self.assertEqual(poll["arguments"], {"run_dir": handle["output_dir"], "documents": [server.JOB_RECORD_NAME],
                                             "fields": ["state", "progress"]})
        self.assertEqual(poll["poll_after_s"], 5)
        self.assertEqual([server._poll_interval(value) for value in (None, 92, 860, 3600)], [5, 5, 28, 30])
        self.assertEqual(handle["mcp"]["cancel"]["arguments"], {
            "output_dir": handle["output_dir"], "job_id": handle["job_id"], "cancel_token": handle["cancel_token"]})
        self.assertEqual(handle["mcp"]["cod_definition"]["L0_y"], 10)
        self.assertLessEqual(server._json_size(handle), 4096)

        # Poll exactly as the handle says, until the job has ended: every poll is complete and small.
        seen: list[dict] = []
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            reply = server.dic_query_manifest(**poll["arguments"])
            self.assertTrue(reply["complete"], reply)
            self.assertNotIn("missing_fields", reply)
            self.assertLess(server._json_size(reply), 1024)
            record = reply["documents"][server.JOB_RECORD_NAME]
            self.assertEqual(list(record), ["state", "progress"])
            seen.append(record)
            if record["state"] not in {"starting", "running"}:
                break
            time.sleep(0.05)
        self.assertEqual(seen[-1]["state"], "exited")
        self.assertEqual(seen[0]["progress"]["frames_total"], 5)  # the plan's totals are there from the start
        final = self.poll("bg")  # the whole record
        self.assertEqual(final["state"], "exited", final)
        self.assert_fields_listed(final)
        self.assertEqual(final["exit_code"], 0)
        self.assertEqual((final["verdict"], final["events_ok"], final["events_failed"]), ("ok", 1, 0))
        self.assertIs(final["wait"], False)
        self.assertEqual(final["call_id"], handle["job_id"])
        progress = final["progress"]
        self.assertEqual((progress["events_done"], progress["events_total"]), (1, 1))
        self.assertEqual((progress["frames_done"], progress["frames_total"]), (5, 5))
        self.assertIsNone(progress["current_event"])
        running = [record["progress"] for record in seen if record["state"] == "running"]
        self.assertTrue(running, "no progress was seen while the job ran")
        self.assertTrue(any(0 < item["frames_done"] < 5 for item in running), [item["frames_done"] for item in running])
        self.assertTrue(all(item["frames_total"] == 5 and item["events_total"] == 1 for item in running))
        self.assertTrue(any(item.get("current_event") == "tick-handle" for item in running))
        frames_seen = [item["frames_done"] for item in running]
        self.assertEqual(frames_seen, sorted(frames_seen))
        self.assertLess(server._json_size(server.dic_query_manifest(**poll["arguments"])),
                        server._json_size(server.dic_query_manifest(
                            poll["arguments"]["run_dir"], documents=[server.JOB_RECORD_NAME])) // 2)

        # The token is kept only as a hash; neither the record nor the audit log holds it.
        self.assertEqual(final["cancel_token_sha256"], hashlib.sha256(handle["cancel_token"].encode()).hexdigest())
        self.assertNotIn(handle["cancel_token"], self.record_path("bg").read_text(encoding="utf-8"))
        self.wait_until_idle()
        self.assertNotIn(handle["cancel_token"], (self.jobs / server.AUDIT_LOG_NAME).read_text(encoding="utf-8"))
        self.assertEqual(self.audit_codes(handle["job_id"]), ["started", "job_started", "job_finished:ok"])
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

        # Same output as a blocking run of the same event, whose reply is the summary as before.
        blocking = server.dic_run_event({**params, "output_dir": str(self.jobs / "fg"), "wait": True})
        self.assertEqual(blocking["verdict"], "ok")
        for key in ("job_id", "cancel_token", "state"):
            self.assertNotIn(key, blocking)
        self.assertEqual((self.jobs / "bg" / "cod_result.csv").read_bytes(),
                         (self.jobs / "fg" / "cod_result.csv").read_bytes())
        summary = server.dic_query_manifest(**handle["mcp"]["result"]["arguments"])["documents"]["summary.json"]
        self.assertEqual(summary["verdict"], "ok")
        blocking_record = json.loads(self.record_path("fg").read_text(encoding="utf-8"))
        self.assertIs(blocking_record["wait"], True)
        self.assertNotIn("cancel_token_sha256", blocking_record)
        self.assertEqual(blocking_record["progress"]["frames_done"], 5)  # a blocking run reports progress too

        # After the job has ended a cancel does nothing, however often it is repeated.
        for _ in range(2):
            late = server.dic_job_cancel(handle["output_dir"], handle["job_id"], handle["cancel_token"])
            self.assertTrue(late["ok"])
            self.assertEqual((late["action"], late["state"]), ("none", "exited"))
        self.assertEqual(json.loads(self.record_path("bg").read_text(encoding="utf-8"))["state"], "exited")

    def test_wait_is_validated_and_refusals_before_the_start_are_unchanged(self) -> None:
        for tool in (server.dic_plan, server.dic_run_event):
            bad = tool(self.job_params("wait-bad", f"wait-bad-{tool.__name__}", wait="no"))
            self.assertEqual((bad["code"], bad["field"]), ("invalid_parameter", "wait"))
        plan = server.dic_plan(self.job_params("wait-plan", "wait-plan", wait=False))
        self.assertTrue(plan["valid"])  # dic_plan accepts and ignores it
        self.assertFalse((self.jobs / "wait-plan").exists())
        null = self.job_params("wait-null", "wait-null", wait=None)
        self.assertEqual(server.dic_run_event(null)["verdict"], "ok")  # null = not given = blocking

        # What a blocking call refuses before the job starts, a wait=false call refuses the same way.
        invalid = self.job_params("wait-invalid", "wait-invalid", wait=False)
        invalid["line_y"] = [10, 200]
        refused = server.dic_run_event(invalid)
        self.assertEqual(refused["code"], "plan_invalid")
        self.assertEqual(json.loads(refused["error"]), refused["errors"])
        self.assertFalse((self.jobs / "wait-invalid").exists())
        with server._gpu_lock(0, {"tool": "dic_run_event", "call_id": "e" * 32, "output_dir": "/elsewhere"}):
            busy = server.dic_run_event(self.job_params("wait-busy", "wait-busy", wait=False))
        self.assertEqual(busy["code"], "gpu_busy")
        self.assertTrue(busy["retryable"])
        existing = self.jobs / "wait-existing"
        existing.mkdir(parents=True)
        self.assertEqual(server.dic_run_event(self.job_params("wait-existing", "wait-existing", wait=False))["code"],
                         "output_exists")
        self.wait_until_idle()
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

    def test_a_second_job_on_a_busy_gpu_is_refused_while_a_background_job_runs(self) -> None:
        handle = server.dic_run_event(self.job_params("sleep-busy", "bg-busy", wait=False))
        self.assertTrue(handle["ok"], handle)
        record = self.wait_for_record(self.record_path("bg-busy"), lambda r: r.get("state") == "running")
        busy = server.dic_run_event(self.job_params("second", "second"))
        self.assertEqual(busy["code"], "gpu_busy")
        self.assertEqual(busy["holder"]["call_id"], handle["job_id"])
        self.assertEqual(busy["holder"]["runner_pid"], record["runner_pid"])
        cancelled = server.dic_job_cancel(handle["output_dir"])
        self.assertEqual(cancelled["action"], "cancelled")
        self.wait_until_idle()
        self.assertEqual(server.dic_run_event(self.job_params("second", "second"))["verdict"], "ok")

    def test_cancel_stops_the_job_records_it_and_releases_the_lock(self) -> None:
        handle = server.dic_run_event(self.job_params("sleep-cancel", "cancel-me", wait=False))
        record = self.wait_for_record(self.record_path("cancel-me"),
                                      lambda r: r.get("state") == "running" and r.get("job_pgid"))
        self.wait_for_binary(record)
        self.assertFalse(_lock_is_free(server._gpu_lock_path(0)))
        # The record names the two processes for good: pid and start time (review 2026-10-05 F2).
        self.assertEqual(record["runner_start_ticks"], server._proc_start_ticks(record["runner_pid"]))
        self.assertEqual(record["job_start_ticks"], server._proc_start_ticks(record["job_pgid"]))
        self.assertIsInstance(record["job_start_ticks"], int)
        # While the job runs the record has its progress, and no outcome field yet (they appear when known).
        self.assertEqual(record["progress"]["events_total"], 1)
        for key in ("exit_code", "verdict", "ended_at", "cancel_reason"):
            self.assertNotIn(key, record)

        started = time.monotonic()
        cancelled = server.dic_job_cancel(handle["output_dir"], job_id=handle["job_id"])  # own session: no token
        self.assertLess(time.monotonic() - started, 10)  # the fake binary would sleep 60 s
        self.assertTrue(cancelled["ok"], cancelled)
        self.assertEqual((cancelled["action"], cancelled["state"]), ("cancelled", "cancelled"))
        self.assertEqual(cancelled["job_id"], handle["job_id"])
        self.assertEqual(cancelled["processes"], {"runner_alive": False, "job_group_alive": False})
        self.assertEqual(cancelled["partial"]["summary_json"], False)
        self.assertIn("NEW output_dir", cancelled["hint"])
        self.assertTrue(_group_gone(record["job_pgid"]), "pipeline or binary survived the cancellation")
        self.assertTrue(_group_gone(record["runner_pid"]), "the job runner survived the cancellation")
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))
        final = json.loads(self.record_path("cancel-me").read_text(encoding="utf-8"))
        self.assertEqual(final["state"], "cancelled")
        self.assert_fields_listed(final)
        self.assertEqual(final["cancel_reason"], f"dic_job_cancel (call {cancelled['call_id']})")
        self.assertEqual(final["cancel_requested_by"]["call_id"], cancelled["call_id"])
        self.assertIn("ended_at", final)
        self.assertEqual(final["progress"]["events_done"], 0)

        again = server.dic_job_cancel(handle["output_dir"])
        self.assertTrue(again["ok"])
        self.assertEqual((again["action"], again["state"]), ("none", "cancelled"))
        self.assertNotIn("hint", again)
        self.wait_until_idle()
        self.assertEqual(self.audit_codes(handle["job_id"]), ["started", "job_started", "job_finished:job_cancelled"])
        self.assertEqual(self.audit_codes(cancelled["call_id"]), ["started", "cancel:cancelled"])
        self.assertEqual(self.audit_codes(again["call_id"]), ["started", "cancel:none"])
        # The device is usable at once, and the cancelled directory is not.
        self.assertEqual(server.dic_run_event(self.job_params("after-cancel", "after-cancel"))["verdict"], "ok")
        self.assertEqual(server.dic_run_event(self.job_params("sleep-cancel", "cancel-me"))["code"], "output_exists")

    def test_cancelling_a_blocking_call_returns_job_cancelled(self) -> None:
        params = self.job_params("sleep-blocking", "blocking")
        thread, result = self.run_in_thread(params)
        record = self.wait_for_record(self.record_path("blocking"),
                                      lambda r: r.get("state") == "running" and r.get("job_pgid"))
        self.assertIs(record["wait"], True)
        cancelled = server.dic_job_cancel(str(self.jobs / "blocking"))
        thread.join(timeout=30)
        self.assertFalse(thread.is_alive())
        self.assertEqual(cancelled["action"], "cancelled")
        self.assertEqual(result["code"], "job_cancelled", result)
        self.assertTrue(result["retryable"])
        self.assertIn("dic_job_cancel", result["error"])
        self.assertIn("NEW output_dir", result["hint"])
        self.assertNotIn("larger timeout", result["hint"])
        self.assertTrue(_group_gone(record["job_pgid"]))
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

    def test_cancel_refuses_what_is_not_a_job(self) -> None:
        self.jobs.mkdir(exist_ok=True)
        (self.jobs / "empty").mkdir()
        (self.jobs / "not-a-record").mkdir()
        (self.jobs / "not-a-record" / server.JOB_RECORD_NAME).write_text('{"state": "running"}', encoding="utf-8")
        for name in ("missing", "empty", "not-a-record"):
            with self.subTest(output_dir=name):
                result = server.dic_job_cancel(str(self.jobs / name))
                self.assertFalse(result["ok"])
                self.assertEqual((result["code"], result["field"]), ("job_not_found", "output_dir"))
                self.assertFalse(result["retryable"])
        self.assertFalse((self.jobs / "missing").exists())
        self.assertEqual(server.dic_job_cancel(str(self.outside / "run"))["code"], "write_path_denied")
        self.assertEqual(server.dic_job_cancel(str(self.jobs))["code"], "write_path_denied")
        self.assertEqual(server.dic_job_cancel("relative/run")["code"], "invalid_path")
        self.assertEqual(server.dic_job_cancel(str(self.jobs / server.CONTROL_DIR_NAME))["code"],
                         "reserved_write_path")

        done = server.dic_run_event(self.job_params("done", "done"))
        self.assertEqual(done["verdict"], "ok")
        finished = server.dic_job_cancel(done["mcp"]["output_dir"], job_id=done["call_id"])
        self.assertTrue(finished["ok"])
        self.assertEqual((finished["action"], finished["state"]), ("none", "exited"))
        self.assertEqual(finished["job"]["exit_code"], 0)
        mismatch = server.dic_job_cancel(done["mcp"]["output_dir"], job_id="0" * 32)
        self.assertEqual((mismatch["code"], mismatch["field"]), ("job_id_mismatch", "job_id"))
        self.assertEqual(mismatch["expected"], done["call_id"])
        for bad_id in ("xyz", 12, "A" * 32):
            self.assertEqual(server.dic_job_cancel(done["mcp"]["output_dir"], job_id=bad_id)["code"],
                             "invalid_parameter")
        self.assertEqual(server.dic_job_cancel(done["mcp"]["output_dir"], cancel_token=5)["code"], "invalid_parameter")
        records = [entry for entry in self.audit_records() if entry["tool"] == "dic_job_cancel"]
        self.assertTrue(all(entry["result_code"] != "internal_error" for entry in records))

    def stale_job(self, name: str, **fields) -> Path:
        """A job record nobody will update: what a killed runner and a gone server leave behind."""
        directory = self.jobs / name
        directory.mkdir(parents=True)
        record = {"schema": server.JOB_RECORD_SCHEMA, "state": "running", "call_id": "a" * 32,
                  "tool": "dic_run_event", "output_dir": str(directory), "server_pid": 1,
                  "runner_lost_at": "2026-10-05T00:00:00Z", "server_detached_at": "2026-10-05T00:00:01Z"}
        record.update(fields)
        (directory / server.JOB_RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
        return directory

    def dead_pid(self) -> int:
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait()
        return process.pid

    def helper(self, *arguments: str) -> subprocess.Popen:
        """A process group of its own that sleeps; its argv carries ``arguments``."""
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", *arguments],
                                   start_new_session=True)
        self.addCleanup(process.wait)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        return process

    def test_a_stale_record_is_settled_by_anyone(self) -> None:
        """Review 2026-10-04 recheck R4: runner killed, server gone, record left at running."""
        dead = self.dead_pid()
        lost = self.stale_job("stale-lost", runner_pid=dead, job_pgid=dead)
        settled = server.dic_job_cancel(str(lost))
        self.assertTrue(settled["ok"], settled)
        self.assertEqual((settled["action"], settled["state"]), ("settled", "job_lost"))
        record = json.loads((lost / server.JOB_RECORD_NAME).read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "job_lost")
        self.assert_fields_listed(record)
        self.assertEqual(record["settled_by"], f"dic_job_cancel (call {settled['call_id']})")
        self.assertIn("ended_at", record)
        self.assertEqual(server.dic_job_cancel(str(lost))["action"], "none")

        # The pipeline had finished by itself: its summary is there, so the state says so.
        finished = self.stale_job("stale-finished", runner_pid=dead, job_pgid=dead)
        (finished / "summary.json").write_text('{"verdict": "ok"}', encoding="utf-8")
        settled = server.dic_job_cancel(str(finished))
        self.assertEqual((settled["action"], settled["state"]), ("settled", "runner_lost"))
        self.assertIn("summary.json", settled["note"])
        self.assertTrue(settled["partial"]["summary_json"])

    def test_a_job_of_another_session_needs_its_token_and_a_reused_pid_is_never_signalled(self) -> None:
        token = "t" * 48
        digest = hashlib.sha256(token.encode()).hexdigest()
        # The runner is gone, the pipeline still runs: a live job of somebody else.
        directory = self.jobs / "foreign"
        pipeline_like = self.helper(f"--output-dir={directory}")
        self.stale_job("foreign", runner_pid=self.dead_pid(), job_pgid=pipeline_like.pid, cancel_token_sha256=digest)
        for attempt in ({}, {"cancel_token": "wrong"}, {"job_id": "a" * 32}):
            with self.subTest(attempt=attempt):
                refused = server.dic_job_cancel(str(directory), **attempt)
                self.assertFalse(refused["ok"])
                self.assertEqual(refused["code"], "job_not_owned")
                self.assertFalse(refused["retryable"])
                self.assertEqual(refused["job"]["state"], "running")
        self.assertIsNone(pipeline_like.poll(), "a refused cancellation signalled the job")
        self.assertEqual(json.loads((directory / server.JOB_RECORD_NAME).read_text(encoding="utf-8"))["state"],
                         "running")
        cancelled = server.dic_job_cancel(str(directory), cancel_token=token)
        self.assertEqual((cancelled["action"], cancelled["state"]), ("cancelled", "cancelled"), cancelled)
        self.assertIsNotNone(pipeline_like.wait(timeout=10))
        record = json.loads((directory / server.JOB_RECORD_NAME).read_text(encoding="utf-8"))
        self.assertIn("the job runner was gone", record["cancel_reason"])
        self.assertEqual(self.audit_records()[-1]["params"]["cancel_token"], "sha256:" + digest[:12])
        self.assertNotIn(token, (self.jobs / server.AUDIT_LOG_NAME).read_text(encoding="utf-8"))

        # A blocking run of another session has no token at all.
        blocking_dir = self.jobs / "foreign-blocking"
        other = self.helper(f"--output-dir={blocking_dir}")
        self.stale_job("foreign-blocking", runner_pid=self.dead_pid(), job_pgid=other.pid)
        refused = server.dic_job_cancel(str(blocking_dir), cancel_token=token)
        self.assertEqual(refused["code"], "job_not_owned")
        self.assertIn("wait=true", refused["error"])
        self.assertIsNone(other.poll())

        # The record's pids now belong to a stranger (reused): nothing of this job is alive, so the
        # record is settled and the stranger is left alone, even for a caller with the token.
        stranger = self.helper("unrelated")
        reused = self.stale_job("reused", runner_pid=stranger.pid, job_pgid=stranger.pid, cancel_token_sha256=digest)
        settled = server.dic_job_cancel(str(reused), cancel_token=token)
        self.assertEqual((settled["action"], settled["state"]), ("settled", "job_lost"), settled)
        self.assertIsNone(stranger.poll(), "dic_job_cancel signalled a process that only reused the pid")

    def test_a_job_whose_runner_was_killed_gets_a_second_runner_and_can_be_cancelled(self) -> None:
        """Recheck R4: the job's timeout, progress and final state no longer hang on the server alone."""
        params = self.job_params("sleep-second-runner", "second-runner")
        thread, result = self.run_in_thread(params)
        record_path = self.record_path("second-runner")
        record = self.wait_for_record(record_path, lambda r: r.get("state") == "running" and r.get("job_pgid"))
        self.wait_for_binary(record)
        first_runner = record["runner_pid"]
        os.kill(first_runner, signal.SIGKILL)
        replaced = self.wait_for_record(record_path, lambda r: "runner_replaced_at" in r)
        self.assertIn("runner_lost_at", replaced)
        self.assertNotEqual(replaced["runner_pid"], first_runner)
        self.assertEqual(replaced["state"], "running")
        self.assertTrue(server._is_job_runner(replaced["runner_pid"], record_path))
        beat = replaced["progress"]["updated_at"] if "progress" in replaced else None
        later = self.wait_for_record(
            record_path, lambda r: r.get("progress", {}).get("elapsed_s", 0) > replaced.get("progress", {}).get(
                "elapsed_s", 0))  # the second runner goes on writing progress
        self.assertGreaterEqual(later["progress"]["updated_at"], beat or "")
        # A server that leaves now leaves a job that still has a runner: the ordinary mark.
        self.assertEqual(server._mark_jobs_detached("MCP server received SIGTERM (simulated)"), {record["call_id"]})
        marked = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertIn("the job keeps running", marked["server_detach_reason"])
        server._DETACH_MARKED.discard(record["call_id"])

        cancelled = server.dic_job_cancel(str(self.jobs / "second-runner"))
        thread.join(timeout=30)
        self.assertFalse(thread.is_alive())
        self.assertEqual((cancelled["action"], cancelled["state"]), ("cancelled", "cancelled"), cancelled)
        self.assertEqual(result["code"], "job_cancelled", result)
        final = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(final["cancel_reason"], f"dic_job_cancel (call {cancelled['call_id']})")
        self.assertIsNone(final["exit_code"])
        self.assert_fields_listed(final)
        self.assertTrue(_group_gone(record["job_pgid"]))
        self.assertTrue(_group_gone(replaced["runner_pid"]))
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

    def test_without_a_second_runner_the_server_waits_itself_and_says_so_when_it_leaves(self) -> None:
        """The fallback of F4, and the R4 mark: a server leaving such a job writes that nobody is left."""
        params = self.job_params("sleep-no-second", "no-second")
        thread, result = self.run_in_thread(params)
        record_path = self.record_path("no-second")
        record = self.wait_for_record(record_path, lambda r: r.get("state") == "running" and r.get("job_pgid"))
        self.wait_for_binary(record)
        # The second runner dies at once without a word: as if it could not be started.
        with mock.patch.object(server, "_JOB_RUNNER_SOURCE", "import sys; sys.exit(3)"):
            os.kill(record["runner_pid"], signal.SIGKILL)
            self.wait_for_record(record_path, lambda r: "runner_lost_at" in r)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:  # until the server holds the reaped second runner
                with server._REGISTRY_GUARD:
                    job = server._ACTIVE_JOBS.get(record["call_id"])
                if job is not None and job["runner"].pid != record["runner_pid"] and job["runner"].returncode is not None:
                    break
                time.sleep(0.05)
        self.assertNotIn("runner_replaced_at", json.loads(record_path.read_text(encoding="utf-8")))
        self.assertEqual(server._mark_jobs_detached("MCP server received SIGTERM (simulated)"), {record["call_id"]})
        marked = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(marked["state"], "running")
        self.assertIn("server_detached_at", marked)
        self.assertIn("nobody updates this record any more", marked["server_detach_reason"])
        self.assertEqual(self.audit_records()[-1]["result_code"], "detached")
        server._DETACH_MARKED.discard(record["call_id"])

        # No runner is left, so the cancellation stops the pipeline's process group directly.
        cancelled = server.dic_job_cancel(str(self.jobs / "no-second"))
        thread.join(timeout=30)
        self.assertFalse(thread.is_alive())
        self.assertEqual((cancelled["action"], cancelled["state"]), ("cancelled", "cancelled"), cancelled)
        self.assertEqual(result["code"], "job_cancelled", result)
        self.assertTrue(_group_gone(record["job_pgid"]))
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))
        self.assertIn("the job runner was gone", json.loads(record_path.read_text(encoding="utf-8"))["cancel_reason"])

    # ── review 2026-10-05 (independent review of the job model): F2, F3, the runner and final states ──

    def test_arguments_name_the_job_directory_only_exactly(self) -> None:
        """F2: '/jobs/run1' is not '/jobs/run10'. The directory itself or a path inside it, alone or
        as the value of an --option=<path>."""

        directory = b"/jobs/run1"
        ours = [
            b"python\0dic_pipeline.py\0--image-dir=/data/e\0--output-dir=/jobs/run1\0--strict",
            b"python\0dic_pipeline.py\0--json-summary=/jobs/run1/summary.json",
            b"opencorr_cli_cuda_mvp\0--output-dir\0/jobs/run1/_frames_e\0--tar-list\0/jobs/run1/_frames_e/tar_list.txt",
            b"x\0/jobs/run1",
            b"/jobs/run1\0",
        ]
        strangers = [
            b"python\0dic_pipeline.py\0--output-dir=/jobs/run10\0--strict",
            b"python\0dic_pipeline.py\0--output-dir=/jobs/run1-retry",
            b"opencorr_cli_cuda_mvp\0--output-dir\0/jobs/run1.bak/_frames_e",
            b"x\0/jobs/run12/_frames_e/tar_list.txt",
            b"x\0/data/jobs/run1",
            b"x\0--output-dir=/data/jobs/run1/",
            b"x\0note:/jobs/run1",
            b"x\0/jobs/run",
            b"",
        ]
        for cmdline in ours:
            self.assertTrue(server._argv_names_directory(cmdline, directory), cmdline)
        for cmdline in strangers:
            self.assertFalse(server._argv_names_directory(cmdline, directory), cmdline)

    @unittest.skipUnless(os.path.exists("/proc/self/stat"), "needs procfs")
    def test_a_job_in_a_directory_with_the_same_prefix_is_never_signalled(self) -> None:
        """F2: a stale record whose process group number now belongs to the job of a directory that
        merely starts with the same characters. Even the caller with the stale job's own token must
        not stop that job."""

        token = "9" * 48
        digest = hashlib.sha256(token.encode()).hexdigest()
        cases = (("run1", "run10", "--output-dir={victim}"),
                 ("job", "job-retry", "--output-dir={victim}"),
                 ("ev", "ev.2", "{victim}/_frames_e/tar_list.txt"))
        for stale_name, victim_name, argument in cases:
            with self.subTest(stale=stale_name, victim=victim_name):
                victim = self.helper(argument.format(victim=self.jobs / victim_name))
                stale = self.stale_job(stale_name, runner_pid=self.dead_pid(), job_pgid=victim.pid,
                                       cancel_token_sha256=digest)
                result = server.dic_job_cancel(str(stale), cancel_token=token)
                self.assertTrue(result["ok"], result)
                self.assertEqual((result["action"], result["state"]), ("settled", "job_lost"), result)
                self.assertEqual(result["processes"], {"runner_alive": False, "job_group_alive": False})
                self.assertIsNone(victim.poll(), "the job of another directory was signalled")
                record = json.loads((stale / server.JOB_RECORD_NAME).read_text(encoding="utf-8"))
                self.assertNotIn("cancel_requested_by", record)

    @unittest.skipUnless(os.path.exists("/proc/self/stat"), "needs procfs")
    def test_a_reused_pid_is_told_apart_by_its_start_time(self) -> None:
        """F2: pid and start time name one process for as long as the system is up. A process that
        reused the pid is not the job even when its arguments name the job's directory."""

        token = "8" * 48
        digest = hashlib.sha256(token.encode()).hexdigest()
        directory = self.jobs / "ticks"
        impostor = self.helper(f"--output-dir={directory}")
        ticks = server._proc_start_ticks(impostor.pid)
        self.assertIsInstance(ticks, int)
        self.stale_job("ticks", runner_pid=self.dead_pid(), job_pgid=impostor.pid, job_start_ticks=ticks + 1,
                       cancel_token_sha256=digest)
        settled = server.dic_job_cancel(str(directory), cancel_token=token)
        self.assertEqual((settled["action"], settled["state"]), ("settled", "job_lost"), settled)
        self.assertIsNone(impostor.poll(), "a process with another start time was signalled")

        own_directory = self.jobs / "ticks-own"
        own = self.helper(f"--output-dir={own_directory}")
        self.stale_job("ticks-own", runner_pid=self.dead_pid(), job_pgid=own.pid,
                       job_start_ticks=server._proc_start_ticks(own.pid), cancel_token_sha256=digest)
        cancelled = server.dic_job_cancel(str(own_directory), cancel_token=token)
        self.assertEqual((cancelled["action"], cancelled["state"]), ("cancelled", "cancelled"), cancelled)
        self.assertIsNotNone(own.wait(timeout=10))

        # The runner: its arguments name the record, and its start time is in the record.
        record_path = self.jobs / "ticks-runner" / server.JOB_RECORD_NAME
        runner_like = self.helper(json.dumps({"record": str(record_path)}))
        actual = server._proc_start_ticks(runner_like.pid)
        self.assertTrue(server._is_job_runner(runner_like.pid, record_path, actual))
        self.assertTrue(server._is_job_runner(runner_like.pid, record_path, None))  # a record without start times
        self.assertFalse(server._is_job_runner(runner_like.pid, record_path, actual + 1))
        self.assertFalse(server._is_job_runner(runner_like.pid, self.jobs / "ticks-runner2" / server.JOB_RECORD_NAME,
                                               actual))
        self.assertFalse(server._is_job_runner(self.dead_pid(), record_path, actual))

        # Asked the instant after a process was started, while it may still be in exec and its
        # arguments read as empty: it is recognised all the same.
        for index in range(10):
            fresh_directory = self.jobs / f"fresh-{index}"
            fresh = self.helper(f"--output-dir={fresh_directory}")
            self.assertTrue(server._job_group_alive(fresh.pid, fresh_directory, server._proc_start_ticks(fresh.pid)))
            fresh.kill()

    def test_cancel_while_the_runner_is_being_replaced_is_prompt_and_says_cancelled(self) -> None:
        """F3: the first runner was killed and the server has not taken the job over yet (held here
        for as long as the cancellation runs). It used to wait CANCEL_WAIT_SECONDS (20 s) for a
        runner that was no longer there."""

        entered, gate = threading.Event(), threading.Event()
        real_adoption = server._adopt_job_of_lost_runner

        def held_adoption(*args, **kwargs):
            entered.set()
            gate.wait(60)
            return real_adoption(*args, **kwargs)

        record_path = self.record_path("window")
        with mock.patch.object(server, "_adopt_job_of_lost_runner", held_adoption):
            thread, result = self.run_in_thread(self.job_params("sleep-window", "window"))
            record = self.wait_for_record(record_path, lambda r: r.get("state") == "running" and r.get("job_pgid"))
            self.wait_for_binary(record)
            os.kill(record["runner_pid"], signal.SIGKILL)
            self.assertTrue(entered.wait(10), "the server never noticed that the runner was gone")
            started = time.monotonic()
            cancelled = server.dic_job_cancel(str(self.jobs / "window"))
            took = time.monotonic() - started
            gate.set()
            thread.join(timeout=30)
        self.assertFalse(thread.is_alive())
        self.assertLess(took, 10.0, "the cancellation waited for a runner that was gone")
        self.assertEqual((cancelled["action"], cancelled["state"]), ("cancelled", "cancelled"), cancelled)
        self.assertEqual(cancelled["processes"], {"runner_alive": False, "job_group_alive": False})
        final = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(final["state"], "cancelled")
        self.assertIn(f"dic_job_cancel (call {cancelled['call_id']})", final["cancel_reason"])
        self.assertEqual(result["code"], "job_cancelled", result)
        self.assertTrue(_group_gone(record["job_pgid"]))
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

    def test_cancel_while_the_killed_runner_is_still_a_zombie(self) -> None:
        """F3, the first instants: the runner was killed and the server thread has not collected it
        yet (held here: it does not look at its runner until released). A signal to that pid goes
        nowhere; the cancellation must see that the runner has ended and stop the job itself."""

        looking = threading.Event()
        real_wait = subprocess.Popen.wait

        def held_wait(process, timeout=None):
            if list(process.args[:3]) == [sys.executable, "-c", server._JOB_RUNNER_SOURCE]:
                looking.wait(60)
            return real_wait(process, timeout=timeout)

        record_path = self.record_path("zombie")
        with mock.patch.object(subprocess.Popen, "wait", held_wait):
            thread, result = self.run_in_thread(self.job_params("sleep-zombie", "zombie"))
            record = self.wait_for_record(record_path, lambda r: r.get("state") == "running" and r.get("job_pgid"))
            self.wait_for_binary(record)
            os.kill(record["runner_pid"], signal.SIGKILL)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:  # dead, and nobody has collected it: a zombie
                state = Path(f"/proc/{record['runner_pid']}/stat").read_bytes().rsplit(b") ", 1)[1][:1] \
                    if os.path.exists(f"/proc/{record['runner_pid']}/stat") else b"Z"
                if state == b"Z":
                    break
                time.sleep(0.01)
            started = time.monotonic()
            cancelled = server.dic_job_cancel(str(self.jobs / "zombie"))
            took = time.monotonic() - started
            looking.set()
            thread.join(timeout=30)
        self.assertFalse(thread.is_alive())
        self.assertLess(took, 10.0, "the cancellation waited for a runner that was dead")
        self.assertEqual((cancelled["action"], cancelled["state"]), ("cancelled", "cancelled"), cancelled)
        self.assertNotIn("forced", cancelled["note"])
        final = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(final["state"], "cancelled")
        self.assertEqual(result["code"], "job_cancelled", result)
        self.assertTrue(_group_gone(record["job_pgid"]))
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

    def test_a_second_runner_records_a_requested_cancellation_as_cancelled(self) -> None:
        """F3: a cancellation that stops the job's process group itself (the runner it signalled was
        gone) while a second runner watches. That runner used to write job_lost: 'nothing left, no summary'."""

        record_path = self.record_path("requested")
        thread, result = self.run_in_thread(self.job_params("sleep-requested", "requested"))
        record = self.wait_for_record(record_path, lambda r: r.get("state") == "running" and r.get("job_pgid"))
        self.wait_for_binary(record)
        os.kill(record["runner_pid"], signal.SIGKILL)
        self.wait_for_record(record_path, lambda r: "runner_replaced_at" in r)
        request = {"call_id": "f" * 32, "server_pid": os.getpid(), "at": "2026-10-05T00:00:00Z"}
        server._merge_job_record(record_path, cancel_requested_at=request["at"], cancel_requested_by=request)
        os.killpg(record["job_pgid"], signal.SIGKILL)
        thread.join(timeout=30)
        self.assertFalse(thread.is_alive())
        final = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(final["state"], "cancelled", final)
        self.assertEqual(final["cancel_reason"], "dic_job_cancel (call %s)" % ("f" * 32))
        self.assertIsNone(final["exit_code"])
        self.assertEqual(result["code"], "job_cancelled", result)
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

    def test_cancel_right_after_the_runner_was_killed(self) -> None:
        """F3, the review's e13: SIGKILL the runner and cancel 0 or 30 ms later, while it is a zombie,
        just reaped or being replaced. Always 'cancelled', never after a 20 s wait, never job_lost."""

        for index, delay in enumerate((0.0, 0.03)):
            with self.subTest(delay=delay):
                name = f"killed-{index}"
                record_path = self.record_path(name)
                thread, result = self.run_in_thread(self.job_params(f"sleep-killed-{index}", name))
                record = self.wait_for_record(record_path,
                                              lambda r: r.get("state") == "running" and r.get("job_pgid"))
                self.wait_for_binary(record)
                os.kill(record["runner_pid"], signal.SIGKILL)
                time.sleep(delay)
                started = time.monotonic()
                cancelled = server.dic_job_cancel(str(self.jobs / name))
                took = time.monotonic() - started
                thread.join(timeout=30)
                self.assertFalse(thread.is_alive())
                self.assertLess(took, 10.0)
                self.assertEqual((cancelled["action"], cancelled["state"]), ("cancelled", "cancelled"), cancelled)
                final = json.loads(record_path.read_text(encoding="utf-8"))
                self.assertEqual(final["state"], "cancelled", final)
                self.assertEqual(result["code"], "job_cancelled", result)
                self.assertTrue(_group_gone(record["job_pgid"]))
                self.assertTrue(_group_gone(final["runner_pid"]))
                self.wait_until_idle()
                self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

    def test_a_runner_that_does_not_stop_the_job_is_killed_with_it(self) -> None:
        """The last resort of dic_job_cancel: after CANCEL_WAIT_SECONDS the runner that watches the job
        at that moment (not a pid read earlier) and the job's group get SIGKILL."""

        deaf = server._JOB_RUNNER_SOURCE.replace("    stop_signal = signum\n", "    pass\n")
        self.assertNotEqual(deaf, server._JOB_RUNNER_SOURCE)
        record_path = self.record_path("deaf")
        with mock.patch.object(server, "_JOB_RUNNER_SOURCE", deaf), \
                mock.patch.object(server, "CANCEL_WAIT_SECONDS", 1.0):
            thread, result = self.run_in_thread(self.job_params("sleep-deaf", "deaf"))
            record = self.wait_for_record(record_path, lambda r: r.get("state") == "running" and r.get("job_pgid"))
            self.wait_for_binary(record)
            os.kill(record["runner_pid"], signal.SIGKILL)  # the second runner is deaf too
            replaced = self.wait_for_record(record_path, lambda r: "runner_replaced_at" in r)
            started = time.monotonic()
            cancelled = server.dic_job_cancel(str(self.jobs / "deaf"))
            took = time.monotonic() - started
            thread.join(timeout=30)
        self.assertFalse(thread.is_alive())
        self.assertGreaterEqual(took, 0.9)
        self.assertLess(took, 10.0)
        self.assertEqual((cancelled["action"], cancelled["state"]), ("cancelled", "cancelled"), cancelled)
        self.assertIn("forced with SIGKILL", cancelled["note"])
        final = json.loads(record_path.read_text(encoding="utf-8"))
        # Who writes the record first, this call or the server thread that watched the job, is open;
        # both write cancelled and name the call that asked for it.
        self.assertEqual(final["state"], "cancelled")
        self.assertTrue(final["cancel_reason"].startswith(f"dic_job_cancel (call {cancelled['call_id']})"), final)
        self.assertTrue(_group_gone(replaced["runner_pid"]), "the second runner survived")
        self.assertTrue(_group_gone(record["job_pgid"]))
        self.assertEqual(result["code"], "job_cancelled", result)
        self.wait_until_idle()
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

    def start_runner(self, name: str, argv: list[str], spec_extra: dict | None = None,
                     **record) -> tuple[subprocess.Popen, Path]:
        """The job runner alone, as the server starts it, for a record written here."""
        directory = self.jobs / name
        directory.mkdir(parents=True)
        record_path = directory / server.JOB_RECORD_NAME
        record_path.write_text(json.dumps({"schema": server.JOB_RECORD_SCHEMA, "call_id": "b" * 32, **record}),
                               encoding="utf-8")
        spec = {"record": str(record_path), "output_dir": str(directory), "argv": argv, "timeout_s": 30,
                "stdout_log": str(directory / server.JOB_STDOUT_NAME),
                "stderr_log": str(directory / server.JOB_STDERR_NAME), "summary": str(directory / "summary.json"),
                "lock_fds": [], "progress_interval_s": 0.05, "progress": None, **(spec_extra or {})}
        runner = subprocess.Popen([sys.executable, "-c", server._JOB_RUNNER_SOURCE, json.dumps(spec)],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                  start_new_session=True)
        self.addCleanup(lambda: runner.stderr.close())
        self.addCleanup(runner.wait)
        self.addCleanup(lambda: runner.poll() is None and runner.kill())
        return runner, record_path

    @unittest.skipUnless(os.path.exists("/proc/self/stat"), "needs procfs")
    def test_the_second_runner_only_stops_a_group_that_is_its_job(self) -> None:
        """Review 2026-10-05 ('unconfirmed' 3, the place this branch added): the second runner got a
        process group number from the record and signalled it as it was."""

        # The number belongs to a stranger now: the runner sees no job, and a SIGTERM stops nobody.
        stranger = self.helper("unrelated")
        runner, record_path = self.start_runner(
            "adopt-stranger", [], spec_extra={"adopt_pgid": stranger.pid, "adopt_start_ticks": None},
            state="running", job_pgid=stranger.pid)
        self.assertEqual(runner.wait(timeout=15), 0)
        final = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(final["state"], "job_lost", final)
        self.assertIsNone(stranger.poll(), "the second runner signalled a process that is not its job")

        # The same with the right arguments but another start time (a reused pid).
        directory = self.jobs / "adopt-reused"
        impostor = self.helper(f"--output-dir={directory}")
        ticks = server._proc_start_ticks(impostor.pid)
        runner, record_path = self.start_runner(
            "adopt-reused", [], spec_extra={"adopt_pgid": impostor.pid, "adopt_start_ticks": ticks + 1},
            state="running", job_pgid=impostor.pid, job_start_ticks=ticks + 1)
        self.assertEqual(runner.wait(timeout=15), 0)
        self.assertEqual(json.loads(record_path.read_text(encoding="utf-8"))["state"], "job_lost")
        self.assertIsNone(impostor.poll())

        # Its own job: watched, and stopped when the runner is told to.
        directory = self.jobs / "adopt-own"
        job = self.helper(f"--output-dir={directory}")
        ticks = server._proc_start_ticks(job.pid)
        runner, record_path = self.start_runner(
            "adopt-own", [], spec_extra={"adopt_pgid": job.pid, "adopt_start_ticks": ticks},
            state="running", job_pgid=job.pid, job_start_ticks=ticks)
        watching = self.wait_for_record(record_path, lambda r: "runner_replaced_at" in r)
        self.assertEqual(watching["runner_start_ticks"], server._proc_start_ticks(watching["runner_pid"]))
        self.assertIsNone(job.poll())
        os.kill(runner.pid, signal.SIGTERM)
        self.assertEqual(runner.wait(timeout=15), 128 + signal.SIGTERM)
        self.assertIsNotNone(job.wait(timeout=10))
        final = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual((final["state"], final["cancel_reason"]), ("cancelled", "job runner received SIGTERM"))

    def test_the_runner_never_overwrites_a_final_state(self) -> None:
        """Review 2026-10-05, 'unconfirmed' 1: a record settled as job_lost between its creation and
        the runner's start was set back to running by the runner. The first final state stays."""

        # (a) Final before the runner starts: the pipeline is not started at all.
        marker = self.root / "pipeline-ran"
        touch = [sys.executable, "-c", "import sys, time; open(sys.argv[1], 'w').close(); time.sleep(0.3)",
                 str(marker)]
        runner, record_path = self.start_runner("settled-first", touch, state="job_lost", settled_by="somebody")
        self.assertNotEqual(runner.wait(timeout=15), 0)
        self.assertFalse(marker.exists(), "the pipeline was started for a job that was already settled")
        # It left before it prepared anything: not even the pipeline's log files were opened.
        self.assertFalse((record_path.parent / server.JOB_STDOUT_NAME).exists())
        self.assertFalse((record_path.parent / server.JOB_STDERR_NAME).exists())
        self.assertEqual(json.loads(record_path.read_text(encoding="utf-8")),
                         {"schema": server.JOB_RECORD_SCHEMA, "call_id": "b" * 32, "state": "job_lost",
                          "settled_by": "somebody"})
        self.assertIn("job_lost", runner.stderr.read().decode())

        # (b) Final while the job runs (somebody else recorded the end): not overwritten at the end.
        nap = [sys.executable, "-c", "import time; time.sleep(1.0)"]
        runner, record_path = self.start_runner("settled-meanwhile", nap, state="starting")
        self.wait_for_record(record_path, lambda r: r.get("state") == "running")
        server._merge_job_record(record_path, state="timed_out", note="recorded by somebody else")
        runner.wait(timeout=15)
        final = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual((final["state"], final["note"]), ("timed_out", "recorded by somebody else"))
        self.assertNotIn("exit_code", final)

        # (c) The job was stopped by a signal after dic_job_cancel had asked for it (the cancellation
        # stopped the process group itself): cancelled, not "exited with -15".
        sleep = [sys.executable, "-c", "import time; time.sleep(60)"]
        runner, record_path = self.start_runner("stopped-directly", sleep, state="starting")
        running = self.wait_for_record(record_path, lambda r: r.get("state") == "running" and r.get("job_pgid"))
        request = {"call_id": "e" * 32, "server_pid": os.getpid(), "at": "2026-10-05T00:00:00Z"}
        server._merge_job_record(record_path, cancel_requested_at=request["at"], cancel_requested_by=request)
        os.killpg(running["job_pgid"], signal.SIGTERM)
        runner.wait(timeout=15)
        final = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual((final["state"], final["exit_code"]), ("cancelled", -signal.SIGTERM), final)
        self.assertEqual(final["cancel_reason"], "dic_job_cancel (call %s)" % ("e" * 32))

        # (d) Without such a request a job that a signal ended is what it was before: exited.
        runner, record_path = self.start_runner("stopped-by-somebody", sleep, state="starting")
        running = self.wait_for_record(record_path, lambda r: r.get("state") == "running" and r.get("job_pgid"))
        os.killpg(running["job_pgid"], signal.SIGTERM)
        runner.wait(timeout=15)
        final = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual((final["state"], final["exit_code"]), ("exited", -signal.SIGTERM), final)

    def test_without_a_second_runner_the_server_still_enforces_the_timeout(self) -> None:
        """The fallback behind the second runner (review 2026-10-04 F4, R5): both runners gone, the
        server itself stops the job at its deadline."""

        params = self.job_params("sleep-no-second-deadline", "no-second-deadline", timeout=4)
        record_path = self.record_path("no-second-deadline")
        started = time.monotonic()
        thread, result = self.run_in_thread(params)
        record = self.wait_for_record(record_path, lambda r: r.get("state") == "running" and r.get("job_pgid"))
        self.wait_for_binary(record)
        with mock.patch.object(server, "_JOB_RUNNER_SOURCE", "import sys; sys.exit(3)"):
            os.kill(record["runner_pid"], signal.SIGKILL)
            thread.join(timeout=30)  # the fake binary sleeps 60 s
        self.assertFalse(thread.is_alive(), "the job ran past its timeout")
        self.assertEqual(result["code"], "timeout", result)
        self.assertLess(time.monotonic() - started, 20)
        final = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(final["state"], "timed_out")
        self.assertIn("runner_lost_at", final)
        self.assertNotIn("runner_replaced_at", final)
        self.assertTrue(_group_gone(record["job_pgid"]))
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

    # ── review 2026-10-05 F4: the record says exited a moment before the locks are free ──

    def ended_job(self, name: str, call_id: str, state: str = "exited", root: Path | None = None) -> dict:
        """A job directory whose record holds ``state``, and the lock-holder entry of that job."""
        directory = (root or self.jobs) / name
        directory.mkdir(parents=True)
        (directory / server.JOB_RECORD_NAME).write_text(json.dumps({
            "schema": server.JOB_RECORD_SCHEMA, "state": state, "call_id": call_id, "exit_code": 0}), encoding="utf-8")
        return {"tool": "dic_run_event", "call_id": call_id, "output_dir": str(directory)}

    def hold_gpu_lock(self, holder: dict, seconds: float) -> threading.Thread:
        """Another thread of this server keeps the GPU lock for a while, as the thread that waited
        for a job does between the runner's exit and its own return."""
        taken = threading.Event()

        def hold() -> None:
            with server._gpu_lock(0, holder):
                taken.set()
                time.sleep(seconds)

        thread = threading.Thread(target=hold)
        thread.start()
        self.assertTrue(taken.wait(5))
        self.addCleanup(thread.join)
        return thread

    def test_a_gpu_whose_holder_has_recorded_its_end_is_waited_for(self) -> None:
        """F4: a job submitted the moment the last one's record said exited met gpu_busy (10 of 10
        in the review): the lock is released some tens of milliseconds after the record is written."""

        # Held by a thread of this server whose job has ended: the run waits for the lock.
        thread = self.hold_gpu_lock(self.ended_job("ended-here", "1" * 32), 0.4)
        started = time.monotonic()
        run = server.dic_run_event(self.job_params("after-ended-here", "after-ended-here"))
        waited = time.monotonic() - started
        thread.join()
        self.assertEqual(run.get("verdict"), "ok", run)
        self.assertGreaterEqual(waited, 0.3)

        # Held by another process (the server of another session) whose job has ended.
        holder = self.ended_job("ended-elsewhere", "2" * 32)
        lock_path = server._gpu_lock_path(0)
        script = textwrap.dedent(f"""\
            import fcntl, json, os, sys, time
            fd = os.open({str(lock_path)!r}, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.ftruncate(fd, 0)
            os.write(fd, json.dumps({{**json.loads(sys.argv[1]), "server_pid": os.getpid(),
                                      "acquired_epoch": time.time()}}).encode())
            print("locked", flush=True)
            time.sleep(0.4)
        """)
        other = subprocess.Popen([sys.executable, "-c", script, json.dumps(holder)], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(other.stdout.readline().strip(), "locked")
            run = server.dic_run_event(self.job_params("after-ended-elsewhere", "after-ended-elsewhere"))
        finally:
            other.wait(timeout=10)
            other.stdout.close()
        self.assertEqual(run.get("verdict"), "ok", run)

        # A holder whose job is still running: refused at once, as before.
        thread = self.hold_gpu_lock(self.ended_job("still-running", "3" * 32, state="running"), 0.8)
        started = time.monotonic()
        busy = server.dic_run_event(self.job_params("while-running", "while-running"))
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(busy["code"], "gpu_busy")
        self.assertEqual(busy["holder"]["call_id"], "3" * 32)
        self.assertFalse((self.jobs / "while-running").exists())
        thread.join()

        # The holder's job has ended but the lock does not come free: gpu_busy after the bounded wait.
        with mock.patch.object(server, "ENDED_HOLDER_WAIT_SECONDS", 0.5):
            thread = self.hold_gpu_lock(self.ended_job("ended-stuck", "4" * 32), 1.5)
            started = time.monotonic()
            busy = server.dic_run_event(self.job_params("stuck", "stuck"))
            waited = time.monotonic() - started
            thread.join()
        self.assertEqual(busy["code"], "gpu_busy")
        self.assertTrue(busy["retryable"])
        self.assertGreaterEqual(waited, 0.45)
        self.assertLess(waited, 1.4)

        # A lock entry that points outside the job root, or at another job's record, is not waited for.
        for holder in (self.ended_job("outside-job", "5" * 32, root=self.outside),
                       {**self.ended_job("other-id", "6" * 32), "call_id": "7" * 32}):
            thread = self.hold_gpu_lock(holder, 0.6)
            started = time.monotonic()
            busy = server.dic_run_event(self.job_params("no-wait", "no-wait"))
            self.assertLess(time.monotonic() - started, 0.4)
            self.assertEqual(busy["code"], "gpu_busy")
            thread.join()

    def test_jobs_submitted_back_to_back_do_not_meet_gpu_busy(self) -> None:
        """F4, the review's e17: start the next wait=false job the moment the record says exited."""

        for index in range(6):
            name = f"back-to-back-{index}"
            handle = server.dic_run_event(self.job_params("back-to-back", name, frames=2, wait=False))
            self.assertTrue(handle.get("ok"), handle)
            record_path = self.record_path(name)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                with contextlib.suppress(OSError, ValueError):
                    if json.loads(record_path.read_text(encoding="utf-8")).get("state") == "exited":
                        break
                time.sleep(0.001)
        self.wait_until_idle()

    def test_campaign_progress_counts_events_and_frames(self) -> None:
        campaign = self.data / "tick-campaign"
        for index in range(3):
            _write_event(campaign / f"tick{index}", frames=3)
        handle = server.dic_run_campaign({
            "photo_dir": str(campaign), "output_dir": str(self.jobs / "campaign"), "line_y": [10, 20], "cal": 1.0,
            "wait": False})
        self.assertTrue(handle["ok"], handle)
        seen: list[dict] = []
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            record = self.poll("campaign")
            if "progress" in record:
                seen.append(dict(record["progress"], state=record["state"]))
            if record.get("state") not in {"starting", "running"}:
                break
            time.sleep(0.05)
        final = self.poll("campaign")
        self.assertEqual((final["state"], final["exit_code"], final["verdict"]), ("exited", 0, "ok"), final)
        self.assertEqual((final["events_ok"], final["events_failed"]), (3, 0))
        self.assertEqual(final["progress"]["events_done"], 3)
        self.assertEqual(final["progress"]["frames_done"], 6)
        self.assertTrue(all((item["events_total"], item["frames_total"]) == (3, 6) for item in seen))
        events_done = [item["events_done"] for item in seen]
        frames_done = [item["frames_done"] for item in seen]
        self.assertEqual(events_done, sorted(events_done))
        self.assertEqual(frames_done, sorted(frames_done))
        self.assertTrue({1, 2} & set(events_done), events_done)  # the middle of the campaign was seen
        self.assertTrue({"tick0", "tick1", "tick2"} & {item.get("current_event") for item in seen})
        # keep_frames=false as usual: the frame directories are gone (their binary logs stay).
        self.assertFalse(any(path.is_dir() for path in (self.jobs / "campaign").glob("_frames_*")))
        self.wait_until_idle()

    def test_progress_counting_from_the_files_of_a_campaign(self) -> None:
        """The runner's counting, step by step, on the files a campaign leaves (no job is run).

        Seen on the real 30 kN campaign: the pipeline deletes an event's frame directory before it
        publishes the COD CSV; in between the frame count must not fall back to 0.
        """

        output = self.root / "progress-files"
        output.mkdir()
        spec = {"record": str(output / server.JOB_RECORD_NAME), "output_dir": str(output), "progress": {
            "token": server.PROGRESS_EVENT_TOKEN, "frames_dir": f"_frames_{server.PROGRESS_EVENT_TOKEN}",
            "cod_csv": f"{server.PROGRESS_EVENT_TOKEN}.csv",
            "binary_log": f"_frames_{server.PROGRESS_EVENT_TOKEN}.binary.log",
            "events": [["e1", 3], ["e2", 3], ["e3", 3]]}}
        script = server._JOB_RUNNER_LIB + textwrap.dedent("""
            import shutil
            out = spec["output_dir"]
            shots = []

            def shot(label, ended=False):
                shots.append(dict(snapshot(time.monotonic(), ended=ended), label=label))

            def frames(event, count):
                os.makedirs(os.path.join(out, "_frames_" + event), exist_ok=True)
                for number in range(2, 2 + count):
                    open(os.path.join(out, "_frames_" + event, "frame_%06d.csv" % number), "w").close()
                open(os.path.join(out, "_frames_" + event, "frame_000099.csv.tmp"), "w").close()
                open(os.path.join(out, "_frames_" + event, "tar_list.txt"), "w").close()

            shot("nothing yet")
            open(os.path.join(out, "_frames_e1.binary.log"), "w").close()
            shot("e1 binary started")
            frames("e1", 2)
            shot("e1 two frames")
            frames("e1", 3)
            shot("e1 all frames")
            shutil.rmtree(os.path.join(out, "_frames_e1"))
            shot("e1 frames deleted, csv not yet published")
            open(os.path.join(out, "e1.csv"), "w").close()
            shot("e1 published")
            open(os.path.join(out, "_frames_e2.binary.log"), "w").close()
            frames("e2", 1)
            shot("e2 one frame, then its binary fails")
            open(os.path.join(out, "_frames_e3.binary.log"), "w").close()
            frames("e3", 3)
            shot("e3 running: e2 is over without a csv")
            shutil.rmtree(os.path.join(out, "_frames_e3"))
            open(os.path.join(out, "e3.csv"), "w").close()
            shot("the pipeline ended", ended=True)
            print(json.dumps(shots))
        """)
        completed = subprocess.run([sys.executable, "-c", script, json.dumps(spec)], capture_output=True, text=True,
                                   timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        shots = {shot["label"]: shot for shot in json.loads(completed.stdout)}
        rows = {label: (shot["events_done"], shot["frames_done"], shot["current_event"],
                        shot.get("current_event_frames_done")) for label, shot in shots.items()}
        self.assertEqual(rows, {
            "nothing yet": (0, 0, None, None),
            "e1 binary started": (0, 0, "e1", 0),
            "e1 two frames": (0, 2, "e1", 2),
            "e1 all frames": (0, 3, "e1", 3),
            "e1 frames deleted, csv not yet published": (0, 3, "e1", 3),
            "e1 published": (1, 3, None, None),
            "e2 one frame, then its binary fails": (1, 4, "e2", 1),
            "e3 running: e2 is over without a csv": (2, 7, "e3", 3),  # e2 counts the one frame it wrote
            "the pipeline ended": (3, 7, None, None),
        })
        self.assertTrue(all((shot["events_total"], shot["frames_total"]) == (3, 9) for shot in shots.values()))
        self.assertTrue(all("error" not in shot for shot in shots.values()))

    def test_the_runner_marks_an_external_sigterm_as_cancelled_without_a_caller(self) -> None:
        """SIGTERM to the runner from outside (an operator) still ends the job cleanly."""
        thread, result = self.run_in_thread(self.job_params("sleep-external", "external"))
        record = self.wait_for_record(self.record_path("external"),
                                      lambda r: r.get("state") == "running" and r.get("job_pgid"))
        os.killpg(record["runner_pid"], signal.SIGTERM)
        thread.join(timeout=30)
        self.assertEqual(result["code"], "job_cancelled", result)
        final = json.loads(self.record_path("external").read_text(encoding="utf-8"))
        self.assertEqual(final["cancel_reason"], "job runner received SIGTERM")
        self.assertTrue(_group_gone(record["job_pgid"]))
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))


class LegacyLabHomeTests(unittest.TestCase):
    """DIC_HOME 的模块级默认值（McpGuardrailTests 会把它 patch 掉，故单开一类）。"""

    def test_module_level_dic_home_matches_the_real_filesystem(self) -> None:
        legacy = Path(server.pipeline.LEGACY_LAB_HOME)
        if legacy.is_dir():
            self.assertEqual(server.DIC_HOME, legacy)
        else:
            self.assertIsNone(server.DIC_HOME)


class ContractBindingTests(unittest.TestCase):
    def test_pipeline_param_keys_match_the_contract(self) -> None:
        contract = json.loads(server.CONTRACT_PATH.read_text(encoding="utf-8"))
        run_params = contract["mcp"]["run_params"]
        self.assertEqual(set(run_params["allowed_pipeline_keys"]), server.PIPELINE_PARAM_KEYS)
        self.assertEqual(set(run_params["control_keys"]), server.CONTROL_PARAM_KEYS)
        self.assertEqual(set(server.PARAM_SPECS), server.PIPELINE_PARAM_KEYS | server.CONTROL_PARAM_KEYS)

    def test_module_import_was_hermetic(self) -> None:
        self.assertEqual(server.JOB_ROOT, Path(_HERMETIC.name) / "import-job-root")
        self.assertFalse(server.GPU_BIN_SOURCE.startswith("config:"))

    def test_the_long_jobs_paragraph_comes_before_a_client_would_cut_the_instructions(self) -> None:
        """Review 2026-10-05 F6: some MCP clients show only about the first 2000 characters of a
        server's instructions. How to use wait=false and the warning not to cancel other people's
        jobs must be inside that part (they started at character 2124)."""

        text = server.SERVER_INSTRUCTIONS
        warning = "Never cancel a job you did not start"
        paragraph = text[text.index("Long jobs:"):text.index("Traps:")]
        for phrase in ("wait=false", "cancel_token", "dic_query_manifest(run_dir=output_dir", ".mcp-job.json",
                       "dic_job_cancel(output_dir)", warning, "gpu_busy"):
            self.assertIn(phrase, paragraph)
        self.assertLessEqual(text.index("Traps:"), 1500)  # the whole paragraph ends before character 1500
        self.assertLessEqual(text.index(warning) + len(warning), 1500)
        self.assertLess(text.index("Workflow:"), text.index("Long jobs:"))
        self.assertLessEqual(len(text), 3000)
        contract = json.loads(server.CONTRACT_PATH.read_text(encoding="utf-8"))
        self.assertIn("first 1500 characters", contract["mcp"]["instructions"])

    def test_job_record_fields_in_the_contract_cover_what_is_written(self) -> None:
        """Review 2026-10-05 F7: the contract's list of job record fields had gaps."""

        contract = json.loads(server.CONTRACT_PATH.read_text(encoding="utf-8"))
        listed = {entry.split(" ", 1)[0] for entry in contract["mcp"]["job_record"]["fields"]}
        for name in ("schema", "states", "stdout_log", "stderr_log", "error", "pipeline_argv", "wait", "deadline_at",
                     "runner_start_ticks", "job_start_ticks", "cancel_token_sha256", "cancel_requested_by",
                     "server_detached_at", "runner_lost_at", "settled_by"):
            self.assertIn(name, listed)
        # McpJobModelTests.assert_fields_listed checks the records of real jobs against the same list.

    def test_cod_um_is_described_as_the_slip_across_the_interface(self) -> None:
        """2026-10-05: cod_um is a slip (relative displacement of the two sides, L0 side minus L1
        side), not an opening. Names (cod_um, cod_method, COD) are unchanged; the sign is stated by
        the formula only."""

        contract = json.loads(server.CONTRACT_PATH.read_text(encoding="utf-8"))
        column = contract["outputs"]["cod_csv"]["columns"][2]
        self.assertEqual(column["name"], "cod_um")
        self.assertTrue(column["meaning"].startswith("slip across the interface"), column["meaning"][:80])
        self.assertIn("not an opening", column["meaning"])
        self.assertIn("1000 * cal * (u_L0 - u_L1)", column["meaning"])
        self.assertNotIn("opening or slip", json.dumps(contract))
        self.assertIn("slip", contract["mcp"]["results"]["cod_definition"])
        definition = server._cod_definition([10, 20], "naive")
        self.assertEqual(definition["formula"], "cod_um = (u_L0 - u_L1) * cal * 1000")
        self.assertIn("slip across the interface", definition["sign_convention_note"])
        self.assertIn("positive when u_L0 > u_L1", definition["sign_convention_note"])
        self.assertIn("slip across the interface", server.SERVER_INSTRUCTIONS)
        self.assertNotIn("opening displacement (COD", server.SERVER_INSTRUCTIONS)
        self.assertNotIn("crack", server.SERVER_INSTRUCTIONS)
        self.assertIn("cod_method", server.PARAM_SPECS)  # the parameter keeps its name
        help_text = server.pipeline.create_parser().format_help()
        self.assertIn("滑移量", help_text)
        self.assertNotIn("张开位移", help_text)


@unittest.skipIf(server.FastMCP is None, "fastmcp is tested in the production venv")
class FastMcpRegistrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_nine_tools_are_registered(self) -> None:
        """Eight until 2026-10-05; dic_job_cancel is the ninth and comes last, so the first eight keep their order."""
        tools = await server.mcp.list_tools()
        names = [tool.name for tool in tools]
        expected = [function.__name__ for function in server.TOOL_FUNCTIONS]
        contract = json.loads(server.CONTRACT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(names, expected)
        self.assertEqual(names, contract["mcp"]["tool_names"])
        self.assertEqual(contract["mcp"]["tool_count"], 9)
        self.assertEqual(names[:8], ["dic_describe", "dic_get_pitfalls", "dic_probe_images", "dic_plan",
                                     "dic_run_event", "dic_run_campaign", "dic_aggregate_cod", "dic_query_manifest"])
        self.assertEqual(names[8], "dic_job_cancel")

    async def test_tools_list_publishes_typed_documented_schemas(self) -> None:
        from fastmcp import Client

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, "JOB_ROOT", Path(tmp) / "jobs"):
            async with Client(server.mcp) as client:
                listing = await client.list_tools_mcp()
                info = client.initialize_result
        tools = {tool.name: tool for tool in listing.tools}
        self.assertEqual(info.serverInfo.version, server.SERVER_VERSION)
        self.assertEqual(info.instructions, server.SERVER_INSTRUCTIONS)
        self.assertRegex(server.SERVER_VERSION, r"^\d+\.\d+")
        for phrase in ("dic_probe_images", "dic_plan", "SIZE", "input_align='left'", "gpu_busy"):
            self.assertIn(phrase, info.instructions)
        for name, tool in tools.items():
            with self.subTest(tool=name):
                self.assertGreaterEqual(len(re.findall(r"[.!?](\s|$)", tool.description)), 2, tool.description)
                self.assertTrue(tool.title)
                self.assertIsNotNone(tool.annotations)
                for prop_name, prop in tool.inputSchema.get("properties", {}).items():
                    if prop_name != "params":
                        self.assertTrue(prop.get("description"), f"{name}.{prop_name}")
        for name in ("dic_describe", "dic_get_pitfalls", "dic_probe_images", "dic_plan", "dic_query_manifest"):
            self.assertTrue(tools[name].annotations.readOnlyHint, name)
        self.assertFalse(tools["dic_run_event"].annotations.readOnlyHint)
        self.assertTrue(tools["dic_aggregate_cod"].annotations.destructiveHint)
        # 2026-10-05: the job model. wait is a documented control key of the three params tools
        # (default true = block, as before); dic_job_cancel takes the job's output_dir.
        cancel = tools["dic_job_cancel"]
        self.assertFalse(cancel.annotations.readOnlyHint)
        self.assertTrue(cancel.annotations.destructiveHint)
        self.assertTrue(cancel.annotations.idempotentHint)
        self.assertEqual(cancel.inputSchema["required"], ["output_dir"])
        self.assertEqual(list(cancel.inputSchema["properties"]), ["output_dir", "job_id", "cancel_token"])
        for name in server.PARAMS_TOOLS:
            wait = tools[name].inputSchema["properties"]["params"]["properties"]["wait"]
            self.assertEqual((wait["type"], wait["default"]), ("boolean", True), name)
            self.assertIn("dic_job_cancel", wait["description"])
        for phrase in ("wait=false", "dic_job_cancel", ".mcp-job.json"):
            self.assertIn(phrase, info.instructions)

        expected_keys = {
            "dic_plan": server.PIPELINE_PARAM_KEYS | server.CONTROL_PARAM_KEYS,
            "dic_run_event": EVENT_RUN_KEYS,
            "dic_run_campaign": (server.PIPELINE_PARAM_KEYS | server.CONTROL_PARAM_KEYS) - {"image_dir"},
        }
        for name, keys in expected_keys.items():
            with self.subTest(tool=name):
                schema = tools[name].inputSchema
                self.assertEqual(schema["required"], ["params"])
                params = schema["properties"]["params"]
                self.assertEqual(set(params["properties"]), keys)
                self.assertFalse(params["additionalProperties"])
                self.assertIn("output_dir", params["required"])
                self.assertIn("cal", params["required"])
                for key, prop in params["properties"].items():
                    self.assertIn("type", prop, key)
                    self.assertTrue(prop.get("description"), key)
                    if "default" in server.PARAM_SPECS[key]:
                        self.assertEqual(prop["default"], server.PARAM_SPECS[key]["default"], key)
                    else:
                        self.assertNotIn("default", prop, key)
                self.assertEqual(params["properties"]["subset"]["default"], 15)
                self.assertIn("SIZE", params["properties"]["subset"]["description"])
                self.assertEqual(params["properties"]["input_depth"]["enum"], ["8", "12", "16", "auto"])
                self.assertIn("mm", params["properties"]["cal"]["description"])
                self.assertEqual(params["properties"]["line_y"]["maxItems"], 2)
        aggregate = tools["dic_aggregate_cod"].inputSchema["properties"]
        self.assertEqual(aggregate["method"]["enum"], ["fillboundary", "naive"])
        self.assertEqual(aggregate["method"]["default"], "naive")
        self.assertEqual(server.PARAM_SPECS["cod_method"]["default"], "naive")
        self.assertIn("max_bytes", tools["dic_query_manifest"].inputSchema["properties"])
        # F11: an argument that defaults to null admits null in its type.
        for name, tool in tools.items():
            for prop_name, prop in tool.inputSchema.get("properties", {}).items():
                if prop.get("default", "<absent>") is None:
                    with self.subTest(tool=name, argument=prop_name):
                        self.assertIn({"type": "null"}, prop.get("anyOf", []))
        gauge = tools["dic_plan"].inputSchema["properties"]["params"]["properties"]["gauge"]
        self.assertIn("2*(gauge//2)", gauge["description"])
        for name in ("dic_plan", "dic_run_campaign"):
            properties = tools[name].inputSchema["properties"]["params"]["properties"]
            self.assertEqual(properties["events"]["items"], {"type": "string", "minLength": 1})
            override = properties["event_interface_y"]["items"]
            self.assertEqual(override["required"], ["pattern", "interface_y"])
            self.assertEqual(override["properties"]["interface_y"]["type"], "integer")
            self.assertFalse(override["additionalProperties"])
            self.assertIn("Campaign only", properties["event_interface_y"]["description"])
        self.assertIn("event_interface_y", info.instructions)
        self.assertIn("isError", info.instructions)
        # 2026-10-05: dic_describe gives less by default; the schema and the instructions say how to get the rest.
        section_schema = tools["dic_describe"].inputSchema["properties"]["section"]
        section_enum = next(option["enum"] for option in section_schema["anyOf"] if "enum" in option)
        for name in ("index", "all", "mcp", "exit_codes", "mcp.results", "mcp.known_issues", "mcp.run_params"):
            self.assertIn(name, section_enum)
        self.assertIn("mcp.results", section_schema["description"])
        self.assertIn("section='mcp.results'", info.instructions)

    async def test_contract_tool_entries_match_the_published_tools(self) -> None:
        from fastmcp import Client

        contract = json.loads(server.CONTRACT_PATH.read_text(encoding="utf-8"))["mcp"]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, "JOB_ROOT", Path(tmp) / "jobs"):
            async with Client(server.mcp) as client:
                tools = {tool.name: tool for tool in (await client.list_tools_mcp()).tools}
        self.assertEqual(contract["interface_revision"], server.MCP_INTERFACE_REVISION)
        self.assertEqual(set(contract["run_params"]["cli_only_keys"]), server.CLI_ONLY_KEYS)
        self.assertEqual(set(contract["run_params"]["forbidden_keys"]), server.OPERATOR_KEYS)
        for entry in contract["tools"]:
            with self.subTest(tool=entry["name"]):
                tool = tools[entry["name"]]
                self.assertEqual(entry["parameters"], list(tool.inputSchema.get("properties", {})))
                self.assertEqual(entry["read_only"], bool(tool.annotations.readOnlyHint))

    async def test_legal_calls_through_the_schema_reach_the_same_validation(self) -> None:
        from fastmcp import Client

        with tempfile.TemporaryDirectory() as tmp:
            data = _write_event(Path(tmp) / "data" / "event")
            jobs = Path(tmp) / "jobs"
            params = {"image_dir": str(data), "output_dir": str(jobs / "plan"), "line_y": "10,20", "cal": 1.0}
            with mock.patch.object(server, "JOB_ROOT", jobs), \
                    mock.patch.object(server, "READ_ROOTS", (Path(tmp) / "data",)), \
                    mock.patch.object(server, "GPU_BIN", Path(sys.executable)):
                direct = server.dic_plan(dict(params))
                async with Client(server.mcp) as client:
                    via_mcp = (await client.call_tool_mcp("dic_plan", {"params": dict(params)})).structuredContent
                    unknown = (await client.call_tool_mcp(
                        "dic_plan", {"params": {**params, "subset_size": 15}})).structuredContent
                    unwrapped = await client.call_tool_mcp("dic_plan", dict(params))
        for plan in (direct, via_mcp):
            plan.pop("call_id")
        self.assertEqual(via_mcp, direct)
        self.assertEqual(unknown["code"], "unknown_params")
        self.assertTrue(unwrapped.isError)
        self.assertIn('{"params": {...}}', unwrapped.content[0].text)

    async def test_schema_rejection_is_audited_by_middleware(self) -> None:
        from fastmcp import Client

        with tempfile.TemporaryDirectory() as tmp:
            job_root = Path(tmp) / "jobs"
            with mock.patch.object(server, "JOB_ROOT", job_root):
                async with Client(server.mcp) as client:
                    with self.assertRaises(Exception):
                        await client.call_tool("dic_run_event", {})
            records = [
                json.loads(line)
                for line in (job_root / server.AUDIT_LOG_NAME).read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["result_code"], "started")
        self.assertTrue(records[1]["result_code"].startswith("mcp_rejected:"))

    async def test_a_cancel_token_never_reaches_the_audit_log_or_the_server_log(self) -> None:
        """Review 2026-10-05 F1. A dic_job_cancel call that the schema rejects never enters the tool
        (where the token used to be hashed): the audit middleware wrote the arguments as they came,
        and FastMCP's own warning printed them. Through the real FastMCP call path, three ways an
        agent gets the call wrong, one nested variant and one well-formed call."""

        from fastmcp import Client

        token = "c0ffee11" * 6
        digest = "sha256:" + hashlib.sha256(token.encode()).hexdigest()[:12]
        with tempfile.TemporaryDirectory() as tmp:
            job_root = Path(tmp) / "jobs"
            target = str(job_root / "no-such-job")
            calls = {
                "output_dir forgotten": {"job_id": "a" * 32, "cancel_token": token},
                "key misspelt as token": {"output_dir": target, "token": token},
                "wrapped in params like a run tool": {"params": {"output_dir": target, "cancel_token": token}},
                "nested under another name": {"output_dir": target, "auth": [{"Cancel-Token": token}]},
                "well-formed, no such job": {"output_dir": target, "cancel_token": token},
            }
            with mock.patch.object(server, "JOB_ROOT", job_root), \
                    self.assertLogs("fastmcp", level="WARNING") as logged:
                async with Client(server.mcp) as client:
                    replies = {label: await client.call_tool_mcp("dic_job_cancel", arguments)
                               for label, arguments in calls.items()}
            audit_text = (job_root / server.AUDIT_LOG_NAME).read_text(encoding="utf-8")
        self.assertEqual([reply.isError for reply in replies.values()], [True, True, True, True, False])
        self.assertEqual(replies["well-formed, no such job"].structuredContent["code"], "job_not_found")
        self.assertNotIn(token, audit_text)
        records = [json.loads(line) for line in audit_text.splitlines()]
        self.assertEqual([record["tool"] for record in records], ["dic_job_cancel"] * (2 * len(calls)))
        self.assertEqual([record["result_code"].split(":")[0] for record in records],
                         ["started", "mcp_rejected"] * 4 + ["started", "job_not_found"])
        logged_params = [record["params"] for record in records[::2]]
        self.assertEqual(logged_params, [
            {"job_id": "a" * 32, "cancel_token": digest},
            {"output_dir": target, "token": digest},
            {"params": {"output_dir": target, "cancel_token": digest}},
            {"output_dir": target, "auth": [{"Cancel-Token": digest}]},
            {"output_dir": target, "cancel_token": digest},
        ])
        self.assertEqual([record["params"] for record in records[1::2]], logged_params)
        # FastMCP's own "Invalid arguments for tool" warning goes to the server's stderr.
        warnings = [record.getMessage() for record in logged.records]
        self.assertEqual(sum("Invalid arguments for tool 'dic_job_cancel'" in text for text in warnings), 4)
        self.assertFalse([text for text in warnings if token in text])
        self.assertTrue(all(digest in text for text in warnings if "Invalid arguments" in text), warnings)


@unittest.skipIf(server.FastMCP is None, "fastmcp is tested in the production venv")
class StdioServerTests(McpTestBase):
    """Real stdio servers (subprocesses) driven by the FastMCP client."""

    def transport(self):
        from fastmcp.client.transports import StdioTransport

        return StdioTransport(
            command=sys.executable, args=[str(server.PIPELINE_DIR / "mcp_server.py")],
            env=self.server_env(), keep_alive=False,
            log_file=self.root / "server-stderr.log",
        )

    def test_negative_line_y_returns_instead_of_hanging(self) -> None:
        from fastmcp import Client

        async def scenario():
            async with Client(self.transport()) as client:
                started = time.monotonic()
                raw = await client.call_tool_mcp(
                    "dic_plan", {"params": {**self.base_params("stdio-negative"), "line_y": [-5, 20]}},
                    timeout=20,
                )
                elapsed = time.monotonic() - started
                follow_up = await client.call_tool_mcp("dic_get_pitfalls", {}, timeout=20)
            return raw, elapsed, follow_up

        raw, elapsed, follow_up = asyncio.run(scenario())
        self.assertEqual(raw.structuredContent["code"], "invalid_parameter")
        self.assertEqual(raw.structuredContent["field"], "line_y")
        self.assertLess(elapsed, 5.0)
        self.assertFalse(follow_up.isError)

    def test_event_selection_refusals_return_and_the_server_keeps_answering(self) -> None:
        from fastmcp import Client

        campaign = self.data / "stdio-campaign"
        for name in ("event01", "event02"):
            _write_event(campaign / name, frames=2)
        base = {"photo_dir": str(campaign), "output_dir": str(self.jobs / "stdio-sel"), "interface_y": 15,
                "gauge": 10, "cal": 1.0}
        requests = [
            {**base, "events": ["event9*"]},
            {**base, "event_interface_y": [{"pattern": "event02", "interface_y": "16"}]},
            {**base, "events": ["event01", "event02"], "event_interface_y": [{"pattern": "event02", "interface_y": 16}]},
        ]

        async def scenario():
            replies = []
            async with Client(self.transport()) as client:
                for request in requests:
                    started = time.monotonic()
                    raw = await client.call_tool_mcp("dic_plan", {"params": request}, timeout=20)
                    replies.append((raw, time.monotonic() - started))
            return replies

        (no_match, t1), (bad_type, t2), (good, t3) = asyncio.run(scenario())
        self.assertFalse(no_match.isError)
        self.assertIn("event_pattern_no_match", [item["code"] for item in no_match.structuredContent["errors"]])
        self.assertEqual(bad_type.structuredContent["code"], "invalid_event_interface_y")
        self.assertTrue(good.structuredContent["valid"], good.structuredContent.get("errors"))
        self.assertEqual(len(good.structuredContent["mcp"]["cod_definition"]["per_event_lines"]), 2)
        self.assertLess(max(t1, t2, t3), 5.0)

    def test_closed_session_keeps_the_gpu_lock_until_the_job_ends(self) -> None:
        """Review 2026-10-03 F2: the job outlives the session, and so does its lock."""

        from fastmcp import Client

        slow = _write_event(self.data / "slow-event", frames=2)
        params = self.base_params("detached")
        params["image_dir"] = str(slow)
        record_path = self.jobs / "detached" / server.JOB_RECORD_NAME

        def record() -> dict:
            try:
                return json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}

        async def scenario():
            async with Client(self.transport()) as client:
                call = asyncio.create_task(client.call_tool_mcp("dic_run_event", {"params": params}))
                deadline = time.monotonic() + 15
                while record().get("state") != "running" and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                call.cancel()
                with contextlib.suppress(BaseException):
                    await call
            # Leaving the context closes stdin, waits ~2 s, then SIGTERMs the server.

        asyncio.run(scenario())
        detached = record()
        self.assertEqual(detached.get("state"), "running", detached)
        self.assertIn("server_detached_at", detached)
        self.assertFalse(_lock_is_free(server._gpu_lock_path(0)))
        busy = server.dic_run_event(self.base_params("while-detached"))
        self.assertEqual(busy["code"], "gpu_busy")
        self.assertFalse(busy["holder"]["server_alive"])

        deadline = time.monotonic() + 30
        while record().get("state") == "running" and time.monotonic() < deadline:
            time.sleep(0.1)
        finished = record()
        self.assertEqual(finished["state"], "exited")
        self.assertEqual(finished["exit_code"], 0)
        self.assertTrue((self.jobs / "detached" / "summary.json").is_file())
        self.assertTrue(_group_gone(finished["runner_pid"]))
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))
        codes = [entry["result_code"] for entry in self.audit_records() if entry["tool"] == "dic_run_event"
                 and entry["params"].get("output_dir") == params["output_dir"]]
        self.assertEqual(codes, ["started", "detached"])
        after = server.dic_run_event(self.base_params("after-detached"))
        self.assertEqual(after["verdict"], "ok")

    def test_a_wait_false_job_outlives_its_session_and_is_cancelled_with_its_token(self) -> None:
        """2026-10-05: start with wait=false, close the session, come back from another one."""

        from fastmcp import Client

        slow = _write_event(self.data / "sleep-background", frames=2)
        quick = _write_event(self.data / "tick-background", frames=4)
        params = {**self.base_params("bg-session"), "image_dir": str(slow), "wait": False}
        record_path = self.jobs / "bg-session" / server.JOB_RECORD_NAME

        async def scenario():
            async with Client(self.transport()) as client:
                names = [tool.name for tool in (await client.list_tools_mcp()).tools]
                # A job that ends inside the session: started, polled and read over the protocol.
                started = time.monotonic()
                first = (await client.call_tool_mcp("dic_run_event", {"params": {
                    **self.base_params("bg-quick"), "image_dir": str(quick), "wait": False}}, timeout=20)
                         ).structuredContent
                states = []
                while time.monotonic() - started < 20:
                    polled = (await client.call_tool_mcp("dic_query_manifest", first["mcp"]["poll"]["arguments"],
                                                         timeout=20)).structuredContent
                    document = polled["documents"][server.JOB_RECORD_NAME]
                    states.append(document["state"])
                    if document["state"] not in {"starting", "running"}:
                        break
                    await asyncio.sleep(0.1)
                late = (await client.call_tool_mcp("dic_job_cancel", {"output_dir": first["output_dir"]},
                                                   timeout=20)).structuredContent
                # A job that is still running when the session closes.
                handle = (await client.call_tool_mcp("dic_run_event", {"params": params}, timeout=20)).structuredContent
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    with contextlib.suppress(OSError, ValueError):
                        if json.loads(record_path.read_text(encoding="utf-8")).get("state") == "running":
                            break
                    await asyncio.sleep(0.05)
            return names, first, states, late, handle
            # Leaving the context closes stdin; the server exits (or is sent SIGTERM after ~2 s).

        names, first, states, late, handle = asyncio.run(scenario())
        self.assertEqual(len(names), 9)
        self.assertEqual(names[-1], "dic_job_cancel")
        self.assertTrue(first["ok"], first)
        self.assertIn(first["state"], {"starting", "running"})  # returned while the job was going
        self.assertIn(states[0], {"starting", "running"})
        self.assertEqual(states[-1], "exited", states)
        self.assertEqual((late["action"], late["state"]), ("none", "exited"))
        self.assertTrue((self.jobs / "bg-quick" / "cod_result.csv").is_file())

        self.assertTrue(handle["ok"], handle)
        detached = self.wait_for_record(record_path, lambda r: "server_detached_at" in r, timeout=10)
        self.assertEqual(detached["state"], "running", detached)
        self.assertFalse(_lock_is_free(server._gpu_lock_path(0)))
        self.assertFalse(_group_gone(detached["job_pgid"], wait=0.1))
        self.assertEqual([entry["result_code"] for entry in self.audit_records()
                          if entry["call_id"] == handle["job_id"]], ["started", "job_started", "detached"])

        # Another session (this process, same job root): the job id alone is not enough.
        refused = server.dic_job_cancel(handle["output_dir"], job_id=handle["job_id"])
        self.assertEqual(refused["code"], "job_not_owned")
        self.assertEqual(server.dic_job_cancel(handle["output_dir"], cancel_token="0" * 48)["code"], "job_not_owned")
        self.assertFalse(_group_gone(detached["job_pgid"], wait=0.1))
        busy = server.dic_run_event(self.base_params("while-background"))
        self.assertEqual(busy["code"], "gpu_busy")
        self.assertFalse(busy["holder"]["server_alive"])

        cancelled = server.dic_job_cancel(handle["output_dir"], job_id=handle["job_id"],
                                          cancel_token=handle["cancel_token"])
        self.assertEqual((cancelled["action"], cancelled["state"]), ("cancelled", "cancelled"), cancelled)
        self.assertEqual(cancelled["processes"], {"runner_alive": False, "job_group_alive": False})
        self.assertTrue(_group_gone(detached["job_pgid"]))
        self.assertTrue(_group_gone(detached["runner_pid"]))
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))
        final = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(final["state"], "cancelled")
        self.assertEqual(final["cancel_reason"], f"dic_job_cancel (call {cancelled['call_id']})")
        self.assertEqual(server.dic_run_event(self.base_params("after-background"))["verdict"], "ok")

    def test_runner_killed_then_server_terminated_still_ends_with_a_final_state(self) -> None:
        """Review 2026-10-04 recheck R4, end to end: the record used to stay at running for ever."""

        from fastmcp import Client

        slow = _write_event(self.data / "slow-r4", frames=2)  # the fake binary takes 6 s, then succeeds
        params = self.base_params("r4")
        params["image_dir"] = str(slow)
        record_path = self.jobs / "r4" / server.JOB_RECORD_NAME

        def record() -> dict:
            try:
                return json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}

        async def scenario():
            async with Client(self.transport()) as client:
                call = asyncio.create_task(client.call_tool_mcp("dic_run_event", {"params": params}))
                deadline = time.monotonic() + 15
                while not (record().get("state") == "running" and record().get("job_pgid")) \
                        and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                first = record()
                os.kill(first["runner_pid"], signal.SIGKILL)
                while "runner_replaced_at" not in record() and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                os.kill(first["server_pid"], signal.SIGTERM)  # the server dies while the job is adopted
                with contextlib.suppress(BaseException):
                    await asyncio.wait_for(call, timeout=10)
                return first

        try:
            first = asyncio.run(scenario())
        except Exception:  # the client may complain about the server it lost
            first = record()
        after = self.wait_for_record(record_path, lambda r: "server_detached_at" in r, timeout=10)
        self.assertIn("runner_replaced_at", after)
        self.assertNotEqual(after["runner_pid"], first["runner_pid"])
        final = self.wait_for_record(record_path, lambda r: r.get("state") not in {"starting", "running"}, timeout=30)
        self.assertEqual(final["state"], "runner_lost", final)
        self.assertIsNone(final["exit_code"])
        self.assertTrue(final["summary_present"])
        self.assertEqual(final["verdict"], "ok")
        self.assertEqual(final["progress"]["events_done"], 1)
        self.assertTrue((self.jobs / "r4" / "cod_result.csv").is_file())
        self.assertTrue(_group_gone(final["runner_pid"]))
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))
        codes = [entry["result_code"] for entry in self.audit_records() if entry["tool"] == "dic_run_event"]
        self.assertEqual(codes, ["started", "detached"])

    def test_a_blocking_call_is_polled_and_cancelled_from_its_own_session(self) -> None:
        """The server answers other requests while a run blocks: progress and cancellation over the protocol."""

        from fastmcp import Client

        sleepy = _write_event(self.data / "sleep-concurrent", frames=2)
        params = self.base_params("concurrent")
        params["image_dir"] = str(sleepy)
        output = str(self.jobs / "concurrent")

        async def scenario():
            async with Client(self.transport()) as client:
                call = asyncio.create_task(client.call_tool_mcp("dic_run_event", {"params": params}, timeout=60))
                state = None
                deadline = time.monotonic() + 15
                while state != "running" and time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                    polled = (await client.call_tool_mcp(
                        "dic_query_manifest", {"run_dir": output, "documents": [server.JOB_RECORD_NAME]},
                        timeout=10)).structuredContent
                    state = polled.get("documents", {}).get(server.JOB_RECORD_NAME, {}).get("state")
                started = time.monotonic()
                cancelled = (await client.call_tool_mcp("dic_job_cancel", {"output_dir": output}, timeout=30)
                             ).structuredContent
                blocked = (await asyncio.wait_for(call, timeout=30)).structuredContent
                return state, cancelled, blocked, time.monotonic() - started

        state, cancelled, blocked, seconds = asyncio.run(scenario())
        self.assertEqual(state, "running")
        self.assertEqual((cancelled["action"], cancelled["state"]), ("cancelled", "cancelled"), cancelled)
        self.assertEqual(blocked["code"], "job_cancelled", blocked)
        self.assertLess(seconds, 15)  # the fake binary would sleep 60 s
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

    def test_cancel_from_another_session_right_after_the_runner_was_killed(self) -> None:
        """Review 2026-10-05 F3 (e13), the other session: the job of a stdio server, its runner
        SIGKILLed, cancelled from this process with the token 0 or 30 ms later."""

        from fastmcp import Client

        async def scenario():
            rows = []
            async with Client(self.transport()) as client:
                for index, delay in enumerate((0.0, 0.03)):
                    event = _write_event(self.data / f"sleep-other-{index}", frames=2)
                    params = {**self.base_params(f"other-{index}"), "image_dir": str(event), "wait": False}
                    record_path = self.jobs / f"other-{index}" / server.JOB_RECORD_NAME
                    handle = (await client.call_tool_mcp("dic_run_event", {"params": params}, timeout=20)
                              ).structuredContent
                    record: dict = {}
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline:
                        with contextlib.suppress(OSError, ValueError):
                            record = json.loads(record_path.read_text(encoding="utf-8"))
                        if record.get("state") == "running" and record.get("job_pgid"):
                            break
                        await asyncio.sleep(0.05)
                    await asyncio.sleep(0.5)  # the binary is up
                    os.kill(record["runner_pid"], signal.SIGKILL)
                    time.sleep(delay)
                    started = time.monotonic()
                    cancelled = await asyncio.to_thread(server.dic_job_cancel, handle["output_dir"],
                                                        cancel_token=handle["cancel_token"])
                    took = time.monotonic() - started
                    released = time.monotonic() + 10
                    while not _lock_is_free(server._gpu_lock_path(0)) and time.monotonic() < released:
                        await asyncio.sleep(0.05)
                    rows.append((cancelled, took, record, json.loads(record_path.read_text(encoding="utf-8"))))
            return rows

        for cancelled, took, record, final in asyncio.run(scenario()):
            self.assertLess(took, 10.0)
            self.assertEqual((cancelled["action"], cancelled["state"]), ("cancelled", "cancelled"), cancelled)
            self.assertEqual(final["state"], "cancelled", final)
            self.assertTrue(_group_gone(record["job_pgid"]))
            self.assertTrue(_group_gone(final["runner_pid"]))
        self.assertTrue(_lock_is_free(server._gpu_lock_path(0)))

    def test_the_gpu_lock_is_free_when_a_cancellation_from_another_session_returns(self) -> None:
        """The reply of dic_job_cancel says that the locks are released: also when the server that
        started the job is another process, whose thread leaves the lock a moment after the runner
        (the review's e17, second half)."""

        from fastmcp import Client

        async def scenario():
            free_at_reply = []
            async with Client(self.transport()) as client:
                for index in range(4):
                    event = _write_event(self.data / f"sleep-lock-{index}", frames=2)
                    params = {**self.base_params(f"lock-{index}"), "image_dir": str(event), "wait": False}
                    record_path = self.jobs / f"lock-{index}" / server.JOB_RECORD_NAME
                    handle = (await client.call_tool_mcp("dic_run_event", {"params": params}, timeout=20)
                              ).structuredContent
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline:
                        with contextlib.suppress(OSError, ValueError):
                            if json.loads(record_path.read_text(encoding="utf-8")).get("state") == "running":
                                break
                        await asyncio.sleep(0.05)
                    await asyncio.sleep(0.3)
                    cancelled = await asyncio.to_thread(server.dic_job_cancel, handle["output_dir"],
                                                        cancel_token=handle["cancel_token"])
                    free_at_reply.append((cancelled.get("action"), _lock_is_free(server._gpu_lock_path(0)),
                                          cancelled.get("note")))
                    await asyncio.sleep(0.2)
            return free_at_reply

        rows = asyncio.run(scenario())
        self.assertEqual([(action, free) for action, free, _ in rows], [("cancelled", True)] * 4, rows)
        self.assertTrue(all("locks are released" in note for _, _, note in rows), rows)


if __name__ == "__main__":
    unittest.main()
