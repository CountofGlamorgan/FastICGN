#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit and fake-binary integration tests for the agent-facing pipeline."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dic_pipeline as pipeline
import fasticgn_io
import sync_agent_docs


CSV_HEADER = "x,y,u,v,zncc,uy,vy,ux,vx\n"

# Hermetic: no test may see the real user's ~/.config/fasticgn/config.toml or a
# FASTICGN_* variable of the shell that runs the suite (review-2026-10-03 F13).
_MODULE_FIXTURES: list = []


def setUpModule() -> None:
    temp = tempfile.TemporaryDirectory()
    _MODULE_FIXTURES.append(temp)
    environment = mock.patch.dict(os.environ, {}, clear=False)
    environment.start()
    _MODULE_FIXTURES.append(environment)
    for key in [key for key in os.environ if key.startswith(pipeline.ENV_PREFIX)]:
        del os.environ[key]
    config = mock.patch.object(
        pipeline, "USER_CONFIG_PATH", Path(temp.name) / "no-user-config.toml",
    )
    config.start()
    _MODULE_FIXTURES.append(config)


def tearDownModule() -> None:
    while _MODULE_FIXTURES:
        fixture = _MODULE_FIXTURES.pop()
        if isinstance(fixture, tempfile.TemporaryDirectory):
            fixture.cleanup()
        else:
            fixture.stop()


def write_frame(path: Path, rows: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(CSV_HEADER)
        for row in rows:
            handle.write(",".join(str(value) for value in row) + "\n")


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def argparse_namespace(**values):
    """A Namespace for helper-level tests (only the fields the helper reads)."""
    return argparse.Namespace(**values)


class FrameStatsTests(unittest.TestCase):
    def test_pooled_stats_use_binary_valid_ratio_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame_dir = Path(tmp)
            write_frame(frame_dir / "frame_000002.csv", [
                [1, 10, 0, 0, 0.99, 0, 0, 0, 0],
                [2, 10, math.nan, 0, 0.95, 0, 0, 0, 0],
                [3, 20, 0, 0, -4, 0, 0, 0, 0],
                [4, 20, 0, 0, 0.90, 0, 0, 0, 0],
            ])
            write_frame(frame_dir / "frame_000003.csv", [
                [1, 10, 0, 0, 0.80, 0, 0, 0, 0],
                [2, 20, 0, 0, math.nan, 0, 0, 0, 0],
            ])

            stats = pipeline.collect_frame_stats(frame_dir)

        self.assertEqual(stats["frames"], 2)
        self.assertIsNone(stats["poi_per_frame"])
        self.assertAlmostEqual(stats["valid_ratio"], 4 / 6)
        self.assertAlmostEqual(stats["zncc_median"], 0.925)
        # u=NaN 但 zncc 有效，必须与 binary FrameQa 一样计入 valid_ratio。
        self.assertEqual(stats["frame_read_errors"], 0)

    def test_unique_poi_mode_and_warning_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame_dir = Path(tmp)
            base = [[1, 10, 0, 0, 0.9, 0, 0, 0, 0]]
            write_frame(frame_dir / "frame_000002.csv", base * 4)
            write_frame(frame_dir / "frame_000003.csv", base * 4)
            write_frame(frame_dir / "frame_000004.csv", base * 2)
            stats = pipeline.collect_frame_stats(frame_dir)

        self.assertEqual(stats["poi_per_frame"], 4)
        self.assertEqual(pipeline.event_warnings(stats, "8", 254), [])
        warnings = pipeline.event_warnings(
            {**stats, "valid_ratio": 0.94999, "zncc_median": 0.89999}, "8", 255,
        )
        self.assertTrue(any(item.startswith("valid_ratio_below") for item in warnings))
        self.assertTrue(any(item.startswith("zncc_median_below") for item in warnings))
        self.assertTrue(any(item.startswith("input_8bit_max_255") for item in warnings))

    def test_event_frame_mode_and_verdict(self) -> None:
        events = [
            {"ok": True, "status": "ok", "name": "a", "frames": 3, "warnings": []},
            {"ok": True, "status": "ok", "name": "b", "frames": 3, "warnings": []},
            {"ok": True, "status": "ok", "name": "c", "frames": 2, "warnings": []},
        ]
        args = pipeline.create_parser().parse_args([
            "--image-dir", "/tmp/event", "--output-dir", "/tmp/out",
            "--line-y", "10,20", "--cal", "1",
        ])
        summary = pipeline.build_run_summary(
            args, {"path": "fake", "git_hash": "abc", "build_date": "now"},
            [], events, 1.25,
        )
        self.assertEqual(summary["verdict"], "warn")
        self.assertFalse(summary["events"][0]["warnings"])
        self.assertTrue(summary["events"][2]["warnings"][0].startswith("frame_count_differs"))

        events[2]["ok"] = False
        events[2]["status"] = "fail"
        failed = pipeline.build_run_summary(
            args, summary["binary"], [], events, 1.25,
        )
        self.assertEqual(failed["verdict"], "fail")
        self.assertEqual(failed["global"]["events_failed"], 1)


class PathResolutionTests(unittest.TestCase):
    """CLI argument -> FASTICGN_* env -> config file -> repo build -> lab path."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.binary = self.root / "opencorr_cli_cuda_mvp"
        self.binary.write_text("#!/bin/sh\nexit 0\n")
        self.binary.chmod(0o755)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_argument_wins_over_environment_and_config(self) -> None:
        config = self.root / "config.toml"
        config.write_text('[paths]\ngpu_bin = "/from/config"\n', encoding="utf-8")
        env = {pipeline.ENV_GPU_BIN: "/from/env", pipeline.ENV_CONFIG: str(config)}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                pipeline.resolve_gpu_bin(str(self.binary)), (str(self.binary), "argument"),
            )

    def test_environment_wins_over_config(self) -> None:
        config = self.root / "config.toml"
        config.write_text('[paths]\ngpu_bin = "/from/config"\n', encoding="utf-8")
        env = {pipeline.ENV_GPU_BIN: "/from/env", pipeline.ENV_CONFIG: str(config)}
        with mock.patch.dict(os.environ, env, clear=False):
            path, source = pipeline.resolve_gpu_bin()
        self.assertEqual(path, "/from/env")
        self.assertEqual(source, f"environment:{pipeline.ENV_GPU_BIN}")

    def test_config_file_is_used_when_no_argument_or_environment(self) -> None:
        config = self.root / "config.toml"
        config.write_text(f'[paths]\ngpu_bin = "{self.binary}"\n', encoding="utf-8")
        env = {pipeline.ENV_CONFIG: str(config)}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop(pipeline.ENV_GPU_BIN, None)
            path, source = pipeline.resolve_gpu_bin()
        self.assertEqual(path, str(self.binary))
        self.assertTrue(source.startswith("config:"))

    def test_home_build_tree_is_searched_before_the_lab_path(self) -> None:
        home = self.root / "deployment"
        build = home / "build" / "cuda-release"
        build.mkdir(parents=True)
        binary = build / "opencorr_cli_cuda_mvp"
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        env = {pipeline.ENV_HOME: str(home)}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop(pipeline.ENV_GPU_BIN, None)
            os.environ.pop(pipeline.ENV_CONFIG, None)
            path, source = pipeline.resolve_gpu_bin()
        self.assertEqual(path, str(binary))
        self.assertEqual(source, "repository_build")

    def test_missing_binary_reports_the_search_order(self) -> None:
        env = {pipeline.ENV_HOME: str(self.root / "empty")}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop(pipeline.ENV_GPU_BIN, None)
            os.environ.pop(pipeline.ENV_CONFIG, None)
            description = pipeline.binary_search_description(pipeline.GPU_BINARY_NAME)
        self.assertIn(pipeline.ENV_GPU_BIN, description)
        self.assertIn("deprecated lab path", description)

    def test_read_allowlist_default_and_override(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(pipeline.ENV_READ_ALLOWLIST, None)
            os.environ.pop(pipeline.ENV_CONFIG, None)
            roots, source = pipeline.resolve_read_allowlist()
        self.assertEqual(source, "default")
        self.assertIn(pipeline.REPO_ROOT, roots)
        self.assertIn(Path("/data"), roots)

        override = os.pathsep.join([str(self.root), "/srv/data"])
        with mock.patch.dict(os.environ, {pipeline.ENV_READ_ALLOWLIST: override}, clear=False):
            roots, source = pipeline.resolve_read_allowlist()
        self.assertEqual(source, f"environment:{pipeline.ENV_READ_ALLOWLIST}")
        self.assertEqual(roots, [self.root, Path("/srv/data"), pipeline.REPO_ROOT])


class ImageDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_default_rule_is_unchanged(self) -> None:
        for name in ("shot_000003.png", "shot_000001.png", "notes.txt", "shot_000002.png"):
            (self.root / name).write_bytes(b"")
        images = pipeline.find_images(str(self.root))
        self.assertEqual([number for number, _ in images], [1, 2, 3])

    def test_custom_pattern_without_digits_numbers_from_one(self) -> None:
        for name in ("b.tif", "a.tif", "c.tif"):
            (self.root / name).write_bytes(b"")
        images = pipeline.find_images(str(self.root), r"\.tif$", "name")
        self.assertEqual([number for number, _ in images], [1, 2, 3])
        self.assertEqual([Path(path).name for _, path in images], ["a.tif", "b.tif", "c.tif"])

    def test_custom_pattern_with_digit_group(self) -> None:
        for name in ("cam-12.bmp", "cam-3.bmp"):
            (self.root / name).write_bytes(b"")
        images = pipeline.find_images(str(self.root), r"cam-(\d+)\.bmp$")
        self.assertEqual([number for number, _ in images], [3, 12])


class CodAggregationTests(unittest.TestCase):
    """review-2026-09-16 section 4: a missing uy column must not degrade silently."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_five_column_frame(self) -> None:
        with (self.root / "frame_000002.csv").open("w", encoding="utf-8") as handle:
            handle.write("x,y,u,v,zncc\n")
            handle.write("10,10,1.0,0,0.99\n")
            handle.write("10,20,0.5,0,0.99\n")

    def test_fillboundary_rejects_a_frame_csv_without_uy(self) -> None:
        self.write_five_column_frame()
        with self.assertRaises(pipeline.CodAggregationError) as raised:
            pipeline.aggregate_cod(self.root, 10, 20, 1.0, None, "fillboundary")
        self.assertEqual(raised.exception.code, "cod_uy_column_missing")
        self.assertIn("--cod-method naive", str(raised.exception))

    def test_explicit_naive_accepts_a_frame_csv_without_uy(self) -> None:
        self.write_five_column_frame()
        cod_um, x_mm, frames, gauge_mm = pipeline.aggregate_cod(
            self.root, 10, 20, 1.0, None, "naive",
        )
        self.assertAlmostEqual(float(cod_um[0, 0]), 500.0)
        self.assertEqual(list(frames), [2])
        self.assertAlmostEqual(gauge_mm, 10.0)
        self.assertEqual(list(x_mm), [10.0])

    def test_fillboundary_uses_the_uy_gradient(self) -> None:
        with (self.root / "frame_000002.csv").open("w", encoding="utf-8") as handle:
            handle.write("x,y,u,v,zncc,uy,vy,ux,vx\n")
            handle.write("10,10,1.0,0,0.99,0.1,0,0,0\n")
            handle.write("10,20,0.5,0,0.99,0.1,0,0,0\n")
        cod_um, _, _, _ = pipeline.aggregate_cod(self.root, 10, 20, 1.0, None, "fillboundary")
        # y0 = 15, so (1.0 + 0.1*5) - (0.5 + 0.1*(-5)) = 1.5 - 0.0 px = 1500 um.
        # The naive difference would be 1.0 - 0.5 = 0.5 px = 500 um.
        self.assertAlmostEqual(float(cod_um[0, 0]), 1500.0)

    def test_default_method_is_naive(self) -> None:
        # 2026-10-05: the default COD method went back to naive (fillboundary stays selectable).
        self.assertEqual(pipeline.create_parser().get_default("cod_method"), "naive")
        with (self.root / "frame_000002.csv").open("w", encoding="utf-8") as handle:
            handle.write("x,y,u,v,zncc,uy,vy,ux,vx\n")
            handle.write("10,10,1.0,0,0.99,0.1,0,0,0\n")
            handle.write("10,20,0.5,0,0.99,0.1,0,0,0\n")
        cod_um, _, _, _ = pipeline.aggregate_cod(self.root, 10, 20, 1.0, None)
        # No extrapolation by default: 1.0 - 0.5 px = 500 um, not the 1500 um above.
        self.assertAlmostEqual(float(cod_um[0, 0]), 500.0)


class SaturationBandTests(unittest.TestCase):
    """2026-10-04: input_8bit_max_255 grades the pixels the COD actually correlates."""

    GEOMETRY = {"cod": True, "line_ys": [10, 20], "subset_radius": 2, "step": 1, "roi_cols": None}

    def test_band_follows_the_binary_poi_grid_and_the_roi(self) -> None:
        # 32 x 32, radius 2: the binary's margin is 4, so its POIs sit at x = 4..27.
        band = pipeline.line_band(10, 32, 32, 2, 1, None)
        self.assertEqual((band["rows"], band["cols"], band["poi_columns"]),
                         ([8, 12], [2, 29], list(range(4, 28))))
        band = pipeline.line_band(10, 32, 32, 2, 1, (6, 20))
        self.assertEqual((band["rows"], band["cols"], band["poi_columns"]),
                         ([8, 12], [4, 22], list(range(6, 21))))
        band = pipeline.line_band(10, 32, 32, 2, 1, (0, 31))  # clipped to the image
        self.assertEqual((band["cols"], band["poi_columns"]), ([0, 31], list(range(4, 28))))
        band = pipeline.line_band(20, 32, 32, 2, 3, None)
        self.assertEqual((band["rows"], band["cols"], band["poi_columns"]),
                         ([18, 22], [2, 27], list(range(4, 28, 3))))
        self.assertEqual(pipeline.line_band(27, 32, 32, 2, 1, None)["rows"], [25, 29])
        # No subset of the line enters the COD: the binary skips a line inside its
        # margin, and an ROI may hold no POI column.
        for y, roi in ((3, None), (28, None), (10, (0, 3)), (10, (28, 31))):
            with self.subTest(y=y, roi=roi):
                self.assertIsNone(pipeline.line_band(y, 32, 32, 2, 1, roi))
        self.assertEqual(pipeline.binary_poi_columns(32, 2, 1), list(range(4, 28)))
        self.assertTrue(pipeline.binary_computes_line(4, 32, 2))
        self.assertFalse(pipeline.binary_computes_line(28, 32, 2))

    def test_subset_counts_use_exactly_the_subset_windows(self) -> None:
        saturated = np.zeros((32, 32), dtype=bool)
        saturated[10, 15] = True    # one pixel in the subsets centred at x = 13..17
        saturated[9:12, 20] = True  # 3 of 25 pixels (> 10 %) in those at x = 18..22
        saturated[13, 15] = True    # one row below the band of y = 10
        saturated[10, 1] = True     # left of every subset (the first POI is x = 4)
        line = pipeline.measure_line_band(saturated, "L0", 10, 2, 1, None)
        self.assertEqual((line["label"], line["y"], line["band_rows"], line["band_cols"]),
                         ("L0", 10, [8, 12], [2, 29]))
        self.assertEqual(line["subsets"], 24)
        self.assertAlmostEqual(line["band_fraction"], 4 / (5 * 28))
        self.assertAlmostEqual(line["subsets_with_saturation"], 10 / 24)
        self.assertAlmostEqual(line["subsets_over_10pct"], 5 / 24)
        roi = pipeline.measure_line_band(saturated, "L0", 10, 2, 1, (13, 17))
        self.assertEqual((roi["band_cols"], roi["subsets"]), ([11, 19], 5))
        self.assertAlmostEqual(roi["band_fraction"], 1 / (5 * 9))
        self.assertEqual((roi["subsets_with_saturation"], roi["subsets_over_10pct"]), (1.0, 0.0))
        skipped = pipeline.measure_line_band(saturated, "L1", 30, 2, 1, None)
        self.assertEqual(skipped, {
            "label": "L1", "y": 30, "band_rows": None, "band_cols": None, "band_fraction": None,
            "subsets": 0, "subsets_with_saturation": None, "subsets_over_10pct": None})

    def test_cod_mode_grades_the_largest_band_and_full_field_the_frame(self) -> None:
        saturated = np.zeros((32, 32), dtype=bool)
        saturated[26:, :] = True        # 192 px away from both lines: 18.75 % of the frame
        saturated[19, 3:6] = True       # 3 px in the L1 band (rows 18-22, cols 2-29)
        cod = pipeline.saturation_report(saturated, 8, 255, self.GEOMETRY, 0.05)
        self.assertEqual((cod["basis"], cod["severity"]), ("line_bands", "info"))
        self.assertEqual([line["band_fraction"] for line in cod["lines"]], [0.0, 3 / 140])
        self.assertEqual(cod["value"], 3 / 140)
        self.assertEqual(cod["frame_fraction"], 195 / 1024)
        field = pipeline.saturation_report(saturated, 8, 255, {**self.GEOMETRY, "cod": False}, 0.05)
        self.assertEqual((field["basis"], field["value"], field["severity"], field["lines"]),
                         ("frame", 195 / 1024, "warn", []))
        self.assertEqual(pipeline.saturation_report(saturated, 8, 255, None, 0.05)["basis"], "frame")

    def test_threshold_boundary_depth_and_unknown_value(self) -> None:
        saturated = np.zeros((32, 32), dtype=bool)
        saturated[10, 3:10] = True  # 7 of the 140 pixels of the L0 band: exactly 5 %
        report = pipeline.saturation_report(saturated, 8, 255, self.GEOMETRY, 0.05)
        self.assertEqual((report["value"], report["severity"]), (0.05, "info"))
        saturated[11, 3] = True     # 8 of 140
        self.assertEqual(pipeline.saturation_report(saturated, 8, 255, self.GEOMETRY, 0.05)["severity"], "warn")
        self.assertEqual(pipeline.DEFAULT_SATURATION_WARN_FRACTION, 0.05)
        args = pipeline.create_parser().parse_args(["--image-dir", "e", "--output-dir", "o"])
        self.assertEqual(args.saturation_warn_fraction, 0.05)
        empty = np.zeros((32, 32), dtype=bool)
        # A container deeper than 8 bits read as 8-bit is a warning whatever the fraction.
        self.assertEqual(pipeline.saturation_report(empty, 16, 255, self.GEOMETRY, 0.05)["severity"], "warn")
        # No pixel at 255: no warning, so no level.
        self.assertIsNone(pipeline.saturation_report(empty, 8, 254, self.GEOMETRY, 0.05)["severity"])
        # Both lines inside the binary margin: nothing to grade, which is a warning.
        skipped = pipeline.saturation_report(saturated, 8, 255, {**self.GEOMETRY, "line_ys": [2, 30]}, 0.05)
        self.assertEqual((skipped["value"], skipped["severity"]), (None, "warn"))
        self.assertEqual([line["subsets"] for line in skipped["lines"]], [0, 0])
        self.assertIn("no measurement line has a subset",
                      pipeline.saturation_warning_detail(skipped, 8))


class RawIoTests(unittest.TestCase):
    def test_reader_round_trips_a_constructed_raw_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame_000002.bin"
            records = np.zeros(3, dtype=fasticgn_io.RAW_DTYPE)
            records["x"] = [4, 5, 6]
            records["y"] = [7, 7, 7]
            records["u"] = [1.5, -2.0, 0.0]
            records["v"] = [0.25, 0.0, 0.0]
            records["uy"] = [0.5, 0.0, 0.0]
            records["zncc"] = [0.99, -4.0, np.nan]
            path.write_bytes(records.tobytes())

            read_back = fasticgn_io.read_raw_frame(path)
            self.assertEqual(fasticgn_io.RECORD_BYTES, 36)
            self.assertEqual(read_back.shape[0], 3)
            np.testing.assert_array_equal(read_back["x"], [4, 5, 6])
            np.testing.assert_array_almost_equal(read_back["u"], [1.5, -2.0, 0.0])
            np.testing.assert_array_equal(
                fasticgn_io.valid_mask(read_back), [True, False, False],
            )
            columns = fasticgn_io.to_csv_columns(read_back)
            self.assertEqual(columns.shape, (3, 9))
            # Legacy CSV order is x,y,u,v,zncc,uy,... so column 5 is uy.
            self.assertAlmostEqual(columns[0, 5], 0.5)

    def test_truncated_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame_000002.bin"
            path.write_bytes(b"\x00" * (fasticgn_io.RECORD_BYTES + 7))
            with self.assertRaises(fasticgn_io.RawFormatError):
                fasticgn_io.read_raw_frame(path)

    def test_manifest_and_ledger_readers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = np.zeros(2, dtype=fasticgn_io.RAW_DTYPE)
            records["zncc"] = [0.9, 0.8]
            (root / "frame_000002.bin").write_bytes(records.tobytes())
            manifest = {
                "format": fasticgn_io.RAW_FORMAT,
                "endianness": "little",
                "record_bytes": fasticgn_io.RECORD_BYTES,
                "engine": {"implementation": "FastICGN", "icgn_kernel": "warp"},
                "frames": [{"file": "frame_000002.bin", "rows": 2,
                            "bytes": records.nbytes, "output_name": "frame_000002.csv"}],
            }
            fasticgn_io.manifest_path(root).write_text(json.dumps(manifest), encoding="utf-8")
            state = root / fasticgn_io.STATE_DIR_NAME
            state.mkdir()
            (state / "complete.json").write_text('{"schema_version": 1}', encoding="utf-8")

            parsed = fasticgn_io.read_manifest(fasticgn_io.manifest_path(root))
            self.assertEqual(fasticgn_io.engine_metadata(parsed)["icgn_kernel"], "warp")
            frames = list(fasticgn_io.iter_frames(fasticgn_io.manifest_path(root)))
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0][1].shape[0], 2)
            ledger = fasticgn_io.read_ledger(root)
            self.assertTrue(ledger["present"])
            self.assertEqual(ledger["complete.json"]["schema_version"], 1)


class CsvIoAndRunStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_csv_frame_has_the_raw_field_names_and_values(self) -> None:
        raw = np.zeros(3, dtype=fasticgn_io.RAW_DTYPE)
        raw["x"], raw["y"] = [9, 10, 11], [20, 20, 20]
        raw["u"], raw["v"] = [0.125, -1.5, 0.0], [0.25, 0.0, 0.0]
        raw["ux"], raw["uy"], raw["vx"], raw["vy"] = [0.001, 0, 0], [0.5, 0, 0], [0.002, 0, 0], [0.003, 0, 0]
        raw["zncc"] = [0.99, -4.0, np.nan]
        columns = fasticgn_io.to_csv_columns(raw)
        path = self.root / "frame_000002.csv"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(CSV_HEADER)
            for row in columns:
                handle.write(f"{int(row[0])},{int(row[1])}," + ",".join(f"{v:.6g}" for v in row[2:]) + "\n")

        records = fasticgn_io.read_csv_frame(path)
        self.assertEqual(records.dtype.names, fasticgn_io.RAW_DTYPE.names)
        self.assertEqual(records.dtype, fasticgn_io.CSV_RECORD_DTYPE)
        for name in fasticgn_io.RAW_DTYPE.names:
            np.testing.assert_allclose(records[name], raw[name].astype(np.float64), rtol=1e-6)
        np.testing.assert_array_equal(fasticgn_io.valid_mask(records), [True, False, False])
        dispatched = fasticgn_io.read_frame(path)
        for name in records.dtype.names:
            np.testing.assert_array_equal(dispatched[name], records[name])

    def test_csv_reader_handles_legacy_empty_and_malformed_files(self) -> None:
        legacy = self.root / "frame_000003.csv"
        legacy.write_text("x,y,u,v,zncc\n1,2,0.5,0.25,0.9\n", encoding="utf-8")
        records = fasticgn_io.read_csv_frame(legacy)
        self.assertEqual(records.shape, (1,))
        self.assertTrue(np.isnan(records["uy"][0]))  # missing gradient is NaN, never 0
        empty = self.root / "frame_000004.csv"
        empty.write_text(CSV_HEADER, encoding="utf-8")
        self.assertEqual(fasticgn_io.read_csv_frame(empty).shape, (0,))
        for name, text in (
            ("no-zncc.csv", "x,y,u,v\n1,2,3,4\n"),
            ("ragged.csv", CSV_HEADER + "1,2,3\n"),
            ("float-x.csv", CSV_HEADER + "1.5,2,0,0,0.9,0,0,0,0\n"),
            # review 2026-10-04 R12: no silent int32 overflow, no silently skipped line
            ("huge-x.csv", CSV_HEADER + "3000000000,2,0,0,0.9,0,0,0,0\n"),
            ("comment.csv", CSV_HEADER + "# 1,2,0,0,0.9,0,0,0,0\n1,2,0,0,0.9,0,0,0,0\n"),
        ):
            with self.subTest(name=name):
                (self.root / name).write_text(text, encoding="utf-8")
                with self.assertRaises(fasticgn_io.CsvFormatError):
                    fasticgn_io.read_csv_frame(self.root / name)
        bom = self.root / "bom.csv"
        bom.write_text("﻿" + CSV_HEADER + "1,2,0,0,0.9,0,0,0,0\n", encoding="utf-8")
        with self.assertRaisesRegex(fasticgn_io.CsvFormatError, "byte-order mark"):
            fasticgn_io.read_csv_frame(bom)

    def test_iter_frame_dir_orders_by_frame_number_and_skips_other_files(self) -> None:
        for number in (10, 2, 7):
            write_frame(self.root / f"frame_{number:06d}.csv", [[1, 2, 0, 0, 0.9, 0, 0, 0, 0]])
        (self.root / "notes.txt").write_text("x", encoding="utf-8")
        (self.root / ".opencorr-state").mkdir()
        (self.root / ".opencorr-tmp-1-2").write_text("x", encoding="utf-8")
        names = [path.name for path, _ in fasticgn_io.iter_frame_dir(self.root)]
        self.assertEqual(names, ["frame_000002.csv", "frame_000007.csv", "frame_000010.csv"])
        self.assertEqual(list(fasticgn_io.iter_frame_dir(self.root, suffix=".bin")), [])
        self.assertEqual(fasticgn_io.frame_number("frame_072424.bin"), 72424)
        self.assertIsNone(fasticgn_io.frame_number("cod_result.csv"))

    def make_run(self, raw: bool = True) -> Path:
        run = self.root / ("raw-run" if raw else "csv-run")
        state = run / fasticgn_io.STATE_DIR_NAME
        state.mkdir(parents=True)
        targets = []
        manifest_frames = []
        for index, number in enumerate((2, 3)):
            output_name = f"frame_{number:06d}.csv"
            name = output_name.replace(".csv", ".bin") if raw else output_name
            payload = (np.zeros(2, dtype=fasticgn_io.RAW_DTYPE).tobytes() if raw
                       else (CSV_HEADER + "1,2,0,0,0.9,0,0,0,0\n").encode())
            (run / name).write_bytes(payload)
            targets.append({"file": name, "output_name": output_name, "input": {}})
            (state / f"{index:032x}.json").write_text(json.dumps({
                "schema_version": 1, "key": f"frame/{name}", "rows": 2 if raw else 1,
                "bytes": len(payload), "algorithm": "xxh3-128", "digest": "0" * 32,
            }), encoding="utf-8")
            manifest_frames.append({"file": name, "output_name": output_name, "rows": 2,
                                    "bytes": len(payload)})
        (state / "contract.json").write_text(json.dumps({
            "schema_version": 1, "targets": targets, "git_hash": "abc", "build_date": "now",
            "options": {"format": "raw" if raw else "csv", "radius": 7, "step": 1, "line_ys": []},
        }), encoding="utf-8")
        (state / "complete.json").write_text(json.dumps({
            "schema_version": 1, "contract_digest": "f" * 32, "durability": "rename", "metadata": {},
        }), encoding="utf-8")
        if raw:
            fasticgn_io.manifest_path(run).write_text(json.dumps({
                "format": fasticgn_io.RAW_FORMAT, "endianness": "little",
                "record_bytes": fasticgn_io.RECORD_BYTES, "frames": manifest_frames,
            }), encoding="utf-8")
        return run

    def snapshot(self, root: Path) -> list:
        return sorted(
            (str(path.relative_to(root)), path.stat().st_mtime_ns, path.stat().st_size)
            for path in root.rglob("*")
        )

    def test_run_status_accepts_a_complete_run_without_changing_it(self) -> None:
        for raw in (True, False):
            with self.subTest(raw=raw):
                run = self.make_run(raw)
                before = self.snapshot(run)
                status = fasticgn_io.run_status(run)
                self.assertEqual(self.snapshot(run), before)
                self.assertEqual(status["verdict"], "complete", status["problems"])
                self.assertTrue(status["complete"])
                self.assertFalse(status["content_verified"])
                self.assertTrue(status["not_verified"])
                self.assertEqual(status["frames"]["expected"], 2)
                self.assertEqual(status["contract"]["subset_radius"], 7)
                if raw:
                    self.assertTrue(status["manifest"]["matches_targets"])

    def test_run_status_reports_each_kind_of_damage(self) -> None:
        run = self.make_run(raw=True)
        (run / "frame_000003.bin").unlink()
        (run / "frame_000002.bin").write_bytes(b"\x00" * 10)
        (run / "stray.txt").write_text("x", encoding="utf-8")
        (run / ".opencorr-tmp-12-3").write_text("x", encoding="utf-8")
        (run / fasticgn_io.STATE_DIR_NAME / "contract.pending.json").write_text("{}", encoding="utf-8")
        (run / fasticgn_io.STATE_DIR_NAME / "complete.json").unlink()
        status = fasticgn_io.run_status(run)
        self.assertEqual(status["verdict"], "incomplete")
        self.assertEqual(status["frames"]["missing"], ["frame_000003.bin"])
        self.assertEqual(status["frames"]["size_mismatch"], ["frame_000002.bin"])
        self.assertEqual(status["frames"]["rows_mismatch"], ["frame_000002.bin"])
        self.assertEqual(status["unexpected_entries"], ["stray.txt"])
        self.assertEqual(status["residue"], [".opencorr-tmp-12-3"])
        self.assertTrue(status["pending_contract"])
        self.assertFalse(status["complete_record"]["present"])
        self.assertGreaterEqual(len(status["problems"]), 6)

        plain = self.root / "plain"
        plain.mkdir()
        write_frame(plain / "frame_000002.csv", [[1, 2, 0, 0, 0.9, 0, 0, 0, 0]])
        status = fasticgn_io.run_status(plain)
        self.assertEqual(status["verdict"], "not_a_run")
        self.assertEqual(status["frames"]["present"], 1)
        self.assertEqual(fasticgn_io.run_status(self.root / "missing")["verdict"], "not_a_run")

    def test_run_status_knows_the_binary_diagnostics_directory(self) -> None:
        """Review 2026-10-04 R7: --diagnostics writes diagnostics/ itself; not a foreign entry."""
        run = self.make_run(raw=False)
        state = run / fasticgn_io.STATE_DIR_NAME
        contract = json.loads((state / "contract.json").read_text(encoding="utf-8"))
        contract["options"].update({"diagnostics": True, "diagnostics_dir": "default"})
        (state / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
        diagnostics = run / "diagnostics"
        diagnostics.mkdir()
        for index, number in enumerate((2, 3)):
            name = f"frame_{number:06d}.diagnostics.csv"
            payload = b"x,y,iterations\n1,2,3\n"
            (diagnostics / name).write_bytes(payload)
            (state / f"{index + 10:032x}.json").write_text(json.dumps({
                "schema_version": 1, "key": f"diagnostics/{name}", "rows": 1, "bytes": len(payload),
            }), encoding="utf-8")
        (diagnostics / fasticgn_io.DIAGNOSTICS_MANIFEST_NAME).write_text("{}", encoding="utf-8")
        status = fasticgn_io.run_status(run)
        self.assertEqual(status["verdict"], "complete", status["problems"])
        self.assertEqual(status["diagnostics"]["present"], 2)
        self.assertEqual(status["unexpected_entries"], [])

        (diagnostics / "frame_000003.diagnostics.csv").unlink()
        (diagnostics / "stray.txt").write_text("x", encoding="utf-8")
        status = fasticgn_io.run_status(run)
        self.assertEqual(status["verdict"], "incomplete")
        self.assertEqual(status["diagnostics"]["missing"], ["frame_000003.diagnostics.csv"])
        self.assertEqual(status["diagnostics"]["unexpected_entries"], ["stray.txt"])


class FakeBinaryIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fake_bin = self.root / "fake_gpu.py"
        self.log_path = self.root / "argv.jsonl"
        self.fake_bin.write_text(textwrap.dedent(f"""\
            #!/usr/bin/env python3
            import json
            import sys
            import time
            from pathlib import Path

            log = Path({str(self.log_path)!r})
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sys.argv[1:], ensure_ascii=False) + "\\n")

            if sys.argv[1:] == ["--version"]:
                print("opencorr_cli_cuda_mvp")
                print("git_hash=fake123")
                print("build_date=2099-01-01T00:00:00Z")
                raise SystemExit(0)

            args = sys.argv[1:]
            ref = Path(args[args.index("--ref") + 1])
            name = str(ref)
            if "errorusage" in name:
                print("Error: --conv-threshold must be a finite positive number, got 'nan'",
                      file=sys.stderr)
                for index in range(30):
                    print(f"  usage line {{index}} --repair --no-repair", file=sys.stderr)
                raise SystemExit(1)
            if "resumereject" in name:
                print("Error: resume rejected: run configuration/reference/binary changed",
                      file=sys.stderr)
                raise SystemExit(1)
            if "fail" in name:
                print("synthetic failure", file=sys.stderr)
                raise SystemExit(7)
            if "sleepy" in name:
                print("Warning: streaming marker before a long computation", file=sys.stderr, flush=True)
                time.sleep(60)
            print("Reference input: source_depth=CV_8U, source_range=[0,200], "
                  "effective_depth=8, effective_align=auto, output_depth=CV_32F, "
                  "output_range=[0,200], scale=1", file=sys.stderr)
            tar_list = Path(args[args.index("--tar-list") + 1])
            output_dir = Path(args[args.index("--output-dir") + 1])
            # Output-ledger semantics of the real binary (src/cli_output_integrity.cpp):
            # a run contract with a configuration fingerprint and the target list;
            # a different configuration or a target list that is not the recorded
            # one plus appended frames is "resume rejected"; files in a directory
            # without a contract are an "untracked entry"; duplicate output names fail.
            targets = [entry.rsplit(maxsplit=1)[1]
                       for entry in tar_list.read_text(encoding="utf-8").splitlines()]
            if len(set(targets)) != len(targets):
                print("Error: duplicate output filename: " + targets[0], file=sys.stderr)
                raise SystemExit(1)
            skip = {{"--tar-list", "--output-dir"}}
            fingerprint = [value for index, value in enumerate(args)
                           if value not in skip and (index == 0 or args[index - 1] not in skip)]
            contract_path = output_dir / ".opencorr-state" / "contract.json"
            if contract_path.exists():
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
                if contract["fingerprint"] != fingerprint:
                    print("Error: resume rejected: run configuration/reference/binary changed",
                          file=sys.stderr)
                    raise SystemExit(1)
                recorded = [target["file"] for target in contract["targets"]]
                if targets[:len(recorded)] != recorded:
                    print("Error: resume rejected: targets changed/reordered", file=sys.stderr)
                    raise SystemExit(1)
            elif output_dir.is_dir():
                foreign = sorted(entry.name for entry in output_dir.iterdir()
                                 if entry.name not in ("tar_list.txt", ".opencorr-state"))
                if foreign:
                    print("Error: untracked entry in output directory: " + str(output_dir / foreign[0]),
                          file=sys.stderr)
                    raise SystemExit(1)
            contract_path.parent.mkdir(parents=True, exist_ok=True)
            contract_path.write_text(json.dumps({{
                "fingerprint": fingerprint,
                "targets": [{{"file": target, "output_name": target}} for target in targets],
            }}), encoding="utf-8")
            # No --line means the general full-field mode: emit a small grid.
            if "--line" in args:
                line_ys = [int(v) for v in args[args.index("--line") + 1].split(",")]
            else:
                line_ys = [10, 11, 12]
            # Like the real binary: a line inside the subset margin of these
            # 32-pixel test images is skipped with a warning.
            kept = []
            for y in line_ys:
                if y < 4 or y >= 28:
                    print(f"Warning: line y={{y}} too close to edge, skipping", file=sys.stderr)
                else:
                    kept.append(y)
            output_dir.mkdir(parents=True, exist_ok=True)
            entries = tar_list.read_text(encoding="utf-8").splitlines()
            existing = sum(
                (output_dir / entry.rsplit(maxsplit=1)[1]).exists() for entry in entries
            )
            if existing:
                print(f"Resume: {{existing}} verified files; replay from frame 1 to restore "
                      "temporal state (no overwrite)", file=sys.stderr)
            for entry in entries:
                _, output_name = entry.rsplit(maxsplit=1)
                with (output_dir / output_name).open("w", encoding="utf-8") as out:
                    out.write("x,y,u,v,zncc,uy,vy,ux,vx\\n")
                    for x in (8, 9):
                        for index, y in enumerate(line_ys):
                            if y not in kept:
                                continue
                            u = 1.0 if "zero" in name else 1.0 - 0.5 * index
                            zncc = -4 if "invalidzncc" in name else 0.98
                            out.write(f"{{x}},{{y}},{{u}},0,{{zncc}},0,0,0,0\\n")
            print(f"Processed frames: {{len(entries)}}, skipped: 0", file=sys.stderr)
            print(f"Frames with fallback FFTCC: {{1 if 'fallback' in name else 0}}",
                  file=sys.stderr)
        """), encoding="utf-8")
        self.fake_bin.chmod(0o755)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_event(self, parent: Path, name: str, max_value: int = 200,
                   frames: int = 3, pixels: np.ndarray | None = None) -> Path:
        event = parent / name
        event.mkdir(parents=True)
        if pixels is None:
            pixels = np.full((32, 32), max_value, dtype=np.uint8)
        for frame in range(1, frames + 1):
            Image.fromarray(pixels).save(event / f"{frame:06d}.png")
        return event

    def run_main(self, args: list[str]) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = pipeline.main(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def read_summary(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_json_constant)

    def codes(self, event: dict) -> dict:
        return {item["code"]: item["severity"] for item in event["warning_details"]}

    def base_args(self, event: Path, output: Path) -> list[str]:
        return [
            "--image-dir", str(event),
            "--output-dir", str(output),
            "--line-y", "10,20",
            "--cal", "1",
            "--subset", "5",
            "--gpu-bin", str(self.fake_bin),
        ]

    def test_single_event_summary_and_no_flag_compatibility(self) -> None:
        event = self.make_event(self.root, "事件 with spaces")
        output_with = self.root / "out summary"
        summary_path = output_with / "summary.json"

        return_code = pipeline.main(
            self.base_args(event, output_with) + ["--json-summary", str(summary_path)]
        )
        self.assertEqual(return_code, 0)
        text = summary_path.read_text(encoding="utf-8")
        summary = json.loads(text, parse_constant=reject_json_constant)
        self.assertEqual(summary["binary"]["git_hash"], "fake123")
        self.assertEqual(summary["binary"]["build_date"], "2099-01-01T00:00:00Z")
        self.assertEqual(summary["params"]["subset"], 5)
        self.assertIsNone(summary["params"]["input_depth"])
        self.assertEqual(summary["params"]["input_depth_effective"], "8")
        self.assertEqual(summary["events"][0]["frames"], 2)
        self.assertEqual(summary["events"][0]["poi_per_frame"], 4)
        self.assertEqual(summary["events"][0]["valid_ratio"], 1.0)
        self.assertAlmostEqual(summary["events"][0]["zncc_median"], 0.98)
        self.assertEqual(summary["events"][0]["input_reference"]["effective_depth"], "8")
        self.assertEqual(summary["verdict"], "ok")
        self.assertFalse((output_with / f"_frames_{event.name}").exists())

        cod_with = (output_with / "cod_result.csv").read_bytes()
        self.log_path.write_text("", encoding="utf-8")
        output_without = self.root / "out-no-summary"
        self.assertEqual(pipeline.main(self.base_args(event, output_without)), 0)
        self.assertEqual(cod_with, (output_without / "cod_result.csv").read_bytes())
        invocations = [json.loads(line) for line in self.log_path.read_text().splitlines()]
        self.assertEqual(len(invocations), 1)
        self.assertNotIn("--version", invocations[0])
        self.assertNotIn("--json-summary", invocations[0])

    def test_full_field_mode_writes_frames_and_no_cod(self) -> None:
        event = self.make_event(self.root, "field-event")
        output = self.root / "field-out"
        summary_path = output / "summary.json"
        args = [
            "--image-dir", str(event), "--output-dir", str(output),
            "--subset", "5", "--gpu-bin", str(self.fake_bin),
            "--json-summary", str(summary_path),
        ]

        self.assertEqual(pipeline.main(args), 0)
        invocation = json.loads(self.log_path.read_text().splitlines()[-1])
        self.assertNotIn("--line", invocation)
        frames = sorted((output / "field-event").glob("frame_*.csv"))
        self.assertEqual(len(frames), 2)
        self.assertFalse((output / "cod_result.csv").exists())
        summary = json.loads(summary_path.read_text(), parse_constant=reject_json_constant)
        self.assertEqual(summary["params"]["field_mode"], "full_field")
        self.assertIsNone(summary["params"]["cal"])
        event_summary = summary["events"][0]
        self.assertEqual(event_summary["field_mode"], "full_field")
        self.assertIsNone(event_summary["cod_csv"])
        self.assertIsNone(event_summary["cod_mode"])
        self.assertEqual(event_summary["field_dir"], str((output / "field-event").resolve()))
        self.assertEqual(summary["verdict"], "ok")

    def test_cod_mode_records_the_aggregation_actually_used(self) -> None:
        event = self.make_event(self.root, "cod-mode-event")
        output = self.root / "cod-mode-out"
        summary_path = output / "summary.json"
        args = self.base_args(event, output) + [
            "--cod-method", "naive", "--json-summary", str(summary_path),
        ]
        self.assertEqual(pipeline.main(args), 0)
        summary = json.loads(summary_path.read_text(), parse_constant=reject_json_constant)
        self.assertEqual(summary["events"][0]["cod_mode"], "naive")
        self.assertEqual(summary["params"]["field_mode"], "cod")

    def test_cod_method_defaults_to_naive_in_the_summary(self) -> None:
        # 2026-10-05: without --cod-method a COD run is naive, with no extrapolation fields.
        event = self.make_event(self.root, "default-method-event")
        output = self.root / "default-method-out"
        summary_path = output / "summary.json"
        self.assertEqual(self.run_main(
            self.base_args(event, output) + ["--json-summary", str(summary_path)])[0], 0)
        summary = self.read_summary(summary_path)
        self.assertEqual(summary["params"]["cod_method"], "naive")
        event_summary = summary["events"][0]
        self.assertEqual(event_summary["cod_method"], "naive")
        self.assertEqual(event_summary["cod_mode"], "naive")
        self.assertIsNone(event_summary["geometry"]["extrapolation_target_y"])
        self.assertIsNone(event_summary["geometry"]["extrapolation_px"])
        self.assertEqual(event_summary["cod_stats"]["max_um"], 500.0)
        self.assertIsNone(event_summary["cod_stats"]["naive_at_max_um"])
        self.assertIsNone(event_summary["cod_stats"]["naive_at_min_um"])
        plan_code, plan_stdout, _ = self.run_main(self.base_args(event, self.root / "default-plan")
                                                  + ["--dry-run"])
        self.assertEqual(plan_code, 0)
        self.assertEqual(json.loads(plan_stdout)["resolved_params"]["cod_method"], "naive")

    def test_explicit_fillboundary_is_still_selectable(self) -> None:
        event = self.make_event(self.root, "fillboundary-event")
        output = self.root / "fillboundary-out"
        summary_path = output / "summary.json"
        self.assertEqual(self.run_main(
            self.base_args(event, output)
            + ["--cod-method", "fillboundary", "--json-summary", str(summary_path)])[0], 0)
        summary = self.read_summary(summary_path)
        self.assertEqual(summary["params"]["cod_method"], "fillboundary")
        event_summary = summary["events"][0]
        self.assertEqual(event_summary["cod_mode"], "fillboundary")
        self.assertEqual(event_summary["geometry"]["extrapolation_target_y"], 15.0)
        self.assertEqual(event_summary["cod_stats"]["naive_at_max_um"], 500.0)

    def test_cod_mode_without_cal_is_rejected(self) -> None:
        event = self.make_event(self.root, "no-cal-event")
        output = self.root / "no-cal-out"
        args = [
            "--image-dir", str(event), "--output-dir", str(output),
            "--line-y", "10,20", "--gpu-bin", str(self.fake_bin),
        ]
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(pipeline.main(args), 1)
        self.assertIn("--cal", stderr.getvalue())

    def test_dry_run_full_field_plan_has_no_line_and_no_cod(self) -> None:
        event = self.make_event(self.root, "field-plan")
        output = self.root / "field-plan-out"
        args = [
            "--image-dir", str(event), "--output-dir", str(output),
            "--subset", "5", "--gpu-bin", str(self.fake_bin), "--dry-run",
        ]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(pipeline.main(args), 0)
        plan = json.loads(stdout.getvalue(), parse_constant=reject_json_constant)
        self.assertTrue(plan["valid"])
        self.assertEqual(plan["resolved_params"]["field_mode"], "full_field")
        self.assertNotIn("--line", plan["events"][0]["binary_command"])
        self.assertIsNone(plan["events"][0]["cod_csv"])
        self.assertIsNotNone(plan["events"][0]["field_dir"])

    def test_dry_run_rejects_raw_output_in_cod_mode(self) -> None:
        event = self.make_event(self.root, "raw-cod")
        output = self.root / "raw-cod-out"
        args = self.base_args(event, output) + ["--output-format", "raw", "--dry-run"]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(pipeline.main(args), 2)
        plan = json.loads(stdout.getvalue(), parse_constant=reject_json_constant)
        self.assertTrue(any(
            error["code"] == "cod_requires_csv_frames" for error in plan["errors"]
        ))

    def test_campaign_partial_failure_still_writes_fail_summary(self) -> None:
        campaign = self.root / "campaign"
        self.make_event(campaign, "event-ok")
        self.make_event(campaign, "event-fail")
        short_event = self.make_event(campaign, "event-short")
        (short_event / "000002.png").unlink()
        (short_event / "000003.png").unlink()
        output = self.root / "campaign-out"
        summary_path = output / "summary.json"
        args = [
            "--photo-dir", str(campaign), "--output-dir", str(output),
            "--line-y", "10,20", "--cal", "1", "--subset", "5",
            "--gpu-bin", str(self.fake_bin), "--json-summary", str(summary_path),
        ]

        self.assertEqual(pipeline.main(args), 0)  # 保持既有 batch 部分失败退出行为
        summary = json.loads(summary_path.read_text(), parse_constant=reject_json_constant)
        self.assertEqual(summary["global"], {
            "wall_time_s": summary["global"]["wall_time_s"],
            "events_ok": 1,
            "events_failed": 2,
        })
        self.assertEqual(summary["verdict"], "fail")
        statuses = {event["name"]: event["status"] for event in summary["events"]}
        self.assertEqual(statuses, {
            "event-fail": "fail", "event-ok": "ok", "event-short": "fail",
        })
        short = next(event for event in summary["events"] if event["name"] == "event-short")
        self.assertTrue(short["warnings"][0].startswith("insufficient_images"))

    def test_cli_failure_preserves_existing_frames_and_control_file(self) -> None:
        event = self.make_event(self.root, "single-fail")
        output = self.root / "preserved-out"
        frames = output / "_frames_single-fail"
        frames.mkdir(parents=True)
        sentinel = frames / "frame_999999.csv"
        sentinel.write_text("do not delete prior evidence\n")
        self.assertEqual(pipeline.main(self.base_args(event, output)), 1)
        self.assertEqual(sentinel.read_text(), "do not delete prior evidence\n")
        self.assertTrue((frames / "tar_list.txt").exists())

    def test_single_event_failure_writes_summary_before_exit(self) -> None:
        event = self.make_event(self.root, "single-fail")
        output = self.root / "failed-out"
        summary_path = output / "summary.json"
        args = self.base_args(event, output) + ["--json-summary", str(summary_path)]

        self.assertEqual(pipeline.main(args), 1)
        summary = json.loads(summary_path.read_text(), parse_constant=reject_json_constant)
        self.assertEqual(summary["verdict"], "fail")
        self.assertEqual(summary["global"]["events_failed"], 1)
        self.assertEqual(summary["events"][0]["status"], "fail")

    def test_auto_depth_uses_reference_effective_mode_for_saturation_warning(self) -> None:
        event = self.make_event(self.root, "auto-event", max_value=255)
        output = self.root / "auto-out"
        summary_path = output / "summary.json"
        args = self.base_args(event, output) + [
            "--input-depth", "auto", "--json-summary", str(summary_path),
        ]

        self.assertEqual(pipeline.main(args), 0)
        summary = json.loads(summary_path.read_text(), parse_constant=reject_json_constant)
        self.assertEqual(summary["params"]["input_depth_effective"], "runtime_per_image")
        self.assertEqual(summary["events"][0]["input_reference"]["effective_depth"], "8")
        self.assertTrue(any(
            warning.startswith("input_8bit_max_255")
            for warning in summary["events"][0]["warnings"]
        ))

    def test_preflight_failure_writes_machine_readable_summary(self) -> None:
        output = self.root / "preflight-out"
        summary_path = self.root / "preflight-summary.json"
        args = [
            "--image-dir", str(self.root / "missing-event"),
            "--output-dir", str(output), "--line-y", "10,20", "--cal", "1",
            "--gpu-bin", str(self.fake_bin), "--json-summary", str(summary_path),
        ]

        self.assertEqual(pipeline.main(args), 1)
        summary = json.loads(summary_path.read_text(), parse_constant=reject_json_constant)
        self.assertEqual(summary["verdict"], "fail")
        self.assertEqual(summary["global"]["events_failed"], 1)
        self.assertTrue(summary["warnings"][0].startswith("pipeline_error"))

    def test_dry_run_builds_exact_plan_without_writes_or_processes(self) -> None:
        event = self.make_event(self.root, "dry event")
        output = self.root / "must-not-exist"
        ignored_summary = self.root / "must-not-write-summary.json"
        args = self.base_args(event, output) + [
            "--dry-run", "--json-summary", str(ignored_summary),
        ]
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            return_code = pipeline.main(args)
        plan = json.loads(stdout.getvalue(), parse_constant=reject_json_constant)

        self.assertEqual(return_code, 0)
        self.assertTrue(plan["valid"])
        self.assertEqual(plan["exit_code"], 0)
        self.assertFalse(plan["gpu_started"])
        self.assertFalse(plan["filesystem_writes"])
        self.assertEqual(plan["events"][0]["target_frames"], 2)
        self.assertEqual(plan["events"][0]["first_frame"]["width"], 32)
        command = plan["events"][0]["binary_command"]
        self.assertIn("--conv-threshold", command)
        self.assertNotIn("--dry-run", command)
        self.assertNotIn("--json-summary", command)
        self.assertFalse(output.exists())
        self.assertFalse(ignored_summary.exists())
        self.assertFalse(self.log_path.exists())

    def test_dry_run_invalid_line_returns_json_exit_2(self) -> None:
        event = self.make_event(self.root, "invalid-line")
        output = self.root / "invalid-output"
        args = self.base_args(event, output)
        line_index = args.index("--line-y") + 1
        args[line_index] = "10,99999"
        args.append("--dry-run")
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            return_code = pipeline.main(args)
        plan = json.loads(stdout.getvalue(), parse_constant=reject_json_constant)

        self.assertEqual(return_code, 2)
        self.assertFalse(plan["valid"])
        self.assertTrue(any(
            error["code"] == "line_y_out_of_bounds" for error in plan["errors"]
        ))
        self.assertFalse(output.exists())
        self.assertFalse(self.log_path.exists())

    def test_dry_run_parse_error_is_stdout_json(self) -> None:
        event = self.make_event(self.root, "parse-error")
        output = self.root / "parse-error-out"
        args = self.base_args(event, output)
        subset_index = args.index("--subset") + 1
        args[subset_index] = "not-an-int"
        args.append("--dry-run")
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                pipeline.main(args)
        plan = json.loads(stdout.getvalue(), parse_constant=reject_json_constant)

        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(plan["errors"][0]["code"], "argument_parse_error")
        self.assertFalse(output.exists())

    def test_every_dry_run_json_exit_has_the_same_top_level_fields(self) -> None:
        """Review 2026-10-04 recheck N1: notes (and the rest) also in the parse-error and exception JSON."""
        contract = json.loads(pipeline.INTERFACE_CONTRACT_PATH.read_text(encoding="utf-8"))["outputs"]["dry_run"]
        required = set(contract["required_top_level_fields"])
        full_extra = set(contract["full_plan_additional_fields"])
        event = self.make_event(self.root, "keys")

        def plan_of(args: list[str], *, exits: bool = False) -> dict:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                if exits:
                    with self.assertRaises(SystemExit):
                        pipeline.main(args)
                else:
                    pipeline.main(args)
            return json.loads(stdout.getvalue(), parse_constant=reject_json_constant)

        valid = plan_of(self.base_args(event, self.root / "keys-valid") + ["--dry-run"])
        invalid = plan_of(self.base_args(event, self.root / "keys-invalid") + ["--line-y", "10,999", "--dry-run"])
        strict = plan_of(self.base_args(event, self.root / "keys-strict") + ["--strict", "--dry-run"])
        parse_error = plan_of(self.base_args(event, self.root / "keys-parse")
                              + ["--subset", "x", "--dry-run"], exits=True)
        abbreviated = plan_of(self.base_args(event, self.root / "keys-abbrev") + ["--subset", "x", "--dry"],
                              exits=True)
        with mock.patch.object(pipeline, "build_dry_run", side_effect=RuntimeError("synthetic")):
            exception = plan_of(self.base_args(event, self.root / "keys-exception") + ["--dry-run"])
        self.assertTrue(valid["valid"])
        self.assertFalse(invalid["valid"])
        self.assertEqual(parse_error["errors"][0]["code"], "argument_parse_error")
        self.assertEqual(abbreviated["errors"][0]["code"], "argument_parse_error")
        self.assertEqual(exception["errors"][0]["code"], "dry_run_exception")
        for name, plan in (("valid", valid), ("invalid", invalid), ("strict", strict)):
            with self.subTest(exit=name):
                self.assertEqual(set(plan), required | full_extra)
        for name, plan in (("parse_error", parse_error), ("abbreviated", abbreviated), ("exception", exception)):
            with self.subTest(exit=name):
                self.assertEqual(set(plan), required)
                self.assertEqual(plan["notes"], [])
        self.assertEqual(contract["minimal_variants"], ["argument_parse_error", "dry_run_exception"])

    def test_dry_run_line_y_overrides_interface_and_gauge(self) -> None:
        event = self.make_event(self.root, "line-override")
        output = self.root / "line-override-out"
        args = self.base_args(event, output) + [
            "--interface-y", "99999", "--gauge", "0", "--dry-run",
        ]
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            return_code = pipeline.main(args)
        plan = json.loads(stdout.getvalue(), parse_constant=reject_json_constant)

        self.assertEqual(return_code, 0)
        self.assertTrue(plan["valid"])
        self.assertEqual(plan["resolved_params"]["line_ys"], [10, 20])
        self.assertFalse(output.exists())

    def test_image_probe_recommends_left_aligned_12_bit(self) -> None:
        image_path = self.root / "left12.tif"
        values = (np.arange(64, dtype=np.uint16).reshape(8, 8) * 1024)
        Image.fromarray(values).save(image_path)

        probe = pipeline.probe_image(str(image_path))
        recommendation = pipeline.recommend_input_mode(probe)

        self.assertTrue(probe["low_4_bits_all_zero"])
        self.assertGreater(probe["pixel_max"], 4095)
        self.assertEqual(recommendation["input_depth"], "12")
        self.assertEqual(recommendation["input_align"], "left")
        self.assertNotIn("highspeed2", recommendation["auto_limitation"])

    def test_png_probe_reports_storage_depth_even_for_dark_or_renamed_files(self) -> None:
        # A dark uint16 PNG must not be guessed to be 8-bit from its maximum.
        # On older Pillow its decoded dtype is int32, but its storage is 16-bit.
        for name, values, depth in (
            ("eight", np.array([[0, 16], [100, 255]], dtype=np.uint8), 8),
            ("dark16", np.array([[0, 16], [100, 255]], dtype=np.uint16), 16),
            ("bright16", np.array([[0, 16], [30000, 65535]], dtype=np.uint16), 16),
        ):
            with self.subTest(name=name):
                path = self.root / f"{name}.dat"
                Image.fromarray(values).save(path, format="PNG")
                self.assertEqual(path.read_bytes()[24], depth)  # PNG IHDR bit depth
                with Image.open(path) as image:
                    mode, dtype = image.mode, str(np.asarray(image).dtype)
                    np.testing.assert_array_equal(np.asarray(image), values)
                probe = pipeline.probe_image(str(path))
                saturation = pipeline.probe_input_saturation(str(path), "8")
                self.assertEqual(probe["container_bit_depth"], depth)
                self.assertEqual(saturation["container_bit_depth"], depth)
                self.assertEqual((probe["pillow_mode"], probe["dtype"]), (mode, dtype))
                self.assertEqual(probe["pixel_max"], int(values.max()))
                expected = values.astype(np.uint64) >> 8 if depth == 16 else values
                self.assertEqual(saturation["pixel_max"], int(expected.max()))
                self.assertEqual(saturation["saturated_fraction"],
                                 float(np.count_nonzero(expected == 255) / expected.size))

    def test_int32_tiff_probe_keeps_its_real_storage_depth(self) -> None:
        path = self.root / "integer32.tif"
        values = np.array([[0, 255], [65535, 100000]], dtype=np.int32)
        Image.fromarray(values).save(path)
        with Image.open(path) as image:
            self.assertEqual(image.tag_v2[258], (32,))  # TIFF BitsPerSample
            self.assertEqual(image.mode, "I")
        probe = pipeline.probe_image(str(path))
        saturation = pipeline.probe_input_saturation(str(path), "16")
        self.assertEqual(probe["container_bit_depth"], 32)
        self.assertEqual(saturation["container_bit_depth"], 32)
        self.assertEqual(probe["dtype"], "int32")
        self.assertEqual(probe["pixel_max"], 100000)
        self.assertEqual(saturation["pixel_max"], 100000)

    # ── review-2026-10-03 batch 2: truthful results, --strict, diagnostics ──

    def test_empty_cod_fails_the_event_but_keeps_the_legacy_exit_code(self) -> None:
        event = self.make_event(self.root, "edge-event")
        output = self.root / "edge-out"
        summary_path = output / "summary.json"
        args = self.base_args(event, output) + ["--json-summary", str(summary_path)]
        args[args.index("--line-y") + 1] = "10,30"  # L1 inside the fake binary's margin

        code, stdout, _ = self.run_main(args)
        self.assertEqual(code, 0)  # historical exit code: the old pipeline printed a check mark
        self.assertNotIn("✅ 完成", stdout)
        self.assertIn("❌", stdout)
        summary = self.read_summary(summary_path)
        event_summary = summary["events"][0]
        self.assertEqual(summary["verdict"], "fail")
        self.assertEqual(event_summary["status"], "fail")
        self.assertEqual(summary["global"]["events_failed"], 1)
        codes = self.codes(event_summary)
        self.assertEqual(codes["cod_empty"], "fail")
        self.assertEqual(codes["cod_line_missing"], "fail")
        self.assertEqual(codes["binary_warning"], "warn")
        self.assertTrue(any("line y=30 too close to edge" in text
                            for text in event_summary["warnings"]))
        self.assertIsNone(event_summary["cod_csv"])
        self.assertEqual(event_summary["cod_stats"]["rows"], 0)
        self.assertEqual(event_summary["cod_stats"]["line_rows_present"], {"L0": True, "L1": False})
        # The header-only file is still written and, as in 638ad18, the frames are
        # deleted without --keep-frames (review 2026-10-04 R1).
        self.assertEqual((output / "cod_result.csv").read_text(), "frame,x_mm,cod_um\n")
        self.assertFalse((output / f"_frames_{event.name}").exists())
        # field_dir keeps its 638ad18 value; field_dir_exists says it is gone.
        self.assertEqual(event_summary["field_dir"], str((output / f"_frames_{event.name}").resolve()))
        self.assertFalse(event_summary["field_dir_exists"])
        self.assertIn("deleted as usual", next(
            item["detail"] for item in event_summary["warning_details"] if item["code"] == "cod_empty"))

        # An empty COD the dry-run cannot predict (every row invalid): default
        # exit 0 as before, --strict exit 3 and the frames stay for diagnosis.
        invalid = self.make_event(self.root, "invalidzncc-event")
        for strict, expected in ((False, 0), (True, pipeline.EXIT_STRICT_EVENT_FAILED)):
            with self.subTest(strict=strict):
                out = self.root / f"invalid-out-{strict}"
                extra = ["--json-summary", str(out / "summary.json")] + (["--strict"] if strict else [])
                code, _, _ = self.run_main(self.base_args(invalid, out) + extra)
                self.assertEqual(code, expected)
                event_summary = self.read_summary(out / "summary.json")["events"][0]
                self.assertEqual(event_summary["status"], "fail")
                self.assertIn("cod_empty", self.codes(event_summary))
                self.assertNotIn("cod_line_missing", self.codes(event_summary))
                self.assertEqual((out / "_frames_invalidzncc-event").is_dir(), strict)

    def test_strict_rejects_what_the_dry_run_rejects_before_running(self) -> None:
        event = self.make_event(self.root, "strict-edge")
        output = self.root / "strict-edge-out"
        summary_path = self.root / "strict-edge-summary.json"
        args = self.base_args(event, output) + ["--json-summary", str(summary_path), "--strict"]
        args[args.index("--line-y") + 1] = "10,30"

        code, _, stderr = self.run_main(args)
        self.assertEqual(code, pipeline.EXIT_INVALID)
        self.assertIn("line_y_inside_binary_margin", stderr)
        self.assertFalse(self.log_path.exists())  # the binary never ran
        self.assertFalse(output.exists())
        summary = self.read_summary(summary_path)
        self.assertEqual(summary["verdict"], "fail")
        self.assertTrue(all(text.startswith("pipeline_error") for text in summary["warnings"]))
        self.assertTrue(any("line_y_inside_binary_margin" in text for text in summary["warnings"]))

    def test_all_zero_cod_is_a_warning_and_a_failure_under_strict(self) -> None:
        event = self.make_event(self.root, "zero-event")
        output = self.root / "zero-out"
        summary_path = output / "summary.json"
        args = self.base_args(event, output) + ["--json-summary", str(summary_path)]

        code, stdout, _ = self.run_main(args)
        self.assertEqual(code, 0)
        self.assertIn("⚠️", stdout)
        summary = self.read_summary(summary_path)
        self.assertEqual(summary["verdict"], "warn")
        self.assertEqual(summary["events"][0]["status"], "ok")
        self.assertEqual(self.codes(summary["events"][0])["cod_all_zero"], "warn")
        self.assertTrue(summary["events"][0]["cod_stats"]["all_zero"])

        strict_output = self.root / "zero-strict-out"
        strict_summary = strict_output / "summary.json"
        args[args.index("--output-dir") + 1] = str(strict_output)
        args[args.index("--json-summary") + 1] = str(strict_summary)
        code, _, _ = self.run_main(args + ["--strict"])
        self.assertEqual(code, pipeline.EXIT_STRICT_EVENT_FAILED)
        summary = self.read_summary(strict_summary)
        self.assertEqual(summary["events"][0]["status"], "fail")
        self.assertEqual(self.codes(summary["events"][0])["cod_all_zero"], "fail")
        self.assertIsNone(summary["events"][0]["cod_csv"])

    def test_failed_event_does_not_point_at_a_stale_cod_csv(self) -> None:
        event = self.make_event(self.root, "stale-fail")
        output = self.root / "stale-out"
        output.mkdir()
        stale = output / "cod_result.csv"
        stale.write_text("frame,x_mm,cod_um\n2,8.0000,123.0000\n", encoding="utf-8")
        summary_path = output / "summary.json"

        code, _, _ = self.run_main(
            self.base_args(event, output) + ["--json-summary", str(summary_path)]
        )
        self.assertEqual(code, 1)
        self.assertEqual(stale.read_text(), "frame,x_mm,cod_um\n2,8.0000,123.0000\n")
        event_summary = self.read_summary(summary_path)["events"][0]
        self.assertIsNone(event_summary["cod_csv"])
        self.assertIn("stale_cod_csv", self.codes(event_summary))
        # warnings keeps the 638ad18 text; the exit code is in warning_details.
        self.assertEqual(event_summary["warnings"][0], "gpu_cli_failed")
        self.assertEqual(event_summary["warning_details"][0]["detail"], "exit=7")

    def test_successful_rerun_reports_the_replaced_cod_csv(self) -> None:
        event = self.make_event(self.root, "replace-event")
        output = self.root / "replace-out"
        output.mkdir()
        (output / "cod_result.csv").write_text("old\n", encoding="utf-8")
        summary_path = output / "summary.json"
        self.assertEqual(self.run_main(
            self.base_args(event, output) + ["--json-summary", str(summary_path)])[0], 0)
        event_summary = self.read_summary(summary_path)["events"][0]
        self.assertEqual(self.codes(event_summary)["cod_csv_replaced"], "warn")
        self.assertEqual(event_summary["cod_csv"], str((output / "cod_result.csv").resolve()))
        self.assertNotEqual((output / "cod_result.csv").read_text(), "old\n")
        self.assertEqual(sorted(p.name for p in output.glob(".cod_result.csv.*")), [])

    def test_campaign_exit_codes_default_and_strict(self) -> None:
        campaign = self.root / "campaign"
        self.make_event(campaign, "event-ok")
        self.make_event(campaign, "event-fail")
        output = self.root / "campaign-default"
        code, stdout, _ = self.run_main([
            "--photo-dir", str(campaign), "--output-dir", str(output),
            "--line-y", "10,20", "--cal", "1", "--subset", "5",
            "--gpu-bin", str(self.fake_bin),
        ])
        self.assertEqual(code, 0)
        self.assertNotIn("✅ 全部完成", stdout)
        self.assertIn("event-fail", stdout.split("运行结束")[-1])

        code, stdout, _ = self.run_main([
            "--photo-dir", str(campaign), "--output-dir", str(self.root / "campaign-strict"),
            "--line-y", "10,20", "--cal", "1", "--subset", "5",
            "--gpu-bin", str(self.fake_bin), "--strict",
        ])
        self.assertEqual(code, pipeline.EXIT_STRICT_EVENT_FAILED)

        clean = self.root / "campaign-clean"
        self.make_event(clean, "event-a")
        code, stdout, _ = self.run_main([
            "--photo-dir", str(clean), "--output-dir", str(self.root / "campaign-clean-out"),
            "--line-y", "10,20", "--cal", "1", "--subset", "5",
            "--gpu-bin", str(self.fake_bin), "--strict",
        ])
        self.assertEqual(code, 0)
        self.assertIn("✅ 全部完成", stdout)

    def test_relative_gpu_bin_is_made_absolute_for_dry_run_and_run(self) -> None:
        event = self.make_event(self.root, "relative-bin")
        output = self.root / "relative-out"
        previous = os.getcwd()
        os.chdir(self.root)
        try:
            args = self.base_args(event, output)
            args[args.index("--gpu-bin") + 1] = "./fake_gpu.py"
            code, stdout, _ = self.run_main(args + ["--dry-run"])
            self.assertEqual(code, 0)
            plan = json.loads(stdout, parse_constant=reject_json_constant)
            command0 = plan["events"][0]["binary_command"][0]
            self.assertTrue(os.path.isabs(command0))
            self.assertEqual(os.path.realpath(command0), os.path.realpath(self.fake_bin))
            self.assertEqual(self.run_main(args)[0], 0)
        finally:
            os.chdir(previous)
        self.assertTrue((output / "cod_result.csv").is_file())

    def test_non_finite_parameters_are_rejected_where_they_are_used(self) -> None:
        event = self.make_event(self.root, "nan-event")
        # COD mode multiplies by --cal: NaN/inf stop the run (exit 2); the dry-run says why.
        for value in ("nan", "inf"):
            with self.subTest(cal=value):
                args = self.base_args(event, self.root / f"nan-out-{value}")
                args[args.index("--cal") + 1] = value
                with self.assertRaises(SystemExit) as raised:
                    self.run_main(args)
                self.assertEqual(raised.exception.code, 2)
                code, stdout, _ = self.run_main(args + ["--dry-run"])
                self.assertEqual(code, 2)
                plan = json.loads(stdout, parse_constant=reject_json_constant)
                self.assertIn("invalid_cal", [error["code"] for error in plan["errors"]])
        # --conv-threshold always reaches the binary: still a parse error.
        args = self.base_args(event, self.root / "nan-conv") + ["--conv-threshold", "nan"]
        with self.assertRaises(SystemExit) as raised:
            self.run_main(args)
        self.assertEqual(raised.exception.code, 2)
        code_out = io.StringIO()
        with contextlib.redirect_stdout(code_out), self.assertRaises(SystemExit):
            pipeline.main(args + ["--dry-run"])
        plan = json.loads(code_out.getvalue(), parse_constant=reject_json_constant)
        self.assertEqual(plan["errors"][0]["code"], "argument_parse_error")
        self.assertFalse(self.log_path.exists())
        # Full-field mode never reads --cal: 638ad18 ran this command, and so does this one.
        field_args = ["--image-dir", str(event), "--output-dir", str(self.root / "nan-field"),
                      "--subset", "5", "--gpu-bin", str(self.fake_bin), "--cal", "nan"]
        code, stdout, _ = self.run_main(field_args + ["--dry-run"])
        self.assertEqual(code, 0, stdout)
        self.assertEqual(self.run_main(field_args)[0], 0)

    def test_unambiguous_abbreviations_work_as_in_638ad18(self) -> None:
        """Review 2026-10-04 R3: --dry/--desc/--keep/--json are prefixes argparse accepts."""
        event = self.make_event(self.root, "abbrev")
        code, stdout, _ = self.run_main(self.base_args(event, self.root / "abbrev-dry") + ["--dry"])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(stdout, parse_constant=reject_json_constant)["dry_run"])
        self.assertFalse(self.log_path.exists())
        # A parse error of an abbreviated dry-run is still machine-readable JSON.
        bad = self.base_args(event, self.root / "abbrev-bad") + ["--dry", "--subset", "x"]
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer), self.assertRaises(SystemExit) as raised:
            pipeline.main(bad)
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(json.loads(stdout_buffer.getvalue())["errors"][0]["code"], "argument_parse_error")
        code, stdout, _ = self.run_main(["--desc"])
        self.assertEqual((code, stdout), (0, pipeline.INTERFACE_CONTRACT_PATH.read_text(encoding="utf-8")))
        code, _, stderr = self.run_main(["--desc", "--image-dir", str(event)])
        self.assertEqual(code, 2)
        self.assertIn("--describe", stderr)
        output = self.root / "abbrev-keep"
        summary_path = output / "summary.json"
        args = self.base_args(event, output) + ["--keep", "--json", str(summary_path)]
        args[args.index("--image-dir")] = "--image-d"
        self.assertEqual(self.run_main(args)[0], 0)
        self.assertTrue((output / "_frames_abbrev").is_dir())
        self.assertTrue(summary_path.is_file())
        with self.assertRaises(SystemExit) as raised:  # --d is ambiguous: --dry-run / --describe
            self.run_main(self.base_args(event, self.root / "abbrev-ambiguous") + ["--d"])
        self.assertEqual(raised.exception.code, 2)

    def test_binary_error_line_log_and_remediation_reach_the_summary(self) -> None:
        for name, needle, advice in (
            ("errorusage-event", "Error: --conv-threshold must be a finite positive number",
             "rerun the dry-run binary_command"),
            ("resumereject-event", "Error: resume rejected", "use a new --output-dir"),
        ):
            with self.subTest(name=name):
                event = self.make_event(self.root, name)
                output = self.root / f"{name}-out"
                summary_path = output / "summary.json"
                code, stdout, _ = self.run_main(
                    self.base_args(event, output) + ["--json-summary", str(summary_path)]
                )
                self.assertEqual(code, 1)
                self.assertIn(needle, stdout)
                event_summary = self.read_summary(summary_path)["events"][0]
                self.assertEqual(event_summary["binary_exit_code"], 1)
                self.assertEqual(event_summary["warnings"][0], "gpu_cli_failed")
                self.assertTrue(event_summary["warning_details"][0]["detail"].startswith(
                    f"exit=1: {needle}"))
                self.assertIn(advice, event_summary["remediation"])
                log = Path(event_summary["binary_stderr_log"])
                self.assertEqual(log, (output / f"_frames_{name}.binary.log").resolve())
                self.assertIn(needle, log.read_text(encoding="utf-8"))
                # The log must not sit inside the frame directory the binary scans.
                self.assertFalse(any(p.suffix == ".log" for p in (output / f"_frames_{name}").iterdir()))

    def test_dry_run_argv_matches_the_real_run(self) -> None:
        event = self.make_event(self.root, "argv-event")
        output = self.root / "argv-out"
        args = self.base_args(event, output)
        code, stdout, _ = self.run_main(args + ["--dry-run"])
        self.assertEqual(code, 0)
        planned = json.loads(stdout, parse_constant=reject_json_constant)["events"][0]["binary_command"]
        self.assertEqual(self.run_main(args)[0], 0)
        executed = json.loads(self.log_path.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(planned[0], str(self.fake_bin))
        self.assertEqual(planned[1:], executed)

    def test_max_frames_limits_plan_run_and_summary_alike(self) -> None:
        event = self.make_event(self.root, "max-frames", frames=6)
        output = self.root / "max-frames-out"
        summary_path = output / "summary.json"
        args = self.base_args(event, output) + ["--max-frames", "2"]
        code, stdout, _ = self.run_main(args + ["--dry-run"])
        self.assertEqual(code, 0)
        plan = json.loads(stdout, parse_constant=reject_json_constant)
        self.assertEqual(plan["events"][0]["target_frames"], 2)
        self.assertEqual(plan["events"][0]["target_frames_available"], 5)
        self.assertEqual(plan["events"][0]["target_frame_range"], [2, 3])
        # Information, so it is a note and the warnings list stays empty.
        self.assertIn("max_frames_applied", [w["code"] for w in plan["notes"]])
        self.assertEqual(plan["warnings"], [])

        # --keep-frames keeps the frame directory, so the tar list check means something.
        self.assertEqual(self.run_main(
            args + ["--keep-frames", "--json-summary", str(summary_path)])[0], 0)
        frame_dir = output / f"_frames_{event.name}"
        self.assertTrue(frame_dir.is_dir())
        self.assertFalse((frame_dir / "tar_list.txt").exists())  # removed after success, as before
        executed = json.loads(self.log_path.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(executed[executed.index("--ref") + 1], str(event / "000001.png"))
        event_summary = self.read_summary(summary_path)["events"][0]
        self.assertEqual(event_summary["frames"], 2)
        self.assertEqual(event_summary["target_frames_selected"], 2)
        self.assertEqual(event_summary["target_frames_available"], 5)
        self.assertEqual(self.codes(event_summary)["max_frames_applied"], "info")
        self.assertEqual(self.read_summary(summary_path)["verdict"], "ok")
        rows = (output / "cod_result.csv").read_text().splitlines()[1:]
        self.assertEqual(sorted({row.split(",")[0] for row in rows}), ["2", "3"])

    def test_severity_decides_the_verdict(self) -> None:
        args = pipeline.create_parser().parse_args([
            "--image-dir", "/tmp/event", "--output-dir", "/tmp/out",
            "--line-y", "10,20", "--cal", "1",
        ])
        binary = {"path": "fake", "git_hash": "abc", "build_date": "now"}

        def event(*warnings: str) -> dict:
            return {"ok": True, "status": "ok", "name": "e", "frames": 2, "warnings": list(warnings)}

        info = pipeline.build_run_summary(args, binary, [], [event("max_frames_applied: x")], 1.0)
        self.assertEqual(info["verdict"], "ok")
        self.assertEqual(info["events"][0]["warning_details"][0]["severity"], "info")
        warn = pipeline.build_run_summary(args, binary, [], [event("cod_all_zero: x")], 1.0)
        self.assertEqual(warn["verdict"], "warn")
        fail = pipeline.build_run_summary(args, binary, [], [event("cod_empty: x")], 1.0)
        self.assertEqual(fail["verdict"], "fail")
        self.assertEqual(fail["events"][0]["status"], "fail")
        self.assertEqual(fail["global"]["events_failed"], 1)
        top_info = pipeline.build_run_summary(
            args, binary, ["gpu_bin_outside_checkout: x"], [event()], 1.0)
        self.assertEqual(top_info["verdict"], "ok")
        self.assertEqual(top_info["warning_details"][0]["severity"], "info")
        args.strict = True
        strict = pipeline.build_run_summary(args, binary, [], [event("cod_all_zero: x")], 1.0)
        self.assertEqual(strict["verdict"], "fail")
        unknown = pipeline.build_run_summary(args, binary, [], [event("new_code: x")], 1.0)
        self.assertEqual(unknown["verdict"], "warn")  # unknown codes are never silenced

    def field_args(self, event: Path, output: Path) -> list[str]:
        """base_args without measurement lines: full-field mode."""
        return ["--image-dir", str(event), "--output-dir", str(output), "--subset", "5",
                "--gpu-bin", str(self.fake_bin)]

    def test_saturation_grades_the_line_bands_in_cod_mode_and_the_frame_in_full_field(self) -> None:
        """2026-10-04: COD mode grades the pixels of the subsets the COD uses (the largest
        band fraction), full-field mode the whole frame; the default threshold is 0.05."""
        in_band = np.full((32, 32), 120, dtype=np.uint8)
        in_band[9:12, 10:15] = 255  # 15 px in the L0 band (rows 8-12, cols 2-29): 10.7 %; frame 1.5 %
        outside = np.full((32, 32), 120, dtype=np.uint8)
        outside[26:, :] = 255  # 192 px away from both bands: frame 18.75 %, bands 0
        for name, pixels, args_of, extra, basis, value, expected in (
            ("band-cod", in_band, self.base_args, [], "line_bands", 15 / 140, "warn"),
            ("band-field", in_band, self.field_args, [], "frame", 15 / 1024, "info"),
            ("outside-cod", outside, self.base_args, [], "line_bands", 0.0, "info"),
            ("outside-field", outside, self.field_args, [], "frame", 192 / 1024, "warn"),
            ("band-cod-0.2", in_band, self.base_args, ["--saturation-warn-fraction", "0.2"],
             "line_bands", 15 / 140, "info"),
        ):
            with self.subTest(name=name):
                event = self.make_event(self.root, name, pixels=pixels)
                output = self.root / f"{name}-out"
                summary_path = output / "summary.json"
                self.assertEqual(self.run_main(
                    args_of(event, output) + extra + ["--json-summary", str(summary_path)])[0], 0)
                summary = self.read_summary(summary_path)
                event_summary = summary["events"][0]
                saturation = event_summary["input_saturation"]
                threshold = 0.2 if extra else 0.05
                self.assertEqual((saturation["basis"], saturation["threshold"]), (basis, threshold))
                self.assertEqual(summary["params"]["saturation_warn_fraction"], threshold)
                self.assertAlmostEqual(saturation["value"], value)
                self.assertAlmostEqual(event_summary["input_saturated_fraction"],
                                       int(np.count_nonzero(pixels == 255)) / pixels.size)
                self.assertEqual(saturation["frame_fraction"], event_summary["input_saturated_fraction"])
                self.assertEqual(saturation["severity"], expected)
                self.assertEqual(self.codes(event_summary)["input_8bit_max_255"], expected)
                self.assertEqual(summary["verdict"], "ok" if expected == "info" else "warn")
                # warnings: the exact 638ad18 text when it is a warning, nothing when
                # it is information (review 2026-10-04 R2); the numbers are in the detail.
                self.assertEqual(event_summary["warnings"], [] if expected == "info" else [
                    "input_8bit_max_255: possible saturation_or_high_bit_truncation"])
                detail = next(item["detail"] for item in event_summary["warning_details"]
                              if item["code"] == "input_8bit_max_255")
                self.assertIn(f"warn above {threshold:.2%}", detail)
                self.assertIn("of reference pixels are 255", detail)
                if basis == "line_bands":
                    self.assertEqual([line["y"] for line in saturation["lines"]], [10, 20])
                    self.assertIn(f"largest measurement-line band {value:.4%}", detail)
                    self.assertIn("L0 y=10 rows 8-12 cols 2-29", detail)
                else:
                    self.assertEqual(saturation["lines"], [])

        deep = np.full((32, 32), 30000, dtype=np.uint16)
        deep[0, :2] = 65535  # high byte 255 outside both bands: still always a warning
        event = self.make_event(self.root, "sat-deep", pixels=deep)
        summary_path = self.root / "sat-deep-out" / "summary.json"
        self.assertEqual(self.run_main(
            self.base_args(event, self.root / "sat-deep-out") + ["--json-summary", str(summary_path)]
        )[0], 0)
        deep_event = self.read_summary(summary_path)["events"][0]
        self.assertEqual(self.codes(deep_event)["input_8bit_max_255"], "warn")
        self.assertEqual(deep_event["input_saturation"]["value"], 0.0)
        self.assertEqual(deep_event["input_saturation"]["container_bit_depth"], 16)
        self.assertEqual(deep_event["input_saturation"]["severity"], "warn")
        self.assertEqual(deep_event["warnings"],
                         ["input_8bit_max_255: possible saturation_or_high_bit_truncation"])
        # The detail names the real reason, the 16-bit container (R13).
        detail = deep_event["warning_details"][0]["detail"]
        self.assertIn("16-bit container read as 8-bit", detail)
        self.assertNotIn("warn above", detail)

    def test_saturation_bands_follow_the_effective_geometry(self) -> None:
        """Band rows L +/- r and columns from --roi-cols or the binary's POI range, with the
        lines, radius and ROI the run itself uses (32 px wide, radius 2: POIs at x 4..27)."""
        pixels = np.full((32, 32), 120, dtype=np.uint8)
        pixels[0, 0] = 255
        event = self.make_event(self.root, "geometry-event", pixels=pixels)
        for name, extra, cols, subsets in (
            ("lines", [], [2, 29], 24),
            ("derived", ["--interface-y", "15", "--gauge", "10"], [2, 29], 24),
            ("roi", ["--roi-cols", "6,20"], [4, 22], 15),
            ("roi-edges", ["--roi-cols", "0,31"], [0, 31], 24),  # x0 - r, x1 + r clipped
            ("step", ["--step", "3"], [2, 27], 8),  # POIs at x 4, 7, ..., 25
        ):
            with self.subTest(name=name):
                args = self.base_args(event, self.root / f"geometry-{name}") + extra
                if name == "derived":
                    index = args.index("--line-y")
                    del args[index:index + 2]
                code, stdout, _ = self.run_main(args + ["--dry-run"])
                self.assertEqual(code, 0, stdout)
                plan = json.loads(stdout, parse_constant=reject_json_constant)
                geometry = plan["resolved_params"]["geometry"]
                radius = geometry["subset_radius"]
                saturation = plan["events"][0]["input_saturation"]
                self.assertEqual(geometry["line_ys"], [10, 20])
                self.assertEqual([line["y"] for line in saturation["lines"]], geometry["line_ys"])
                for line in saturation["lines"]:
                    self.assertEqual(line["band_rows"], [line["y"] - radius, line["y"] + radius])
                    self.assertEqual(line["band_cols"], cols)
                    self.assertEqual(line["subsets"], subsets)
                    self.assertEqual(line["band_fraction"], 0.0)
                self.assertEqual(saturation["frame_fraction"], 1 / 1024)

    def test_dry_run_forecasts_the_runs_input_saturation_and_level(self) -> None:
        """The plan's events[].input_saturation is the run's, its severity the run's level."""
        mixed = np.full((32, 32), 120, dtype=np.uint8)
        mixed[9:12, 10:15] = 255
        mixed[26:, :] = 255
        dark = np.full((32, 32), 120, dtype=np.uint8)  # no pixel at 255: no warning
        deep = np.full((32, 32), 30000, dtype=np.uint16)
        deep[0, :2] = 65535
        for name, pixels, args_of, extra in (
            ("cod", mixed, self.base_args, []),
            ("cod-roi", mixed, self.base_args, ["--roi-cols", "6,12"]),
            ("cod-threshold", mixed, self.base_args, ["--saturation-warn-fraction", "0.2"]),
            ("cod-auto", mixed, self.base_args, ["--input-depth", "auto"]),
            ("field", mixed, self.field_args, []),
            ("dark", dark, self.base_args, []),
            ("deep", deep, self.base_args, []),
        ):
            with self.subTest(name=name):
                event = self.make_event(self.root, f"forecast-{name}", pixels=pixels)
                output = self.root / f"forecast-{name}-out"
                args = args_of(event, output) + extra
                code, stdout, _ = self.run_main(args + ["--dry-run"])
                self.assertEqual(code, 0, stdout)
                plan = json.loads(stdout, parse_constant=reject_json_constant)
                planned = plan["events"][0]["input_saturation"]
                self.assertNotIn("input_8bit_max_255", [item["code"] for item in plan["warnings"]])
                self.assertEqual(self.run_main(
                    args + ["--json-summary", str(output / "summary.json")])[0], 0)
                event_summary = self.read_summary(output / "summary.json")["events"][0]
                self.assertIsNotNone(planned)
                self.assertEqual(planned, event_summary["input_saturation"])
                self.assertEqual(planned["severity"],
                                 self.codes(event_summary).get("input_8bit_max_255"))
        contract = json.loads(pipeline.INTERFACE_CONTRACT_PATH.read_text(encoding="utf-8"))["outputs"]
        self.assertIn("input_saturation", contract["json_summary"]["event_fields"])
        self.assertIn("input_saturation", contract["dry_run"]["event_fields"])
        self.assertEqual(list(planned), contract["json_summary"]["input_saturation_fields"])
        self.assertEqual(list(event_summary["input_saturation"]["lines"][0]),
                         contract["json_summary"]["input_saturation_line_fields"])
        # Read at another depth, there is nothing to grade.
        code, stdout, _ = self.run_main(
            self.base_args(self.make_event(self.root, "forecast-12", pixels=dark), self.root / "f12")
            + ["--input-depth", "12", "--input-align", "right", "--dry-run"])
        self.assertIsNone(json.loads(stdout)["events"][0]["input_saturation"])

    def test_default_mode_reports_rejected_arguments_and_strict_refuses_them(self) -> None:
        event = self.make_event(self.root, "even-subset")
        output = self.root / "even-out"
        summary_path = output / "summary.json"
        args = self.base_args(event, output) + ["--json-summary", str(summary_path)]
        args[args.index("--subset") + 1] = "6"
        code, stdout, stderr = self.run_main(args)
        self.assertEqual(code, 0)
        self.assertIn("invalid_subset_size", stderr)
        self.assertIn("⚠️ 完成", stdout)
        self.assertNotIn("✅ 完成", stdout)
        summary = self.read_summary(summary_path)
        self.assertEqual(summary["verdict"], "warn")
        self.assertTrue(summary["warnings"][0].startswith("argument_warning: invalid_subset_size"))
        executed = json.loads(self.log_path.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(executed[executed.index("--output-dir") + 2], "2")  # radius 2, as before

        self.log_path.unlink()
        strict_args = self.base_args(event, self.root / "even-strict") + ["--subset", "6", "--strict"]
        code, _, stderr = self.run_main(strict_args)
        self.assertEqual(code, pipeline.EXIT_INVALID)
        self.assertIn("invalid_subset_size", stderr)
        self.assertFalse(self.log_path.exists())

    def test_summary_write_failure_falls_back_to_a_failure_summary(self) -> None:
        event = self.make_event(self.root, "summary-fail")
        output = self.root / "summary-fail-out"
        summary_path = output / "summary.json"
        with mock.patch.object(pipeline, "build_run_summary", side_effect=ValueError("boom")):
            code, _, stderr = self.run_main(
                self.base_args(event, output) + ["--json-summary", str(summary_path)]
            )
        self.assertEqual(code, 1)
        self.assertIn("boom", stderr)
        summary = self.read_summary(summary_path)
        self.assertEqual(summary["verdict"], "fail")
        self.assertTrue(summary["warnings"][0].startswith("pipeline_error: summary_write_failed"))

    def test_dry_run_reports_output_state_and_duplicate_frames(self) -> None:
        event = self.make_event(self.root, "state-event")
        occupied = self.root / "occupied"
        occupied.write_text("a file, not a directory", encoding="utf-8")
        code, stdout, _ = self.run_main(self.base_args(event, occupied) + ["--dry-run"])
        self.assertEqual(code, 2)
        plan = json.loads(stdout, parse_constant=reject_json_constant)
        self.assertIn("output_dir_not_directory", [e["code"] for e in plan["errors"]])

        reused = self.root / "reused"
        frames = reused / f"_frames_{event.name}"
        frames.mkdir(parents=True)
        (reused / "cod_result.csv").write_text("old\n", encoding="utf-8")
        (frames / "frame_000002.csv").write_text("x\n", encoding="utf-8")
        code, stdout, _ = self.run_main(self.base_args(event, reused) + ["--dry-run"])
        plan = json.loads(stdout, parse_constant=reject_json_constant)
        self.assertEqual(code, 2)
        self.assertIn("frame_dir_untracked", [e["code"] for e in plan["errors"]])
        self.assertIn("cod_csv_exists", [w["code"] for w in plan["warnings"]])
        state = plan["events"][0]["output_state"]
        self.assertEqual(state["binary_expectation"], "reject_untracked")
        self.assertTrue(plan["output_state"]["exists"])

        (frames / ".opencorr-state").mkdir()
        (frames / ".opencorr-state" / "contract.json").write_text("{}", encoding="utf-8")
        code, stdout, _ = self.run_main(self.base_args(event, reused) + ["--dry-run"])
        plan = json.loads(stdout, parse_constant=reject_json_constant)
        self.assertEqual(code, 0)
        self.assertIn("frame_dir_has_ledger", [w["code"] for w in plan["warnings"]])
        self.assertEqual(plan["events"][0]["output_state"]["binary_expectation"],
                         "resume_if_unchanged")

        duplicated = self.make_event(self.root, "dup-event")
        Image.fromarray(np.full((32, 32), 100, dtype=np.uint8)).save(duplicated / "b_000002.png")
        code, stdout, _ = self.run_main(
            self.base_args(duplicated, self.root / "dup-out") + ["--dry-run"])
        plan = json.loads(stdout, parse_constant=reject_json_constant)
        self.assertEqual(code, 2)
        self.assertIn("duplicate_frame_number", [e["code"] for e in plan["errors"]])
        self.assertFalse((self.root / "dup-out").exists())

        # A second image with the reference's number is never used, and the run
        # succeeds: a warning, not an error (review 2026-10-04 R4).
        ref_dup = self.make_event(self.root, "refdup-event")
        Image.fromarray(np.full((32, 32), 100, dtype=np.uint8)).save(ref_dup / "b_000001.png")
        code, stdout, _ = self.run_main(
            self.base_args(ref_dup, self.root / "refdup-out") + ["--dry-run"])
        plan = json.loads(stdout, parse_constant=reject_json_constant)
        self.assertEqual(code, 0, plan["errors"])
        self.assertIn(("duplicate_reference_frame_number", "warn"),
                      [(w["code"], w["severity"]) for w in plan["warnings"]])
        self.assertEqual(plan["events"][0]["target_frames"], 2)

    def test_resume_verified_by_the_binary_is_information(self) -> None:
        event = self.make_event(self.root, "resume-event")
        output = self.root / "resume-out"
        summary_path = output / "summary.json"
        args = self.base_args(event, output) + [
            "--keep-frames", "--json-summary", str(summary_path)]
        self.assertEqual(self.run_main(args)[0], 0)
        self.assertEqual(self.run_main(args)[0], 0)
        codes = self.codes(self.read_summary(summary_path)["events"][0])
        self.assertEqual(codes["resumed_verified"], "info")
        self.assertNotIn("preexisting_frame_csv_detected", codes)
        self.assertEqual(codes["cod_csv_replaced"], "warn")

    def test_cod_stats_geometry_and_binary_totals_in_the_summary(self) -> None:
        event = self.make_event(self.root, "fallback-stats")
        output = self.root / "stats-out"
        summary_path = output / "summary.json"
        self.assertEqual(self.run_main(
            self.base_args(event, output)
            + ["--cod-method", "fillboundary", "--json-summary", str(summary_path)])[0], 0)
        event_summary = self.read_summary(summary_path)["events"][0]
        stats = event_summary["cod_stats"]
        self.assertEqual(stats["rows"], 4)
        self.assertEqual(stats["x_points"], 2)
        self.assertEqual(stats["frames_with_points"], 2)
        self.assertFalse(stats["all_zero"])
        self.assertEqual(stats["max_um"], 500.0)
        self.assertEqual((stats["max_frame"], stats["max_x_px"], stats["max_x_mm"]), (2, 8, 8.0))
        self.assertEqual(stats["naive_at_max_um"], 500.0)
        self.assertEqual(stats["abs_p95_um"], 500.0)
        geometry = event_summary["geometry"]
        self.assertEqual(geometry["line_ys"], [10, 20])
        self.assertEqual(geometry["subset_radius"], 2)
        self.assertEqual(geometry["extrapolation_target_y"], 15.0)
        self.assertEqual(geometry["extrapolation_px"], [5.0, -5.0])
        self.assertEqual(geometry["gauge_mm"], 10.0)
        self.assertEqual(event_summary["binary_totals"]["frames_with_fallback_fftcc"], 1)
        self.assertEqual(self.codes(event_summary)["frame_fallback_used"], "info")
        # frames deleted without --keep-frames: the path as in 638ad18, and the flag
        self.assertEqual(event_summary["field_dir"], str((output / "_frames_fallback-stats").resolve()))
        self.assertFalse(event_summary["field_dir_exists"])
        self.assertIsNone(event_summary["remediation"])
        self.assertEqual(event_summary["binary_exit_code"], 0)

    def test_binary_outside_the_checkout_is_an_info_note(self) -> None:
        event = self.make_event(self.root, "env-bin")
        output = self.root / "env-bin-out"
        summary_path = output / "summary.json"
        args = [arg for arg in self.base_args(event, output)]
        index = args.index("--gpu-bin")
        del args[index:index + 2]
        with mock.patch.dict(os.environ, {pipeline.ENV_GPU_BIN: str(self.fake_bin)}):
            code, stdout, _ = self.run_main(args + ["--dry-run"])
            plan = json.loads(stdout, parse_constant=reject_json_constant)
            self.assertEqual(code, 0)
            notes = [w for w in plan["notes"] if w["code"] == "gpu_bin_outside_checkout"]
            self.assertEqual(notes[0]["severity"], "info")
            self.assertEqual(plan["warnings"], [])  # a configured deployment: as in 638ad18
            code, stdout, _ = self.run_main(args + ["--json-summary", str(summary_path)])
        self.assertEqual(code, 0)
        self.assertIn("ℹ️", stdout)
        summary = self.read_summary(summary_path)
        self.assertEqual(summary["verdict"], "ok")
        self.assertEqual(summary["warnings"], [])  # review 2026-10-04 R2
        self.assertEqual(summary["warning_details"][0]["code"], "gpu_bin_outside_checkout")

    def test_strict_refuses_the_deprecated_lab_path(self) -> None:
        event = self.make_event(self.root, "legacy-bin")
        lab = self.root / "lab-build"
        lab.mkdir()
        legacy = lab / pipeline.GPU_BINARY_NAME
        legacy.write_bytes(self.fake_bin.read_bytes())
        legacy.chmod(0o755)
        args = [arg for arg in self.base_args(event, self.root / "legacy-out")]
        index = args.index("--gpu-bin")
        del args[index:index + 2]
        with mock.patch.object(pipeline, "LEGACY_LAB_BUILD_DIR", str(lab)), \
                mock.patch.object(pipeline, "binary_candidates", return_value=[]):
            code, stdout, _ = self.run_main(args + ["--dry-run"])
            plan = json.loads(stdout, parse_constant=reject_json_constant)
            self.assertEqual(code, 0)
            self.assertIn("path_resolution", [w["code"] for w in plan["warnings"]])
            code, stdout, _ = self.run_main(args + ["--dry-run", "--strict"])
            plan = json.loads(stdout, parse_constant=reject_json_constant)
            self.assertEqual(code, 2)
            self.assertIn("gpu_bin_legacy_lab_path", [e["code"] for e in plan["errors"]])
            code, _, stderr = self.run_main(args + ["--strict"])
            self.assertEqual(code, pipeline.EXIT_INVALID)
            self.assertIn("gpu_bin_legacy_lab_path", stderr)

    # ── review 2026-10-04: default-mode compatibility with 638ad18 ──

    def test_empty_cod_then_corrected_rerun_into_the_same_directory(self) -> None:
        """R1: 638ad18 deleted the frames of an empty COD too, so a rerun with corrected
        parameters into the same --output-dir started afresh instead of being rejected."""
        event = self.make_event(self.root, "rerun-event")
        output = self.root / "rerun-out"
        wrong = self.base_args(event, output)
        wrong[wrong.index("--line-y") + 1] = "10,30"  # L1 inside the margin: empty COD
        self.assertEqual(self.run_main(wrong)[0], 0)
        self.assertEqual((output / "cod_result.csv").read_text(), "frame,x_mm,cod_um\n")
        self.assertFalse((output / "_frames_rerun-event").exists())
        fixed = self.base_args(event, output) + ["--json-summary", str(output / "summary.json")]
        code, _, _ = self.run_main(fixed)
        self.assertEqual(code, 0)
        self.assertEqual(len((output / "cod_result.csv").read_text().splitlines()), 5)
        self.assertEqual(self.read_summary(output / "summary.json")["events"][0]["status"], "ok")

        campaign = self.root / "rerun-campaign"
        self.make_event(campaign, "event-a")
        self.make_event(campaign, "event-b")
        out = self.root / "rerun-campaign-out"
        common = ["--photo-dir", str(campaign), "--output-dir", str(out), "--cal", "1",
                  "--subset", "5", "--gpu-bin", str(self.fake_bin)]
        self.assertEqual(self.run_main(common + ["--line-y", "10,30"])[0], 0)
        self.assertEqual(self.run_main(common + ["--line-y", "10,20"])[0], 0)
        for name in ("event-a", "event-b"):
            self.assertEqual(len((out / f"{name}.csv").read_text().splitlines()), 5, name)

        # --keep-frames (as in 638ad18) and --strict keep the frames of an empty COD
        # in place; the binary then refuses a changed configuration there.
        invalid = self.make_event(self.root, "invalidzncc-rerun")
        for flag, first_code, second_code in (
            ("--keep-frames", 0, 1),
            ("--strict", pipeline.EXIT_STRICT_EVENT_FAILED, pipeline.EXIT_STRICT_EVENT_FAILED),
        ):
            with self.subTest(flag=flag):
                out = self.root / f"kept{flag}"
                first = self.base_args(invalid, out) + [flag]
                self.assertEqual(self.run_main(first)[0], first_code)
                self.assertTrue((out / "_frames_invalidzncc-rerun").is_dir())
                changed = first + ["--json-summary", str(out / "summary.json")]
                changed[changed.index("--subset") + 1] = "7"
                code, stdout, _ = self.run_main(changed)
                self.assertEqual(code, second_code)
                self.assertIn("resume rejected", stdout)
                event_summary = self.read_summary(out / "summary.json")["events"][0]
                self.assertIn("use a new --output-dir", event_summary["remediation"])

    def test_summary_warnings_keep_their_638ad18_wording(self) -> None:
        """R2: warnings holds warn/fail entries in the old wording; everything else is in
        warning_details. A deployment configured through the config file (no --gpu-bin)
        keeps an empty top-level list."""
        config = self.root / "config.toml"
        config.write_text(f'[paths]\ngpu_bin = "{self.fake_bin}"\n', encoding="utf-8")
        event = self.make_event(self.root, "config-event", max_value=255)  # 100 % at 255: warn
        output = self.root / "config-out"
        args = [arg for arg in self.base_args(event, output)]
        index = args.index("--gpu-bin")
        del args[index:index + 2]
        with mock.patch.dict(os.environ, {pipeline.ENV_CONFIG: str(config)}):
            code, stdout, _ = self.run_main(args + ["--dry-run"])
            plan = json.loads(stdout, parse_constant=reject_json_constant)
            self.assertEqual((code, plan["warnings"]), (0, []))
            self.assertEqual(plan["resolved_params"]["gpu_bin_source"], f"config:{config}")
            code, _, _ = self.run_main(args + ["--json-summary", str(output / "summary.json")])
        self.assertEqual(code, 0)
        summary = self.read_summary(output / "summary.json")
        self.assertEqual(summary["warnings"], [])
        self.assertEqual([(d["code"], d["severity"]) for d in summary["warning_details"]],
                         [("gpu_bin_outside_checkout", "info")])
        self.assertEqual(summary["events"][0]["warnings"],
                         ["input_8bit_max_255: possible saturation_or_high_bit_truncation"])
        self.assertEqual(summary["verdict"], "warn")

    def test_duplicate_frame_numbers_at_run_time(self) -> None:
        """A target duplicate fails at the binary (duplicate output name) and is named in
        the summary; a duplicate of the reference number is dropped and only warned about."""
        target_dup = self.make_event(self.root, "tdup-event")
        Image.fromarray(np.full((32, 32), 100, dtype=np.uint8)).save(target_dup / "b_000002.png")
        output = self.root / "tdup-out"
        code, _, _ = self.run_main(
            self.base_args(target_dup, output) + ["--json-summary", str(output / "summary.json")])
        self.assertEqual(code, 1)
        event_summary = self.read_summary(output / "summary.json")["events"][0]
        self.assertEqual(self.codes(event_summary)["duplicate_frame_number"], "warn")
        self.assertIn("duplicate_frame_number: 1 target frame number(s)",
                      " ".join(event_summary["warnings"]))
        self.assertIn("same frame number", event_summary["remediation"])

        ref_dup = self.make_event(self.root, "rdup-event")
        Image.fromarray(np.full((32, 32), 100, dtype=np.uint8)).save(ref_dup / "b_000001.png")
        output = self.root / "rdup-out"
        for strict in (False, True):
            with self.subTest(strict=strict):
                out = output / str(strict)
                code, _, _ = self.run_main(
                    self.base_args(ref_dup, out) + ["--json-summary", str(out / "summary.json")]
                    + (["--strict"] if strict else []))
                self.assertEqual(code, 0)
                event_summary = self.read_summary(out / "summary.json")["events"][0]
                self.assertEqual(event_summary["status"], "ok")
                self.assertEqual(self.codes(event_summary)["duplicate_reference_frame_number"], "warn")
                self.assertEqual(event_summary["frames"], 2)

    def test_gpu_bin_with_dotdot_after_a_symlinked_directory(self) -> None:
        """R5: link/.. must follow the symlink, not be folded away textually."""
        real = self.root / "real"
        (real / "sub").mkdir(parents=True)
        binary = real / "fake_bin_r5.py"
        binary.write_bytes(self.fake_bin.read_bytes())
        binary.chmod(0o755)
        (self.root / "link").symlink_to(real / "sub", target_is_directory=True)
        event = self.make_event(self.root, "dotdot-event")
        previous = os.getcwd()
        os.chdir(self.root)
        try:
            args = self.base_args(event, self.root / "dotdot-out")
            args[args.index("--gpu-bin") + 1] = "link/../fake_bin_r5.py"
            code, stdout, _ = self.run_main(args + ["--dry-run"])
            self.assertEqual(code, 0, stdout)
            self.assertEqual(json.loads(stdout)["events"][0]["binary_command"][0],
                             str(Path(os.getcwd()) / "link" / ".." / "fake_bin_r5.py"))
            self.assertEqual(self.run_main(args)[0], 0)
        finally:
            os.chdir(previous)

    # ── review 2026-10-04: --strict gaps ──

    def test_strict_refuses_a_deeper_container_read_as_8_bit(self) -> None:
        """R6a: the first pitfall. The default --input-depth 8 on a 16-bit file runs with
        a warning, --strict refuses it and names the probe's recommendation."""
        deep = np.full((32, 32), 3000, dtype=np.uint16)
        deep.flat[:4] = 4000
        event = self.make_event(self.root, "deep-event", pixels=deep)
        output = self.root / "deep-out"
        code, _, _ = self.run_main(self.base_args(event, output))
        self.assertEqual(code, 0)
        code, stdout, _ = self.run_main(self.base_args(event, self.root / "deep-plan") + ["--dry-run", "--strict"])
        self.assertEqual(code, 2)
        error = next(e for e in json.loads(stdout)["errors"] if e["code"] == "input_depth_truncates_container")
        self.assertIn("--input-depth 12 --input-align right", error["message"])
        self.assertIn("input_depth='12'", error["message"])
        code, _, stderr = self.run_main(self.base_args(event, self.root / "deep-strict") + ["--strict"])
        self.assertEqual(code, pipeline.EXIT_INVALID)
        self.assertIn("input_depth_truncates_container", stderr)
        # An explicit depth other than 8 passes the check.
        explicit = self.base_args(event, self.root / "deep-explicit") + [
            "--strict", "--input-depth", "12", "--input-align", "right"]
        self.assertEqual(self.run_main(explicit)[0], 0)

    def test_strict_precheck_warnings_reach_the_summary(self) -> None:
        """R6b: the pre-check's warnings are recorded at the top level of the summary."""
        deep = np.full((32, 32), 3000, dtype=np.uint16)
        event = self.make_event(self.root, "precheck-event", pixels=deep)
        output = self.root / "precheck-out"
        args = self.base_args(event, output) + [
            "--strict", "--input-depth", "16", "--json-summary", str(output / "summary.json")]
        code, _, _ = self.run_main(args)
        self.assertEqual(code, 0)
        summary = self.read_summary(output / "summary.json")
        recorded = [d for d in summary["warning_details"] if d.get("stage") == "strict_precheck"]
        self.assertEqual([(d["code"], d["severity"], d["event"]) for d in recorded],
                         [("input_mode_differs_from_probe", "warn", "precheck-event")])
        self.assertTrue(any(text.startswith("input_mode_differs_from_probe:") for text in summary["warnings"]))
        self.assertEqual(summary["verdict"], "warn")

    def test_strict_campaign_precheck_is_all_or_nothing(self) -> None:
        """R6c (documented, unchanged): one refused event stops the whole campaign."""
        campaign = self.root / "strict-campaign"
        self.make_event(campaign, "event-ok")
        lonely = self.make_event(campaign, "event-one-image")
        for frame in ("000002.png", "000003.png"):
            (lonely / frame).unlink()
        output = self.root / "strict-campaign-out"
        args = ["--photo-dir", str(campaign), "--output-dir", str(output), "--line-y", "10,20",
                "--cal", "1", "--subset", "5", "--gpu-bin", str(self.fake_bin)]
        code, _, stderr = self.run_main(args + ["--strict"])
        self.assertEqual(code, pipeline.EXIT_INVALID)
        self.assertIn("insufficient_images", stderr)
        self.assertFalse(self.log_path.exists())
        self.assertEqual(self.run_main(args)[0], 0)  # default mode runs the others

    # ── review 2026-10-04: diagnostics of the run itself ──

    def test_binary_stderr_is_in_the_log_while_the_binary_runs(self) -> None:
        """R9: the log is written as the binary writes, so it survives a killed pipeline."""
        import signal
        import time

        event = self.make_event(self.root, "sleepy-event")
        output = self.root / "sleepy-out"
        log = output / "_frames_sleepy-event.binary.log"
        process = subprocess.Popen(
            [sys.executable, pipeline.__file__, *self.base_args(event, output)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if log.is_file() and "streaming marker" in log.read_text(encoding="utf-8", errors="replace"):
                    break
                time.sleep(0.05)
        finally:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        text = log.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# command: "))
        self.assertIn("Warning: streaming marker", text)
        self.assertNotIn("# exit code:", text)  # the pipeline never finished

    def test_dry_run_predicts_that_a_shorter_target_list_is_rejected(self) -> None:
        """R16: --max-frames that shortens an existing run cannot resume it."""
        event = self.make_event(self.root, "shorten-event", frames=6)
        output = self.root / "shorten-out"
        args = self.base_args(event, output) + ["--keep-frames"]
        self.assertEqual(self.run_main(args)[0], 0)
        code, stdout, _ = self.run_main(args + ["--max-frames", "2", "--dry-run"])
        plan = json.loads(stdout, parse_constant=reject_json_constant)
        self.assertEqual(code, 2)
        self.assertIn("frame_dir_targets_changed", [e["code"] for e in plan["errors"]])
        self.assertEqual(plan["events"][0]["output_state"]["binary_expectation"], "reject_targets_changed")
        code, stdout, _ = self.run_main(args + ["--max-frames", "2"])
        self.assertEqual(code, 1)  # the binary agrees
        self.assertIn("targets changed/reordered", stdout)
        code, stdout, _ = self.run_main(args + ["--dry-run"])  # the same list resumes
        self.assertEqual(code, 0, stdout)


    # ── 2026-10-05: --events and --event-interface-y (campaign only) ──

    def make_selection_campaign(self, name: str) -> Path:
        campaign = self.root / name
        for event in ("ev01", "ev02", "ev03", "other"):
            self.make_event(campaign, event)
        (campaign / "empty").mkdir()
        return campaign

    def selection_args(self, campaign: Path, output: Path) -> list[str]:
        return ["--photo-dir", str(campaign), "--output-dir", str(output),
                "--interface-y", "12", "--gauge", "4", "--cal", "1", "--subset", "5",
                "--gpu-bin", str(self.fake_bin)]

    def test_events_and_event_interface_y_plan_run_and_summary_agree(self) -> None:
        campaign = self.make_selection_campaign("selection")
        output = self.root / "selection-out"
        summary_path = self.root / "selection-summary.json"
        args = self.selection_args(campaign, output) + [
            "--events", "ev0*", "--event-interface-y", "ev03=16",
            "--json-summary", str(summary_path),
        ]
        code, stdout, _ = self.run_main(args + ["--dry-run"])
        self.assertEqual(code, 0, stdout)
        plan = json.loads(stdout, parse_constant=reject_json_constant)
        self.assertEqual([event["name"] for event in plan["events"]], ["ev01", "ev02", "ev03"])
        # the left-out empty directory is not reported; "empty" does not match ev0*
        self.assertNotIn("event_without_images", [item["code"] for item in plan["warnings"]])
        self.assertEqual(plan["input"]["events_with_images"], 4)
        selection = plan["event_selection"]
        self.assertEqual(selection["events_argument"], "ev0*")
        self.assertEqual(selection["patterns"], ["ev0*"])
        self.assertEqual((selection["events_discovered"], selection["events_selected"],
                          selection["events_excluded"]), (4, 3, 1))
        self.assertEqual(selection["excluded_events"], ["other"])
        self.assertEqual(selection["interface_y_default"], 12)
        self.assertEqual(selection["interface_y_overrides"], [{
            "argument": "ev03=16", "pattern": "ev03", "interface_y": 16, "events": ["ev03"],
        }])
        expected = {
            "ev01": ([10, 14], 12, "interface_y", []),
            "ev02": ([10, 14], 12, "interface_y", []),
            "ev03": ([14, 18], 16, "event_interface_y", ["ev03=16"]),
        }
        planned = {}
        for event in plan["events"]:
            geometry = event["geometry"]
            self.assertEqual((geometry["line_ys"], geometry["interface_y"],
                              geometry["interface_y_source"], geometry["interface_y_overrides"]),
                             expected[event["name"]])
            command = event["binary_command"]
            self.assertEqual(command[command.index("--line") + 1],
                             ",".join(map(str, expected[event["name"]][0])))
            planned[event["name"]] = command[1:]

        code, stdout, _ = self.run_main(args)
        self.assertEqual(code, 0, stdout)
        executed = [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines()
                    if line != '["--version"]']
        self.assertEqual(executed, [planned["ev01"], planned["ev02"], planned["ev03"]])
        self.assertEqual(sorted(path.name for path in output.glob("*.csv")),
                         ["ev01.csv", "ev02.csv", "ev03.csv"])
        summary = self.read_summary(summary_path)
        self.assertEqual(summary["event_selection"], selection)
        for event in summary["events"]:
            geometry = event["geometry"]
            self.assertEqual((geometry["line_ys"], geometry["interface_y"],
                              geometry["interface_y_source"], geometry["interface_y_overrides"]),
                             expected[event["name"]])
        self.assertEqual(summary["params"]["events"], "ev0*")
        self.assertEqual(summary["params"]["event_interface_y"], ["ev03=16"])
        self.assertIn("ev03=16", stdout)

    def test_campaign_without_the_new_options_selects_everything(self) -> None:
        campaign = self.make_selection_campaign("no-selection")
        output = self.root / "no-selection-out"
        summary_path = self.root / "no-selection-summary.json"
        args = self.selection_args(campaign, output) + ["--json-summary", str(summary_path)]
        code, stdout, _ = self.run_main(args + ["--dry-run"])
        plan = json.loads(stdout, parse_constant=reject_json_constant)
        self.assertEqual(code, 0, stdout)
        self.assertEqual([event["name"] for event in plan["events"]], ["ev01", "ev02", "ev03", "other"])
        self.assertIn("event_without_images", [item["code"] for item in plan["warnings"]])
        self.assertEqual(plan["event_selection"]["events_argument"], None)
        self.assertEqual(plan["event_selection"]["excluded_events"], [])
        self.assertEqual(plan["event_selection"]["interface_y_overrides"], [])
        self.assertEqual(self.run_main(args)[0], 0)
        summary = self.read_summary(summary_path)
        for event in summary["events"]:
            self.assertEqual(event["geometry"]["line_ys"], [10, 14])
            self.assertEqual(event["geometry"]["interface_y_source"], "interface_y")
        self.assertEqual(summary["event_selection"]["events_selected"], 4)

        single = self.root / "selection-single-summary.json"
        event = campaign / "ev01"
        code, _, _ = self.run_main(self.base_args(event, self.root / "single-out")
                                   + ["--json-summary", str(single)])
        self.assertEqual(code, 0)
        summary = self.read_summary(single)
        self.assertIsNone(summary["event_selection"])
        geometry = summary["events"][0]["geometry"]
        self.assertEqual(geometry["line_source"], "line_y")  # base_args uses --line-y
        self.assertIsNone(geometry["interface_y"])
        self.assertIsNone(geometry["interface_y_source"])

    def test_identical_overrides_are_not_a_conflict(self) -> None:
        campaign = self.make_selection_campaign("same-row")
        args = self.selection_args(campaign, self.root / "same-row-out") + [
            "--event-interface-y", "ev0[23]=16", "--event-interface-y", "ev03=16", "--dry-run",
        ]
        code, stdout, _ = self.run_main(args)
        self.assertEqual(code, 0, stdout)
        plan = json.loads(stdout, parse_constant=reject_json_constant)
        ev03 = [event for event in plan["events"] if event["name"] == "ev03"][0]
        self.assertEqual(ev03["geometry"]["interface_y_overrides"], ["ev0[23]=16", "ev03=16"])

    def test_event_interface_y_bounds_are_checked_per_event(self) -> None:
        campaign = self.make_selection_campaign("override-bounds")
        args = self.selection_args(campaign, self.root / "override-bounds-out") + [
            "--event-interface-y", "ev02=999", "--dry-run",
        ]
        code, stdout, _ = self.run_main(args)
        self.assertEqual(code, 2)
        errors = json.loads(stdout, parse_constant=reject_json_constant)["errors"]
        bounds = [item for item in errors if item["code"] == "interface_y_out_of_bounds"]
        self.assertEqual([item["event"] for item in bounds], ["ev02"])
        self.assertIn("--event-interface-y ev02=999", bounds[0]["message"])
        self.assertEqual({item.get("event") for item in errors if item["code"] == "line_y_out_of_bounds"},
                         {"ev02"})

    def test_event_selection_errors_exit_2_before_anything_is_written(self) -> None:
        campaign = self.make_selection_campaign("selection-errors")
        single = campaign / "ev01"
        cases = [
            ("glob without match", ["--events", "zz*"], ["event_pattern_no_match", "no_events_selected"]),
            ("unknown event name", ["--events", "ev01,ev09"], ["event_not_found"]),
            ("directory without images", ["--events", "empty"], ["event_not_found", "no_events_selected"]),
            ("empty list item", ["--events", "ev01,"], ["invalid_events"]),
            ("override conflict", ["--event-interface-y", "ev0*=16", "--event-interface-y", "ev03=18"],
             ["event_interface_y_conflict"]),
            ("override matches nothing", ["--event-interface-y", "zz=16"], ["event_interface_y_no_match"]),
            ("override matches only left-out events",
             ["--events", "ev01", "--event-interface-y", "ev02=16"], ["event_interface_y_no_match"]),
            ("override with --line-y", ["--line-y", "10,20", "--event-interface-y", "ev01=16"],
             ["event_interface_y_with_line_y"]),
            ("override not an integer", ["--event-interface-y", "ev01=abc"], ["invalid_event_interface_y"]),
            ("override negative", ["--event-interface-y", "ev01=-3"], ["invalid_event_interface_y"]),
            ("override without =", ["--event-interface-y", "ev01"], ["invalid_event_interface_y"]),
            ("override with commas", ["--event-interface-y", "ev01,ev02=16"], ["invalid_event_interface_y"]),
        ]
        for index, (label, extra, codes) in enumerate(cases):
            for strict in (False, True):
                with self.subTest(case=label, strict=strict):
                    output = self.root / f"selection-error-{index}-{int(strict)}"
                    args = self.selection_args(campaign, output) + extra + (["--strict"] if strict else [])
                    code, stdout, stderr = self.run_main(args)
                    self.assertEqual(code, pipeline.EXIT_INVALID, stderr)
                    self.assertFalse(output.exists())
                    self.assertFalse(self.log_path.exists())
                    first = stderr.splitlines()[0]
                    self.assertTrue(first.startswith("错误: "), first)
                    for expected in codes:
                        self.assertIn(expected, stderr)
            code, stdout, _ = self.run_main(self.selection_args(campaign, self.root / f"plan-{index}")
                                            + extra + ["--dry-run"])
            self.assertEqual(code, 2)
            plan_codes = [item["code"] for item in json.loads(stdout)["errors"]]
            for expected in codes:
                self.assertIn(expected, plan_codes, label)

        no_default = ["--photo-dir", str(campaign), "--output-dir", str(self.root / "no-default"),
                      "--gauge", "4", "--cal", "1", "--subset", "5", "--gpu-bin", str(self.fake_bin),
                      "--event-interface-y", "ev01=16"]
        code, _, stderr = self.run_main(no_default)
        self.assertEqual(code, pipeline.EXIT_INVALID)
        self.assertIn("event_interface_y_requires_interface_y", stderr)
        self.assertFalse((self.root / "no-default").exists())
        for option, value, expected in (("--events", "ev01", "events_requires_photo_dir"),
                                        ("--event-interface-y", "ev01=16",
                                         "event_interface_y_requires_photo_dir")):
            output = self.root / f"single-{expected}"
            code, _, stderr = self.run_main(
                ["--image-dir", str(single), "--output-dir", str(output), "--interface-y", "12",
                 "--gauge", "4", "--cal", "1", "--subset", "5", "--gpu-bin", str(self.fake_bin),
                 option, value])
            self.assertEqual(code, pipeline.EXIT_INVALID)
            self.assertIn(expected, stderr)
            self.assertFalse(output.exists())
        self.assertFalse(self.log_path.exists())

    def test_event_patterns_are_case_sensitive_globs(self) -> None:
        names = ["event01", "event20", "event21", "event25", "Event22"]
        args = argparse_namespace(events="event2[1-5],event01", interface_y=131, line_y=None,
                                  event_interface_y=["event2[1-5]=133"], photo_dir="/nonexistent")
        selection = pipeline.plan_event_selection(args, names)
        self.assertEqual(selection["errors"], [])
        self.assertEqual(selection["selected"], ["event01", "event21", "event25"])
        self.assertEqual(selection["excluded"], ["event20", "Event22"])
        self.assertEqual({name: value["interface_y"] for name, value in selection["interface"].items()},
                         {"event01": 131, "event21": 133, "event25": 133})
        self.assertEqual(pipeline.parse_event_interface_y(" event2[1-5] = 133 "), ("event2[1-5]", 133))


class CodPublicationTests(unittest.TestCase):
    def test_publication_is_atomic_and_byte_identical_to_write_cod_csv(self) -> None:
        cod = np.array([[1.23456, np.nan], [-0.00004, 7.0]])
        x_mm = np.array([1.0, 2.0])
        frames = np.array([2, 3])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct, published = root / "direct.csv", root / "published.csv"
            pipeline.write_cod_csv(direct, cod, x_mm, frames)
            published.write_text("previous\n", encoding="utf-8")
            pipeline.publish_cod_csv(published, cod, x_mm, frames)
            self.assertEqual(direct.read_bytes(), published.read_bytes())
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["direct.csv", "published.csv"])

            with mock.patch.object(pipeline, "write_cod_csv", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    pipeline.publish_cod_csv(published, cod, x_mm, frames)
            self.assertEqual(direct.read_bytes(), published.read_bytes())
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["direct.csv", "published.csv"])

    def test_a_write_that_dies_halfway_never_reaches_the_published_name(self) -> None:
        """Review 2026-10-04 R11 (mutant M6): a crash after half the bytes were written
        must leave the previous file intact. Writing the target directly fails this."""
        cod = np.array([[1.5, 2.5], [3.5, 4.5]])
        x_mm, frames = np.array([1.0, 2.0]), np.array([2, 3])
        real_write = pipeline.write_cod_csv

        def dies_halfway(path, *arguments):
            real_write(path, *arguments)
            with open(path, "r+b") as handle:
                handle.truncate(max(1, Path(path).stat().st_size // 2))
            raise OSError("process killed while writing")

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "cod_result.csv"
            target.write_text("frame,x_mm,cod_um\n2,1.0000,9.0000\n", encoding="utf-8")
            with mock.patch.object(pipeline, "write_cod_csv", side_effect=dies_halfway):
                with self.assertRaises(OSError):
                    pipeline.publish_cod_csv(target, cod, x_mm, frames)
            self.assertEqual(target.read_text(encoding="utf-8"), "frame,x_mm,cod_um\n2,1.0000,9.0000\n")
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["cod_result.csv"])

    def test_cod_statistics_match_the_written_rows(self) -> None:
        cod = np.array([[0.0, np.nan], [-0.00004, 0.00001]])
        stats = pipeline.cod_statistics(
            cod, np.array([1.0, 2.0]), np.array([2, 3]), 1.0,
            {"line_rows": {"L0": 4, "L1": 4}, "x_px": np.array([1, 2]),
             "u_l0": np.zeros((2, 2)), "u_l1": np.zeros((2, 2))}, "naive",
        )
        self.assertEqual(stats["rows"], 3)
        self.assertTrue(stats["all_zero"])  # every written value prints as 0.0000
        self.assertEqual(stats["frames_with_points"], 2)
        self.assertIsNone(stats["naive_at_max_um"])  # naive mode: no separate naive value


class InterfaceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract_path = Path(__file__).with_name("interface_contract.json")
        cls.contract_text = cls.contract_path.read_text(encoding="utf-8")
        cls.contract = json.loads(cls.contract_text, parse_constant=reject_json_constant)

    def test_describe_outputs_contract_byte_for_byte(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            return_code = pipeline.main(["--describe"])
        self.assertEqual(return_code, 0)
        self.assertEqual(stdout.getvalue(), self.contract_text)
        self.assertEqual(json.loads(stdout.getvalue())["schema_version"], 1)

    def test_describe_resolves_contract_when_pipeline_is_invoked_via_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            symlink = Path(tmp) / "dic_pipeline.py"
            symlink.symlink_to(Path(pipeline.__file__).resolve())
            result = subprocess.run(
                [sys.executable, str(symlink), "--describe"],
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, self.contract_text)

    def test_pipeline_contract_matches_argparse_defaults_and_choices(self) -> None:
        parser = pipeline.create_parser()
        actions = {}
        for action in parser._actions:
            long_options = [option for option in action.option_strings if option.startswith("--")]
            if long_options and long_options[0] != "--help":
                actions[long_options[0]] = action
        parameters = {
            parameter["name"]: parameter
            for parameter in self.contract["pipeline"]["parameters"]
        }

        self.assertEqual(set(parameters), set(actions))
        for name, action in actions.items():
            with self.subTest(name=name):
                self.assertEqual(parameters[name]["dest"], action.dest)
                self.assertEqual(parameters[name]["default"], action.default)
                self.assertEqual(parameters[name]["required"], action.required)
                choices = list(action.choices) if action.choices is not None else None
                self.assertEqual(parameters[name].get("choices"), choices)
                self.assertIn("semantics", parameters[name])
                self.assertIn("unit", parameters[name])

    def test_binary_contract_covers_every_common_option_in_source(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "opencorr_cli_cuda_mvp.cpp").read_text(
            encoding="utf-8"
        )
        block = source.split("bool parseCommonOption", 1)[1].split("bool loadTarList", 1)[0]
        source_options = set(re.findall(r'arg == "(--[a-z0-9-]+)"', block))
        contract_options = {
            parameter["name"] for parameter in self.contract["binary"]["parameters"]
        }
        self.assertEqual(contract_options, source_options)
        for parameter in self.contract["binary"]["parameters"]:
            with self.subTest(parameter=parameter["name"]):
                self.assertIn("semantics", parameter)
                self.assertIn("unit", parameter)

        modes = self.contract["binary"]["modes"]
        positional_items = (
            modes["single_frame"]["positionals"]
            + modes["multi_frame"]["positionals"]
            + modes["multi_frame"]["required_options"]
        )
        for parameter in positional_items:
            with self.subTest(mode_parameter=parameter["name"]):
                self.assertIn("semantics", parameter)
                self.assertIn("unit", parameter)

    def test_output_columns_and_pitfall_ids_are_stable(self) -> None:
        columns = self.contract["outputs"]["frame_csv"]["columns"]
        self.assertEqual(
            [column["name"] for column in columns],
            ["x", "y", "u", "v", "zncc", "uy", "vy", "ux", "vx"],
        )
        pitfall_ids = [item["id"] for item in self.contract["pitfalls"]]
        self.assertEqual(len(pitfall_ids), 6)
        self.assertEqual(len(pitfall_ids), len(set(pitfall_ids)))
        # 2026-10-19: the entries that hold only for the author's laboratory
        # moved to lab_notes (same record shape, disjoint ids), outside
        # dic_get_pitfalls and the default dic_describe() reply.
        lab_note_ids = [item["id"] for item in self.contract["lab_notes"]]
        self.assertEqual(len(lab_note_ids), 5)
        self.assertEqual(len(lab_note_ids), len(set(lab_note_ids)))
        self.assertEqual(set(pitfall_ids) & set(lab_note_ids), set())
        for item in self.contract["pitfalls"] + self.contract["lab_notes"]:
            with self.subTest(entry=item["id"]):
                self.assertEqual(list(item), ["id", "title", "fact", "action", "agent_markdown"])
        self.assertEqual(list(self.contract)[-3:], ["pitfalls", "lab_notes", "known_limitations"])

    def test_emitted_warning_codes_are_declared_in_the_vocabulary(self) -> None:
        source = Path(pipeline.__file__).read_text(encoding="utf-8")
        vocabulary = {
            item["code"]: item
            for item in self.contract["outputs"]["json_summary"]["warning_vocabulary"]
        }
        emitted = set(pipeline.WARNING_SEVERITY)
        for pattern in (
            r'add_warning\(\s*\w+,\s*"([a-z][a-z0-9_.]*)"',
            r'warnings(?:"\])?\.append\(\s*f?"([a-z][a-z0-9_.]*)[:"]',
            r'warnings = \[f?"([a-z][a-z0-9_.]*):',
            r'CodAggregationError\(\s*"([a-z][a-z0-9_.]*)"',
        ):
            emitted |= set(re.findall(pattern, source))
        # Sanity: the scan really sees the producers it is meant to see.
        self.assertTrue({"cod_empty", "gpu_cli_failed", "argument_warning",
                         "valid_ratio_below_0.95", "pipeline_error"} <= emitted)
        self.assertLessEqual(emitted, set(vocabulary), sorted(emitted - set(vocabulary)))
        self.assertEqual(set(pipeline.WARNING_SEVERITY), set(vocabulary))
        for code, level in pipeline.WARNING_SEVERITY.items():
            with self.subTest(code=code):
                declared = vocabulary[code]["severity"]
                if code == "input_8bit_max_255":
                    self.assertEqual(declared, "info or warn")
                else:
                    self.assertEqual(declared, level)
                strict = vocabulary[code].get("strict_severity")
                self.assertEqual(strict == "fail", code in pipeline.STRICT_FAIL_CODES)

    def test_dry_run_issue_codes_are_declared(self) -> None:
        source = Path(pipeline.__file__).read_text(encoding="utf-8")
        dry_run = self.contract["outputs"]["dry_run"]
        declared = set(dry_run["error_codes"]) | set(dry_run["warning_codes"])
        used = set(re.findall(r'_issue\(\s*"([a-z][a-z0-9_]*)"', source))
        self.assertTrue({"output_dir_not_directory", "duplicate_frame_number"} <= used)
        self.assertLessEqual(used, declared, sorted(used - declared))
        self.assertEqual(set(dry_run["warning_codes"]), set(dry_run["warning_code_semantics"]))

    def test_exit_codes_include_the_strict_code(self) -> None:
        codes = {item["code"] for item in self.contract["exit_codes"]["pipeline"]}
        self.assertEqual(codes, {pipeline.EXIT_OK, pipeline.EXIT_FAILURE,
                                 pipeline.EXIT_INVALID, pipeline.EXIT_STRICT_EVENT_FAILED})

    def test_agents_pitfall_appendix_is_generated_from_contract(self) -> None:
        agents_text = (
            Path(__file__).resolve().parents[1] / "docs" / "lab" / "simulacrum.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(
            sync_agent_docs.synchronized_text(agents_text, self.contract),
            agents_text,
        )


if __name__ == "__main__":
    unittest.main()
