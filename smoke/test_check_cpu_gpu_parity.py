# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The parity gate must reject lost frames and errors hidden by pooled medians."""
import copy
import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from check_cpu_gpu_parity import compare_frame


class ParityCoverageTests(unittest.TestCase):
    def setUp(self):
        self.cpu = {(x, 10): {"u": 5., "v": 0., "zncc": .99} for x in range(200)}

    def test_matching_frame_passes(self):
        self.assertTrue(compare_frame(self.cpu, self.cpu)["pass"])

    def test_entire_lost_frame_fails_even_after_a_good_frame(self):
        good = compare_frame(self.cpu, self.cpu)
        gpu = copy.deepcopy(self.cpu)
        for row in gpu.values(): row["zncc"] = -4.
        bad = compare_frame(self.cpu, gpu)
        self.assertTrue(good["pass"])
        self.assertFalse(bad["pass"])
        self.assertEqual(bad["cpu_valid_gpu_invalid"], 200)

    def test_lost_ten_percent_cannot_hide_in_zero_displacement_error(self):
        gpu = copy.deepcopy(self.cpu)
        for key in list(gpu)[:20]: gpu[key]["zncc"] = -4.
        result = compare_frame(self.cpu, gpu)
        self.assertEqual(result["median_abs_du"], 0.)
        self.assertFalse(result["pass"])

    def test_tail_error_cannot_hide_in_median(self):
        gpu = copy.deepcopy(self.cpu)
        for key in list(gpu)[:4]: gpu[key]["u"] += 1.
        result = compare_frame(self.cpu, gpu)
        self.assertEqual(result["median_abs_du"], 0.)
        self.assertFalse(result["pass"])

    def test_nonfinite_displacement_is_invalid_even_with_positive_zncc(self):
        gpu = copy.deepcopy(self.cpu)
        for row in gpu.values(): row["u"] = float("nan")
        self.assertFalse(compare_frame(self.cpu, gpu)["pass"])

    def test_all_invalid_on_both_sides_is_not_parity_evidence(self):
        rows = copy.deepcopy(self.cpu)
        for row in rows.values(): row["zncc"] = -3.
        self.assertFalse(compare_frame(rows, rows)["pass"])

    def test_empty_common_domain_fails(self):
        self.assertFalse(compare_frame(self.cpu, {})["pass"])

    def test_single_outlier_fails_max_when_p99_is_zero(self):
        gpu = copy.deepcopy(self.cpu)
        gpu[(0, 10)]["u"] += 1.
        result = compare_frame(self.cpu, gpu)
        self.assertEqual(result["p99_abs_du"], 0.)
        self.assertFalse(result["pass"])
        self.assertTrue(compare_frame(self.cpu, gpu, max_threshold=1.)["pass"])

    def write_frames(self, folder: Path, names, u=5.0):
        folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            rows = "".join(f"{x},10,{u},0,0.99\n" for x in range(200))
            (folder / name).write_text("x,y,u,v,zncc\n" + rows)

    def run_checker(self, cpu: Path, gpu: Path, *extra: str):
        report = cpu.parent / "report.json"
        run = subprocess.run([sys.executable, str(Path(__file__).with_name("check_cpu_gpu_parity.py")),
                              "--cpu-dir", str(cpu), "--gpu-dir", str(gpu),
                              "--json-report", str(report), *extra],
                             capture_output=True, text=True)
        self.assertNotIn("Traceback", run.stderr)
        return run.returncode, json.loads(report.read_text())

    def test_default_pairing_still_uses_zero_based_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_frames(root / "cpu", ["frame_000000.csv", "frame_000001.csv"])
            self.write_frames(root / "gpu", ["frame_000000.csv", "frame_000001.csv"])
            code, report = self.run_checker(root / "cpu", root / "gpu", "--frames", "2")
            self.assertEqual(code, 0)
            self.assertEqual([f["frame"] for f in report["frames"]],
                             ["frame_000000.csv", "frame_000001.csv"])
            self.assertEqual(report["pairing"], "index")

    def test_glob_pairs_arbitrary_frame_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = ["frame_072425.csv", "frame_072424.csv", "frame_072426.csv"]
            self.write_frames(root / "cpu", names)
            self.write_frames(root / "gpu", names)
            (root / "gpu" / "notes.txt").write_text("not a frame")
            code, report = self.run_checker(root / "cpu", root / "gpu", "--glob", "frame_*.csv")
            self.assertEqual(code, 0)
            self.assertEqual([f["frame"] for f in report["frames"]], sorted(names))
            code, report = self.run_checker(root / "cpu", root / "gpu",
                                            "--glob", "frame_*.csv", "--frames", "2")
            self.assertEqual(code, 0)
            self.assertEqual(len(report["frames"]), 2)

    def test_glob_fails_a_name_present_on_one_side_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_frames(root / "cpu", ["frame_000002.csv", "frame_000003.csv"])
            self.write_frames(root / "gpu", ["frame_000002.csv"])
            code, report = self.run_checker(root / "cpu", root / "gpu", "--glob", "frame_*.csv")
            self.assertEqual(code, 1)
            verdicts = {f["frame"]: f["pass"] for f in report["frames"]}
            self.assertEqual(verdicts, {"frame_000002.csv": True, "frame_000003.csv": False})

    def test_glob_without_any_match_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "cpu").mkdir()
            (root / "gpu").mkdir()
            code, report = self.run_checker(root / "cpu", root / "gpu", "--glob", "frame_*.csv")
            self.assertEqual(code, 1)
            self.assertEqual(report["frames"], [])
            self.assertFalse(report["pass"])

    def test_missing_csv_column_is_reported_without_traceback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "frame_000000.csv").write_text("x,y,u,v,zncc\n1,2,3\n")
            report = root / "report.json"
            run = subprocess.run([sys.executable, str(Path(__file__).with_name("check_cpu_gpu_parity.py")),
                                  "--cpu-dir", folder, "--gpu-dir", folder, "--frames", "1",
                                  "--json-report", str(report)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            self.assertNotIn("Traceback", run.stderr)
            self.assertFalse(json.loads(report.read_text())["frames"][0]["pass"])


if __name__ == "__main__":
    unittest.main()
