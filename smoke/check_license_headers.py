#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Licence-header coverage gate and vendored-source integrity assertion.

Two independent checks, both offline:

1. Every first-party source file carries the MPL-2.0 Exhibit A notice and the
   project copyright line.
2. Every vendored OpenCorr file that we did not modify is still byte-identical
   to the recorded upstream baseline, and every vendored file we did modify
   still carries its original upstream notice plus our modification marker.

Baseline: OpenCorr upstream commit 994337a74b051f5b28a305299df3bfc12d08288f
(2026-08-28, github.com/vincentjzy/OpenCorr, branch main, untagged -- upstream
has issued no library release tag since v1.0, so the commit hash is the only
precise identifier).  The previous baseline was 5f84be6f (2025-05-06); NOTICE.md
section 2.4 records what moved between the two.
`src/nanoflann.hpp` is nanoflann v1.9.0 (github.com/jlblancoc/nanoflann),
which upstream OpenCorr expects as an external dependency rather than vendoring.

Usage:  python3 smoke/check_license_headers.py [--root <repo>]
Exit status 0 means both checks pass, 1 that one failed, and 77 that the
root is not a git checkout (an unpacked archive): the file list comes from
`git ls-files`, so the check is skipped there, as scripts/verify.sh does.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import sys

MPL_NOTICE = "This Source Code Form is subject to the terms of the Mozilla Public"
COPYRIGHT = "Copyright (c) 2026 Chonglang Wang"
MODIFICATION_MARKER = "Modified for OpenCorr-GPU"
HEADER_WINDOW = 1600
EXIT_SKIPPED = 77  # not a git checkout: the same code scripts/verify.sh gives a skipped step

SOURCE_SUFFIXES = {".cpp", ".cu", ".h", ".hpp", ".in", ".py", ".sh"}
SOURCE_NAMES = {"CMakeLists.txt"}

# Paths carrying no first-party header by design.
SKIP_EXACT = {
    "LICENSE",
    "NOTICE.md",
    "CITATION.cff",
    ".gitignore",
    "pipeline/requirements-mcp.txt",
}
SKIP_PREFIXES = ("third_party/",)

# Vendored third-party header shipped inside src/ (BSD-2-Clause, unmodified).
NANOFLANN = "src/nanoflann.hpp"
NANOFLANN_SHA256 = "e91b910b6fcdeaefb0253d50438c519de675980cbf3374397da730dfc23bbfe7"

# Vendored OpenCorr files carrying local modifications (see NOTICE.md).
VENDORED_MODIFIED = {
    "src/oc_dic.h",
    "src/oc_fftcc.cpp",
    "src/oc_fftcc.h",
    "src/oc_icgn.cpp",
    "src/oc_image.cpp",
    "src/oc_image.h",
}

# Vendored OpenCorr files that must stay byte-identical to the baseline.
UPSTREAM_SHA256 = {
    "src/oc_array.h": "cfbd2ef7c0c5eef4a8f5505c25e55a3eaf3edc66e5b91d6e73ec599382ffa886",
    "src/oc_calibration.cpp": "c809a6dbf5d83e4ddac5d79b018f71e560c5b42212b7710f34576c6180735bf1",
    "src/oc_calibration.h": "1326efef484ebeb2ebcab2a6a2d19e7fcac8fd51b0a3d90368fb3f52e3f09d14",
    "src/oc_cubic_bspline.cpp": "607b8181c4991f7e368c6465deaeece576a9c4dc2aa647fc4146b5c14a2db264",
    "src/oc_cubic_bspline.h": "6f5fc8a8b6339c830fe8b575db13515f260f704a6efe57c0dfdf72c718760a5e",
    "src/oc_deformation.cpp": "6366697d58737d410f5369086196e0d053a01447d2d4d80f849446edd38b8a2a",
    "src/oc_deformation.h": "e99ee491cbefd6a01d148f2aad8dd6b10209dc69b61f18391bc7c4fd94aa37bf",
    "src/oc_dic.cpp": "8fc3d796397dd80fc2fa3c9d1ae2ac4e6ad6fd8cef425b89ddea902749a26987",
    "src/oc_epipolar_search.cpp": "b862a97d0f8c3421842261f468968789e374967ab504db6d2ec968ccd61ba913",
    "src/oc_epipolar_search.h": "121f3271588689fc549887f4ef1e0a39b982546284603217889d094783b7adc0",
    "src/oc_feature.h": "cc36de8d7c36fce2ee8076b2d15ec2dc27b51339b71460898fc02fd95af89cd4",
    "src/oc_feature_affine.cpp": "849df372a1a0c35ac7d862b6ea99a2f28ee0cfa58f794f703efaf1fd91418c5f",
    "src/oc_feature_affine.h": "c761aa886c50334b2247a66411c848321625d9cf12f97d4c97b582dcf1131e04",
    "src/oc_gradient.cpp": "a9132928072fed1e6da70acf06d571ef5e42453bcc2e83d67660b8dfc59c0868",
    "src/oc_gradient.h": "fe3a9548906f0e7b68ec1407aa7a51058fe7728e9284190b8ab7b05295995e1c",
    "src/oc_icgn.h": "5c7d7e741b6d312dd28eb5713456a36aa62db068ffb9ae23daa4b054223e0ee5",
    "src/oc_iclm.cpp": "6ab36ee6112db56d9fe0df7810ebe717d9dafbdc6afdeac57dd4ac5604d7027c",
    "src/oc_iclm.h": "2c384573a12e8e7dfaea3e9da69fe8996051e796289e7ec6a923eb9c21d92deb",
    "src/oc_interpolation.h": "21065945649e0d1a50f508468e95755d2bc8e55dc7be693cf5bbdba54f26d9bc",
    "src/oc_io.cpp": "7ff9dc2ba0a64669aa03929b7d24df37923650d2958bc249a0b4804968bba84c",
    "src/oc_io.h": "760fa749f5592caa5a44040e7b00eb4d749b59719c49973356279f7958b62385",
    "src/oc_nearest_neighbor.cpp": "64800e22ba0b62444dff23acb5340ec99e01c0d901af2be360261c54025981b8",
    "src/oc_nearest_neighbor.h": "2a74a9f7375efaed6b23ef18be4dc5638033da8ae63fb0c954dcee02f9cfd0e5",
    "src/oc_nr.cpp": "ffd6494ab07703ed2538afd7bd02c5cdfa1d16ff00afd3d745c6780b408383ec",
    "src/oc_nr.h": "c340cb602740b8e8fadb021a5079812511dd3aa4ff41db132a6a393eddeeed6e",
    "src/oc_poi.h": "4e7de25ef1b24ce1d5430409c4b0a9d2f4a0d0ac9607a60806fa413f8e63e456",
    "src/oc_point.h": "2f6b8d7e1ad0c8c6b73e0d4f7ee483ebfa59f820757722454fddcae9e87841a5",
    "src/oc_region_fit.cpp": "dda3e56969656abfe42cc67fa09f86b5669d8cd6d8ba7d5d801d58b02b5d64e9",
    "src/oc_region_fit.h": "81e202fd67e6ddfa071558febadde169cfd1a7091cd66151a11a9486e6d918ce",
    "src/oc_sift.cpp": "9a98d9e5adaca32fdd01f2183666407c72b35def55b91f898fc8626df11690ce",
    "src/oc_sift.h": "374283391a0b859736eea0a93d376f4cfc4bffce4539e944beab3f5d1f561ee6",
    "src/oc_stereovision.cpp": "eb9b07535c3365cda986b0b0885409d03ddf2a67f4048fdeb7071e2965916626",
    "src/oc_stereovision.h": "ff23885f4fe999590ecb38277eea134cf753262e7593a34ad16d07d1e4413458",
    "src/oc_strain.cpp": "9ba6ecd3f1f7fc36094c5e201fa4d1e6d09f21a233d97d6ca8ab3061581be42a",
    "src/oc_strain.h": "3ea300d38d5416ced42c0f2cc115d9e8eef619c0d6239df8551e2e1d78bbad93",
    "src/oc_subset.cpp": "2c40b20d78d018990154b9d3c3a407e0c4eb5ee838e4937dcc1dde3df8226303",
    "src/oc_subset.h": "b58381e409d0474669f6f5a6bf0a32216ac0c3e959041157d867b74517a33c5c",
    "src/opencorr.h": "747d48b9b4a5f7f3b1f34a51f9f0122b37fa8139b084983f0eb57d66aba8f511",
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_toplevel(root: pathlib.Path) -> pathlib.Path | None:
    """The work tree `root` belongs to, or None (not a checkout, or no git)."""
    try:
        probe = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return None
    if probe.returncode != 0 or not probe.stdout.strip():
        return None
    return pathlib.Path(probe.stdout.strip()).resolve()


def tracked_files(root: pathlib.Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        capture_output=True, text=True, check=True,
    ).stdout
    return sorted(line for line in out.splitlines() if line)


def needs_first_party_header(rel: str) -> bool:
    if rel in SKIP_EXACT or rel.startswith(SKIP_PREFIXES):
        return False
    if rel in UPSTREAM_SHA256 or rel in VENDORED_MODIFIED or rel == NANOFLANN:
        return False
    path = pathlib.PurePosixPath(rel)
    if path.name in SOURCE_NAMES:
        return True
    if path.name.endswith(".h.in"):
        return True
    return path.suffix in SOURCE_SUFFIXES and path.suffix not in {".in"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(pathlib.Path(__file__).resolve().parent.parent),
        help="repository root (default: parent of this script)",
    )
    args = parser.parse_args()
    root = pathlib.Path(args.root).resolve()

    # The file list comes from git ls-files, so an unpacked archive (GitHub's
    # "Download ZIP", `git archive`) cannot be checked: say so and report a
    # skip (exit 77, the code scripts/verify.sh uses for a skipped step)
    # instead of failing or passing vacuously on an outer repository's list.
    if git_toplevel(root) != root:
        print(f"SKIPPED: {root} is not the top of a git checkout (or git is not "
              "installed); this check reads `git ls-files`, so run it inside a "
              "git clone of the repository.")
        return EXIT_SKIPPED

    missing_header: list[str] = []
    missing_copyright: list[str] = []
    drifted_vendored: list[str] = []
    missing_marker: list[str] = []
    missing_upstream_notice: list[str] = []
    unchecked_vendored: list[str] = []
    checked = 0

    files = tracked_files(root)
    for rel in files:
        path = root / rel
        if not path.is_file():
            continue
        if needs_first_party_header(rel):
            checked += 1
            head = path.read_text(errors="replace")[:HEADER_WINDOW]
            if MPL_NOTICE not in head:
                missing_header.append(rel)
            if COPYRIGHT not in head:
                missing_copyright.append(rel)

    for rel, expected in sorted(UPSTREAM_SHA256.items()):
        path = root / rel
        if not path.is_file():
            unchecked_vendored.append(rel)
            continue
        if sha256(path) != expected:
            drifted_vendored.append(rel)

    nano = root / NANOFLANN
    if not nano.is_file():
        unchecked_vendored.append(NANOFLANN)
    elif sha256(nano) != NANOFLANN_SHA256:
        drifted_vendored.append(NANOFLANN)

    for rel in sorted(VENDORED_MODIFIED):
        path = root / rel
        if not path.is_file():
            unchecked_vendored.append(rel)
            continue
        head = path.read_text(errors="replace")[:HEADER_WINDOW]
        if MPL_NOTICE not in head:
            missing_upstream_notice.append(rel)
        if MODIFICATION_MARKER not in head:
            missing_marker.append(rel)

    def report(label: str, items: list[str]) -> None:
        if items:
            print(f"{label} ({len(items)}):")
            for item in items:
                print(f"  {item}")

    print(f"tracked files:                  {len(files)}")
    print(f"first-party files checked:      {checked}")
    print(f"vendored pinned to baseline:    {len(UPSTREAM_SHA256) + 1}")
    print(f"vendored modified (marker req): {len(VENDORED_MODIFIED)}")

    report("MISSING MPL-2.0 notice", missing_header)
    report("MISSING copyright line", missing_copyright)
    report("VENDORED FILE DRIFTED FROM BASELINE", drifted_vendored)
    report("MODIFIED VENDORED FILE LOST UPSTREAM NOTICE", missing_upstream_notice)
    report("MODIFIED VENDORED FILE LOST MODIFICATION MARKER", missing_marker)
    report("EXPECTED VENDORED FILE NOT FOUND", unchecked_vendored)

    failed = (
        missing_header
        or missing_copyright
        or drifted_vendored
        or missing_upstream_notice
        or missing_marker
        or unchecked_vendored
    )
    if failed:
        print("\nFAIL")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
