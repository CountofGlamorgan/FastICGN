#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The version the binary prints (CMakeLists.txt) must be the one CITATION.cff
and pyproject.toml declare. Needs no build: scripts/verify.sh runs it in the
static tier, ctest as version_metadata."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cmake_version():
    text = (ROOT / 'CMakeLists.txt').read_text(encoding='utf-8')
    project = re.search(r'project\(opencorr_gpu_mvp\s+VERSION\s+(\d+\.\d+\.\d+)', text)
    suffix = re.search(r'set\(OPENCORR_GPU_MVP_VERSION_SUFFIX\s+"([^"]*)"\)', text)
    assert project and suffix, 'CMakeLists.txt no longer declares the version the way this test reads it'
    return project.group(1), suffix.group(1)


class VersionMetadataTests(unittest.TestCase):
    def test_suffix_is_empty_or_a_semver_prerelease(self):
        self.assertRegex(cmake_version()[1], r'^(|-[0-9A-Za-z.]+)$')

    def test_citation_carries_the_binary_version_with_a_leading_v(self):
        number, suffix = cmake_version()
        citation = (ROOT / 'CITATION.cff').read_text(encoding='utf-8')
        declared = re.search(r'(?m)^version:\s*"([^"]+)"\s*$', citation)
        self.assertIsNotNone(declared, 'CITATION.cff has no quoted top-level version')
        self.assertEqual(declared.group(1), 'v' + number + suffix)

    def test_pyproject_carries_the_pep440_spelling(self):
        number, suffix = cmake_version()
        pyproject = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
        declared = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', pyproject)
        self.assertIsNotNone(declared, 'pyproject.toml has no version')
        expected = {'': number, '-dev': number + '.dev0'}.get(suffix)
        self.assertIsNotNone(expected, f'add the PEP 440 spelling of suffix {suffix!r} to this test')
        self.assertEqual(declared.group(1), expected)


if __name__ == '__main__':
    unittest.main()
