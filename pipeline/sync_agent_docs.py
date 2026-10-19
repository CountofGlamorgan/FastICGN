#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Synchronize the generated pitfall section of docs/lab/simulacrum.md from interface_contract.json.

The block renders the contract's general `pitfalls` first and then, numbered on,
its `lab_notes` (facts that hold only for the author's laboratory; since
2026-10-19 they are a section of their own, outside dic_get_pitfalls)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
CONTRACT_PATH = PIPELINE_DIR / "interface_contract.json"
TARGET_PATH = REPO_ROOT / "docs" / "lab" / "simulacrum.md"
BEGIN = "<!-- BEGIN GENERATED: interface_contract.json#pitfalls -->"
END = "<!-- END GENERATED: interface_contract.json#pitfalls -->"
LAB_NOTES_HEADING = (
    "以下是契约 `lab_notes`（只对本实验室成立，不在 `dic_get_pitfalls` 里；"
    "MCP 用 `dic_describe(section='lab_notes')` 取）："
)


def render_pitfalls(contract: dict) -> str:
    lines = [BEGIN]
    index = 0
    for index, pitfall in enumerate(contract["pitfalls"], start=1):
        lines.append(f"{index}. {pitfall['agent_markdown']}")
    lab_notes = contract.get("lab_notes") or []
    if lab_notes:
        lines.extend(["", LAB_NOTES_HEADING, ""])
        for index, note in enumerate(lab_notes, start=index + 1):
            lines.append(f"{index}. {note['agent_markdown']}")
    lines.append(END)
    return "\n".join(lines)


def synchronized_text(agents_text: str, contract: dict) -> str:
    begin = agents_text.find(BEGIN)
    end = agents_text.find(END)
    if begin < 0 or end < 0 or end < begin:
        raise ValueError("docs/lab/simulacrum.md generated pitfall markers are missing or out of order")
    end += len(END)
    return agents_text[:begin] + render_pitfalls(contract) + agents_text[end:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail instead of updating drift")
    args = parser.parse_args()

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    current = TARGET_PATH.read_text(encoding="utf-8")
    expected = synchronized_text(current, contract)
    if current == expected:
        return 0
    if args.check:
        print("docs/lab/simulacrum.md generated pitfalls are out of sync")
        return 1
    TARGET_PATH.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
