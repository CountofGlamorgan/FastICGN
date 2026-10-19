#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""NumPy reader for FastICGN frame output (raw and CSV), its manifest and its ledger.

This module is the documented way to read ``--output-format raw`` results
without re-deriving the binary layout, and it reads the per-frame CSV files
into the same field names. It adds no numerics: every value is exactly what
the CLI wrote. ``run_status()`` reports, without changing anything, whether a
multi-frame output directory looks complete.

Record layout (``opencorr-gpu-raw-v1``)
---------------------------------------
One little-endian 36-byte record per point of interest (POI), in the order the
POI queue was built (row-major: y outer, x inner, both ascending)::

    offset  0  x     int32
    offset  4  y     int32
    offset  8  u     float32
    offset 12  v     float32
    offset 16  ux    float32
    offset 20  uy    float32
    offset 24  vx    float32
    offset 28  vy    float32
    offset 32  zncc  float32

Note that the raw field order is **not** the legacy CSV column order, which is
``x,y,u,v,zncc,uy,vy,ux,vx``. The manifest carries both: ``fields`` describes
the binary record and ``csv_field_order`` describes the CSV conversion.

Coordinates, units and signs
----------------------------
* ``x`` is the image **column** index, increasing to the right; ``y`` is the
  image **row** index, increasing downward. Both are integer pixel coordinates
  in the *reference* image, and they are the POI identity: a POI keeps its
  ``(x, y)`` for the whole sequence.
* ``u`` is the displacement along ``+x`` and ``v`` the displacement along
  ``+y``, both in **pixels**, both measured from the fixed reference image to
  that frame's target image. Multiply by the calibration (mm/px) for physical
  units; the CLI never applies a calibration.
* ``ux, uy, vx, vy`` are the first-order shape-function gradients
  ``∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y``, in **1/px** (dimensionless strain-like
  quantities). They are the warp parameters the IC-GN solver converged to, not
  a post-hoc finite difference of the displacement field.
* ``zncc`` is the zero-mean normalized cross-correlation coefficient of the
  converged subset, nominally in ``[-1, 1]``.

Valid state
-----------
A row is a **valid measurement** only when ``zncc`` is finite and ``>= 0``.
This is the same rule the CLI's own frame QA and the pipeline's
``valid_ratio`` use. Rows with ``zncc < 0`` are failure/status rows: the
negative value mirrors a status code, and their ``u/v`` are retained diagnostic
state (initial seed, previous-frame seed, or the last failed iterate). They
must be excluded from displacement analysis. ``valid_mask()`` implements
exactly this rule; ``finite_displacement_mask()`` additionally requires finite
``u`` and ``v``, which analysis code usually also wants.

Output cropping
---------------
``--output-roi x1,y1,x2,y2`` and ``--output-step N`` change only what is
*written*: the ROI is an inclusive pixel rectangle and the step keeps every
N-th planned grid point in each axis. Computation and the QA statistics still
cover the full field, so a cropped file is a subset of rows, never a different
solution. When either is active the manifest carries an ``output_filter``
block; without it the file holds the whole POI grid.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Iterator

import numpy as np

RAW_FORMAT = "opencorr-gpu-raw-v1"
RAW_MANIFEST_NAME = "manifest.opencorr-raw.json"
DIAGNOSTICS_MANIFEST_NAME = "manifest.opencorr-diagnostics.json"
STATE_DIR_NAME = ".opencorr-state"
RESIDUE_PREFIX = ".opencorr-tmp-"
FRAME_FILE_RE = re.compile(r"^frame_(\d+)\.(csv|bin)$")
INTENT_NAME_RE = re.compile(r"^[0-9a-f]{32}\.json$")

#: Structured dtype of one raw record. ``itemsize`` is 36 by construction.
RAW_DTYPE = np.dtype([
    ("x", "<i4"),
    ("y", "<i4"),
    ("u", "<f4"),
    ("v", "<f4"),
    ("ux", "<f4"),
    ("uy", "<f4"),
    ("vx", "<f4"),
    ("vy", "<f4"),
    ("zncc", "<f4"),
])
RECORD_BYTES = RAW_DTYPE.itemsize

#: Legacy per-frame CSV column order, for conversion helpers.
CSV_FIELD_ORDER = ("x", "y", "u", "v", "zncc", "uy", "vy", "ux", "vx")

#: Structured dtype returned by :func:`read_csv_frame`: the raw record's field
#: names in the raw record's order, with float64 values (the CSV text carries
#: six significant digits; float64 keeps exactly the parsed number).
CSV_RECORD_DTYPE = np.dtype([
    ("x", "<i4"),
    ("y", "<i4"),
    ("u", "<f8"),
    ("v", "<f8"),
    ("ux", "<f8"),
    ("uy", "<f8"),
    ("vx", "<f8"),
    ("vy", "<f8"),
    ("zncc", "<f8"),
])


class RawFormatError(ValueError):
    """The bytes on disk do not match the documented raw layout."""


class CsvFormatError(ValueError):
    """A frame CSV does not have the documented header or row shape."""


def read_raw_frame(path: str | Path) -> np.ndarray:
    """Read one ``frame_*.bin`` into a structured array of :data:`RAW_DTYPE`.

    Raises :class:`RawFormatError` when the file length is not a whole number
    of 36-byte records, which is how a truncated payload shows up here.
    """
    path = Path(path)
    payload = path.read_bytes()
    if len(payload) % RECORD_BYTES:
        raise RawFormatError(
            f"{path} holds {len(payload)} bytes, not a multiple of the "
            f"{RECORD_BYTES}-byte record size"
        )
    return np.frombuffer(payload, dtype=RAW_DTYPE)


def read_csv_frame(path: str | Path) -> np.ndarray:
    """Read one ``frame_*.csv`` into a structured array with the raw field names.

    Columns are matched by their header names, so both the 9-column file
    (``x,y,u,v,zncc,uy,vy,ux,vx``) and the legacy 5-column one
    (``x,y,u,v,zncc``) work; a gradient column the file lacks is NaN, never 0,
    so it cannot be mistaken for a measured zero gradient. A header-only file
    gives an empty array. The result has :data:`CSV_RECORD_DTYPE` (float64
    values); :func:`valid_mask` and :func:`finite_displacement_mask` apply to it
    unchanged.

    Raises :class:`CsvFormatError` when x, y, u, v or zncc is missing, a row has
    the wrong number of fields or is not numeric (a ``#`` line included), x/y
    are not integers or do not fit int32, or the header starts with a BOM.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline()
        body = handle.read()
    if header.startswith("﻿"):
        raise CsvFormatError(
            f"{path}: the header starts with a UTF-8 byte-order mark (BOM); the binary never "
            "writes one, so the file was re-saved by another program; save it without the BOM"
        )
    names = [name.strip() for name in header.strip().split(",")]
    missing = [name for name in ("x", "y", "u", "v", "zncc") if name not in names]
    if missing:
        raise CsvFormatError(f"{path}: header lacks column(s) {missing}: {header.strip()!r}")
    if not body.strip():
        return np.zeros(0, dtype=CSV_RECORD_DTYPE)
    try:
        # comments=None: a line starting with '#' is malformed, not skipped silently.
        data = np.loadtxt(io.StringIO(body), delimiter=",", dtype=np.float64, ndmin=2,
                          comments=None)
    except ValueError as exc:
        raise CsvFormatError(f"{path}: {exc}") from exc
    if data.shape[1] != len(names):
        raise CsvFormatError(
            f"{path}: rows have {data.shape[1]} fields, the header names {len(names)}"
        )
    records = np.zeros(data.shape[0], dtype=CSV_RECORD_DTYPE)
    for field in CSV_RECORD_DTYPE.names:
        if field not in names:
            records[field] = np.nan
            continue
        column = data[:, names.index(field)]
        if field in ("x", "y"):
            if not np.all(np.isfinite(column)) or not np.all(column == np.round(column)):
                raise CsvFormatError(f"{path}: column {field} must hold integers")
            limits = np.iinfo(np.int32)
            if column.size and (column.min() < limits.min or column.max() > limits.max):
                raise CsvFormatError(f"{path}: column {field} holds values outside the int32 range")
            records[field] = column.astype(np.int32)
        else:
            records[field] = column
    return records


def frame_number(path: str | Path) -> int | None:
    """The NNNNNN of ``frame_NNNNNN.csv``/``.bin``, or None for any other name."""
    match = FRAME_FILE_RE.match(Path(path).name)
    return int(match.group(1)) if match else None


def read_frame(path: str | Path) -> np.ndarray:
    """Read a ``frame_*.bin`` (raw dtype) or ``frame_*.csv`` (CSV dtype) by suffix."""
    path = Path(path)
    if path.suffix == ".bin":
        return read_raw_frame(path)
    if path.suffix == ".csv":
        return read_csv_frame(path)
    raise ValueError(f"{path}: not a .bin or .csv frame file")


def iter_frame_dir(directory: str | Path,
                   suffix: str | None = None) -> Iterator[tuple[Path, np.ndarray]]:
    """Yield ``(path, records)`` for every ``frame_NNNNNN.csv``/``.bin`` in a directory.

    Files are ordered by frame number (then name); other files, hidden
    directories such as ``.opencorr-state`` and crash temporaries are skipped.
    ``suffix`` (``".csv"`` or ``".bin"``) restricts the kind. Unlike
    :func:`iter_frames` this needs no manifest, so it also reads CSV runs; it
    does not check a raw manifest's recorded rows and bytes.
    """
    directory = Path(directory)
    files = []
    for entry in directory.iterdir():
        number = frame_number(entry)
        if number is None or not entry.is_file():
            continue
        if suffix is not None and entry.suffix != suffix:
            continue
        files.append((number, entry.name, entry))
    for _, _, entry in sorted(files):
        yield entry, read_frame(entry)


def valid_mask(records: np.ndarray) -> np.ndarray:
    """Valid-measurement mask: finite ``zncc`` and ``zncc >= 0``."""
    zncc = records["zncc"]
    return np.isfinite(zncc) & (zncc >= 0)


def finite_displacement_mask(records: np.ndarray) -> np.ndarray:
    """:func:`valid_mask` plus finite ``u`` and ``v``."""
    return valid_mask(records) & np.isfinite(records["u"]) & np.isfinite(records["v"])


def read_manifest(path: str | Path) -> dict:
    """Read and sanity-check a raw manifest JSON file."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RawFormatError(f"{path} root must be a JSON object")
    if manifest.get("format") != RAW_FORMAT:
        raise RawFormatError(f"{path} format must be {RAW_FORMAT!r}")
    if manifest.get("endianness") != "little":
        raise RawFormatError(f"{path} endianness must be 'little'")
    if manifest.get("record_bytes") != RECORD_BYTES:
        raise RawFormatError(f"{path} record_bytes must be {RECORD_BYTES}")
    if not isinstance(manifest.get("frames"), list):
        raise RawFormatError(f"{path} frames must be an array")
    return manifest


def manifest_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / RAW_MANIFEST_NAME


def read_frame_from_manifest(frame: dict, base_dir: str | Path) -> np.ndarray:
    """Read one manifest frame entry and check its recorded rows/bytes."""
    name = frame.get("file") or frame.get("output_name")
    if not name:
        raise RawFormatError(f"manifest frame entry has no file name: {frame}")
    records = read_raw_frame(Path(base_dir) / name)
    expected_rows = frame.get("rows")
    if isinstance(expected_rows, int) and records.shape[0] != expected_rows:
        raise RawFormatError(
            f"{name} holds {records.shape[0]} rows, manifest says {expected_rows}"
        )
    expected_bytes = frame.get("bytes")
    if isinstance(expected_bytes, int) and records.nbytes != expected_bytes:
        raise RawFormatError(
            f"{name} holds {records.nbytes} bytes, manifest says {expected_bytes}"
        )
    return records


def iter_frames(manifest: str | Path | dict,
                base_dir: str | Path | None = None
                ) -> Iterator[tuple[dict, np.ndarray]]:
    """Yield ``(frame_entry, records)`` in manifest order.

    ``manifest`` may be a path (``base_dir`` then defaults to its directory) or
    an already-parsed manifest dict (``base_dir`` is then required).
    """
    if isinstance(manifest, dict):
        if base_dir is None:
            raise ValueError("base_dir is required when passing a parsed manifest")
        data = manifest
    else:
        path = Path(manifest)
        data = read_manifest(path)
        base_dir = path.parent if base_dir is None else base_dir
    for frame in data["frames"]:
        yield frame, read_frame_from_manifest(frame, base_dir)


def to_csv_columns(records: np.ndarray) -> np.ndarray:
    """Return a 2-D float64 array in the legacy CSV column order."""
    return np.column_stack([records[name].astype(np.float64) for name in CSV_FIELD_ORDER])


def read_ledger(output_dir: str | Path) -> dict:
    """Read the hidden ``.opencorr-state/`` integrity ledger of a run.

    Returns ``{"state_dir": str, "<file name>": <parsed JSON>}`` for every JSON
    record present (``contract.json``, ``complete.json``, per-frame intents,
    ``contract.pending.json``). The ledger is provenance, not measurement: it
    is read-only here and never repaired.
    """
    state = Path(output_dir) / STATE_DIR_NAME
    ledger: dict = {"state_dir": str(state), "present": state.is_dir()}
    if not state.is_dir():
        return ledger
    for item in sorted(state.iterdir()):
        if not item.is_file() or item.suffix != ".json":
            continue
        try:
            ledger[item.name] = json.loads(item.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # pragma: no cover - corrupt ledger
            ledger[item.name] = {"unreadable": str(exc)}
    return ledger


def engine_metadata(manifest: dict) -> dict:
    """Return the manifest ``engine`` block, or ``{}`` for pre-2026-09 files."""
    engine = manifest.get("engine")
    return engine if isinstance(engine, dict) else {}


RUN_STATUS_NOT_VERIFIED = (
    "payload content: the ledger stores XXH3-128 digests, which the Python standard "
    "library cannot compute, so a same-size corruption is not detected",
    "inputs: the reference and target images are not compared with the fingerprints "
    "in contract.json",
    "complete.json: its contract_digest is not checked against contract.json, so it may "
    "describe an earlier, shorter target list",
)


def _load_json(path: Path) -> tuple[object, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError) as exc:
        return None, str(exc)


def run_status(output_dir: str | Path) -> dict:
    """Report whether a multi-frame output directory looks complete. Changes nothing.

    File-level checks an agent can make without the binary (review 2026-10-03,
    cli-output section 7, steps 2-6):

    * ``.opencorr-state/contract.json`` lists the expected frame files;
    * ``complete.json`` is present (necessary, not sufficient);
    * every expected file exists, has a commit intent in the ledger, and its size
      equals the intent's ``bytes`` (raw files also ``rows * 36``);
    * raw runs: the manifest lists exactly the expected files;
    * ``--diagnostics`` runs: every target's ``<stem>.diagnostics.csv`` in
      ``diagnostics/`` (or the recorded ``diagnostics_dir``) with its intent, and
      the diagnostics manifest; ``diagnostics/`` is not a foreign entry;
    * no ``contract.pending.json`` (an interrupted append) and no
      ``.opencorr-tmp-*`` crash temporaries;
    * no entry the binary would reject on a resume.

    ``verdict`` is ``complete`` only when every check passes, ``incomplete``
    when the ledger is there but a check fails, and ``not_a_run`` without a
    readable ledger contract (a pre-ledger output, or not a binary output
    directory at all). ``content_verified`` is always False: see
    ``not_verified``. The strongest check remains rerunning the identical
    command, which re-reads every file and prints "Verified complete run".
    For the pipeline, pass ``<output-dir>/<event>`` (full-field mode) or
    ``<output-dir>/_frames_<event>`` (COD mode with --keep-frames).
    """
    root = Path(output_dir)
    state = root / STATE_DIR_NAME
    problems: list[str] = []
    status: dict = {
        "schema_version": 1,
        "path": str(root),
        "exists": root.is_dir(),
        "ledger_present": state.is_dir(),
        "contract": None,
        "complete_record": None,
        "pending_contract": (state / "contract.pending.json").exists(),
        "manifest": None,
        "frames": {
            "expected": 0, "present": 0, "missing": [], "size_mismatch": [],
            "without_intent": [], "rows_mismatch": [],
        },
        "intents": 0,
        "residue": [],
        "unexpected_entries": [],
        "problems": problems,
        "complete": False,
        "verdict": "not_a_run",
        "content_verified": False,
        "not_verified": list(RUN_STATUS_NOT_VERIFIED),
    }
    if not root.is_dir():
        problems.append("the directory does not exist")
        return status
    contract, error = _load_json(state / "contract.json")
    if contract is None or not isinstance(contract, dict) or not isinstance(contract.get("targets"), list):
        problems.append(
            "no readable .opencorr-state/contract.json: not a ledger-tracked multi-frame run "
            + (f"({error})" if error else "(targets missing)")
        )
        status["frames"]["present"] = sum(1 for entry in root.iterdir() if frame_number(entry) is not None)
        return status

    targets = contract["targets"]
    options = contract.get("options") if isinstance(contract.get("options"), dict) else {}
    status["contract"] = {
        "targets": len(targets),
        "git_hash": contract.get("git_hash"),
        "build_date": contract.get("build_date"),
        "format": options.get("format"),
        "subset_radius": options.get("radius"),
        "grid_step": options.get("step"),
        "line_ys": options.get("line_ys"),
    }
    complete, error = _load_json(state / "complete.json")
    status["complete_record"] = {
        "present": (state / "complete.json").exists(),
        "readable": isinstance(complete, dict),
        "contract_digest": complete.get("contract_digest") if isinstance(complete, dict) else None,
        "durability": complete.get("durability") if isinstance(complete, dict) else None,
    }
    if not status["complete_record"]["present"]:
        problems.append("complete.json is missing: the run never completed, or is being extended")
    elif not status["complete_record"]["readable"]:
        problems.append(f"complete.json is unreadable: {error}")
    if status["pending_contract"]:
        problems.append("contract.pending.json exists: an append was interrupted")

    intents: dict[str, dict] = {}
    for item in sorted(state.iterdir()):
        if item.is_file() and INTENT_NAME_RE.match(item.name):
            record, _ = _load_json(item)
            if isinstance(record, dict) and isinstance(record.get("key"), str):
                intents[record["key"]] = record
    status["intents"] = len(intents)

    frames = status["frames"]
    expected_names: set[str] = set()
    for target in targets:
        name = target.get("file") if isinstance(target, dict) else None
        if not isinstance(name, str):
            problems.append(f"contract target without a file name: {target!r}")
            continue
        expected_names.add(name)
        frames["expected"] += 1
        path = root / name
        if not path.is_file():
            frames["missing"].append(name)
            continue
        frames["present"] += 1
        size = path.stat().st_size
        intent = intents.get(f"frame/{name}")
        if intent is None:
            frames["without_intent"].append(name)
            continue
        if intent.get("bytes") != size:
            frames["size_mismatch"].append(name)
        rows = intent.get("rows")
        if name.endswith(".bin") and isinstance(rows, int) and rows * RECORD_BYTES != size:
            frames["rows_mismatch"].append(name)
    for label, key in (("missing", "missing"), ("size differs from its intent", "size_mismatch"),
                       ("without a commit intent", "without_intent"),
                       ("rows x 36 differs from the size", "rows_mismatch")):
        if frames[key]:
            problems.append(f"{len(frames[key])} frame file(s) {label}: {frames[key][:5]}")

    # --diagnostics (review 2026-10-04 R7): the binary writes one
    # <frame stem>.diagnostics.csv per target plus a diagnostics manifest into
    # <dir>/diagnostics/ (or --diagnostics-dir) and records an intent for each.
    expected_entries: set[str] = set()
    if options.get("diagnostics") is True:
        configured = options.get("diagnostics_dir")
        diagnostics_dir = (root / "diagnostics" if configured in (None, "", "default")
                           else Path(configured))
        diag = {"enabled": True, "dir": str(diagnostics_dir), "expected": 0, "present": 0,
                "missing": [], "size_mismatch": [], "without_intent": [],
                "manifest_present": (diagnostics_dir / DIAGNOSTICS_MANIFEST_NAME).is_file(),
                "unexpected_entries": []}
        diagnostic_names = set()
        for name in sorted(expected_names):
            diagnostic = Path(name).stem + ".diagnostics.csv"
            diagnostic_names.add(diagnostic)
            diag["expected"] += 1
            path = diagnostics_dir / diagnostic
            if not path.is_file():
                diag["missing"].append(diagnostic)
                continue
            diag["present"] += 1
            intent = intents.get(f"diagnostics/{diagnostic}")
            if intent is None:
                diag["without_intent"].append(diagnostic)
            elif intent.get("bytes") != path.stat().st_size:
                diag["size_mismatch"].append(diagnostic)
        if diagnostics_dir.is_dir():
            for entry in sorted(diagnostics_dir.iterdir()):
                if entry.name.startswith(RESIDUE_PREFIX):
                    status["residue"].append(str(entry))
                elif entry.name not in diagnostic_names and entry.name != DIAGNOSTICS_MANIFEST_NAME:
                    diag["unexpected_entries"].append(entry.name)
        for label, key in (("missing", "missing"), ("size differs from its intent", "size_mismatch"),
                           ("without a commit intent", "without_intent"),
                           ("the binary did not write", "unexpected_entries")):
            if diag[key]:
                problems.append(f"{len(diag[key])} diagnostics file(s) {label}: {diag[key][:5]}")
        if not diag["manifest_present"]:
            problems.append(f"diagnostics manifest {DIAGNOSTICS_MANIFEST_NAME} is missing")
        status["diagnostics"] = diag
        if diagnostics_dir.parent == root:
            expected_entries.add(diagnostics_dir.name)

    raw = any(name.endswith(".bin") for name in expected_names) or options.get("format") == "raw"
    metadata_names = {RAW_MANIFEST_NAME, DIAGNOSTICS_MANIFEST_NAME}
    if raw:
        manifest_file = root / RAW_MANIFEST_NAME
        manifest_status = {"present": manifest_file.is_file(), "frames": None, "matches_targets": False}
        if manifest_file.is_file():
            try:
                manifest = read_manifest(manifest_file)
                listed = [frame.get("file") or frame.get("output_name") for frame in manifest["frames"]]
                manifest_status["frames"] = len(listed)
                manifest_status["matches_targets"] = sorted(listed) == sorted(expected_names)
            except (OSError, ValueError) as exc:
                problems.append(f"raw manifest unreadable: {exc}")
        if not manifest_status["present"]:
            problems.append("raw manifest is missing")
        elif manifest_status["frames"] is not None and not manifest_status["matches_targets"]:
            problems.append(
                f"raw manifest lists {manifest_status['frames']} frames, the contract "
                f"{len(expected_names)}"
            )
        status["manifest"] = manifest_status

    for directory in (root, state):
        for entry in sorted(directory.iterdir()):
            if entry.name.startswith(RESIDUE_PREFIX):
                status["residue"].append(str(entry.relative_to(root)))
    if status["residue"]:
        problems.append(f"{len(status['residue'])} crash temporar(y/ies) {RESIDUE_PREFIX}*")
    for entry in sorted(root.iterdir()):
        name = entry.name
        if (entry == state or name in expected_names or name in metadata_names
                or name in expected_entries
                or name == "tar_list.txt" or name.startswith(RESIDUE_PREFIX)):
            continue
        status["unexpected_entries"].append(name)
    if status["unexpected_entries"]:
        problems.append(
            f"{len(status['unexpected_entries'])} entr(y/ies) the binary did not write "
            f"(it would reject a resume): {status['unexpected_entries'][:5]}"
        )

    status["complete"] = not problems
    status["verdict"] = "complete" if not problems else "incomplete"
    return status


__all__ = [
    "RAW_DTYPE", "RAW_FORMAT", "RAW_MANIFEST_NAME", "RECORD_BYTES",
    "CSV_FIELD_ORDER", "CSV_RECORD_DTYPE", "RawFormatError", "CsvFormatError",
    "read_raw_frame", "read_csv_frame", "read_frame", "frame_number", "iter_frame_dir",
    "valid_mask", "finite_displacement_mask", "read_manifest", "manifest_path",
    "read_frame_from_manifest", "iter_frames", "to_csv_columns", "read_ledger",
    "engine_metadata", "run_status",
]
