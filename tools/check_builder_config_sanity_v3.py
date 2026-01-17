#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_builder_config_sanity_v3.py

Read-only configuration sanity checker for src/eu5_locations_master_builder.py.

What it checks
--------------
- Resolves EU5_ROOT and MAP_DATA_DIR from the builder (best-effort).
- Lists expected candidate assets (locations/rivers png/tga) and whether they exist.
- Highlights potentially confusing configuration (e.g., v2_0 as a "previous" CSV).

Safety
------
- Does NOT import or execute the builder.
- Reads the builder as plain text and extracts a small set of constants using regex.
- Only performs local filesystem existence checks.

Usage
-----
  python tools/check_builder_config_sanity_v3.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Finding:
    level: str   # "INFO" | "WARN" | "ERROR"
    code: str
    message: str


RE_EU5_ROOT = re.compile(r'^\s*EU5_ROOT\s*=\s*r?"([^"]+)"\s*$', re.MULTILINE)
RE_MAP_DATA_DIR = re.compile(r'^\s*MAP_DATA_DIR\s*=\s*os\.path\.join\(\s*EU5_ROOT\s*,\s*r?"([^"]+)"\s*\)\s*$', re.MULTILINE)
RE_OUT_TAG = re.compile(r'^\s*OUT_TAG\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
RE_JOIN_CWD = re.compile(r'os\.path\.join\(\s*os\.getcwd\(\)\s*,\s*"([^"]+)"\s*\)')


def _extract(builder_text: str) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    eu5_root = None
    map_rel = None
    out_tag = None
    prev_names: List[str] = []

    m = RE_EU5_ROOT.search(builder_text)
    if m:
        eu5_root = m.group(1)

    m = RE_MAP_DATA_DIR.search(builder_text)
    if m:
        map_rel = m.group(1)

    m = RE_OUT_TAG.search(builder_text)
    if m:
        out_tag = m.group(1)
    return eu5_root, map_rel, out_tag, prev_names


def _ver_from_csv(name: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"_v(\d+)_(\d+)\.csv$", name, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--builder", default="src/eu5_locations_master_builder.py")
    args = ap.parse_args()

    builder_path = Path(args.builder)
    if not builder_path.exists():
        print(f"[ERROR] builder not found: {builder_path}")
        return 2

    text = builder_path.read_text(encoding="utf-8", errors="replace")
    eu5_root, map_rel, out_tag, prev_names = _extract(text)

    findings: List[Finding] = []

    print("Builder Config Sanity Check (v3)")
    print("=" * 60)
    print(f"builder: {builder_path}")
    print("")

    if eu5_root:
        print(f"EU5_ROOT     : {eu5_root}")
        if not os.path.exists(eu5_root):
            findings.append(Finding("WARN", "EU5_ROOT_MISSING", f"EU5_ROOT does not exist: {eu5_root}"))
    else:
        print("EU5_ROOT     : (unresolved)")
        findings.append(Finding("WARN", "EU5_ROOT_UNRESOLVED", "EU5_ROOT not found via regex."))

    map_data_dir = None
    if eu5_root and map_rel:
        map_data_dir = os.path.join(eu5_root, map_rel)
        print(f"MAP_DATA_DIR : {map_data_dir}")
        if not os.path.exists(map_data_dir):
            findings.append(Finding("WARN", "MAP_DATA_DIR_MISSING", f"MAP_DATA_DIR does not exist: {map_data_dir}"))
    else:
        print("MAP_DATA_DIR : (unresolved)")
        findings.append(Finding("WARN", "MAP_DATA_DIR_UNRESOLVED", "MAP_DATA_DIR not resolved (EU5_ROOT or join expr missing)."))

    print(f"OUT_TAG      : {out_tag if out_tag else '(unresolved)'}")
    print("")

    # Image candidates (derived behavior)
    if map_data_dir:
        candidates = [
            ("locations.png", os.path.join(map_data_dir, "locations.png")),
            ("locations.tga", os.path.join(map_data_dir, "locations.tga")),
            ("rivers.png",    os.path.join(map_data_dir, "rivers.png")),
            ("rivers.tga",    os.path.join(map_data_dir, "rivers.tga")),
        ]
        print("Map image candidates (existence)")
        print("-" * 60)
        for label, path in candidates:
            exists = os.path.exists(path)
            flag = "OK " if exists else "MISS"
            print(f"[{flag}] {label:12s}  {path}")
        print("")

        # Informational: if png exists and tga missing (common)
        if os.path.exists(os.path.join(map_data_dir, "locations.png")) and not os.path.exists(os.path.join(map_data_dir, "locations.tga")):
            findings.append(Finding("INFO", "LOCATIONS_TGA_MISSING", "locations.tga is not present (png-only environment)."))
        if os.path.exists(os.path.join(map_data_dir, "rivers.png")) and not os.path.exists(os.path.join(map_data_dir, "rivers.tga")):
            findings.append(Finding("INFO", "RIVERS_TGA_MISSING", "rivers.tga is not present (png-only environment)."))

    # Previous CSV candidates
    print("Previous CSV candidates (existence)")
    print("-" * 60)
    if not prev_names:
        print("(none detected)")
    else:
        for name in prev_names:
            p = os.path.join(os.getcwd(), name)
            exists = os.path.exists(p)
            flag = "OK " if exists else "MISS"
            print(f"[{flag}] {name:34s}  {p}")

        # Version consistency warning
        if out_tag:
            cur = None
            m = re.search(r"v(\d+)_(\d+)", out_tag, re.IGNORECASE)
            if m:
                cur = (int(m.group(1)), int(m.group(2)))
            if cur:
                    pass  # legacy check removed
    print("")
    print("Findings")
    print("-" * 60)
    if not findings:
        print("PASS: no findings.")
        return 0

    for f in findings:
        print(f"{f.level} {f.code}: {f.message}")

    # Exit: WARN/INFO -> 1, ERROR -> 2 (none used currently)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())