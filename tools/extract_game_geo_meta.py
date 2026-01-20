#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_game_geo_meta.py

Adds to an existing locations CSV:
  - Continent
  - Subcontinent
  - MaxWinter

Inputs:
  - EU5_GAME_DIR (env) or --game-dir
  - EU5_RAW_CSV  (env) or --csv
  - definitions:
      <EU5_GAME_DIR>/in_game/map_data/definitions.txt
  - climates (winter):
      1) EU5_CLIMATES_DIR (env)
      2) <EU5_GAME_DIR>/in_game/common/climates
      3) <EU5_GAME_DIR>/common/climates

Output:
  python tools/extract_game_geo_meta.py --out ./Eeu5_locations_master_raw.csv

Notes:
  - Continent/Subcontinent are resolved by matching Province -> Area -> Region against keys
    found under (continent -> subcontinent -> ...) scope in definitions.txt.
  - Fully empty CSV records (all columns empty/whitespace) are skipped.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path


WINTER_RE = re.compile(
    r"^\s*(max_winter|winter|winter_severity|winter_level)\s*=\s*([A-Za-z0-9_]+)\s*$"
)


def _read_text_lines(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            yield line


def _is_empty_record(r: dict) -> bool:
    for v in (r or {}).values():
        if (v or "").strip():
            return False
    return True


def _read_rows(csv_path: Path) -> tuple[list[dict], list[str], int]:
    """Returns (rows, fieldnames, skipped_empty_rows)."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("CSV header not found.")
        fieldnames = list(reader.fieldnames)

        rows: list[dict] = []
        skipped = 0
        for r in reader:
            if _is_empty_record(r):
                skipped += 1
                continue
            rows.append(r)

    return rows, fieldnames, skipped


def _norm(s: str) -> str:
    return (s or "").strip()


def _alt_keys(val: str, suffix: str) -> list[str]:
    """
    Try both with and without a suffix (build-dependent).
      "foo" <-> "foo_province"
    """
    v = _norm(val)
    if not v:
        return []
    out = [v]
    if suffix:
        if v.endswith(suffix):
            out.append(v[: -len(suffix)])
        else:
            out.append(v + suffix)

    seen = set()
    uniq = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _parse_definitions_nested(defs_path: Path) -> dict[str, tuple[str | None, str | None]]:
    """
    Map any key under (continent -> subcontinent -> ...) scope to (continent, subcontinent).
    We treat:
      stack[0] = continent
      stack[1] = subcontinent
    """
    key_map: dict[str, tuple[str | None, str | None]] = {}
    stack: list[str] = []

    def cur_cont_sub() -> tuple[str | None, str | None]:
        cont = stack[0] if len(stack) >= 1 else None
        sub = stack[1] if len(stack) >= 2 else None
        return cont, sub

    for raw in _read_text_lines(defs_path):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue

        key = None
        if "=" in line and "{" in line:
            left = line.split("=", 1)[0].strip()
            if re.fullmatch(r"[A-Za-z0-9_]+", left):
                key = left

        cont, sub = cur_cont_sub()
        if key and cont and sub:
            key_map.setdefault(key, (cont, sub))

        opens = line.count("{")
        closes = line.count("}")

        for i in range(opens):
            if i == 0 and key:
                stack.append(key)
            else:
                stack.append(stack[-1] if stack else "unknown")

        for _ in range(closes):
            if stack:
                stack.pop()

    return key_map


def _find_climate_files(game_dir: Path) -> tuple[list[Path], str]:
    env_dir = _norm(os.environ.get("EU5_CLIMATES_DIR", ""))
    if env_dir:
        d = Path(env_dir)
        if d.exists() and d.is_dir():
            return sorted(d.glob("*.txt")), f"EU5_CLIMATES_DIR={d}"

    cand1 = game_dir / "in_game" / "common" / "climates"
    if cand1.exists() and cand1.is_dir():
        return sorted(cand1.glob("*.txt")), str(cand1)

    cand2 = game_dir / "common" / "climates"
    if cand2.exists() and cand2.is_dir():
        return sorted(cand2.glob("*.txt")), str(cand2)

    return [], "none"


def _parse_climate_winter(files: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}

    for p in files:
        active: str | None = None
        brace_depth = 0
        active_depth: int | None = None

        for raw in _read_text_lines(p):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue

            if active is None and "=" in line and "{" in line:
                left = line.split("=", 1)[0].strip()
                if ":" not in left and re.fullmatch(r"[A-Za-z0-9_]+", left):
                    active = left
                    active_depth = brace_depth + line.count("{") - line.count("}")

            if active is not None:
                m = WINTER_RE.match(line)
                if m:
                    val = _norm(m.group(2)).lower()
                    if val:
                        out[active] = val

            brace_depth += line.count("{")
            brace_depth -= line.count("}")

            if active is not None and active_depth is not None and brace_depth < active_depth:
                active = None
                active_depth = None

    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-dir", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    game_dir = Path(args.game_dir) if args.game_dir else Path(_norm(os.environ.get("EU5_GAME_DIR", "")))
    if not str(game_dir):
        print("ERROR: EU5_GAME_DIR is not set (or use --game-dir).", file=sys.stderr)
        return 2
    if not game_dir.exists():
        print(f"ERROR: EU5 game folder not found: {game_dir}", file=sys.stderr)
        return 2

    csv_path = Path(args.csv) if args.csv else Path(_norm(os.environ.get("EU5_RAW_CSV", "")))
    if not str(csv_path):
        print("ERROR: EU5_RAW_CSV is not set (or use --csv).", file=sys.stderr)
        return 2
    if not csv_path.exists():
        print(f"ERROR: Input CSV not found: {csv_path}", file=sys.stderr)
        return 2

    out_path = Path(args.out) if args.out else csv_path

    defs = game_dir / "in_game" / "map_data" / "definitions.txt"
    if not defs.exists():
        print(f"ERROR: definitions.txt not found: {defs}", file=sys.stderr)
        return 2

    rows, fieldnames, skipped_empty = _read_rows(csv_path)

    region_col = "Region" if "Region" in fieldnames else None
    area_col = "Area" if "Area" in fieldnames else None
    prov_col = "Province" if "Province" in fieldnames else None
    climate_col = "Climate" if "Climate" in fieldnames else None

    key_map = _parse_definitions_nested(defs)

    climate_files, climate_mode = _find_climate_files(game_dir)
    climate_to_winter = _parse_climate_winter(climate_files) if climate_files else {}

    add_cols = ["Continent", "Subcontinent", "MaxWinter"]
    new_fields = list(fieldnames)
    for c in add_cols:
        if c not in new_fields:
            new_fields.append(c)

    miss_cont = miss_sub = miss_wint = 0

    for r in rows:
        cont = sub = None

        prov_val = _norm(r.get(prov_col, "")) if prov_col else ""
        area_val = _norm(r.get(area_col, "")) if area_col else ""
        region_val = _norm(r.get(region_col, "")) if region_col else ""

        # Province -> Area -> Region
        if prov_col and prov_val:
            for k in _alt_keys(prov_val, "_province"):
                if k in key_map:
                    cont, sub = key_map[k]
                    break

        if (not cont or not sub) and area_col and area_val:
            for k in _alt_keys(area_val, "_area"):
                if k in key_map:
                    cont, sub = key_map[k]
                    break

        if (not cont or not sub) and region_col and region_val:
            for k in _alt_keys(region_val, "_region"):
                if k in key_map:
                    cont, sub = key_map[k]
                    break

        if not cont:
            miss_cont += 1
        if not sub:
            miss_sub += 1

        # MaxWinter from Climate
        wint = None
        clim = _norm(r.get(climate_col, "")) if climate_col else ""
        if climate_col and clim:
            wint = climate_to_winter.get(clim)

        if not wint:
            miss_wint += 1

        r["Continent"] = cont or ""
        r["Subcontinent"] = sub or ""
        r["MaxWinter"] = wint or ""

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=new_fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"OK: wrote {out_path.name}")
    print(f"Rows: {len(rows)}")
    print(f"Skipped empty rows: {skipped_empty}")
    print(f"Climate winter source: {climate_mode} (files={len(climate_files)})")
    print(f"Missing Continent:    {miss_cont}")
    print(f"Missing Subcontinent: {miss_sub}")
    print(f"Missing MaxWinter:    {miss_wint}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
