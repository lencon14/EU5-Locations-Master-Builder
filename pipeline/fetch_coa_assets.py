"""Fetch EU5 CoA DDS texture assets from Windows PC via SSH/SCP.

Fetches patterns, colored emblems, and textured emblems needed for
CoA rendering. Only fetches files not already present locally.

Usage:
    python pipeline/fetch_coa_assets.py
    python pipeline/fetch_coa_assets.py patterns          # single category
    python pipeline/fetch_coa_assets.py colored_emblems   # single category
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
RAW_DIR = PIPELINE_DIR / "raw" / "coa"

EU5_BASE = r"C:\Program Files (x86)\Steam\steamapps\common\Europa Universalis V"
COA_GFX = rf"{EU5_BASE}\game\main_menu\gfx\coat_of_arms"
NAMED_COLORS = rf"{EU5_BASE}\game\main_menu\common\named_colors"

CATEGORIES = {
    "patterns": rf"{COA_GFX}\patterns",
    "colored_emblems": rf"{COA_GFX}\colored_emblems",
    "textured_emblems": rf"{COA_GFX}\textured_emblems",
}


def list_remote_dds(remote_dir: str) -> list[str]:
    result = subprocess.run(
        ["ssh", "winpc",
         f'powershell -Command "Get-ChildItem \'{remote_dir}\' -Filter \'*.dds\' -Name"'],
        capture_output=True, text=True, timeout=30,
    )
    return [f.strip() for f in result.stdout.splitlines() if f.strip()]


def fetch_category(name: str, remote_dir: str) -> int:
    local_dir = RAW_DIR / name
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_files = list_remote_dds(remote_dir)
    existing = {p.name for p in local_dir.glob("*.dds")}
    needed = [f for f in remote_files if f not in existing]

    if not needed:
        print(f"  [{name}] {len(existing)} already present, 0 to fetch")
        return 0

    print(f"  [{name}] fetching {len(needed)} / {len(remote_files)} DDS files...")

    # Try batch scp first
    scp_src = f"winpc:{remote_dir.replace(chr(92), '/')}/*.dds"
    result = subprocess.run(
        ["scp", scp_src, str(local_dir) + "/"],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        # Fallback: individual scp
        for fname in needed:
            remote_path = f"winpc:{remote_dir.replace(chr(92), '/')}/{fname}"
            subprocess.run(
                ["scp", remote_path, str(local_dir) + "/"],
                capture_output=True, text=True, timeout=60,
            )

    fetched = list(local_dir.glob("*.dds"))
    print(f"  [{name}] {len(fetched)} DDS files total")
    return len(fetched)


def fetch_named_colors() -> None:
    """Fetch named color definitions (text file)."""
    local_dir = RAW_DIR / "named_colors"
    local_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ssh", "winpc",
         f"""powershell -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; """
         f"""chcp 65001 | Out-Null; """
         f"""[System.IO.File]::ReadAllText('{NAMED_COLORS}\\01_coa.txt', [System.Text.Encoding]::UTF8)" """],
        capture_output=True, text=True, timeout=30,
    )
    content = result.stdout.replace("\r\n", "\n")
    (local_dir / "01_coa.txt").write_text(content, encoding="utf-8")
    print(f"  [named_colors] 01_coa.txt ({len(content)} bytes)")


def main():
    args = sys.argv[1:]
    cats = args if args else list(CATEGORIES.keys())

    print("=== EU5 CoA Asset Fetch ===")
    fetch_named_colors()

    for cat in cats:
        if cat in CATEGORIES:
            fetch_category(cat, CATEGORIES[cat])
        else:
            print(f"[WARN] Unknown category: {cat}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
