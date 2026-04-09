"""Fetch EU5 icon DDS files from Windows PC via SSH and convert to PNG.

Usage:
    python pipeline/fetch_icons.py [category ...]
    python pipeline/fetch_icons.py              # fetch all
    python pipeline/fetch_icons.py trade_goods buildings
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

PIPELINE_DIR = Path(__file__).parent
RAW_DIR = PIPELINE_DIR / "raw" / "icons"
OUTPUT_DIR = PIPELINE_DIR / "output" / "icons"

EU5_BASE = r"C:\Program Files (x86)\Steam\steamapps\common\Europa Universalis V"
ICONS_BASE = rf"{EU5_BASE}\game\main_menu\gfx\interface\icons"

# category name -> remote subdirectory
ICON_CATEGORIES = {
    "trade_goods": "trade_goods",
    "buildings": "buildings",
    "building_categories": "building_categories",
    "religion": "religion",
    "religious_aspects": "religious_aspects",
    "religious_doctrines": "religious_doctrines",
    "religious_tenets": "religious_tenets",
    "religious_schools": "religious_schools",
    "government_types": "government_types",
    "laws": "laws",
    "pops": "pops",
    "holy_site_types": "holy_site_types",
}


def list_remote_dds(remote_dir: str) -> list[str]:
    """List .dds files in a remote directory."""
    result = subprocess.run(
        [
            "ssh", "winpc",
            f'powershell -Command "Get-ChildItem \'{remote_dir}\' -Filter \'*.dds\' -Name"',
        ],
        capture_output=True, text=True, timeout=30,
    )
    return [f.strip() for f in result.stdout.splitlines() if f.strip()]


def fetch_dds_files(remote_dir: str, local_dir: Path) -> list[Path]:
    """Fetch all DDS files from a remote directory using scp."""
    local_dir.mkdir(parents=True, exist_ok=True)
    files = list_remote_dds(remote_dir)
    if not files:
        print(f"    No DDS files found")
        return []

    # scp the entire directory's DDS files
    scp_src = f"winpc:{remote_dir.replace(chr(92), '/')}/*.dds"
    result = subprocess.run(
        ["scp", scp_src, str(local_dir) + "/"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        # scp glob might not work; fall back to individual files
        for fname in files:
            remote_path = f"winpc:{remote_dir.replace(chr(92), '/')}/{fname}"
            subprocess.run(
                ["scp", remote_path, str(local_dir) + "/"],
                capture_output=True, text=True, timeout=30,
            )

    fetched = list(local_dir.glob("*.dds"))
    print(f"    {len(fetched)} DDS files fetched")
    return fetched


def convert_dds_to_png(dds_files: list[Path], output_dir: Path, size: int = 64) -> int:
    """Convert DDS files to PNG, resized for web use."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for dds_path in dds_files:
        png_name = dds_path.stem + ".png"
        png_path = output_dir / png_name
        try:
            img = Image.open(dds_path)
            if img.size != (size, size):
                img = img.resize((size, size), Image.LANCZOS)
            img.save(png_path, "PNG")
            count += 1
        except Exception as e:
            print(f"    [WARN] {dds_path.name}: {e}")
    print(f"    {count} PNG files converted ({size}x{size})")
    return count


def fetch_category(name: str) -> None:
    """Fetch and convert icons for a single category."""
    subdir = ICON_CATEGORIES[name]
    remote_dir = rf"{ICONS_BASE}\{subdir}"
    raw_dir = RAW_DIR / name
    out_dir = OUTPUT_DIR / name

    print(f"[{name}]")
    dds_files = fetch_dds_files(remote_dir, raw_dir)
    if dds_files:
        convert_dds_to_png(dds_files, out_dir)


def main():
    args = sys.argv[1:]
    cats = args if args else list(ICON_CATEGORIES.keys())

    print("=== EU5 Icon Fetch & Convert ===")

    for cat in cats:
        if cat in ICON_CATEGORIES:
            fetch_category(cat)
        else:
            print(f"[WARN] Unknown category: {cat}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
