"""Fetch EU5 game data from Windows PC via SSH.

Usage:
    python pipeline/fetch_raw.py [category ...]
    python pipeline/fetch_raw.py              # fetch all
    python pipeline/fetch_raw.py building_types religions
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
RAW_DIR = PIPELINE_DIR / "raw"

EU5_BASE = r"C:\Program Files (x86)\Steam\steamapps\common\Europa Universalis V"
GAME = rf"{EU5_BASE}\game\in_game"
LOC = rf"{EU5_BASE}\game\main_menu\localization"


def ssh_cmd(cmd: str) -> str:
    """Run a command on winpc via SSH and return stdout."""
    result = subprocess.run(
        ["ssh", "winpc", f'powershell -Command "chcp 65001 | Out-Null; {cmd}"'],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout


def ssh_read_file(remote_path: str) -> str:
    """Read a single file from winpc with proper UTF-8 encoding."""
    result = subprocess.run(
        [
            "ssh",
            "winpc",
            f"""powershell -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; chcp 65001 | Out-Null; [System.IO.File]::ReadAllText('{remote_path}', [System.Text.Encoding]::UTF8)" """,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout


def list_remote_files(remote_dir: str, pattern: str = "*.txt") -> list[str]:
    """List files in a remote directory."""
    output = ssh_cmd(
        f"Get-ChildItem '{remote_dir}' -Filter '{pattern}' -File "
        f"| Select-Object -ExpandProperty Name"
    )
    return [f.strip() for f in output.splitlines() if f.strip()]


def fetch_dir(remote_dir: str, local_dir: Path, pattern: str = "*.txt") -> int:
    """Fetch all matching files from remote dir."""
    local_dir.mkdir(parents=True, exist_ok=True)
    files = list_remote_files(remote_dir, pattern)
    print(f"  {remote_dir}")
    for fname in files:
        remote_path = rf"{remote_dir}\{fname}"
        content = ssh_read_file(remote_path)
        # Normalize line endings
        content = content.replace("\r\n", "\n")
        (local_dir / fname).write_text(content, encoding="utf-8")
    print(f"    {len(files)} files")
    return len(files)


def fetch_loc(loc_name: str) -> None:
    """Fetch English and Japanese localization for a given name."""
    loc_dir = RAW_DIR / "localization"
    loc_dir.mkdir(parents=True, exist_ok=True)
    for lang in ("english", "japanese"):
        fname = f"{loc_name}_l_{lang}.yml"
        remote_path = rf"{LOC}\{lang}\{fname}"
        try:
            content = ssh_read_file(remote_path)
            if content.strip():
                content = content.replace("\r\n", "\n")
                (loc_dir / fname).write_text(content, encoding="utf-8")
                print(f"    {fname}")
            else:
                print(f"    {fname} (empty, skipping)")
        except Exception:
            print(f"    {fname} (not found)")


# --- categories ---

CATEGORIES: dict[str, callable] = {}


def category(name: str):
    def decorator(func):
        CATEGORIES[name] = func
        return func
    return decorator


@category("goods")
def fetch_goods():
    print("[goods]")
    fetch_dir(rf"{GAME}\common\goods", RAW_DIR / "goods")
    fetch_loc("goods")


@category("building_types")
def fetch_building_types():
    print("[building_types]")
    fetch_dir(rf"{GAME}\common\building_types", RAW_DIR / "building_types")
    fetch_loc("buildings")


@category("religions")
def fetch_religions():
    print("[religions]")
    fetch_dir(rf"{GAME}\common\religions", RAW_DIR / "religions")
    fetch_dir(rf"{GAME}\common\religion_groups", RAW_DIR / "religion_groups")
    fetch_loc("religion")


@category("countries")
def fetch_countries():
    print("[countries]")
    fetch_dir(rf"{GAME}\setup\countries", RAW_DIR / "countries")
    fetch_loc("countries")
    fetch_loc("country_names")


@category("cultures")
def fetch_cultures():
    print("[cultures]")
    fetch_dir(rf"{GAME}\common\cultures", RAW_DIR / "cultures")
    fetch_dir(rf"{GAME}\common\culture_groups", RAW_DIR / "culture_groups")
    fetch_loc("cultures")


@category("government_types")
def fetch_government_types():
    print("[government_types]")
    fetch_dir(rf"{GAME}\common\government_types", RAW_DIR / "government_types")
    fetch_loc("government")


@category("laws")
def fetch_laws():
    print("[laws]")
    fetch_dir(rf"{GAME}\common\laws", RAW_DIR / "laws")
    fetch_loc("laws")


def fetch_version():
    print("[version]")
    branch = ssh_cmd(f"Get-Content '{EU5_BASE}\\caesar_branch.txt' -Encoding UTF8").strip()
    rev = ssh_cmd(f"Get-Content '{EU5_BASE}\\caesar_rev.txt' -Encoding UTF8").strip()
    version = branch.split("/")[-1] if "/" in branch else branch
    ver_file = RAW_DIR / "VERSION.txt"
    ver_file.write_text(f"version: {version}\nbranch: {branch}\nrev: {rev}\n")
    print(f"  {version} ({branch})")


def main():
    args = sys.argv[1:]
    cats = args if args else list(CATEGORIES.keys())

    print("=== EU5 Raw Data Fetch ===")
    fetch_version()

    for cat in cats:
        if cat in CATEGORIES:
            CATEGORIES[cat]()
        else:
            print(f"[WARN] Unknown category: {cat}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
