from __future__ import annotations

from pathlib import Path
import re

TARGET = Path(r"src/eu5_locations_master_builder.py")

PAYLOAD_CORE = """# Input / palette sizes (pure observation)
edges_n = int(getattr(edges_u32, "shape", [0])[0]) if edges_u32 is not None else 0
rgb_n = len(rgb_to_id) if rgb_to_id is not None else 0
lake_n = len(lake_ids_set) if lake_ids_set is not None else 0
sea_n = len(sea_ids_set) if sea_ids_set is not None else 0

# Palette-derived counts (cheap recompute; avoids threading logger through builders)
lake_rgbs_n = 0
sea_rgbs_n = 0
if rgb_to_id is not None and lake_ids_set is not None:
    lake_rgbs_n = sum(1 for _, loc_id in rgb_to_id.items() if loc_id in lake_ids_set)
if rgb_to_id is not None and sea_ids_set is not None:
    sea_rgbs_n = sum(1 for _, loc_id in rgb_to_id.items() if loc_id in sea_ids_set)

logger.info(
    "lake_adjacency(input): edges=%d rgb_map=%d lake_ids=%d sea_ids=%d lake_rgbs=%d sea_rgbs=%d",
    edges_n, rgb_n, lake_n, sea_n, lake_rgbs_n, sea_rgbs_n,
)
"""

def fail(msg: str) -> None:
    raise SystemExit(f"[apply_diag_log_expand] ERROR: {msg}")

def main() -> None:
    if not TARGET.exists():
        fail(f"Target not found: {TARGET}")

    text = TARGET.read_text(encoding="utf-8")

    if "lake_adjacency(input): edges=" in text:
        print("[apply_diag_log_expand] No changes needed.")
        return

    # Find logger.info that logs lake_adjacency(image) (multi-line or single-line)
    pat_multiline = re.compile(
        r"^(?P<indent>[ \t]*)logger\.info\(\s*\n(?:(?:.|\n)*?)^[ \t]*[\"']lake_adjacency\(image\):",
        re.M,
    )
    m = pat_multiline.search(text)

    if m:
        indent = m.group("indent")
        insert_at = m.start()
    else:
        pat_single = re.compile(r"^(?P<indent>[ \t]*)logger\.info\(.*lake_adjacency\(image\):", re.M)
        m2 = pat_single.search(text)
        if not m2:
            fail("Could not locate lake_adjacency(image) logger.info block in source.")
        indent = m2.group("indent")
        insert_at = m2.start()

    payload = "\n".join(indent + line if line else "" for line in PAYLOAD_CORE.splitlines())
    payload += "\n"

    text = text[:insert_at] + payload + "\n" + text[insert_at:]
    TARGET.write_text(text, encoding="utf-8")
    print("[apply_diag_log_expand] Applied changes successfully.")

if __name__ == "__main__":
    main()
