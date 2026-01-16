from __future__ import annotations

import re
from pathlib import Path

TARGET = Path(r"src/eu5_locations_master_builder.py")

HELPER_BLOCK = r'''
# -----------------------------------------------------------------------------
# Diagnostic logging (opt-in; must not affect results)
# -----------------------------------------------------------------------------
_DIAGNOSTIC_LOGGER = None


def _diagnostic_enabled(argv) -> bool:
    # Minimal argv scan. Default behavior must remain identical when absent.
    try:
        return "--diagnostic" in argv[1:]
    except Exception:
        return False


def _enable_diagnostic_logger():
    """
    Enable file-only diagnostic logger.
    Must not change stdout/stderr output or any computation results.
    """
    global _DIAGNOSTIC_LOGGER
    if _DIAGNOSTIC_LOGGER is not None:
        return _DIAGNOSTIC_LOGGER

    log_path = os.path.join(".", "artifacts", "diagnostic_lake_adjacency.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("eu5_locations_master.diagnostic")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if main() is executed multiple times in one process
    if not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", "").endswith("diagnostic_lake_adjacency.log")
        for h in logger.handlers
    ):
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)

    _DIAGNOSTIC_LOGGER = logger
    return logger
'''.lstrip("\n")

def fail(msg: str) -> None:
    raise SystemExit(f"[apply_diagnostic_flag] ERROR: {msg}")

def main() -> None:
    if not TARGET.exists():
        fail(f"Target file not found: {TARGET}")

    original = TARGET.read_text(encoding="utf-8")
    text = original

    # 1) Ensure import logging exists
    if re.search(r"^import\s+logging\s*$", text, flags=re.M) is None:
        # Insert after the last contiguous import line near top
        m = re.search(r"^(?:import|from)\s+.+$", text, flags=re.M)
        if not m:
            fail("No import block found to insert 'import logging'.")
        # Find end of initial import region (first blank line after imports)
        imp_end = None
        for mm in re.finditer(r"^(?:import|from)\s+.+$", text, flags=re.M):
            imp_end = mm.end()
        assert imp_end is not None
        text = text[:imp_end] + "\nimport logging" + text[imp_end:]

    # 2) Insert helper block before resolve_adjacent_to_lake_ids definition
    if "_diagnostic_enabled" not in text:
        anchor = "def resolve_adjacent_to_lake_ids"
        pos = text.find(anchor)
        if pos < 0:
            fail("Anchor not found: def resolve_adjacent_to_lake_ids")
        text = text[:pos] + HELPER_BLOCK + "\n\n" + text[pos:]

    # 3) Add logger fallback inside resolve_adjacent_to_lake_ids
    # Insert immediately after the comment '# Optional diagnostic logging' if present,
    # otherwise before 'if logger is not None:' inside the function.
    if "logger = _DIAGNOSTIC_LOGGER" not in text:
        # Prefer comment anchor
        pat = r"(^\s*# Optional diagnostic logging.*$\n)"
        m = re.search(pat, text, flags=re.M)
        if m:
            insert_at = m.end(1)
            # Determine indentation from following lines in function body
            # Use 4 spaces (function-level) as conservative default.
            fallback = "    # If --diagnostic is enabled, allow global file-only logger when logger is not explicitly provided.\n" \
                       "    if logger is None and _DIAGNOSTIC_LOGGER is not None:\n" \
                       "        logger = _DIAGNOSTIC_LOGGER\n"
            text = text[:insert_at] + fallback + text[insert_at:]
        else:
            # Fallback: insert before first 'if logger is not None:' within the function
            m2 = re.search(r"(^\s*if\s+logger\s+is\s+not\s+None\s*:\s*$)", text, flags=re.M)
            if not m2:
                fail("Could not find insertion point for logger fallback.")
            insert_at = m2.start(1)
            fallback = "    # Optional diagnostic logging (no effect on results)\n" \
                       "    # If --diagnostic is enabled, allow global file-only logger when logger is not explicitly provided.\n" \
                       "    if logger is None and _DIAGNOSTIC_LOGGER is not None:\n" \
                       "        logger = _DIAGNOSTIC_LOGGER\n\n"
            text = text[:insert_at] + fallback + text[insert_at:]

    # 4) Enable diagnostic logger in main() immediately after the version print
    if "_enable_diagnostic_logger()" not in text or "if _diagnostic_enabled(sys.argv):" not in text:
        needle = 'print(f"[INFO] {TOOL_NAME} {TOOL_VERSION}")'
        idx = text.find(needle)
        if idx < 0:
            fail("Anchor not found in main(): print(f\"[INFO] {TOOL_NAME} {TOOL_VERSION}\")")
        after = idx + len(needle)
        insert = "\n\n    # Diagnostic flag: opt-in only; must not affect results/output when absent.\n" \
                 "    if _diagnostic_enabled(sys.argv):\n" \
                 "        _enable_diagnostic_logger()\n"
        # Avoid double insert if already present
        if "Diagnostic flag: opt-in only" not in text[idx:idx+300]:
            text = text[:after] + insert + text[after:]

    if text == original:
        print("[apply_diagnostic_flag] No changes needed.")
        return

    TARGET.write_text(text, encoding="utf-8")
    print("[apply_diagnostic_flag] Applied changes successfully.")

if __name__ == "__main__":
    main()
