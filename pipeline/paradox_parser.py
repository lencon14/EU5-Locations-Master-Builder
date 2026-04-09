"""Parser for Paradox script files (.txt) used in EU5.

Converts brace-delimited key-value structures into Python dicts/lists.

Format examples:
    horses = {
        method = farming
        category = raw_material
        default_market_price = 3
        demand_add = { nobles = 0.25 }
        custom_tags = { old_world_goods }
    }
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

# Tokens
_TOKEN_RE = re.compile(
    r"""
    (?P<comment>\#[^\n]*)          |  # line comment
    (?P<lbrace>\{)                 |  # open brace
    (?P<rbrace>\})                 |  # close brace
    (?P<eq>=)                      |  # equals sign
    (?P<quoted>"[^"]*")            |  # quoted string
    (?P<word>[^\s={}#"]+)             # bare word / number
    """,
    re.VERBOSE,
)

Value = Union[str, int, float, bool, dict, list]


def _coerce(raw: str) -> Value:
    """Convert a raw token string to an appropriate Python type."""
    if raw == "yes":
        return True
    if raw == "no":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def tokenize(text: str) -> list[tuple[str, str]]:
    """Tokenize Paradox script text into (type, value) pairs."""
    tokens = []
    for m in _TOKEN_RE.finditer(text):
        if m.group("comment"):
            continue
        for name in ("lbrace", "rbrace", "eq", "quoted", "word"):
            val = m.group(name)
            if val is not None:
                if name == "quoted":
                    val = val[1:-1]  # strip quotes
                    name = "word"
                tokens.append((name, val))
                break
    return tokens


def _parse_block(tokens: list[tuple[str, str]], pos: int) -> tuple[dict | list, int]:
    """Parse a brace-delimited block starting after the opening '{'.

    Returns either a dict (if contents are key=value pairs)
    or a list (if contents are bare values).
    """
    items: list[tuple[str | None, Value]] = []
    while pos < len(tokens):
        typ, val = tokens[pos]
        if typ == "rbrace":
            pos += 1
            break

        if typ == "word":
            # Peek ahead for '='
            if pos + 1 < len(tokens) and tokens[pos + 1][0] == "eq":
                # key = value
                key = val
                pos += 2  # skip key and '='
                if pos >= len(tokens):
                    break
                typ2, val2 = tokens[pos]
                if typ2 == "lbrace":
                    pos += 1
                    child, pos = _parse_block(tokens, pos)
                    items.append((key, child))
                else:
                    items.append((key, _coerce(val2)))
                    pos += 1
            else:
                # Bare value (list element)
                items.append((None, _coerce(val)))
                pos += 1
        elif typ == "lbrace":
            # Nested anonymous block
            pos += 1
            child, pos = _parse_block(tokens, pos)
            items.append((None, child))
        else:
            pos += 1

    # Decide if this is a dict or list
    has_keys = any(k is not None for k, _ in items)
    has_bare = any(k is None for k, _ in items)

    if not items:
        return {}, pos

    if has_keys and not has_bare:
        # All key-value pairs → dict (handle duplicate keys by merging)
        result: dict[str, Value] = {}
        for k, v in items:
            if k in result:
                existing = result[k]
                if isinstance(existing, list):
                    existing.append(v)
                else:
                    result[k] = [existing, v]
            else:
                result[k] = v
        return result, pos

    if not has_keys:
        # All bare values → list
        return [v for _, v in items], pos

    # Mixed: key-value pairs + bare values (unusual but handle gracefully)
    result = {}
    bare = []
    for k, v in items:
        if k is not None:
            result[k] = v
        else:
            bare.append(v)
    if bare:
        result["_values"] = bare
    return result, pos


def parse(text: str) -> dict[str, Value]:
    """Parse a Paradox script text file into a dict."""
    tokens = tokenize(text)
    result: dict[str, Value] = {}
    pos = 0
    while pos < len(tokens):
        typ, val = tokens[pos]
        if typ == "word":
            if pos + 1 < len(tokens) and tokens[pos + 1][0] == "eq":
                key = val
                pos += 2
                if pos >= len(tokens):
                    break
                typ2, val2 = tokens[pos]
                if typ2 == "lbrace":
                    pos += 1
                    child, pos = _parse_block(tokens, pos)
                    v = child
                else:
                    v = _coerce(val2)
                    pos += 1
                # Handle duplicate keys (same logic as _parse_block)
                if key in result:
                    existing = result[key]
                    if isinstance(existing, list):
                        existing.append(v)
                    else:
                        result[key] = [existing, v]
                else:
                    result[key] = v
            else:
                pos += 1
        else:
            pos += 1
    return result


def parse_file(path: str | Path) -> dict[str, Value]:
    """Parse a Paradox script file."""
    text = Path(path).read_text(encoding="utf-8-sig")
    return parse(text)
