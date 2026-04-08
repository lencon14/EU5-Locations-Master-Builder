"""Parser for Paradox localization files (*_l_english.yml, *_l_japanese.yml).

Format:
    l_english:
     key: "value"
     key_desc: "long description..."

Values may contain Paradox markup like [SCOPE.func], #T ... #!, @icon, etc.
"""

from __future__ import annotations

import re
from pathlib import Path

# Matches: <space>key: "value"
_LOC_LINE_RE = re.compile(r'^\s+(\S+):\s*"(.*)"$')


def parse_loc(text: str) -> dict[str, str]:
    """Parse a Paradox localization text into a {key: value} dict."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        m = _LOC_LINE_RE.match(line)
        if m:
            result[m.group(1)] = m.group(2)
    return result


def parse_loc_file(path: str | Path) -> dict[str, str]:
    """Parse a Paradox localization file."""
    text = Path(path).read_text(encoding="utf-8-sig")
    return parse_loc(text)


def strip_markup(text: str) -> str:
    """Remove Paradox markup tags from a localization string.

    Strips: #T ... #!, #tooltip_subheading, #italic ... #!, @icon!, [SCOPE...],
    and other common inline tags.
    """
    # Remove #T ... #! wrappers (keep inner text)
    text = re.sub(r"#T\s*", "", text)
    text = re.sub(r"#!", "", text)
    # Remove #italic ... wrappers
    text = re.sub(r"#italic\s*", "", text)
    # Remove #tooltip_subheading
    text = re.sub(r"#tooltip_subheading\s*", "", text)
    # Remove @icon! references
    text = re.sub(r"@\w+!", "", text)
    # Remove [SCOPE...] references
    text = re.sub(r"\[[\w.'()| ]+\]", "", text)
    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
