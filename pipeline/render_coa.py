"""Render EU5 Coat of Arms from definitions + DDS textures to PNG.

Three-stage pipeline: parse -> resolve -> render.

Usage:
    python pipeline/render_coa.py                 # all countries
    python pipeline/render_coa.py FRA ENG SWE     # specific tags
    python pipeline/render_coa.py --list-textures  # list needed DDS files
"""

from __future__ import annotations

import colorsys
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
import numpy as np

PIPELINE_DIR = Path(__file__).parent
RAW_DIR = PIPELINE_DIR / "raw"
COA_DIR = RAW_DIR / "coat_of_arms"
COA_ASSETS = RAW_DIR / "coa"
OUTPUT_DIR = PIPELINE_DIR / "output" / "icons" / "coa"

# Internal render size (3:2 ratio), then resize to final
INTERNAL_W, INTERNAL_H = 384, 256
FINAL_W, FINAL_H = 96, 64

_GREY = (128, 128, 128)
_MAX_SUB_DEPTH = 5  # prevent infinite recursion in sub resolution


# ─── Stage 1: Parse ─────────────────────────────────────────────────────

def _preprocess_variables(text: str) -> str:
    """Resolve @variable definitions and @[expressions] in CoA text."""
    variables: dict[str, str] = {}

    for m in re.finditer(r'^\s*@(\w+)\s*=\s*(.+?)(?:\s*#.*)?$', text, re.MULTILINE):
        name, value = m.group(1), m.group(2).strip()
        if value.startswith("@["):
            expr = value[2:-1]
            for vk, vv in variables.items():
                expr = expr.replace(vk, vv)
            try:
                value = str(eval(expr))
            except Exception:
                pass
        variables[name] = value

    text = re.sub(r'^\s*@\w+\s*=\s*.+$', '', text, flags=re.MULTILINE)

    for name in sorted(variables, key=len, reverse=True):
        text = text.replace(f"@{name}", variables[name])

    # Resolve remaining @[expressions] in body text
    def _eval_inline(m):
        try:
            return str(eval(m.group(1)))
        except Exception:
            return m.group(0)
    text = re.sub(r'@\[([^\]]+)\]', _eval_inline, text)

    return text


def parse_coa_files() -> tuple[dict[str, dict], dict[str, dict]]:
    """Parse all CoA definition files.

    Returns:
        (country_coas, sub_coas) where each maps name -> definition dict.
    """
    country_coas: dict[str, dict] = {}
    sub_coas: dict[str, dict] = {}

    files = [
        COA_DIR / "00_subs.txt",
        COA_DIR / "00_subs_usa.txt",
        COA_DIR / "pre_scripted_countries.txt",
        COA_DIR / "pre_scripted_countries_formable.txt",
        COA_DIR / "pre_scripted_countries_japanese_clans.txt",
        COA_DIR / "pre_scripted_countries_usa.txt",
    ]

    from paradox_parser import tokenize, _parse_block

    for fpath in files:
        if not fpath.exists():
            continue
        text = fpath.read_text(encoding="utf-8")
        text = _preprocess_variables(text)

        tokens = tokenize(text)
        pos = 0
        while pos < len(tokens):
            typ, val = tokens[pos]
            if typ == "word" and pos + 1 < len(tokens) and tokens[pos + 1][0] == "eq":
                tag = val
                pos += 2
                if pos < len(tokens) and tokens[pos][0] == "lbrace":
                    pos += 1
                    block, pos = _parse_block(tokens, pos)
                    if tag.startswith("sub_"):
                        sub_coas[tag] = block
                    else:
                        country_coas[tag] = block
                else:
                    pos += 1
            else:
                pos += 1

    print(f"Parsed {len(country_coas)} country CoAs, {len(sub_coas)} sub definitions")
    return country_coas, sub_coas


# ─── Stage 2: Resolve ───────────────────────────────────────────────────

def _parse_named_colors() -> dict[str, tuple[int, int, int]]:
    """Parse named color definitions (CoA + map) to RGB tuples."""
    colors: dict[str, tuple[int, int, int]] = {}

    for filename in ("01_coa.txt", "02_map.txt"):
        path = COA_ASSETS / "named_colors" / filename
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")
        for m in re.finditer(
            r'(\w+)\s*=\s*hsv360\s*\{\s*(\d+)\s+(\d+)\s+(\d+)\s*\}', text
        ):
            name = m.group(1)
            h, s, v = int(m.group(2)), int(m.group(3)), int(m.group(4))
            r, g, b = colorsys.hsv_to_rgb(h / 360.0, s / 100.0, v / 100.0)
            colors[name] = (int(r * 255), int(g * 255), int(b * 255))

        for m in re.finditer(
            r'(\w+)\s*=\s*rgb\s*\{\s*(\d+)\s+(\d+)\s+(\d+)\s*\}', text
        ):
            name = m.group(1)
            colors[name] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))

        # HSV 0-1 scale (not hsv360)
        for m in re.finditer(
            r'(\w+)\s*=\s*hsv\s*\{\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\}', text
        ):
            name = m.group(1)
            h, s, v = float(m.group(2)), float(m.group(3)), float(m.group(4))
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            colors[name] = (int(r * 255), int(g * 255), int(b * 255))

    return colors


NAMED_COLORS: dict[str, tuple[int, int, int]] = {}


@dataclass
class ResolvedEmblem:
    texture: str
    colors: list[tuple[int, int, int]]
    instances: list[dict]
    is_textured: bool = False


@dataclass
class SubLayer:
    """A sub CoA composited on top of the parent at a given transform."""
    coa: ResolvedCoA
    scale: tuple[float, float]
    offset: tuple[float, float]


@dataclass
class ResolvedCoA:
    tag: str
    pattern: str
    color1: tuple[int, int, int]
    color2: tuple[int, int, int]
    color3: tuple[int, int, int]
    emblems: list[ResolvedEmblem] = field(default_factory=list)
    sub_layers: list[SubLayer] = field(default_factory=list)


def _resolve_color(value, parent_colors: dict[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    """Resolve a color value to RGB."""
    if isinstance(value, tuple) and len(value) == 3:
        return value
    if isinstance(value, str):
        if value in parent_colors:
            return parent_colors[value]
        if value in NAMED_COLORS:
            return NAMED_COLORS[value]
        return _GREY
    if isinstance(value, dict):
        vals = value.get("_values", [])
        if len(vals) >= 3:
            h, s, v = float(vals[0]), float(vals[1]), float(vals[2])
            r, g, b = colorsys.hsv_to_rgb(h / 360.0, s / 100.0, v / 100.0)
            return (int(r * 255), int(g * 255), int(b * 255))
    if isinstance(value, list) and len(value) >= 3:
        try:
            h, s, v = float(value[0]), float(value[1]), float(value[2])
            r, g, b = colorsys.hsv_to_rgb(h / 360.0, s / 100.0, v / 100.0)
            return (int(r * 255), int(g * 255), int(b * 255))
        except (ValueError, TypeError):
            pass
    return _GREY


def _extract_pair(data, default=(0.0, 0.0)) -> tuple[float, float]:
    """Extract (x, y) from dict with _values, list, or default."""
    if isinstance(data, dict):
        vals = data.get("_values", list(default))
        if len(vals) >= 2:
            return (float(vals[0]), float(vals[1]))
    elif isinstance(data, list) and len(data) >= 2:
        return (float(data[0]), float(data[1]))
    return default


def _resolve_instances(inst_data) -> list[dict]:
    """Normalize instance data to list of {position, scale, offset}."""
    if not inst_data:
        return [{"position": (0.5, 0.5), "scale": (1.0, 1.0), "offset": (0.0, 0.0)}]

    instances = inst_data if isinstance(inst_data, list) else [inst_data]
    result = []
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        result.append({
            "position": _extract_pair(inst.get("position"), (0.5, 0.5)),
            "scale": _extract_pair(inst.get("scale"), (1.0, 1.0)),
            "offset": _extract_pair(inst.get("offset"), (0.0, 0.0)),
        })
    return result if result else [{"position": (0.5, 0.5), "scale": (1.0, 1.0), "offset": (0.0, 0.0)}]


def _resolve_emblem(emb_data: dict, parent_colors: dict[str, tuple],
                    is_textured: bool = False) -> ResolvedEmblem:
    texture = emb_data.get("texture", "")
    if isinstance(texture, str):
        texture = texture.strip('"')

    colors = []
    for cn in ("color1", "color2", "color3"):
        cv = emb_data.get(cn)
        if cv is not None:
            colors.append(_resolve_color(cv, parent_colors))
        else:
            colors.append(parent_colors.get(cn, _GREY))

    return ResolvedEmblem(
        texture=texture,
        colors=colors,
        instances=_resolve_instances(emb_data.get("instance")),
        is_textured=is_textured,
    )


def _build_parent_colors(coa_def: dict) -> dict[str, tuple[int, int, int]]:
    """Extract and resolve color1-4 from a CoA definition."""
    parent_colors: dict[str, tuple[int, int, int]] = {}
    for cn in ("color1", "color2", "color3", "color4"):
        cv = coa_def.get(cn)
        if cv is not None:
            parent_colors[cn] = _resolve_color(cv, parent_colors)
    return parent_colors


def _resolve_emblems(coa_def: dict, parent_colors: dict) -> list[ResolvedEmblem]:
    """Resolve all colored/textured emblems from a definition."""
    emblems: list[ResolvedEmblem] = []

    ce_data = coa_def.get("colored_emblem")
    if ce_data is not None:
        ce_list = ce_data if isinstance(ce_data, list) else [ce_data]
        for ce in ce_list:
            if isinstance(ce, dict):
                emblems.append(_resolve_emblem(ce, parent_colors, is_textured=False))

    te_data = coa_def.get("textured_emblem")
    if te_data is not None:
        te_list = te_data if isinstance(te_data, list) else [te_data]
        for te in te_list:
            if isinstance(te, dict):
                emblems.append(_resolve_emblem(te, parent_colors, is_textured=True))

    return emblems


def _lookup_parent(name: str, sub_coas: dict, country_coas: dict) -> dict | None:
    """Look up a parent definition in sub_coas first, then country_coas."""
    return sub_coas.get(name) or country_coas.get(name)


def resolve_coa(tag: str, coa_def: dict, sub_coas: dict[str, dict],
                country_coas: dict[str, dict] | None = None,
                depth: int = 0) -> ResolvedCoA:
    """Resolve a country's CoA definition to renderable form.

    Handles sub references as compositing layers, not merges.
    """
    global NAMED_COLORS
    if not NAMED_COLORS:
        NAMED_COLORS = _parse_named_colors()
    if country_coas is None:
        country_coas = {}

    # Build parent colors from this definition
    parent_colors = _build_parent_colors(coa_def)
    color1 = parent_colors.get("color1", _GREY)
    color2 = parent_colors.get("color2", _GREY)
    color3 = parent_colors.get("color3", _GREY)

    # Extract pattern
    pattern = coa_def.get("pattern", "pattern_solid.dds")
    if isinstance(pattern, str):
        pattern = pattern.strip('"')

    # Resolve direct emblems
    emblems = _resolve_emblems(coa_def, parent_colors)

    # Resolve sub layers
    sub_layers: list[SubLayer] = []
    sub_data = coa_def.get("sub")
    if sub_data is not None and depth < _MAX_SUB_DEPTH:
        sub_refs = sub_data if isinstance(sub_data, list) else [sub_data]
        for sub_ref in sub_refs:
            if not isinstance(sub_ref, dict):
                continue

            parent_name = sub_ref.get("parent", "")
            if isinstance(parent_name, str):
                parent_name = parent_name.strip('"')

            parent_def = _lookup_parent(parent_name, sub_coas, country_coas)
            if not parent_def:
                continue

            # Build the effective definition: parent + sub overrides
            effective = dict(parent_def)
            for k, v in sub_ref.items():
                if k not in ("parent", "instance"):
                    effective[k] = v

            # Colors from sub_ref override parent's colors;
            # also inherit from the outer coa_def as context
            for cn in ("color1", "color2", "color3", "color4"):
                if cn not in effective and cn in parent_colors:
                    effective[cn] = parent_colors[cn]

            # Recursively resolve the sub's content
            sub_coa = resolve_coa(
                f"{tag}_{parent_name}", effective, sub_coas, country_coas, depth + 1
            )

            # Extract transform from the sub_ref's instance
            inst_data = sub_ref.get("instance")
            if isinstance(inst_data, dict):
                scale = _extract_pair(inst_data.get("scale"), (1.0, 1.0))
                offset = _extract_pair(inst_data.get("offset"), (0.0, 0.0))
            else:
                scale = (1.0, 1.0)
                offset = (0.0, 0.0)

            sub_layers.append(SubLayer(coa=sub_coa, scale=scale, offset=offset))

    # If this CoA has ONLY subs and no direct pattern (other than default),
    # and exactly one sub fills the canvas, promote it
    if (sub_layers and not emblems
            and coa_def.get("pattern") is None
            and len(sub_layers) == 1
            and sub_layers[0].scale == (1.0, 1.0)
            and sub_layers[0].offset == (0.0, 0.0)):
        promoted = sub_layers[0].coa
        return ResolvedCoA(
            tag=tag, pattern=promoted.pattern,
            color1=promoted.color1, color2=promoted.color2, color3=promoted.color3,
            emblems=promoted.emblems, sub_layers=promoted.sub_layers,
        )

    return ResolvedCoA(
        tag=tag, pattern=pattern,
        color1=color1, color2=color2, color3=color3,
        emblems=emblems, sub_layers=sub_layers,
    )


# ─── Stage 3: Render ────────────────────────────────────────────────────

_texture_cache: dict[str, Image.Image] = {}


def _load_texture(name: str) -> Image.Image | None:
    """Load a DDS texture, with caching."""
    if name in _texture_cache:
        return _texture_cache[name]

    for subdir in ("patterns", "colored_emblems", "textured_emblems"):
        path = COA_ASSETS / subdir / name
        if path.exists():
            try:
                img = Image.open(path).convert("RGBA")
                _texture_cache[name] = img
                return img
            except Exception as e:
                print(f"  [WARN] Failed to load {path}: {e}")
                return None
    return None


def _apply_color_mask(img: Image.Image, colors: list[tuple[int, int, int]],
                      is_emblem: bool = False) -> Image.Image:
    """Apply color tinting using channel masks.

    Patterns:  R→color1, G→color2, B→color3
    Emblems:   B→color1, G→color2, R→color3 (BGR)
    """
    arr = np.array(img, dtype=np.float32) / 255.0
    a = arr[:, :, 3]

    if is_emblem:
        m1, m2, m3 = arr[:, :, 2], arr[:, :, 1], arr[:, :, 0]
    else:
        m1, m2, m3 = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    c1 = np.array(colors[0] if len(colors) > 0 else (0, 0, 0), dtype=np.float32) / 255.0
    c2 = np.array(colors[1] if len(colors) > 1 else (0, 0, 0), dtype=np.float32) / 255.0
    c3 = np.array(colors[2] if len(colors) > 2 else (0, 0, 0), dtype=np.float32) / 255.0

    out = np.zeros((*m1.shape, 4), dtype=np.float32)
    for ch in range(3):
        out[:, :, ch] = m1 * c1[ch] + m2 * c2[ch] + m3 * c3[ch]
    out[:, :, 3] = a

    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8), "RGBA")


def _render_emblem_layer(emb: ResolvedEmblem, W: int, H: int) -> Image.Image | None:
    """Render a single emblem with all its instances onto a canvas."""
    tex = _load_texture(emb.texture)
    if tex is None:
        return None

    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    for inst in emb.instances:
        px, py = inst["position"]
        sx, sy = inst["scale"]
        ox_off, oy_off = inst.get("offset", (0.0, 0.0))

        flip_h, flip_v = sx < 0, sy < 0
        sx, sy = abs(sx), abs(sy)

        ew, eh = int(W * sx), int(H * sy)
        if ew <= 0 or eh <= 0:
            continue

        if emb.is_textured:
            layer = tex.resize((ew, eh), Image.LANCZOS).convert("RGBA")
        else:
            scaled = tex.resize((ew, eh), Image.LANCZOS).convert("RGBA")
            layer = _apply_color_mask(scaled, emb.colors, is_emblem=True)

        if flip_h:
            layer = layer.transpose(Image.FLIP_LEFT_RIGHT)
        if flip_v:
            layer = layer.transpose(Image.FLIP_TOP_BOTTOM)

        # Position: center-based + offset
        dest_x = int(px * W - ew / 2 + ox_off * W)
        dest_y = int(py * H - eh / 2 + oy_off * H)

        # Paste with bounds handling
        temp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        temp.paste(layer, (dest_x, dest_y))
        canvas.alpha_composite(temp)

    return canvas


def render_coa(resolved: ResolvedCoA) -> Image.Image:
    """Render a resolved CoA to an image."""
    W, H = INTERNAL_W, INTERNAL_H

    # Base: pattern with color1/color2/color3
    pattern_tex = _load_texture(resolved.pattern)
    if pattern_tex:
        base = pattern_tex.resize((W, H), Image.LANCZOS)
        base = _apply_color_mask(base, [resolved.color1, resolved.color2, resolved.color3])
    else:
        base = Image.new("RGBA", (W, H), resolved.color1 + (255,))

    # Composite direct emblems
    for emb in resolved.emblems:
        layer = _render_emblem_layer(emb, W, H)
        if layer:
            base.alpha_composite(layer)

    # Composite sub layers
    for sub in resolved.sub_layers:
        sub_img = render_coa(sub.coa)  # recursive render
        sx, sy = abs(sub.scale[0]), abs(sub.scale[1])
        ox, oy = sub.offset

        sw, sh = int(W * sx), int(H * sy)
        if sw <= 0 or sh <= 0:
            continue

        sub_img = sub_img.resize((sw, sh), Image.LANCZOS)

        if sub.scale[0] < 0:
            sub_img = sub_img.transpose(Image.FLIP_LEFT_RIGHT)
        if sub.scale[1] < 0:
            sub_img = sub_img.transpose(Image.FLIP_TOP_BOTTOM)

        dest_x = int(ox * W)
        dest_y = int(oy * H)

        temp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        temp.paste(sub_img, (dest_x, dest_y))
        base.alpha_composite(temp)

    return base


def render_and_save(tag: str, coa_def: dict, sub_coas: dict,
                    country_coas: dict, output_dir: Path) -> bool:
    """Render a single country's CoA and save as PNG."""
    try:
        resolved = resolve_coa(tag, coa_def, sub_coas, country_coas)
        img = render_coa(resolved)
        img = img.resize((FINAL_W, FINAL_H), Image.LANCZOS)
        output_dir.mkdir(parents=True, exist_ok=True)
        img.save(output_dir / f"{tag}.png", "PNG")
        return True
    except Exception as e:
        print(f"  [WARN] {tag}: {e}")
        return False


def list_needed_textures(tags: list[str] | None = None) -> set[str]:
    """List all DDS textures needed for given tags (or all)."""
    country_coas, sub_coas = parse_coa_files()
    textures: set[str] = set()

    def _collect(resolved: ResolvedCoA):
        textures.add(resolved.pattern)
        for emb in resolved.emblems:
            textures.add(emb.texture)
        for sl in resolved.sub_layers:
            _collect(sl.coa)

    targets = tags if tags else list(country_coas.keys())
    for tag in targets:
        coa_def = country_coas.get(tag)
        if not coa_def:
            continue
        try:
            resolved = resolve_coa(tag, dict(coa_def), sub_coas, country_coas)
            _collect(resolved)
        except Exception:
            pass
    return textures


def generate_fallback(tag: str, country_defs: dict, output_dir: Path) -> bool:
    """Generate a solid-color fallback flag from the country's map color."""
    global NAMED_COLORS
    if not NAMED_COLORS:
        NAMED_COLORS = _parse_named_colors()

    defn = country_defs.get(tag, {})
    color_val = defn.get("color")
    if not color_val:
        return False

    rgb = None
    if isinstance(color_val, str):
        if color_val in NAMED_COLORS:
            rgb = NAMED_COLORS[color_val]
        else:
            # Try _values (rgb { R G B } or map_* with color2 = rgb { ... })
            vals = defn.get("_values", [])
            if isinstance(vals, list):
                for item in vals:
                    if isinstance(item, list) and len(item) >= 3:
                        try:
                            rgb = (int(float(item[0])), int(float(item[1])), int(float(item[2])))
                        except (ValueError, TypeError):
                            pass
                        break
                if not rgb and len(vals) >= 3 and all(isinstance(v, (int, float)) for v in vals[:3]):
                    rgb = (int(vals[0]), int(vals[1]), int(vals[2]))

    if not rgb:
        return False

    img = Image.new("RGBA", (INTERNAL_W, INTERNAL_H), rgb + (255,))
    img = img.resize((FINAL_W, FINAL_H), Image.LANCZOS)
    output_dir.mkdir(parents=True, exist_ok=True)
    img.save(output_dir / f"{tag}.png", "PNG")
    return True


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if "--list-textures" in args:
        args.remove("--list-textures")
        textures = list_needed_textures(args or None)
        print(f"\nNeeded textures: {len(textures)}")
        for t in sorted(textures):
            print(f"  {t}")
        return

    with_fallback = "--with-fallback" in args
    if with_fallback:
        args.remove("--with-fallback")

    country_coas, sub_coas = parse_coa_files()

    if args:
        tags = [t.upper() for t in args]
    else:
        tags = list(country_coas.keys())

    print(f"\nRendering {len(tags)} CoAs...")
    ok = fail = skip = 0
    for tag in tags:
        coa_def = country_coas.get(tag)
        if not coa_def:
            skip += 1
            continue
        if render_and_save(tag, dict(coa_def), sub_coas, country_coas, OUTPUT_DIR):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} rendered, {fail} failed, {skip} skipped (no CoA definition)")

    # Generate fallback flags for countries without CoA definitions
    if with_fallback:
        from paradox_parser import parse_file as _pf
        country_defs: dict[str, dict] = {}
        ctr_dir = RAW_DIR / "countries"
        if ctr_dir.exists():
            for f in sorted(ctr_dir.glob("*.txt")):
                if f.name.lower() in ("readme.txt", "00_readme.info"):
                    continue
                data = _pf(f)
                for k, v in data.items():
                    if isinstance(v, dict):
                        country_defs[k] = v

        existing = {p.stem for p in OUTPUT_DIR.glob("*.png")}
        need_fallback = {t for t in country_defs if t not in existing}
        fb_ok = fb_fail = 0
        for tag in sorted(need_fallback):
            if generate_fallback(tag, country_defs, OUTPUT_DIR):
                fb_ok += 1
            else:
                fb_fail += 1
        print(f"Fallback: {fb_ok} generated, {fb_fail} failed (no color)")


if __name__ == "__main__":
    main()
