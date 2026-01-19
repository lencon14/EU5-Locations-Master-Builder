# -*- coding: utf-8 -*-
"""EU5 Locations Master Builder v1.1

Build a master CSV of Europa Universalis V (EU5) locations by joining map_data
sources from an EU5 installation and deriving image-based features.

Inputs (under the EU5 installation directory):
  - game/in_game/map_data/named_locations/00_default.txt
  - game/in_game/map_data/definitions.txt
  - game/in_game/map_data/location_templates.txt
  - game/in_game/map_data/ports.csv
  - game/in_game/map_data/locations.png
  - game/in_game/map_data/rivers.png

Outputs (current working directory):
  - eu5_locations_master_raw.csv
  - eu5_locations_master_river_overlap_water_v1_1.csv
  - eu5_locations_master_qc_flags_v1_1.csv
  - eu5_locations_master_run_report_v1_1.json
  - debug_rivers_overlay_v1_1.png

Key columns:
  - Has Coast: derived from ports.csv (LandProvince).
  - Has River: derived from rivers.png ink detection (land only).

Dependencies:
  - Required: pillow, numpy, pandas
  - Optional: scipy (used when COAST_GUARD_MODE is 'edt' or 'auto')
"""
# =============================================================================
# Imports
# =============================================================================

import os
import re
import sys
import json
import time
import hashlib
import platform
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# =============================================================================
# 0) Paths
# =============================================================================

EU5_ROOT = r"C:\Program Files (x86)\Steam\steamapps\common\Europa Universalis V"
MAP_DATA_DIR = os.path.join(EU5_ROOT, r"game\in_game\map_data")
NAMED_LOCATIONS_DIR = os.path.join(MAP_DATA_DIR, "named_locations")

FILE_00_DEFAULT  = os.path.join(NAMED_LOCATIONS_DIR, "00_default.txt")
FILE_DEFINITIONS = os.path.join(MAP_DATA_DIR, "definitions.txt")
FILE_TEMPLATES   = os.path.join(MAP_DATA_DIR, "location_templates.txt")
FILE_PORTS       = os.path.join(MAP_DATA_DIR, "ports.csv")

LOCATIONS_IMG_CANDIDATES = [
    os.path.join(MAP_DATA_DIR, "locations.png"),
]
RIVERS_IMG_CANDIDATES = [
    os.path.join(MAP_DATA_DIR, "rivers.png"),
]

# =============================================================================
# 1) Versioning + outputs
# =============================================================================

TOOL_NAME = "EU5 Locations Master Builder"
TOOL_VERSION = "v1.1"
SCHEMA_VERSION = "v1.1"   # bump only when MASTER CSV schema changes

OUT_TAG = "v1_1"          # filesystem-friendly tag
OUT_PREFIX = "eu5_locations_master"

OUTPUT_CSV          = os.path.join(os.getcwd(), f"{OUT_PREFIX}_raw.csv")
OUTPUT_RUN_REPORT   = os.path.join(os.getcwd(), f"{OUT_PREFIX}_run_report_{OUT_TAG}.json")
OUTPUT_QC_FLAGS     = os.path.join(os.getcwd(), f"{OUT_PREFIX}_qc_flags_{OUT_TAG}.csv")
OUTPUT_DIFF_SUMMARY = os.path.join(os.getcwd(), f"{OUT_PREFIX}_diff_summary_{OUT_TAG}.json")

# Diagnostic: non-land tiles with river ink overlap (two-track)
OUTPUT_RIVER_OVERLAP_WATER_CSV = os.path.join(os.getcwd(), f"{OUT_PREFIX}_river_overlap_water_{OUT_TAG}.csv")

# Features: estuary/coastal flags for all Types

DEBUG_RIVERS_OVERLAY = os.path.join(os.getcwd(), f"debug_rivers_overlay_{OUT_TAG}.png")
# =============================================================================
# 2) Parameters (accuracy-first)
# =============================================================================

SEA_TOL  = 4
LAND_TOL = 2
COAST_GUARD = 2
MIN_PIXELS = 200
MIN_RATIO  = 0.0007

COAST_GUARD_MODE = 'auto'  # 'dilate' | 'edt' | 'auto'

DEBUG_DOWNSAMPLE = 4

CACHE_ENABLE = (os.environ.get('EU5_CACHE') or '1').strip().lower() in ('1','true','yes','on')
CACHE_ENABLE = CACHE_ENABLE and (os.environ.get('EU5_NO_CACHE') or '').strip().lower() not in ('1','true','yes','on')
CACHE_DIR = os.environ.get('EU5_CACHE_DIR') or os.path.join(os.getcwd(), '.tmp', f"eu5_locations_cache_{OUT_TAG}")
# Cache schema tag for rivers payload (two-track)
CACHE_FORMAT_RIVERS = "rivers_cache_v2_two_track"

# Cache schema tag for adjacency payloads (coastal)

# =============================================================================
# 3) Domain constants
# =============================================================================

SEA_TOPO = {
    'coastal_ocean',
    'ocean',
    'deep_ocean',
    'inland_sea',
    'narrows',
    'ocean_wasteland',
}

SEA_ID_PATTERNS = [
    'sea_zones',
    'sea_zone',
    'high_sea',
    'ocean',
    'inland_sea',
    'coastal_ocean',
    'narrows',
    'sea_province',
]

WASTELAND_HINTS = ['wasteland', '_wasteland']

# =============================================================================
# 4) Small utilities
# =============================================================================

def pick_existing(path_list, label):
    for p in path_list:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"[ERROR] {label} not found. Tried: " + " | ".join(path_list))


def strip_comments(line: str) -> str:
    return line.split('#', 1)[0].strip()


def normalize_id(x) -> str:
    if x is None:
        return ''
    s = str(x).strip()
    if not s:
        return ''
    if s.lower() in {'nan', 'none', 'null'}:
        return ''
    return s


def max_abs_diff(A: np.ndarray, b_rgb: np.ndarray) -> np.ndarray:
    return np.max(np.abs(A.astype(np.int16) - b_rgb.astype(np.int16)), axis=-1)


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

# - Hashing policy (speed-first)
#   - EU5_HASH_INPUTS=1 : hash all inputs (strict but slower)
#   - EU5_HASH_MAX_BYTES: hash small files only (default 10MB)
HASH_INPUTS = (os.environ.get('EU5_HASH_INPUTS') or '').strip().lower() in ('1','true','yes','on')
HASH_MAX_BYTES = int(os.environ.get('EU5_HASH_MAX_BYTES') or str(10 * 1024 * 1024))


def safe_stat_and_hash(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {'exists': False}
    st = os.stat(path)
    sha = None
    try:
        if HASH_INPUTS or int(st.st_size) <= HASH_MAX_BYTES:
            sha = file_sha256(path)
    except Exception:
        sha = None
    return {
        'exists': True,
        'bytes': int(st.st_size),
        'mtime_utc': datetime.datetime.fromtimestamp(st.st_mtime, datetime.UTC).isoformat().replace('+00:00', 'Z'),
        'sha256': sha,
    }

# =============================================================================
# 5) Cache utilities
# =============================================================================

def _cache_mkdir():
    if CACHE_ENABLE:
        os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_paths(prefix: str):
    meta = os.path.join(CACHE_DIR, f"{prefix}.meta.json")
    data = os.path.join(CACHE_DIR, f"{prefix}.data.json")
    return meta, data


def cache_load(prefix: str, signature: dict):
    if not CACHE_ENABLE:
        return None
    _cache_mkdir()
    meta_path, data_path = _cache_paths(prefix)
    if not (os.path.exists(meta_path) and os.path.exists(data_path)):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        if meta.get('signature') != signature:
            return None
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def cache_save(prefix: str, signature: dict, data_obj):
    if not CACHE_ENABLE:
        return
    _cache_mkdir()
    meta_path, data_path = _cache_paths(prefix)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'signature': signature}, f, ensure_ascii=False, indent=2)
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(data_obj, f, ensure_ascii=False)

# =============================================================================
# 6) Parse EU5 text assets
# =============================================================================

def read_text_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def load_00_default(path: str) -> dict:
    hex_map = {}
    pat = re.compile(r'^\s*([A-Za-z0-9_\-\.]+)\s*=\s*(#?[0-9A-Fa-f]{1,6})\s*$')
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = strip_comments(line)
            if not line:
                continue
            m = pat.match(line)
            if not m:
                continue
            loc_id = normalize_id(m.group(1))
            if not loc_id:
                continue
            val = m.group(2).strip().lstrip('#')
            if re.fullmatch(r'[0-9A-Fa-f]{1,6}', val):
                val = val.zfill(6)
                hex_map[loc_id] = '#' + val.upper()
    return hex_map


def parse_paradox_blocks(text: str):
    lines = [ln.split('#', 1)[0] for ln in text.splitlines()]
    text_nc = '\n'.join(lines)
    start_re = re.compile(r'\s*([A-Za-z0-9_\-\.]+)\s*=\s*\{')
    i, n = 0, len(text_nc)
    blocks = []
    while i < n:
        m = start_re.search(text_nc, i)
        if not m:
            break
        loc_id = normalize_id(m.group(1))
        if not loc_id:
            i = m.end();
            continue
        brace_start = m.end() - 1
        depth = 0
        j = brace_start
        while j < n:
            c = text_nc[j]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            i = m.end();
            continue
        body = text_nc[brace_start+1:j]
        blocks.append((loc_id, body))
        i = j + 1
    return blocks


def load_location_templates(path: str) -> dict:
    templates = {}
    text = read_text_file(path)
    def grab(body: str, key: str) -> str:
        m = re.search(rf'\b{re.escape(key)}\s*=\s*([A-Za-z0-9_\-\.]+)', body)
        return m.group(1) if m else ''
    for loc_id, body in parse_paradox_blocks(text):
        templates[loc_id] = {
            'Topography': grab(body, 'topography'),
            'Vegetation': grab(body, 'vegetation'),
            'Climate': grab(body, 'climate'),
            'Raw Material': grab(body, 'raw_material'),
            'Harbor': grab(body, 'natural_harbor_suitability') or grab(body, 'harbor'),
        }
    return templates


def load_definitions_hierarchy(path: str) -> dict:
    hierarchy = {}
    content = ' '.join([strip_comments(ln) for ln in read_text_file(path).splitlines()])
    tokens = re.findall(r'\{|\}|=|[^\s{}=]+', content)
    stack = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ('{', '}', '='):
            if t == '}' and stack:
                stack.pop()
            i += 1
            continue
        if i + 2 < len(tokens) and tokens[i+1] == '=' and tokens[i+2] == '{':
            key = t.strip()
            if key:
                stack.append(key)
            i += 3
            continue
        leaf = normalize_id(t)
        if leaf and stack:
            region = stack[-3] if len(stack) >= 3 else ''
            area   = stack[-2] if len(stack) >= 2 else ''
            prov   = stack[-1] if len(stack) >= 1 else ''
            hierarchy[leaf] = {'RegionID': region, 'AreaID': area, 'ProvinceID': prov}
        i += 1
    return hierarchy


def load_ports(path: str) -> set:
    ports = set()
    try:
        df = pd.read_csv(path, sep=';')
        if 'LandProvince' in df.columns:
            for v in df['LandProvince'].dropna().astype(str):
                v = normalize_id(v)
                if v:
                    ports.add(v)
    except Exception as e:
        print('[WARN] ports.csv read error:', e)
    return ports

# =============================================================================
# 7) Type classification
# =============================================================================

def is_explicit_sea_id(loc_id: str, h: dict) -> bool:
    s = (loc_id or '').lower()
    for p in SEA_ID_PATTERNS:
        if p in s:
            return True
    prov = (h.get('ProvinceID') or '').lower()
    area = (h.get('AreaID') or '').lower()
    if 'sea_zones' in prov or 'sea_zone' in prov:
        return True
    if 'sea_zones' in area or 'sea_zone' in area:
        return True
    return False


def classify_type(loc_id: str, topo: str, h: dict):
    topo_l = (topo or '').lower()
    s = (loc_id or '').lower()
    if topo_l == 'lakes':
        return 'lake', 'topography=lakes'
    if topo_l in SEA_TOPO:
        return 'sea', f'topography={topo_l}'
    if any(w in topo_l for w in WASTELAND_HINTS) or any(w in s for w in WASTELAND_HINTS):
        return 'wasteland', 'wasteland_hint'
    if is_explicit_sea_id(loc_id, h):
        return 'sea', 'explicit_sea_pattern'
    return 'land', 'default'

# =============================================================================
# 8) River detection (two-track)
# =============================================================================

def _dilate_bool_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    m = mask
    for _ in range(int(max(0, iterations))):
        pad = np.pad(m, ((1,1),(1,1)), mode='constant', constant_values=False)
        m = (
            pad[0:-2,0:-2] | pad[0:-2,1:-1] | pad[0:-2,2:  ] |
            pad[1:-1,0:-2] | pad[1:-1,1:-1] | pad[1:-1,2:  ] |
            pad[2:  ,0:-2] | pad[2:  ,1:-1] | pad[2:  ,2:  ]
        )
    return m


def _count_pixels_by_id(L_rgb: np.ndarray, mask: np.ndarray, rgb_to_id: dict) -> dict:
    pixels = L_rgb[mask]
    out = defaultdict(int)
    if pixels.size == 0:
        return out
    rc, cnt = np.unique(pixels, axis=0, return_counts=True)
    for rgb, c in zip(rc, cnt):
        key = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        loc_id = rgb_to_id.get(key)
        if loc_id:
            out[loc_id] += int(c)
    return out


def detect_river_two_track(loc_img_path: str, river_img_path: str, rgb_to_id: dict) -> dict:
    img_r = Image.open(river_img_path).convert('RGB')
    img_l = Image.open(loc_img_path).convert('RGB')
    if img_r.size != img_l.size:
        raise ValueError('rivers image and locations image sizes differ')

    R = np.array(img_r, dtype=np.uint8)
    L = np.array(img_l, dtype=np.uint8)

    flat_R = R.reshape(-1, 3)
    colors, counts = np.unique(flat_R, axis=0, return_counts=True)
    sea_rgb_arr = colors[np.argmax(counts)].astype(np.uint8)
    sea_rgb = (int(sea_rgb_arr[0]), int(sea_rgb_arr[1]), int(sea_rgb_arr[2]))

    exact_mode = (colors.shape[0] <= 64)
    land_rgb_arr = np.array([255,255,255], dtype=np.uint8)

    if exact_mode:
        sea_like  = np.all(R == sea_rgb_arr, axis=-1)
        land_like = np.all(R == land_rgb_arr, axis=-1)
    else:
        sea_like  = (max_abs_diff(R, sea_rgb_arr) <= SEA_TOL)
        land_like = (max_abs_diff(R, land_rgb_arr) <= LAND_TOL)

    raw_river_mask = (~sea_like) & (~land_like)
    guarded_river_mask = raw_river_mask.copy()

    used_scipy = False
    used_fallback = False

    if COAST_GUARD and COAST_GUARD > 0:
        mode = (COAST_GUARD_MODE or 'dilate').lower()
        if mode in ('edt', 'auto'):
            try:
                from scipy.ndimage import distance_transform_edt
                sea_dist = distance_transform_edt(~sea_like)
                guarded_river_mask &= (sea_dist >= COAST_GUARD)
                used_scipy = True
            except Exception:
                if mode == 'auto':
                    used_fallback = True
                    sea_expanded = _dilate_bool_mask(sea_like, COAST_GUARD)
                    guarded_river_mask &= (~sea_expanded)
        else:
            used_fallback = True
            sea_expanded = _dilate_bool_mask(sea_like, COAST_GUARD)
            guarded_river_mask &= (~sea_expanded)

    raw_counts = _count_pixels_by_id(L, raw_river_mask, rgb_to_id)
    guarded_counts = _count_pixels_by_id(L, guarded_river_mask, rgb_to_id)

    flat_L = L.reshape(-1, 3)
    allc, alln = np.unique(flat_L, axis=0, return_counts=True)
    area_map = {(int(c[0]),int(c[1]),int(c[2])): int(n) for c,n in zip(allc, alln)}

    return {
        'sea_rgb': sea_rgb,
        'exact_mode': bool(exact_mode),
        'used_scipy': bool(used_scipy),
        'used_fallback': bool(used_fallback),
        'sea_like': sea_like,
        'land_like': land_like,
        'raw_river_mask': raw_river_mask,
        'guarded_river_mask': guarded_river_mask,
        'raw_counts': raw_counts,
        'guarded_counts': guarded_counts,
        'area_map': area_map,
    }


def finalize_river_locs_from_counts(guarded_counts: dict, id_to_rgb: dict, area_map: dict) -> set:
    river_locs = set()
    for loc_id, rp in guarded_counts.items():
        rgb = id_to_rgb.get(loc_id)
        if not rgb:
            continue
        area = area_map.get(rgb, 0)
        if area <= 0:
            continue
        if rp >= MIN_PIXELS or (rp / area) >= MIN_RATIO:
            river_locs.add(loc_id)
    return river_locs

# =============================================================================
# 10) Debug overlays
# =============================================================================

def write_rivers_debug_overlay(art: dict, out_path: str, downsample: int):
    sea_like = art['sea_like']
    land_like = art['land_like']
    river_mask = art['guarded_river_mask']
    if downsample > 1:
        sea_like = sea_like[::downsample, ::downsample]
        land_like = land_like[::downsample, ::downsample]
        river_mask = river_mask[::downsample, ::downsample]
    h, w = sea_like.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    out[land_like] = (200,200,200)
    out[sea_like]  = (40,120,255)
    out[river_mask]= (255,60,60)
    Image.fromarray(out, mode='RGB').save(out_path)


# =============================================================================
# 11) QC flags
# =============================================================================

def build_qc_flags(all_ids: set, templates: dict, hierarchy: dict, df_master: pd.DataFrame) -> pd.DataFrame:
    flags = []
    ids_set = set(all_ids)
    tpl_set = set(templates.keys())
    hie_set = set(hierarchy.keys())

    for loc_id in sorted(ids_set - tpl_set):
        flags.append({'ID': loc_id, 'Flag': 'MISSING_TEMPLATES', 'Details': 'ID not present in location_templates.txt'})
    for loc_id in sorted(ids_set - hie_set):
        flags.append({'ID': loc_id, 'Flag': 'MISSING_DEFINITIONS', 'Details': 'ID not present in definitions.txt hierarchy'})

    sea_river = df_master[(df_master['Type']=='sea') & (df_master['Has River'].fillna('')=='Yes')]
    for _, r in sea_river.iterrows():
        flags.append({'ID': r['ID'], 'Flag': 'RIVER_ON_SEA', 'Details': 'Has River=Yes while Type=sea'})

    df_flags = pd.DataFrame(flags)
    if df_flags.empty:
        return pd.DataFrame(columns=['ID','Flag','Details'])
    return df_flags.drop_duplicates(subset=['ID','Flag','Details'], keep='first')

# =============================================================================
# 12) Diff summary
# =============================================================================

def build_diff_summary(prev_csv_path: str, curr_df: pd.DataFrame) -> dict:
    try:
        prev_df = pd.read_csv(prev_csv_path, encoding='utf-8-sig')
    except Exception:
        prev_df = pd.read_csv(prev_csv_path)

    def yes_count(df, col):
        return int((df[col].fillna('')=='Yes').sum()) if col in df.columns else None

    summary = {
        'prev_csv': prev_csv_path,
        'curr_csv': OUTPUT_CSV,
        'prev_rows': int(prev_df.shape[0]),
        'curr_rows': int(curr_df.shape[0]),
        'row_delta': int(curr_df.shape[0] - prev_df.shape[0]),
        'counts': {},
    }

    for col in ['Has Coast','Has River']:
        summary['counts'][col] = {'prev': yes_count(prev_df, col), 'curr': yes_count(curr_df, col)}
        if summary['counts'][col]['prev'] is not None:
            summary['counts'][col]['delta'] = summary['counts'][col]['curr'] - summary['counts'][col]['prev']

    if 'Type' in prev_df.columns and 'Type' in curr_df.columns:
        summary['type_counts'] = {
            'prev': prev_df['Type'].value_counts(dropna=False).to_dict(),
            'curr': curr_df['Type'].value_counts(dropna=False).to_dict(),
        }

    return summary

# =============================================================================
# 13) Main
# =============================================================================

def main():
    t0 = time.time()
    step_times = {}

    def mark(name, t_start):
        step_times[name] = round(time.time() - t_start, 3)

    print(f"[INFO] {TOOL_NAME} {TOOL_VERSION}")

    # Enable diagnostic logging when requested.
    if _diagnostic_enabled(sys.argv):
        _enable_diagnostic_logger()


    required = [FILE_00_DEFAULT, FILE_DEFINITIONS, FILE_TEMPLATES, FILE_PORTS]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print('[ERROR] Missing required files:')
        for m in missing:
            print('  -', m)
        sys.exit(1)

    loc_img_path = pick_existing(LOCATIONS_IMG_CANDIDATES, 'locations image')
    riv_img_path = pick_existing(RIVERS_IMG_CANDIDATES, 'rivers image')

    # --- Load sources
    t = time.time()
    hex_map = load_00_default(FILE_00_DEFAULT)
    templates = load_location_templates(FILE_TEMPLATES)
    hierarchy = load_definitions_hierarchy(FILE_DEFINITIONS)
    ports = load_ports(FILE_PORTS)
    mark('load_text_sources', t)

    if not hex_map:
        print('[ERROR] 00_default.txt parsed empty.')
        sys.exit(1)

    # --- Palette maps
    t = time.time()
    id_to_rgb = {}
    rgb_to_id = {}
    for loc_id, hx in hex_map.items():
        loc_id = normalize_id(loc_id)
        if not loc_id:
            continue
        v = hx.lstrip('#')
        rgb = (int(v[0:2],16), int(v[2:4],16), int(v[4:6],16))
        id_to_rgb[loc_id] = rgb
        rgb_to_id[rgb] = loc_id
    palette_rgb_set = set(rgb_to_id.keys())
    mark('build_palette_maps', t)

    # --- Rivers (two-track, cache)
    t = time.time()
    rivers_sig = {
        'cache_format': CACHE_FORMAT_RIVERS,
        'tool': {'name': TOOL_NAME, 'version': TOOL_VERSION},
        'schema': SCHEMA_VERSION,
        'locations_image_sha256': safe_stat_and_hash(loc_img_path).get('sha256'),
        'rivers_image_sha256': safe_stat_and_hash(riv_img_path).get('sha256'),
        '00_default_sha256': safe_stat_and_hash(FILE_00_DEFAULT).get('sha256'),
        'SEA_TOL': SEA_TOL,
        'LAND_TOL': LAND_TOL,
        'COAST_GUARD': COAST_GUARD,
        'COAST_GUARD_MODE': COAST_GUARD_MODE,
        'MIN_PIXELS': MIN_PIXELS,
        'MIN_RATIO': MIN_RATIO,
    }

    river_cache = cache_load('rivers', rivers_sig)
    river_cache_used = False

    if river_cache:
        river_cache_used = True
        raw_counts = defaultdict(int, {k: int(v) for k, v in river_cache.get('raw_counts', {}).items()})
        guarded_counts = defaultdict(int, {k: int(v) for k, v in river_cache.get('guarded_counts', {}).items()})
        area_map = {tuple(map(int, k.split(','))): int(v) for k, v in river_cache.get('area_map', {}).items()} if isinstance(river_cache.get('area_map'), dict) else {}
        rivers_meta = {
            'sea_rgb': tuple(river_cache.get('sea_rgb', (0,0,0))),
            'exact_mode': bool(river_cache.get('exact_mode', False)),
            'used_scipy': bool(river_cache.get('used_scipy', False)),
            'used_fallback': bool(river_cache.get('used_fallback', False)),
            'sea_like': None,
            'land_like': None,
            'raw_river_mask': None,
            'guarded_river_mask': None,
        }
    else:
        art = detect_river_two_track(loc_img_path, riv_img_path, rgb_to_id)
        raw_counts = art['raw_counts']
        guarded_counts = art['guarded_counts']
        area_map = art['area_map']
        rivers_meta = art

        area_map_ser = {f"{k[0]},{k[1]},{k[2]}": int(v) for k, v in area_map.items()}
        cache_save('rivers', rivers_sig, {
            'sea_rgb': list(art['sea_rgb']),
            'exact_mode': art['exact_mode'],
            'used_scipy': art['used_scipy'],
            'used_fallback': art['used_fallback'],
            'raw_counts': {k: int(v) for k, v in raw_counts.items()},
            'guarded_counts': {k: int(v) for k, v in guarded_counts.items()},
            'area_map': area_map_ser,
        })

    river_locs = finalize_river_locs_from_counts(guarded_counts, id_to_rgb, area_map)

    mark('detect_rivers', t)



    # --- Build master
    t = time.time()
    all_ids = set(hex_map.keys()) | set(templates.keys()) | set(hierarchy.keys())
    all_ids = {normalize_id(x) for x in all_ids}
    all_ids = {x for x in all_ids if x and x not in ('rgb','hsv')}

    rows = []
    for loc_id in sorted(all_ids):
        h = hierarchy.get(loc_id, {})
        td = templates.get(loc_id, {})
        topo = td.get('Topography','')
        typ, typ_reason = classify_type(loc_id, topo, h)

        has_river = 'Yes' if (typ == 'land' and loc_id in river_locs) else ''

        rows.append({
            'ID': loc_id,
            'Region': h.get('RegionID',''),
            'Area': h.get('AreaID',''),
            'Province': h.get('ProvinceID',''),
            'Type': typ,
            'Type Reason': typ_reason,
            'Topography': topo,
            'Vegetation': td.get('Vegetation',''),
            'Climate': td.get('Climate',''),
            'Raw Material': td.get('Raw Material',''),
            'Harbor': td.get('Harbor',''),
            'Has Coast': 'Yes' if loc_id in ports else '',
            'Has River': has_river,
        })

    df = pd.DataFrame(rows)
    df['ID'] = df['ID'].map(normalize_id)
    df = df[df['ID'] != '']
    df = df.drop_duplicates(subset=['ID'], keep='first')

    cols = ['ID','Region','Area','Province','Type','Type Reason','Topography','Vegetation','Climate','Raw Material','Harbor','Has Coast','Has River']
    df = df.reindex(columns=cols)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    mark('build_and_write_csv', t)


    # --- Diagnostic: river overlap on non-land (sea/lake/wasteland), two-track
    t = time.time()
    df_idx = df.set_index('ID', drop=False)
    diag_rows = []
    candidate_ids = set(raw_counts.keys()) | set(guarded_counts.keys())

    for loc_id in candidate_ids:
        loc_id_n = normalize_id(loc_id)
        if not loc_id_n or loc_id_n not in df_idx.index:
            continue
        typ = str(df_idx.loc[loc_id_n, 'Type'])
        if typ == 'land':
            continue

        rgb = id_to_rgb.get(loc_id_n)
        area_pix = int(area_map.get(rgb, 0)) if rgb is not None else 0

        raw_p = int(raw_counts.get(loc_id_n, 0))
        grd_p = int(guarded_counts.get(loc_id_n, 0))

        raw_ratio = (raw_p / area_pix) if area_pix > 0 else None
        grd_ratio = (grd_p / area_pix) if area_pix > 0 else None

        raw_meets = (raw_p >= MIN_PIXELS) or (raw_ratio is not None and raw_ratio >= MIN_RATIO)
        grd_meets = (grd_p >= MIN_PIXELS) or (grd_ratio is not None and grd_ratio >= MIN_RATIO)

        removed = raw_p - grd_p
        removed_pct = (removed / raw_p) if raw_p > 0 else None

        diag_rows.append({
            'ID': loc_id_n,
            'Type': typ,
            'Region': df_idx.loc[loc_id_n, 'Region'],
            'Area': df_idx.loc[loc_id_n, 'Area'],
            'Province': df_idx.loc[loc_id_n, 'Province'],
            'Topography': df_idx.loc[loc_id_n, 'Topography'],
            'Raw River Pixels': raw_p,
            'Guarded River Pixels': grd_p,
            'Removed by Guard': removed,
            'Removed %': removed_pct,
            'Area Pixels': area_pix,
            'Raw River Ratio': raw_ratio,
            'Guarded River Ratio': grd_ratio,
            'Raw Meets Threshold': 'Yes' if raw_meets else '',
            'Guarded Meets Threshold': 'Yes' if grd_meets else '',
        })

    df_diag = pd.DataFrame(diag_rows)
    if df_diag.empty:
        df_diag = pd.DataFrame(columns=[
            'ID','Type','Region','Area','Province','Topography',
            'Raw River Pixels','Guarded River Pixels','Removed by Guard','Removed %',
            'Area Pixels','Raw River Ratio','Guarded River Ratio',
            'Raw Meets Threshold','Guarded Meets Threshold'
        ])
    else:
        df_diag['Guarded Meets Threshold'] = df_diag['Guarded Meets Threshold'].fillna('')
        df_diag = df_diag.sort_values(by=['Guarded Meets Threshold','Raw River Pixels'], ascending=[False, False])

    df_diag.to_csv(OUTPUT_RIVER_OVERLAP_WATER_CSV, index=False, encoding='utf-8-sig')
    mark('river_overlap_water_csv', t)

    # --- QC
    t = time.time()
    df_qc = build_qc_flags(set(df['ID'].tolist()), templates, hierarchy, df)
    df_qc.to_csv(OUTPUT_QC_FLAGS, index=False, encoding='utf-8-sig')
    mark('qc_flags', t)

    # --- Debug overlays
    t = time.time()
    try:
        if rivers_meta.get('sea_like') is not None:
            write_rivers_debug_overlay(rivers_meta, DEBUG_RIVERS_OVERLAY, DEBUG_DOWNSAMPLE)
    except Exception as e:
        print('[WARN] debug overlay failed:', e)
    mark('debug_overlays', t)

    # --- Diff summary
    t = time.time()
    prev_csv = os.environ.get("EU5_PREV_CSV")
    diff_summary = None
    if prev_csv and os.path.exists(prev_csv):
        try:
            diff_summary = build_diff_summary(prev_csv, df)
            with open(OUTPUT_DIFF_SUMMARY, 'w', encoding='utf-8') as f:
                json.dump(diff_summary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print('[WARN] diff summary failed:', e)
    mark('diff_summary', t)

    # --- Run report
    t = time.time()

    raw_hits = int((df_diag['Raw Meets Threshold'].fillna('')=='Yes').sum()) if not df_diag.empty else 0
    grd_hits = int((df_diag['Guarded Meets Threshold'].fillna('')=='Yes').sum()) if not df_diag.empty else 0

    report = {
        'tool': {'name': TOOL_NAME, 'version': TOOL_VERSION},
        'schema': {'version': SCHEMA_VERSION, 'csv_columns': list(df.columns)},
        'generated_utc': datetime.datetime.now(datetime.UTC).isoformat().replace('+00:00', 'Z'),
        'python': {'version': sys.version, 'executable': sys.executable},
        'platform': {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine()},
        'paths': {
            'EU5_ROOT': EU5_ROOT,
            'MAP_DATA_DIR': MAP_DATA_DIR,
            'NAMED_LOCATIONS_DIR': NAMED_LOCATIONS_DIR,
            '00_default': FILE_00_DEFAULT,
            'definitions': FILE_DEFINITIONS,
            'location_templates': FILE_TEMPLATES,
            'ports': FILE_PORTS,
            'locations_image': loc_img_path,
            'rivers_image': riv_img_path,
        },
        'input_files': {
            '00_default': safe_stat_and_hash(FILE_00_DEFAULT),
            'definitions': safe_stat_and_hash(FILE_DEFINITIONS),
            'location_templates': safe_stat_and_hash(FILE_TEMPLATES),
            'ports': safe_stat_and_hash(FILE_PORTS),
            'locations_image': safe_stat_and_hash(loc_img_path),
            'rivers_image': safe_stat_and_hash(riv_img_path),
        },
        'parameters': {
            'SEA_TOL': SEA_TOL,
            'LAND_TOL': LAND_TOL,
            'COAST_GUARD': COAST_GUARD,
            'COAST_GUARD_MODE': COAST_GUARD_MODE,
            'MIN_PIXELS': MIN_PIXELS,
            'MIN_RATIO': MIN_RATIO,
            'DEBUG_DOWNSAMPLE': DEBUG_DOWNSAMPLE,
            'CACHE_ENABLE': CACHE_ENABLE,
        },
        'computed': {
            'rows': int(df.shape[0]),
            'blank_id_rows': int((df['ID'].astype(str).str.strip().str.lower().isin(['','nan','none','null'])).sum()),
            'type_counts': df['Type'].value_counts(dropna=False).to_dict(),
            'has_coast_yes': int((df['Has Coast'].fillna('')=='Yes').sum()),
            'has_river_yes': int((df['Has River'].fillna('')=='Yes').sum()),
            'rivers': {
                'estimated_sea_rgb': list(rivers_meta.get('sea_rgb')) if rivers_meta.get('sea_rgb') else None,
                'exact_mode': bool(rivers_meta.get('exact_mode', False)),
                'used_scipy': bool(rivers_meta.get('used_scipy', False)),
                'used_fallback': bool(rivers_meta.get('used_fallback', False)),
                'cache_used': bool(river_cache_used),
            },
            'river_overlap_water': {
                'rows': int(df_diag.shape[0]),
                'counts_by_type': df_diag['Type'].value_counts().to_dict() if not df_diag.empty else {},
                'raw_threshold_hits': raw_hits,
                'guarded_threshold_hits': grd_hits,
            },
            'qc': {
                'qc_flag_rows': int(df_qc.shape[0]),
                'flags_by_type': df_qc['Flag'].value_counts().to_dict() if not df_qc.empty else {},
            },
        },
        'diff_summary': diff_summary,
        'timing_seconds': {**step_times, 'total': round(time.time() - t0, 3)},
        'outputs': {
            'csv': OUTPUT_CSV,
            'run_report': OUTPUT_RUN_REPORT,
            'qc_flags': OUTPUT_QC_FLAGS,
            'diff_summary': OUTPUT_DIFF_SUMMARY if diff_summary else None,
            'river_overlap_water_csv': OUTPUT_RIVER_OVERLAP_WATER_CSV,
            'debug_rivers_overlay': DEBUG_RIVERS_OVERLAY,
        },
    }

    report['version'] = TOOL_VERSION

    with open(OUTPUT_RUN_REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    mark('run_report', t)

    print('[DONE] wrote:', OUTPUT_CSV)
    print('[DONE] wrote:', OUTPUT_RIVER_OVERLAP_WATER_CSV)
    print('[DONE] wrote:', OUTPUT_QC_FLAGS)
    print('[DONE] wrote:', OUTPUT_RUN_REPORT)
    if diff_summary:
        print('[DONE] wrote:', OUTPUT_DIFF_SUMMARY)
    print('[DONE] wrote:', DEBUG_RIVERS_OVERLAY)
    print(f"[DONE] rows={len(df)} time={report['timing_seconds']['total']:.1f}s")


if __name__ == '__main__':
    main()