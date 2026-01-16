# -*- coding: utf-8 -*-
"""EU5 Locations Master Builder v1.0

EU5のマップデータ（named_locations / definitions / location_templates / ports.csv / locations画像 / rivers画像）を
突き合わせて、ロケーション一覧CSV（MASTER）と診断用CSVを生成します。

目的
- 後段（分析・レビュー・都市化スコア等）が、ゲーム内の地理判定に近い入力列を利用できるようにする。
- 特に「沿岸」は ports.csv 由来の判定を正とし、MASTER CSV に `Has Coast` として出力する。

入力（EU5インストール配下）
- game/in_game/map_data/named_locations/00_default.txt
  - location_id ↔ 色（RGB/HEX）の対応
- game/in_game/map_data/definitions.txt
  - Region/Area/Province の階層
- game/in_game/map_data/location_templates.txt
  - Topography/Vegetation/Climate/Raw Material/Harbor 等
- game/in_game/map_data/ports.csv
  - 沿岸（Has Coast）判定の根拠
- game/in_game/map_data/locations.(png|tga)
  - 位置（色）から隣接関係を計算
- game/in_game/map_data/rivers.(png|tga)
  - 河川インクを検出して Has River を決定

出力（カレントディレクトリ）
- eu5_locations_raw_1.0.10.txt
  - MASTER 一覧
  - 重要列：
    - Has Coast : ports.csv（LandProvince）由来。沿岸扱い（港/沿岸条件）に対応する想定。
    - Harbor : natural_harbor_suitability 等（テンプレ由来、文字列）。後段で数値化推奨。
    - Has River : rivers 画像の河川インク検出（陸地のみ Yes）。
    - Is Adjacent To Lake : locations 画像の隣接判定で湖に接する陸地（Yes）。

- eu5_locations_master_river_overlap_water_v1_0.csv
  - 診断用（two-track）
  - 非陸地（sea/lake/wasteland）上に河川インクが重なった量を Raw/Guarded で出力

- eu5_locations_master_qc_flags_v1_0.csv
  - QC用（テンプレ欠落など）

- eu5_locations_master_run_report_v1_0.json
  - 入力ファイルのハッシュ、件数、処理時間などの実行レポート

判定ポリシー
- 沿岸（Has Coast）は ports.csv を唯一の根拠とする。
- 説明用の地形ラベル（例: Coastal Land / Estuary Land など）は出力しない。
  - ゲーム内判定と一致しない可能性があり、後段スコアリングでノイズになるため。

Dependencies
- Required: pillow, numpy, pandas
- Optional: scipy（COAST_GUARD_MODE='edt' または 'auto' の場合に使用）
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
import logging

Image.MAX_IMAGE_PIXELS = None

# =============================================================================
# 0) Fixed paths (NO CLI args)
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
    os.path.join(MAP_DATA_DIR, "locations.tga"),
]
RIVERS_IMG_CANDIDATES = [
    os.path.join(MAP_DATA_DIR, "rivers.png"),
    os.path.join(MAP_DATA_DIR, "rivers.tga"),
]

# =============================================================================
# 1) Versioning + outputs
# =============================================================================

TOOL_NAME = "EU5 Locations Master Builder"
TOOL_VERSION = "v1.0"     # stays v1.0 until final release is confirmed
SCHEMA_VERSION = "v1.0"   # bump only when MASTER CSV schema changes

OUT_TAG = "v1_0"          # filesystem-friendly tag
OUT_PREFIX = "eu5_locations_master"

OUTPUT_CSV          = os.path.join(os.getcwd(), f"eu5_locations_raw_1.0.10.txt")
OUTPUT_RUN_REPORT   = os.path.join(os.getcwd(), f"{OUT_PREFIX}_run_report_{OUT_TAG}.json")
OUTPUT_QC_FLAGS     = os.path.join(os.getcwd(), f"{OUT_PREFIX}_qc_flags_{OUT_TAG}.csv")
OUTPUT_DIFF_SUMMARY = os.path.join(os.getcwd(), f"{OUT_PREFIX}_diff_summary_{OUT_TAG}.json")

# Diagnostic: non-land tiles with river ink overlap (two-track)
OUTPUT_RIVER_OVERLAP_WATER_CSV = os.path.join(os.getcwd(), f"{OUT_PREFIX}_river_overlap_water_{OUT_TAG}.csv")

# Features: estuary/coastal flags for all Types

DEBUG_RIVERS_OVERLAY = os.path.join(os.getcwd(), f"debug_rivers_overlay_{OUT_TAG}.png")
DEBUG_LAKE_OVERLAY   = os.path.join(os.getcwd(), f"debug_lake_adjacency_overlay_{OUT_TAG}.png")

PREV_CSV_CANDIDATES = [
    os.path.join(os.getcwd(), "eu5_locations_master_v2_0.csv"),
    os.path.join(os.getcwd(), "eu5_locations_master_v1_2.csv"),
    os.path.join(os.getcwd(), "eu5_locations_master_v1_1.csv"),
]

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

CACHE_ENABLE = False
CACHE_DIR = os.path.join(os.getcwd(), f"eu5_locations_cache_{OUT_TAG}")

# Cache schema tag for rivers payload (two-track)
CACHE_FORMAT_RIVERS = "rivers_cache_v2_two_track"

# Cache schema tag for adjacency payloads (coastal)
CACHE_FORMAT_ADJ = "adjacency_cache_v1"

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


def safe_stat_and_hash(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {'exists': False}
    st = os.stat(path)
    return {
        'exists': True,
        'bytes': int(st.st_size),
        'mtime_utc': datetime.datetime.utcfromtimestamp(st.st_mtime).isoformat() + 'Z',
        'sha256': file_sha256(path),
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
# 9) Adjacency (coastal / lake)
# =============================================================================

def build_adjacency_edges(loc_img_path: str, palette_rgb_set: set) -> tuple:
    """Return unique undirected edges between palette colors (packed uint32) using 4-neighborhood."""
    img = Image.open(loc_img_path).convert('RGB')
    A = np.array(img, dtype=np.uint8)

    Ai = (A[:,:,0].astype(np.uint32)<<16) | (A[:,:,1].astype(np.uint32)<<8) | A[:,:,2].astype(np.uint32)
    palette_ints = np.fromiter(((r<<16)|(g<<8)|b for (r,g,b) in palette_rgb_set), dtype=np.uint32)
    valid = np.isin(Ai, palette_ints)

    left = Ai[:,:-1]; right = Ai[:,1:]
    v_h = valid[:,:-1] & valid[:,1:] & (left != right)
    hp = np.stack([left[v_h], right[v_h]], axis=1) if np.any(v_h) else np.empty((0,2), dtype=np.uint32)

    up = Ai[:-1,:]; down = Ai[1:,:]
    v_v = valid[:-1,:] & valid[1:,:] & (up != down)
    vp = np.stack([up[v_v], down[v_v]], axis=1) if np.any(v_v) else np.empty((0,2), dtype=np.uint32)

    allp = np.vstack([hp, vp]) if (hp.size or vp.size) else np.empty((0,2), dtype=np.uint32)
    if allp.size == 0:
        return np.empty((0,2), dtype=np.uint32), {'valid_pixel_ratio': float(valid.mean()), 'unique_edges': 0}

    mn = np.minimum(allp[:,0], allp[:,1])
    mx = np.maximum(allp[:,0], allp[:,1])
    edges = np.unique(np.stack([mn,mx], axis=1), axis=0)

    return edges, {'valid_pixel_ratio': float(valid.mean()), 'unique_edges': int(edges.shape[0])}


def get_adjacency_edges_cached(loc_img_path: str, palette_rgb_set: set, cache_load_fn, cache_save_fn):
    """Load or build location adjacency edges from the locations image.

    NOTE: Refactor only. Logic must be identical to v1.0.
    """
    adj_sig = {
        'cache_format': CACHE_FORMAT_ADJ,
        'tool': {'name': TOOL_NAME, 'version': TOOL_VERSION},
        'schema': SCHEMA_VERSION,
        'locations_image_sha256': safe_stat_and_hash(loc_img_path).get('sha256'),
        '00_default_sha256': safe_stat_and_hash(FILE_00_DEFAULT).get('sha256'),
    }

    adj_cache = cache_load_fn('adjacency_edges', adj_sig)
    adj_cache_used = False

    if adj_cache:
        adj_cache_used = True
        edges_list = adj_cache.get('edges_u32', [])
        edges_u32 = np.array(edges_list, dtype=np.uint32) if edges_list else np.empty((0,2), dtype=np.uint32)
        edge_stats = adj_cache.get('edge_stats', {})
    else:
        edges_u32, edge_stats = build_adjacency_edges(loc_img_path, palette_rgb_set)
        cache_save_fn('adjacency_edges', adj_sig, {
            'edges_u32': edges_u32.astype(int).tolist(),
            'edge_stats': edge_stats,
        })

    return edges_u32, edge_stats, adj_cache_used


def build_lake_adjacency_from_edges(edges_u32: np.ndarray, rgb_to_id: dict, lake_ids_set: set, sea_ids_set: set):
    lake_rgbs = [rgb for rgb, loc_id in rgb_to_id.items() if loc_id in lake_ids_set]
    lake_ints = {((r<<16)|(g<<8)|b) for (r,g,b) in lake_rgbs}

    sea_rgbs = [rgb for rgb, loc_id in rgb_to_id.items() if loc_id in sea_ids_set]
    sea_ints = {((r<<16)|(g<<8)|b) for (r,g,b) in sea_rgbs}

    adjacent = set()
    for u,v in edges_u32.tolist():
        u=int(u); v=int(v)
        u_is_lake = u in lake_ints
        v_is_lake = v in lake_ints
        if u_is_lake == v_is_lake:
            continue
        other = v if u_is_lake else u
        if other in lake_ints or other in sea_ints:
            continue
        rgb_other = ((other>>16)&255, (other>>8)&255, other&255)
        other_id = rgb_to_id.get(rgb_other)
        if other_id:
            adjacent.add(other_id)

    return adjacent


def detect_adjacent_to_lake_ids_image(
    edges_u32: np.ndarray,
    rgb_to_id: dict,
    lake_ids_set: set,
    sea_ids_set: set,
) -> set:
    """Authoritative lake-adjacency detection based on the locations image.

    This function must remain the single source of truth for lake adjacency.
    Future auxiliary sources (e.g., text triggers) may be consulted elsewhere,
    but MUST NOT override this result.

    NOTE: Refactor-only boundary. No logic change is permitted here.
    """
    return build_lake_adjacency_from_edges(
        edges_u32, rgb_to_id, lake_ids_set, sea_ids_set
    )


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


def resolve_adjacent_to_lake_ids(
    edges_u32: np.ndarray,
    rgb_to_id: dict,
    lake_ids_set: set,
    sea_ids_set: set,
    *,
    trigger_hints=None,
    logger=None,
) -> set:
    """Resolve lake adjacency with image as the final authority.

    - Current behavior: image-only (identical to v1.0 / current accuracy-first).
    - Extension point (future): optional trigger hints / logging / multi-source validation.
      Image result must always be the final decision.

    NOTE: Refactor-only boundary. No logic change is permitted here.
    """
    adjacent = detect_adjacent_to_lake_ids_image(
        edges_u32, rgb_to_id, lake_ids_set, sea_ids_set
    )

    # Optional diagnostic logging (no effect on results)
    # If --diagnostic is enabled, allow global file-only logger when logger is not explicitly provided.
    if logger is None and _DIAGNOSTIC_LOGGER is not None:
        logger = _DIAGNOSTIC_LOGGER
    # Optional diagnostic logging (no effect on results)
    if logger is not None:
        try:
            logger.info(
                "lake_adjacency(image): %d locations adjacent to lakes",
                len(adjacent),
            )

            # Diagnostic-only: compare with trigger-derived hints (must not override image result)
            hint_ids = None
            if trigger_hints is not None:
                try:
                    hint_ids = trigger_hints if isinstance(trigger_hints, set) else set(trigger_hints)
                except Exception:
                    hint_ids = None

            if hint_ids is not None:
                logger.info(
                    "lake_adjacency(hints): %d hinted locations",
                    len(hint_ids),
                )
                overlap = len(adjacent & hint_ids)
                image_only = len(adjacent - hint_ids)
                hint_only = len(hint_ids - adjacent)
                logger.info(
                    "lake_adjacency(compare): overlap=%d image_only=%d hint_only=%d",
                    overlap, image_only, hint_only,
                )

        except Exception:
            # Logging must never affect execution
            pass

    return adjacent



def build_coastal_flags_from_edges(edges_u32: np.ndarray, rgb_to_id: dict, land_ids: set, sea_ids: set):
    """Return (coastal_land_set, coastal_sea_set)."""
    # Build int sets
    land_rgbs = [rgb for rgb, loc_id in rgb_to_id.items() if loc_id in land_ids]
    sea_rgbs  = [rgb for rgb, loc_id in rgb_to_id.items() if loc_id in sea_ids]
    land_ints = {((r<<16)|(g<<8)|b) for (r,g,b) in land_rgbs}
    sea_ints  = {((r<<16)|(g<<8)|b) for (r,g,b) in sea_rgbs}

    coastal_land = set()
    coastal_sea = set()

    for u,v in edges_u32.tolist():
        u=int(u); v=int(v)
        u_land = u in land_ints
        v_land = v in land_ints
        u_sea  = u in sea_ints
        v_sea  = v in sea_ints

        # land-sea adjacency
        if u_land and v_sea:
            land_rgb = ((u>>16)&255, (u>>8)&255, u&255)
            sea_rgb  = ((v>>16)&255, (v>>8)&255, v&255)
            land_id = rgb_to_id.get(land_rgb)
            sea_id  = rgb_to_id.get(sea_rgb)
            if land_id:
                coastal_land.add(land_id)
            if sea_id:
                coastal_sea.add(sea_id)
        elif v_land and u_sea:
            land_rgb = ((v>>16)&255, (v>>8)&255, v&255)
            sea_rgb  = ((u>>16)&255, (u>>8)&255, u&255)
            land_id = rgb_to_id.get(land_rgb)
            sea_id  = rgb_to_id.get(sea_rgb)
            if land_id:
                coastal_land.add(land_id)
            if sea_id:
                coastal_sea.add(sea_id)

    return coastal_land, coastal_sea

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


def write_lake_debug_overlay(loc_img_path: str, adjacent_ids: set, id_to_rgb: dict,
                            out_path: str, downsample: int):
    base = np.array(Image.open(loc_img_path).convert('RGB'), dtype=np.uint8)
    base = base[::downsample, ::downsample] if downsample > 1 else base

    adj_rgbs = [id_to_rgb.get(i) for i in adjacent_ids]
    adj_rgbs = [x for x in adj_rgbs if x is not None]
    adj_set = set((int(r),int(g),int(b)) for (r,g,b) in adj_rgbs)

    packed = (base[:,:,0].astype(np.uint32)<<16) | (base[:,:,1].astype(np.uint32)<<8) | base[:,:,2].astype(np.uint32)
    adj_ints = np.fromiter(((r<<16)|(g<<8)|b for (r,g,b) in adj_set), dtype=np.uint32)

    mask = np.isin(packed, adj_ints)
    out = base.copy()
    out[mask] = (60,255,60)
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

    for col in ['Has Coast','Has River','Is Adjacent To Lake']:
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

    # Diagnostic flag: opt-in only; must not affect results/output when absent.
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

    # --- Type sets
    t = time.time()
    lake_ids_set = {loc_id for loc_id, td in templates.items() if (td.get('Topography','') or '').lower() == 'lakes'}

    sea_ids_set = set()
    for loc_id in hex_map.keys():
        topo = (templates.get(loc_id, {}).get('Topography','') or '').lower()
        if topo in SEA_TOPO:
            sea_ids_set.add(loc_id)
        elif is_explicit_sea_id(loc_id, hierarchy.get(loc_id, {})):
            sea_ids_set.add(loc_id)

    # land IDs will be resolved after we build the master df (safer if any missing template)
    mark('build_water_id_sets', t)

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

    # --- Adjacency edges (cache)
    t = time.time()
    edges_u32, edge_stats, adj_cache_used = get_adjacency_edges_cached(
        loc_img_path, palette_rgb_set, cache_load, cache_save
    )

    mark('adjacency_edges', t)

    # --- Lake adjacency from edges
    t = time.time()
    adjacent_to_lake_ids = resolve_adjacent_to_lake_ids(edges_u32, rgb_to_id, lake_ids_set, sea_ids_set)
    lake_stats = {**edge_stats}
    mark('lake_adjacency', t)


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
            'Is Adjacent To Lake': 'Yes' if (loc_id in adjacent_to_lake_ids and typ == 'land') else '',
        })

    df = pd.DataFrame(rows)
    df['ID'] = df['ID'].map(normalize_id)
    df = df[df['ID'] != '']
    df = df.drop_duplicates(subset=['ID'], keep='first')

    cols = ['ID','Region','Area','Province','Type','Type Reason','Topography','Vegetation','Climate','Raw Material','Harbor','Has Coast','Has River','Is Adjacent To Lake']
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
        write_lake_debug_overlay(loc_img_path, adjacent_to_lake_ids, id_to_rgb, DEBUG_LAKE_OVERLAY, DEBUG_DOWNSAMPLE)
    except Exception as e:
        print('[WARN] debug overlay failed:', e)
    mark('debug_overlays', t)

    # --- Diff summary
    t = time.time()
    prev_csv = next((p for p in PREV_CSV_CANDIDATES if os.path.exists(p)), None)
    diff_summary = None
    if prev_csv:
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
        'generated_utc': datetime.datetime.utcnow().isoformat() + 'Z',
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
            'adjacent_to_lake_yes': int((df['Is Adjacent To Lake'].fillna('')=='Yes').sum()),
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
            'coastal': {
                'adjacency_edges': edge_stats,
                'adj_cache_used': bool(adj_cache_used),
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
            'debug_lake_overlay': DEBUG_LAKE_OVERLAY,
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
    print('[DONE] wrote:', DEBUG_LAKE_OVERLAY)
    print(f"[DONE] rows={len(df)} time={report['timing_seconds']['total']:.1f}s")


if __name__ == '__main__':
    main()
