# EU5 Locations Master Builder

EU5 Locations Master Builder generates a master CSV of Europa Universalis V (EU5) locations by joining multiple map data sources from an EU5 installation.

This repository is intended to produce a high-fidelity, analysis-ready location table that stays aligned with in-game geography rules (especially coastline/coastal eligibility).

## What it produces

When you run the builder, it writes these files to the current working directory:

- `eu5_locations_master_raw.csv`
  - The master location table.
- `eu5_locations_master_river_overlap_water_v1_0.csv`
  - Diagnostics: river ink overlap over non-land tiles (useful for validating the river mask).
- `eu5_locations_master_qc_flags_v1_0.csv`
  - Quality-control flags (missing templates, unexpected mappings, etc.).
- `eu5_locations_master_run_report_v1_0.json`
  - Run metadata (inputs seen, cache status, timings).
- `debug_rivers_overlay_v1_0.png`
  - Debug visualization of the river mask.
- `debug_lake_adjacency_overlay_v1_0.png`
  - Debug visualization of lake adjacency detection.

## Requirements

- Windows 10/11
- Python 3.11+ (recommended)
- EU5 installed locally (Steam default path is supported out of the box)

Python dependencies are listed in `requirements.txt`.

## Quick start (recommended)

1. Clone this repository.
2. Ensure EU5 is installed.
3. Run the wrapper script:

```powershell
pwsh -NoProfile -File .\run.ps1
```

The first run may take longer because it builds persistent cache files. Subsequent runs are typically much faster.

## EU5 installation path

By default, the builder expects EU5 at:

- `C:\Program Files (x86)\Steam\steamapps\common\Europa Universalis V`

If your EU5 install is elsewhere, set `EU5_ROOT` before running:

```powershell
$env:EU5_ROOT = "D:\\SteamLibrary\\steamapps\\common\\Europa Universalis V"
pwsh -NoProfile -File .\run.ps1
```

## Persistent cache (enabled by default)

The builder uses a persistent cache to avoid recomputing expensive intermediate results.

- Default cache location: `.tmp\eu5_locations_cache_v1_0\`
- You can override the cache directory:

```powershell
$env:EU5_CACHE_DIR = "C:\\work\\eu5_cache"
pwsh -NoProfile -File .\run.ps1
```

To disable caching for a single run:

```powershell
$env:EU5_NO_CACHE = "1"
pwsh -NoProfile -File .\run.ps1
Remove-Item Env:EU5_NO_CACHE -ErrorAction SilentlyContinue
```

### Cache invalidation policy

To keep cached runs fast, the default invalidation policy hashes only small input files (up to 10 MB) and uses size/mtime for larger files.

- Hash everything (stricter, slower):

```powershell
$env:EU5_HASH_INPUTS = "1"
pwsh -NoProfile -File .\run.ps1
```

- Change the maximum hashed size (bytes):

```powershell
$env:EU5_HASH_MAX_BYTES = "20971520"  # 20 MB
pwsh -NoProfile -File .\run.ps1
```

## Notes on key fields

- `Has Coast` is derived from `ports.csv` (LandProvince) and is treated as authoritative.
- `Has River` is derived from river ink overlap on land tiles in `rivers.png`.
- `Is Adjacent To Lake` is derived from pixel adjacency between land tiles and lake tiles in `locations.png`.

## Troubleshooting

### "[ERROR] ... not found" for EU5 files

Confirm your `EU5_ROOT` is correct and points to the EU5 installation folder.

### SciPy is not installed

SciPy is optional. If installed, it enables a faster/more accurate distance-transform mode for coastline guarding.

### Large image warnings

The script disables Pillow's image pixel limit (`Image.MAX_IMAGE_PIXELS = None`) because EU5 map images can be large.

## License

See `LICENSE`.
