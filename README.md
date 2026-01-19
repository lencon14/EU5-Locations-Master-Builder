# EU5 Locations Master Builder

EU5 Locations Master Builder generates a master CSV of Europa Universalis V (EU5) locations by joining EU5 `map_data` sources and deriving image-based features from the map textures.

## Outputs

Running the builder writes the following files to the current working directory:

- `eu5_locations_master_raw.csv` - Master location table
- `eu5_locations_master_river_overlap_water_v1_0.csv` - Diagnostics: river ink overlap over non-land tiles
- `eu5_locations_master_qc_flags_v1_0.csv` - Quality-control flags (missing templates, unexpected mappings)
- `eu5_locations_master_run_report_v1_0.json` - Run metadata (inputs seen, cache status, timings)
- `debug_rivers_overlay_v1_0.png` - Debug visualization of the river mask
- `debug_lake_adjacency_overlay_v1_0.png` - Debug visualization of lake adjacency

## Requirements

- Windows 10/11
- Python 3.11+ recommended
- EU5 installed locally

Python dependencies are listed in `requirements.txt`.

## Quick start

1. Clone this repository.
2. Install Python dependencies.
3. Run the builder.

```powershell
python -m pip install -r requirements.txt
python .\src\eu5_locations_master_builder.py
```

## EU5 installation path

By default, the builder expects EU5 at the Steam default path:

- `C:\Program Files (x86)\Steam\steamapps\common\Europa Universalis V`

If your EU5 install is elsewhere, set `EU5_ROOT` before running:

```powershell
$env:EU5_ROOT = "D:\\SteamLibrary\\steamapps\\common\\Europa Universalis V"
python .\src\eu5_locations_master_builder.py
```

## Persistent cache (enabled by default)

The builder uses a persistent cache to avoid recomputing expensive intermediate results. This typically makes the second and subsequent runs much faster.

- Default cache location: `.tmp\eu5_locations_cache_v1_0\`
- Override cache directory:

```powershell
$env:EU5_CACHE_DIR = "C:\\work\\eu5_cache"
python .\src\eu5_locations_master_builder.py
```

- Disable caching for one run:

```powershell
$env:EU5_NO_CACHE = "1"
python .\src\eu5_locations_master_builder.py
Remove-Item Env:EU5_NO_CACHE -ErrorAction SilentlyContinue
```

### Cache invalidation policy

For speed, the default invalidation policy hashes only small input files (up to 10 MB) and uses file size and modification time for larger files.

- Hash all inputs (stricter, slower):

```powershell
$env:EU5_HASH_INPUTS = "1"
python .\src\eu5_locations_master_builder.py
Remove-Item Env:EU5_HASH_INPUTS -ErrorAction SilentlyContinue
```

- Change the maximum hashed size (bytes):

```powershell
$env:EU5_HASH_MAX_BYTES = "20971520"  # 20 MB
python .\src\eu5_locations_master_builder.py
Remove-Item Env:EU5_HASH_MAX_BYTES -ErrorAction SilentlyContinue
```

## Key columns

- `Has Coast` - Derived from `ports.csv` (`LandProvince`).
- `Has River` - Derived from river ink overlap in `rivers.png` (land only).
- `Is Adjacent To Lake` - Derived from pixel adjacency between land tiles and lake tiles in `locations.png`, and only when `Has Coast` is `Yes`.

## Troubleshooting

### Missing EU5 input files

If the run prints `[ERROR] Missing required files`, confirm `EU5_ROOT` points to the EU5 installation folder.

### Cache seems stale

If you suspect the cache is out of date, delete the cache directory and run again:

```powershell
Remove-Item -Recurse -Force .\.tmp\eu5_locations_cache_v1_0
python .\src\eu5_locations_master_builder.py
```

### SciPy is not installed

SciPy is optional. If installed, it can be used for distance-transform based coastline guarding.

## License

See `LICENSE`.
