# Phase 0: 国家ページ データソース調査（修正版 v2）

## データソース一覧

### 取得済み（game/in_game/ 配下）

| ソース | フィールド | 備考 |
|--------|-----------|------|
| `setup/countries/*.txt` | tag, color, color2, culture_definition, religion_definition, difficulty, description_category, is_historic | 2,328タグ。colorは`map_*`名前or`rgb{}`構文 |
| `common/cultures/*.txt` | culture → culture_groups マッピング | **多対多**: 23%のcultureが複数groupに所属（例: breton → celtic_group + french_group） |
| `common/culture_groups/00_culture_groups.txt` | グループ定義 | |

### 新規fetch必要（game/main_menu/ 配下）

| ソース | フィールド | サイズ | 備考 |
|--------|-----------|-------|------|
| `setup/start/10_countries.txt` | capital, country_rank, government.type, heir_selection, include(template名), starting_technology_level, stability, prestige, gold, tolerated_cultures, court_language, societal_values(13種) | 1.5MB | **最重要**。2,327タグ。setup/countriesとのタグ差分監査必要 |
| `setup/templates/*.txt` | 政体テンプレート（government.type, laws, privileges等のデフォルト値） | 180+ファイル | includeで継承 |
| `common/formable_countries/00_formable_countries.txt` | level, rule, required_locations_fraction, regions/areas, potential, allow, form_effect | 60KB, 139件 | tag_f形式（GBR_f等） |
| `common/country_ranks/00_default.txt` | 4ランク定義（county/duchy/kingdom/empire） | 小 | |
| `common/coat_of_arms/coat_of_arms/*.txt` | CoA定義 | 1.1MB+ | pre_scripted_countries.txt他 |
| `common/flag_definitions/00_flag_definitions.txt` | flag→CoAマッピング | 186KB | trigger付き条件分岐 |
| `gfx/coat_of_arms/` | patterns(66) + colored_emblems(3,593) + textured_emblems(476) DDS | ~500MB | バイナリ |

### 新規fetch必要（localization）

| loc ファイル | 理由 | 優先度 |
|-------------|------|--------|
| `common_used_strings` | country名の `$common_string_prefix_article$` 解決（15+タグ） | **必須** |
| `formable_countries` | formable国名・説明文 | **必須** |
| `location_names` | capital名のloc + country名の `$hamadan$` 等解決 | **必須**（部分取得済み） |
| `region_names` | formable説明の `[ShowRegionName(...)]` | あると良い |
| `cultural_and_languages` | 個別culture名のfallback（cultures_l_*.ymlが空のため） | **必須** |

### 既存取得済みlocの活用

| loc ファイル | 用途 | 追加fetch |
|-------------|------|----------|
| `government_names` | rank名（rank_county等）+ 政体名 | 不要 |
| `government_reforms` | heir_selection等の改革名 | 不要 |
| `culture_groups` | 文化圏名 | 不要 |
| `religion` | 宗教名 | 不要 |

## 特殊タグ除外

| タグ | 理由 | 判定 |
|------|------|------|
| DUMMY | プレースホルダ（culture/religion無し） | **除外** |
| PIR | 海賊（システムエンティティ） | **除外** |
| MER | 傭兵（システムエンティティ） | **除外** |
| is_historic=yes (54ユニークタグ) | イベント発生国・旧体制 | **表示するがフラグ付き** |

## 欠損ポリシー

| フィールド | 欠損時 | 理由 |
|-----------|--------|------|
| capital | omit | county級は首都なしもあり得る |
| country_rank | `"rank_county"` | ゲームデフォルト |
| government_type | templateから継承→なければomit | include → template の2段解決 |
| difficulty | omit | 大国のみ設定（推定60国前後） |
| description_category | omit | 大国のみ |
| culture_groups | culture定義から逆引き→失敗ならomit | **多対多注意** |
| formable info | omit | 大多数はformableでない |
| color | `rgb{}`後処理 or `map_*`解決→失敗なら`[128,128,128]` | parser制約あり（後述） |

## スキーマ案

### 一覧用 (index — 軽量、Astroビルド入力)

```jsonc
{
  "tag": "FRA",
  "icon": "icons/coa/FRA.png",
  "file_region": "france",         // ファイル由来（ゲーム内リージョンではない）
  "culture_groups": ["french_group"],  // string[] — 多対多対応
  "religion_definition": "catholic",
  "country_rank": "rank_kingdom",
  "difficulty": 3,                 // optional
  "description_category": "military",  // optional
  "is_formable": false
}
```

### 詳細用 (core — フル)

```jsonc
{
  // index全フィールド +
  "culture_definition": "french",
  "capital": "paris",              // optional
  "government_type": "monarchy",   // optional
  "heir_selection": "salic_law",   // optional
  "template": "catholic_monarchy", // optional
  "color": [65, 105, 225],        // optional, RGB
  "is_historic": false,
  "starting_tech": 5,             // optional
  "stability": 0,                 // optional
  "prestige": 0,                  // optional
  "gold": 0,                      // optional
  "formable_level": null,         // optional (formableのみ)
  "formable_rule": null           // optional
}
```

### 注意: `file_region` vs ゲームリージョン

現在の `region` フィールドはcountry定義ファイルのstem（france, british_isles等）であり、ゲーム内の地理的リージョンではない。一覧のグルーピングには使えるが、UIラベルには注意が必要。`file_region` にリネームして混同を防ぐ。

## 既知リスク（Phase 2 前に対処）

### R1: paradox_parser 重複キーバグ (HIGH)
- **問題**: 3個以上の同名キー（ruler_term, include等）でネストリスト化する
- **影響**: 10_countries.txtに`include`が5,115件 → データ構造破壊
- **対処**: Phase 2 開始前に `_parse_block` lines 122-128 を修正（フラットリスト化）

### R2: rgb/hsv360 色構文 (MEDIUM)
- **問題**: `color = rgb { 16 41 202 }` が `color: "rgb"` + orphaned `_values` に分離
- **対処**: extract_countries.py内で後処理（`color == "rgb" and "_values" in parent` → 再構築）

### R3: $variable$ 解決チェーン (HIGH)
- **問題**: country名に `$common_string_prefix_article$`, `$hamadan$`, `$ilkhanate$` 等の未解決参照
- **対処**: extract_countries.pyに `_resolve_var_refs()` 実装。lookup chain: country_names → location_names → government_names → common_used_strings

### R4: cultures_l_*.yml 空 (MEDIUM)
- **問題**: 個別culture名のlocがない
- **対処**: `cultural_and_languages_l_*.yml` をfallbackソースに使用

### R5: setup/countries ↔ 10_countries タグ差分 (LOW)
- **問題**: 2ソース間でタグ集合が完全一致しない可能性
- **対処**: join時にINNER JOINとし、片方のみのタグはwarning出力

### R6: CoA文化依存 (LOW)
- **問題**: 文化定義の `tags` に `*_coa_gfx` タグがあり、CoAレンダリングに影響する可能性
- **対処**: Phase 1 CoA MVPで確認

## fetch_raw.py 拡張計画

```python
# 追加定数
MAIN_MENU = rf"{EU5_BASE}\game\main_menu"

# 新規カテゴリ
@category("country_start")
def fetch_country_start():
    """10_countries.txt + templates"""
    ssh_read_file(rf"{MAIN_MENU}\setup\start\10_countries.txt",
                  RAW_DIR / "country_start" / "10_countries.txt")
    fetch_dir(rf"{MAIN_MENU}\setup\templates", RAW_DIR / "country_start" / "templates")
    fetch_loc("common_used_strings")
    fetch_loc("formable_countries")

@category("formable_countries")
def fetch_formable_countries():
    fetch_dir(rf"{GAME}\common\formable_countries", RAW_DIR / "formable_countries")

@category("country_ranks")
def fetch_country_ranks():
    fetch_dir(rf"{GAME}\common\country_ranks", RAW_DIR / "country_ranks")

@category("coat_of_arms")
def fetch_coat_of_arms():
    fetch_dir(rf"{MAIN_MENU}\common\coat_of_arms\coat_of_arms", RAW_DIR / "coat_of_arms")
    fetch_dir(rf"{MAIN_MENU}\common\flag_definitions", RAW_DIR / "flag_definitions")
    # DDS textures は fetch_coa_assets.py で別途取得
```
