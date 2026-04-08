# EU5 Locations Master Builder

## EU5 ゲームパス

- 標準パス: `C:\Program Files (x86)\Steam\steamapps\common\Europa Universalis V`
- SSH経由: `ssh winpc` で接続（ユーザー: kazuk）

## Windows SSH 経由の注意

- `cmd.exe` の `dir` 等は日本語環境のエンコーディング問題でパスを認識できないことがある
- PowerShell (`powershell -Command "chcp 65001 | Out-Null; ..."`) を使うこと
- 日本語ローカライズファイルの取得は `[System.IO.File]::ReadAllText(path, [System.Text.Encoding]::UTF8)` で正しくUTF-8出力する

## 攻略サイト（site/）

- Astro 静的サイト。`site/` ディレクトリ
- デザインシステム: `site/DESIGN.md` に定義。UI変更時は必ず参照する
- データソース: `pipeline/output/*.json` → `site/src/data/` にコピー

## データパイプライン（pipeline/）

- `fetch_raw.py` — SSH経由で全カテゴリのゲームデータを一括取得（シェル版 fetch_raw.sh は非推奨、Python版を使う）
- 抽出スクリプト: `extract_goods.py`, `extract_buildings.py`, `extract_countries.py`, `extract_religions.py`, `extract_governments.py`
- 共通モジュール: `paradox_parser.py`（Paradoxスクリプトパーサー）, `loc_parser.py`（ローカライズパーサー）
- 全スクリプトは `cd pipeline && python3 <script>.py` で実行

### 抽出済みデータ（v1.1.10）

| カテゴリ | 件数 | スクリプト |
|---------|------|----------|
| 交易品 | 74 | extract_goods.py |
| 建物 | 439 | extract_buildings.py |
| 国家 | 2,328 | extract_countries.py |
| 宗教 | 293 | extract_religions.py |
| 政体 | 5 | extract_governments.py |
| 法律 | 191 | extract_governments.py |

### アイコンパイプライン

- `fetch_icons.py` — SSH経由でDDSアイコンを取得し、Pillow で PNG(64x64) に変換
- アイコン格納先: `pipeline/output/icons/{category}/` → `site/public/icons/` にコピー
- 各 extract スクリプトが JSON に `icon` フィールド（相対パス）を出力
- ファイル名規則: trade_goods は `icon_goods_{id}.png`、他は `{id}.png`

| カテゴリ | アイコン数 |
|---------|----------|
| trade_goods | 75 |
| buildings | 447 |
| religion | 294 |
| government_types | 8 |
| laws | 205 |
| building_categories / religious_* | 計 134 |

## データ更新フロー（ゲームバージョンアップ時）

1. `cd pipeline && python3 fetch_raw.py` で全データ再取得
2. `python3 fetch_icons.py` でアイコン再取得・変換
3. `git diff pipeline/raw/` で変更点を確認
4. 各 `extract_*.py` を実行してJSON再生成
5. `cp pipeline/output/*.json site/src/data/`
6. `cp -r pipeline/output/icons site/public/`
7. `cd site && npm run build` でサイトリビルド
8. 変更をコミット

## 課題管理

- タスク一覧: `TODO.md`
- 設計プラン: `.claude/plans/lucky-bubbling-globe.md`
