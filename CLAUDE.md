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

## データ更新フロー（ゲームバージョンアップ時）

1. `cd pipeline && python3 fetch_raw.py` で全データ再取得
2. `git diff pipeline/raw/` で変更点を確認
3. 各 `extract_*.py` を実行してJSON再生成
4. `cp pipeline/output/*.json site/src/data/`
5. `cd site && npm run build` でサイトリビルド
6. 変更をコミット

## 課題管理

- タスク一覧: `TODO.md`
- 設計プラン: `.claude/plans/lucky-bubbling-globe.md`
