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
- **【必須】ページの新規作成・UI変更時は frontend-design スキルを使用する。DESIGN.md を読むだけでなく、スキルの審美眼でデザイン品質を担保すること**
- **【必須】`font-size` を直書きしない。`var(--type-*)` トークンを使う（DESIGN.md Type Scale 参照）。`scripts/check-type-scale.sh` で検証**

## データパイプライン（pipeline/）

- `fetch_raw.py` — SSH経由で全カテゴリのゲームデータを一括取得（シェル版 fetch_raw.sh は非推奨、Python版を使う）
- 抽出スクリプト: `extract_goods.py`, `extract_buildings.py`, `extract_countries.py`, `extract_religions.py`, `extract_governments.py`, `extract_holy_sites.py`
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
| 聖地 | 209 | extract_holy_sites.py |
| 聖地タイプ | 10 | extract_holy_sites.py |
| 宗教アスペクト | 163 | extract_aspects.py |

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
| holy_site_types | 4 |
| holy_site_illustrations | 16 |

## データ更新フロー（ゲームバージョンアップ時）

1. `cd pipeline && python3 fetch_raw.py` で全データ再取得
2. `python3 fetch_icons.py` でアイコン再取得・変換
3. `git diff pipeline/raw/` で変更点を確認
4. 各 `extract_*.py` を実行してJSON再生成
5. `python3 generate_all_loc.py` で統合loc辞書を再生成（all_loc.json）
6. `cp pipeline/output/*.json site/src/data/`
7. `cp -r pipeline/output/icons site/public/`
8. `cd site && npm run build` でサイトリビルド
9. 変更をコミット

## 翻訳品質ルール

### 【必須】ゲーム用語の独自翻訳禁止
- ゲーム内用語は必ず公式ローカライズファイル（`pipeline/raw/localization/`）から取得する
- 公式訳が見つからない場合は英語のまま表示するか、ユーザーに確認する
- 「だいたいこういう意味だろう」で訳を作らない

### 【必須】ビルド後の翻訳漏れ全ページ監査
新ページ作成・データ追加後は、ビルド後に `dist/ja/` の全ページから英語混入を自動検出するスクリプトを実行すること。レビュー依頼前に必ず実施する。

チェック対象:
1. `$variable$` 未解決
2. modifier/mechanic名が raw key のまま（underscore 連結の英語）
3. 日本語テキスト中の英単語混入（Clergy Estate 等）
4. タイプ名・場所名が英語のまま
5. 説明文が助詞で始まる（$変数$ が strip で消えた痕跡）

### 【必須】新カテゴリ追加時の $variable$ 解決チェックリスト
extract スクリプト作成後、出力 loc JSON の全言語・全値を `$` でスキャンし、未解決変数がゼロであることを確認する。

確認項目:
1. var_lookup に必要な loc ソースが全て含まれているか（pops, estate, buildings, location_names, game_concepts, cultures, country_names 等）
2. メカニクス/タグ名のマッピングに全フラグが含まれているか
3. modifier名の loc カバレッジが100%か
4. タイプ名/カテゴリ名が全言語で loc にあるか

## 課題管理

- タスク一覧: `TODO.md`
- タスク完了時・方針変更時は必ず TODO.md を更新すること

## デプロイ先（決定済み）

- ドメイン: `paradoxpedia.com`（ムームードメインで取得予定）
- ホスティング: Cloudflare Workers Static Assets + R2（月$5）
- 契約はサイト完成後。今は開発に集中
