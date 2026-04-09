# TODO

## 多言語対応（i18n）
- [x] fetch_raw.py を全11言語対応に改修
- [x] 各extractスクリプトをcore/loc分離に改修（goods, religions, governments, laws, countries）
- [x] site ui 文言ファイル作成（site/src/i18n/ui.ts — en/ja、他はen fallback）
- [x] 共通Layout多言語対応（hreflang, canonical, html lang, meta title/desc）
- [x] Astro i18n設定（manual routing, /{lang}/eu5/... ルーティング）
- [x] 交易品ページi18n対応（core+loc、全11言語）
- [x] 言語切替UI（ヘッダードロップダウン、キーボードナビ対応）
- [x] ホームページ多言語化 + ルートリダイレクト（cookie→Accept-Language→en）
- [x] 建物ページURL移行（/{lang}/eu5/buildings/...、名前 en/ja 切替）
- [x] extract_buildings.py のcore/loc分離（requirements AST + facets + 全11言語condition_lines）
- [x] 建物ページのフルi18n化（facetsベースフィルタUI・条件テキスト・modifier/PM多言語化）
- [ ] requirements AST 未対応パターン拡張（ネスト dict/list 形の culture/religion/tag、has_variable、has_estate_privilege 等 57キー）
- [x] 残り9言語のサイトUI文言追加（de, es, fr, ko, pl, pt-br, ru, tr, zh-hans）

## サイトページ作成
- [x] 交易品（74件）— 一覧 + 詳細ページ（全11言語、5,677ページ中825ページ）
  - [x] Building Uses セクション（逆引き: 交易品→建物、input/output表示）
  - [x] 統合POPテーブル（需要 + 富の影響係数、POPアイコン付き）
  - [x] Base Production 列（一覧ページ）
- [x] 建物（439件）— 一覧 + 詳細ページ（フィルタUI・ローカライズ・シミュレーター）
- [x] トップページ（全11言語 + ルートリダイレクト）
- [x] ナビゲーションメニュー（多言語対応済み）
- [x] 宗教（293件）— 一覧 + 詳細ページ（modifier/opinions/mechanics/聖地逆引き、$概念参照解決）
- [x] 聖地（209件）— 一覧（宗教別グループ）+ 詳細ページ（タイプmodifier、イラスト、対象宗教リンク）
- [x] 宗教アスペクト（163件）— B案: 独立ページ + 宗教詳細に逆引き（modifier/排他条件/opinions）
- [ ] 国家（2,328件）— 一覧 + 詳細ページ
- [x] 政体（5件）— 一覧ページ（カード形式、modifier表示、国家数リンク）
- [ ] 法律（191件）— 一覧 + 詳細ページ
- [ ] ロケーション（28,000件）— sql.js基盤で構築

## SEO
- [ ] robots.txt（デプロイ時に即配置）
- [ ] sitemap（locale × entity type で分割、@astrojs/sitemap + ロケーション用外部生成）
- [ ] Search Console登録（ドメイン取得後すぐ）

## DB基盤（ロケーション着手時）
- [ ] Pythonパイプラインからeu5.sqlite並行生成
- [ ] sql.js + Web Worker + 共通クエリ層（db-worker.ts）
- [ ] ページネーション・facetカウント実装

## インタラクティブマップ＋セーブデータ（将来課題）
- [ ] map_data/ からプロヴィンス境界データ取得・解析
- [ ] Canvas/WebGL でマップ描画、クリックでロケーション詳細表示
- [ ] セーブデータ(.eu5)のブラウザ内ストリーミング解析（File API + ReadableStream）
- [ ] セーブ内ロケーションセクションの構造特定（所有国・建物・人口の格納形式）
- [ ] 解析結果をマップに反映（サーバー送信なし、ブラウザ完結）

## 攻略記事
- [ ] Astro Content Collections 設定
- [ ] 記事テンプレート作成
- [ ] 記事一覧・詳細ページ

## タイポグラフィ
- [x] 全サイトのフォントサイズを DESIGN.md Type Scale に統一（見出し 1.15rem / 本文 1rem）
- [x] ナビ・言語切替・テーブルヘッダー・タグを 1rem に統一
- [x] CSS変数トークン化（--type-* 11トークン、207箇所置換、check-type-scale.sh でビルド時検証）

## パイプライン改善
- [x] loc_parser.py — [word] / [ShowPopTypeName] のマークアップ処理修正
- [x] extract_game_terms.py — $game_concept_X$ / $pop$ 参照の解決
- [x] extract_buildings.py — unique_production_methods の list 型対応（8建物修正）
- [x] fetch_icons.py — pops カテゴリ追加（8種POPアイコン）

## 開発ツール
- [x] /review-2model スキル（Claude + Codex 並行レビュー）
- [x] paradoxpedia-reviewer エージェント（既知バグパターン7種チェック）

## アイコン・画像
- [ ] CoAレンダラー — パターン+エンブレム+色の定義から国旗画像を生成（2,328国）

## デプロイ
- [ ] ムームードメインで paradoxpedia.com 取得
- [ ] Cloudflare Workers + R2 ($5/月) セットアップ
- [ ] Workers Builds で git push デプロイ構築
- [ ] Accept-Language リダイレクト（/ のみ、302、cookie優先）
- [ ] 重いアセット（.db/.wasm）は R2 に配置
- [ ] 動作確認
