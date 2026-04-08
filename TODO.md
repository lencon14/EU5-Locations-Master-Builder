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
- [ ] extract_buildings.py のcore/loc分離（25個のloc参照、条件テキストi18n設計が必要）
- [ ] 建物ページのフルi18n化（フィルターUI・ツールチップ・条件テキスト）
- [ ] 残り9言語のサイトUI文言追加（de, es, fr, ko, pl, pt-br, ru, tr, zh-hans）

## サイトページ作成
- [x] 交易品（74件）— 一覧 + 詳細ページ（全11言語、5,677ページ中825ページ）
- [x] 建物（439件）— 一覧 + 詳細ページ（フィルタUI・ローカライズ・シミュレーター）
- [x] トップページ（全11言語 + ルートリダイレクト）
- [x] ナビゲーションメニュー（多言語対応済み）
- [ ] 宗教（293件）— 一覧 + 詳細ページ
- [ ] 国家（2,328件）— 一覧 + 詳細ページ
- [ ] 政体（5件）— 一覧ページ
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

## アイコン・画像
- [ ] CoAレンダラー — パターン+エンブレム+色の定義から国旗画像を生成（2,328国）

## デプロイ
- [ ] ムームードメインで paradoxpedia.com 取得
- [ ] Cloudflare Workers + R2 ($5/月) セットアップ
- [ ] Workers Builds で git push デプロイ構築
- [ ] Accept-Language リダイレクト（/ のみ、302、cookie優先）
- [ ] 重いアセット（.db/.wasm）は R2 に配置
- [ ] 動作確認
