# DESIGN.md — EU5 攻略DB

## 1. Visual Theme & Atmosphere

中世ヨーロッパの書斎。羊皮紙の上に金箔で押された文字、蝋燭の灯りに照らされた古地図、革装丁の書物が並ぶ空間。豪華だが控えめ、荘重だが読みやすい。

キーワード: **重厚、荘厳、羊皮紙、金箔、暗室、古地図**

Paradox社のグランドストラテジーゲームのUIを想起させつつ、情報密度の高いデータベースサイトとして機能する。装飾は目的を持つ場所にのみ使い、データの可読性を最優先する。

## 2. Color Palette & Roles

### Primary Surface (背景の層)

| Name | Hex | Role |
|------|-----|------|
| Deep Obsidian | `#0d0f14` | 最深層の背景。ページ全体のベース |
| Dark Walnut | `#161a23` | カード・パネルの背景。主要サーフェス |
| Charred Oak | `#1e2330` | ヘッダー・フッター・テーブルヘッダー |
| Slate | `#2a3042` | ホバー状態・選択状態の背景 |

純粋な `#000000` は使わない。常に青みがかった暗色で、中世の夜空を感じさせる。

### Accent (差し色)

| Name | Hex | Role |
|------|-----|------|
| Royal Gold | `#c9a84c` | 主要アクセント。見出し装飾、アクティブ要素、CTA |
| Pale Gold | `#e8d5a3` | セカンダリアクセント。ホバーリンク、小見出し |
| Parchment | `#f0e6d0` | 強調テキスト、バッジ背景 |

Gold は権威と歴史を表す。派手にならないよう彩度を抑えた Old Gold 系統を使う。

### Text

| Name | Hex | Role |
|------|-----|------|
| Ivory | `#e8e4dc` | 本文テキスト。温かみのある白 |
| Dusty Silver | `#9a9bab` | 補助テキスト、説明文、キャプション |
| Dim Stone | `#5c5f72` | 非活性テキスト、プレースホルダ |

純白 `#FFFFFF` は使わない。常にわずかに黄味のある Ivory 系で、羊皮紙の上のインクを再現する。

### Border & Divider

| Name | Hex | Role |
|------|-----|------|
| Iron Edge | `#2e3447` | カード・テーブルの境界線 |
| Gold Thread | `rgba(201,168,76,0.3)` | アクセント用の境界線。セクション区切り |

### Semantic (状態色)

| Name | Hex | Role |
|------|-----|------|
| Crest Red | `#c44f4f` | エラー、危険、マイナス効果 |
| Forest Green | `#4c9a5a` | 成功、プラス効果、成長 |
| Azure Banner | `#4a8ec9` | 情報、リンクのデフォルト色 |
| Harvest Amber | `#c9944c` | 警告、注意 |

## 3. Typography Rules

### Font Stack

- **見出し（Display/Heading）**: `"Cinzel", "Cormorant Garamond", Georgia, serif`
  - Cinzel は碑文・古典彫刻から着想を得たセリフ体。EU5の時代感に合う。Google Fonts で利用可能
- **本文（Body/UI）**: `"Inter", "Helvetica Neue", Arial, "Hiragino Kaku Gothic ProN", "Hiragino Sans", Meiryo, sans-serif`
  - 日本語テキストの可読性を確保するため、本文はサンセリフ体

### Type Scale

| Token | Size | Weight | Font | Use |
|-------|------|--------|------|-----|
| display | 2.5rem (40px) | 700 | Cinzel | ページタイトル |
| h1 | 2rem (32px) | 700 | Cinzel | セクション見出し |
| h2 | 1.5rem (24px) | 600 | Cinzel | サブセクション |
| h3 | 1.15rem (18.4px) | 600 | Cinzel | カード見出し、テーブルグループ |
| body | 1rem (16px) | 400 | Inter | 本文 |
| small | 0.875rem (14px) | 400 | Inter | 補助テキスト、テーブルセル |
| caption | 0.75rem (12px) | 500 | Inter | ラベル、タグ、バッジ |

### Rules

- 見出しの `letter-spacing`: `0.03em`（Cinzel の碑文感を引き立てる）
- 本文の `line-height`: `1.7`（データサイトなので行間は広めに）
- 見出しの `line-height`: `1.3`
- 英語テキストの見出しは `text-transform: uppercase` を使ってもよい
- 日本語テキストには `text-transform` を適用しない

## 4. Component Stylings

### Navigation Bar

- Background: Charred Oak `#1e2330`
- Border-bottom: `1px solid` Iron Edge
- サイトタイトルは Royal Gold で表示
- ナビリンクは Ivory、ホバーで Pale Gold
- 高さ: 56px、内側 padding `0 1.5rem`

### Buttons

**Primary CTA:**
- Background: Royal Gold `#c9a84c`
- Text: Deep Obsidian `#0d0f14`
- Border-radius: `2px`（中世の角張った意匠）
- Padding: `0.5rem 1.25rem`
- Font: Inter 500, `text-transform: uppercase`, `letter-spacing: 0.05em`
- Hover: Pale Gold `#e8d5a3` に明るく

**Ghost Button:**
- Background: transparent
- Border: `1px solid` Royal Gold
- Text: Royal Gold
- Hover: background `rgba(201,168,76,0.1)`

**Text Button / Link:**
- Color: Azure Banner `#4a8ec9`
- Hover: Pale Gold `#e8d5a3`、下線表示
- Active: Royal Gold

### Cards

- Background: Dark Walnut `#161a23`
- Border: `1px solid` Iron Edge `#2e3447`
- Border-radius: `4px`
- Padding: `1.25rem`
- Hover: border を Gold Thread に変化させる。background は変えない
- タイトルは h3 スタイル（Cinzel）

### Data Tables

- Table header: Charred Oak 背景、caption スタイル（uppercase, Dusty Silver）
- Row border: `1px solid` Iron Edge
- Row hover: Slate `#2a3042` 背景
- 数値セル: `text-align: right`, `font-variant-numeric: tabular-nums`
- 交互行の色分けはしない（ボーダーで十分区別できる）

### Tags / Badges

- Background: `rgba(201,168,76,0.15)`
- Text: Pale Gold
- Border-radius: `2px`
- Padding: `0.15rem 0.5rem`
- Font: caption サイズ

### Search Input

- Background: Dark Walnut
- Border: `1px solid` Iron Edge
- Border-radius: `4px`
- Focus: border を Royal Gold に変化、`box-shadow: 0 0 0 2px rgba(201,168,76,0.2)`
- Placeholder: Dim Stone

### Breadcrumb

- Font: small サイズ
- Color: Dusty Silver
- Separator: `›`（右向き山括弧）
- Current page: Ivory（リンクなし）

## 5. Layout Principles

### Container

- Max-width: `960px`
- Padding: `0 1.5rem`
- 中央揃え（`margin: 0 auto`）

### Spacing Scale (8px base unit)

| Token | Value |
|-------|-------|
| xs | 4px (0.25rem) |
| sm | 8px (0.5rem) |
| md | 16px (1rem) |
| lg | 24px (1.5rem) |
| xl | 32px (2rem) |
| 2xl | 48px (3rem) |

### Section Spacing

- セクション間: `2xl` (48px)
- セクション見出しと内容の間: `lg` (24px)
- カードグリッドのガター: `md` (16px)

### Grid

- カードグリッド: `repeat(auto-fill, minmax(280px, 1fr))`
- データテーブル: 全幅（100%）

## 6. Depth & Elevation

シャドウは控えめに使う。中世の書斎は蝋燭の柔らかい光なので、鋭いドロップシャドウは避ける。

| Level | Shadow | Use |
|-------|--------|-----|
| Surface (0) | none | カード、テーブル。ボーダーで区別する |
| Raised (1) | `0 2px 8px rgba(0,0,0,0.3)` | ドロップダウン、ツールチップ |
| Overlay (2) | `0 4px 16px rgba(0,0,0,0.5)` | モーダル、ポップオーバー |

背景色のレイヤリング（Deep Obsidian → Dark Walnut → Charred Oak → Slate）でサーフェスの階層を表現する。シャドウに頼らない。

## 7. Do's and Don'ts

### Do

- Gold アクセントは見出し装飾とインタラクティブ要素に限定して使う
- データの可読性を最優先する。装飾は情報を邪魔しない範囲で
- 見出しには Cinzel セリフ体、本文には Inter サンセリフ体を一貫して使う
- テーブルの数値は右揃え、`tabular-nums` で桁を揃える
- サーフェスの区別はボーダーと背景色の階層で表現する

### Don't

- 純黒 `#000000` や純白 `#FFFFFF` を使わない。温かみのある暗色と Ivory 系で統一
- Gold を背景色としてべた塗りしない。テキストやボーダーのアクセントとして使う
- 大きな border-radius（8px 超）を使わない。中世の角張った意匠に合わない
- グラデーション背景を使わない。ソリッドカラーのレイヤリングで深度を出す
- 影を多用しない。Level 1 以上のシャドウはドロップダウンとモーダルのみ
- テーブルの交互行カラーリング（ストライプ）を使わない

## 8. Responsive Behavior

### Breakpoints

| Name | Width | Behavior |
|------|-------|----------|
| mobile | < 640px | 単一カラム。テーブルは横スクロール |
| tablet | 640-959px | 2カラムグリッド |
| desktop | ≥ 960px | フルレイアウト（max-width 960px） |

### Mobile Adaptations

- ナビゲーションはハンバーガーメニューに折り畳む
- テーブルは `overflow-x: auto` のスクロールコンテナで包む
- カードグリッドは1列に
- フォントサイズは変更しない（rem ベースなのでブラウザ設定に従う）

## 9. Agent Prompt Guide

### Quick Color Reference

```
背景:    #0d0f14 → #161a23 → #1e2330 → #2a3042
Gold:    #c9a84c (primary) → #e8d5a3 (light) → #f0e6d0 (parchment)
テキスト: #e8e4dc (primary) → #9a9bab (muted) → #5c5f72 (dim)
ボーダー: #2e3447 (default) → rgba(201,168,76,0.3) (gold)
リンク:   #4a8ec9 (default) → #e8d5a3 (hover)
```

### Component Quick Start

新しいページやコンポーネントを作るとき:
1. 背景は Dark Walnut `#161a23` のカードに Iron Edge ボーダーで囲む
2. 見出しは Cinzel、h2 以上には `letter-spacing: 0.03em`
3. テーブルは全幅、ヘッダーは Charred Oak 背景 + uppercase ラベル
4. アクセントが必要な場所に Royal Gold を控えめに使う
5. インタラクティブ要素のホバーは Gold 系に変化させる
