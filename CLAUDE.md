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

## データ更新フロー（ゲームバージョンアップ時）

1. `pipeline/raw/` のゲームファイルをSSH経由で再取得して上書き
2. `pipeline/raw/VERSION.txt` を更新
3. `git diff pipeline/raw/` で変更点を確認
4. `cd pipeline && python3 extract_goods.py` でJSON再生成
5. `cp pipeline/output/goods.json site/src/data/`
6. `cd site && npm run build` でサイトリビルド
7. 変更をコミット
