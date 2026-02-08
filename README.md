# Easy PNGTuber

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

PNGTuber用の表情差分画像を簡単に作成するツールです。

📝 [使い方の解説記事（note）](https://note.com/rotejin/n/n106abaaa3957)

[English](README_EN.md)

---

## 更新情報（2026-02-08）

- **位置合わせ余白の自動トリミング**を追加しました。  
  `mask_composer.py` / `parts_mixer.py` / `simple_aligner_app.py` の保存時に、位置合わせで生じた端の空白を自動で除去できます。
- 保存欄の **「位置合わせ余白を自動トリミング」** をONにし、必要に応じて **マージン(px)** を調整してください。
- 余白を最小化したい場合は、マージンを `0px` に設定してください。

---

## 特徴

- 目と口を別々のソースから選んで4パターン自動生成
- 2x2 表情シートから分割＆位置合わせ
- AKAZE / ORB 特徴点マッチングによる高精度位置合わせ
- 位置合わせで発生した余白を自動トリミングして保存
- 位置合わせスコア/成功率のステータス可視化
- 低スコア画像への再試行導線（しきい値・ジャンプ）
- ツール設定の自動保存（前回ディレクトリ/主要UI設定）
- オーバーレイ表示で差分確認しながらマスク描画
- 日本語ファイルパス対応
- シンプルなGUI（PySide6）

---

## クイックスタート

まず **表情差分シート**（2x2の表情バリエーション画像）を用意する必要があります（[Step 2](#step-2-表情差分シートの作成) 参照）。

### Step 1: ツールのインストール

```bash
# リポジトリをダウンロードまたはクローン
git clone https://github.com/rotejin/EasyPNGTuber.git
cd EasyPNGTuber

# 依存パッケージをインストール（仮想環境も自動作成）
uv sync
```

> [uv](https://docs.astral.sh/uv/) がインストールされていない場合: `pip install uv` または [公式サイト](https://docs.astral.sh/uv/getting-started/installation/) 参照

### Step 2: 表情差分シートの作成

1. **キャラクター画像を用意**
   - PNGTuberにしたいキャラクターの画像（1枚）を用意します

2. **Grid Tilerで2x2画像を作成**
   ```bash
   uv run python grid_tiler.py
   ```
   - 用意した画像をドラッグ＆ドロップ
   - 「タイリング画像を保存」で2x2画像を出力

3. **AIで表情差分を生成**
   - [Google AI Studio](https://aistudio.google.com/) の画像生成AI（Nano Banana）を使用
   - 2x2画像をアップロードし、以下のプロンプトで表情差分を生成:

   <details>
   <summary>プロンプトを表示（クリックで展開）</summary>

   ```yaml
   expression_sheet:
     task: "edit_reference_image"
     format:
       grid: "2x2"
       preserve: "exact_pixel_position"

     critical_rule:
       source: "use provided reference image as base"
       maintain:
         - original_art_style
         - original_character_design
         - original_face_angle
         - original_head_tilt
         - original_color_palette
         - original_lighting
         - original_line_weight
       do_not_change:
         - head_position
         - face_outline
         - hair
         - background
         - overall_composition

     editable_elements:
       - eyelids_only
       - mouth_only

     parts_definition:
       eyes:
         open: "natural relaxed open state"
         closed: "gentle blink, relaxed eyelids"
       mouth:
         closed: "lips together, neutral"
         open: "natural speaking, showing inside"

     panels:
       top_left:
         action: "keep_unchanged"

       top_right:
         eyes: "open"
         mouth: "open"

       bottom_left:
         eyes: "closed"
         mouth: "closed"

       bottom_right:
         eyes: "closed"
         mouth: "open"
   ```

   </details>

### Step 3: Parts Mixerで仕上げ

```bash
uv run python parts_mixer.py
```

生成された表情差分シートをParts Mixerで読み込み、目と口のパーツを組み合わせてPNGTuber用の4パターンを出力します。

---

## Parts Mixerの使い方

目と口のパーツを別々のソース画像から選択し、4パターン（目ON/OFF × 口ON/OFF）を自動生成します。

AI画像生成で表情差分を一括生成した際、一部のパーツが期待通りにならない場合に便利です。

```bash
uv run python parts_mixer.py
```

1. 表情シートをドラッグ＆ドロップ
2. 「分割＆位置合わせ」を実行
3. ベース画像 / 目ソース / 口ソースをそれぞれ選択（ベースが目閉じ口閉じの場合、目ソースは目開き、口ソースは口開きを選ぶ）
4. 目キャンバスで目の領域をマスク
5. 口キャンバスで口の領域をマスク
6. 4パターンがプレビューに表示
7. 「4パターン一括保存」で出力

### 出力ファイル

- `{元画像名}_eyeOFF_mouthOFF.png` - 目OFF 口OFF
- `{元画像名}_eyeON_mouthOFF.png` - 目ON 口OFF
- `{元画像名}_eyeOFF_mouthON.png` - 目OFF 口ON
- `{元画像名}_eyeON_mouthON.png` - 目ON 口ON

---

## サンプル画像

`sample/` フォルダに動作確認用のサンプル画像が含まれています。

- `tomari_sample.png` - 2x2表情シートのサンプル

---

## トラブルシューティング

### ツールが起動しない

- Python 3.10以上がインストールされているか確認
- `.venv` フォルダを削除して `uv sync` を再実行

### 画像が読み込めない

- 対応形式: PNG, JPG, BMP, WebP
- 日本語ファイル名も対応しています

### 位置合わせがうまくいかない

- 画像の差異が大きすぎる場合は失敗することがあります
- 回転角が±30度を超える場合は対応できません

### 位置合わせ後に端の空白が気になる

- 保存時に「位置合わせ余白を自動トリミング」をONにしてください
- 必要に応じて「マージン(px)」で余白量を微調整できます

---

## 技術仕様

### 位置合わせアルゴリズム

- **AKAZE特徴点マッチング**（メイン）
- **ORB**（フォールバック）
- **RANSAC** によるアフィン変換推定
- AKAZE/ORBともにバイナリ記述子に適した距離指標でマッチング

### 余白トリミング仕様

- 変換時に有効領域マスク（元画像由来の画素）を同時生成
- 保存時に複数画像の有効領域の共通部分を自動トリミング
- 全出力で同一矩形を使うため、PNGTuber用フレームサイズを維持

### 制限パラメータ

| パラメータ | 値 |
|-----------|-----|
| 最大回転角 | ±30度 |
| スケール範囲 | 0.8～1.2倍 |
| 成功スコア閾値 | 0.6以上 |

---

## 推奨環境

| 項目 | 要件 |
|------|------|
| Python | 3.10 以上 |
| OS | Windows / macOS / Linux |
| パッケージ管理 | [uv](https://docs.astral.sh/uv/) 推奨 |

---

## ファイル構成

```
EasyPNGTuber/
├── parts_mixer.py        # メインツール: パーツ合成
├── grid_tiler.py         # 画像タイリング
├── mask_composer.py      # マスク合成
├── simple_aligner_app.py # 位置合わせ
├── aligner.py            # 位置合わせエンジン
├── compositor.py         # 画像合成エンジン
├── cv2_utils.py          # OpenCVユーティリティ
├── mask_canvas.py        # マスクキャンバスUI
├── preview_widget.py     # プレビューUI
├── gemini_prompt.txt     # AI用プロンプト
├── pyproject.toml        # 依存パッケージ定義
└── sample/               # サンプル画像
```

---

## ライセンス

[MIT License](LICENSE)

Copyright (c) 2026 rotejin
