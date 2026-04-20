# GoshuinScan-OSS
[English](README_en.md)

御朱印のための AI デジタルアーカイブ化ツール

## 機能
Python GUI を使用して御朱印の写真を処理します：
1. **視角補正と切り抜き**：RMBG-2.0 を使用してページの輪郭を検出し、パース（傾きや立体的な歪み）を自動で補正して長方形に切り抜きます。
2. **ドキュメント補正**：docTR と古典的な画像補正アルゴリズム（CLAHE およびシャープネス）を組み合わせ、残りの微小な傾きを修正し、コントラストを強化します。
3. **背景除去とインク抽出**：RMBG-2.0 を再度適用して背景を完璧に取り除き、適応的閾値（Adaptive Threshold）などで朱印（赤）と墨跡（黒）のみを保持した高精度な透過 PNG を出力します。

## 必須環境
- Python 3.10+
- Windows / Linux / macOS
- NVIDIA GPU 推奨（CUDA が自動的に使用されます）

## インストール
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> 初回実行時に docTR および RMBG-2.0 のモデル重みがダウンロードされるため、インターネット接続が必要です。

## 実行方法
```bash
python app.py
```

## Hugging Face 認証 (RMBG-2.0 の利用に必須)
`briaai/RMBG-2.0` は gated model（アクセス制限付きモデル）のため、アクセス権の申請が必要です。

1. 以下のリンクにアクセスし、アクセス権を申請してください: `https://huggingface.co/briaai/RMBG-2.0`
2. `hf` コマンドでログインします（推奨）:

```powershell
.\.venv\Scripts\hf auth login
```

> **fine-grained token** を使用する場合は、トークンの設定で `Read access to public gated repositories you can access` を有効にしてください。そうしないと `403 Forbidden` エラーが発生します。

環境変数を使用することもできます:

```powershell
$env:HF_TOKEN = "hf_xxx"
```

まだ `RMBG-2.0` のアクセス権を取得していない場合、一時的に公開モデルに切り替えることも可能です:

```powershell
$env:RMBG_MODEL_ID = "briaai/RMBG-1.4"
```

## GUI の使い方
1. `画像を選択` をクリックして、御朱印の画像を選択します。
2. `フォルダー選択` をクリックして、出力先のディレクトリを指定します。
3. 任意：`GPU (CUDA) を使用` にチェックを入れます。
4. `処理開始` をクリックします。

処理が完了すると、指定した出力ディレクトリに以下のファイルが生成されます：
- `*_enhanced_doctr.png`：視角およびドキュメント補正済みの画像
- `*_ink_stamp_transparent.png`：黒墨と赤朱印のみを保持した透過背景 PNG
