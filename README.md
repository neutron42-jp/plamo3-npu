# PLaMo-3 Translate 0.8B on AMD Ryzen AI NPU

[pfnet/plamo-3-translate-0.8b](https://huggingface.co/pfnet/plamo-3-translate-0.8b) を AMD Ryzen AI NPU（XDNA2 / `aie2p`）上でネイティブ実行・高速化するための変換パイプライン、CLI ランナー、および Google 翻訳風 WebUI です。

## 主な特徴

- **DynamicDispatch Full-Fusion NPU 実行**: Prefill（321 fused ops）・Token decode（209 fused ops）の双方を NPU 制御パケットへ統合し、`lm_head` には AVX-512 VNNI (`accuracy_level=4`) を組み合わせることで安定 **~56–57 tok/s** のデコード速度を実現
- **PLaMo-3 → Gemma3-style OGA INT4 変換**: `hidden_size` の 1536 → 2048 ゼロパディング＋RMSNorm スケーリング、offset 畳み込み、GQA KV 複製（1 → 8 heads）、Local / Global 2 系統 RoPE キャッシュを実装
- **長文対応（履歴引き継ぎチャンク分割）**: Ryzen AI 1.8.0 の Strix カーネルにおける `window_size=2048` 非対応制約（`local_window_size=-1` 強制による RoPE 局所エイリアシング・反復崩壊）を回避するため、直前 1 ターンの訳文履歴を保ちつつ 1 NPU パスあたり `<= 1850` トークンへ自動分割＋反復検知ガードを実装
- **WebUI (`webui/`)**: SSE ストリーミング、54 言語＋やさしい日本語 (`ja-easy`)、敬体／常体切替、文脈 (`context`)・用語集 (`glossary`)、チャンク進捗表示・停止ボタン・リクエストログに対応

## 検証済み環境・対応ハードウェア

- **検証実機**: AMD Ryzen AI 9 HX 370 (Strix Point, `NPU Strix` / `aie2p` / Topology `6x8`)、Linux + XRT 2.26.0 + `amdxdna`
- **NPU 世代の目安**:
  - **動作対象**: XDNA2 (`aie2p`) 搭載 Ryzen AI 製品（Ryzen AI 300 / Ryzen AI Max / Ryzen AI 400 系など。XRT / `amdxdna` が対象デバイスの PCI ID に対応している必要があります）
  - **非対応**: XDNA1 世代（Ryzen 7040 / 8040 / 200 系など）、および将来の別世代アーキテクチャ（再ビルド要）

## 実測ベンチマーク（Ryzen AI 9 HX 370 / NPU Strix）

`python3 run_plamo3_npu.py --verify_ref --benchmark`（116 トークン入力、ウォーム実行 3 回）の結果:

| 指標 | 実測値 |
| :--- | :--- |
| Prefill 平均レイテンシ | **70.17 ms** |
| Steady Decode 平均速度 | **17.77 ms/tok (56.27 tok/s)** |
| Steady Decode 中央値 | **17.58 ms/tok (56.90 tok/s)** |
| リファレンス一致確認 (`--verify_ref`) | **3 / 3 ケースで top-1 logits 一致** |

## セットアップ

### 1. 前提環境

1. Linux + XRT / `amdxdna` ドライバが導入され、`xrt-smi examine` で NPU が認識されていること
2. **AMD Ryzen AI Software 1.8.0**（Python 3.12 仮想環境 `.venv`）がインストール済みであること
3. WebUI / ダウンロード用パッケージが必要な場合は仮想環境へ追加:

```bash
source env.sh
uv pip install aiohttp huggingface_hub
```

> **Note**: 仮想環境を `.venv` 以外の場所に構築した場合は `export RYZEN_AI_VENV=/path/to/venv` を指定してから `source env.sh` を実行してください。

### 2. モデルの取得・変換・NPU ビルド

生成されたモデル本体やキャッシュ（数 GB）はリポジトリには含まれません。以下の順でローカルに生成します（`models/` 配下に出力され `.gitignore` 対象になります）。

```bash
source env.sh

# 1. Hugging Face から PLaMo-3 Translate 0.8B を取得
python3 download_model.py

# 2. (任意) PyTorch FP32 基準出力を更新する場合
# python3 cpu_ref.py

# 3. OGA INT4 ONNX へエクスポート（hidden 2048 pad, kv_heads=8, block_size=128）
python3 export_plamo3_oga.py

# 4. (任意) CPU ベースラインでの等価性検証
python3 verify_oga_cpu.py

# 5. DynamicDispatch Full-Fusion NPU モデルをビルド
python3 build_plamo3_npu.py

# 6. (任意) 融合された ONNX オペレータ構成を監査
python3 audit_npu_model.py
```

## 使い方

### CLI 推論・検証・ベンチマーク

```bash
source env.sh

# リファレンス検証とベンチマーク
python3 run_plamo3_npu.py --verify_ref --benchmark

# 任意テキストの翻訳（長文は自動で履歴付きチャンク分割）
python3 run_plamo3_npu.py --source_lang en --target_lang ja --text "Hello from Ryzen AI NPU."
```

### WebUI

```bash
source env.sh
python3 webui/webui_plamo3_npu.py --host 127.0.0.1 --port 7860
```

ブラウザで `http://127.0.0.1:7860` を開きます。

> **注意 (XRT 排他制御)**: NPU デバイスコンテキストは 1 プロセス専有のため、WebUI 起動中は別ターミナルから `run_plamo3_npu.py` 等の NPU CLI を同時実行しないでください。

## ファイル構成

```text
.
├── env.sh                    # Ryzen AI .venv / XRT 環境変数設定
├── download_model.py         # pfnet/plamo-3-translate-0.8b 取得スクリプト
├── cpu_ref.py                # PyTorch FP32 リファレンス出力生成
├── cpu_reference.json        # 検証用リファレンス入出力・logits
├── export_plamo3_oga.py      # PLaMo-3 -> OGA INT4 ONNX 変換（1536->2048 pad等）
├── verify_oga_cpu.py         # 変換後 OGA ONNX の CPU 等価性テスト
├── build_plamo3_npu.py       # DynamicDispatch Full-Fusion NPU ビルド
├── audit_npu_model.py        # 融合済み ONNX / キャッシュメタデータ監査
├── run_plamo3_npu.py         # NPU CLI 推論・チャンク分割・検証・ベンチマーク
└── webui/
    ├── webui_plamo3_npu.py   # aiohttp SSE WebUI バックエンドサーバ
    └── index.html            # Google 翻訳風 WebUI フロントエンド
```

## ライセンス・注意事項

- 本リポジトリ内のスクリプト群とは別に、ベースモデル（`pfnet/plamo-3-translate-0.8b`）およびトークナイザ等は Preferred Networks の **PLaMo Community License**（https://plamo.preferredai.jp/info/plamo-community-license-en ）に従います。
- AMD Ryzen AI Software および XRT バイナリはそれぞれの AMD EULA / サードパーティライセンスに従います。
