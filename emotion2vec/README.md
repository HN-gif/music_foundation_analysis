# emotion2vec Analysis

Hugging Face 版 `emotion2vec/emotion2vec_base` を FunASR 経由で読み込み、IEMOCAP に対して layer-wise hidden state 解析を行うディレクトリです。

実行環境:

```bash
cd /home/takamichi-lab-pc07/research/music_foundation_analysis/emotion2vec
python3 -m venv .venv
.venv/bin/pip install -U -r requirements.txt
```

実装上は Hugging Face Hub の `emotion2vec/emotion2vec_base` を既定で使います。初回実行時にモデル一式が自動ダウンロードされます。

## Hidden State の定義

`emotion2vec_base` は audio prenet/context encoder の後ろに 8 個の Transformer block を持つため、この実装では次の 9 層を抽出します。

- `layer 0`: audio prenet/context encoder 出力
- `layer 1..8`: 各 Transformer block 出力

各層 hidden state は時間方向に mean pooling して発話ベクトル化します。`iemocap_mds` の解析はこの発話ベクトルを使います。

## 解析スクリプト

- `iemocap_mds/mds_iemocap_emotions.py`
  IEMOCAP の離散感情ラベルに対して A/B/C 方式の距離行列と MDS を作成
- `iemocap_mds/plot_iemocap_norms.py`
  hidden state ベクトルの L2 norm を話者別・男女別・層別に可視化
- `iemocap_rsa/rsa_iemocap_layers.py`
  IEMOCAP の VAD 値を使って layer-wise RSA を実行

詳細は [iemocap_mds/README.md](/home/takamichi-lab-pc07/research/music_foundation_analysis/emotion2vec/iemocap_mds/README.md) を参照してください。
RSA については [iemocap_rsa/README.md](/home/takamichi-lab-pc07/research/music_foundation_analysis/emotion2vec/iemocap_rsa/README.md) を参照してください。
