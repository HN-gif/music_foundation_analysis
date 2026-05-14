# emotion2vec IEMOCAP RSA

IEMOCAP の発話単位音声について、`emotion2vec/emotion2vec_base` の各層内部表現からモデル側 RDM を作り、人間アノテーションの VAD 距離 RDM と比較します。

## 目的

- モデル側: 発話間距離行列 `D_model`
- 人間側: VAD 距離行列 `D_VAD`
- RSA: RDM の上三角成分同士の Spearman / Pearson 相関

## デフォルト仕様

- データセット: `data/IEMOCAP_full_release`
- 音声: `Session*/sentences/wav/*/*.wav`
- アノテーション: `Session*/dialog/EmoEvaluation/*.txt`
- `emotion_label=xxx` は除外
- 感情ラベルごとの発話数を数え、`100` を超えるラベルだけを解析対象にする
- emotion2vec 表現: 全 hidden-state 層
- 発話ベクトル: 時間方向 mean pooling
- モデル距離: cosine distance
- VAD: `[valence, activation, dominance]`
- VAD 前処理: analysis group 内で z-score 標準化
- VAD 距離: euclidean distance
- RSA 指標: Spearman と Pearson

## 実行例

```bash
cd /home/takamichi-lab-pc07/research/music_foundation_analysis
emotion2vec/.venv/bin/python emotion2vec/iemocap_rsa/rsa_iemocap_layers.py
```

層を限定:

```bash
emotion2vec/.venv/bin/python emotion2vec/iemocap_rsa/rsa_iemocap_layers.py \
  --layers 0 4 8
```

impro のみを入力として使う:

```bash
emotion2vec/.venv/bin/python emotion2vec/iemocap_rsa/rsa_iemocap_layers.py \
  --dialog-types impro \
  --layers 0 4 8
```

Hugging Face を明示:

```bash
emotion2vec/.venv/bin/python emotion2vec/iemocap_rsa/rsa_iemocap_layers.py \
  --hub hf \
  --model-name emotion2vec/emotion2vec_base
```

## 出力

既定の出力先は `research/music_foundation_analysis/emotion2vec/iemocap_rsa/outputs` です。

```text
outputs/
  utterance_metadata.csv
  summary_metrics.csv
  layer_correlation_plot.png
  analyses/
    all/
      layer_00/
        model_rdm.npy
        model_rdm.csv
        vad_rdm.npy
        vad_rdm.csv
        rdm_scatter.png
        model_mds_coordinates.csv
        model_mds.png
        vad_mds_coordinates.csv
        vad_mds.png
        model_umap_coordinates.csv
        model_umap.png
        vad_umap_coordinates.csv
        vad_umap.png
```
