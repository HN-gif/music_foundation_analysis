# wav2vec2 IEMOCAP RSA

IEMOCAP の発話単位音声について、wav2vec2 の各層内部表現からモデル側 RDM を作り、人間アノテーションの VAD 距離 RDM と比較します。

## 目的

- モデル側: 発話間距離行列 `D_model`
- 人間側: VAD 距離行列 `D_VAD`
- RSA: RDM の上三角成分同士の Spearman / Pearson 相関

2 次元に潰す前の距離構造そのものを比較するため、MDS 研究と自然につながる解析です。

## デフォルト仕様

- データセット: `data/IEMOCAP_full_release`
- 音声: `Session*/sentences/wav/*/*.wav`
- アノテーション: `Session*/dialog/EmoEvaluation/*.txt`
- 対象: 発話単位
- `emotion_label=xxx` は除外
- 感情ラベルごとの発話数を数え、`100` を超えるラベルだけを解析対象にする
- wav2vec 表現: 全 hidden states
- 発話ベクトル: 時間方向 mean pooling
- モデル距離: cosine distance
- VAD: summary 行の平均 `[valence, activation, dominance]`
- VAD 前処理: analysis group 内で z-score 標準化
- VAD 距離: euclidean distance
- RSA 指標: Spearman と Pearson
- 解析単位:
  - `all`: 発話者を区別せず全発話を同じ RDM にまとめる
  - `speakers`: `SessionX_F` / `SessionX_M` ごとに話者別 RDM を作る
- dialog type 条件:
  - `combined`: improvisation と script をまとめる
  - `impro`: improvisation のみ
  - `script`: script のみ

## セットアップ

```bash
cd /home/takamichi-lab-pc07/research/music_foundation_analysis/wav2vec_iemocap_rsa
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

既存の `wav2vec_rdm/.venv` に依存が入っている場合は、それを使って実行しても構いません。

## 実行例

全発話、全層、全体分析と話者別分析:

```bash
cd /home/takamichi-lab-pc07/research/music_foundation_analysis
wav2vec_rdm/.venv/bin/python wav2vec_iemocap_rsa/rsa_iemocap_layers.py
```

層を限定:

```bash
wav2vec_rdm/.venv/bin/python wav2vec_iemocap_rsa/rsa_iemocap_layers.py --layers 0 6 12
```

全体分析のみ:

```bash
wav2vec_rdm/.venv/bin/python wav2vec_iemocap_rsa/rsa_iemocap_layers.py --analysis-groups all
```

話者別分析のみ:

```bash
wav2vec_rdm/.venv/bin/python wav2vec_iemocap_rsa/rsa_iemocap_layers.py --analysis-groups speakers
```

improvisation / script / 両方まとめた条件を比較:

```bash
wav2vec_rdm/.venv/bin/python wav2vec_iemocap_rsa/rsa_iemocap_layers.py \
  --layers 0 6 12 \
  --dialog-type-analyses combined impro script
```

improvisation のみを入力として使う:

```bash
wav2vec_rdm/.venv/bin/python wav2vec_iemocap_rsa/rsa_iemocap_layers.py \
  --dialog-types impro \
  --layers 0 6 12
```

デバッグ用に発話数を制限:

```bash
wav2vec_rdm/.venv/bin/python wav2vec_iemocap_rsa/rsa_iemocap_layers.py --max-utterances 200 --layers 0 6 12
```

## 出力

既定の出力先は `research/music_foundation_analysis/wav2vec_iemocap_rsa/outputs` です。

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
      layer_01/
        ...
    speaker_Session1_F/
      layer_00/
        ...
    impro_all/
      layer_00/
        ...
    script_all/
      layer_00/
        ...
```

`summary_metrics.csv` には dialog type 条件、analysis group、層、発話数、Spearman、Pearson、MDS stress などが保存されます。

全発話の RDM CSV は非常に大きくなります。`.npy` は常に保存されるため、CSV が不要な確認実行では `--skip-csv-rdms` を使えます。
