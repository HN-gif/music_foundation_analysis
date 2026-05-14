# emotion2vec IEMOCAP MDS

IEMOCAP の離散感情ラベル付き発話音声に対して、Hugging Face 版 `emotion2vec/emotion2vec_base` の hidden states を抽出し、感情ラベル間距離行列を作成して MDS で 2 次元可視化します。

## 解析方式

- データセット: `data/IEMOCAP_full_release`
- 音声: `Session*/sentences/wav/*/*.wav`
- アノテーション: `Session*/dialog/EmoEvaluation/*.txt`
- `emotion_label=xxx` は既定で除外
- `emotion_label=oth` は常に除外
- 各発話から emotion2vec hidden states を取得
- `layer 0` は audio prenet/context encoder, `layer 1..8` は Transformer block 出力
- 発話ベクトルは時間方向 mean pooling
- 距離は既定で cosine distance
- `--distance-metric norm-only` では L2 ノルム差 `| ||x||_2 - ||y||_2 |` を距離にする
- 既定では `--layers 0 4 8` の 3 層だけを解析する
- 既定では `--dialog-types script` に限定する
- 実行開始時に出力先 `outputs/` の中身を削除し、最新実行の結果だけを残す

## MDS 方式

`wav2vec_iemocap_mds` と同じ A/B/C 方式です。

- A: 感情ラベルごとの平均ベクトル間距離
- B: `speaker_id × emotion_label` 平均のあとに感情平均
- C: 感情間の全発話ペア距離平均

## 実行例

```bash
cd /home/takamichi-lab-pc07/research/music_foundation_analysis
emotion2vec/.venv/bin/python emotion2vec/iemocap_mds/mds_iemocap_emotions.py
```

Hugging Face から初回ダウンロードを明示する場合:

```bash
emotion2vec/.venv/bin/python emotion2vec/iemocap_mds/mds_iemocap_emotions.py \
  --hub hf \
  --model-name emotion2vec/emotion2vec_base
```

解析層の変更:

```bash
emotion2vec/.venv/bin/python emotion2vec/iemocap_mds/mds_iemocap_emotions.py \
  --layers 0 1 2 3 4 5 6 7 8
```

話者ごと、テキストごとも実行:

```bash
emotion2vec/.venv/bin/python emotion2vec/iemocap_mds/mds_iemocap_emotions.py \
  --group-scopes all genders speakers texts
```

デバッグ用に感情ラベルをランダムにシャッフルして解析:

```bash
emotion2vec/.venv/bin/python emotion2vec/iemocap_mds/mds_iemocap_emotions.py \
  --shuffle-emotion-labels \
  --shuffle-label-seed 42
```

## ノルム分析

```bash
emotion2vec/.venv/bin/python emotion2vec/iemocap_mds/plot_iemocap_norms.py
```

ノルム分析側でも同じシャッフルオプションを受け付けます。主に `utterance_norms.csv` のデバッグ確認用です。

既定では `layer 4` を話者別・男女別に使い、`layer 0/4/8` の層別平均ノルムも出力します。

## 出力

- `emotion2vec/iemocap_mds/outputs`
- `emotion2vec/iemocap_mds/norm_outputs`

出力ファイル構造は `wav2vec_iemocap_mds` と揃えています。
