# wav2vec2 Layer-wise Emotion MDS

RAVDESS の離散感情ラベル付き音声に対して、`facebook/wav2vec2-base-960h` の各層 hidden state を抽出し、感情ラベル間距離行列を作成して MDS で 2 次元可視化します。

## 解析方式

- 各発話から全層 hidden states を取得
- 各層の発話ベクトルは時間方向 mean pooling
- 距離は既定で cosine distance
- `--distance-metric norm-only` ではベクトル方向を無視し、L2 ノルムの差 `| ||x||_2 - ||y||_2 |` を距離にする
- `speaker_id` と `emotion_label` は必須メタデータとして扱う
- `text_id` は RAVDESS の `statement_code` として保持
- neutral (`emotion_code=01`) は既定で除外
- neutral を含める場合のみ `--include-neutral` を指定する
- 欠損行は除外
- `random_state` は既定で `42`

### A 方式

感情ラベルごとに発話ベクトルを平均して感情代表ベクトルを作り、代表ベクトル間の距離行列を MDS に入力します。

### B 方式

まず `speaker_id × emotion_label` ごとに発話ベクトルを平均し、その後 `emotion_label` ごとに話者平均して感情代表ベクトルを作ります。話者ごとの発話数差の影響を抑える方式です。

### C 方式

感情ラベル間距離を、2 つの感情に属する発話ベクトル全組み合わせ距離の平均として定義します。同一感情内距離も異なる発話ペアの平均で定義します。データ量が多い場合は `--pair-sample-size` でペアサンプリングできます。

## セットアップ

```bash
cd /home/takamichi-lab-pc07/research/music_foundation_analysis/wav2vec_mds
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 実行例

全データ、全層、A/B/C を解析:

```bash
python mds_emotion_layers.py
```

この既定実行では neutral (`emotion_code=01`) は使いません。

neutral も含めて解析:

```bash
python mds_emotion_layers.py --include-neutral
```

方式を限定:

```bash
python mds_emotion_layers.py --methods A C
```

層を限定:

```bash
python mds_emotion_layers.py --layers 0 6 12
```

C 方式の組み合わせ距離をサンプリング:

```bash
python mds_emotion_layers.py --pair-sample-size 5000
```

ノルム差だけを距離にする:

```bash
python mds_emotion_layers.py --distance-metric norm-only
```

デバッグ用にファイル数を限定:

```bash
python mds_emotion_layers.py --max-files 40 --layers 0 6 12
```

## 出力

既定の出力先は `research/music_foundation_analysis/wav2vec_mds/outputs` です。

```text
outputs/
  group_sample_counts.csv
  utterance_metadata.csv
  summary_metrics.csv
  layer_00/
    method_A/
      distance_matrix.csv
      mds_coordinates.csv
      mds_plot.png
    method_B/
      ...
    method_C/
      ...
  layer_01/
    ...
```

`summary_metrics.csv` には層、方式、stress、感情数、発話数、距離計算に使ったサンプル数などを保存します。

`group_sample_counts.csv` には、MDS を作る前の各 group のサンプル数を保存します。RAVDESS 版では group は `gender × statement(text_id)` です。

主な列:

```text
gender             female / male
text_id            statement code。01 または 02
output_name        female_text01 などの出力ディレクトリ名
utterance_count    その group に含まれる発話数
emotion_count      1件以上ある感情ラベル数
analyzed           MDS 分析対象になったか
skip_reason        スキップされた理由
label_counts       02_calm:8;03_happy:8 のような要約
count_01_neutral   neutral の発話数
count_02_calm      calm の発話数
count_03_happy     happy の発話数
...
```

neutral は既定で除外されるため、通常 `count_01_neutral` は `0` です。`--include-neutral` を指定した場合のみ neutral の件数が入ります。
