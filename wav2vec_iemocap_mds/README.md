# wav2vec2 IEMOCAP Emotion MDS

IEMOCAP の離散感情ラベル付き発話音声に対して、wav2vec2 の各層 hidden state を抽出し、感情ラベル間距離行列を作成して MDS で 2 次元可視化します。

RAVDESS 用の `wav2vec_mds` と同じ A/B/C 方式を、IEMOCAP の `ang`, `hap`, `sad`, `neu`, `fru`, `exc` などのカテゴリ感情ラベルに適用します。

## 解析方式

- データセット: `data/IEMOCAP_full_release`
- 音声: `Session*/sentences/wav/*/*.wav`
- アノテーション: `Session*/dialog/EmoEvaluation/*.txt`
- `emotion_label=xxx` は既定で除外
- `emotion_label=oth` は常に除外
- 各発話から wav2vec2 hidden states を取得
- 発話ベクトルは時間方向 mean pooling
- 距離は既定で cosine distance
- `--distance-metric norm-only` ではベクトル方向を無視し、L2 ノルムの差 `| ||x||_2 - ||y||_2 |` を距離にする
- 既定では `--layers 0 6 12` の 3 層だけを解析する
- 既定では `--dialog-types script` に限定する
- 実行開始時に出力先 `outputs/` の中身を削除し、最新実行の結果だけを残す
- MDS の `random_state` は既定で `42`
- MDS 次元数は `--mds-components` で変更可能。既定は 2 次元

### A 方式

感情ラベルごとに発話ベクトルを平均して感情代表ベクトルを作り、代表ベクトル間距離を MDS に入力します。

### B 方式

まず `speaker_id × emotion_label` ごとに発話ベクトルを平均し、その後 `emotion_label` ごとに話者平均して感情代表ベクトルを作ります。話者ごとの発話数差の影響を抑える方式です。

### C 方式

感情ラベル間距離を、2 つの感情に属する発話ベクトル全組み合わせ距離の平均として定義します。データ量が多い場合は `--pair-sample-size` でペアサンプリングできます。

## グループ分け

`--group-scopes` で MDS を作る単位を選べます。

```text
all       全発話をまとめる
genders   male / female に分ける
speakers  Session1_F など、10 話者ごとに分ける
texts     script01_1 / impro01 など、発話内容・シナリオ ID ごとに分ける
```

既定では `all` と `genders` を実行します。男女別は既定で必ず作られます。

### 出力ディレクトリ名の意味

`outputs/` 配下には、選んだ `--group-scopes` に応じて解析単位ごとのディレクトリが作られます。

```text
all/
gender_female/
gender_male/
speaker_Session1_F/
speaker_Session1_M/
text_impro01/
text_script01_1/
```

`speaker_Session1_F` は、**Session 1 の female speaker だけ**を使った分析です。IEMOCAP は 5 セッションそれぞれに男女 1 名ずつ、合計 10 話者がいるため、話者 ID は以下のように作っています。

```text
Session1_F, Session1_M, ..., Session5_F, Session5_M
```

したがって `speaker_Session3_M` なら、Session 3 の male speaker の発話だけで感情代表ベクトル、距離行列、MDS を作っています。

`text_impro01` は、**improvisation 01 という同じ即興シナリオ ID**に属する発話だけを集めた分析です。IEMOCAP の元 ID は `Ses01F_impro01`, `Ses02M_impro01` のように session や marker 装着者情報を含みますが、このスクリプトでは先頭の `SesXXF` / `SesXXM` を落として `impro01` を `content_id` としています。つまり `text_impro01` は複数セッションにまたがる `impro01` 系の発話をまとめます。

`text_script01_1` は、**script 01_1 という同じ台本 ID**に属する発話だけを集めた分析です。`text_script02_1`, `text_script03_2` なども同様に、session をまたいで同じ script ID の発話をまとめます。

注意点として、`text_*` は「完全に同一の短い文」ではなく、IEMOCAP の dialog/scenario ID から作った **シナリオまたは台本ブロック単位**です。発話ごとの厳密な単語列で揃えたい場合は、transcription や forced alignment を使った別の text ID 設計が必要です。

## 実行例

既定実行。script のみ、層 0/6/12、全体 + 男女別、A/B/C:

```bash
cd /home/takamichi-lab-pc07/research/music_foundation_analysis
wav2vec_rdm/.venv/bin/python wav2vec_iemocap_mds/mds_iemocap_emotions.py
```

解析層を変更:

```bash
wav2vec_rdm/.venv/bin/python wav2vec_iemocap_mds/mds_iemocap_emotions.py --layers 0 3 6 9 12
```

3 次元 MDS:

```bash
wav2vec_rdm/.venv/bin/python wav2vec_iemocap_mds/mds_iemocap_emotions.py \
  --layers 0 6 12 \
  --mds-components 3
```

話者ごと、テキストごとも実行:

```bash
wav2vec_rdm/.venv/bin/python wav2vec_iemocap_mds/mds_iemocap_emotions.py \
  --layers 0 6 12 \
  --group-scopes all genders speakers texts
```

improvisation / script を限定:

```bash
wav2vec_rdm/.venv/bin/python wav2vec_iemocap_mds/mds_iemocap_emotions.py \
  --dialog-types impro script \
  --layers 0 6 12
```

ノルム差だけを距離にする:

```bash
wav2vec_rdm/.venv/bin/python wav2vec_iemocap_mds/mds_iemocap_emotions.py \
  --distance-metric norm-only \
  --layers 6 \
  --dialog-types script
```

デバッグ用:

```bash
wav2vec_rdm/.venv/bin/python wav2vec_iemocap_mds/mds_iemocap_emotions.py \
  --max-utterances 100 \
  --layers 0 \
  --skip-groups-with-fewer-emotions 2
```

## ノルム分析

話者性に応じた wav2vec2 ベクトルの L2 ノルム差を見る場合は、別スクリプト `plot_iemocap_norms.py` を使います。

既定では `script` 発話だけを使い、話者別・男女別の棒グラフは 6 層目、層別の棒グラフは 0 / 6 / 12 層で作成します。実行開始時に `norm_outputs/` の中身を削除し、最新実行の結果だけを残します。

```bash
cd /home/takamichi-lab-pc07/research/music_foundation_analysis
wav2vec_rdm/.venv/bin/python wav2vec_iemocap_mds/plot_iemocap_norms.py
```

主な出力先は `research/music_foundation_analysis/wav2vec_iemocap_mds/norm_outputs` です。

```text
norm_outputs/
  utterance_norms.csv
  speaker_mean_norms_layer_06.csv
  speaker_mean_norms_layer_06.png
  gender_mean_norms_layer_06.csv
  gender_mean_norms_layer_06.png
  layer_mean_norms.csv
  layer_mean_norms.png
```

`utterance_norms.csv` には発話ごとのノルム、`speaker_mean_norms_layer_06.csv` には各話者の平均ノルム、`gender_mean_norms_layer_06.csv` には男女別の平均ノルム、`layer_mean_norms.csv` には話者を区別しない層別平均ノルムを保存します。

## 出力

既定の出力先は `research/music_foundation_analysis/wav2vec_iemocap_mds/outputs` です。
スクリプトは実行開始時にこの出力先の中身を削除してから新しい結果を書き出します。

```text
outputs/
  utterance_metadata.csv
  group_sample_counts.csv
  summary_metrics.csv
  all/
    layer_00/
      method_A/
        distance_matrix.csv
        mds_coordinates.csv
        mds_plot.png
      method_B/
      method_C/
  gender_female/
  gender_male/
  speaker_Session1_F/
  text_script01_1/
```

`summary_metrics.csv` には group scope、group id、層、方式、stress、感情数、発話数、出力パスなどを保存します。

`group_sample_counts.csv` には、MDS を作る前の各 group のサンプル数を保存します。`text_script01_1` などで MDS の点が少ない場合に、そもそも発話数が少ないのか、感情ラベルの多様性が低いのかを確認するための表です。

主な列:

```text
group_scope       all / gender / speaker / text
group_id          all, female, Session1_F, script01_1 など
output_name       出力ディレクトリ名
utterance_count   その group に含まれる発話数
emotion_count     1件以上ある感情ラベル数
analyzed          MDS 分析対象になったか
skip_reason       スキップされた理由
label_counts      neu:10;hap:3;fru:8 のような要約
count_neu         neutral の発話数
count_hap         happy の発話数
count_exc         excited の発話数
...
```

例えば `text_script01_1` の `utterance_count` が大きいのに `emotion_count` が小さい場合は、サンプル数不足ではなく、感情ラベルの多様性が低い group だと判断できます。

`mds_coordinates.csv` は `--mds-components 3` の場合、`mds_1`, `mds_2`, `mds_3` を保存します。PNG は 3D プロットになります。4 次元以上を指定した場合、CSV には全次元を保存し、PNG は先頭 3 次元を描画します。
