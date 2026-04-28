# wav2vec2 Middle-Layer UMAP

`facebook/wav2vec2-base-960h` の中間層 hidden state を使い、同一話者・同一文の `emotion - neutral` 差分ベクトルを各性別・各 statement ごとに UMAP で可視化します。

## 解析方針

- モデル: `facebook/wav2vec2-base-960h`
- 表現: hidden states のちょうど中間の層 `len(hidden_states) // 2`
- 発話ベクトル化: 中間層 hidden state を時間方向に平均
- ペア定義:
  - 同じ話者
  - 同じ statement
  - neutral (`emotion_code=01`) とそれ以外
  - `repetition` は無視
  - `intensity` はペア条件に使わない
- ペア表現:
  - 同一話者・同一 statement の neutral 発話群を平均して neutral 基準ベクトルを作る
  - 各 non-neutral 発話から neutral 基準ベクトルを引いた差分ベクトルを 1 ペアの表現とする
  - 例えば `Calm - Neutral` は `ΔCalm` という 1 つの属性として扱う
- 可視化:
  - 差分の計算は話者ごとに行う
  - その後、同じ性別・同じ statement の差分ベクトルをまとめて UMAP に入力する
  - 1つ目の UMAP では `ΔCalm`, `ΔHappy` などの差分属性ごとに点群を描く
  - 色は差分属性、マーカーは intensity に対応する
  - 各差分属性ごとに分布楕円と重心を重ねて表示する
  - 2つ目の UMAP では色分けを話者 ID ごとに切り替える
  - 話者 ID ごとの分布楕円と重心も重ねて表示する
  - さらに各話者・各 statement ごとにも `Δ感情` 色分けの UMAP を出力する

## neutral の扱い

RAVDESS 解析では neutral (`emotion_code=01`) は既定で使用しません。

ただし、この `wav2vec_rdm` は `emotion - neutral` 差分ベクトルを作る解析なので、neutral 基準ベクトルなしでは実行できません。そのため、解析を実行する場合は neutral を使う意思を明示するために `--include-neutral` が必須です。

## venv セットアップ

```bash
cd /home/takamichi-lab-pc07/research/wav2vec_rdm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 実行例

全話者を解析:

```bash
python rdm_speaker_midlayer.py --include-neutral
```

このスクリプトは実行時に自動で `research` ルートを import path に追加するので、`wav2vec_rdm` ディレクトリ内からそのまま実行できます。

女性話者だけを解析:

```bash
python rdm_speaker_midlayer.py --include-neutral --genders female
```

特定の話者だけを解析:

```bash
python rdm_speaker_midlayer.py --include-neutral --actor-ids 2 4 6
```

Statement を限定:

```bash
python rdm_speaker_midlayer.py --include-neutral --statement-codes 01
```

## 出力

既定の出力先は `research/wav2vec_rdm/outputs` です。

```text
outputs/
  actor_01/
    statement_01/
      umap_by_delta.png
      umap.npy
      metadata.json
    statement_02/
      ...
  gender_female/
    statement_01/
      umap_clusters.png
      umap_by_actor.png
      umap.npy
      metadata.json
    statement_02/
      ...
  gender_male/
    ...
  summary.json
```

`summary.json` には `by_gender` と `by_actor` の 2 系統の出力メタデータが入ります。
