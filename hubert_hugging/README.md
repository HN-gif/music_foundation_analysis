# HuBERT Internal Representation Extraction

`research/data` 配下の音声を Hugging Face の HuBERT モデルへ入力し、内部表現を取り出して保存するためのスクリプトです。

RAVDESS のファイル名解析とデータ選別ロジックは、`wav2vec_docker` の外に切り出した共有モジュール [audio_dataset/ravdess.py](/home/takamichi-lab-pc07/research/audio_dataset/ravdess.py) にまとめています。今後ほかの音声モデルでも同じ条件で入力データをそろえられます。

対応モデル:

- `facebook/hubert-base-ls960`
- `facebook/hubert-large-ll60k`
- `facebook/hubert-large-ls960-ft`
- `facebook/hubert-xlarge-ls960-ft`

補足:

- 依頼文の `163` は Hugging Face 上で対応モデルを確認できなかったため、この実装では alias として `facebook/hubert-xlarge-ls960-ft` に割り当てています。
- HuBERT は 16kHz 入力を想定するため、入力音声はスクリプト内で 16kHz にリサンプリングされます。

## venv セットアップ

```bash
cd /home/takamichi-lab-pc07/research/hubert_hugging
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

以後は、このディレクトリで作業する前に毎回次を実行してください。

```bash
cd /home/takamichi-lab-pc07/research/hubert_hugging
source .venv/bin/activate
```

依存パッケージのバージョンは `requirements.txt` で管理しています。別環境を作り直したい場合は、同じ `requirements.txt` を使って `.venv` を再作成すれば同じ系列のバージョン構成を再現できます。

## 実行例

4 モデルすべてで `data` 配下を処理:

```bash
python extract_hubert_representations.py --model all
```

最初の 3 ファイルだけを `hubert-base-ls960` で処理:

```bash
python extract_hubert_representations.py --model base-ls960 --limit 3
```

`wav2vec_docker` と同じように、強い感情表現 `02`、文 `01`、女性話者だけに限定:

```bash
python extract_hubert_representations.py \
  --model base-ls960 \
  --intensity-codes 02 \
  --statement-codes 01 \
  --genders female
```

`163` alias を使って extra large の fine-tuned モデルを実行:

```bash
python extract_hubert_representations.py --model 163 --limit 1
```

全 hidden states と CTC logits も保存:

```bash
python extract_hubert_representations.py \
  --model large-ls960-ft \
  --limit 1 \
  --save-all-hidden-states \
  --save-logits
```

## 出力

出力先の既定値は `research/hubert_hugging/outputs` です。入力のディレクトリ構造を保ったまま、モデルごとに `.npz` を保存します。

例:

```text
outputs/
  base-ls960/
    Audio_Speech_Actors_01-24/Actor_21/03-01-07-02-01-02-21.npz
    manifest.jsonl
  large-ll60k/
  large-ls960-ft/
  xlarge-ls960-ft/
```

各 `.npz` には次の配列が入ります:

- `last_hidden_state`: フレームごとの最終層表現 `[time, hidden_size]`
- `utterance_embedding`: `last_hidden_state` の時間平均 `[hidden_size]`
- `extract_features`: モデルが返す中間特徴 `[time, conv_dim]`。返らないモデルでは保存されません
- `hidden_states`: `--save-all-hidden-states` 指定時のみ保存 `[num_layers_plus_input, time, hidden_size]`
- `logits`: `--save-logits` 指定時のみ保存。CTC fine-tuned モデルのみ

`manifest.jsonl` には各音声のメタデータ、保存先、テンソル形状、RAVDESS の感情ラベル情報が 1 行 1 JSON で保存されます。

## アルゴリズム概要

1. `data` 配下の `.wav` を再帰的に列挙する
2. 共有モジュールで話者性別、感情強度、発話文、反復回数、感情コード、話者 ID によるフィルタを適用する
3. `soundfile` で音声を読み込み、必要ならモノラル化する
4. 16kHz でない場合は線形補間で 16kHz にリサンプリングする
5. Hugging Face の `AutoFeatureExtractor` でモデル入力へ変換する
6. HuBERT モデルを `output_hidden_states=True` で推論する
7. 最終層表現、平均プーリング埋め込み、任意で全 hidden states / logits を `.npz` に保存する
8. 形状や感情ラベルなどを `manifest.jsonl` に保存する
