# Docker実行手順書

この手順書は、`research_2026` を VS Code 内で Docker を使って実行するためのものです。  
ホスト側の Docker / NVIDIA 設定は [docker_command_history.txt](/home/takamichi-lab-pc07/research_2026/docker_command_history.txt:1) を参照してください。

## 1. 前提条件

- Ubuntu 上で Docker がインストール済みである
- `docker` コマンドを `sudo` なしで実行できる
- NVIDIA GPU を使う場合は `nvidia-container-toolkit` の設定が済んでいる
- VS Code に `Dev Containers` 拡張が入っている

## 2. ホスト側の確認

ターミナルで次を実行します。

```bash
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

`nvidia-smi` の結果が表示されれば、GPU コンテナは利用可能です。

## 3. VS Codeでプロジェクトを開く

1. VS Code で `/home/takamichi-lab-pc07/research_2026` を開きます。
2. コマンドパレットを開きます。(Ctrl + SHIft + P)
3. `Dev Containers: Reopen in Container` を実行します。

初回は次のファイルを使ってイメージがビルドされます。

- [.devcontainer/devcontainer.json](/home/takamichi-lab-pc07/research_2026/.devcontainer/devcontainer.json:1)
- [.devcontainer/Dockerfile](/home/takamichi-lab-pc07/research_2026/.devcontainer/Dockerfile:1)
- [requirements.txt](/home/takamichi-lab-pc07/research_2026/requirements.txt:1)

## 4. コンテナ起動後の確認

VS Code のターミナルはコンテナ内になります。次を実行してください。

```bash
python3 --version
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import umap, librosa, transformers; print('imports ok')"
```

`torch.cuda.is_available()` が `True` なら GPU を使える状態です。  
`False` でも CPU 実行は可能です。

## 5. UMAPスクリプトの実行

少量データで確認する場合:

```bash
python3 U-MAP.py --max-files 50 --output umap_test.png
```

全データで実行する場合:

```bash
python3 U-MAP.py --output umap_emotion_plot.png
```

層番号を指定する場合:

```bash
python3 U-MAP.py --layer-index 6 --output umap_layer6.png
python3 U-MAP.py --layer-index 12 --output umap_layer12.png
```

## 6. スクリプトの仕様

- 入力データは `data/Audio_Speech_Actors_01-24` を使います
- 感情ラベルはファイル名の3つ目の数字を使います
- 感情対応表は [data/readme.md](/home/takamichi-lab-pc07/research_2026/data/readme.md:1) に基づきます
- `wav2vec2` の hidden state を抽出し、時間方向平均で 1 音声 1 ベクトルにしています
- UMAP 上では感情ごとに色分けされます

## 7. 出力ファイル

- 既定の出力画像: `umap_emotion_plot.png`
- 任意の名前は `--output` で指定できます

画像はワークスペース直下に保存されるため、VS Code のエクスプローラーからそのまま確認できます。

## 8. よくある問題

`Dev Containers: Reopen in Container` が失敗する場合:

- Docker Desktop ではなく Docker Engine が起動しているか確認してください
- `docker ps` がホスト側で実行できるか確認してください

GPU が使えない場合:

- ホスト側で `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi` を再確認してください
- `docker_command_history.txt` にある `nvidia-container-toolkit` の設定が済んでいるか確認してください

モデルのダウンロードで時間がかかる場合:

- 初回実行時に Hugging Face のモデル取得が走るため少し待ちます

## 9. 再ビルドが必要なケース

次の変更をした場合は、VS Code で `Dev Containers: Rebuild and Reopen in Container` を実行してください。

- `.devcontainer/Dockerfile` を変更した
- `requirements.txt` を変更した
- ベースイメージや GPU 設定を変更した
