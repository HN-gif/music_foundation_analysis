import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

try:
    import umap
except ImportError as exc:
    raise ImportError(
        "umap-learn is not installed. Please install it with "
        "`pip install umap-learn` before running this script."
    ) from exc

from wav2vec import extract_mean_pooled_embedding


EMOTION_MAP = {
    "01": ("Neutral", "中立"),
    "02": ("Calm", "冷静"),
    "03": ("Happy", "幸せ"),
    "04": ("Sad", "悲しみ"),
    "05": ("Angry", "怒り"),
    "06": ("Fearful", "恐怖"),
    "07": ("Disgust", "嫌悪"),
    "08": ("Surprised", "驚き"),
}

EMOTION_ORDER = list(EMOTION_MAP.keys())
EMOTION_COLORS = {
    "01": "#4C78A8",
    "02": "#72B7B2",
    "03": "#F2CF5B",
    "04": "#9C755F",
    "05": "#E45756",
    "06": "#B279A2",
    "07": "#54A24B",
    "08": "#FF9DA6",
}

TARGET_INTENSITY = "02"
TARGET_STATEMENT = "01"


def parse_args():
    parser = argparse.ArgumentParser(
        description="wav2vec2 の中間層特徴を UMAP で可視化します。"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/Audio_Speech_Actors_01-24"),
        help="RAVDESS 音声データセットのルートディレクトリ",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=None,
        help="可視化する hidden_states の層番号。0 は特徴抽出器出力、1-12 は Transformer 層。",
    )
    parser.add_argument(
        "--layer-indices",
        type=int,
        nargs="+",
        default=None,
        help="複数の層番号をまとめて可視化する場合に指定します。",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="先頭から使用する最大ファイル数。未指定なら全件を使用。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("umap_emotion_plot.png"),
        help="保存先画像ファイル名",
    )
    return parser.parse_args()


def get_emotion_code(audio_path: Path):
    parts = audio_path.stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {audio_path.name}")
    return parts[2]


def should_use_file(audio_path: Path):
    parts = audio_path.stem.split("-")
    if len(parts) < 7:
        raise ValueError(f"Unexpected filename format: {audio_path.name}")

    intensity = parts[3]
    statement = parts[4]
    return intensity == TARGET_INTENSITY and statement == TARGET_STATEMENT


def get_actor_id(audio_path: Path):
    parts = audio_path.stem.split("-")
    if len(parts) < 7:
        raise ValueError(f"Unexpected filename format: {audio_path.name}")
    return parts[6]


def get_gender_label(audio_path: Path):
    actor_id = int(get_actor_id(audio_path))
    return "female" if actor_id % 2 == 0 else "male"


def collect_audio_files(data_dir: Path):
    audio_files = sorted(data_dir.rglob("*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No wav files found under: {data_dir}")
    return audio_files


def build_dataset(audio_files, layer_index):
    all_embeddings = []
    emotion_codes = []
    gender_labels = []

    for idx, audio_path in enumerate(audio_files, start=1):
        if not should_use_file(audio_path):
            continue

        emotion_code = get_emotion_code(audio_path)
        if emotion_code not in EMOTION_MAP:
            continue

        embedding = extract_mean_pooled_embedding(audio_path, layer_index=layer_index)
        all_embeddings.append(embedding)
        emotion_codes.append(emotion_code)
        gender_labels.append(get_gender_label(audio_path))

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(audio_files)} files")

    if not all_embeddings:
        raise RuntimeError("No embeddings were extracted.")

    return np.vstack(all_embeddings), np.array(emotion_codes), np.array(gender_labels)


def resolve_layer_indices(args):
    if args.layer_indices:
        return args.layer_indices
    if args.layer_index is not None:
        return [args.layer_index]
    return [6]


def add_confidence_ellipse(points, color):
    if len(points) < 3:
        return

    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # 2 standard deviations gives a readable summary of each cluster spread.
    width, height = 2 * 2 * np.sqrt(np.maximum(eigenvalues, 0))
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    center = points.mean(axis=0)

    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.12,
        linewidth=2,
    )
    plt.gca().add_patch(ellipse)


def plot_umap(embedding_2d, emotion_codes, layer_index, output_path: Path, gender_label: str):
    plt.figure(figsize=(10, 8))

    for emotion_code in EMOTION_ORDER:
        mask = emotion_codes == emotion_code
        if not np.any(mask):
            continue

        emotion_points = embedding_2d[mask]
        emotion_en, _ = EMOTION_MAP[emotion_code]
        plt.scatter(
            emotion_points[:, 0],
            emotion_points[:, 1],
            s=18,
            alpha=0.55,
            color=EMOTION_COLORS[emotion_code],
            label=f"{emotion_code}: {emotion_en}",
        )

        add_confidence_ellipse(emotion_points, EMOTION_COLORS[emotion_code])

        centroid = emotion_points.mean(axis=0)
        plt.scatter(
            centroid[0],
            centroid[1],
            s=220,
            color=EMOTION_COLORS[emotion_code],
            edgecolor="black",
            linewidth=1.2,
            marker="X",
            zorder=5,
        )
        plt.text(
            centroid[0],
            centroid[1],
            f" {emotion_en}",
            fontsize=9,
            weight="bold",
            va="center",
            ha="left",
        )

    plt.title(f"wav2vec2 hidden state layer {layer_index} UMAP ({gender_label})")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def main():
    args = parse_args()
    audio_files = collect_audio_files(args.data_dir)
    layer_indices = resolve_layer_indices(args)

    if args.max_files is not None:
        audio_files = audio_files[: args.max_files]

    output_stem = args.output.stem
    output_suffix = args.output.suffix or ".png"

    for layer_index in layer_indices:
        print(
            f"Extracting wav2vec2 layer {layer_index} embeddings from "
            f"{len(audio_files)} files "
            f"(Intensity={TARGET_INTENSITY}, Statement={TARGET_STATEMENT})..."
        )
        all_embeddings, emotion_codes, gender_labels = build_dataset(
            audio_files=audio_files,
            layer_index=layer_index,
        )
        unique_gender_labels = sorted(np.unique(gender_labels))

        for gender_label in unique_gender_labels:
            gender_mask = gender_labels == gender_label
            gender_embeddings = all_embeddings[gender_mask]
            gender_emotions = emotion_codes[gender_mask]

            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2,
                random_state=42,
            )
            embedding_2d = reducer.fit_transform(gender_embeddings)

            base_name = output_stem if len(layer_indices) == 1 else f"{output_stem}_layer{layer_index}"
            output_path = args.output.with_name(
                f"{base_name}_{gender_label}{output_suffix}"
            )

            plot_umap(
                embedding_2d,
                gender_emotions,
                layer_index,
                output_path,
                gender_label,
            )
            print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
