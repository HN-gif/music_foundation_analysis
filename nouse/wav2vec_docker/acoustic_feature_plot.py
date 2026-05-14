import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")

import librosa
import matplotlib.pyplot as plt
import numpy as np


EMOTION_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised",
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
        description=(
            "Intensity=02, Statement=01 にそろえた音声について、"
            "感情ごとの平均 F0 を可視化します。"
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/Audio_Speech_Actors_01-24"),
        help="RAVDESS 音声データセットのルートディレクトリ",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="先頭から使用する最大ファイル数。未指定なら条件に合う全件を使用。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mean_f0_by_emotion.png"),
        help="保存先画像ファイル名",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("mean_f0_by_emotion.csv"),
        help="感情ごとの平均 F0 集計を保存する CSV",
    )
    return parser.parse_args()


def parse_filename(audio_path: Path):
    parts = audio_path.stem.split("-")
    if len(parts) < 7:
        raise ValueError(f"Unexpected filename format: {audio_path.name}")
    return parts


def should_use_file(audio_path: Path):
    parts = parse_filename(audio_path)
    intensity = parts[3]
    statement = parts[4]
    return intensity == TARGET_INTENSITY and statement == TARGET_STATEMENT


def get_emotion_code(audio_path: Path):
    return parse_filename(audio_path)[2]


def collect_audio_files(data_dir: Path, max_files=None):
    audio_files = [path for path in sorted(data_dir.rglob("*.wav")) if should_use_file(path)]
    if max_files is not None:
        audio_files = audio_files[:max_files]
    if not audio_files:
        raise FileNotFoundError(
            "No wav files matched the requested condition under: "
            f"{data_dir} (Intensity={TARGET_INTENSITY}, Statement={TARGET_STATEMENT})"
        )
    return audio_files


def compute_mean_f0(audio_path: Path):
    speech, sr = librosa.load(str(audio_path), sr=16000)
    f0, voiced_flag, _ = librosa.pyin(
        speech,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
    )

    if voiced_flag is None:
        voiced_values = f0[np.isfinite(f0)]
    else:
        voiced_values = f0[np.isfinite(f0) & voiced_flag]

    if len(voiced_values) == 0:
        return np.nan
    return float(np.mean(voiced_values))


def summarize_mean_f0(audio_files):
    grouped_values = {emotion_code: [] for emotion_code in EMOTION_ORDER}

    for idx, audio_path in enumerate(audio_files, start=1):
        emotion_code = get_emotion_code(audio_path)
        if emotion_code not in EMOTION_MAP:
            continue

        grouped_values[emotion_code].append(compute_mean_f0(audio_path))

        if idx % 20 == 0:
            print(f"Processed {idx}/{len(audio_files)} files")

    summary = []
    for emotion_code in EMOTION_ORDER:
        values = np.array(grouped_values[emotion_code], dtype=float)
        valid_values = values[np.isfinite(values)]
        if len(valid_values) == 0:
            continue

        summary.append(
            {
                "emotion_code": emotion_code,
                "emotion_name": EMOTION_MAP[emotion_code],
                "count": int(len(valid_values)),
                "mean_f0_hz": float(np.mean(valid_values)),
                "std_f0_hz": float(np.std(valid_values)),
            }
        )

    if not summary:
        raise RuntimeError("No valid F0 values were extracted.")

    return summary


def save_summary_csv(summary, output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "emotion_code",
                "emotion_name",
                "count",
                "mean_f0_hz",
                "std_f0_hz",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)


def plot_mean_f0(summary, output_path: Path):
    names = [item["emotion_name"] for item in summary]
    means = [item["mean_f0_hz"] for item in summary]
    stds = [item["std_f0_hz"] for item in summary]
    counts = [item["count"] for item in summary]
    colors = [EMOTION_COLORS[item["emotion_code"]] for item in summary]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(names, means, yerr=stds, color=colors, alpha=0.88, capsize=4)

    for bar, mean_value, count in zip(bars, means, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean_value:.1f} Hz\nn={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title("Mean F0 by emotion")
    plt.ylabel("Mean F0 [Hz]")
    plt.xlabel("Emotion")
    plt.grid(axis="y", alpha=0.25)
    plt.figtext(
        0.5,
        0.01,
        f"Condition: Intensity={TARGET_INTENSITY}, Statement={TARGET_STATEMENT}",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(output_path, dpi=300)
    plt.show()


def main():
    args = parse_args()
    audio_files = collect_audio_files(args.data_dir, max_files=args.max_files)

    print(
        f"Extracting mean F0 from {len(audio_files)} files "
        f"(Intensity={TARGET_INTENSITY}, Statement={TARGET_STATEMENT})..."
    )

    summary = summarize_mean_f0(audio_files)
    save_summary_csv(summary, args.summary_csv)
    plot_mean_f0(summary, args.output)

    print(f"Saved plot to: {args.output}")
    print(f"Saved summary csv to: {args.summary_csv}")


if __name__ == "__main__":
    main()
