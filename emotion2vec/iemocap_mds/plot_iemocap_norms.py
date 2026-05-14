#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emotion2vec.emotion2vec import DEFAULT_MODEL_NAME, extract_layer_embeddings, validate_layers
from wav2vec_iemocap_rsa.rsa_iemocap_layers import (
    DEFAULT_MIN_EMOTION_COUNT_EXCLUSIVE,
    IemocapUtterance,
    collect_iemocap_utterances,
)


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "IEMOCAP_full_release"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "emotion2vec" / "iemocap_mds" / "norm_outputs"
DEFAULT_DIALOG_TYPES = ("script",)
DEFAULT_LAYERS = (0, 4, 8)
DEFAULT_SPEAKER_LAYER = 4
EXCLUDED_EMOTION_LABELS = {"oth"}
DEFAULT_RANDOM_STATE = 42
EMOTION_NAMES = {
    "ang": "angry",
    "hap": "happy",
    "sad": "sad",
    "neu": "neutral",
    "fru": "frustrated",
    "exc": "excited",
    "fea": "fearful",
    "sur": "surprised",
    "dis": "disgusted",
    "oth": "other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract emotion2vec IEMOCAP utterance embeddings and plot L2 norm "
            "averages by speaker, gender, and layer."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--hub", default="hf", choices=["hf", "ms", "modelscope"])
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device. Example: cpu, cuda, cuda:0",
    )
    parser.add_argument("--dialog-types", nargs="+", choices=["impro", "script"], default=DEFAULT_DIALOG_TYPES)
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--speaker-layer", type=int, default=DEFAULT_SPEAKER_LAYER)
    parser.add_argument("--sessions", nargs="+", default=None)
    parser.add_argument("--emotion-labels", nargs="+", default=None)
    parser.add_argument("--include-xxx", action="store_true")
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--shuffle-emotion-labels",
        action="store_true",
        help="Ignore dataset emotion labels and randomly permute labels across utterances for debugging.",
    )
    parser.add_argument(
        "--shuffle-label-seed",
        type=int,
        default=None,
        help="Random seed for --shuffle-emotion-labels. Defaults to --random-state.",
    )
    return parser.parse_args()


def gender_name(item: IemocapUtterance) -> str:
    return "female" if item.speaker_gender == "F" else "male"


def clear_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def write_dict_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_filtered_utterances(args: argparse.Namespace) -> List[IemocapUtterance]:
    utterances = collect_iemocap_utterances(
        args.data_dir,
        args.sessions,
        args.emotion_labels,
        args.dialog_types,
        args.include_xxx,
        args.max_utterances,
        min_emotion_count_exclusive=DEFAULT_MIN_EMOTION_COUNT_EXCLUSIVE,
    )
    utterances = [utterance for utterance in utterances if utterance.emotion_label not in EXCLUDED_EMOTION_LABELS]
    if not utterances:
        raise RuntimeError("No IEMOCAP utterances remained after filtering.")
    return utterances


def maybe_shuffle_emotion_labels(
    utterances: Sequence[IemocapUtterance],
    enabled: bool,
    seed: int,
) -> List[IemocapUtterance]:
    if not enabled:
        return list(utterances)
    rng = np.random.default_rng(seed)
    shuffled_labels = [utterances[index].emotion_label for index in rng.permutation(len(utterances))]
    shuffled = []
    for utterance, shuffled_label in zip(utterances, shuffled_labels):
        shuffled.append(
            replace(
                utterance,
                emotion_label=shuffled_label,
                emotion_name=EMOTION_NAMES.get(shuffled_label, utterance.emotion_label),
            )
        )
    return shuffled


def build_norm_rows(
    utterances: Sequence[IemocapUtterance],
    model_name: str,
    device_name: str,
    hub: str,
    layers: Sequence[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    checked_layers: List[int] | None = None
    for index, utterance in enumerate(utterances, start=1):
        print(f"[{index}/{len(utterances)}] extracting hidden states: {utterance.utterance_id}")
        layer_vectors = extract_layer_embeddings(utterance.wav_path, model_name=model_name, device_name=device_name, hub=hub)
        if checked_layers is None:
            checked_layers = validate_layers(layers, len(layer_vectors))
        for layer_index in checked_layers:
            rows.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "wav_path": str(utterance.wav_path),
                    "session_id": utterance.session_id,
                    "dialog_id": utterance.dialog_id,
                    "dialog_type": utterance.dialog_type,
                    "speaker_id": utterance.speaker_id,
                    "speaker_gender": utterance.speaker_gender,
                    "gender": gender_name(utterance),
                    "emotion_label": utterance.emotion_label,
                    "layer": layer_index,
                    "norm": float(np.linalg.norm(layer_vectors[layer_index])),
                }
            )
    return rows


def summarize_norms(rows: Sequence[Dict[str, object]], group_key: str, label_key: str, layer: int | None = None):
    grouped: Dict[str, List[float]] = {}
    labels: Dict[str, str] = {}
    for row in rows:
        if layer is not None and int(row["layer"]) != layer:
            continue
        key = str(row[group_key])
        grouped.setdefault(key, []).append(float(row["norm"]))
        labels[key] = str(row[label_key])
    summary_rows = []
    for key in sorted(grouped):
        values = np.array(grouped[key], dtype=np.float64)
        summary_rows.append(
            {
                group_key: key,
                label_key: labels[key],
                "layer": layer if layer is not None else "all",
                "utterance_count": int(values.size),
                "mean_norm": float(values.mean()),
                "std_norm": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }
        )
    return summary_rows


def summarize_layers(rows: Sequence[Dict[str, object]], layers: Sequence[int]):
    summary_rows = []
    for layer in layers:
        values = np.array([float(row["norm"]) for row in rows if int(row["layer"]) == layer], dtype=np.float64)
        if values.size == 0:
            continue
        summary_rows.append(
            {
                "layer": layer,
                "utterance_count": int(values.size),
                "mean_norm": float(values.mean()),
                "std_norm": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }
        )
    return summary_rows


def plot_bar(rows: Sequence[Dict[str, object]], label_field: str, title: str, output_path: Path, color: str) -> None:
    labels = [str(row[label_field]) for row in rows]
    means = [float(row["mean_norm"]) for row in rows]
    counts = [int(row["utterance_count"]) for row in rows]
    fig_width = max(7.0, 0.7 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    x_positions = np.arange(len(labels))
    bars = ax.bar(x_positions, means, color=color, edgecolor="#222222", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Mean L2 norm")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35 if len(labels) > 4 else 0, ha="right")
    ax.grid(axis="y", alpha=0.25)
    top = max(means) if means else 0.0
    ax.set_ylim(0, top * 1.18 if top > 0 else 1.0)
    for bar, mean, count in zip(bars, means, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mean:.3f}\nn={count}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    clear_output_dir(args.output_dir)
    shuffle_seed = args.random_state if args.shuffle_label_seed is None else args.shuffle_label_seed
    requested_layers = sorted(set(args.layers) | {args.speaker_layer})
    utterances = collect_filtered_utterances(args)
    utterances = maybe_shuffle_emotion_labels(utterances, args.shuffle_emotion_labels, shuffle_seed)
    print(f"Collected {len(utterances)} IEMOCAP utterances")
    if args.shuffle_emotion_labels:
        print(f"Emotion labels were shuffled across utterances with seed={shuffle_seed}")
    norm_rows = build_norm_rows(utterances, args.model_name, args.device, args.hub, requested_layers)
    available_layers = sorted({int(row["layer"]) for row in norm_rows})
    if args.speaker_layer not in available_layers:
        raise RuntimeError(f"Layer {args.speaker_layer} was not extracted.")
    write_dict_rows(
        args.output_dir / "utterance_norms.csv",
        norm_rows,
        ["utterance_id", "wav_path", "session_id", "dialog_id", "dialog_type", "speaker_id", "speaker_gender", "gender", "emotion_label", "layer", "norm"],
    )
    speaker_rows = summarize_norms(norm_rows, "speaker_id", "gender", layer=args.speaker_layer)
    gender_rows = summarize_norms(norm_rows, "gender", "gender", layer=args.speaker_layer)
    layer_rows = summarize_layers(norm_rows, args.layers)
    write_dict_rows(args.output_dir / f"speaker_mean_norms_layer_{args.speaker_layer:02d}.csv", speaker_rows, ["speaker_id", "gender", "layer", "utterance_count", "mean_norm", "std_norm"])
    write_dict_rows(args.output_dir / f"gender_mean_norms_layer_{args.speaker_layer:02d}.csv", gender_rows, ["gender", "layer", "utterance_count", "mean_norm", "std_norm"])
    write_dict_rows(args.output_dir / "layer_mean_norms.csv", layer_rows, ["layer", "utterance_count", "mean_norm", "std_norm"])
    plot_bar(speaker_rows, "speaker_id", f"IEMOCAP mean emotion2vec L2 norm by speaker (layer {args.speaker_layer:02d})", args.output_dir / f"speaker_mean_norms_layer_{args.speaker_layer:02d}.png", "#4C78A8")
    plot_bar(gender_rows, "gender", f"IEMOCAP mean emotion2vec L2 norm by gender (layer {args.speaker_layer:02d})", args.output_dir / f"gender_mean_norms_layer_{args.speaker_layer:02d}.png", "#F58518")
    plot_bar(layer_rows, "layer", "IEMOCAP mean emotion2vec L2 norm by layer", args.output_dir / "layer_mean_norms.png", "#54A24B")
    print(f"Saved IEMOCAP norm analysis outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
