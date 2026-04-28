#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import librosa
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from transformers import Wav2Vec2Model, Wav2Vec2Processor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_dataset import EMOTION_MAP, RAVDESSRecord, collect_ravdess_files


DEFAULT_MODEL_NAME = "facebook/wav2vec2-base-960h"
DEFAULT_OUTPUT_DIR = Path("/home/takamichi-lab-pc07/research/wav2vec_rdm/outputs")
TARGET_SAMPLE_RATE = 16_000
NEUTRAL_EMOTION_CODE = "01"
EMOTION_COLORS = {
    "02": "#72B7B2",
    "03": "#F2CF5B",
    "04": "#9C755F",
    "05": "#E45756",
    "06": "#B279A2",
    "07": "#54A24B",
    "08": "#FF9DA6",
}
INTENSITY_MARKERS = {
    "01": "o",
    "02": "^",
}


@dataclass(frozen=True)
class PairRecord:
    actor_id: int
    gender: str
    statement_code: str
    neutral_reference_count: int
    emotion_code: str
    intensity_code: str
    source_path: str
    label: str
    vector: np.ndarray

    @property
    def delta_label(self) -> str:
        emotion_name = EMOTION_MAP.get(self.emotion_code, {}).get("en", "unknown")
        return f"Δ{emotion_name.capitalize()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "wav2vec2-base-960h の中間層 hidden state から "
            "emotion-minus-neutral 差分ベクトルを作成し、"
            "各性別・各 statement ごとに UMAP を可視化します。"
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/takamichi-lab-pc07/research/data"),
        help="RAVDESS wav files are searched recursively under this directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures and metadata are saved.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face wav2vec2 checkpoint to use.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device. Example: cpu, cuda, cuda:0",
    )
    parser.add_argument(
        "--actor-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of actor IDs to analyze.",
    )
    parser.add_argument(
        "--genders",
        nargs="+",
        choices=["male", "female"],
        default=None,
        help="Optional subset of speaker genders.",
    )
    parser.add_argument(
        "--statement-codes",
        nargs="+",
        default=None,
        help="Optional subset of statement codes. Example: 01 02",
    )
    parser.add_argument(
        "--include-neutral",
        action="store_true",
        help=(
            "Allow neutral utterances to be used as the baseline. "
            "This script requires neutral data, so it only runs when this flag is set."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of wav files to inspect after filtering.",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=5,
        help="Base n_neighbors value for UMAP. Automatically clipped by sample count.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter.",
    )
    return parser.parse_args()


@lru_cache(maxsize=2)
def load_model(model_name: str, device_name: str):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    device = torch.device(device_name)
    model.to(device)
    model.eval()
    return processor, model, device


def load_waveform(audio_path: Path) -> np.ndarray:
    waveform, _ = librosa.load(str(audio_path), sr=TARGET_SAMPLE_RATE)
    return waveform.astype(np.float32, copy=False)


def extract_middle_layer_embedding(
    audio_path: Path,
    model_name: str,
    device_name: str,
) -> Tuple[np.ndarray, int]:
    processor, model, device = load_model(model_name, device_name)
    waveform = load_waveform(audio_path)

    inputs = processor(
        waveform,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    model_inputs = {
        "input_values": inputs.input_values.to(device),
        "output_hidden_states": True,
    }

    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        model_inputs["attention_mask"] = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(**model_inputs)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("wav2vec2 model did not return hidden states.")

    middle_index = len(hidden_states) // 2
    embedding = hidden_states[middle_index][0].mean(dim=0).detach().cpu().numpy()
    return embedding.astype(np.float32), middle_index


def collect_records(args: argparse.Namespace) -> List[RAVDESSRecord]:
    return collect_ravdess_files(
        args.data_dir,
        statement_codes=args.statement_codes,
        genders=args.genders,
        actor_ids=args.actor_ids,
        max_files=args.max_files,
    )


def build_pair_records(
    records: Iterable[RAVDESSRecord],
    model_name: str,
    device_name: str,
) -> Tuple[Dict[Tuple[int, str], List[PairRecord]], int]:
    grouped: Dict[Tuple[int, str], List[RAVDESSRecord]] = defaultdict(list)
    group_pairs: Dict[Tuple[int, str], List[PairRecord]] = defaultdict(list)
    embedding_cache: Dict[Path, np.ndarray] = {}
    resolved_middle_index: Optional[int] = None

    for record in records:
        grouped[(record.actor_id, record.statement_code)].append(record)

    for (actor_id, statement_code), statement_records in sorted(grouped.items()):
        neutral_records = [record for record in statement_records if record.emotion_code == NEUTRAL_EMOTION_CODE]
        emotional_records = [record for record in statement_records if record.emotion_code != NEUTRAL_EMOTION_CODE]

        if not neutral_records or not emotional_records:
            continue

        neutral_embeddings = []
        for record in neutral_records:
            if record.path not in embedding_cache:
                embedding_cache[record.path], resolved_middle_index = extract_middle_layer_embedding(
                    record.path, model_name, device_name
                )
            neutral_embeddings.append(embedding_cache[record.path])

        neutral_reference = np.mean(np.stack(neutral_embeddings, axis=0), axis=0).astype(np.float32)

        for record in emotional_records:
            if record.path not in embedding_cache:
                embedding_cache[record.path], resolved_middle_index = extract_middle_layer_embedding(
                    record.path, model_name, device_name
                )
            pair_vector = embedding_cache[record.path] - neutral_reference
            emotion_name = EMOTION_MAP.get(record.emotion_code, {}).get("en", "unknown")
            label = f"emo{record.emotion_code}_{emotion_name}_int{record.intensity_code}"
            group_pairs[(actor_id, statement_code)].append(
                PairRecord(
                    actor_id=actor_id,
                    gender=record.gender,
                    statement_code=statement_code,
                    neutral_reference_count=len(neutral_records),
                    emotion_code=record.emotion_code,
                    intensity_code=record.intensity_code,
                    source_path=str(record.path),
                    label=label,
                    vector=pair_vector.astype(np.float32),
                )
            )

    if resolved_middle_index is None:
        raise RuntimeError("No valid neutral-vs-emotional pairs could be built.")

    return group_pairs, resolved_middle_index


def regroup_by_gender_and_statement(
    actor_statement_pairs: Dict[Tuple[int, str], List[PairRecord]]
) -> Dict[Tuple[str, str], List[PairRecord]]:
    grouped: Dict[Tuple[str, str], List[PairRecord]] = defaultdict(list)
    for (_, statement_code), pair_records in actor_statement_pairs.items():
        for record in pair_records:
            grouped[(record.gender, statement_code)].append(record)
    return grouped


def filter_groups_with_min_pairs(
    groups: Dict[Tuple[object, str], List[PairRecord]],
    minimum_pairs: int,
    group_name: str,
) -> Dict[Tuple[object, str], List[PairRecord]]:
    filtered: Dict[Tuple[object, str], List[PairRecord]] = {}
    for (group_id, statement_code), pair_records in sorted(groups.items()):
        if len(pair_records) < minimum_pairs:
            print(
                f"Skipping {group_name} {group_id} statement {statement_code}: "
                f"need at least {minimum_pairs} emotion-neutral pairs for UMAP, got {len(pair_records)}"
            )
            continue
        filtered[(group_id, statement_code)] = pair_records
    return filtered


def compute_umap_projection(
    vectors: np.ndarray,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    sample_count = vectors.shape[0]
    adjusted_neighbors = min(max(2, n_neighbors), max(2, sample_count - 1))
    reducer = umap.UMAP(
        n_components=2,
        metric="euclidean",
        n_neighbors=adjusted_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    return reducer.fit_transform(vectors)


def add_legends(ax, pair_records: List[PairRecord]) -> None:
    emotion_codes = sorted({record.emotion_code for record in pair_records})
    delta_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="w",
            markerfacecolor=EMOTION_COLORS.get(emotion_code, "#888888"),
            markeredgecolor="black",
            markersize=9,
            label=f"Δ{EMOTION_MAP.get(emotion_code, {}).get('en', 'unknown').capitalize()}",
        )
        for emotion_code in emotion_codes
    ]
    intensity_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            color="black",
            markerfacecolor="white",
            markersize=9,
            label=f"intensity {code}",
        )
        for code, marker in INTENSITY_MARKERS.items()
    ]
    legend_delta = ax.legend(handles=delta_handles, title="Delta Attribute", loc="upper left")
    ax.add_artist(legend_delta)
    ax.legend(handles=intensity_handles, title="Intensity", loc="lower right")


def add_actor_legends(ax, pair_records: List[PairRecord], actor_colors: Dict[int, Tuple[float, float, float, float]]) -> None:
    actor_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="w",
            markerfacecolor=actor_colors[actor_id],
            markeredgecolor="black",
            markersize=9,
            label=f"actor {actor_id:02d}",
        )
        for actor_id in sorted(actor_colors)
    ]
    intensity_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            color="black",
            markerfacecolor="white",
            markersize=9,
            label=f"intensity {code}",
        )
        for code, marker in INTENSITY_MARKERS.items()
    ]
    legend_actor = ax.legend(handles=actor_handles, title="Actor", loc="upper left")
    ax.add_artist(legend_actor)
    ax.legend(handles=intensity_handles, title="Intensity", loc="lower right")


def add_confidence_ellipse(ax, points: np.ndarray, color: str) -> None:
    if len(points) < 3:
        return

    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

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
        alpha=0.14,
        linewidth=2,
    )
    ax.add_patch(ellipse)


def plot_emotion_ellipses_and_centroids(
    ax,
    umap_points: np.ndarray,
    pair_records: List[PairRecord],
) -> None:
    emotion_codes = sorted({record.emotion_code for record in pair_records})
    for emotion_code in emotion_codes:
        mask = np.array([record.emotion_code == emotion_code for record in pair_records], dtype=bool)
        emotion_points = umap_points[mask]
        color = EMOTION_COLORS.get(emotion_code, "#888888")

        add_confidence_ellipse(ax, emotion_points, color)

        centroid = emotion_points.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=260,
            color=color,
            edgecolor="black",
            linewidth=1.4,
            marker="X",
            zorder=5,
        )
        ax.text(
            centroid[0] + 0.04,
            centroid[1] + 0.04,
            f"{pair_records[np.where(mask)[0][0]].delta_label} centroid",
            fontsize=9,
            weight="bold",
            color=color,
        )


def plot_actor_ellipses_and_centroids(
    ax,
    umap_points: np.ndarray,
    pair_records: List[PairRecord],
    actor_colors: Dict[int, Tuple[float, float, float, float]],
) -> None:
    actor_ids = sorted({record.actor_id for record in pair_records})
    for actor_id in actor_ids:
        mask = np.array([record.actor_id == actor_id for record in pair_records], dtype=bool)
        actor_points = umap_points[mask]
        color = actor_colors[actor_id]

        add_confidence_ellipse(ax, actor_points, color)

        centroid = actor_points.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=260,
            color=color,
            edgecolor="black",
            linewidth=1.4,
            marker="X",
            zorder=5,
        )
        ax.text(
            centroid[0] + 0.04,
            centroid[1] + 0.04,
            f"actor {actor_id:02d} centroid",
            fontsize=9,
            weight="bold",
            color=color,
        )


def build_actor_colors(pair_records: List[PairRecord]) -> Dict[int, Tuple[float, float, float, float]]:
    actor_ids = sorted({record.actor_id for record in pair_records})
    cmap = plt.cm.get_cmap("tab10", max(len(actor_ids), 1))
    return {actor_id: cmap(index % cmap.N) for index, actor_id in enumerate(actor_ids)}


def plot_group_outputs(
    gender: str,
    statement_code: str,
    pair_records: List[PairRecord],
    umap_neighbors: int,
    umap_min_dist: float,
    middle_index: int,
    output_dir: Path,
) -> Dict[str, object]:
    vectors = np.stack([record.vector for record in pair_records], axis=0)
    umap_points = compute_umap_projection(vectors, umap_neighbors, umap_min_dist)
    output_group_dir = output_dir / f"gender_{gender}" / f"statement_{statement_code}"
    output_group_dir.mkdir(parents=True, exist_ok=True)

    fig_umap, ax_umap = plt.subplots(figsize=(10, 8))
    for point, record in zip(umap_points, pair_records):
        ax_umap.scatter(
            point[0],
            point[1],
            s=180,
            color=EMOTION_COLORS.get(record.emotion_code, "#888888"),
            marker=INTENSITY_MARKERS.get(record.intensity_code, "o"),
            edgecolors="black",
            linewidths=2.2,
            alpha=0.9,
        )

    plot_emotion_ellipses_and_centroids(ax_umap, umap_points, pair_records)

    ax_umap.set_title(
        f"Gender {gender} / Statement {statement_code} UMAP\n"
        f"wav2vec2 middle layer {middle_index} from emotion-neutral differences"
    )
    ax_umap.set_xlabel("UMAP-1")
    ax_umap.set_ylabel("UMAP-2")
    ax_umap.grid(alpha=0.2)
    add_legends(ax_umap, pair_records)

    umap_figure_path = output_group_dir / "umap_clusters.png"
    fig_umap.savefig(umap_figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig_umap)

    actor_colors = build_actor_colors(pair_records)
    fig_actor, ax_actor = plt.subplots(figsize=(10, 8))
    for point, record in zip(umap_points, pair_records):
        ax_actor.scatter(
            point[0],
            point[1],
            s=180,
            color=actor_colors[record.actor_id],
            marker=INTENSITY_MARKERS.get(record.intensity_code, "o"),
            edgecolors="black",
            linewidths=2.2,
            alpha=0.9,
        )

    plot_actor_ellipses_and_centroids(ax_actor, umap_points, pair_records, actor_colors)

    ax_actor.set_title(
        f"Gender {gender} / Statement {statement_code} UMAP by actor\n"
        f"wav2vec2 middle layer {middle_index} from emotion-neutral differences"
    )
    ax_actor.set_xlabel("UMAP-1")
    ax_actor.set_ylabel("UMAP-2")
    ax_actor.grid(alpha=0.2)
    add_actor_legends(ax_actor, pair_records, actor_colors)

    umap_actor_figure_path = output_group_dir / "umap_by_actor.png"
    fig_actor.savefig(umap_actor_figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig_actor)

    umap_path = output_group_dir / "umap.npy"
    np.save(umap_path, umap_points)

    metadata = {
        "gender": gender,
        "statement_code": statement_code,
        "middle_layer_index": middle_index,
        "umap_neighbors": min(max(2, umap_neighbors), max(2, len(pair_records) - 1)),
        "umap_min_dist": umap_min_dist,
        "num_pairs": len(pair_records),
        "umap_figure_path": str(umap_figure_path),
        "umap_by_actor_figure_path": str(umap_actor_figure_path),
        "umap_path": str(umap_path),
        "pairs": [
            {
                "label": record.label,
                "delta_label": record.delta_label,
                "actor_id": record.actor_id,
                "gender": record.gender,
                "emotion_code": record.emotion_code,
                "emotion_en": EMOTION_MAP.get(record.emotion_code, {}).get("en", "unknown"),
                "intensity_code": record.intensity_code,
                "neutral_reference_count": record.neutral_reference_count,
                "source_path": record.source_path,
            }
            for record in pair_records
        ],
    }

    metadata_path = output_group_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return metadata


def plot_actor_outputs(
    actor_id: int,
    statement_code: str,
    pair_records: List[PairRecord],
    umap_neighbors: int,
    umap_min_dist: float,
    middle_index: int,
    output_dir: Path,
) -> Dict[str, object]:
    vectors = np.stack([record.vector for record in pair_records], axis=0)
    umap_points = compute_umap_projection(vectors, umap_neighbors, umap_min_dist)
    output_group_dir = output_dir / f"actor_{actor_id:02d}" / f"statement_{statement_code}"
    output_group_dir.mkdir(parents=True, exist_ok=True)

    fig_umap, ax_umap = plt.subplots(figsize=(10, 8))
    for point, record in zip(umap_points, pair_records):
        ax_umap.scatter(
            point[0],
            point[1],
            s=180,
            color=EMOTION_COLORS.get(record.emotion_code, "#888888"),
            marker=INTENSITY_MARKERS.get(record.intensity_code, "o"),
            edgecolors="black",
            linewidths=2.2,
            alpha=0.9,
        )

    plot_emotion_ellipses_and_centroids(ax_umap, umap_points, pair_records)

    ax_umap.set_title(
        f"Actor {actor_id:02d} / Statement {statement_code} UMAP\n"
        f"wav2vec2 middle layer {middle_index} from emotion-neutral differences"
    )
    ax_umap.set_xlabel("UMAP-1")
    ax_umap.set_ylabel("UMAP-2")
    ax_umap.grid(alpha=0.2)
    add_legends(ax_umap, pair_records)

    umap_figure_path = output_group_dir / "umap_by_delta.png"
    fig_umap.savefig(umap_figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig_umap)

    umap_path = output_group_dir / "umap.npy"
    np.save(umap_path, umap_points)

    metadata = {
        "actor_id": actor_id,
        "gender": pair_records[0].gender,
        "statement_code": statement_code,
        "middle_layer_index": middle_index,
        "umap_neighbors": min(max(2, umap_neighbors), max(2, len(pair_records) - 1)),
        "umap_min_dist": umap_min_dist,
        "num_pairs": len(pair_records),
        "umap_figure_path": str(umap_figure_path),
        "umap_path": str(umap_path),
        "pairs": [
            {
                "label": record.label,
                "delta_label": record.delta_label,
                "emotion_code": record.emotion_code,
                "emotion_en": EMOTION_MAP.get(record.emotion_code, {}).get("en", "unknown"),
                "intensity_code": record.intensity_code,
                "neutral_reference_count": record.neutral_reference_count,
                "source_path": record.source_path,
            }
            for record in pair_records
        ],
    }

    metadata_path = output_group_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return metadata


def main() -> None:
    args = parse_args()
    if not args.include_neutral:
        raise RuntimeError(
            "Neutral utterances are excluded by default. wav2vec_rdm computes "
            "emotion-minus-neutral differences, so rerun with --include-neutral "
            "if you want to use neutral data as the baseline."
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(args)
    actor_statement_pairs, middle_index = build_pair_records(records, args.model_name, args.device)
    group_pairs = regroup_by_gender_and_statement(actor_statement_pairs)
    filtered_actor_pairs = filter_groups_with_min_pairs(
        actor_statement_pairs, minimum_pairs=3, group_name="actor"
    )
    filtered_gender_pairs = filter_groups_with_min_pairs(
        group_pairs, minimum_pairs=3, group_name="gender"
    )

    summary = {"by_gender": [], "by_actor": []}
    for (gender, statement_code), pair_records in sorted(filtered_gender_pairs.items()):
        print(
            f"Analyzing gender {gender} statement {statement_code} "
            f"with {len(pair_records)} emotion-neutral pairs"
        )
        summary["by_gender"].append(
            plot_group_outputs(
                gender=gender,
                statement_code=statement_code,
                pair_records=pair_records,
                umap_neighbors=args.umap_neighbors,
                umap_min_dist=args.umap_min_dist,
                middle_index=middle_index,
                output_dir=output_dir,
            )
        )

    for (actor_id, statement_code), pair_records in sorted(filtered_actor_pairs.items()):
        print(
            f"Analyzing actor {actor_id} statement {statement_code} "
            f"with {len(pair_records)} emotion-neutral pairs"
        )
        summary["by_actor"].append(
            plot_actor_outputs(
                actor_id=actor_id,
                statement_code=statement_code,
                pair_records=pair_records,
                umap_neighbors=args.umap_neighbors,
                umap_min_dist=args.umap_min_dist,
                middle_index=middle_index,
                output_dir=output_dir,
            )
        )

    if not summary["by_gender"] and not summary["by_actor"]:
        raise RuntimeError(
            "No gender/statement or actor/statement groups had enough neutral-vs-emotional pairs for UMAP."
        )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Saved gender-and-statement-level outputs under {output_dir}")


if __name__ == "__main__":
    main()
