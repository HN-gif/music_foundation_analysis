#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from transformers import Wav2Vec2Model, Wav2Vec2Processor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_dataset import EMOTION_MAP, RAVDESSRecord, collect_ravdess_files


DEFAULT_MODEL_NAME = "facebook/wav2vec2-base-960h"
DEFAULT_OUTPUT_DIR = Path("/home/takamichi-lab-pc07/research/wav2vec_mds/outputs")
TARGET_SAMPLE_RATE = 16_000
DEFAULT_RANDOM_STATE = 42
NEUTRAL_EMOTION_CODE = "01"
METHODS = ("A", "B", "C")
EMOTION_COLORS = {
    "neutral": "#4C78A8",
    "calm": "#72B7B2",
    "happy": "#F2CF5B",
    "sad": "#9C755F",
    "angry": "#E45756",
    "fearful": "#B279A2",
    "disgust": "#54A24B",
    "surprised": "#FF9DA6",
}


@dataclass(frozen=True)
class UtteranceEmbedding:
    source_path: str
    speaker_id: str
    emotion_label: str
    emotion_code: str
    text_id: str
    intensity_code: str
    repetition_code: str
    gender: str
    layer_vectors: Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class DistanceMatrix:
    labels: Tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class MethodResult:
    method: str
    distance_matrix: DistanceMatrix
    coordinates: Tuple[Dict[str, float | str], ...]
    stress: float
    emotion_count: int
    utterance_count: int
    distance_sample_count: int
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract layer-wise wav2vec2 hidden-state embeddings from RAVDESS, "
            "build emotion-label distance matrices with methods A/B/C, and plot MDS."
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
        help="Directory where CSV files, figures, and summary metrics are saved.",
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
        "--distance-metric",
        default="cosine",
        help="Distance metric passed to sklearn.metrics.pairwise_distances.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Methods to run.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Optional layer indices to analyze. By default all hidden-state layers are used.",
    )
    parser.add_argument(
        "--pair-sample-size",
        type=int,
        default=None,
        help=(
            "Maximum vector-pair count sampled per emotion pair for method C. "
            "If omitted, all combinations are used."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for MDS and method-C pair sampling.",
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
        "--emotion-codes",
        nargs="+",
        default=None,
        help="Optional subset of emotion codes. Example: 01 03 05",
    )
    parser.add_argument(
        "--include-neutral",
        action="store_true",
        help="Include neutral emotion utterances. By default emotion_code=01 is excluded.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of wav files to inspect after filtering.",
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


def extract_layer_embeddings(
    audio_path: Path,
    model_name: str,
    device_name: str,
) -> Tuple[np.ndarray, ...]:
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

    return tuple(
        hidden_state[0].mean(dim=0).detach().cpu().numpy().astype(np.float32)
        for hidden_state in hidden_states
    )


def collect_records(args: argparse.Namespace) -> List[RAVDESSRecord]:
    if args.emotion_codes is None:
        emotion_codes = [
            emotion_code
            for emotion_code in sorted(EMOTION_MAP)
            if args.include_neutral or emotion_code != NEUTRAL_EMOTION_CODE
        ]
    else:
        emotion_codes = [
            emotion_code
            for emotion_code in args.emotion_codes
            if args.include_neutral or emotion_code != NEUTRAL_EMOTION_CODE
        ]
        if not emotion_codes:
            raise RuntimeError(
                "No emotion codes remain after excluding neutral. "
                "Use --include-neutral if emotion_code=01 should be analyzed."
            )

    return collect_ravdess_files(
        args.data_dir,
        statement_codes=args.statement_codes,
        emotion_codes=emotion_codes,
        genders=args.genders,
        actor_ids=args.actor_ids,
        max_files=args.max_files,
    )


def record_to_embedding(
    record: RAVDESSRecord,
    model_name: str,
    device_name: str,
) -> Optional[UtteranceEmbedding]:
    speaker_id = f"actor_{record.actor_id:02d}"
    emotion_label = record.emotion_en
    text_id = record.statement_code
    required_values = [speaker_id, emotion_label]
    if any(value is None or value == "" or value == "unknown" for value in required_values):
        return None

    layer_vectors = extract_layer_embeddings(record.path, model_name, device_name)
    return UtteranceEmbedding(
        source_path=str(record.path),
        speaker_id=speaker_id,
        emotion_label=emotion_label,
        emotion_code=record.emotion_code,
        text_id=text_id,
        intensity_code=record.intensity_code,
        repetition_code=record.repetition_code,
        gender=record.gender,
        layer_vectors=layer_vectors,
    )


def build_utterance_embeddings(
    records: Iterable[RAVDESSRecord],
    model_name: str,
    device_name: str,
) -> List[UtteranceEmbedding]:
    embeddings: List[UtteranceEmbedding] = []
    for index, record in enumerate(records, start=1):
        print(f"[{index}] extracting hidden states: {record.path}")
        embedding = record_to_embedding(record, model_name, device_name)
        if embedding is not None:
            embeddings.append(embedding)

    if not embeddings:
        raise RuntimeError("No valid utterance embeddings were built after dropping missing rows.")

    return embeddings


def write_dict_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_utterance_metadata(path: Path, embeddings: Sequence[UtteranceEmbedding]) -> None:
    rows = [
        {
            "source_path": item.source_path,
            "speaker_id": item.speaker_id,
            "emotion_label": item.emotion_label,
            "emotion_code": item.emotion_code,
            "text_id": item.text_id,
            "intensity_code": item.intensity_code,
            "repetition_code": item.repetition_code,
            "gender": item.gender,
        }
        for item in embeddings
    ]
    write_dict_rows(
        path,
        rows,
        [
            "source_path",
            "speaker_id",
            "emotion_label",
            "emotion_code",
            "text_id",
            "intensity_code",
            "repetition_code",
            "gender",
        ],
    )


def layer_matrix(embeddings: Sequence[UtteranceEmbedding], layer_index: int) -> np.ndarray:
    return np.stack([item.layer_vectors[layer_index] for item in embeddings], axis=0)


def labels_for(embeddings: Sequence[UtteranceEmbedding]) -> np.ndarray:
    return np.array([item.emotion_label for item in embeddings], dtype=object)


def speaker_ids_for(embeddings: Sequence[UtteranceEmbedding]) -> np.ndarray:
    return np.array([item.speaker_id for item in embeddings], dtype=object)


def ordered_emotion_labels(embeddings: Sequence[UtteranceEmbedding]) -> List[str]:
    code_by_label = {item.emotion_label: item.emotion_code for item in embeddings}
    return sorted(code_by_label, key=lambda label: code_by_label[label])


def build_distance_matrix(vectors: np.ndarray, labels: Sequence[str], metric: str) -> DistanceMatrix:
    matrix = pairwise_distances(vectors, metric=metric)
    return DistanceMatrix(labels=tuple(labels), values=matrix)


def method_a(
    vectors: np.ndarray,
    emotion_labels: np.ndarray,
    ordered_labels: Sequence[str],
    metric: str,
) -> Tuple[DistanceMatrix, int, str]:
    representatives = []
    for label in ordered_labels:
        representatives.append(vectors[emotion_labels == label].mean(axis=0))
    representative_matrix = np.stack(representatives, axis=0)
    return build_distance_matrix(representative_matrix, ordered_labels, metric), len(vectors), ""


def method_b(
    vectors: np.ndarray,
    emotion_labels: np.ndarray,
    speaker_ids: np.ndarray,
    ordered_labels: Sequence[str],
    metric: str,
) -> Tuple[DistanceMatrix, int, str]:
    representatives = []
    used_speaker_emotion_groups = 0
    for label in ordered_labels:
        speaker_means = []
        for speaker_id in sorted(set(speaker_ids[emotion_labels == label])):
            mask = (emotion_labels == label) & (speaker_ids == speaker_id)
            if not np.any(mask):
                continue
            speaker_means.append(vectors[mask].mean(axis=0))
            used_speaker_emotion_groups += 1
        if not speaker_means:
            raise RuntimeError(f"Method B has no speaker means for emotion label: {label}")
        representatives.append(np.stack(speaker_means, axis=0).mean(axis=0))

    representative_matrix = np.stack(representatives, axis=0)
    notes = f"speaker_emotion_groups={used_speaker_emotion_groups}"
    return build_distance_matrix(
        representative_matrix, ordered_labels, metric
    ), used_speaker_emotion_groups, notes


def sampled_mean_distance(
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
    metric: str,
    rng: np.random.Generator,
    pair_sample_size: Optional[int],
    same_label: bool,
) -> Tuple[float, int]:
    if same_label:
        return 0.0, 0
    else:
        pair_count = len(left_vectors) * len(right_vectors)
        if pair_count == 0:
            return np.nan, 0
        if pair_sample_size is None or pair_count <= pair_sample_size:
            distances = pairwise_distances(left_vectors, right_vectors, metric=metric)
            return float(distances.mean()), pair_count
        left_indices = rng.integers(0, len(left_vectors), size=pair_sample_size)
        right_indices = rng.integers(0, len(right_vectors), size=pair_sample_size)

    sampled_distances = pairwise_distances(
        left_vectors[left_indices],
        right_vectors[right_indices],
        metric=metric,
    )
    return float(np.diag(sampled_distances).mean()), int(pair_sample_size)


def method_c(
    vectors: np.ndarray,
    emotion_labels: np.ndarray,
    ordered_labels: Sequence[str],
    metric: str,
    random_state: int,
    pair_sample_size: Optional[int],
) -> Tuple[DistanceMatrix, int, str]:
    rng = np.random.default_rng(random_state)
    matrix = np.zeros((len(ordered_labels), len(ordered_labels)), dtype=np.float64)
    sampled_pairs = 0

    vectors_by_label = {
        label: vectors[emotion_labels == label]
        for label in ordered_labels
    }
    for left_index, left_label in enumerate(ordered_labels):
        for right_index in range(left_index, len(ordered_labels)):
            right_label = ordered_labels[right_index]
            distance, count = sampled_mean_distance(
                vectors_by_label[left_label],
                vectors_by_label[right_label],
                metric,
                rng,
                pair_sample_size,
                same_label=left_label == right_label,
            )
            matrix[left_index, right_index] = distance
            matrix[right_index, left_index] = distance
            sampled_pairs += count

    notes = f"pair_sample_size={pair_sample_size if pair_sample_size is not None else 'all'}"
    return DistanceMatrix(labels=tuple(ordered_labels), values=matrix), sampled_pairs, notes


def run_mds(distance_matrix: DistanceMatrix, random_state: int) -> Tuple[Tuple[Dict[str, float | str], ...], float]:
    if distance_matrix.values.shape[0] < 2:
        raise RuntimeError("MDS needs at least two emotion labels.")
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
        normalized_stress="auto",
    )
    coordinates = mds.fit_transform(distance_matrix.values.astype(np.float64))
    coordinate_rows = tuple(
        {
            "emotion_label": label,
            "mds_1": float(point[0]),
            "mds_2": float(point[1]),
        }
        for label, point in zip(distance_matrix.labels, coordinates)
    )
    return coordinate_rows, float(mds.stress_)


STATEMENT_LABEL = {
    "01": 'Statement 01 ("Kids are talking by the door")',
    "02": 'Statement 02 ("Dogs are sitting by the door")',
}


def plot_mds(
    coordinates: Sequence[Dict[str, float | str]],
    layer_index: int,
    method: str,
    stress: float,
    output_path: Path,
    gender: str = "",
    text_id: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for row in coordinates:
        label = str(row["emotion_label"])
        ax.scatter(
            row["mds_1"],
            row["mds_2"],
            s=180,
            color=EMOTION_COLORS.get(label, "#888888"),
            edgecolors="black",
            linewidths=1.5,
            alpha=0.92,
        )
        ax.annotate(
            label,
            xy=(row["mds_1"], row["mds_2"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )

    ax.axhline(0, color="#BBBBBB", linewidth=0.8)
    ax.axvline(0, color="#BBBBBB", linewidth=0.8)
    subtitle_parts = []
    if gender:
        subtitle_parts.append(gender.capitalize())
    if text_id:
        subtitle_parts.append(STATEMENT_LABEL.get(text_id, f"Text {text_id}"))
    subtitle = " / ".join(subtitle_parts)
    title = f"Layer {layer_index:02d} Method {method} Emotion MDS"
    if subtitle:
        title = f"{title}\n{subtitle}"
    title = f"{title}\nstress={stress:.6g}"
    ax.set_title(title)
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.grid(alpha=0.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_method(
    method: str,
    vectors: np.ndarray,
    embeddings: Sequence[UtteranceEmbedding],
    ordered_labels: Sequence[str],
    metric: str,
    random_state: int,
    pair_sample_size: Optional[int],
) -> MethodResult:
    emotion_labels = labels_for(embeddings)
    speaker_ids = speaker_ids_for(embeddings)

    if method == "A":
        distance_frame, sample_count, notes = method_a(vectors, emotion_labels, ordered_labels, metric)
    elif method == "B":
        distance_frame, sample_count, notes = method_b(
            vectors, emotion_labels, speaker_ids, ordered_labels, metric
        )
    elif method == "C":
        distance_frame, sample_count, notes = method_c(
            vectors, emotion_labels, ordered_labels, metric, random_state, pair_sample_size
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    coordinates, stress = run_mds(distance_frame, random_state)
    return MethodResult(
        method=method,
        distance_matrix=distance_frame,
        coordinates=coordinates,
        stress=stress,
        emotion_count=len(ordered_labels),
        utterance_count=len(embeddings),
        distance_sample_count=sample_count,
        notes=notes,
    )


def save_method_outputs(
    result: MethodResult,
    layer_index: int,
    output_dir: Path,
    gender: str = "",
    text_id: str = "",
) -> Dict[str, object]:
    method_dir = output_dir / f"layer_{layer_index:02d}" / f"method_{result.method}"
    method_dir.mkdir(parents=True, exist_ok=True)

    distance_path = method_dir / "distance_matrix.csv"
    coordinates_path = method_dir / "mds_coordinates.csv"
    plot_path = method_dir / "mds_plot.png"

    write_distance_matrix(distance_path, result.distance_matrix)
    write_dict_rows(coordinates_path, result.coordinates, ["emotion_label", "mds_1", "mds_2"])
    plot_mds(result.coordinates, layer_index, result.method, result.stress, plot_path, gender, text_id)

    return {
        "gender": gender,
        "text_id": text_id,
        "layer": layer_index,
        "method": result.method,
        "stress": result.stress,
        "emotion_count": result.emotion_count,
        "utterance_count": result.utterance_count,
        "distance_sample_count": result.distance_sample_count,
        "distance_matrix_path": str(distance_path),
        "mds_coordinates_path": str(coordinates_path),
        "plot_path": str(plot_path),
        "notes": result.notes,
    }


def write_distance_matrix(path: Path, distance_matrix: DistanceMatrix) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["emotion_label", *distance_matrix.labels])
        for label, row in zip(distance_matrix.labels, distance_matrix.values):
            writer.writerow([label, *[float(value) for value in row]])


def validate_layers(requested_layers: Optional[Sequence[int]], available_layers: int) -> List[int]:
    if requested_layers is None:
        return list(range(available_layers))
    invalid = [layer for layer in requested_layers if layer < 0 or layer >= available_layers]
    if invalid:
        raise ValueError(f"Invalid layer indices {invalid}; available range is 0..{available_layers - 1}")
    return list(dict.fromkeys(requested_layers))


def emotion_count_columns() -> List[str]:
    return [
        f"count_{emotion_code}_{EMOTION_MAP[emotion_code]['en']}"
        for emotion_code in sorted(EMOTION_MAP)
    ]


def summarize_group_labels(embeddings: Sequence[UtteranceEmbedding]) -> str:
    counts: Dict[str, int] = {}
    label_by_code = {
        item.emotion_code: item.emotion_label
        for item in embeddings
    }
    for item in embeddings:
        counts[item.emotion_code] = counts.get(item.emotion_code, 0) + 1
    return ";".join(
        f"{emotion_code}_{label_by_code[emotion_code]}:{counts[emotion_code]}"
        for emotion_code in sorted(counts)
    )


def build_group_sample_count_rows(
    groups: Dict[Tuple[str, str], List[UtteranceEmbedding]]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for (gender, text_id), embeddings in sorted(groups.items()):
        counts = {emotion_code: 0 for emotion_code in sorted(EMOTION_MAP)}
        for item in embeddings:
            counts[item.emotion_code] = counts.get(item.emotion_code, 0) + 1

        emotion_count = sum(1 for count in counts.values() if count > 0)
        analyzed = emotion_count >= 2
        row: Dict[str, object] = {
            "gender": gender,
            "text_id": text_id,
            "output_name": f"{gender}_text{text_id}",
            "utterance_count": len(embeddings),
            "emotion_count": emotion_count,
            "analyzed": analyzed,
            "skip_reason": "" if analyzed else "fewer_than_2_emotion_labels",
            "label_counts": summarize_group_labels(embeddings),
        }
        for emotion_code in sorted(EMOTION_MAP):
            emotion_name = EMOTION_MAP[emotion_code]["en"]
            row[f"count_{emotion_code}_{emotion_name}"] = counts.get(emotion_code, 0)
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(args)
    all_embeddings = build_utterance_embeddings(records, args.model_name, args.device)
    available_layers = len(all_embeddings[0].layer_vectors)
    layers = validate_layers(args.layers, available_layers)

    metadata_path = args.output_dir / "utterance_metadata.csv"
    write_utterance_metadata(metadata_path, all_embeddings)

    # Group by (gender, text_id)
    groups: Dict[Tuple[str, str], List[UtteranceEmbedding]] = {}
    for emb in all_embeddings:
        groups.setdefault((emb.gender, emb.text_id), []).append(emb)

    write_dict_rows(
        args.output_dir / "group_sample_counts.csv",
        build_group_sample_count_rows(groups),
        [
            "gender",
            "text_id",
            "output_name",
            "utterance_count",
            "emotion_count",
            "analyzed",
            "skip_reason",
            "label_counts",
            *emotion_count_columns(),
        ],
    )

    summary_rows = []
    for (gender, text_id), embeddings in sorted(groups.items()):
        group_label = f"{gender}_text{text_id}"
        group_dir = args.output_dir / group_label
        group_dir.mkdir(parents=True, exist_ok=True)

        ordered_labels = ordered_emotion_labels(embeddings)
        if len(ordered_labels) < 2:
            print(f"Skipping {group_label}: fewer than 2 emotion labels.")
            continue

        for layer_index in layers:
            print(f"[{group_label}] Analyzing layer {layer_index:02d}")
            vectors = layer_matrix(embeddings, layer_index)
            for method in args.methods:
                print(f"  method {method}")
                result = run_method(
                    method=method,
                    vectors=vectors,
                    embeddings=embeddings,
                    ordered_labels=ordered_labels,
                    metric=args.distance_metric,
                    random_state=args.random_state,
                    pair_sample_size=args.pair_sample_size,
                )
                summary_rows.append(
                    save_method_outputs(result, layer_index, group_dir, gender, text_id)
                )

    summary_path = args.output_dir / "summary_metrics.csv"
    write_dict_rows(
        summary_path,
        summary_rows,
        [
            "gender",
            "text_id",
            "layer",
            "method",
            "stress",
            "emotion_count",
            "utterance_count",
            "distance_sample_count",
            "distance_matrix_path",
            "mds_coordinates_path",
            "plot_path",
            "notes",
        ],
    )
    print(f"Saved MDS outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
