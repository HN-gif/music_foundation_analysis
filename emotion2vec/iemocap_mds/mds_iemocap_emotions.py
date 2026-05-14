#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emotion2vec.emotion2vec import DEFAULT_MODEL_NAME, extract_layer_embeddings, validate_layers
from wav2vec_iemocap_rsa.rsa_iemocap_layers import IemocapUtterance, collect_iemocap_utterances


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "IEMOCAP_full_release"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "emotion2vec" / "iemocap_mds" / "outputs"
DEFAULT_RANDOM_STATE = 42
DEFAULT_LAYERS = (0, 4, 8)
DEFAULT_DIALOG_TYPES = ("script",)
METHODS = ("A", "B", "C")
GROUP_SCOPES = ("all", "genders", "speakers", "texts")
EMOTION_ORDER = ("neu", "hap", "exc", "sad", "ang", "fru", "fea", "sur", "dis", "oth")
EXCLUDED_EMOTION_LABELS = {"oth"}
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
EMOTION_COLORS = {
    "ang": "#E45756",
    "hap": "#F2CF5B",
    "sad": "#9C755F",
    "neu": "#4C78A8",
    "fru": "#B279A2",
    "exc": "#F58518",
    "fea": "#7F7F7F",
    "sur": "#FF9DA6",
    "dis": "#54A24B",
    "oth": "#BAB0AC",
}


@dataclass(frozen=True)
class IemocapEmbedding:
    metadata: IemocapUtterance
    content_id: str
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


@dataclass(frozen=True)
class AnalysisGroup:
    scope: str
    group_id: str
    output_name: str
    indices: Tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract layer-wise emotion2vec embeddings from IEMOCAP utterances, "
            "build discrete-emotion distance matrices, and plot MDS."
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
    parser.add_argument(
        "--distance-metric",
        default="cosine",
        help=(
            "Distance metric passed to sklearn.metrics.pairwise_distances. "
            "Use norm-only to compare only L2 norm magnitudes."
        ),
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    parser.add_argument(
        "--group-scopes",
        nargs="+",
        choices=GROUP_SCOPES,
        default=["all", "genders"],
    )
    parser.add_argument("--pair-sample-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--mds-components", type=int, default=2)
    parser.add_argument("--sessions", nargs="+", default=None)
    parser.add_argument("--emotion-labels", nargs="+", default=None)
    parser.add_argument(
        "--dialog-types",
        nargs="+",
        choices=["impro", "script"],
        default=DEFAULT_DIALOG_TYPES,
    )
    parser.add_argument("--include-xxx", action="store_true")
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--skip-groups-with-fewer-emotions", type=int, default=2)
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


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace(":", "_")


def speaker_gender_name(item: IemocapUtterance) -> str:
    return "female" if item.speaker_gender == "F" else "male"


def infer_content_id(dialog_id: str) -> str:
    parts = dialog_id.split("_", 1)
    if len(parts) == 2:
        return parts[1]
    return dialog_id


def build_embeddings(
    utterances: Sequence[IemocapUtterance],
    model_name: str,
    device_name: str,
    hub: str,
) -> List[IemocapEmbedding]:
    embeddings: List[IemocapEmbedding] = []
    for index, utterance in enumerate(utterances, start=1):
        print(f"[{index}/{len(utterances)}] extracting hidden states: {utterance.utterance_id}")
        embeddings.append(
            IemocapEmbedding(
                metadata=utterance,
                content_id=infer_content_id(utterance.dialog_id),
                layer_vectors=extract_layer_embeddings(
                    utterance.wav_path,
                    model_name=model_name,
                    device_name=device_name,
                    hub=hub,
                ),
            )
        )
    if not embeddings:
        raise RuntimeError("No valid IEMOCAP utterance embeddings were built.")
    return embeddings


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
                emotion_name=EMOTION_NAMES.get(shuffled_label, utterance.emotion_name),
            )
        )
    return shuffled


def write_dict_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clear_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def write_metadata(path: Path, embeddings: Sequence[IemocapEmbedding]) -> None:
    rows = []
    for emb in embeddings:
        item = emb.metadata
        rows.append(
            {
                "utterance_id": item.utterance_id,
                "wav_path": str(item.wav_path),
                "session_id": item.session_id,
                "dialog_id": item.dialog_id,
                "dialog_type": item.dialog_type,
                "content_id": emb.content_id,
                "speaker_id": item.speaker_id,
                "speaker_gender": item.speaker_gender,
                "gender": speaker_gender_name(item),
                "start_time": item.start_time,
                "end_time": item.end_time,
                "duration": item.end_time - item.start_time,
                "emotion_label": item.emotion_label,
                "emotion_name": item.emotion_name,
                "valence": item.valence,
                "activation": item.activation,
                "dominance": item.dominance,
            }
        )
    write_dict_rows(
        path,
        rows,
        [
            "utterance_id",
            "wav_path",
            "session_id",
            "dialog_id",
            "dialog_type",
            "content_id",
            "speaker_id",
            "speaker_gender",
            "gender",
            "start_time",
            "end_time",
            "duration",
            "emotion_label",
            "emotion_name",
            "valence",
            "activation",
            "dominance",
        ],
    )


def make_groups(embeddings: Sequence[IemocapEmbedding], scopes: Sequence[str]) -> List[AnalysisGroup]:
    groups: List[AnalysisGroup] = []
    if "all" in scopes:
        groups.append(AnalysisGroup("all", "all", "all", tuple(range(len(embeddings)))))
    if "genders" in scopes:
        by_gender: Dict[str, List[int]] = {}
        for index, emb in enumerate(embeddings):
            by_gender.setdefault(speaker_gender_name(emb.metadata), []).append(index)
        for gender, indices in sorted(by_gender.items()):
            groups.append(AnalysisGroup("gender", gender, f"gender_{gender}", tuple(indices)))
    if "speakers" in scopes:
        by_speaker: Dict[str, List[int]] = {}
        for index, emb in enumerate(embeddings):
            by_speaker.setdefault(emb.metadata.speaker_id, []).append(index)
        for speaker_id, indices in sorted(by_speaker.items()):
            groups.append(AnalysisGroup("speaker", speaker_id, f"speaker_{speaker_id}", tuple(indices)))
    if "texts" in scopes:
        by_text: Dict[str, List[int]] = {}
        for index, emb in enumerate(embeddings):
            by_text.setdefault(emb.content_id, []).append(index)
        for content_id, indices in sorted(by_text.items()):
            groups.append(AnalysisGroup("text", content_id, f"text_{safe_name(content_id)}", tuple(indices)))
    return groups


def subset_embeddings(embeddings: Sequence[IemocapEmbedding], indices: Sequence[int]) -> List[IemocapEmbedding]:
    return [embeddings[index] for index in indices]


def layer_matrix(embeddings: Sequence[IemocapEmbedding], layer_index: int) -> np.ndarray:
    return np.stack([item.layer_vectors[layer_index] for item in embeddings], axis=0)


def labels_for(embeddings: Sequence[IemocapEmbedding]) -> np.ndarray:
    return np.array([item.metadata.emotion_label for item in embeddings], dtype=object)


def speaker_ids_for(embeddings: Sequence[IemocapEmbedding]) -> np.ndarray:
    return np.array([item.metadata.speaker_id for item in embeddings], dtype=object)


def ordered_emotion_labels(embeddings: Sequence[IemocapEmbedding]) -> List[str]:
    present = {item.metadata.emotion_label for item in embeddings}
    ordered = [label for label in EMOTION_ORDER if label in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def vector_distances(left_vectors: np.ndarray, right_vectors: Optional[np.ndarray], metric: str) -> np.ndarray:
    if metric == "norm-only":
        left_norms = np.linalg.norm(left_vectors, axis=1)
        right_norms = left_norms if right_vectors is None else np.linalg.norm(right_vectors, axis=1)
        return np.abs(left_norms[:, np.newaxis] - right_norms[np.newaxis, :])
    if right_vectors is None:
        return pairwise_distances(left_vectors, metric=metric)
    return pairwise_distances(left_vectors, right_vectors, metric=metric)


def build_distance_matrix(vectors: np.ndarray, labels: Sequence[str], metric: str) -> DistanceMatrix:
    matrix = vector_distances(vectors, None, metric)
    return DistanceMatrix(labels=tuple(labels), values=matrix.astype(np.float64))


def method_a(vectors: np.ndarray, emotion_labels: np.ndarray, ordered_labels: Sequence[str], metric: str):
    representatives = [vectors[emotion_labels == label].mean(axis=0) for label in ordered_labels]
    representative_matrix = np.stack(representatives, axis=0)
    return build_distance_matrix(representative_matrix, ordered_labels, metric), len(vectors), ""


def method_b(
    vectors: np.ndarray,
    emotion_labels: np.ndarray,
    speaker_ids: np.ndarray,
    ordered_labels: Sequence[str],
    metric: str,
):
    representatives = []
    used_groups = 0
    for label in ordered_labels:
        speaker_means = []
        for speaker_id in sorted(set(speaker_ids[emotion_labels == label])):
            mask = (emotion_labels == label) & (speaker_ids == speaker_id)
            if np.any(mask):
                speaker_means.append(vectors[mask].mean(axis=0))
                used_groups += 1
        if not speaker_means:
            raise RuntimeError(f"Method B has no speaker means for emotion label: {label}")
        representatives.append(np.stack(speaker_means, axis=0).mean(axis=0))
    representative_matrix = np.stack(representatives, axis=0)
    return build_distance_matrix(representative_matrix, ordered_labels, metric), used_groups, f"speaker_emotion_groups={used_groups}"


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
    pair_count = len(left_vectors) * len(right_vectors)
    if pair_count == 0:
        return np.nan, 0
    if pair_sample_size is None or pair_count <= pair_sample_size:
        distances = vector_distances(left_vectors, right_vectors, metric)
        return float(distances.mean()), pair_count
    left_indices = rng.integers(0, len(left_vectors), size=pair_sample_size)
    right_indices = rng.integers(0, len(right_vectors), size=pair_sample_size)
    sampled = vector_distances(left_vectors[left_indices], right_vectors[right_indices], metric)
    return float(np.diag(sampled).mean()), int(pair_sample_size)


def method_c(
    vectors: np.ndarray,
    emotion_labels: np.ndarray,
    ordered_labels: Sequence[str],
    metric: str,
    random_state: int,
    pair_sample_size: Optional[int],
):
    rng = np.random.default_rng(random_state)
    matrix = np.zeros((len(ordered_labels), len(ordered_labels)), dtype=np.float64)
    sampled_pairs = 0
    vectors_by_label = {label: vectors[emotion_labels == label] for label in ordered_labels}
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
    return DistanceMatrix(labels=tuple(ordered_labels), values=matrix), sampled_pairs, f"pair_sample_size={pair_sample_size if pair_sample_size is not None else 'all'}"


def run_mds(distance_matrix: DistanceMatrix, random_state: int, n_components: int):
    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=4,
        normalized_stress="auto",
    )
    coordinates = mds.fit_transform(distance_matrix.values.astype(np.float64))
    rows = []
    for label, point in zip(distance_matrix.labels, coordinates):
        row: Dict[str, float | str] = {
            "emotion_label": label,
            "emotion_name": EMOTION_NAMES.get(label, "unknown"),
        }
        for dim_index, value in enumerate(point, start=1):
            row[f"mds_{dim_index}"] = float(value)
        rows.append(row)
    return tuple(rows), float(mds.stress_)


def run_method(
    method: str,
    vectors: np.ndarray,
    embeddings: Sequence[IemocapEmbedding],
    ordered_labels: Sequence[str],
    metric: str,
    random_state: int,
    pair_sample_size: Optional[int],
    n_components: int,
) -> MethodResult:
    emotion_labels = labels_for(embeddings)
    speaker_ids = speaker_ids_for(embeddings)
    if method == "A":
        distance_matrix, sample_count, notes = method_a(vectors, emotion_labels, ordered_labels, metric)
    elif method == "B":
        distance_matrix, sample_count, notes = method_b(vectors, emotion_labels, speaker_ids, ordered_labels, metric)
    elif method == "C":
        distance_matrix, sample_count, notes = method_c(vectors, emotion_labels, ordered_labels, metric, random_state, pair_sample_size)
    else:
        raise ValueError(f"Unsupported method: {method}")
    coordinates, stress = run_mds(distance_matrix, random_state, n_components)
    return MethodResult(method, distance_matrix, coordinates, stress, len(ordered_labels), len(embeddings), sample_count, notes)


def write_distance_matrix(path: Path, distance_matrix: DistanceMatrix) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["emotion_label", *distance_matrix.labels])
        for label, row in zip(distance_matrix.labels, distance_matrix.values):
            writer.writerow([label, *[float(value) for value in row]])


def plot_mds(
    coordinates: Sequence[Dict[str, float | str]],
    layer_index: int,
    method: str,
    group: AnalysisGroup,
    stress: float,
    output_path: Path,
) -> None:
    coordinate_dims = sorted([key for key in coordinates[0] if key.startswith("mds_")], key=lambda value: int(value.split("_")[1]))
    plot_dims = min(len(coordinate_dims), 3)
    fig = plt.figure(figsize=(8, 7))
    if plot_dims >= 3:
        ax = fig.add_subplot(111, projection="3d")
        for row in coordinates:
            label = str(row["emotion_label"])
            ax.scatter(float(row["mds_1"]), float(row["mds_2"]), float(row["mds_3"]), s=160, color=EMOTION_COLORS.get(label, "#888888"), edgecolors="black", linewidths=1.2, alpha=0.92)
            ax.text(float(row["mds_1"]), float(row["mds_2"]), float(row["mds_3"]), f"{label} ({row['emotion_name']})", fontsize=9, fontweight="bold")
        ax.set_zlabel("MDS-3")
    else:
        ax = fig.add_subplot(111)
        for row in coordinates:
            label = str(row["emotion_label"])
            x_value = float(row["mds_1"])
            y_value = float(row["mds_2"]) if len(coordinate_dims) >= 2 else 0.0
            ax.scatter(x_value, y_value, s=180, color=EMOTION_COLORS.get(label, "#888888"), edgecolors="black", linewidths=1.5, alpha=0.92)
            ax.annotate(f"{label} ({row['emotion_name']})", xy=(x_value, y_value), xytext=(6, 6), textcoords="offset points", fontsize=10, fontweight="bold")
        ax.axhline(0, color="#BBBBBB", linewidth=0.8)
        ax.axvline(0, color="#BBBBBB", linewidth=0.8)
    ax.set_title(f"{group.scope}: {group.group_id}\nLayer {layer_index:02d} Method {method} IEMOCAP emotion2vec MDS ({len(coordinate_dims)}D; plotting first {plot_dims}D)\nstress={stress:.6g}")
    ax.set_xlabel("MDS-1")
    if plot_dims >= 2:
        ax.set_ylabel("MDS-2")
    ax.grid(alpha=0.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def coordinate_fieldnames(coordinates: Sequence[Dict[str, float | str]]) -> List[str]:
    dims = sorted([key for key in coordinates[0] if key.startswith("mds_")], key=lambda value: int(value.split("_")[1]))
    return ["emotion_label", "emotion_name", *dims]


def save_method_outputs(result: MethodResult, layer_index: int, group: AnalysisGroup, output_dir: Path) -> Dict[str, object]:
    method_dir = output_dir / group.output_name / f"layer_{layer_index:02d}" / f"method_{result.method}"
    method_dir.mkdir(parents=True, exist_ok=True)
    distance_path = method_dir / "distance_matrix.csv"
    coordinates_path = method_dir / "mds_coordinates.csv"
    plot_path = method_dir / "mds_plot.png"
    write_distance_matrix(distance_path, result.distance_matrix)
    write_dict_rows(coordinates_path, result.coordinates, coordinate_fieldnames(result.coordinates))
    plot_mds(result.coordinates, layer_index, result.method, group, result.stress, plot_path)
    return {
        "group_scope": group.scope,
        "group_id": group.group_id,
        "layer": layer_index,
        "method": result.method,
        "mds_components": len(coordinate_fieldnames(result.coordinates)) - 2,
        "stress": result.stress,
        "emotion_count": result.emotion_count,
        "utterance_count": result.utterance_count,
        "distance_sample_count": result.distance_sample_count,
        "distance_matrix_path": str(distance_path),
        "mds_coordinates_path": str(coordinates_path),
        "plot_path": str(plot_path),
        "notes": result.notes,
    }


def summarize_group_labels(embeddings: Sequence[IemocapEmbedding]) -> str:
    counts: Dict[str, int] = {}
    for emb in embeddings:
        counts[emb.metadata.emotion_label] = counts.get(emb.metadata.emotion_label, 0) + 1
    return ";".join(f"{label}:{counts[label]}" for label in ordered_emotion_labels(embeddings))


def emotion_count_columns() -> List[str]:
    return [f"count_{label}" for label in EMOTION_ORDER if label not in EXCLUDED_EMOTION_LABELS]


def build_group_sample_count_rows(
    groups: Sequence[AnalysisGroup],
    all_embeddings: Sequence[IemocapEmbedding],
    minimum_emotions: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for group in groups:
        embeddings = subset_embeddings(all_embeddings, group.indices)
        counts: Dict[str, int] = {label: 0 for label in EMOTION_ORDER if label not in EXCLUDED_EMOTION_LABELS}
        for emb in embeddings:
            counts[emb.metadata.emotion_label] = counts.get(emb.metadata.emotion_label, 0) + 1
        present_labels = [label for label, count in counts.items() if count > 0]
        analyzed = len(present_labels) >= minimum_emotions
        row: Dict[str, object] = {
            "group_scope": group.scope,
            "group_id": group.group_id,
            "output_name": group.output_name,
            "utterance_count": len(embeddings),
            "emotion_count": len(present_labels),
            "analyzed": analyzed,
            "skip_reason": "" if analyzed else f"fewer_than_{minimum_emotions}_emotion_labels",
            "label_counts": ";".join(f"{label}:{counts[label]}" for label in EMOTION_ORDER if counts.get(label, 0) > 0),
        }
        for label in EMOTION_ORDER:
            if label not in EXCLUDED_EMOTION_LABELS:
                row[f"count_{label}"] = counts.get(label, 0)
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    clear_output_dir(args.output_dir)
    shuffle_seed = args.random_state if args.shuffle_label_seed is None else args.shuffle_label_seed
    utterances = collect_iemocap_utterances(args.data_dir, args.sessions, args.emotion_labels, args.dialog_types, args.include_xxx, args.max_utterances)
    utterances = [utterance for utterance in utterances if utterance.emotion_label not in EXCLUDED_EMOTION_LABELS]
    if not utterances:
        raise RuntimeError("No IEMOCAP utterances remained after excluding the 'oth' emotion label.")
    utterances = maybe_shuffle_emotion_labels(utterances, args.shuffle_emotion_labels, shuffle_seed)
    print(f"Collected {len(utterances)} IEMOCAP utterances")
    if args.shuffle_emotion_labels:
        print(f"Emotion labels were shuffled across utterances with seed={shuffle_seed}")
    all_embeddings = build_embeddings(utterances, args.model_name, args.device, args.hub)
    layers = validate_layers(args.layers, len(all_embeddings[0].layer_vectors))
    write_metadata(args.output_dir / "utterance_metadata.csv", all_embeddings)
    groups = make_groups(all_embeddings, args.group_scopes)
    group_count_rows = build_group_sample_count_rows(groups, all_embeddings, args.skip_groups_with_fewer_emotions)
    write_dict_rows(
        args.output_dir / "group_sample_counts.csv",
        group_count_rows,
        ["group_scope", "group_id", "output_name", "utterance_count", "emotion_count", "analyzed", "skip_reason", "label_counts", *emotion_count_columns()],
    )
    summary_rows = []
    for group in groups:
        embeddings = subset_embeddings(all_embeddings, group.indices)
        ordered_labels = ordered_emotion_labels(embeddings)
        if len(ordered_labels) < args.skip_groups_with_fewer_emotions:
            print(f"Skipping {group.scope} {group.group_id}: only {len(ordered_labels)} emotion labels ({summarize_group_labels(embeddings)})")
            continue
        print(f"Analyzing {group.scope} {group.group_id}: {len(embeddings)} utterances, labels {summarize_group_labels(embeddings)}")
        for layer_index in layers:
            vectors = layer_matrix(embeddings, layer_index)
            print(f"  layer {layer_index:02d}")
            for method in args.methods:
                print(f"    method {method}")
                result = run_method(method, vectors, embeddings, ordered_labels, args.distance_metric, args.random_state, args.pair_sample_size, args.mds_components)
                summary_rows.append(save_method_outputs(result, layer_index, group, args.output_dir))
    if not summary_rows:
        raise RuntimeError("No MDS groups were analyzed. Check filters and group settings.")
    write_dict_rows(
        args.output_dir / "summary_metrics.csv",
        summary_rows,
        ["group_scope", "group_id", "layer", "method", "mds_components", "stress", "emotion_count", "utterance_count", "distance_sample_count", "distance_matrix_path", "mds_coordinates_path", "plot_path", "notes"],
    )
    print(f"Saved IEMOCAP MDS outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
