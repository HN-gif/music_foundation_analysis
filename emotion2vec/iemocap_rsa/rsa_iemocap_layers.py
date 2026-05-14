#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emotion2vec.emotion2vec import DEFAULT_MODEL_NAME, extract_layer_embeddings, validate_layers
from wav2vec_iemocap_rsa.rsa_iemocap_layers import (
    EMOTION_COLORS,
    IemocapUtterance,
    collect_iemocap_utterances,
)


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "IEMOCAP_full_release"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "emotion2vec" / "iemocap_rsa" / "outputs"
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class EmbeddedUtterance:
    metadata: IemocapUtterance
    layer_vectors: Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class AnalysisGroup:
    name: str
    dialog_type_analysis: str
    utterance_indices: Tuple[int, ...]


@dataclass(frozen=True)
class ProjectionResult:
    coordinates: np.ndarray
    stress: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run layer-wise RSA between emotion2vec utterance RDMs and "
            "IEMOCAP human VAD RDMs."
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
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Optional layer indices to analyze. By default all hidden-state layers are used.",
    )
    parser.add_argument(
        "--model-distance-metric",
        default="cosine",
        help="Distance metric for emotion2vec utterance vectors.",
    )
    parser.add_argument(
        "--vad-distance-metric",
        default="euclidean",
        help="Distance metric for z-scored VAD vectors.",
    )
    parser.add_argument(
        "--analysis-groups",
        nargs="+",
        choices=["all", "speakers"],
        default=["all", "speakers"],
        help="Run one combined RDM, speaker-specific RDMs, or both.",
    )
    parser.add_argument(
        "--dialog-type-analyses",
        nargs="+",
        choices=["combined", "impro", "script"],
        default=["combined"],
        help=(
            "Dialog-type analysis conditions. combined uses impro+script together; "
            "impro and script run separate RDM/RSA analyses."
        ),
    )
    parser.add_argument(
        "--dialog-types",
        nargs="+",
        choices=["impro", "script"],
        default=None,
        help="Optional input filter before analysis. By default both impro and script utterances are loaded.",
    )
    parser.add_argument("--sessions", nargs="+", default=None)
    parser.add_argument("--emotion-labels", nargs="+", default=None)
    parser.add_argument(
        "--include-xxx",
        action="store_true",
        help="Include utterances whose categorical emotion label is xxx. Default excludes them.",
    )
    parser.add_argument(
        "--max-utterances",
        type=int,
        default=None,
        help="Optional maximum number of utterances after filtering, useful for smoke tests.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for MDS, UMAP, and visualization subsampling.",
    )
    parser.add_argument(
        "--viz-sample-size",
        type=int,
        default=2000,
        help="Maximum points used for scatter/MDS/UMAP plots. RDM and RSA still use all utterances.",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="Base n_neighbors value for UMAP, clipped by sample count.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--skip-csv-rdms",
        action="store_true",
        help="Skip large RDM CSV files and save only .npy RDMs.",
    )
    return parser.parse_args()


def build_embeddings(
    utterances: Sequence[IemocapUtterance],
    model_name: str,
    device_name: str,
    hub: str,
) -> List[EmbeddedUtterance]:
    embedded: List[EmbeddedUtterance] = []
    for index, utterance in enumerate(utterances, start=1):
        print(f"[{index}/{len(utterances)}] extracting hidden states: {utterance.utterance_id}")
        embedded.append(
            EmbeddedUtterance(
                metadata=utterance,
                layer_vectors=extract_layer_embeddings(
                    utterance.wav_path,
                    model_name=model_name,
                    device_name=device_name,
                    hub=hub,
                ),
            )
        )
    return embedded


def write_dict_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path: Path, embedded: Sequence[EmbeddedUtterance]) -> None:
    rows = []
    for item in embedded:
        meta = item.metadata
        rows.append(
            {
                "utterance_id": meta.utterance_id,
                "wav_path": str(meta.wav_path),
                "session_id": meta.session_id,
                "dialog_id": meta.dialog_id,
                "dialog_type": meta.dialog_type,
                "speaker_id": meta.speaker_id,
                "speaker_gender": meta.speaker_gender,
                "start_time": meta.start_time,
                "end_time": meta.end_time,
                "duration": meta.end_time - meta.start_time,
                "emotion_label": meta.emotion_label,
                "emotion_name": meta.emotion_name,
                "valence": meta.valence,
                "activation": meta.activation,
                "dominance": meta.dominance,
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
            "speaker_id",
            "speaker_gender",
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


def make_analysis_groups(
    embedded: Sequence[EmbeddedUtterance],
    requested_groups: Sequence[str],
    dialog_type_analyses: Sequence[str],
) -> List[AnalysisGroup]:
    groups: List[AnalysisGroup] = []
    for dialog_type_analysis in dialog_type_analyses:
        if dialog_type_analysis == "combined":
            condition_indices = list(range(len(embedded)))
            prefix = ""
        else:
            condition_indices = [
                index
                for index, item in enumerate(embedded)
                if item.metadata.dialog_type == dialog_type_analysis
            ]
            prefix = f"{dialog_type_analysis}_"

        if "all" in requested_groups:
            group_name = "all" if dialog_type_analysis == "combined" else f"{prefix}all"
            groups.append(
                AnalysisGroup(
                    name=group_name,
                    dialog_type_analysis=dialog_type_analysis,
                    utterance_indices=tuple(condition_indices),
                )
            )
        if "speakers" in requested_groups:
            by_speaker: Dict[str, List[int]] = {}
            for index in condition_indices:
                item = embedded[index]
                by_speaker.setdefault(item.metadata.speaker_id, []).append(index)
            for speaker_id, indices in sorted(by_speaker.items()):
                group_name = (
                    f"speaker_{speaker_id}"
                    if dialog_type_analysis == "combined"
                    else f"{prefix}speaker_{speaker_id}"
                )
                groups.append(
                    AnalysisGroup(
                        name=group_name,
                        dialog_type_analysis=dialog_type_analysis,
                        utterance_indices=tuple(indices),
                    )
                )
    return groups


def subset_metadata(embedded: Sequence[EmbeddedUtterance], indices: Sequence[int]) -> List[IemocapUtterance]:
    return [embedded[index].metadata for index in indices]


def layer_matrix(embedded: Sequence[EmbeddedUtterance], indices: Sequence[int], layer_index: int) -> np.ndarray:
    return np.stack([embedded[index].layer_vectors[layer_index] for index in indices], axis=0)


def vad_matrix(metadata: Sequence[IemocapUtterance]) -> np.ndarray:
    return np.array([[item.valence, item.activation, item.dominance] for item in metadata], dtype=np.float32)


def compute_vad_rdm(metadata: Sequence[IemocapUtterance], metric: str) -> Tuple[np.ndarray, np.ndarray]:
    values = vad_matrix(metadata)
    standardized = StandardScaler().fit_transform(values)
    return pairwise_distances(standardized, metric=metric).astype(np.float32), standardized.astype(np.float32)


def compute_model_rdm(vectors: np.ndarray, metric: str) -> np.ndarray:
    return pairwise_distances(vectors, metric=metric).astype(np.float32)


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def compute_rsa_metrics(model_rdm: np.ndarray, vad_rdm: np.ndarray) -> Dict[str, float]:
    model_values = upper_triangle_values(model_rdm)
    vad_values = upper_triangle_values(vad_rdm)
    valid = np.isfinite(model_values) & np.isfinite(vad_values)
    model_values = model_values[valid]
    vad_values = vad_values[valid]
    if len(model_values) < 2:
        return {
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "rdm_pair_count": int(len(model_values)),
        }
    spearman = spearmanr(model_values, vad_values)
    pearson = pearsonr(model_values, vad_values)
    return {
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "rdm_pair_count": int(len(model_values)),
    }


def choose_visualization_indices(sample_count: int, max_points: int, random_state: int) -> np.ndarray:
    if sample_count <= max_points:
        return np.arange(sample_count)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(sample_count, size=max_points, replace=False))


def safe_group_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def write_square_matrix_csv(path: Path, matrix: np.ndarray, labels: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utterance_id", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[float(value) for value in row]])


def run_mds_from_rdm(rdm: np.ndarray, random_state: int) -> ProjectionResult:
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=4,
        normalized_stress="auto",
    )
    coordinates = mds.fit_transform(rdm.astype(np.float64))
    return ProjectionResult(coordinates=coordinates.astype(np.float32), stress=float(mds.stress_))


def run_umap(vectors: np.ndarray, metric: str, random_state: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    import umap

    sample_count = vectors.shape[0]
    adjusted_neighbors = min(max(2, n_neighbors), max(2, sample_count - 1))
    reducer = umap.UMAP(
        n_components=2,
        metric=metric,
        n_neighbors=adjusted_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(vectors).astype(np.float32)


def metadata_colors(metadata: Sequence[IemocapUtterance]) -> List[str]:
    return [EMOTION_COLORS.get(item.emotion_label, "#888888") for item in metadata]


def add_emotion_legend(ax, metadata: Sequence[IemocapUtterance]) -> None:
    labels = sorted({item.emotion_label for item in metadata})
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=EMOTION_COLORS.get(label, "#888888"),
            markeredgecolor="none",
            markersize=8,
            label=label,
        )
        for label in labels
    ]
    ax.legend(handles=handles, title="Emotion", loc="best", fontsize=8)


def plot_projection(
    coordinates: np.ndarray,
    metadata: Sequence[IemocapUtterance],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=metadata_colors(metadata),
        s=28,
        alpha=0.75,
        edgecolors="none",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    add_emotion_legend(ax, metadata)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rdm_scatter(
    model_rdm: np.ndarray,
    vad_rdm: np.ndarray,
    title: str,
    output_path: Path,
    random_state: int,
    max_points: int,
) -> None:
    model_values = upper_triangle_values(model_rdm)
    vad_values = upper_triangle_values(vad_rdm)
    valid = np.isfinite(model_values) & np.isfinite(vad_values)
    model_values = model_values[valid]
    vad_values = vad_values[valid]
    if len(model_values) > max_points:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(model_values), size=max_points, replace=False)
        model_values = model_values[indices]
        vad_values = vad_values[indices]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(vad_values, model_values, s=8, alpha=0.25, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel("VAD distance")
    ax.set_ylabel("Model distance")
    ax.grid(alpha=0.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_coordinates_csv(
    path: Path,
    coordinates: np.ndarray,
    metadata: Sequence[IemocapUtterance],
    x_name: str,
    y_name: str,
) -> None:
    rows = []
    for point, item in zip(coordinates, metadata):
        rows.append(
            {
                "utterance_id": item.utterance_id,
                "speaker_id": item.speaker_id,
                "dialog_type": item.dialog_type,
                "emotion_label": item.emotion_label,
                x_name: float(point[0]),
                y_name: float(point[1]),
            }
        )
    write_dict_rows(path, rows, ["utterance_id", "speaker_id", "dialog_type", "emotion_label", x_name, y_name])


def save_analysis_layer_outputs(
    output_dir: Path,
    group: AnalysisGroup,
    layer_index: int,
    metadata: Sequence[IemocapUtterance],
    vectors: np.ndarray,
    model_rdm: np.ndarray,
    vad_rdm: np.ndarray,
    standardized_vad: np.ndarray,
    metrics: Dict[str, float],
    args: argparse.Namespace,
) -> Dict[str, object]:
    layer_dir = output_dir / "analyses" / safe_group_name(group.name) / f"layer_{layer_index:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    labels = [item.utterance_id for item in metadata]

    model_rdm_npy = layer_dir / "model_rdm.npy"
    vad_rdm_npy = layer_dir / "vad_rdm.npy"
    np.save(model_rdm_npy, model_rdm)
    np.save(vad_rdm_npy, vad_rdm)

    model_rdm_csv = layer_dir / "model_rdm.csv"
    vad_rdm_csv = layer_dir / "vad_rdm.csv"
    if not args.skip_csv_rdms:
        write_square_matrix_csv(model_rdm_csv, model_rdm, labels)
        write_square_matrix_csv(vad_rdm_csv, vad_rdm, labels)

    sampled_indices = choose_visualization_indices(len(metadata), args.viz_sample_size, args.random_state)
    sampled_metadata = [metadata[index] for index in sampled_indices]
    sampled_vectors = vectors[sampled_indices]
    sampled_model_rdm = model_rdm[np.ix_(sampled_indices, sampled_indices)]
    sampled_vad_rdm = vad_rdm[np.ix_(sampled_indices, sampled_indices)]
    sampled_vad_vectors = standardized_vad[sampled_indices]

    plot_rdm_scatter(
        sampled_model_rdm,
        sampled_vad_rdm,
        f"{group.name} layer {layer_index:02d}: model RDM vs VAD RDM",
        layer_dir / "rdm_scatter.png",
        args.random_state,
        args.viz_sample_size,
    )

    model_mds = run_mds_from_rdm(sampled_model_rdm, args.random_state)
    write_coordinates_csv(layer_dir / "model_mds_coordinates.csv", model_mds.coordinates, sampled_metadata, "mds_1", "mds_2")
    plot_projection(
        model_mds.coordinates,
        sampled_metadata,
        f"{group.name} layer {layer_index:02d} model MDS",
        "MDS-1",
        "MDS-2",
        layer_dir / "model_mds.png",
    )

    vad_mds = run_mds_from_rdm(sampled_vad_rdm, args.random_state)
    write_coordinates_csv(layer_dir / "vad_mds_coordinates.csv", vad_mds.coordinates, sampled_metadata, "mds_1", "mds_2")
    plot_projection(
        vad_mds.coordinates,
        sampled_metadata,
        f"{group.name} layer {layer_index:02d} VAD MDS",
        "MDS-1",
        "MDS-2",
        layer_dir / "vad_mds.png",
    )

    umap_path = ""
    try:
        model_umap = run_umap(
            sampled_vectors,
            args.model_distance_metric,
            args.random_state,
            args.umap_neighbors,
            args.umap_min_dist,
        )
        write_coordinates_csv(layer_dir / "model_umap_coordinates.csv", model_umap, sampled_metadata, "umap_1", "umap_2")
        plot_projection(
            model_umap,
            sampled_metadata,
            f"{group.name} layer {layer_index:02d} model UMAP",
            "UMAP-1",
            "UMAP-2",
            layer_dir / "model_umap.png",
        )
        umap_path = str(layer_dir / "model_umap.png")
    except Exception as exc:
        print(f"Skipping UMAP for {group.name} layer {layer_index:02d}: {exc}")

    vad_umap = run_umap(
        sampled_vad_vectors,
        "euclidean",
        args.random_state,
        args.umap_neighbors,
        args.umap_min_dist,
    )
    write_coordinates_csv(layer_dir / "vad_umap_coordinates.csv", vad_umap, sampled_metadata, "umap_1", "umap_2")
    plot_projection(
        vad_umap,
        sampled_metadata,
        f"{group.name} layer {layer_index:02d} VAD UMAP",
        "UMAP-1",
        "UMAP-2",
        layer_dir / "vad_umap.png",
    )

    return {
        "dialog_type_analysis": group.dialog_type_analysis,
        "analysis_group": group.name,
        "layer": layer_index,
        "utterance_count": len(metadata),
        "rdm_pair_count": metrics["rdm_pair_count"],
        "spearman_r": metrics["spearman_r"],
        "spearman_p": metrics["spearman_p"],
        "pearson_r": metrics["pearson_r"],
        "pearson_p": metrics["pearson_p"],
        "model_mds_stress": model_mds.stress,
        "vad_mds_stress": vad_mds.stress,
        "model_rdm_npy": str(model_rdm_npy),
        "vad_rdm_npy": str(vad_rdm_npy),
        "model_rdm_csv": "" if args.skip_csv_rdms else str(model_rdm_csv),
        "vad_rdm_csv": "" if args.skip_csv_rdms else str(vad_rdm_csv),
        "rdm_scatter": str(layer_dir / "rdm_scatter.png"),
        "model_mds_plot": str(layer_dir / "model_mds.png"),
        "vad_mds_plot": str(layer_dir / "vad_mds.png"),
        "model_umap_plot": umap_path,
        "vad_umap_plot": str(layer_dir / "vad_umap.png"),
    }


def plot_layer_correlations(output_path: Path, summary_rows: Sequence[Dict[str, object]]) -> None:
    by_group: Dict[str, List[Dict[str, object]]] = {}
    for row in summary_rows:
        by_group.setdefault(str(row["analysis_group"]), []).append(row)
    fig, ax = plt.subplots(figsize=(10, 6))
    for group_name, rows in sorted(by_group.items()):
        ordered = sorted(rows, key=lambda row: int(row["layer"]))
        layers = [int(row["layer"]) for row in ordered]
        spearman_values = [float(row["spearman_r"]) for row in ordered]
        ax.plot(layers, spearman_values, marker="o", linewidth=1.8, label=group_name)
    ax.set_title("Layer-wise RSA Spearman correlation")
    ax.set_xlabel("emotion2vec layer")
    ax.set_ylabel("Spearman r")
    ax.axhline(0, color="#BBBBBB", linewidth=0.8)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_group_layer(
    output_dir: Path,
    group: AnalysisGroup,
    layer_index: int,
    embedded: Sequence[EmbeddedUtterance],
    args: argparse.Namespace,
) -> Dict[str, object]:
    metadata = subset_metadata(embedded, group.utterance_indices)
    if len(metadata) < 3:
        raise RuntimeError(f"Analysis group {group.name} needs at least 3 utterances.")
    vectors = layer_matrix(embedded, group.utterance_indices, layer_index)
    model_rdm = compute_model_rdm(vectors, args.model_distance_metric)
    vad_rdm, standardized_vad = compute_vad_rdm(metadata, args.vad_distance_metric)
    metrics = compute_rsa_metrics(model_rdm, vad_rdm)
    return save_analysis_layer_outputs(
        output_dir,
        group,
        layer_index,
        metadata,
        vectors,
        model_rdm,
        vad_rdm,
        standardized_vad,
        metrics,
        args,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    utterances = collect_iemocap_utterances(
        args.data_dir,
        args.sessions,
        args.emotion_labels,
        args.dialog_types,
        args.include_xxx,
        args.max_utterances,
    )
    print(f"Collected {len(utterances)} IEMOCAP utterances")

    embedded = build_embeddings(utterances, args.model_name, args.device, args.hub)
    layers = validate_layers(args.layers, len(embedded[0].layer_vectors))
    write_metadata(args.output_dir / "utterance_metadata.csv", embedded)

    groups = make_analysis_groups(embedded, args.analysis_groups, args.dialog_type_analyses)
    summary_rows: List[Dict[str, object]] = []
    for group in groups:
        if len(group.utterance_indices) < 3:
            print(f"Skipping analysis group {group.name}: need at least 3 utterances")
            continue
        print(f"Running analysis group {group.name} with {len(group.utterance_indices)} utterances")
        for layer_index in layers:
            print(f"  layer {layer_index:02d}")
            summary_rows.append(run_group_layer(args.output_dir, group, layer_index, embedded, args))

    write_dict_rows(
        args.output_dir / "summary_metrics.csv",
        summary_rows,
        [
            "dialog_type_analysis",
            "analysis_group",
            "layer",
            "utterance_count",
            "rdm_pair_count",
            "spearman_r",
            "spearman_p",
            "pearson_r",
            "pearson_p",
            "model_mds_stress",
            "vad_mds_stress",
            "model_rdm_npy",
            "vad_rdm_npy",
            "model_rdm_csv",
            "vad_rdm_csv",
            "rdm_scatter",
            "model_mds_plot",
            "vad_mds_plot",
            "model_umap_plot",
            "vad_umap_plot",
        ],
    )
    plot_layer_correlations(args.output_dir / "layer_correlation_plot.png", summary_rows)
    print(f"Saved IEMOCAP RSA outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
