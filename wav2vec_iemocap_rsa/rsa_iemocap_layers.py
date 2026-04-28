#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
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
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2Model, Wav2Vec2Processor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DATA_DIR = Path("/home/takamichi-lab-pc07/research/data/IEMOCAP_full_release")
DEFAULT_OUTPUT_DIR = Path("/home/takamichi-lab-pc07/research/wav2vec_iemocap_rsa/outputs")
DEFAULT_MODEL_NAME = "facebook/wav2vec2-base-960h"
TARGET_SAMPLE_RATE = 16_000
DEFAULT_RANDOM_STATE = 42
ANNOTATION_PATTERN = re.compile(
    r"^\[(?P<start>[0-9.]+) - (?P<end>[0-9.]+)\]\s+"
    r"(?P<utterance_id>\S+)\s+"
    r"(?P<emotion_label>\S+)\s+"
    r"\[(?P<valence>[0-9.]+), (?P<activation>[0-9.]+), (?P<dominance>[0-9.]+)\]"
)
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
class IemocapUtterance:
    utterance_id: str
    wav_path: Path
    session_id: str
    dialog_id: str
    dialog_type: str
    speaker_id: str
    speaker_gender: str
    start_time: float
    end_time: float
    emotion_label: str
    emotion_name: str
    valence: float
    activation: float
    dominance: float


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
            "Run layer-wise RSA between wav2vec2 utterance RDMs and "
            "IEMOCAP human VAD RDMs."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory of IEMOCAP_full_release.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where RSA outputs are saved.",
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
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Optional layer indices to analyze. By default all hidden-state layers are used.",
    )
    parser.add_argument(
        "--model-distance-metric",
        default="cosine",
        help="Distance metric for wav2vec utterance vectors.",
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
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="Optional subset of sessions. Example: Session1 Session2 or 1 2.",
    )
    parser.add_argument(
        "--emotion-labels",
        nargs="+",
        default=None,
        help="Optional subset of emotion labels. Example: ang sad neu.",
    )
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


def normalize_session_filters(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not values:
        return None
    normalized = set()
    for value in values:
        if value.startswith("Session"):
            normalized.add(value)
        else:
            normalized.add(f"Session{int(value)}")
    return normalized


def infer_dialog_type(dialog_id: str) -> str:
    if "_impro" in dialog_id:
        return "impro"
    if "_script" in dialog_id:
        return "script"
    return "unknown"


def parse_annotation_file(path: Path, data_dir: Path) -> List[IemocapUtterance]:
    session_id = path.parts[path.parts.index(data_dir.name) + 1]
    rows: List[IemocapUtterance] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = ANNOTATION_PATTERN.match(line)
        if match is None:
            continue

        utterance_id = match.group("utterance_id")
        dialog_id = utterance_id.rsplit("_", 1)[0]
        dialog_type = infer_dialog_type(dialog_id)
        speaker_gender = utterance_id.split("_")[-1][0]
        speaker_id = f"{session_id}_{speaker_gender}"
        wav_matches = list((data_dir / session_id / "sentences" / "wav").glob(f"*/{utterance_id}.wav"))
        if len(wav_matches) != 1:
            raise FileNotFoundError(
                f"Expected one wav for utterance {utterance_id}, found {len(wav_matches)}"
            )

        emotion_label = match.group("emotion_label")
        rows.append(
            IemocapUtterance(
                utterance_id=utterance_id,
                wav_path=wav_matches[0],
                session_id=session_id,
                dialog_id=dialog_id,
                dialog_type=dialog_type,
                speaker_id=speaker_id,
                speaker_gender=speaker_gender,
                start_time=float(match.group("start")),
                end_time=float(match.group("end")),
                emotion_label=emotion_label,
                emotion_name=EMOTION_NAMES.get(emotion_label, "unknown"),
                valence=float(match.group("valence")),
                activation=float(match.group("activation")),
                dominance=float(match.group("dominance")),
            )
        )
    return rows


def collect_iemocap_utterances(
    data_dir: Path,
    sessions: Optional[Sequence[str]],
    emotion_labels: Optional[Sequence[str]],
    dialog_types: Optional[Sequence[str]],
    include_xxx: bool,
    max_utterances: Optional[int],
) -> List[IemocapUtterance]:
    session_filter = normalize_session_filters(sessions)
    emotion_filter = set(emotion_labels) if emotion_labels else None
    dialog_type_filter = set(dialog_types) if dialog_types else None
    utterances: List[IemocapUtterance] = []

    for annotation_path in sorted(data_dir.glob("Session*/dialog/EmoEvaluation/*.txt")):
        session_id = annotation_path.parts[annotation_path.parts.index(data_dir.name) + 1]
        if session_filter is not None and session_id not in session_filter:
            continue
        for item in parse_annotation_file(annotation_path, data_dir):
            if item.emotion_label == "xxx" and not include_xxx:
                continue
            if emotion_filter is not None and item.emotion_label not in emotion_filter:
                continue
            if dialog_type_filter is not None and item.dialog_type not in dialog_type_filter:
                continue
            utterances.append(item)
            if max_utterances is not None and len(utterances) >= max_utterances:
                return utterances

    if not utterances:
        raise RuntimeError("No IEMOCAP utterances matched the requested filters.")
    return utterances


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


def extract_layer_embeddings(audio_path: Path, model_name: str, device_name: str) -> Tuple[np.ndarray, ...]:
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


def build_embeddings(
    utterances: Sequence[IemocapUtterance],
    model_name: str,
    device_name: str,
) -> List[EmbeddedUtterance]:
    embedded: List[EmbeddedUtterance] = []
    for index, utterance in enumerate(utterances, start=1):
        print(f"[{index}/{len(utterances)}] extracting hidden states: {utterance.utterance_id}")
        embedded.append(
            EmbeddedUtterance(
                metadata=utterance,
                layer_vectors=extract_layer_embeddings(utterance.wav_path, model_name, device_name),
            )
        )
    return embedded


def validate_layers(requested_layers: Optional[Sequence[int]], available_layers: int) -> List[int]:
    if requested_layers is None:
        return list(range(available_layers))
    invalid = [layer for layer in requested_layers if layer < 0 or layer >= available_layers]
    if invalid:
        raise ValueError(f"Invalid layer indices {invalid}; available range is 0..{available_layers - 1}")
    return list(dict.fromkeys(requested_layers))


def write_dict_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
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
                group_name = f"speaker_{speaker_id}" if dialog_type_analysis == "combined" else f"{prefix}speaker_{speaker_id}"
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
    return np.array(
        [[item.valence, item.activation, item.dominance] for item in metadata],
        dtype=np.float32,
    )


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


def plot_projection(
    coordinates: np.ndarray,
    metadata: Sequence[IemocapUtterance],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = metadata_colors(metadata)
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=colors,
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
    write_coordinates_csv(
        layer_dir / "model_mds_coordinates.csv",
        model_mds.coordinates,
        sampled_metadata,
        "mds_1",
        "mds_2",
    )
    plot_projection(
        model_mds.coordinates,
        sampled_metadata,
        f"{group.name} layer {layer_index:02d} model MDS",
        "MDS-1",
        "MDS-2",
        layer_dir / "model_mds.png",
    )

    vad_mds = run_mds_from_rdm(sampled_vad_rdm, args.random_state)
    write_coordinates_csv(
        layer_dir / "vad_mds_coordinates.csv",
        vad_mds.coordinates,
        sampled_metadata,
        "mds_1",
        "mds_2",
    )
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
        write_coordinates_csv(
            layer_dir / "model_umap_coordinates.csv",
            model_umap,
            sampled_metadata,
            "umap_1",
            "umap_2",
        )
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
    write_coordinates_csv(
        layer_dir / "vad_umap_coordinates.csv",
        vad_umap,
        sampled_metadata,
        "umap_1",
        "umap_2",
    )
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
    ax.set_xlabel("wav2vec2 layer")
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

    embedded = build_embeddings(utterances, args.model_name, args.device)
    available_layers = len(embedded[0].layer_vectors)
    layers = validate_layers(args.layers, available_layers)
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

    summary_path = args.output_dir / "summary_metrics.csv"
    write_dict_rows(
        summary_path,
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
