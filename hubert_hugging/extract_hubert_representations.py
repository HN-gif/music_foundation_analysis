#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel, AutoModelForCTC

from audio_dataset import collect_ravdess_files, parse_ravdess_path


TARGET_SAMPLE_RATE = 16_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    description: str


MODEL_SPECS: Dict[str, ModelSpec] = {
    "base-ls960": ModelSpec(
        key="base-ls960",
        model_id="facebook/hubert-base-ls960",
        description="HuBERT Base pretrained on LibriSpeech 960h",
    ),
    "large-ll60k": ModelSpec(
        key="large-ll60k",
        model_id="facebook/hubert-large-ll60k",
        description="HuBERT Large pretrained on Libri-Light 60k hours",
    ),
    "large-ls960-ft": ModelSpec(
        key="large-ls960-ft",
        model_id="facebook/hubert-large-ls960-ft",
        description="HuBERT Large fine-tuned on LibriSpeech 960h",
    ),
    "xlarge-ls960-ft": ModelSpec(
        key="xlarge-ls960-ft",
        model_id="facebook/hubert-xlarge-ls960-ft",
        description="HuBERT XLarge fine-tuned on LibriSpeech 960h",
    ),
}

MODEL_ALIASES = {
    "hubert-base-ls960": "base-ls960",
    "hubert-large-ll60k": "large-ll60k",
    "hubert-large-ls960-ft": "large-ls960-ft",
    "hubert-xlarge-ls960-ft": "xlarge-ls960-ft",
    "163": "xlarge-ls960-ft",
    "hubert-163": "xlarge-ls960-ft",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract HuBERT internal representations from wav files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory containing .wav files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "hubert_hugging" / "outputs",
        help="Directory to store representation files.",
    )
    parser.add_argument(
        "--model",
        choices=["all", *sorted(MODEL_SPECS.keys()), *sorted(MODEL_ALIASES.keys())],
        default="all",
        help="Model key or alias. Use 'all' to run every supported model.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device. Example: cpu, cuda, cuda:0",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of wav files to process.",
    )
    parser.add_argument(
        "--pattern",
        default="*.wav",
        help="Filename pattern under input-dir. Default: *.wav",
    )
    parser.add_argument(
        "--intensity-codes",
        nargs="+",
        default=None,
        help="Optional RAVDESS intensity codes to keep. Example: 01 02",
    )
    parser.add_argument(
        "--statement-codes",
        nargs="+",
        default=None,
        help="Optional RAVDESS statement codes to keep. Example: 01",
    )
    parser.add_argument(
        "--repetition-codes",
        nargs="+",
        default=None,
        help="Optional RAVDESS repetition codes to keep. Example: 01 02",
    )
    parser.add_argument(
        "--emotion-codes",
        nargs="+",
        default=None,
        help="Optional RAVDESS emotion codes to keep. Example: 03 04 05",
    )
    parser.add_argument(
        "--genders",
        nargs="+",
        choices=["male", "female"],
        default=None,
        help="Optional speaker genders to keep.",
    )
    parser.add_argument(
        "--actor-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional actor IDs to keep. Example: 1 2 12",
    )
    parser.add_argument(
        "--save-all-hidden-states",
        action="store_true",
        help="Store every hidden state layer in addition to the last layer.",
    )
    parser.add_argument(
        "--save-logits",
        action="store_true",
        help="Store logits for fine-tuned CTC checkpoints.",
    )
    return parser.parse_args()


def resolve_model_keys(model_arg: str) -> List[str]:
    if model_arg == "all":
        return list(MODEL_SPECS.keys())
    if model_arg in MODEL_SPECS:
        return [model_arg]
    if model_arg in MODEL_ALIASES:
        return [MODEL_ALIASES[model_arg]]
    raise KeyError(f"Unsupported model key: {model_arg}")


def list_audio_files(
    input_dir: Path,
    pattern: str,
    limit: Optional[int],
    intensity_codes: Optional[List[str]],
    statement_codes: Optional[List[str]],
    repetition_codes: Optional[List[str]],
    emotion_codes: Optional[List[str]],
    genders: Optional[List[str]],
    actor_ids: Optional[List[int]],
) -> List[Path]:
    records = collect_ravdess_files(
        input_dir,
        pattern=pattern,
        intensity_codes=intensity_codes,
        statement_codes=statement_codes,
        repetition_codes=repetition_codes,
        emotion_codes=emotion_codes,
        genders=genders,
        actor_ids=actor_ids,
        max_files=limit,
    )
    return [record.path for record in records]


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = resample_audio(audio, sample_rate, TARGET_SAMPLE_RATE)
        sample_rate = TARGET_SAMPLE_RATE
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio, sample_rate


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    duration = audio.shape[0] / float(source_rate)
    target_length = max(int(round(duration * target_rate)), 1)
    source_positions = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
    resampled = np.interp(target_positions, source_positions, audio)
    return resampled.astype(np.float32, copy=False)


def infer_model_loader(model_id: str):
    config = AutoConfig.from_pretrained(model_id)
    architectures = config.architectures or []
    if any("ForCTC" in architecture for architecture in architectures):
        return AutoModelForCTC
    return AutoModel


def to_serializable_shape(array: Optional[np.ndarray]) -> Optional[List[int]]:
    if array is None:
        return None
    return [int(dim) for dim in array.shape]


def save_representation(
    destination: Path,
    last_hidden_state: np.ndarray,
    utterance_embedding: np.ndarray,
    extract_features: Optional[np.ndarray],
    hidden_states: Optional[np.ndarray],
    logits: Optional[np.ndarray],
) -> None:
    payload = {
        "last_hidden_state": last_hidden_state.astype(np.float32),
        "utterance_embedding": utterance_embedding.astype(np.float32),
    }
    if extract_features is not None:
        payload["extract_features"] = extract_features.astype(np.float32)
    if hidden_states is not None:
        payload["hidden_states"] = hidden_states.astype(np.float32)
    if logits is not None:
        payload["logits"] = logits.astype(np.float32)
    np.savez_compressed(destination, **payload)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_output_path(output_dir: Path, model_key: str, input_dir: Path, audio_path: Path) -> Path:
    relative_path = audio_path.relative_to(input_dir)
    return output_dir / model_key / relative_path.with_suffix(".npz")


def stack_hidden_states(hidden_states: Iterable[torch.Tensor]) -> np.ndarray:
    stacked = torch.stack([layer[0].detach().cpu() for layer in hidden_states], dim=0)
    return stacked.numpy().astype(np.float32)


def extract_one_file(
    audio_path: Path,
    input_dir: Path,
    output_dir: Path,
    model_key: str,
    model_id: str,
    feature_extractor,
    model,
    device: str,
    save_all_hidden_states: bool,
    save_logits: bool,
) -> Dict[str, object]:
    waveform, sample_rate = load_audio(audio_path)
    inputs = feature_extractor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=False,
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states_tuple = outputs.hidden_states
    if hidden_states_tuple is None:
        raise RuntimeError(f"Model {model_id} did not return hidden states.")

    last_hidden_state = hidden_states_tuple[-1][0].detach().cpu().numpy().astype(np.float32)
    utterance_embedding = last_hidden_state.mean(axis=0).astype(np.float32)

    extract_features = None
    if hasattr(outputs, "extract_features") and outputs.extract_features is not None:
        extract_features = (
            outputs.extract_features[0].detach().cpu().numpy().astype(np.float32)
        )

    hidden_states = None
    if save_all_hidden_states:
        hidden_states = stack_hidden_states(hidden_states_tuple)

    logits = None
    if save_logits and hasattr(outputs, "logits") and outputs.logits is not None:
        logits = outputs.logits[0].detach().cpu().numpy().astype(np.float32)

    output_path = build_output_path(output_dir, model_key, input_dir, audio_path)
    ensure_parent(output_path)
    save_representation(
        destination=output_path,
        last_hidden_state=last_hidden_state,
        utterance_embedding=utterance_embedding,
        extract_features=extract_features,
        hidden_states=hidden_states,
        logits=logits,
    )

    num_input_samples = int(waveform.shape[0])
    num_frames = int(last_hidden_state.shape[0])
    approx_stride_ms = float(num_input_samples / sample_rate / max(num_frames, 1) * 1000.0)

    metadata = {
        "input_path": str(audio_path),
        "output_path": str(output_path),
        "model_key": model_key,
        "model_id": model_id,
        "sample_rate": sample_rate,
        "duration_sec": round(num_input_samples / float(sample_rate), 6),
        "num_input_samples": num_input_samples,
        "num_frames": num_frames,
        "hidden_size": int(last_hidden_state.shape[1]),
        "approx_frame_stride_ms": round(approx_stride_ms, 6),
        "saved_arrays": {
            "last_hidden_state": to_serializable_shape(last_hidden_state),
            "utterance_embedding": to_serializable_shape(utterance_embedding),
            "extract_features": to_serializable_shape(extract_features),
            "hidden_states": to_serializable_shape(hidden_states),
            "logits": to_serializable_shape(logits),
        },
        "ravdess": parse_ravdess_path(audio_path).to_metadata(),
    }
    return metadata


def write_manifest(output_dir: Path, model_key: str, manifest_rows: List[Dict[str, object]]) -> None:
    manifest_path = output_dir / model_key / "manifest.jsonl"
    ensure_parent(manifest_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    model_keys = resolve_model_keys(args.model)
    audio_files = list_audio_files(
        args.input_dir,
        args.pattern,
        args.limit,
        args.intensity_codes,
        args.statement_codes,
        args.repetition_codes,
        args.emotion_codes,
        args.genders,
        args.actor_ids,
    )

    if not audio_files:
        raise FileNotFoundError(
            f"No audio files matching {args.pattern!r} were found under {args.input_dir}"
        )

    print(f"Found {len(audio_files)} audio files under {args.input_dir}")

    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        print(f"\nLoading {spec.model_id} on {args.device}")
        feature_extractor = AutoFeatureExtractor.from_pretrained(spec.model_id)
        model_loader = infer_model_loader(spec.model_id)
        model = model_loader.from_pretrained(spec.model_id).to(args.device)
        model.eval()

        manifest_rows: List[Dict[str, object]] = []

        for index, audio_path in enumerate(audio_files, start=1):
            print(f"[{model_key}] {index}/{len(audio_files)} {audio_path}")
            row = extract_one_file(
                audio_path=audio_path,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                model_key=model_key,
                model_id=spec.model_id,
                feature_extractor=feature_extractor,
                model=model,
                device=args.device,
                save_all_hidden_states=args.save_all_hidden_states,
                save_logits=args.save_logits,
            )
            manifest_rows.append(row)

        write_manifest(args.output_dir, model_key, manifest_rows)
        print(f"Saved manifest to {args.output_dir / model_key / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
