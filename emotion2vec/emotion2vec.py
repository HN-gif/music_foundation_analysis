#!/usr/bin/env python3
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from funasr import AutoModel

DEFAULT_MODEL_NAME = "emotion2vec/emotion2vec_base"
DEFAULT_HUB = "hf"
TARGET_SAMPLE_RATE = 16_000


@lru_cache(maxsize=4)
def load_emotion2vec_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device_name: str = "cpu",
    hub: str = DEFAULT_HUB,
) -> AutoModel:
    auto_model = AutoModel(
        model=model_name,
        hub=hub,
        device=device_name,
        disable_update=True,
    )
    auto_model.model.eval()
    return auto_model


def load_audio(wav_path: Path | str, target_sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    waveform, _ = librosa.load(Path(wav_path), sr=target_sample_rate, mono=True)
    return waveform.astype(np.float32, copy=False)


def _trim_extra_tokens(hidden_state: torch.Tensor, num_extra_tokens: int) -> torch.Tensor:
    if num_extra_tokens <= 0:
        return hidden_state
    return hidden_state[:, num_extra_tokens:, :]


def extract_hidden_states(
    wav_path: Path | str,
    model_name: str = DEFAULT_MODEL_NAME,
    device_name: str = "cpu",
    hub: str = DEFAULT_HUB,
) -> Tuple[np.ndarray, ...]:
    auto_model = load_emotion2vec_model(model_name=model_name, device_name=device_name, hub=hub)
    model = auto_model.model
    feature_extractor = model.modality_encoders["AUDIO"]
    num_extra_tokens = int(feature_extractor.modality_cfg.num_extra_tokens)

    waveform = load_audio(wav_path)
    source = torch.from_numpy(waveform).to(device=device_name, dtype=torch.float32)
    if model.cfg.normalize:
        source = F.layer_norm(source, source.shape)
    source = source.view(1, -1)

    hidden_states = []
    with torch.inference_mode():
        extractor_out = feature_extractor(
            source,
            padding_mask=None,
            mask=False,
            remove_masked=False,
            clone_batch=1,
        )
        x = extractor_out["x"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias")
        alibi_scale = extractor_out.get("alibi_scale")

        hidden_states.append(_trim_extra_tokens(x, num_extra_tokens))
        if model.dropout_input is not None:
            x = model.dropout_input(x)

        for layer_index, block in enumerate(model.blocks):
            alibi_bias = masked_alibi_bias
            if alibi_bias is not None and alibi_scale is not None:
                scale = (
                    alibi_scale[layer_index]
                    if alibi_scale.size(0) > 1
                    else alibi_scale.squeeze(0)
                )
                alibi_bias = alibi_bias * scale.type_as(alibi_bias)
            x, layer_output = block(
                x,
                padding_mask=masked_padding_mask,
                alibi_bias=alibi_bias,
            )
            hidden_states.append(_trim_extra_tokens(layer_output, num_extra_tokens))

        if model.norm is not None:
            x = model.norm(x)
            hidden_states[-1] = _trim_extra_tokens(x, num_extra_tokens)

    return tuple(state.squeeze(0).cpu().numpy().astype(np.float32, copy=False) for state in hidden_states)


def extract_layer_embeddings(
    wav_path: Path | str,
    model_name: str = DEFAULT_MODEL_NAME,
    device_name: str = "cpu",
    hub: str = DEFAULT_HUB,
) -> Tuple[np.ndarray, ...]:
    hidden_states = extract_hidden_states(
        wav_path=wav_path,
        model_name=model_name,
        device_name=device_name,
        hub=hub,
    )
    return tuple(state.mean(axis=0).astype(np.float32, copy=False) for state in hidden_states)


def validate_layers(requested_layers: Optional[Sequence[int]], available_layers: int) -> list[int]:
    if requested_layers is None:
        return list(range(available_layers))
    invalid = [layer for layer in requested_layers if layer < 0 or layer >= available_layers]
    if invalid:
        raise ValueError(
            f"Invalid layer indices {invalid}; available range is 0..{available_layers - 1}."
        )
    return list(dict.fromkeys(requested_layers))
