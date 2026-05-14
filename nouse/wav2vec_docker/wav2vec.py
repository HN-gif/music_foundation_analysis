from functools import lru_cache
from pathlib import Path

import librosa
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor


DEFAULT_MODEL_NAME = "facebook/wav2vec2-base-960h"


@lru_cache(maxsize=1)
def load_wav2vec(model_name: str = DEFAULT_MODEL_NAME):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device


def get_hidden_states(audio_path: str | Path, model_name: str = DEFAULT_MODEL_NAME):
    processor, model, device = load_wav2vec(model_name)
    speech, _ = librosa.load(str(audio_path), sr=16000)

    inputs = processor(
        speech,
        sampling_rate=16000,
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

    return outputs.hidden_states


def extract_mean_pooled_embedding(
    audio_path: str | Path,
    layer_index: int = 6,
    model_name: str = DEFAULT_MODEL_NAME,
):
    hidden_states = get_hidden_states(audio_path, model_name=model_name)

    if not 0 <= layer_index < len(hidden_states):
        raise ValueError(
            f"layer_index must be between 0 and {len(hidden_states) - 1}, "
            f"but got {layer_index}."
        )

    # (batch, time, hidden) -> (hidden,)
    embedding = hidden_states[layer_index][0].mean(dim=0)
    return embedding.detach().cpu().numpy()
