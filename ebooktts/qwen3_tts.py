# A very simple wrapper to abstract stuff away.

import io
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io.wavfile
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
from qwen_tts.cli.demo import _normalize_audio
from tqdm import tqdm


# https://github.com/QwenLM/Qwen3-TTS/blob/3b30a4e509657d8df1387554394141a0d68be4f0/qwen_tts/cli/demo.py#L528-L563
# Apache 2 license, just like this.
def load_voice(file_obj: Path):
    path = (
        getattr(file_obj, "name", None)
        or getattr(file_obj, "path", None)
        or str(file_obj)
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "items" not in payload:
        return None, "Invalid file format (文件格式不正确)."

    items_raw = payload["items"]
    if not isinstance(items_raw, list) or len(items_raw) == 0:
        return None, "Empty voice items (音色为空)."

    items: list[VoiceClonePromptItem] = []
    for d in items_raw:
        if not isinstance(d, dict):
            return None, "Invalid item format in file (文件内部格式错误)."
        ref_code = d.get("ref_code", None)
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        ref_spk = d.get("ref_spk_embedding", None)
        if ref_spk is None:
            return None, "Missing ref_spk_embedding (缺少说话人向量)."
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)

        items.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                icl_mode=bool(
                    d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))
                ),
                ref_text=d.get("ref_text", None),
            )
        )
    return items


def save_voice(out_path, voice_items):
    from dataclasses import asdict

    payload = {
        "items": [asdict(it) for it in voice_items],
    }
    torch.save(payload, out_path)


def load_text_file(path):
    with open(path) as f:
        return f.read().strip()


def load_audio_files_to_qwentts_b64(paths):
    data = None
    use_samplerate = None
    for path in paths:
        waveform, samplerate = sf.read(path)
        if use_samplerate is None:
            use_samplerate = samplerate
        else:
            if use_samplerate != samplerate:
                raise ValueError("got two samplerates")
        if data is None:
            data = waveform
        else:
            data = np.vstack([data, waveform])

    data = np.mean(data, axis=-1)

    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, use_samplerate, data)
    import base64

    as_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    z = "data:audio," + as_b64
    return z


class AudioObject:
    def __init__(self, waveform, sample_rate):
        self._waveform = waveform
        self._sample_rate = sample_rate

    def save(self, path: Path):
        # should switch to soundfile.
        scipy.io.wavfile.write(path, self._sample_rate, self._waveform)

    @staticmethod
    def from_list(z: "list[AudioObject]"):
        # should just be a concatenation.

        sample_rate = z[0]._sample_rate
        data = z[0]._waveform
        for more_data in z[1:]:
            if more_data._sample_rate != sample_rate:
                raise ValueError(
                    f"Got two sample rates for concat {sample_rate} and {more_data._sample_rate}"
                )
            data = np.hstack([data, more_data._waveform])
        return AudioObject(data, sample_rate)


class Qwen3TTSInterface:
    def __init__(
        self,
        model_path: Path,
        device: str,
        dtype: str,
        attn_impl: str | None,
    ):
        self._tts: Qwen3TTSModel = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        self._voice = None

    def load_voice(self, voice_path: str):
        self._voice = load_voice(voice_path)

    def generate(self, text, language="Auto", **kwargs):
        wavs, sr = self._tts.generate_voice_clone(
            text=text.strip(),
            language=language.lower(),
            voice_clone_prompt=self._voice,
            **kwargs,
        )
        return AudioObject(wavs[0], sr)

    def generate_chunked_progress(self, list_of_texts, **kwargs):
        audio_segments = []
        for text in tqdm(list_of_texts):
            audio_segments.append(self.generate(text, **kwargs))

        return AudioObject.from_list(audio_segments)

    def voice_clone(self, audio_and_text_pairs, use_xvec=False):
        audios = []
        texts = []
        for audio_path, text_path in audio_and_text_pairs:
            # Defer loading of file to the the module itself, such that it doesn't try to assign to a tuple.
            audio_data = audio_path
            text_data = load_text_file(text_path)
            audios.append(audio_data)
            texts.append(text_data)
        combined_audio = load_audio_files_to_qwentts_b64(audios)
        combined_text = "\n\n".join(texts)
        items = self._tts.create_voice_clone_prompt(
            ref_audio=combined_audio,
            ref_text=combined_text,
            x_vector_only_mode=bool(use_xvec),
        )
        print(items)
        return items


if __name__ == "__main__":
    import os

    voice_path = os.environ["QWEN_TTS_VOICE"]
    model_path = os.environ["QWEN_TTS_BASE_MODEL"]
    device = "cuda:0"
    dtype = torch.bfloat16
    z = Qwen3TTSInterface(
        model_path=model_path,
        device=device,
        dtype=dtype,
        attn_impl=None,
    )
    z.load_voice(voice_path=str(voice_path))

    audio = z.generate("hello there, how are you")
    audio.save("/tmp/first_tts.wav")
