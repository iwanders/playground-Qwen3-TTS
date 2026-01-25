# A very simple wrapper to abstract stuff away.

from pathlib import Path

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
from scipy.io.wavfile import write


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


class AudioObject:
    def __init__(self, waveform, sample_rate):
        self._waveform = waveform
        self._sample_rate = sample_rate

    def save(self, path: Path):
        write(path, self._sample_rate, self._waveform)

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
        voice_path: Path,
        device: str,
        dtype: str,
        attn_impl: str,
    ):
        self._tts: Qwen3TTSModel = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )

        self._voice = load_voice(voice_path)

    def generate(self, text, language="Auto", **kwargs):
        wavs, sr = self._tts.generate_voice_clone(
            text=text.strip(),
            language=language.lower(),
            voice_clone_prompt=self._voice,
            **kwargs,
        )
        return AudioObject(wavs[0], sr)


if __name__ == "__main__":
    import os

    voice_path = os.environ["QWEN_TTS_VOICE"]
    model_path = os.environ["QWEN_TTS_BASE_MODEL"]
    device = "cuda:0"
    dtype = torch.bfloat16
    z = Qwen3TTSInterface(
        model_path=model_path,
        voice_path=voice_path,
        device=device,
        dtype=dtype,
        attn_impl=None,
    )

    audio = z.generate("hello there, how are you")
    audio.save("/tmp/first_tts.wav")
