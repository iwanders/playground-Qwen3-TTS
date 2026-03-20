# ~A very simple wrapper to abstract stuff away.~
# A not so simple anymore wrapper, that does caching on chunks and handles chunk concatenation.

import io
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io.wavfile
import soundfile as sf
import torch

# torch.set_float32_matmul_precision("high")
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
from tqdm import tqdm

from . import chunks

logger = logging.getLogger(__name__)


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

    @staticmethod
    def quiet_from(base: "AudioObject", duration: float):
        sample_rate = base.get_sample_rate()
        samples = int(duration * sample_rate)
        dtype = base._waveform.dtype
        waveform = np.zeros((samples,), dtype=dtype)
        return AudioObject(waveform, sample_rate)

    def save(self, path: Path, metadata: dict[str, str] = {}):
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)

        # https://github.com/bastibe/python-soundfile/issues/294#issuecomment-971324753
        metadata_keys = {
            "title",
            "copyright",
            "software",
            "artist",
            "comment",
            "date",
            "album",
            "license",
            "tracknumber",
            "genre",
        }
        with sf.SoundFile(path, "w", samplerate=self._sample_rate, channels=1) as file:
            file.write(self._waveform)
            for k, v in metadata.items():
                if k not in metadata_keys:
                    raise KeyError("unsupported metadata key for this file format")
                # workaround from https://github.com/bastibe/python-soundfile/issues/294#issuecomment-975768080
                file.__setattr__(k, v)

    def concat(self, other):
        if other._sample_rate != self._sample_rate:
            raise ValueError(
                f"Got two sample rates for concat {self._sample_rate} and {other._sample_rate}"
            )
        data = np.hstack([self._waveform, other._waveform])
        return AudioObject(data, self._sample_rate)

    def get_sample_rate(self):
        return self._sample_rate

    def audio_length(self):
        return (1.0 / self._sample_rate) * len(self._waveform)

    @staticmethod
    def from_list(z: "list[AudioObject]", inter_chunk_duration=0.0):
        data = z[0]
        for more_data in z[1:]:
            if inter_chunk_duration != 0.0:
                data = data.concat(AudioObject.quiet_from(data, inter_chunk_duration))
            data = data.concat(more_data)

        return data


class AudioCache:
    def __init__(self, path):
        self._path = Path(path)
        self._path.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def _key(text, voice):
        import hashlib

        # calculate the md5sum of text
        text_md5 = hashlib.md5(f"{text} - {voice}".encode("utf-8")).hexdigest()
        return text_md5

    def add_cache(self, text, voice, audio_object):
        key = AudioCache._key(text, voice)

        payload = {
            "key": key,
            "voice_path": str(voice),
            "text": text,
        }
        with open(self._path / f"{key}.json", "w") as f:
            import json

            json.dump(payload, f, indent=2)
        audio_object.save(self._path / f"{key}.flac")

    def retrieve_cache(self, text, voice):
        key = AudioCache._key(text, voice)

        file_path = self._path / f"{key}.flac"
        if file_path.is_file():
            data, sample_rate = sf.read(file_path)
            return AudioObject(data, sample_rate)


class Qwen3TTSInterface:
    def __init__(
        self,
        model_path: Path,
        device: str,
        dtype: str,
        attn_impl: str | None,
        compile=True,
        audio_cache="/tmp/cache_tts/",
    ):
        if model_path is None:
            print("Missing model path, set the env var")
            sys.exit(1)

        self._tts: Qwen3TTSModel = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        if True:
            self._tts.model = torch.compile(
                self._tts.model,
                mode="reduce-overhead",
                dynamic=False,
            )

        if audio_cache:
            self._audio_cache = AudioCache(audio_cache)
        else:
            self._audio_cache = None

    def load_voice(self, voice_path: str):
        # we MUST pass this in as str, otherwise it throws somewhere in the tts system.
        self._voice_path = voice_path
        self._voice = load_voice(voice_path)

    def generate(self, text, language="Auto", **kwargs):
        if isinstance(text, str):
            text = text.strip()

        if isinstance(text, str) and self._audio_cache is not None:
            r = self._audio_cache.retrieve_cache(text, self._voice_path)
            if r is not None:
                return r

        wavs, sr = self._tts.generate_voice_clone(
            text=text,
            language=language.lower(),
            voice_clone_prompt=self._voice,
            **kwargs,
        )
        if isinstance(text, str):
            audio_res = AudioObject(wavs[0], sr)
            if self._audio_cache is not None:
                self._audio_cache.add_cache(text, self._voice_path, audio_res)
            return audio_res
        else:
            return [AudioObject(wav, sr) for wav in wavs]

    def generate_chunked_progress(
        self, list_of_texts, inter_chunk_duration=1.0, save_intermittent=None, **kwargs
    ):
        audio_segments = []
        if False:
            # I don't have the vram to do this... :<
            batch_size = 2
            for text in tqdm(list(chunks(list_of_texts, batch_size))):
                audio_batch = self.generate(list_of_texts, **kwargs)
                for audio in audio_batch:
                    audio_segments.append(audio)
        else:
            s = time.time()
            pbar = tqdm(list_of_texts)
            for text in pbar:
                this_segment = self.generate(text, **kwargs)

                audio_segments.append(this_segment)
                total_audio_length = sum([a.audio_length() for a in audio_segments])
                realtime = total_audio_length / (time.time() - s)
                pbar.set_postfix({"realtime_factor": realtime})
                if save_intermittent:
                    combined = AudioObject.from_list(
                        audio_segments, inter_chunk_duration
                    )
                    combined.save(save_intermittent)

        return AudioObject.from_list(audio_segments, inter_chunk_duration)

    def voice_clone(self, audio_and_text_pairs, use_xvec_only=False):
        audios = []
        texts = []
        for audio_path, text_path in audio_and_text_pairs:
            if text_path is not None:
                text_data = load_text_file(text_path)
                texts.append(text_data)

            # Defer loading of file to the the module itself, such that it doesn't try to assign to a tuple.
            audio_data = audio_path
            audios.append(str(audio_data))
        combined_audio = load_audio_files_to_qwentts_b64(audios)

        combined_text = "\n\n".join(texts) if not use_xvec_only else None

        items = self._tts.create_voice_clone_prompt(
            ref_audio=combined_audio,
            ref_text=combined_text,
            x_vector_only_mode=bool(use_xvec_only),
        )
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
