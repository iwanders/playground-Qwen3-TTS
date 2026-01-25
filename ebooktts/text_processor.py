import copy
import json
import os
import sys
from pathlib import Path

import ollama
from ollama import ChatResponse
from pydantic import BaseModel, ValidationError

# export OLLAMA_CONTEXT_LENGTH=8192
# export OLLAMA_KV_CACHE_TYPE=q8_0
# qwen3:8b

OLLAMA_MODEL_TO_USE = os.environ.get("OLLAMA_MODEL_TO_USE", "qwen3:8b")


class OllamaCache:
    def __init__(self, path="/tmp/ebooktts_cache/"):
        self._path = Path(path)
        self._cache_file = self._path / "ollama_cache.json"
        self._cache = self.load_cache()

    def load_cache(self):
        if self._cache_file.exists():
            try:
                with open(self._cache_file) as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Failed to open file {e}")

                self._cache_file.unlink()
                return dict()
        else:
            return dict()

    def save_cache(self):
        self._cache_file.parent.mkdir(exist_ok=True, parents=True)
        with open(self._cache_file, "w") as f:
            json.dump(self._cache, f, indent=1)

    def make_key(self, kwargs):
        return json.dumps(dict(kwargs), indent=1, sort_keys=True)

    def retrieve(self, kwargs):
        key = self.make_key(kwargs)
        if key in self._cache:
            # return a copy to avoid the cache getting modified.
            return copy.deepcopy(self._cache[key])

    def insert(self, key, response):
        self._cache[key] = response
        self.save_cache()

    def chat(self, **kwargs):
        cache_hit = self.retrieve(kwargs)
        if cache_hit is not None:
            return cache_hit
        # Need to do work.
        #
        response: ChatResponse = ollama.chat(**kwargs)
        content = response.message.content
        key = self.make_key(kwargs)
        self.insert(key, content)
        return copy.deepcopy(content)


cache = OllamaCache()


class Section(BaseModel):
    ids: list[int]
    reasoning: str

    def get_strings(self, numbered_lines):
        lookup = dict(numbered_lines)
        return [lookup[i] for i in self.ids]

    def get_numbered_lines(self, numbered_lines):
        lookup = dict(numbered_lines)
        return [(i, lookup[i]) for i in self.ids]


class SectionList(BaseModel):
    sections: list[Section]


# Logical sections may be too long.
# TTS starts degrading after 30s with custom voice? ... limit to ~60  words


class TextProcessor:
    def __init__(self, input_text):
        input_lines = input_text
        if isinstance(input_text, str):
            input_lines = input_text.split("\n")
        # Make numbered lines out of this,  + 1 here such that it matches line numbers from the export.
        self._numbered_lines = list((k + 1, v) for k, v in enumerate(input_lines))

        self._sections = []

    def create_sections(self):
        # Iterate over the seed, such that if the model doesn't produce json, or drops ids, we try the next seed.
        for seed in range(1, 3):
            try:
                sections = self.send_prompt_for_sections(
                    self._numbered_lines, seed=seed
                )
                # Verify that it did not actually lose any ids, or created duplicates.
                ids = []
                for s in sections:
                    ids.extend(s.ids)
                expected = list(range(1, len(self._numbered_lines) + 1))
                if ids == expected:
                    # Splendid, we're all good.
                    self._sections = sections
                    return
                else:
                    print(f"Ids were not consectutive, got {ids}, expected {expected}")

                self._sections = sections
            except ValidationError as e:
                print(f"Invalid json: {e}")

    def get_sections(self):
        return self._sections

    @staticmethod
    def send_prompt_for_sections(numbered_lines, seed=1):
        payload = json.dumps(
            list({"id": k, "text": v} for k, v in numbered_lines),
            indent=2,
        )

        # Using id's is much lighter than actually having the text in the sections.
        # It also kinda failed at splitting on who is speaking... often having 'said Foo in a grumpy voice' etc instead
        # of just having the section that was the quote.
        # Qwen3-TTS does a reasonable job at identifying quotes, so we don't _actually_ need to break on quotes.

        response = cache.chat(
            model=OLLAMA_MODEL_TO_USE,
            messages=[
                {
                    "role": "system",
                    "content": """
                    Provided are the text lines of a book in json format, each line has an id.
                    Split the lines into a logical sections by stating which ids are present in each section.
                    Briefly explain the reasoning for each section and why it makes a logical section.
                    Each section should be short enough to be spoken out loud in one breath.
                    Make sure no text (or ids) are lost. 
                    Respond in json.
                    """,
                },
                {
                    "role": "user",
                    "content": payload,
                },
            ],
            format=SectionList.model_json_schema(),
            # Make things completely deterministic, such that if weird things happen, I can at least reproduce weird things.
            options={"temperature": 0, "seed": seed},
        )

        response = SectionList.model_validate_json(response)

        return response.sections


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        d = f.read()

    if False:
        parts = d.split("\n")[0:5]
        d = "\n".join(parts)
    print(d)

    z = TextProcessor(d)
    z.create_sections()
    for s in z.get_sections():
        print(s)
