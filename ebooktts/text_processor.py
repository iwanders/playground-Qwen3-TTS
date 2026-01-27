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


class InternalSection(BaseModel):
    ids: list[int]
    reasoning: str

    def is_multiple_lines(self):
        return len(self.ids) > 1

    def get_strings(self, numbered_lines):
        lookup = dict(numbered_lines)
        return [lookup[i] for i in self.ids]

    def get_numbered_lines(self, numbered_lines):
        lookup = dict(numbered_lines)
        return [(i, lookup[i]) for i in self.ids]

    def get_text(self, numbered_lines):
        return "\n".join(self.get_strings(numbered_lines))

    def get_word_count(self, numbered_lines):
        return len(self.get_text(numbered_lines).split(" "))


class SectionList(BaseModel):
    sections: list[InternalSection]


# Logical sections may be too long.
# TTS starts degrading after 30s with custom voice? ... limit to ~60  words
# Sounds like https://github.com/QwenLM/Qwen3-TTS/issues/80


class Section:
    def __init__(self, lines: list[str], reasoning: str):
        self._lines = lines
        self._reasoning = reasoning

    def get_text(self):
        return "\n".join(self._lines)

    def get_reasoning(self):
        return self._reasoning

    def __repr__(self):
        shorteneds = [s[:20] + "..." + s[-20:] for s in self._lines]
        lines = ",".join(f"'{short}'" for short in shorteneds)
        return f"<Section {lines} @ 0x{id(self):x}>"


class TextProcessor:
    def __init__(self, input_text, word_count_limit=300):
        input_lines = input_text
        if isinstance(input_text, str):
            input_lines = input_text.split("\n")
        # Make numbered lines out of this,  + 1 here such that it matches line numbers from the export.
        self._numbered_lines = list((k + 1, v) for k, v in enumerate(input_lines))
        self._word_count_limit = word_count_limit

        self._sections = []

    def create_sections(self):
        # Chop sections until they fit.
        numbered_lines = self._numbered_lines
        work_sections = []
        work_sections.append(
            InternalSection(ids=list(i for i, _ in numbered_lines), reasoning="root")
        )
        while work_sections:
            front = work_sections.pop(0)
            words = front.get_word_count(numbered_lines)
            if words > self._word_count_limit and front.is_multiple_lines():
                # print(f"splitting section {front} because it is {words} long")
                section_numbered_lines = front.get_numbered_lines(numbered_lines)

                subsections = self.work_on_subsection(section_numbered_lines)
                if not subsections:
                    raise ValueError(f"Failed to converge on a solution at {front} ")

                # Check if the LLM returned the same entry as front...
                # Now use the results
                subsections, remainder_numbered_lines = subsections
                remaining_ids = list(i for i, _ in remainder_numbered_lines)
                # print(f"Subsec: {subsections} with remainder: {remaining_ids}")
                remaining_line_section_insert = []
                if remaining_ids:
                    remaining_line_section_insert = [
                        InternalSection(
                            ids=remaining_ids,
                            reasoning="remaining",
                        )
                    ]

                if len(subsections) == 1 and subsections[0].ids == front.ids:
                    # No split happened, lets just force it in for now.
                    print(f"no split happened for {front.ids}, forcing in.")
                    self._sections.append(subsections[0])
                    subsections = []

                work_sections = (
                    subsections + remaining_line_section_insert + work_sections
                )
                # print(f"new work sections: {work_sections}")
                # print("\n\n")
                # print(f"Split into {subsections}")

            else:
                # this is good, move it to sections.
                print(f"Section {front} is ready to go")
                self._sections.append(front)
        # We are at the end, lets do a sanity check!
        ids = []
        for s in self._sections:
            ids.extend(s.ids)
        expected = list(lineid for lineid, _ in numbered_lines)
        if ids != expected:
            raise ValueError(
                f"uh oh, text preprocessor change order or lost ids got {ids}, expected {expected} "
            )

    def work_on_subsection(
        self, numbered_lines
    ) -> tuple[list[Section], list[tuple[int, str]]] | None:
        # Iterate over the seed, such that if the model doesn't produce json, or drops ids, we try the next seed.
        for seed in range(1, 4):
            try:
                sections = self.send_prompt_for_sections(
                    numbered_lines=numbered_lines, seed=seed
                )

                # For large chapters, it is pretty common that the LLM only returns the first ids and some sentences
                # remain avaialble for chunking.
                for s in sections:
                    print(s)
                # Verify that the ids in the sections are the consecutive block at the start.
                ids = []
                for s in sections:
                    ids.extend(s.ids)
                expected = list(lineid for lineid, _ in numbered_lines[0 : len(ids)])

                # also verify it didn't actually put all the lines in the first section.
                # all_in_one_section = len(sections) == 1 and len(sections[0].ids) == len(
                #     ids
                # )

                if ids == expected:
                    # Splendid, this is a correct prefix.
                    return sections, numbered_lines[len(ids) :]
                else:
                    raise ValueError(f"got {ids}, expected {expected} ")

            except ValidationError as e:
                print(f"Invalid json: {e}")

            except ValueError as e:
                print(f"ids not consecutive or all in one section: {e}")

    def get_sections(self):
        return [
            Section(s.get_strings(self._numbered_lines), s.reasoning)
            for s in self._sections
        ]

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
            # Add this to ensure it is immediately evicted from the ollama server to free vram for the tts model.
            keep_alive=0,
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
