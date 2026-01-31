import copy
import json
import os
import re
import sys
from collections import namedtuple
from pathlib import Path

import ollama
from ollama import ChatResponse
from pydantic import BaseModel, ValidationError

# export OLLAMA_CONTEXT_LENGTH=8192
# export OLLAMA_KV_CACHE_TYPE=q8_0
# qwen3:8b

OLLAMA_MODEL_TO_USE = os.environ.get("OLLAMA_MODEL_TO_USE", "qwen3:8b")
import logging

logger = logging.getLogger(__name__)

from .ollama_cache import OllamaCache

cache = OllamaCache()


class InternalSection(BaseModel):
    ids: list[int]
    reasoning: str

    def is_multiple_lines(self):
        return len(self.ids) > 1

    def get_strings(self, numbered_lines):
        numbered_lines = (
            numbered_lines.get_numbered_chunks()
            if isinstance(numbered_lines, NumberedChunks)
            else numbered_lines
        )
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


"""
The recursive approach breaks down when we pass in too much text, and it's also just hard to deal with.
Lets do linear chunking instead, with a window that slides...
And lets also account for situations where the single lines are really long and contain an entire paragraph.
Breaking on '.' and '\n' seems reasonable in the input text, but we want this to be opaque to the LLM.
So we do a bunch more bookkeeping
"""


def split_by_delim_chunk(input_text, delims):
    """
    THis splits the string by delims, without losing the delims, and rejoins consecutive delims.
    """

    delims = set(delims)
    full = [input_text]
    for d in delims:
        new_split = []
        for k in full:
            k_d = k.split(d)
            # Append at the end of each token except the last.
            appended = []
            for i, z in enumerate(k_d):
                segment = z
                if i < len(k_d) - 1:
                    segment = z + d
                appended.append(segment)
            new_split.extend(appended)
        full = new_split

    if not full:
        return input_text

    # Now that it is full split, we want to coalesce delimeters again, such that ".\n" does not result in two
    # chunks.... just because deubging is easier without too many splits.
    re_joined = [full[0]]
    # If the stripped segment is one of these characters, we also just want to join it back with the original one.
    # todo; pass this in as an arg, or something, this is very hacky.
    extra_merge = set(["’", ",", " ", "\n", "'", '"'])
    for s in full[1:]:
        if s.strip() in (delims | extra_merge) or s.strip() == "":
            re_joined[-1] = re_joined[-1] + s
        else:
            re_joined.append(s)

    return re_joined


def test_split():
    def t(s, d):
        res_one = list(split_by_delim_chunk(s, d))
        print(s, " -> ", res_one)
        assert s == "".join(res_one)

    t("Hello there, how are you", [" ", ","])
    t("Hello there\nhow are you\n", ["\n"])
    t("Hello there\n,how are you\n", ["\n", ","])
    t("Hello there\n...how are you\n", ["\n", "."])
    t("Hello there\n. . ., how are you\n", ["\n", ".", ","])
    t("", ["\n", ","])
    raise ValueError("Success")


# test_split()


class NumberedChunks:
    def __init__(self, numbered_chunks):
        self._numbered_chunks = numbered_chunks

    def split_word_limit(self, word_limit):
        res = []
        word_so_far = 0
        for ni, (i, v) in enumerate(self._numbered_chunks):
            words = list(w for w in re.split(r"[;,.\s*\n’]", v) if len(w) > 1)
            # print("v was: ", v, "words is ", words, "split", re.split(r"[;,\s*\n]", v))
            word_this_segment = len(words)
            # print(
            #     f"words so far: {word_so_far}, limit {word_limit} this segm: {word_this_segment}, {words}"
            # )
            word_so_far += word_this_segment
            res.append((i, v))
            if word_so_far > word_limit:
                return NumberedChunks(res), NumberedChunks(
                    self._numbered_chunks[ni + 1 :]
                )
        return NumberedChunks(self._numbered_chunks), NumberedChunks([])

    def combine(self, other: "NumberedChunks"):
        return NumberedChunks(self._numbered_chunks + other._numbered_chunks)

    def has_chunks(self):
        return len(self._numbered_chunks) != 0

    def get_numbered_chunks(self):
        return self._numbered_chunks

    def print_verbose_chunks(self):
        for i, v in self._numbered_chunks:
            print(f"{i:0>4d}: {v}")

    def __repr__(self):
        return f"<NumberedChunks nums: {', '.join(str(i) for i, v in self._numbered_chunks)}>"


class TextProcessor:
    def __init__(self, input_text, word_count_limit=100, window_words_factor=2):
        if isinstance(input_text, list):
            input_text = "\n".join(input_text)
        self._input_text = input_text
        self._word_count_limit = word_count_limit
        self._window_words_factor = window_words_factor
        # Lets split the input text by chunks.
        text_chunks = list(
            split_by_delim_chunk(
                input_text,
                delims=set(
                    [
                        "\n",
                        ".",
                    ]
                ),
            )
        )
        self._numbered_chunks = NumberedChunks(
            list((i + 1, v) for i, v in enumerate(text_chunks))
        )

        self._sections = []

    def create_sections(self):
        remaining = self._numbered_chunks
        while remaining.has_chunks():
            # Determine the chunks at the start that make up the llm desired word count.
            desired_words = self._word_count_limit * self._window_words_factor
            in_chunk, tail_end = remaining.split_word_limit(desired_words)
            # print("LLM section ready")
            in_chunk.print_verbose_chunks()

            result = self.work_on_subsection(in_chunk.get_numbered_chunks())
            if not result:
                raise ValueError(f"Failed to converge on a solution at {in_chunk} ")

            subsections, remainder_numbered_lines = result
            # print(f"subsections: {subsections}")

            self._sections.extend(subsections)
            remaining = NumberedChunks(remainder_numbered_lines).combine(tail_end)

    def work_on_subsection(
        self, numbered_lines
    ) -> tuple[list[Section], list[tuple[int, str]]] | None:
        # Iterate over the seed, such that if the model doesn't produce json, or drops ids, we try the next seed.
        for seed in range(1, 4):
            try:
                # print(f"Passing {numbered_lines}")
                sections = self.send_prompt_for_sections(
                    numbered_lines=numbered_lines, seed=seed
                )
                # print(f"LLM returned {sections}")

                # for s in sections:
                #    print(s)
                # Verify that the ids in the sections are the consecutive block at the start.
                ids = []
                for s in sections:
                    ids.extend(s.ids)
                expected = list(lineid for lineid, _ in numbered_lines[0 : len(ids)])

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
            Section(s.get_strings(self._numbered_chunks), s.reasoning)
            for s in self._sections
        ]

    @staticmethod
    def send_prompt_for_sections(numbered_lines, seed=1):
        payload = json.dumps(
            list({"line_id": k, "line": v} for k, v in numbered_lines),
            indent=2,
        )
        print(f"payload to llm: {payload}")

        response = cache.chat(
            model=OLLAMA_MODEL_TO_USE,
            messages=[
                {
                    "role": "system",
                    "content": """
                    Provided are the text lines of a book in json format, each line has an id provided with it.
                    Split the lines into a logical sections by stating which ids are present in each section.
                    Briefly explain the reasoning for each section and why it makes a logical section.
                    The provided lines may be in the middle of the book and may not form a cohesive whole.
                    Each section should be short enough to be spoken out loud in one breath. 
                    Respond in json.
                    Do not renumber the lines, you must use the line_id specified.
                    """,
                },
                {
                    "role": "user",
                    "content": payload,
                },
            ],
            format=SectionList.model_json_schema(),
            # Make things completely deterministic, such that if weird things happen, I can at least reproduce weird things.
            options={"temperature": 0.00, "seed": seed},
            # Add this to ensure it is immediately evicted from the ollama server to free vram for the tts model.
            keep_alive=0,
            think=True,
        )
        print(response)

        response = SectionList.model_validate_json(response.message.content)
        print(f"response from llm: {response}")

        return response.sections


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        d = f.read()

    if False:
        parts = d.split("\n")[0:1]
        d = "\n".join(parts)
    print(d)

    z = TextProcessor(d)
    z.create_sections()
    for s in z.get_sections():
        print(s)
