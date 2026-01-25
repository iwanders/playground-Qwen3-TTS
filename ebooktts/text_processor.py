import json
import os
import sys

import ollama
from ollama import ChatResponse
from pydantic import BaseModel, ValidationError

# export OLLAMA_CONTEXT_LENGTH=8192
# export OLLAMA_KV_CACHE_TYPE=q8_0
# qwen3:8b

OLLAMA_MODEL_TO_USE = os.environ.get("OLLAMA_MODEL_TO_USE", "qwen3:8b")


class Section(BaseModel):
    ids: list[int]
    reasoning: str


class SectionList(BaseModel):
    sections: list[Section]


class TextProcessor:
    def __init__(self, input_text):
        if isinstance(input_text, str):
            self._input_lines = input_text.split("\n")
        else:
            self._input_lines = list(input_text)
        self._sections = []

    def create_sections(self):
        # Iterate over the seed, such that if the model doesn't produce json, or drops ids, we try the next seed.
        for seed in range(1, 3):
            try:
                sections = self.send_prompt_for_sections(self._input_lines, seed=seed)
                # Verify that it did not actually lose any ids, or created duplicates.
                ids = []
                for s in sections:
                    ids.extend(s.ids)
                expected = list(range(1, len(self._input_lines) + 1))
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
    def send_prompt_for_sections(input_lines, seed=1):
        payload = json.dumps(
            # + 1 here such that it matches line numbers from the export.
            list({"id": k + 1, "text": v} for k, v in enumerate(input_lines)),
            indent=2,
        )

        # Using id's is much lighter than actually having the text in the sections.
        # It also kinda failed at splitting on who is speaking... often having 'said Foo in a grumpy voice' etc instead
        # of just having the section that was the quote.
        # Qwen3-TTS does a reasonable job at identifying quotes, so we don't _actually_ need to break on quotes.

        response: ChatResponse = ollama.chat(
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

        response = SectionList.model_validate_json(response.message.content)

        return response.sections


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        d = f.read()

    if False:
        parts = d.split("\n")[0:10]
        d = "\n".join(parts)
    print(d)

    z = TextProcessor(d)
    z.create_sections()
    for s in z.get_sections():
        print(s)
