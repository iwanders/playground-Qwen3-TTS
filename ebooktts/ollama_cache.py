import copy
import json
import logging
from pathlib import Path

import ollama
from ollama import ChatResponse

logger = logging.getLogger(__name__)


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

    def generate(self, **kwargs):
        cache_hit = self.retrieve(kwargs)
        if cache_hit is not None:
            return cache_hit
        # Need to do work.
        #
        response: ChatResponse = ollama.generate(**kwargs)

        print(response)
        content = response.response
        key = self.make_key(kwargs)
        self.insert(key, content)
        return copy.deepcopy(content)
