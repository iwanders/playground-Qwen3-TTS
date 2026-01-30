import copy
import hashlib
import json
import logging
from pathlib import Path

import ollama
from ollama import ChatResponse, GenerateResponse

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
        z = hashlib.md5(
            json.dumps(dict(kwargs), indent=1, sort_keys=True).encode("utf-8")
        )

        return z.hexdigest()

    def retrieve(self, kwargs):
        key = self.make_key(kwargs)
        if key in self._cache:
            # return a copy to avoid the cache getting modified.
            z: ChatResponse = ChatResponse.model_validate(self._cache[key]["response"])
            return z

    def insert(self, key, request, response):
        response_json = None
        try:
            if "message" in response:
                response_json = json.loads(response.message.content)
            elif "response" in response:
                response_json = json.loads(response.response)
            else:
                response_json = {}
        except json.JSONDecodeError as e:
            pass
        self._cache[key] = {
            "request": request,
            "response": response.model_dump(),
            "response_json": response_json,
        }
        self.save_cache()

    def cached_worker(self, fun, kwargs):
        cache_hit = self.retrieve(kwargs)
        if cache_hit is None:
            response: ChatResponse = fun(**kwargs)
            key = self.make_key(kwargs)
            self.insert(key, kwargs, response)

        return self.retrieve(kwargs)

    def chat(self, **kwargs) -> ChatResponse:
        return self.cached_worker(ollama.chat, kwargs)

    def generate(self, **kwargs) -> GenerateResponse:
        return self.cached_worker(ollama.generate, kwargs)
