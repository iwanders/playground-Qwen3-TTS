import copy
import hashlib
import json
import logging
import os
from pathlib import Path

import ollama
from ollama import ChatResponse, GenerateResponse

logger = logging.getLogger(__name__)


def strtobool(value: str) -> bool:
    """Convert a string to a boolean value based on standard truthy/falsy values."""
    value = value.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif value in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"Invalid truth value: '{value}'")


USE_CACHE = strtobool(os.environ.get("EBOOKTTS_USE_CACHE", "yes"))


def combine_chat_response(chunks: list[ChatResponse]) -> ChatResponse:
    value = chunks[0].model_dump()
    for c in chunks[1:]:
        c = c.model_dump()
        if c["message"]["thinking"] is not None:
            value["message"]["thinking"] += c["message"]["thinking"]

            # value["message"]["thinking"]
        value["message"]["content"] += c["message"]["content"]
        if c["message"]["tool_calls"]:
            value["message"]["tool_calls"] = c["message"]["tool_calls"]
        value["done"] = c["done"]
        for k, v in c.items():
            if k in value and value[k] is None:
                value[k] = v
    return ChatResponse.model_validate(value)


class StreamCache:
    def __init__(self, stream, cache, start_kwargs):
        self._stream = stream
        self._cache = cache
        self._start_kwargs = start_kwargs
        self._chunks = []

    def __iter__(self) -> "StreamCache":
        return self

    def __next__(self) -> ChatResponse:
        have_value = False
        value = None
        try:
            value = next(self._stream)
            have_value = True
        except StopIteration as e:
            pass

        if have_value:
            self._chunks.append(value)
            combine_chat_response(self._chunks)
            return value
        else:
            self._cache._cache_stream(kwargs=self._start_kwargs, chunks=self._chunks)
            raise StopIteration


class CacheStream:
    def __init__(self, resp):
        self._resp = resp

    def __iter__(self) -> "CacheStream":
        return self

    def __next__(self) -> ChatResponse:
        v = self._resp
        self._resp = None
        if v:
            return v
        else:
            raise StopIteration


class OllamaCache:
    def __init__(self, path="/tmp/ebooktts_cache/"):
        self._path = Path(path)
        self._cache_file = self._path / "ollama_cache.json"
        self._cache = self._load_cache()

    def _load_cache(self):
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

    def _save_cache(self):
        self._cache_file.parent.mkdir(exist_ok=True, parents=True)
        with open(self._cache_file, "w") as f:
            json.dump(self._cache, f, indent=1)

    @staticmethod
    def _serializible_kwargs(kwargs):
        kwargs = copy.deepcopy(kwargs)
        clean_kwargs = {}
        for k, v in kwargs.items():
            if k == "tools":
                continue
            if k == "messages":
                clean_kwargs["messages"] = []
                for m in v:
                    if m.get("tool_calls"):
                        m["tool_calls"] = [t.model_dump() for t in m["tool_calls"]]
                    clean_kwargs["messages"].append(m)
                continue
            clean_kwargs[k] = v
        return clean_kwargs

    def _make_key(self, kwargs):
        z = hashlib.md5(
            json.dumps(dict(kwargs), indent=1, sort_keys=True).encode("utf-8")
        )

        return z.hexdigest()

    def _retrieve(self, kwargs):
        if not USE_CACHE:
            return None
        clean_kwargs = OllamaCache._serializible_kwargs(kwargs)
        key = self._make_key(clean_kwargs)
        if key in self._cache:
            # return a copy to avoid the cache getting modified.
            z: ChatResponse = ChatResponse.model_validate(self._cache[key]["response"])
            return z

    def _insert(self, key, request, response):
        if not USE_CACHE:
            return
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
        self._save_cache()

    def _cached_worker(self, fun, kwargs):
        cache_hit = self._retrieve(kwargs)
        if cache_hit is None:
            response: ChatResponse = fun(**kwargs)
            clean_kwargs = OllamaCache._serializible_kwargs(kwargs)
            key = self._make_key(clean_kwargs)
            self._insert(key, clean_kwargs, response)
            if not USE_CACHE:
                return response

        return self._retrieve(kwargs)

    def chat(self, **kwargs) -> ChatResponse:
        if kwargs.get("stream"):
            raise ValueError("Stream is not supported in cached mode.")
        return self._cached_worker(ollama.chat, kwargs)

    def generate(self, **kwargs) -> GenerateResponse:
        return self._cached_worker(ollama.generate, kwargs)

    def stream(self, **kwargs) -> StreamCache | CacheStream:
        kwargs["stream"] = True
        cache_hit = self._retrieve(kwargs)
        if cache_hit:
            return CacheStream(cache_hit)
        # Extra wrapping for the streaming handler.
        # We can only cache it after the stream concluded.
        return StreamCache(ollama.chat(**kwargs), cache=self, start_kwargs=kwargs)

    def _cache_stream(self, kwargs, chunks):
        response = combine_chat_response(chunks)
        clean_kwargs = OllamaCache._serializible_kwargs(kwargs)
        key = self._make_key(clean_kwargs)
        self._insert(key, clean_kwargs, response)


if __name__ == "__main__":
    OLLAMA_MODEL_TO_USE = os.environ.get("OLLAMA_MODEL_TO_USE", "qwen3.5:4b")
    import logging

    logger = logging.getLogger(__name__)

    from .ollama_cache import OllamaCache

    cache = OllamaCache()

    def magic_function(value: int) -> str:
        """Returns a new integer."""
        """
        Returns:
            A new integer based on the input value..
        """
        return str(int(value) + 1)

    messages = [
        {
            "role": "system",
            "content": """
            Call the `magic_function` tool with a value of 0, then call it again with its return, and so on until it returns 3.
            Don't overthink this or try to predict what the function does, just start calling the tool.
            """,
        },
    ]
    available_functions = {
        "magic_function": magic_function,
    }
    while True:
        stream: ChatResponse = cache.stream(
            model=OLLAMA_MODEL_TO_USE,
            messages=messages,
            tools=list(available_functions.values()),
            # Make things completely deterministic, such that if weird things happen, I can at least reproduce weird things.
            # Since we do a lot of consecutive calls now, we want to keep it in memory until the next call.
            keep_alive=1,
            # Return the thinking data... this is the only debugging insight we get.
            think=True,
            stream=True,
        )

        thinking = ""
        content = ""
        tool_calls = []
        done_thinking = False
        # accumulate the partial fields
        for chunk in stream:
            if chunk.message.thinking:
                if not thinking:
                    # \033[1;30m grey
                    print("Thinking:\n", end="", flush=True)
                thinking += chunk.message.thinking
                print(chunk.message.thinking, end="", flush=True)
            if chunk.message.content:
                if not done_thinking:
                    done_thinking = True
                    # Wipe color \033[0m
                    print("\n\nAnswer:\n", end="", flush=True)
                    print("\n")
                content += chunk.message.content
                print(chunk.message.content, end="", flush=True)
            if chunk.message.tool_calls:
                tool_calls.extend(chunk.message.tool_calls)
                print(chunk.message.tool_calls)

        # append accumulated fields to the messages
        if thinking or content or tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "thinking": thinking,
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )

        if not tool_calls:
            break

        for call in tool_calls:
            fun = available_functions.get(call.function.name)
            if fun is None:
                print(f"Unknown function {call.function.name}")
            args = call.function.arguments
            result = fun(**args)
            print(f"Called {fun} with {args} -> {result}")
            messages.append(
                {
                    "role": "tool",
                    "tool_name": call.function.name,
                    "content": result,
                }
            )
