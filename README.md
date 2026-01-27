# Playground Qwen3-TTS

A quickly hacked up weekend project around [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

Bulk of the code is around the three-stage epub -> audio file.
1. Extract raw text for each chapter from the epub.
2. Segment each chapter into logical segments of max 300 words using an LLM (via ollama running locally).
3. Use the tts on each segment individually, concatenate them and write to disk.

## Architecture
The entry point is the `__main__.py` file, it's a bit of a mess, the remainder follows the three steps outlined above clearly.

### Extract raw text
The [text_extractor](text_extractor) uses [ebooklib](https://github.com/aerkalov/ebooklib) to iterate over the book's spine, then parses the html with
the builtin `html.parser` from stdlib. Output is bunch of `Chapter` objects that have a list of lines that hold the text
of the chapter. These directly correspond to the 'data' html elements extracted from the ebook.

Debug this phase by extracting the text from the book, or for a specific chapter like 47:
```
python3 -m ebooktts extract -f /tmp/our_ebook.epub -o /tmp/processed/ -c 47
```

### Processing text

This is neccessary for me, I found with a cloned voice processing long segments caused problems, sometimes collapse of voice all together.
Probably the same as [this issue](https://github.com/QwenLM/Qwen3-TTS/issues/80), maybe this can be avoided with a fine tune instead?
Processing the text like this works reasonably well though, so it's not really a problem now, and the segments are large enough
such that the tts engine has context.

Splitting the text in logical segments and feeding the individual segments of <300 words to the TTS does wonders.
This is done by the [text_processor](./ebooktts/text_processor.py).
This provides the LLM with a json blob of lines and their ids, and asks it to group it by logical sections.
Each section refers to ids and a reasoning is provided why this makes a logical section.
If sections are still too large a new request is made to the LLM with just those lines until all sections are below the threshold.

There's a caching system for the llm ollama chat.
This is important because this way the first pass can populate the chat with the ollama server running.
After that, the ollama server can be closed to make vram capacity for the tts step.
Chats are executed with a fixed seed and no temperature.

It is possible to write the processed results to disk as json file for inspection, for example for chapter 47:
```
python3 -m ebooktts -- process -f /tmp/our_ebook.epub -o /tmp/processed/ -c 47
```

### TTS on the segments
This is pretty boring, calls into the qwen_tts module for each section, sections are contatenated with a second delay.
Files are written to the output directory as wav files, with a reasonable filename and metadata.

There is a progress indicator for the segments exported, not for the chapters currently.
There is a progress indicator for the segments exported, not for the chapters currently.

Write the wav files to disk at `/tmp/our_book/`, limited to chapter 47:
```
python3 -m ebooktts -- ebook  -f /tmp/our_ebook.epub  -o /tmp/our_book/ -c 47
```

## Use

Fresh virtualenv `pip install .` on this dir, tested with python 3.13.

```
python3 -m ebooktts -- ebook  -f /tmp/our_ebook.epub -o /tmp/my_book/
```

It respects the following environment variables.
```
export QWEN_TTS_BASE_MODEL=/path/to/Qwen3-TTS-12Hz-1.7B-Base
export QWEN_TTS_VOICE="/path/to/cloned_voice.pt"
export OLLAMA_MODEL_TO_USE="qwen3:8b"
```

### Flash attention
Did the following, from apt;

```
apt install nvidia-cuda-toolkit
apt install ninja-build 
```

In this virtualenv;
```
pip install flash-attn --no-build-isolation
```
which takes quite some resources. It almost peaked at over 64 GB of ram, see [MAX_JOBS](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) from the installation guide to limit concurrent jobs.

License is Apache-2.0.
