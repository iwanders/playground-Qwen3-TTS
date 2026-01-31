# Playground Qwen3-TTS

A quickly hacked up weekend project around [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

Bulk of the code is around the three-stage epub -> audio file.
1. Extract raw text for each chapter from the epub.
2. Segment each chapter into logical segments of configurable maximum word count using an LLM (via ollama running locally).
3. Use the tts on each segment individually, concatenate them and write to disk.

## Architecture
The entry point is the `__main__.py` file, the rest of the module follows the three steps outlined above clearly.

### Extract raw text
The [text_extractor](text_extractor) uses [ebooklib](https://github.com/aerkalov/ebooklib) to iterate over the book's spine, then parses the html with
the builtin `html.parser` from stdlib. Output is bunch of `Chapter` objects that have a list of lines that hold the text
of the chapter. These directly correspond to the 'data' html elements extracted from the ebook.

Debug this phase by extracting the text from the book, or for a specific chapter like 47:
```
python3 -m ebooktts extract -f /tmp/our_ebook.epub -o /tmp/processed/ -c 47
```

### Processing text
I found that generating audio with a cloned voice on longer segments sometimes resulted in issues, including complete voice collapse.
Could be the same as [this issue](https://github.com/QwenLM/Qwen3-TTS/issues/80), would fine tuning be better?.
Cloning a voice using longer audio does seem to work better.
Currently, the text processing splits the chapters into smaller segments, which resolves the immediate problem and ensures the TTS engine has sufficient context for proper intonation.

This splitting is handled by the [text_processor](./ebooktts/text_processor.py).
It provides the LLM with a JSON blob containing line segments and their IDs, instructing it to group them into logical sections.
The processor uses a sliding window approach, feeding the LLM text within a window larger than the desired section length.
Each section returns a list of IDs, along with a reasoning explaining the logic behind the grouping.

Chats are executed with a fixed seed, no temperature (and cached to `/tmp/`).

It is possible to write the processed results to disk as json file for inspection, for example for chapter 47:
```
python3 -m ebooktts -- process -f /tmp/our_ebook.epub -o /tmp/processed/ -c 47
```

### TTS on the segments
This is pretty boring, calls into the qwen_tts module for each section, sections are contatenated with a second delay.
Files are written to the output directory as wav files, with a reasonable filename and metadata.

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
Edit: I don't this this is necessary actually, pytorch ships with it, see the docs on [sdpa](https://docs.pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html) stating that all implementations are enabled by default.

<details>

<summary>Original flash attention notes</summary>

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


</details>

License is Apache-2.0.
