# Playground Qwen3-TTS

A quickly hacked weekend project around [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

Bulk of the code is around the three-stage epub -> audio file.
1. Extract raw text for each chapter from the epub.
2. Segment each chapter into logical segments of max 300 words using an LLM (via ollama running locally).
3. Use the tts on each segment individually, concatenate them and write to disk.



License is Apache-2.0.
