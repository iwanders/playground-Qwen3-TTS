#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("ebooktts")
from tqdm import tqdm

from .text_extractor import Chapter, Extractor
from .text_processor import Section, TextProcessor

"""
https://github.com/QwenLM/Qwen3-TTS/issues/89#issuecomment-3800731293
from https://github.com/dffdeeq/Qwen3-TTS-streaming/blob/01b51f0cb1c1c1a1fac9684f837a967226c0b17d/examples/test_optimized_no_streaming.py
"""


# https://github.com/QwenLM/Qwen3-TTS/blob/3b30a4e509657d8df1387554394141a0d68be4f0/qwen_tts/cli/demo.py#L178
def _collect_gen_kwargs(args: argparse.Namespace):
    mapping = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_top_k": args.subtalker_top_k,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def instantiate_tts_model(args):
    # Import here... such that --help is fast.
    from qwen_tts.cli.demo import _dtype_from_str

    from .qwen3_tts import Qwen3TTSInterface

    if args.seed is not None:
        import torch

        torch.manual_seed(args.seed)

    attn_impl = "flash_attention_2" if args.flash_attn else None
    tts = Qwen3TTSInterface(
        model_path=args.model,
        device=args.device,
        dtype=_dtype_from_str(args.dtype),
        attn_impl=attn_impl,
    )
    return tts


def ebook_to_chapter_exports(args) -> list[tuple[Chapter, list[str]]]:
    extractor = Extractor(args.file)
    chapters = extractor.get_chapters()
    to_export: list[Chapter] = []
    print(args.chapter)
    for c in chapters:
        i = c.get_index()
        print(f"{i} : {c}")
        if args.chapter is None or i in args.chapter:
            to_export.append(c)

    if not to_export:
        print("Nothing to export, pass a chapter!")
        sys.exit(0)

    chapter_data = []

    for c in to_export:
        text_segments = [f"Chapter {c.get_title()}"]
        lines = c.get_lines()
        if args.limit_lines is not None:
            print(f"Total lines {len(lines)} limiting to {args.limit_lines}")
            lines = lines[0 : args.limit_lines]
        for line in lines:
            text_segments.append(line)

        total = text_segments
        chapter_data.append((c, total))

    return chapter_data


def run_extract(args):
    chapter_data = ebook_to_chapter_exports(args)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    for c, text_segments in chapter_data:
        out_name = f"{args.output_prefix}{c.get_index():0>2} {c.get_title()}{args.output_suffix}.txt"
        out_path = args.output_dir / out_name
        with open(out_path, "w") as f:
            f.write("\n".join(text_segments))


def process_chapters(
    chapter_data: list[tuple[Chapter, list[str]]], section_word_limit=300
) -> list[tuple[Chapter, list[Section]]]:
    chapter_segments: list[tuple[Chapter, list[Section]]] = []
    for c, text_segments in chapter_data:
        processor = TextProcessor(text_segments, word_count_limit=section_word_limit)
        processor.create_sections()
        for s in processor.get_sections():
            print(s)

        chapter_segments.append((c, processor.get_sections()))
    return chapter_segments


def run_process(args):
    # Step 1, extract text from the ebook, holding lines by chapter.
    chapter_data = ebook_to_chapter_exports(args)

    # Step 2, crack each chapter, this requires ollama, but we'll get cache hits :)
    chapter_segments = process_chapters(
        chapter_data, section_word_limit=args.section_word_limit
    )

    args.output_dir.mkdir(exist_ok=True, parents=True)
    for c, text_sections in chapter_segments:
        out_name = f"{args.output_prefix}{c.get_index():0>2} {c.get_title()}{args.output_suffix}.json"
        out_path = args.output_dir / out_name
        data = {
            "chapter": c.get_title(),
            "sections": [
                {
                    "reasoning": s.get_reasoning(),
                    "text": s.get_text(),
                }
                for s in text_sections
            ],
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)


def run_ebook(args):
    gen_kwargs_default = _collect_gen_kwargs(args)
    book_name = Path(args.file).stem

    # Step 1, extract text from the ebook, holding lines by chapter.
    chapter_data = ebook_to_chapter_exports(args)

    # Step 2, crack each chapter, this requires ollama, but we'll get cache hits :)
    chapter_segments = process_chapters(
        chapter_data, section_word_limit=args.section_word_limit
    )

    # Step 3, now that we have ethe segments, we can perform the actual tts.
    tts = instantiate_tts_model(args)
    tts.load_voice(voice_path=str(args.voice))

    for c, text_sections in chapter_segments:
        combined = tts.generate_chunked_progress(
            [a.get_text() for a in text_sections], **gen_kwargs_default
        )

        out_name = f"{args.output_prefix}{c.get_index():0>2} {c.get_title()}{args.output_suffix}.wav"
        out_path = args.output_dir / out_name

        combined.save(
            out_path,
            metadata={
                "title": c.get_title(),
                "tracknumber": f"{c.get_index():0>2}",
                "album": book_name,
            },
        )


def run_tts(args):
    def get_text_from_args(args):
        if args.file is not None:
            if args.file == "-":
                return sys.stdin.read()
            with open(args.file, "r") as f:
                return f.read()
        if args.text is not None:
            return args.text
        raise ValueError("missing input argument")

    gen_kwargs_default = _collect_gen_kwargs(args)
    text = get_text_from_args(args)

    tts = instantiate_tts_model(args)
    tts.load_voice(voice_path=str(args.voice))

    if args.split_lines:
        textlist = text.split("\n")
        output = tts.generate_chunked_progress(
            textlist, language=args.language, **gen_kwargs_default
        )
    else:
        output = tts.generate(text, language=args.language, **gen_kwargs_default)
    output.save(args.output)


# https://stackoverflow.com/a/312464
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def run_clone(args):
    by_pairs = []
    inputs = list(args.input[0])
    if args.sort_inputs:
        inputs.sort()
    logger.info(f"args.x_vec_only: {args.x_vec_only}")
    if args.x_vec_only:
        for audio in inputs:
            if audio.endswith("txt"):
                continue
            by_pairs.append((audio, None))
            logger.info(f"{audio}")

    else:
        if len(args.input[0]) % 2 != 0:
            raise argparse.ArgumentTypeError(
                "incorrect number of input arguments, must be multiple of two"
            )
        for a, b in chunks(inputs, 2):
            if not a.endswith("txt") and not b.endswith("txt"):
                raise argparse.ArgumentTypeError(
                    f"no txt passed in for this pair {a} {b}"
                )

            text = a if a.endswith("txt") else b
            audio = b if a.endswith("txt") else a
            logger.info(f"{audio} with {text}")
            by_pairs.append((audio, text))

    tts = instantiate_tts_model(args)
    if args.concatenate:
        cloned = tts.voice_clone(by_pairs, use_xvec_only=args.x_vec_only)
        from .qwen3_tts import save_voice

        save_voice(args.output, cloned)
    else:
        # iterate over the inputs and create 'n' voice files.
        for audio_path, text_path in by_pairs:
            output_prefix = args.output.replace(".pt", "")
            cloned = tts.voice_clone(
                [(audio_path, text_path)], use_xvec_only=args.x_vec_only
            )
            from .qwen3_tts import save_voice

            p = Path(audio_path)

            output_prefix = output_prefix + p.stem + ".pt"

            save_voice(output_prefix, cloned)


def add_gen_args(parser):
    # Counterpart to _collect_gen_kwargs
    # https://github.com/QwenLM/Qwen3-TTS/blob/1ab0dd75353392f28a0d05d9ca960c9954b13c83/qwen_tts/cli/demo.py#L91
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for device_map, e.g. cpu, cuda, cuda:0 (default: cuda:0).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype for loading the model (default: bfloat16).",
    )
    parser.add_argument(
        "--flash-attn/--no-flash-attn",
        dest="flash_attn",
        default=False,
        # Leaving this on false
        # With True; get a warning that dtype is not set
        # Demo doesn't have this warning, what flag / property is missing?
        # With that warning, False takes 20s, True takes 30s, for same text and seed.
        # This doesn't actually matter since torch ships with flash attention now.
        action=argparse.BooleanOptionalAction,
        help="Enable FlashAttention-2 (default: enabled).",
    )
    parser.add_argument(
        "--language", default="Auto", help="The language; Auto / English, etc"
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Max new tokens for generation (optional).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (optional). Default is 0.9.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (optional). Default is 50.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling (optional). Default is 1.0.",
    )
    ## IW: Modified this, if this 1.05 (the default) the tts system is much less robust with custom voices and deteriorates much earlier
    # on long segments.
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.01,
        help="Repetition penalty (optional).",
    )
    parser.add_argument(
        "--subtalker-top-k",
        type=int,
        default=None,
        help="Subtalker top-k (optional, only for tokenizer v2).",
    )
    parser.add_argument(
        "--subtalker-top-p",
        type=float,
        default=None,
        help="Subtalker top-p (optional, only for tokenizer v2).",
    )
    parser.add_argument(
        "--subtalker-temperature",
        type=float,
        default=None,
        help="Subtalker temperature (optional, only for tokenizer v2).",
    )


if __name__ == "__main__":
    level = logging.DEBUG

    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser(add_help=False)

    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument(
        "--voice",
        type=Path,
        default=os.environ.get("QWEN_TTS_VOICE"),
        help="Specify the voice, defaults to ${QWEN_TTS_VOICE}, currently %(default)s",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=os.environ.get("QWEN_TTS_BASE_MODEL"),
        help="Specify the model, defaults to ${QWEN_TTS_BASE_MODEL}, currently %(default)s",
    )

    # Settings from  _collect_gen_kwargs
    add_gen_args(parser)

    parser.add_argument(
        "--seed", type=int, default=None, help="If set, seed torch with this."
    )
    ## --  Extract subcommand --
    parser_extract = subparsers.add_parser(
        "extract",
        help="Dump the text data from the ebook to disk.",
    )

    def add_ebook_common_args(subparser):
        subparser.add_argument(
            "-f",
            "--file",
            type=Path,
            help="Input ebook.",
        )
        subparser.add_argument(
            "-c",
            "--chapter",
            type=int,
            nargs="+",
            help="Limit export to these chapters, use bash expansion for a range, like; -c {3..7}",
            default=None,
        )
        subparser.add_argument(
            "--limit-lines",
            type=int,
            help="Limit export of each chapter to this number of lines",
            default=None,
        )
        subparser.add_argument(
            "--output-suffix",
            default="",
            type=str,
            help="Suffix to add to the output file name at the end.",
        )
        subparser.add_argument(
            "--output-prefix",
            default="",
            type=str,
            help="Prefix to prepend to the output file name.",
        )
        subparser.add_argument(
            "--section-word-limit",
            default=300,
            type=int,
            help="Number of words that's allowed in a single section. Single lines output by the text extractor are never broken.",
        )

    add_ebook_common_args(parser_extract)
    parser_extract.add_argument(
        "-o",
        "--output-dir",
        default="/tmp/",
        type=Path,
        help="The output directory for the chapters.",
    )
    parser_extract.set_defaults(func=run_extract)

    ## --- Ebook process ---
    #
    parser_process = subparsers.add_parser(
        "process",
        parents=[parser],
        help="This outputs the processed text to disk for inspection",
    )
    add_ebook_common_args(parser_process)
    parser_process.add_argument(
        "-o",
        "--output-dir",
        default="/tmp/",
        type=Path,
        help="The output directory for the chapters.",
    )
    parser_process.set_defaults(func=run_process)

    ## --  Ebook subcommand --
    parser_ebook = subparsers.add_parser(
        "ebook",
        parents=[parser],
        help="This converts an ebook, or selected chapters to wav files.",
    )

    add_ebook_common_args(parser_ebook)

    parser_ebook.add_argument(
        "-o",
        "--output-dir",
        default="/tmp/",
        type=Path,
        help="The output directory for the chapters.",
    )

    parser_ebook.set_defaults(func=run_ebook)

    ## --  TTS subcommand --

    parser_tts = subparsers.add_parser(
        "tts",
        parents=[parser],
    )
    parser_tts.add_argument(
        "-t",
        "--text",
        default=None,
        help="The text to tts",
        nargs="?",
    )
    parser_tts.add_argument(
        "-o", "--output", default="/tmp/output.wav", help="The output file path"
    )
    parser_tts.add_argument(
        "-f",
        "--file",
        nargs="?",
        type=str,
        default=None,
        help="Input file (or '-' for stdin)",
    )
    parser_tts.add_argument(
        "-s",
        "--split-lines",
        default=False,
        action="store_true",
        help="Split on \\n characters in the text input and read file and process them seperately.",
    )
    parser_tts.set_defaults(func=run_tts)

    ## --  Voice Clone subcommand --

    parser_clone = subparsers.add_parser(
        "clone",
        help="This is a simple wrapper to create a voice clone.",
    )
    parser_clone.add_argument(
        "-o",
        "--output",
        default="/tmp/voice.pt",
        help="The output file path for the voice file. Defaults to %(default)s. If not concatenated extension is stripped and it becomes the prefix.",
    )
    parser_clone.add_argument(
        "input",
        nargs="+",
        action="append",
        help="Input files to make the voice clone from, should be pairs of txt and audio files",
    )
    parser_clone.add_argument(
        "--sort-inputs",
        action="store_true",
        default=False,
        help="Whether to run sort() on the inputs, this is helpful if filenames are good and a glob is used.",
    )
    parser_clone.add_argument(
        "--concatenate",
        action="store_true",
        default=False,
        help="Whether or not to concatenate the inputs, or whether to create multiple voice files (each containing one), suffixed with the stem.",
    )
    parser_clone.add_argument(
        "--x-vec-only",
        action="store_true",
        default=False,
        help="Whether or not to only use the xvec, discards all the txts.",
    )

    parser_clone.set_defaults(func=run_clone)

    ## --  parsing  --
    args = parser.parse_args()

    args.func(args)
