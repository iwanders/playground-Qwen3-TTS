#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

from .text_extractor import Chapter, Extractor


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

    dtype = _dtype_from_str(args.dtype)
    attn_impl = "flash_attention_2" if args.flash_attn else None
    print(args.voice)
    tts = Qwen3TTSInterface(
        model_path=args.model,
        voice_path=str(args.voice),
        device=args.device,
        dtype=_dtype_from_str(args.dtype),
        attn_impl=attn_impl,
    )
    return tts


def run_ebook(args):
    extractor = Extractor(args.file)
    chapters = extractor.get_chapters()
    to_export: list[Chapter] = []
    for c in chapters:
        i = c.get_index()
        print(f"{i} : {c}")
        if args.chapter == i:
            to_export.append(c)

    if not to_export:
        print("Nothing to export, pass a chapter!")
        sys.exit(0)

    tts = instantiate_tts_model(args)
    from .qwen3_tts import AudioObject

    gen_kwargs_default = _collect_gen_kwargs(args)

    for c in to_export:
        audio_segments = []
        audio_segments.append(tts.generate(f"Chapter {c.get_title()}"))
        lines = c.get_lines()
        if args.limit_lines is not None:
            print(f"Total lines {len(lines)} limiting to {args.limit_lines}")
            lines = lines[0 : args.limit_lines]
        if args.chapter_concat:
            total = "\n".join(lines)
            audio_segments.append(tts.generate(total, **gen_kwargs_default))
        else:
            for l in tqdm(lines):
                audio_segments.append(tts.generate(l, **gen_kwargs_default))

        out_name = f"{args.output_prefix}{c.get_index():0>2} {c.get_title()}{args.output_suffix}.wav"
        out_path = args.output_dir / out_name

        combined = AudioObject.from_list(audio_segments)
        combined.save(out_path)


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

    tts = instantiate_tts_model(args)
    output = tts.generate(
        get_text_from_args(args), language=args.language, **gen_kwargs_default
    )
    output.save(args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
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
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
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
    ## --

    parser_ebook = subparsers.add_parser(
        "ebook",
        parents=[parser],
    )

    parser_ebook.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Input ebook.",
    )
    parser_ebook.add_argument(
        "-c",
        "--chapter",
        type=int,
        help="Only export this chapter index",
        default=None,
    )

    parser_ebook.add_argument(
        "--chapter-concat",
        action="store_true",
        help="Concatenate the entire chapter together for tts synthesis. (Doesn't seem to work well)",
        default=False,
    )

    parser_ebook.add_argument(
        "--limit-lines",
        type=int,
        help="Limit export of each chapter to this number of lines",
        default=None,
    )
    parser_ebook.add_argument(
        "-o",
        "--output-dir",
        default="/tmp/",
        type=Path,
        help="The output directory for the chapters.",
    )
    parser_ebook.add_argument(
        "--output-suffix",
        default="",
        type=str,
        help="Suffix to add to the output file name at the end.",
    )
    parser_ebook.add_argument(
        "--output-prefix",
        default="",
        type=str,
        help="Prefix to prepend to the output file name.",
    )

    parser_ebook.set_defaults(func=run_ebook)

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
    parser_tts.set_defaults(func=run_tts)

    args = parser.parse_args()

    args.func(args)
