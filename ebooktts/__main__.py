#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

from .text_extractor import Chapter, Extractor


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

    for c in to_export:
        audio_segments = []
        audio_segments.append(tts.generate(f"Chapter {c.get_title()}"))
        lines = c.get_lines()
        if args.limit_lines is not None:
            print(f"Total lines {len(lines)} limiting to {args.limit_lines}")
            lines = lines[0 : args.limit_lines]
        for l in tqdm(lines):
            audio_segments.append(tts.generate(l))
        out_name = f"{c.get_index():0>2} {c.get_title()}.wav"
        out_path = args.output_dir / out_name

        combined = AudioObject.from_list(audio_segments)
        combined.save(out_path)

    pass


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

    tts = instantiate_tts_model(args)
    output = tts.generate(get_text_from_args(args), language=args.language)
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
