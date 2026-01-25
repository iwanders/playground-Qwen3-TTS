#!/usr/bin/env python3

import argparse
import logging
import os
from pathlib import Path

from qwen_tts.cli.demo import _dtype_from_str

from .text_extractor import Extractor


def run_tts(args):
    extractor = Extractor(args.path)

    chapter_content = extractor.get_chapters()

    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    parser_tts = subparsers.add_parser("tts", help="Used to embed a watermark")
    parser_tts.add_argument(
        "--voice",
        type=Path,
        default=os.environ.get("QWEN_TTS_VOICE"),
        help="Override the voice, defaults to ${QWEN_TTS_VOICE}, currently %(default)s",
    )
    parser_tts.add_argument(
        "path",
        type=Path,
        help="Input path",
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
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable FlashAttention-2 (default: enabled).",
    )

    dtype = _dtype_from_str(args.dtype)
    attn_impl = "flash_attention_2" if args.flash_attn else None

    parser_tts.set_defaults(func=run_tts)

    args = parser.parse_args()

    # no command
    if args.command is None:
        parser.print_help()
        parser.exit()

    args.func(args)
