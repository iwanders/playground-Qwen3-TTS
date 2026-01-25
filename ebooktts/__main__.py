#!/usr/bin/env python3

import argparse
import logging
import os
from pathlib import Path

from .text_extractor import Extractor


def run_tts(args):
    extractor = Extractor(args.path)

    chapter_content = extractor.get_chapters()

    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
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
    parser_tts.set_defaults(func=run_tts)

    args = parser.parse_args()

    # no command
    if args.command is None:
        parser.print_help()
        parser.exit()

    args.func(args)
