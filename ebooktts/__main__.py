#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

from .text_extractor import Chapter, Extractor
from .text_processor import Section, TextProcessor


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
        if args.chapter_concat:
            total = ["\n".join(text_segments)]
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


def run_ebook(args):
    gen_kwargs_default = _collect_gen_kwargs(args)
    book_name = Path(args.file).stem

    # Step 1, extract text from the ebook, holding lines by chapter.
    chapter_data = ebook_to_chapter_exports(args)

    # Step 2, crack each chapter, this requires ollama, but we'll get cache hits :)
    chapter_segments: list[tuple[Chapter, list[Section]]] = []
    for c, text_segments in chapter_data:
        processor = TextProcessor(text_segments)
        processor.create_sections()
        for s in processor.get_sections():
            print(s)

        chapter_segments.append((c, processor.get_sections()))

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
    if len(args.input[0]) % 2 != 0:
        raise argparse.ArgumentTypeError(
            "incorrect number of input arguments, must be multiple of two"
        )
    by_pairs = []
    inputs = list(args.input[0])
    if args.sort_inputs:
        inputs.sort()
    for a, b in chunks(inputs, 2):
        if not a.endswith("txt") and not b.endswith("txt"):
            raise argparse.ArgumentTypeError(f"no txt passed in for this pair {a} {b}")

        text = a if a.endswith("txt") else b
        audio = b if a.endswith("txt") else a
        by_pairs.append((audio, text))

    tts = instantiate_tts_model(args)
    if args.concatenate:
        cloned = tts.voice_clone(by_pairs)
        from .qwen3_tts import save_voice

        save_voice(args.output, cloned)
    else:
        # iterate over the inputs and create 'n' voice files.
        for audio_path, text_path in by_pairs:
            output_prefix = args.output.replace(".pt", "")
            cloned = tts.voice_clone([(audio_path, text_path)])
            from .qwen3_tts import save_voice

            p = Path(text_path)

            output_prefix = output_prefix + p.stem + ".pt"

            save_voice(output_prefix, cloned)


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

    parser.add_argument(
        "--seed", type=int, default=None, help="If set, seed torch with this."
    )
    ## --  Extract subcommand --
    parser_extract = subparsers.add_parser(
        "extract",
        help="dump the content of the ebook to disk",
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
            "--chapter-concat",
            action="store_true",
            help="Concatenate the entire chapter together for tts synthesis. (Doesn't seem to work well)",
            default=False,
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

    add_ebook_common_args(parser_extract)
    parser_extract.add_argument(
        "-o",
        "--output-dir",
        default="/tmp/",
        type=Path,
        help="The output directory for the chapters.",
    )
    parser_extract.set_defaults(func=run_extract)

    ## --  Ebook subcommand --
    parser_ebook = subparsers.add_parser(
        "ebook",
        parents=[parser],
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
        help="Whether or not to concatenate the inputs, or whether to create multiple voice files (each containing one)",
    )

    parser_clone.set_defaults(func=run_clone)

    ## --  parsing  --
    args = parser.parse_args()

    args.func(args)
