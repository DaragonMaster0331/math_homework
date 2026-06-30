#!/usr/bin/env python3
# Copyright (c) Opendatalab. All rights reserved.
"""
Local MinerU PDF extraction (no server-client, no HTTP).

Uses do_parse() / aio_do_parse() directly in-process.

Examples:
  # Preload models (optional, speeds up first document)
  python demo/local_extract.py --preload-only -b pipeline

  # Process one file
  python demo/local_extract.py -p path/to/file.pdf -o ./output

  # Process every supported file in a folder
  python demo/local_extract.py -p path/to/folder -o ./output

  # CPU-friendly pipeline backend
  python demo/local_extract.py -p file.pdf -o ./output -b pipeline

  # Use local models and GPU
  python demo/local_extract.py -p file.pdf -o ./output -b pipeline \\
      --model-path C:/models/PDF-Extract-Kit-1.0 --device cuda

  # VLM with local model path
  python demo/local_extract.py -p file.pdf -o ./output -b vlm-auto-engine \\
      --model-path C:/models/MinerU2.5-Pro --device cuda

Install (pipeline backend):
  pip install "mineru[pipeline]" "transformers>=4.57.3,<5.0.0"
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from loguru import logger

from mineru.cli.common import (
    HybridDependencyError,
    do_parse,
    aio_do_parse,
    image_suffixes,
    office_suffixes,
    pdf_suffixes,
    read_fn,
    uniquify_task_stems,
)
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path

SUPPORTED_SUFFIXES = set(pdf_suffixes + image_suffixes + office_suffixes)


def apply_runtime_config(
    device: str | None,
    model_path: Path | None,
    backend: str,
) -> dict:
    """Set device/model env and return extra kwargs for do_parse."""
    parse_kwargs: dict = {}

    if device:
        os.environ["MINERU_DEVICE_MODE"] = device
        logger.info(f"Using device: {device}")

    if model_path is None:
        return parse_kwargs

    resolved_model_path = model_path.expanduser().resolve()
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {resolved_model_path}")

    if backend == "pipeline":
        _configure_local_pipeline_models(resolved_model_path)
        logger.info(f"Using local pipeline models from: {resolved_model_path}")
        return parse_kwargs

    if backend.startswith("vlm-") or backend.startswith("hybrid-"):
        parse_kwargs["model_path"] = str(resolved_model_path)
        logger.info(f"Using VLM model from: {resolved_model_path}")
        return parse_kwargs

    raise ValueError(f"--model-path is not supported for backend: {backend}")


def _configure_local_pipeline_models(pipeline_root: Path) -> None:
    """Point MinerU pipeline model loader at a local PDF-Extract-Kit root."""
    if not (pipeline_root / "models").is_dir():
        raise FileNotFoundError(
            f"Pipeline model root must contain a 'models' directory: {pipeline_root}"
        )

    models_dir = {
        "pipeline": str(pipeline_root),
        "vlm": "",
    }
    os.environ["MINERU_MODEL_SOURCE"] = "local"

    def local_models_dir() -> dict[str, str]:
        return models_dir

    # MinerU reads MINERU_TOOLS_CONFIG_JSON only at import time, so patch loaders directly.
    import mineru.utils.config_reader as config_reader
    import mineru.utils.models_download_utils as models_download_utils

    config_reader.get_local_models_dir = local_models_dir
    models_download_utils.get_local_models_dir = local_models_dir


def check_transformers_version() -> None:
    """MinerU formula models require transformers 4.x (not 5.x)."""
    try:
        transformers_version = version("transformers")
    except PackageNotFoundError:
        return

    major = int(transformers_version.split(".", 1)[0])
    if major >= 5:
        raise RuntimeError(
            f"Incompatible transformers {transformers_version} installed. "
            "MinerU requires transformers>=4.57.3,<5.0.0. "
            'Fix: pip install "transformers>=4.57.3,<5.0.0" '
            "Or skip formula parsing with --no-formula."
        )


PIPELINE_OPTIONAL_MODULES = ("albumentations", "shapely", "pyclipper", "omegaconf")


def check_pipeline_dependencies(backend: str, formula_enable: bool) -> None:
    if backend != "pipeline" and not backend.startswith("hybrid-"):
        return

    missing = []
    for module_name in PIPELINE_OPTIONAL_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)

    if missing:
        raise RuntimeError(
            f"Missing pipeline dependencies: {', '.join(missing)}. "
            'Install with: pip install "mineru[pipeline]" '
            '"transformers>=4.57.3,<5.0.0"'
        )

    if formula_enable:
        check_transformers_version()


def collect_input_files(input_path: Path) -> list[Path]:
    path = input_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        suffix = guess_suffix_by_path(path)
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {path.name}")
        return [path]

    if not path.is_dir():
        raise ValueError(f"Input path must be a file or directory: {path}")

    files = sorted(
        candidate.resolve()
        for candidate in path.iterdir()
        if candidate.is_file() and guess_suffix_by_path(candidate) in SUPPORTED_SUFFIXES
    )
    if not files:
        raise ValueError(f"No supported files found in directory: {path}")
    return files


def build_document_batch(files: list[Path], lang: str) -> tuple[list[str], list[bytes], list[str]]:
    stems = [file.stem for file in files]
    unique_stems, renamed = uniquify_task_stems(stems)
    if renamed:
        details = ", ".join(f"{src} -> {dst}" for src, dst in renamed)
        logger.warning(f"Normalized duplicate document stems: {details}")

    pdf_bytes_list = [read_fn(file) for file in files]
    lang_list = [lang] * len(files)
    return unique_stems, pdf_bytes_list, lang_list


def preload_models(
    backend: str,
    lang: str = "ch",
    formula_enable: bool = True,
    table_enable: bool = True,
    model_path: str | None = None,
) -> None:
    """Warm up model weights before parsing."""
    if backend == "pipeline":
        from mineru.backend.pipeline.pipeline_analyze import ModelSingleton

        logger.info("Preloading pipeline models...")
        ModelSingleton().get_model(
            lang=lang,
            formula_enable=formula_enable,
            table_enable=table_enable,
        )
        logger.info("Pipeline models ready.")
        return

    if backend.startswith("vlm-") or backend.startswith("hybrid-"):
        if backend.startswith("hybrid-"):
            try:
                from mineru.backend.pipeline.model_init import HybridModelSingleton

                logger.info("Preloading hybrid pipeline sub-models...")
                HybridModelSingleton().get_model(
                    lang=lang,
                    formula_enable=formula_enable,
                )
            except Exception as exc:
                logger.warning(f"Hybrid pipeline preload skipped: {exc}")

        from mineru.backend.vlm.vlm_analyze import ModelSingleton
        from mineru.utils.engine_utils import get_vlm_engine

        engine_name = backend.removeprefix("vlm-").removeprefix("hybrid-")
        if engine_name == "auto-engine":
            engine_name = get_vlm_engine(inference_engine="auto", is_async=False)

        logger.info(f"Preloading VLM engine: {engine_name}...")
        ModelSingleton().get_model(engine_name, model_path, None)
        logger.info(f"VLM engine ready: {engine_name}")
        return

    raise ValueError(f"Unsupported backend for preload: {backend}")


def _backend_needs_async(backend: str) -> bool:
    return "vllm-async-engine" in backend or backend.endswith("vllm-async-engine")


def run_parse(
    output_dir: Path,
    pdf_file_names: list[str],
    pdf_bytes_list: list[bytes],
    lang_list: list[str],
    backend: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    start_page_id: int,
    end_page_id: int | None,
    image_analysis: bool,
    parse_kwargs: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        output_dir=str(output_dir),
        pdf_file_names=pdf_file_names,
        pdf_bytes_list=pdf_bytes_list,
        p_lang_list=lang_list,
        backend=backend,
        parse_method=parse_method,
        formula_enable=formula_enable,
        table_enable=table_enable,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        image_analysis=image_analysis,
    )
    if parse_kwargs:
        kwargs.update(parse_kwargs)

    if _backend_needs_async(backend):
        asyncio.run(aio_do_parse(**kwargs))
    else:
        do_parse(**kwargs)


def process_path(
    input_path: Path,
    output_dir: Path,
    *,
    backend: str,
    lang: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    start_page_id: int,
    end_page_id: int | None,
    image_analysis: bool,
    preload: bool,
    parse_kwargs: dict | None = None,
) -> None:
    parse_kwargs = parse_kwargs or {}

    files = collect_input_files(input_path)
    mode = "folder" if input_path.is_dir() else "file"
    logger.info(f"Local {mode} mode: {len(files)} document(s) -> {output_dir}")

    if preload:
        preload_models(
            backend=backend,
            lang=lang,
            formula_enable=formula_enable,
            table_enable=table_enable,
            model_path=parse_kwargs.get("model_path"),
        )

    names, pdf_bytes_list, lang_list = build_document_batch(files, lang)
    run_parse(
        output_dir=output_dir,
        pdf_file_names=names,
        pdf_bytes_list=pdf_bytes_list,
        lang_list=lang_list,
        backend=backend,
        parse_method=parse_method,
        formula_enable=formula_enable,
        table_enable=table_enable,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        image_analysis=image_analysis,
        parse_kwargs=parse_kwargs,
    )
    logger.info(f"Done. Output written to {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MinerU local PDF extraction (no API server).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-p",
        "--path",
        type=Path,
        help="Input PDF/image/office file or folder of supported files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./output"),
        help="Output directory (default: ./output).",
    )
    parser.add_argument(
        "-b",
        "--backend",
        default="pipeline",
        help=(
            "Parsing backend. Common values: pipeline, hybrid-auto-engine, "
            "vlm-auto-engine, vlm-transformers (default: pipeline)."
        ),
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["auto", "txt", "ocr"],
        default="auto",
        help="Parse method for pipeline/hybrid backends (default: auto).",
    )
    parser.add_argument(
        "-l",
        "--lang",
        default="ch",
        help="Document language for OCR (default: ch).",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        help="Inference device: cuda, cpu, mps, npu, etc. (default: auto-detect).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Local model directory. pipeline: PDF-Extract-Kit root; "
            "vlm/hybrid: VLM model root. Auto-download if omitted."
        ),
    )
    parser.add_argument(
        "-s",
        "--start-page",
        type=int,
        default=0,
        help="Start page index, 0-based (default: 0).",
    )
    parser.add_argument(
        "-e",
        "--end-page",
        type=int,
        default=None,
        help="End page index inclusive; omit for last page.",
    )
    parser.add_argument(
        "--no-formula",
        action="store_true",
        help="Disable formula parsing.",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Disable table parsing.",
    )
    parser.add_argument(
        "--no-image-analysis",
        action="store_true",
        help="Disable image/chart analysis (VLM/hybrid only).",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload models before parsing.",
    )
    parser.add_argument(
        "--preload-only",
        action="store_true",
        help="Only preload models, do not parse.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    formula_enable = not args.no_formula
    table_enable = not args.no_table
    image_analysis = not args.no_image_analysis

    try:
        parse_kwargs = apply_runtime_config(args.device, args.model_path, args.backend)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    try:
        check_pipeline_dependencies(args.backend, formula_enable=formula_enable)
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    if args.preload_only:
        preload_models(
            backend=args.backend,
            lang=args.lang,
            formula_enable=formula_enable,
            table_enable=table_enable,
            model_path=parse_kwargs.get("model_path"),
        )
        return 0

    if args.path is None:
        parser.error("-p/--path is required unless --preload-only is set")

    try:
        process_path(
            input_path=args.path,
            output_dir=args.output,
            backend=args.backend,
            lang=args.lang,
            parse_method=args.method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            start_page_id=args.start_page,
            end_page_id=args.end_page,
            image_analysis=image_analysis,
            preload=args.preload,
            parse_kwargs=parse_kwargs,
        )
    except HybridDependencyError as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:
        logger.exception(exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
