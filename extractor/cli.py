import argparse
from pathlib import Path

from extractor.pipeline import ExtractionConfig, run_extraction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="English Grammar in Use extractor")
    parser.add_argument("--pdf", required=True, help="Path to source PDF")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--dpi", type=int, default=250, help="Render DPI")
    parser.add_argument("--ocr", type=str, default="true", help="Enable OCR true/false")
    parser.add_argument("--units", nargs=2, type=int, metavar=("START", "END"), default=[1, 145])
    parser.add_argument("--debug", type=str, default="true", help="Enable debug overlays true/false")
    return parser


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = ExtractionConfig(
        pdf_path=Path(args.pdf),
        out_dir=Path(args.out),
        dpi=args.dpi,
        ocr_enabled=str_to_bool(args.ocr),
        unit_start=args.units[0],
        unit_end=args.units[1],
        debug_enabled=str_to_bool(args.debug),
    )
    run_extraction(config)
    return 0
