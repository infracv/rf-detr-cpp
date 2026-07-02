#!/usr/bin/env python3
"""compare_devices.py — aggregate rfdetr_bench JSON results into a Markdown table.

Usage:
    python benchmarks/scripts/compare_devices.py benchmarks/results/*.json
    python benchmarks/scripts/compare_devices.py benchmarks/results/  # scans dir
    python benchmarks/scripts/compare_devices.py benchmarks/results/ --csv > table.csv

JSON schema expected (from rfdetr_bench):
    {
        "device":       str,
        "variant":      str,
        "precision":    str,
        "task":         "detection" | "segmentation",
        "batch":        int,
        "trt_version":  str,
        "latency_ms":   {"mean": float, "p50": float, "p90": float, "p99": float, ...},
        "gpu_ms":       {"mean": float, ...},
        "throughput_fps": float,
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_results(paths: list[Path]) -> list[dict]:
    results = []
    for p in paths:
        if p.is_dir():
            results.extend(load_results(sorted(p.glob("*.json"))))
        else:
            try:
                with open(p) as f:
                    data = json.load(f)
                data["_source"] = str(p)
                results.append(data)
            except Exception as e:
                print(f"warning: skipping {p}: {e}", file=sys.stderr)
    return results


def fmt_float(v, decimals: int = 1, suffix: str = "") -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}{suffix}"


def build_table(results: list[dict], csv_mode: bool = False) -> str:
    if not results:
        return "(no results)"

    sep = "," if csv_mode else " | "
    hdr_fmt = "{}" if csv_mode else "{}"

    headers = [
        "Device", "Variant", "Task", "Prec.", "Batch",
        "Lat mean (ms)", "Lat p50", "Lat p90", "Lat p99",
        "GPU mean (ms)", "Throughput (fps)",
        "TRT",
    ]

    rows = []
    for r in sorted(results, key=lambda x: (
        x.get("device", ""), x.get("variant", ""), x.get("precision", ""),
        x.get("batch", 1)
    )):
        lat = r.get("latency_ms", {})
        gpu = r.get("gpu_ms", {})
        rows.append([
            r.get("device", "?"),
            r.get("variant", "?"),
            r.get("task", "detection"),
            r.get("precision", "?"),
            str(r.get("batch", 1)),
            fmt_float(lat.get("mean")),
            fmt_float(lat.get("p50")),
            fmt_float(lat.get("p90")),
            fmt_float(lat.get("p99")),
            fmt_float(gpu.get("mean")),
            fmt_float(r.get("throughput_fps")),
            r.get("trt_version", "?"),
        ])

    if csv_mode:
        lines = [",".join(headers)]
        for row in rows:
            lines.append(",".join(row))
        return "\n".join(lines)

    # Compute column widths.
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    def fmt_sep() -> str:
        return "| " + " | ".join("-" * w for w in col_widths) + " |"

    lines = [fmt_row(headers), fmt_sep()]
    for row in rows:
        lines.append(fmt_row(row))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate rfdetr_bench JSON results into a table.")
    parser.add_argument("paths", nargs="+", type=Path, help="JSON files or directories")
    parser.add_argument("--csv", action="store_true", help="Output CSV instead of Markdown")
    args = parser.parse_args()

    results = load_results(args.paths)
    if not results:
        print("No valid result files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(results)} result(s).\n", file=sys.stderr)
    print(build_table(results, csv_mode=args.csv))


if __name__ == "__main__":
    main()
