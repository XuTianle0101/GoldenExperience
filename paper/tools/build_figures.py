#!/usr/bin/env python3
"""Build deterministic vector figures for the publication-v5 negative result."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import io
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "artifacts/publication_v5/evidence"
OUTPUT_DIR = ROOT / "artifacts/publication_v5/figures"
V2_DIAGNOSTIC = ROOT / "artifacts/publication_v5/development/v2_method_dev_diagnostic.json"
V3_DIAGNOSTIC = ROOT / "artifacts/publication_v5/development/v3_method_dev_diagnostic.json"
V4_DIAGNOSTIC = ROOT / "artifacts/publication_v5/development/v4_method_dev_diagnostic.json"
FAILED_RECEIPT = (
    ROOT / "artifacts/publication_v5/stages" / "qwen3_4b_to_8b.evaluate_method_dev.v4.failed.json"
)
INITIALIZATION = ROOT / "artifacts/publication_v5/initialization_v4.json"

INK = "#172B36"
MUTED = "#60747D"
GRID = "#D8E1E2"
PAPER = "#FFFFFF"
WASH = "#F3F6F5"
TEAL = "#167D80"
TEAL_LIGHT = "#8DB7B5"
BLUE = "#3A6FA1"
NAVY = "#28536B"
AMBER = "#D6922E"
CORAL = "#C94F46"
CORAL_DARK = "#8D352F"
BLOCKED = "#B7C1C4"

FONT_FAMILY = "Aptos, Source Sans 3, DejaVu Sans, sans-serif"
GATE = 0.45
GREEDY_GATE = 0.98
DRIFT_GATE = 2.0

FIGURES = (
    "fig01_candidate_coverage",
    "fig02_full_prefix_by_length",
    "fig03_task_heterogeneity",
    "fig04_failure_overlap",
    "fig05_method_progression",
    "fig06_pipeline_stop",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Rebuild in memory and fail if tracked figure outputs differ.",
    )
    return parser.parse_args()


def _reject_sealed_path(path: Path) -> None:
    if any("sealed" in part.lower() for part in path.parts):
        raise ValueError(f"refusing to access sealed path: {path}")


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    _reject_sealed_path(path)
    raw = path.read_bytes()

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {path}: {value}")

    payload = json.loads(raw, parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload, raw


def _load_csv(path: Path) -> tuple[list[dict[str, str]], bytes]:
    _reject_sealed_path(path)
    raw = path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    if not rows:
        raise ValueError(f"empty CSV input: {path}")
    return rows, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _csv_bytes(fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    _require(len(color) == 6, f"invalid color {color}")
    return tuple(int(color[index : index + 2], 16) / 255 for index in (0, 2, 4))  # type: ignore[return-value]


def _pdf_number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".") or "0"


def _pdf_escape(value: str) -> str:
    value.encode("ascii")
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Drawing:
    def __init__(self, width: int, height: int, *, title: str, description: str) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.description = description
        self._svg: list[str] = []
        self._pdf: list[str] = []
        self.rect(0, 0, width, height, fill=PAPER)

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float = 1,
        radius: float = 0,
    ) -> None:
        attributes = [
            f'x="{_pdf_number(x)}"',
            f'y="{_pdf_number(y)}"',
            f'width="{_pdf_number(width)}"',
            f'height="{_pdf_number(height)}"',
        ]
        if radius:
            attributes.append(f'rx="{_pdf_number(radius)}"')
        attributes.append(f'fill="{fill or "none"}"')
        if stroke:
            attributes.extend([f'stroke="{stroke}"', f'stroke-width="{_pdf_number(stroke_width)}"'])
        self._svg.append(f"<rect {' '.join(attributes)}/>")

        commands: list[str] = []
        if fill:
            r, g, b = _rgb(fill)
            commands.append(f"{_pdf_number(r)} {_pdf_number(g)} {_pdf_number(b)} rg")
        if stroke:
            r, g, b = _rgb(stroke)
            commands.extend(
                [
                    f"{_pdf_number(r)} {_pdf_number(g)} {_pdf_number(b)} RG",
                    f"{_pdf_number(stroke_width)} w",
                ]
            )
        commands.append(
            f"{_pdf_number(x)} {_pdf_number(self.height - y - height)} "
            f"{_pdf_number(width)} {_pdf_number(height)} re"
        )
        commands.append("B" if fill and stroke else "f" if fill else "S")
        self._pdf.extend(commands)

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = INK,
        stroke_width: float = 1,
        dash: Sequence[float] | None = None,
    ) -> None:
        dash_svg = ""
        dash_pdf = "[] 0 d"
        if dash:
            values = ",".join(_pdf_number(value) for value in dash)
            dash_svg = f' stroke-dasharray="{values}"'
            dash_pdf = f"[{' '.join(_pdf_number(value) for value in dash)}] 0 d"
        self._svg.append(
            f'<line x1="{_pdf_number(x1)}" y1="{_pdf_number(y1)}" '
            f'x2="{_pdf_number(x2)}" y2="{_pdf_number(y2)}" stroke="{stroke}" '
            f'stroke-width="{_pdf_number(stroke_width)}"{dash_svg}/>'
        )
        r, g, b = _rgb(stroke)
        self._pdf.extend(
            [
                f"{_pdf_number(r)} {_pdf_number(g)} {_pdf_number(b)} RG",
                f"{_pdf_number(stroke_width)} w",
                dash_pdf,
                f"{_pdf_number(x1)} {_pdf_number(self.height - y1)} m",
                f"{_pdf_number(x2)} {_pdf_number(self.height - y2)} l S",
                "[] 0 d",
            ]
        )

    def polygon(
        self,
        points: Sequence[tuple[float, float]],
        *,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 1,
    ) -> None:
        svg_points = " ".join(f"{_pdf_number(x)},{_pdf_number(y)}" for x, y in points)
        self._svg.append(
            f'<polygon points="{svg_points}" fill="{fill}"'
            + (f' stroke="{stroke}" stroke-width="{_pdf_number(stroke_width)}"' if stroke else "")
            + "/>"
        )
        r, g, b = _rgb(fill)
        commands = [f"{_pdf_number(r)} {_pdf_number(g)} {_pdf_number(b)} rg"]
        if stroke:
            sr, sg, sb = _rgb(stroke)
            commands.extend(
                [
                    f"{_pdf_number(sr)} {_pdf_number(sg)} {_pdf_number(sb)} RG",
                    f"{_pdf_number(stroke_width)} w",
                ]
            )
        first_x, first_y = points[0]
        commands.append(f"{_pdf_number(first_x)} {_pdf_number(self.height - first_y)} m")
        commands.extend(f"{_pdf_number(x)} {_pdf_number(self.height - y)} l" for x, y in points[1:])
        commands.extend(["h", "B" if stroke else "f"])
        self._pdf.extend(commands)

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        *,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 1,
    ) -> None:
        self._svg.append(
            f'<circle cx="{_pdf_number(x)}" cy="{_pdf_number(y)}" '
            f'r="{_pdf_number(radius)}" fill="{fill}"'
            + (f' stroke="{stroke}" stroke-width="{_pdf_number(stroke_width)}"' if stroke else "")
            + "/>"
        )
        r, g, b = _rgb(fill)
        commands = [f"{_pdf_number(r)} {_pdf_number(g)} {_pdf_number(b)} rg"]
        if stroke:
            sr, sg, sb = _rgb(stroke)
            commands.extend(
                [
                    f"{_pdf_number(sr)} {_pdf_number(sg)} {_pdf_number(sb)} RG",
                    f"{_pdf_number(stroke_width)} w",
                ]
            )
        cy = self.height - y
        k = radius * 0.5522847498
        commands.extend(
            [
                f"{_pdf_number(x + radius)} {_pdf_number(cy)} m",
                f"{_pdf_number(x + radius)} {_pdf_number(cy + k)} "
                f"{_pdf_number(x + k)} {_pdf_number(cy + radius)} "
                f"{_pdf_number(x)} {_pdf_number(cy + radius)} c",
                f"{_pdf_number(x - k)} {_pdf_number(cy + radius)} "
                f"{_pdf_number(x - radius)} {_pdf_number(cy + k)} "
                f"{_pdf_number(x - radius)} {_pdf_number(cy)} c",
                f"{_pdf_number(x - radius)} {_pdf_number(cy - k)} "
                f"{_pdf_number(x - k)} {_pdf_number(cy - radius)} "
                f"{_pdf_number(x)} {_pdf_number(cy - radius)} c",
                f"{_pdf_number(x + k)} {_pdf_number(cy - radius)} "
                f"{_pdf_number(x + radius)} {_pdf_number(cy - k)} "
                f"{_pdf_number(x + radius)} {_pdf_number(cy)} c",
                "B" if stroke else "f",
            ]
        )
        self._pdf.extend(commands)

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: float = 10,
        fill: str = INK,
        weight: str = "normal",
        anchor: str = "start",
        letter_spacing: float = 0,
    ) -> None:
        value.encode("ascii")
        escaped = html.escape(value)
        self._svg.append(
            f'<text x="{_pdf_number(x)}" y="{_pdf_number(y)}" fill="{fill}" '
            f'font-family="{FONT_FAMILY}" font-size="{_pdf_number(size)}" '
            f'font-weight="{weight}" text-anchor="{anchor}" '
            f'letter-spacing="{_pdf_number(letter_spacing)}">{escaped}</text>'
        )
        approximate_width = len(value) * size * (0.56 if weight == "bold" else 0.52)
        pdf_x = x
        if anchor == "middle":
            pdf_x -= approximate_width / 2
        elif anchor == "end":
            pdf_x -= approximate_width
        r, g, b = _rgb(fill)
        font = "/F2" if weight == "bold" else "/F1"
        self._pdf.append(
            f"BT {font} {_pdf_number(size)} Tf "
            f"{_pdf_number(r)} {_pdf_number(g)} {_pdf_number(b)} rg "
            f"{_pdf_number(pdf_x)} {_pdf_number(self.height - y)} Td "
            f"({_pdf_escape(value)}) Tj ET"
        )

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = INK,
        stroke_width: float = 1.5,
        dash: Sequence[float] | None = None,
    ) -> None:
        self.line(x1, y1, x2, y2, stroke=stroke, stroke_width=stroke_width, dash=dash)
        angle = math.atan2(y2 - y1, x2 - x1)
        length = 7
        spread = 0.55
        points = [
            (x2, y2),
            (x2 - length * math.cos(angle - spread), y2 - length * math.sin(angle - spread)),
            (x2 - length * math.cos(angle + spread), y2 - length * math.sin(angle + spread)),
        ]
        self.polygon(points, fill=stroke)

    def svg_bytes(self) -> bytes:
        title = html.escape(self.title)
        description = html.escape(self.description)
        body = "\n  ".join(self._svg)
        payload = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}" '
            'role="img" aria-labelledby="figure-title figure-description">\n'
            f'  <title id="figure-title">{title}</title>\n'
            f'  <desc id="figure-description">{description}</desc>\n'
            f"  {body}\n"
            "</svg>\n"
        )
        raw = payload.encode("utf-8")
        ET.fromstring(raw)
        return raw

    def pdf_bytes(self) -> bytes:
        stream = ("\n".join(self._pdf) + "\n").encode("ascii")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
                "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
            ).encode("ascii"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream",
        ]
        output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for index, obj in enumerate(objects, start=1):
            offsets.append(len(output))
            output.extend(f"{index} 0 obj\n".encode("ascii"))
            output.extend(obj)
            output.extend(b"\nendobj\n")
        xref = len(output)
        output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        output.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        output.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
            ).encode("ascii")
        )
        raw = bytes(output)
        _require(raw.startswith(b"%PDF-1.4") and raw.endswith(b"%%EOF\n"), "invalid PDF")
        return raw


def _frame(
    drawing: Drawing,
    *,
    title: str,
    subtitle: str,
    source: str,
    accent: str = TEAL,
) -> None:
    drawing.rect(28, 22, 8, 30, fill=accent)
    drawing.text(48, 38, title, size=19, weight="bold")
    drawing.text(48, 57, subtitle, size=9.5, fill=MUTED)
    drawing.line(28, drawing.height - 28, drawing.width - 28, drawing.height - 28, stroke=GRID)
    drawing.text(28, drawing.height - 12, source, size=7.5, fill=MUTED)


def _axes(
    drawing: Drawing,
    *,
    left: float,
    top: float,
    right: float,
    bottom: float,
    y_ticks: Sequence[float],
    y_max: float,
    formatter: str,
) -> None:
    for tick in y_ticks:
        y = bottom - (tick / y_max) * (bottom - top)
        drawing.line(left, y, right, y, stroke=GRID, stroke_width=0.8)
        label = f"{tick:.0%}" if formatter == "percent" else _number(tick)
        drawing.text(left - 10, y + 3, label, size=8, fill=MUTED, anchor="end")
    drawing.line(left, top, left, bottom, stroke=INK, stroke_width=1)
    drawing.line(left, bottom, right, bottom, stroke=INK, stroke_width=1)


def _candidate_data(
    candidates: Sequence[Mapping[str, str]], safe_sets: Sequence[Mapping[str, str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rows.append(
            {
                "kind": "registered_candidate",
                "candidate_id": candidate["candidate_id"],
                "label": f"r{candidate['rank']}/s{candidate['seed']}",
                "rank": int(candidate["rank"]),
                "seed": int(candidate["seed"]),
                "safe_count": int(candidate["safe_count"]),
                "sample_count": int(candidate["sample_count"]),
                "coverage": float(candidate["oracle_safe_coverage"]),
                "is_deployment_candidate": candidate["is_deployment_candidate"] == "True",
                "is_post_hoc_oracle": False,
                "coverage_gate": GATE,
            }
        )
    oracle = next(row for row in safe_sets if row["choice_set"] == "all_nine_any_candidate")
    rows.append(
        {
            "kind": "all_candidate_oracle",
            "candidate_id": "not_deployable",
            "label": "9-way oracle",
            "rank": "",
            "seed": "",
            "safe_count": int(oracle["safe_count"]),
            "sample_count": 1024,
            "coverage": float(oracle["coverage"]),
            "is_deployment_candidate": False,
            "is_post_hoc_oracle": True,
            "coverage_gate": GATE,
        }
    )
    return rows


def _figure_candidate_coverage(rows: Sequence[Mapping[str, Any]]) -> Drawing:
    drawing = Drawing(
        760,
        450,
        title="Every registered candidate misses the coverage gate",
        description="Coverage bars for nine rank and seed candidates plus a post-hoc oracle.",
    )
    _frame(
        drawing,
        title="Every registered candidate misses the coverage gate",
        subtitle="Qwen3-4B -> Qwen3-8B, 1,024 method-dev prompts; star marks deployment",
        source="Source: full v4 method-dev report; gate = 0.45 oracle-safe coverage.",
        accent=CORAL,
    )
    left, top, right, bottom = 66, 92, 730, 366
    _axes(
        drawing,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        y_ticks=(0, 0.1, 0.2, 0.3, 0.4, 0.5),
        y_max=0.5,
        formatter="percent",
    )
    gate_y = bottom - (GATE / 0.5) * (bottom - top)
    drawing.line(left, gate_y, right, gate_y, stroke=CORAL, stroke_width=1.7, dash=(6, 4))
    drawing.text(right, gate_y - 6, "registered gate 45%", size=8.5, fill=CORAL, anchor="end")

    step = (right - left) / len(rows)
    rank_colors = {32: TEAL_LIGHT, 64: TEAL, 128: NAVY}
    for index, row in enumerate(rows):
        center = left + (index + 0.5) * step
        width = step * 0.58
        coverage = float(row["coverage"])
        y = bottom - (coverage / 0.5) * (bottom - top)
        if row["is_post_hoc_oracle"]:
            drawing.rect(
                center - width / 2,
                y,
                width,
                bottom - y,
                fill=WASH,
                stroke=CORAL,
                stroke_width=2,
            )
        else:
            drawing.rect(
                center - width / 2,
                y,
                width,
                bottom - y,
                fill=rank_colors[int(row["rank"])],
            )
        drawing.text(center, y - 7, f"{coverage:.3f}", size=7.7, anchor="middle")
        if row["is_deployment_candidate"]:
            star_x = center + 18
            star_y = y - 13
            drawing.polygon(
                [
                    (star_x, star_y - 8),
                    (star_x + 2, star_y - 3),
                    (star_x + 8, star_y - 3),
                    (star_x + 3, star_y + 1),
                    (star_x + 5, star_y + 7),
                    (star_x, star_y + 3),
                    (star_x - 5, star_y + 7),
                    (star_x - 3, star_y + 1),
                    (star_x - 8, star_y - 3),
                    (star_x - 2, star_y - 3),
                ],
                fill=AMBER,
                stroke=PAPER,
                stroke_width=0.8,
            )
        label = str(row["label"])
        if row["is_post_hoc_oracle"]:
            drawing.text(center, bottom + 17, "9-way", size=8, anchor="middle", fill=CORAL)
            drawing.text(center, bottom + 29, "oracle", size=8, anchor="middle", fill=CORAL)
        else:
            drawing.text(center, bottom + 19, label, size=8, anchor="middle")
    drawing.text(left, top - 10, "Oracle-safe coverage", size=8.5, fill=MUTED)
    return drawing


def _row_safe(row: Mapping[str, Any]) -> bool:
    greedy = int(row["greedy_matches"]) / int(row["greedy_tokens"])
    drift = (
        abs(
            math.expm1(
                (float(row["bridge_nll"]) - float(row["native_nll"])) / int(row["teacher_tokens"])
            )
        )
        * 100
    )
    task_regression = float(row["native_task_score"]) >= float(
        row["task_pass_threshold"]
    ) and float(row["bridge_task_score"]) < float(row["task_pass_threshold"])
    return not task_regression and greedy >= GREEDY_GATE and drift <= DRIFT_GATE


def _teacher_alignment_data(
    archive_raw: bytes,
    evidence_manifest: Mapping[str, Any],
    v3: Mapping[str, Any],
    v4: Mapping[str, Any],
) -> list[dict[str, Any]]:
    archive_entry = next(
        item
        for item in evidence_manifest["artifacts"]
        if item["role"] == "canonical_compressed_method_dev_report"
    )
    _require(_sha256(archive_raw) == archive_entry["sha256"], "report archive hash differs")
    report_raw = gzip.decompress(archive_raw)
    _require(
        _sha256(report_raw) == archive_entry["uncompressed_sha256"],
        "report archive payload hash differs",
    )
    report = json.loads(report_raw)
    selected = [row for row in report["measurements"] if row["rank"] == 128 and row["seed"] == 17]
    _require(len(selected) == 1024, "fixed v4 candidate is incomplete")
    v4_counts: Counter[int] = Counter()
    for row in selected:
        if _row_safe(row):
            v4_counts[int(str(row["sample_id"]).split(".")[1])] += 1
    deltas = v4["method_progression"]["same_rank_128_seed_17_v4_minus_v3"][
        "safe_count_by_token_bucket"
    ]
    rows: list[dict[str, Any]] = []
    for bucket in (128, 512, 2048, 8192):
        v4_count = v4_counts[bucket]
        v3_count = v4_count - int(deltas[str(bucket)])
        rows.extend(
            [
                {
                    "version": "v3",
                    "supervision": "sampled_prefix_teacher",
                    "rank": 128,
                    "seed": 17,
                    "token_bucket": bucket,
                    "safe_count": v3_count,
                    "sample_count": 256,
                    "coverage": v3_count / 256,
                    "v4_minus_v3_safe_count": int(deltas[str(bucket)]),
                },
                {
                    "version": "v4",
                    "supervision": "full_prefix_teacher",
                    "rank": 128,
                    "seed": 17,
                    "token_bucket": bucket,
                    "safe_count": v4_count,
                    "sample_count": 256,
                    "coverage": v4_count / 256,
                    "v4_minus_v3_safe_count": int(deltas[str(bucket)]),
                },
            ]
        )
    _require(
        sum(int(row["safe_count"]) for row in rows if row["version"] == "v3")
        == v3["full_method_dev"]["deployment_v3"]["safe_count"],
        "v3 bucket counts do not match the fixed-candidate total",
    )
    _require(
        sum(int(row["safe_count"]) for row in rows if row["version"] == "v4") == 159,
        "v4 bucket counts do not match the fixed-candidate total",
    )
    return rows


def _figure_teacher_alignment(rows: Sequence[Mapping[str, Any]]) -> Drawing:
    drawing = Drawing(
        720,
        440,
        title="Full-prefix supervision helps most at long prefixes",
        description="Safe prompt counts by token bucket for v3 and v4 at fixed rank and seed.",
    )
    _frame(
        drawing,
        title="Full-prefix supervision helps most at long prefixes",
        subtitle="Controlled comparison: rank 128, seed 17; 256 prompts in each prefix bucket",
        source="Source: v3/v4 mechanism diagnostics and the archived v4 method-dev report.",
    )
    left, top, right, bottom = 70, 90, 690, 350
    _axes(
        drawing,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        y_ticks=(0, 10, 20, 30, 40, 50),
        y_max=50,
        formatter="count",
    )
    buckets = (128, 512, 2048, 8192)
    group_width = (right - left) / len(buckets)
    bar_width = 36
    for index, bucket in enumerate(buckets):
        center = left + (index + 0.5) * group_width
        values = {
            str(row["version"]): int(row["safe_count"])
            for row in rows
            if int(row["token_bucket"]) == bucket
        }
        for version, offset, color in (("v3", -21, BLUE), ("v4", 21, TEAL)):
            value = values[version]
            y = bottom - value / 50 * (bottom - top)
            drawing.rect(center + offset - bar_width / 2, y, bar_width, bottom - y, fill=color)
            drawing.text(center + offset, y - 7, str(value), size=9, anchor="middle", weight="bold")
        delta = values["v4"] - values["v3"]
        delta_y = (
            min(
                bottom - values["v3"] / 50 * (bottom - top),
                bottom - values["v4"] / 50 * (bottom - top),
            )
            - 26
        )
        drawing.text(
            center,
            delta_y,
            f"{delta:+d}",
            size=9,
            anchor="middle",
            fill=TEAL if delta >= 0 else CORAL,
            weight="bold",
        )
        drawing.text(center, bottom + 21, f"{bucket:,}", size=9, anchor="middle")
    drawing.text(left, top - 9, "Safe prompts", size=8.5, fill=MUTED)
    drawing.text(
        (left + right) / 2, bottom + 42, "Prefix tokens", size=8.5, fill=MUTED, anchor="middle"
    )
    drawing.rect(left + 14, top + 5, 12, 12, fill=BLUE)
    drawing.text(left + 32, top + 15, "v3 sampled-prefix", size=8.5)
    drawing.rect(left + 14, top + 23, 12, 12, fill=TEAL)
    drawing.text(left + 32, top + 33, "v4 full-prefix", size=8.5)
    return drawing


TASK_LABELS = {
    "function_calling": "Function calling",
    "competition_math": "Competition math",
    "grade_school_math": "Grade-school math",
    "long_context_qa": "Long-context QA",
    "python_code_generation": "Python code",
}


def _task_data(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "task": row["task"],
            "display_label": TASK_LABELS[row["task"]],
            "sample_count": int(row["sample_count"]),
            "safe_count": int(row["safe_count"]),
            "oracle_safe_coverage": float(row["oracle_safe_coverage"]),
            "greedy_agreement": float(row["greedy_agreement"]),
            "perplexity_drift_pct": float(row["perplexity_drift_pct"]),
            "coverage_gate": GATE,
            "greedy_gate": GREEDY_GATE,
            "max_perplexity_drift_pct": DRIFT_GATE,
        }
        for row in rows
    ]


def _panel_scale(value: float, minimum: float, maximum: float, left: float, right: float) -> float:
    return left + (value - minimum) / (maximum - minimum) * (right - left)


def _figure_task_heterogeneity(rows: Sequence[Mapping[str, Any]]) -> Drawing:
    drawing = Drawing(
        820,
        440,
        title="Behavioral safety is sharply task-dependent",
        description=(
            "Aligned task panels for safe coverage, greedy agreement, and perplexity drift."
        ),
    )
    _frame(
        drawing,
        title="Behavioral safety is sharply task-dependent",
        subtitle="Registered deployment candidate: rank 64, seed 17; vertical lines show gates",
        source=(
            "Source: archived v4 method-dev report; aggregates use the fixed deployment candidate."
        ),
        accent=AMBER,
    )
    label_x = 28
    panels = (
        ("oracle_safe_coverage", "Safe coverage", 190, 375, 0.0, 1.0, GATE, False),
        ("greedy_agreement", "Greedy agreement", 405, 590, 0.5, 1.0, GREEDY_GATE, False),
        ("perplexity_drift_pct", "PPL drift", 620, 790, 0.0, 70.0, DRIFT_GATE, True),
    )
    top = 112
    row_gap = 52
    for field, title, left, right, minimum, maximum, gate, lower_is_better in panels:
        drawing.text((left + right) / 2, 90, title, size=9.5, anchor="middle", weight="bold")
        drawing.line(left, 100, right, 100, stroke=INK)
        drawing.text(left, 112, f"{minimum:g}", size=7.5, fill=MUTED, anchor="middle")
        drawing.text(right, 112, f"{maximum:g}", size=7.5, fill=MUTED, anchor="middle")
        gate_x = _panel_scale(gate, minimum, maximum, left, right)
        drawing.line(gate_x, 96, gate_x, 354, stroke=CORAL, stroke_width=1.2, dash=(4, 3))
        direction = "max" if lower_is_better else "min"
        drawing.text(gate_x, 368, f"{direction} {gate:g}", size=7.2, fill=CORAL, anchor="middle")
        for index, row in enumerate(rows):
            y = top + 34 + index * row_gap
            value = float(row[field])
            x = _panel_scale(value, minimum, maximum, left, right)
            drawing.line(left, y, right, y, stroke=GRID, stroke_width=0.8)
            drawing.line(left, y, x, y, stroke=TEAL_LIGHT, stroke_width=3)
            passes = value <= gate if lower_is_better else value >= gate
            drawing.circle(x, y, 5.5, fill=TEAL if passes else CORAL, stroke=PAPER, stroke_width=1)
            label = f"{value:.1f}%" if field == "perplexity_drift_pct" else f"{value:.3f}"
            anchor = "end" if x > right - 34 else "start"
            offset = -8 if anchor == "end" else 8
            drawing.text(x + offset, y - 8, label, size=7.3, fill=INK, anchor=anchor)
    for index, row in enumerate(rows):
        y = top + 34 + index * row_gap
        drawing.text(label_x, y + 3, str(row["display_label"]), size=9, weight="bold")
        drawing.text(
            174,
            y + 3,
            f"n={row['sample_count']}",
            size=7.2,
            fill=MUTED,
            anchor="end",
        )
    return drawing


def _failure_data(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "reason_combination": row["reason_combination"],
            "task_regression": row["task_regression"] == "True",
            "greedy_failure": row["greedy_failure"] == "True",
            "drift_failure": row["drift_failure"] == "True",
            "is_unsafe": row["is_unsafe"] == "True",
            "count": int(row["count"]),
            "fraction": float(row["fraction"]),
        }
        for row in rows
    ]


def _figure_failure_overlap(rows: Sequence[Mapping[str, Any]]) -> Drawing:
    drawing = Drawing(
        760,
        440,
        title="Most prompts violate both token-level behavior gates",
        description=(
            "Exact failure partition and marginal violation counts for the deployment candidate."
        ),
    )
    _frame(
        drawing,
        title="Most prompts violate both token-level behavior gates",
        subtitle="Exact partition of all 1,024 prompts for rank 64, seed 17",
        source="Source: archived v4 method-dev report; combinations are mutually exclusive.",
        accent=CORAL,
    )
    color_map = {
        "safe": TEAL,
        "drift_only": AMBER,
        "greedy_only": BLUE,
        "greedy_and_drift": CORAL,
        "task_regression_only": BLOCKED,
        "task_regression_and_drift": CORAL_DARK,
        "task_regression_and_greedy": NAVY,
        "task_regression_and_greedy_and_drift": CORAL_DARK,
    }
    label_map = {
        "safe": "Safe",
        "drift_only": "Drift only",
        "greedy_only": "Greedy only",
        "greedy_and_drift": "Greedy + drift",
        "task_regression_and_greedy": "Task + greedy",
        "task_regression_and_greedy_and_drift": "Task + greedy + drift",
    }
    active = [row for row in rows if int(row["count"]) > 0]
    left, right, y, height = 58, 710, 112, 58
    cursor = left
    for row in active:
        width = (right - left) * int(row["count"]) / 1024
        drawing.rect(cursor, y, width, height, fill=color_map[str(row["reason_combination"])])
        if width > 52:
            text_color = PAPER if row["reason_combination"] in {"safe", "greedy_and_drift"} else INK
            drawing.text(
                cursor + width / 2,
                y + 25,
                label_map[str(row["reason_combination"])],
                size=8,
                fill=text_color,
                anchor="middle",
                weight="bold",
            )
            drawing.text(
                cursor + width / 2,
                y + 41,
                str(row["count"]),
                size=8,
                fill=text_color,
                anchor="middle",
            )
        cursor += width
    drawing.text(left, y - 10, "Mutually exclusive outcome partition", size=8.5, fill=MUTED)
    drawing.text(right, y - 10, "n = 1,024", size=8.5, fill=MUTED, anchor="end")

    legend_columns = (left, left + 220, left + 440)
    for index, row in enumerate(active):
        legend_x = legend_columns[index % 3]
        legend_y = 196 + (index // 3) * 24
        label = label_map[str(row["reason_combination"])]
        drawing.rect(legend_x, legend_y - 9, 10, 10, fill=color_map[str(row["reason_combination"])])
        drawing.text(legend_x + 15, legend_y, f"{label}: {row['count']}", size=7.5)

    marginals = {
        "Task regression": sum(int(row["count"]) for row in rows if row["task_regression"]),
        "Greedy < 0.98": sum(int(row["count"]) for row in rows if row["greedy_failure"]),
        "PPL drift > 2%": sum(int(row["count"]) for row in rows if row["drift_failure"]),
    }
    drawing.text(left, 248, "Marginal violations (overlapping)", size=9, fill=MUTED, weight="bold")
    bar_left, bar_right = 190, 700
    colors = (NAVY, BLUE, AMBER)
    for index, ((label, count), color) in enumerate(zip(marginals.items(), colors, strict=True)):
        bar_y = 270 + index * 38
        drawing.text(left, bar_y + 10, label, size=8.5)
        drawing.rect(bar_left, bar_y, bar_right - bar_left, 15, fill=WASH)
        drawing.rect(bar_left, bar_y, (bar_right - bar_left) * count / 1024, 15, fill=color)
        drawing.text(
            bar_right + 8,
            bar_y + 11,
            f"{count} ({count / 1024:.1%})",
            size=8,
            fill=INK,
        )
    unsafe = sum(int(row["count"]) for row in rows if row["is_unsafe"])
    drawing.rect(568, 225, 142, 27, fill=CORAL, radius=4)
    drawing.text(
        639,
        243,
        f"Unsafe union: {unsafe} ({unsafe / 1024:.1%})",
        size=8.3,
        fill=PAPER,
        anchor="middle",
        weight="bold",
    )
    return drawing


def _progression_data(
    v2: Mapping[str, Any],
    v3: Mapping[str, Any],
    v4: Mapping[str, Any],
    evidence_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    oracle = v4["method_progression"]["candidate_oracle_safe_union"]
    deployment = {
        "v2": {
            "coverage": v2["deployment_candidate"]["oracle_safe_coverage"],
            "safe_count": v2["deployment_candidate"]["safe_count"],
            "rank": v2["deployment_candidate"]["rank"],
        },
        "v3": {
            "coverage": v3["full_method_dev"]["deployment_v3"]["oracle_safe_coverage"],
            "safe_count": v3["full_method_dev"]["deployment_v3"]["safe_count"],
            "rank": 128,
        },
        "v4": {
            "coverage": evidence_manifest["registered_deployment"]["oracle_safe_coverage"],
            "safe_count": evidence_manifest["registered_deployment"]["safe_count"],
            "rank": evidence_manifest["registered_deployment"]["rank"],
        },
    }
    methods = {
        "v2": "detached_generation_terms",
        "v3": "sampled_prefix_teacher",
        "v4": "full_prefix_teacher",
    }
    rows: list[dict[str, Any]] = []
    for version in ("v2", "v3", "v4"):
        rows.extend(
            [
                {
                    "version": version,
                    "method": methods[version],
                    "metric": "registered_deployment",
                    "selected_rank": deployment[version]["rank"],
                    "safe_count": deployment[version]["safe_count"],
                    "sample_count": 1024,
                    "coverage": deployment[version]["coverage"],
                    "deployable": True,
                    "coverage_gate": GATE,
                },
                {
                    "version": version,
                    "method": methods[version],
                    "metric": "all_candidate_oracle_union",
                    "selected_rank": deployment[version]["rank"],
                    "safe_count": oracle[version]["count"],
                    "sample_count": 1024,
                    "coverage": oracle[version]["coverage"],
                    "deployable": False,
                    "coverage_gate": GATE,
                },
            ]
        )
    return rows


def _figure_progression(rows: Sequence[Mapping[str, Any]]) -> Drawing:
    drawing = Drawing(
        720,
        440,
        title="Method progression remains below the registered gate",
        description="Deployment and nine-candidate oracle coverage for v2, v3, and v4.",
    )
    _frame(
        drawing,
        title="Method progression remains below the registered gate",
        subtitle="Descriptive comparison: selected deployment rank changes across versions",
        source="Source: v2/v3/v4 diagnostics; oracle bars use prohibited target-derived selection.",
        accent=CORAL,
    )
    left, top, right, bottom = 72, 92, 690, 345
    _axes(
        drawing,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        y_ticks=(0, 0.1, 0.2, 0.3, 0.4, 0.5),
        y_max=0.5,
        formatter="percent",
    )
    gate_y = bottom - GATE / 0.5 * (bottom - top)
    drawing.line(left, gate_y, right, gate_y, stroke=CORAL, stroke_width=1.6, dash=(6, 4))
    drawing.text(right, gate_y - 6, "gate 45%", size=8, fill=CORAL, anchor="end")
    versions = ("v2", "v3", "v4")
    group_width = (right - left) / len(versions)
    bar_width = 48
    for index, version in enumerate(versions):
        center = left + (index + 0.5) * group_width
        version_rows = {str(row["metric"]): row for row in rows if row["version"] == version}
        for metric, offset, color in (
            ("registered_deployment", -29, TEAL),
            ("all_candidate_oracle_union", 29, CORAL),
        ):
            coverage = float(version_rows[metric]["coverage"])
            y = bottom - coverage / 0.5 * (bottom - top)
            if metric == "registered_deployment":
                drawing.rect(center + offset - bar_width / 2, y, bar_width, bottom - y, fill=color)
            else:
                drawing.rect(
                    center + offset - bar_width / 2,
                    y,
                    bar_width,
                    bottom - y,
                    fill=WASH,
                    stroke=color,
                    stroke_width=2,
                )
            drawing.text(center + offset, y - 7, f"{coverage:.3f}", size=8, anchor="middle")
        rank = version_rows["registered_deployment"]["selected_rank"]
        drawing.text(center, bottom + 22, version, size=10, anchor="middle", weight="bold")
        drawing.text(
            center, bottom + 36, f"deployment rank {rank}", size=7.5, anchor="middle", fill=MUTED
        )
    drawing.rect(left + 8, top + 8, 12, 12, fill=TEAL)
    drawing.text(left + 26, top + 18, "Registered deployment", size=8.5)
    drawing.rect(left + 155, top + 8, 12, 12, fill=WASH, stroke=CORAL, stroke_width=2)
    drawing.text(left + 173, top + 18, "Nine-candidate oracle union", size=8.5)
    return drawing


def _pipeline_data(
    failed: Mapping[str, Any], initialization: Mapping[str, Any]
) -> list[dict[str, Any]]:
    _require(failed["attempt"]["status"] == "failed", "method-dev status changed")
    _require(
        initialization["pipeline"]["semantic_sealed_state"] == "locked",
        "semantic sealed state changed",
    )
    return [
        {
            "stage": "transport_train_collection",
            "display_label": "Transport train",
            "status": "completed",
            "evidence": "4,096 rows collected",
        },
        {
            "stage": "transport_fit",
            "display_label": "Transport fit",
            "status": "completed",
            "evidence": "9 candidates; 1,536 steps",
        },
        {
            "stage": "method_dev_collection",
            "display_label": "Method-dev trace",
            "status": "completed",
            "evidence": "1,024 prompts collected",
        },
        {
            "stage": "method_dev_evaluation",
            "display_label": "Method-dev gate",
            "status": "failed",
            "evidence": "142/1,024 safe; 0.1387 < 0.45",
        },
        {
            "stage": "other_direction_fits",
            "display_label": "Other directions",
            "status": "blocked",
            "evidence": "not fit",
        },
        {
            "stage": "selector_and_calibration",
            "display_label": "Selector + calibration",
            "status": "blocked",
            "evidence": "not fit",
        },
        {
            "stage": "validation",
            "display_label": "Validation x4",
            "status": "blocked",
            "evidence": "not run",
        },
        {
            "stage": "semantic_sealed",
            "display_label": "Semantic sealed",
            "status": "locked",
            "evidence": "payload unopened",
        },
        {
            "stage": "runtime_audit",
            "display_label": "Runtime audit",
            "status": "blocked",
            "evidence": "not run",
        },
    ]


def _stage_box(
    drawing: Drawing,
    x: float,
    y: float,
    width: float,
    height: float,
    row: Mapping[str, Any],
) -> None:
    status = str(row["status"])
    border = TEAL if status == "completed" else CORAL if status == "failed" else BLOCKED
    fill = PAPER if status in {"completed", "failed"} else WASH
    drawing.rect(x, y, width, height, fill=fill, stroke=border, stroke_width=1.7, radius=5)
    drawing.rect(x, y, width, 7, fill=border)
    drawing.text(x + 10, y + 29, str(row["display_label"]), size=9, weight="bold")
    drawing.text(x + 10, y + 47, str(row["evidence"]), size=7.3, fill=MUTED)
    drawing.text(
        x + width - 9, y + 17, status.upper(), size=6.8, fill=border, anchor="end", weight="bold"
    )


def _figure_pipeline_stop(rows: Sequence[Mapping[str, Any]]) -> Drawing:
    drawing = Drawing(
        1180,
        410,
        title="The failed method-dev gate stops the evidence chain",
        description="Pipeline diagram showing completed, failed, blocked, and locked stages.",
    )
    _frame(
        drawing,
        title="The failed method-dev gate stops the evidence chain",
        subtitle=(
            "Implemented stages to the right remain protocol descriptions, not executed evidence"
        ),
        source=(
            "Source: publication-v5 failed-stage receipt and immutable workspace "
            "initialization receipt."
        ),
        accent=CORAL,
    )
    by_stage = {str(row["stage"]): row for row in rows}
    box_w, box_h = 142, 64
    positions = {
        "transport_train_collection": (28, 120),
        "transport_fit": (204, 120),
        "method_dev_collection": (204, 235),
        "method_dev_evaluation": (380, 120),
        "other_direction_fits": (596, 92),
        "selector_and_calibration": (596, 205),
        "validation": (776, 148),
        "semantic_sealed": (948, 96),
        "runtime_audit": (948, 217),
    }
    drawing.arrow(170, 152, 204, 152, stroke=TEAL)
    drawing.arrow(346, 152, 380, 152, stroke=TEAL)
    drawing.arrow(346, 267, 368, 267, stroke=TEAL)
    drawing.line(368, 267, 368, 173, stroke=TEAL, stroke_width=1.5)
    drawing.arrow(368, 173, 380, 173, stroke=TEAL)

    stop_x = 554
    drawing.line(stop_x, 85, stop_x, 300, stroke=CORAL, stroke_width=4)
    drawing.polygon(
        [
            (stop_x - 9, 80),
            (stop_x + 9, 80),
            (stop_x + 13, 84),
            (stop_x + 13, 102),
            (stop_x + 9, 106),
            (stop_x - 9, 106),
            (stop_x - 13, 102),
            (stop_x - 13, 84),
        ],
        fill=CORAL,
    )
    drawing.text(stop_x, 96, "STOP", size=7, fill=PAPER, anchor="middle", weight="bold")
    drawing.text(stop_x, 318, "protocol stop", size=8, fill=CORAL, anchor="middle", weight="bold")
    drawing.arrow(522, 152, stop_x - 14, 152, stroke=CORAL, stroke_width=2)

    dash = (5, 4)
    drawing.arrow(stop_x + 14, 133, 596, 124, stroke=BLOCKED, dash=dash)
    drawing.arrow(stop_x + 14, 180, 596, 237, stroke=BLOCKED, dash=dash)
    drawing.arrow(738, 124, 776, 166, stroke=BLOCKED, dash=dash)
    drawing.arrow(738, 237, 776, 194, stroke=BLOCKED, dash=dash)
    drawing.arrow(918, 180, 948, 128, stroke=BLOCKED, dash=dash)
    drawing.arrow(1090, 128, 1105, 128, stroke=BLOCKED, dash=dash)
    drawing.line(1105, 128, 1105, 249, stroke=BLOCKED, stroke_width=1.5, dash=dash)
    drawing.arrow(1105, 249, 1090, 249, stroke=BLOCKED, dash=dash)

    for stage, (x, y) in positions.items():
        _stage_box(drawing, x, y, box_w, box_h, by_stage[stage])
    drawing.text(
        596,
        297,
        "No selector, calibration, other-direction, validation, sealed-test, or runtime claim",
        size=8.2,
        fill=MUTED,
    )
    drawing.rect(28, 334, 12, 12, fill=TEAL)
    drawing.text(46, 344, "completed", size=8)
    drawing.rect(120, 334, 12, 12, fill=CORAL)
    drawing.text(138, 344, "failed", size=8)
    drawing.rect(198, 334, 12, 12, fill=BLOCKED)
    drawing.text(216, 344, "blocked / locked", size=8)
    return drawing


def _source_entry(path: Path, raw: bytes, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
    }


def _verify_evidence_artifact(manifest: Mapping[str, Any], path: Path, raw: bytes) -> None:
    relative = path.relative_to(ROOT).as_posix()
    entry = next(
        (item for item in manifest["artifacts"] if item["path"] == relative),
        None,
    )
    _require(entry is not None, f"evidence manifest does not bind {relative}")
    _require(entry["sha256"] == _sha256(raw), f"evidence hash differs for {relative}")
    _require(entry["size_bytes"] == len(raw), f"evidence size differs for {relative}")


def _figure_outputs(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    drawing: Drawing,
) -> dict[str, bytes]:
    return {
        f"{name}.csv": _csv_bytes(tuple(rows[0]), rows),
        f"{name}.svg": drawing.svg_bytes(),
        f"{name}.pdf": drawing.pdf_bytes(),
    }


def _build() -> dict[str, bytes]:
    candidates, candidates_raw = _load_csv(EVIDENCE_DIR / "method_dev_candidates.v4.csv")
    safe_sets, safe_sets_raw = _load_csv(EVIDENCE_DIR / "method_dev_safe_sets.v4.csv")
    tasks, tasks_raw = _load_csv(EVIDENCE_DIR / "method_dev_tasks.v4.csv")
    failures, failures_raw = _load_csv(EVIDENCE_DIR / "method_dev_failure_overlap.v4.csv")
    evidence_manifest, evidence_manifest_raw = _load_json(
        EVIDENCE_DIR / "method_dev_evidence_manifest.v4.json"
    )
    archive_path = EVIDENCE_DIR / "method_dev_report.v4.json.gz"
    _reject_sealed_path(archive_path)
    archive_raw = archive_path.read_bytes()
    v2, v2_raw = _load_json(V2_DIAGNOSTIC)
    v3, v3_raw = _load_json(V3_DIAGNOSTIC)
    v4, v4_raw = _load_json(V4_DIAGNOSTIC)
    failed, failed_raw = _load_json(FAILED_RECEIPT)
    initialization, initialization_raw = _load_json(INITIALIZATION)

    for path, raw in (
        (EVIDENCE_DIR / "method_dev_candidates.v4.csv", candidates_raw),
        (EVIDENCE_DIR / "method_dev_safe_sets.v4.csv", safe_sets_raw),
        (EVIDENCE_DIR / "method_dev_tasks.v4.csv", tasks_raw),
        (EVIDENCE_DIR / "method_dev_failure_overlap.v4.csv", failures_raw),
        (archive_path, archive_raw),
    ):
        _verify_evidence_artifact(evidence_manifest, path, raw)

    _require(evidence_manifest["registered_deployment"]["gate_passed"] is False, "gate changed")
    _require(v4["conclusion"]["gate_disposition"].startswith("v4_rejected"), "v4 status changed")
    _require(v2["conclusion"]["rank_or_seed_tuning_is_insufficient"] is True, "v2 changed")
    _require(v3["conclusion"]["rank_or_seed_tuning_is_insufficient"] is True, "v3 changed")

    candidate_rows = _candidate_data(candidates, safe_sets)
    teacher_rows = _teacher_alignment_data(archive_raw, evidence_manifest, v3, v4)
    task_rows = _task_data(tasks)
    failure_rows = _failure_data(failures)
    progression_rows = _progression_data(v2, v3, v4, evidence_manifest)
    pipeline_rows = _pipeline_data(failed, initialization)

    outputs: dict[str, bytes] = {}
    outputs.update(
        _figure_outputs(FIGURES[0], candidate_rows, _figure_candidate_coverage(candidate_rows))
    )
    outputs.update(
        _figure_outputs(FIGURES[1], teacher_rows, _figure_teacher_alignment(teacher_rows))
    )
    outputs.update(_figure_outputs(FIGURES[2], task_rows, _figure_task_heterogeneity(task_rows)))
    outputs.update(_figure_outputs(FIGURES[3], failure_rows, _figure_failure_overlap(failure_rows)))
    outputs.update(
        _figure_outputs(FIGURES[4], progression_rows, _figure_progression(progression_rows))
    )
    outputs.update(_figure_outputs(FIGURES[5], pipeline_rows, _figure_pipeline_stop(pipeline_rows)))

    sources = [
        _source_entry(
            EVIDENCE_DIR / "method_dev_candidates.v4.csv",
            candidates_raw,
            "candidate_metrics",
        ),
        _source_entry(
            EVIDENCE_DIR / "method_dev_safe_sets.v4.csv", safe_sets_raw, "safe_set_metrics"
        ),
        _source_entry(EVIDENCE_DIR / "method_dev_tasks.v4.csv", tasks_raw, "task_metrics"),
        _source_entry(
            EVIDENCE_DIR / "method_dev_failure_overlap.v4.csv",
            failures_raw,
            "failure_partition",
        ),
        _source_entry(
            EVIDENCE_DIR / "method_dev_evidence_manifest.v4.json",
            evidence_manifest_raw,
            "evidence_manifest",
        ),
        _source_entry(archive_path, archive_raw, "compressed_method_dev_report"),
        _source_entry(V2_DIAGNOSTIC, v2_raw, "v2_diagnostic"),
        _source_entry(V3_DIAGNOSTIC, v3_raw, "v3_diagnostic"),
        _source_entry(V4_DIAGNOSTIC, v4_raw, "v4_diagnostic"),
        _source_entry(FAILED_RECEIPT, failed_raw, "failed_stage_receipt"),
        _source_entry(INITIALIZATION, initialization_raw, "locked_workspace_receipt"),
    ]
    artifacts = []
    for name, raw in outputs.items():
        suffix = Path(name).suffix
        media_type = {
            ".csv": "text/csv",
            ".svg": "image/svg+xml",
            ".pdf": "application/pdf",
        }[suffix]
        artifacts.append(
            {
                "path": (OUTPUT_DIR / name).relative_to(ROOT).as_posix(),
                "sha256": _sha256(raw),
                "size_bytes": len(raw),
                "media_type": media_type,
            }
        )
    manifest = {
        "schema_version": "goldenexperience.publication_v5_figures.v1",
        "authority": "derived_terminal_negative_result_figures_not_runtime_approval",
        "figure_count": len(FIGURES),
        "formats": ["csv", "svg", "pdf"],
        "style": {
            "ink": INK,
            "safe": TEAL,
            "failure": CORAL,
            "warning": AMBER,
            "blocked": BLOCKED,
            "font": FONT_FAMILY,
        },
        "source_objects": sources,
        "artifacts": artifacts,
        "reproduction": {
            "command": "python3 paper/tools/build_figures.py",
            "check_command": "python3 paper/tools/build_figures.py --check",
            "external_plotting_dependencies": [],
            "sealed_payload_access": False,
        },
    }
    outputs["figures_manifest.v4.json"] = _canonical_json(manifest)
    return outputs


def _write_or_check(outputs: Mapping[str, bytes], *, check: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for name, expected in outputs.items():
        path = OUTPUT_DIR / name
        _reject_sealed_path(path)
        if check:
            if not path.is_file():
                failures.append(f"missing {path.relative_to(ROOT)}")
                continue
            if path.read_bytes() != expected:
                failures.append(f"content differs for {path.relative_to(ROOT)}")
        else:
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_bytes(expected)
            os.replace(temporary, path)
        print(f"{_sha256(expected)}  {path.relative_to(ROOT)}")
    if failures:
        raise SystemExit("\n".join(failures))


def main() -> None:
    args = _parse_args()
    _write_or_check(_build(), check=args.check)


if __name__ == "__main__":
    main()
