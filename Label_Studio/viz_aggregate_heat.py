#!/usr/bin/env python
"""
viz_aggregate_heat.py
---------------------
Read a fine-tune JSONL where each line contains heatmaps (flattened 12×21 grids)
and visualize the **average** heatmap for each category.

This is a quick dataset-inspection tool; it does not affect training or the plugin.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Key = tag as it appears in JSONL (lowercase)
# Value = display label
CANONICAL_MAP = {
    "title_heat": "Title",
    "location_heat": "Location",
    "time_heat": "Time",
    "host_organization_heat": "Host/organization",
    "call-to-action_purpose_heat": "Call-To-Action/Purpose",
    "text_descriptions_details_heat": "Text descriptions/details",
}

LABELS_ORDER = [
    "Title",
    "Location",
    "Time",
    "Host/organization",
    "Call-To-Action/Purpose",
    "Text descriptions/details",
]


def parse_user_block(user_content: str) -> Dict[str, np.ndarray]:
    """Parse the user content into {label: flattened_vector} for known tags."""
    out: Dict[str, np.ndarray] = {}
    for part in user_content.splitlines():
        bits = part.strip().split()
        if len(bits) < 2:
            continue
        tag = bits[0].lower()
        if tag not in CANONICAL_MAP:
            continue
        out[CANONICAL_MAP[tag]] = np.asarray(list(map(float, bits[1:])), dtype=float)
    return out


def main(jsonl_path: str, hx: int, hy: int, output: str | None, no_show: bool, transparent: bool) -> None:
    sums = {lbl: np.zeros((hy, hx), dtype=float) for lbl in set(CANONICAL_MAP.values())}
    doc_count = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            user = next(m["content"] for m in row["messages"] if m["role"] == "user")
            parsed = parse_user_block(user)
            for lbl, vec in parsed.items():
                if vec.size != hx * hy:
                    continue
                sums[lbl] += vec.reshape(hy, hx)
            doc_count += 1

    for k in sums:
        sums[k] /= max(doc_count, 1)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    if transparent:
        fig.patch.set_facecolor("none")
    axes = axes.flatten()

    for ax, lbl in zip(axes, LABELS_ORDER):
        if transparent:
            ax.set_facecolor("none")
        ax.imshow(sums[lbl], origin="upper", aspect="auto", cmap="plasma")
        ax.axis("off")

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        plt.savefig(output, transparent=transparent, bbox_inches="tight", pad_inches=0, dpi=150)

    if no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        default=os.path.join(SCRIPT_DIR, "layout_heat.jsonl"),
        help="Path to layout_heat.jsonl (default: Label_Studio/layout_heat.jsonl)",
    )
    parser.add_argument("--hx", type=int, default=12, help="grid columns (default 12)")
    parser.add_argument("--hy", type=int, default=21, help="grid rows (default 21)")
    parser.add_argument("--output", default=None, help="Optional output PNG path")
    parser.add_argument("--no-show", action="store_true", default=False)
    parser.add_argument("--transparent", action="store_true", default=True)
    args = parser.parse_args()
    main(args.jsonl, args.hx, args.hy, args.output, args.no_show, args.transparent)
