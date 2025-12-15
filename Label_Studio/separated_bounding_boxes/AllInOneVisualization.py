"""
AllInOneVisualization.py
-----------------------
Render a combined heatmap from multiple category bounding-box JSON files.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CATEGORY_FILES = [
    "Title.json",
    "Location.json",
    "Time.json",
    "Host_organization.json",
    "Call-To-Action_Purpose.json",
    "Text descriptions_details.json",
]


def load_boxes(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_boxes_to_heatmap(
    heatmap: np.ndarray,
    boxes: List[Dict[str, Any]],
    grid_width: int,
    grid_height: int,
) -> None:
    for box in boxes:
        x0 = int(float(box["x"]) / 100.0 * grid_width)
        y0 = int(float(box["y"]) / 100.0 * grid_height)
        x1 = int((float(box["x"]) + float(box["width"])) / 100.0 * grid_width)
        y1 = int((float(box["y"]) + float(box["height"])) / 100.0 * grid_height)

        x0, x1 = max(0, min(x0, grid_width - 1)), max(0, min(x1, grid_width))
        y0, y1 = max(0, min(y0, grid_height - 1)), max(0, min(y1, grid_height))

        heatmap[y0:y1, x0:x1] += 1.0


def render_heatmap(
    heatmap: np.ndarray,
    output_path: str,
    cmap: str,
    transparent: bool,
    no_show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 18))
    if transparent:
        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")

    ax.imshow(heatmap, cmap=cmap, interpolation="bilinear", origin="upper")
    ax.axis("off")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, transparent=transparent, bbox_inches="tight", pad_inches=0, dpi=150)

    if no_show:
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a combined heatmap from multiple category JSON files.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[os.path.join(SCRIPT_DIR, p) for p in DEFAULT_CATEGORY_FILES],
        help="List of category JSON files. Defaults to the 6 standard files in this folder.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(SCRIPT_DIR, "outputs", "combined_heatmap.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--grid-width", type=int, default=1080)
    parser.add_argument("--grid-height", type=int, default=1920)
    parser.add_argument("--sigma", type=float, default=15.0)
    parser.add_argument("--cmap", default="plasma")
    parser.add_argument("--transparent", action="store_true", default=True)
    parser.add_argument("--no-show", action="store_true", default=False)
    args = parser.parse_args()

    heatmap = np.zeros((args.grid_height, args.grid_width), dtype=float)
    for path in args.inputs:
        add_boxes_to_heatmap(heatmap, load_boxes(path), args.grid_width, args.grid_height)

    smoothed = gaussian_filter(heatmap, sigma=args.sigma)
    render_heatmap(smoothed, args.output, args.cmap, args.transparent, args.no_show)


if __name__ == "__main__":
    main()
