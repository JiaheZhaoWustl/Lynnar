"""
HeatMapVisualization.py
----------------------
Render a single category heatmap from bounding boxes (Label Studio export format).

Input format: list of boxes like:
  [{"x": <percent>, "y": <percent>, "width": <percent>, "height": <percent>}, ...]
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


def load_boxes(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def boxes_to_heatmap(
    boxes: List[Dict[str, Any]],
    grid_width: int,
    grid_height: int,
    sigma: float,
) -> np.ndarray:
    heatmap = np.zeros((grid_height, grid_width), dtype=float)

    for box in boxes:
        x0 = int(float(box["x"]) / 100.0 * grid_width)
        y0 = int(float(box["y"]) / 100.0 * grid_height)
        x1 = int((float(box["x"]) + float(box["width"])) / 100.0 * grid_width)
        y1 = int((float(box["y"]) + float(box["height"])) / 100.0 * grid_height)

        x0, x1 = max(0, min(x0, grid_width - 1)), max(0, min(x1, grid_width))
        y0, y1 = max(0, min(y0, grid_height - 1)), max(0, min(y1, grid_height))

        heatmap[y0:y1, x0:x1] += 1.0

    return gaussian_filter(heatmap, sigma=sigma)


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
    parser = argparse.ArgumentParser(description="Render a single-category heatmap from bounding boxes.")
    parser.add_argument(
        "--input",
        default=os.path.join(SCRIPT_DIR, "Call-To-Action_Purpose.json"),
        help="Path to a category JSON file (list of boxes). Defaults to Call-To-Action_Purpose.json in this folder.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(SCRIPT_DIR, "outputs", "category_heatmap.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--grid-width", type=int, default=1080)
    parser.add_argument("--grid-height", type=int, default=1920)
    parser.add_argument("--sigma", type=float, default=15.0)
    parser.add_argument("--cmap", default="plasma")
    parser.add_argument("--transparent", action="store_true", default=True)
    parser.add_argument("--no-show", action="store_true", default=False)
    args = parser.parse_args()

    boxes = load_boxes(args.input)
    heat = boxes_to_heatmap(boxes, args.grid_width, args.grid_height, args.sigma)
    render_heatmap(heat, args.output, args.cmap, args.transparent, args.no_show)


if __name__ == "__main__":
    main()
