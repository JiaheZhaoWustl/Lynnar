#!/usr/bin/env python
# build_heat_dataset.py
# ---------------------
# Batch-convert Label-Studio rectangle annotations into portrait 12×21
# percentage heat-maps for GPT fine-tuning.

import argparse, json, glob, pathlib, numpy as np
from collections import defaultdict
from scipy.ndimage import gaussian_filter

# ---- configuration ----
LABELS = [
    "Title",
    "Location",
    "Time",
    "Host/organization",
    "Call-To-Action/Purpose",
    "Text descriptions/details",
]

# ---- helpers ----
def rects_to_grid(rects, hx, hy, sigma):
    """List of (x0,y0,x1,y1)% → blurred hx×hy grid, origin upper-left."""
    g = np.zeros((hy, hx), float)
    for x0, y0, x1, y1 in rects:
        gx0, gx1 = int(x0 / (100 / hx)), int(x1 / (100 / hx))
        gy0, gy1 = int(y0 / (100 / hy)), int(y1 / (100 / hy))
        g[gy0 : gy1 + 1, gx0 : gx1 + 1] = 1  # binary vote
    if sigma:
        g = gaussian_filter(g, sigma=sigma)
    if g.max() > 0:
        g /= g.max()
    return np.round(g.flatten(), 1)  # 1-dec quantisation


def get_result_list(task):
    """
    Return the list of rectangle objects regardless of LS export flavour.
    """
    if "result" in task:
        return task["result"]                        # bulk export (flat)
    if "annotation" in task and "result" in task["annotation"]:
        return task["annotation"]["result"]          # per-file export
    if "annotations" in task and task["annotations"]:
        return task["annotations"][0]["result"]      # bulk nested
    raise KeyError("Could not find 'result' list in task JSON.")


def poster_to_row(task, hx, hy, sigma):
    """Single LS task → JSONL dict with 6 heat-maps."""
    label_boxes = defaultdict(list)
    results = get_result_list(task)

    for r in results:
        lab = r["value"]["rectanglelabels"][0]
        if lab not in LABELS:
            continue
        v = r["value"]
        # already percentages
        box = (v["x"], v["y"], v["x"] + v["width"], v["y"] + v["height"])
        label_boxes[lab].append(box)

    lines = []
    for lab in LABELS:
        heat = rects_to_grid(label_boxes.get(lab, []), hx, hy, sigma)
        lines.append(
            f"{lab.lower().replace('/','_')}_heat "
            + " ".join(f"{v:.1f}" for v in heat)
        )

    user_msg = "FRAME_PCT 100 100\n" + "\n".join(lines)
    return {
        "messages": [
            {"role": "system", "content": "<LAYOUT_HEAT> Predict bounding boxes."},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": ""},  # blank target
        ]
    }


def walk_tasks(src_path, bulk):
    p = pathlib.Path(src_path)
    if bulk:  # single bulk JSON file
        data = json.load(open(p, encoding="utf-8"))
        for task in data:
            yield task
    else:  # directory of JSON files
        for f in glob.glob(str(p / "*.json")):
            yield json.load(open(f, encoding="utf-8"))


# ---- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src", required=True, help="Dir with per-poster JSONs OR bulk JSON file"
    )
    ap.add_argument("--dst", required=True, help="Output JSONL file")
    ap.add_argument(
        "--grid",
        nargs=2,
        type=int,
        default=[12, 21],
        metavar=("HX", "HY"),
        help="Grid columns rows (default 12 21)",
    )
    ap.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian blur radius in grid cells (default 1.0)",
    )
    ap.add_argument(
        "--bulk",
        action="store_true",
        help="Treat --src as a single bulk-export JSON",
    )
    args = ap.parse_args()

    hx, hy = args.grid
    rows = 0
    with open(args.dst, "w", encoding="utf-8") as out:
        for task in walk_tasks(args.src, args.bulk):
            row_dict = poster_to_row(task, hx, hy, args.sigma)
            out.write(json.dumps(row_dict, ensure_ascii=False) + "\n")
            rows += 1
    print(f"Wrote {rows} posters → {args.dst}")


if __name__ == "__main__":
    main()
