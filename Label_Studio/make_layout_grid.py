#!/usr/bin/env python
"""
make_layout_prompt.py
---------------------
python make_layout_prompt.py poster.png > prompt.txt
"""
import argparse, numpy as np, cv2, pathlib, sys
from PIL import Image

LABELS = [
    "title_heat",
    "location_heat",
    "time_heat",
    "host_organization_heat",
    "call-to-action_purpose_heat",
    "text_descriptions_details_heat",
]

HX, HY   = 12, 21            # grid size
THRESH   = 240               # >240 treated as white background
MIN_CELL = 0.01              # small residual weight inside occupied cells

def build_occupancy(img):
    """Return HY×HX grid: 1 = free, 0 = occupied by imagery."""
    W, H = img.size
    rgb  = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray < THRESH               # True where content exists
    occ  = np.zeros((HY, HX), float)

    ys, xs = np.where(mask)
    if ys.size == 0:                    # blank poster
        occ[:] = 1.0
        return occ

    x0,x1 = xs.min(), xs.max()
    y0,y1 = ys.min(), ys.max()

    gx0, gx1 = int(x0 / W * HX), int(x1 / W * HX)
    gy0, gy1 = int(y0 / H * HY), int(y1 / H * HY)
    occ[:] = 1.0
    occ[gy0:gy1+1, gx0:gx1+1] = MIN_CELL
    return occ

def grid_to_line(tag, grid):
    flat = " ".join(f"{v:.1f}" for v in grid.flatten())
    return f"{tag} {flat}"

def main(path):
    img = Image.open(path)
    base = build_occupancy(img)        # HY×HX free-space grid

    lines = [grid_to_line(tag, base) for tag in LABELS]

    print("<LAYOUT_HEAT>")
    print("FRAME_PCT 100 100")
    print("\n".join(lines))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="PNG/JPG poster frame")
    args = ap.parse_args()
    main(pathlib.Path(args.image))
