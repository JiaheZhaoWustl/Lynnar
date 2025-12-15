# viz_poster.py
# -------------
# Visualise one poster with its Label-Studio bounding boxes.
# Optionally shows the 12×21 percentage heat-maps too.

import argparse, json, numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter

# --------------------- CONFIG --------------------
LABELS = [
    "Title",
    "Location",
    "Time",
    "Host/organization",
    "Call-To-Action/Purpose",
    "Text descriptions/details",
]
COLOURS = {                         # tweak colour palette as you like
    "Title": "#e41a1c",
    "Location": "#4daf4a",
    "Time": "#377eb8",
    "Host/organization": "#984ea3",
    "Call-To-Action/Purpose": "#ff7f00",
    "Text descriptions/details": "#00bbbb",
}
GRID = (12, 21)                    # heat-map resolution
SIGMA = 1.0                        # blur radius in grid cells
SHOW_HEATMAPS = True              # set True to see grids on the right
# -------------------------------------------------

def load_rects(anno_path):
    task = json.load(open(anno_path, encoding="utf-8"))
    # Auto-detect where the rectangles live
    if "result" in task:
        results = task["result"]
    elif "annotation" in task and "result" in task["annotation"]:
        results = task["annotation"]["result"]
    else:
        results = task["annotations"][0]["result"]

    rects = {lab: [] for lab in LABELS}
    for r in results:
        lab = r["value"]["rectanglelabels"][0]
        if lab not in rects:
            continue
        v = r["value"]
        rects[lab].append((v["x"], v["y"], v["width"], v["height"]))  # % space
    return rects

def draw_boxes(ax, rects_pct, W, H):
    for lab, boxes in rects_pct.items():
        for x, y, w, h in boxes:
            rect = patches.Rectangle(
                (x / 100 * W, y / 100 * H),
                w / 100 * W,
                h / 100 * H,
                linewidth=1.5,
                edgecolor=COLOURS[lab],
                facecolor="none",
                alpha=0.9,
            )
            ax.add_patch(rect)
    # legend
    handles = [
        patches.Patch(edgecolor=COLOURS[l], facecolor="none", label=l, linewidth=2)
        for l in LABELS
    ]
    ax.legend(handles=handles, fontsize=6, loc="upper left", frameon=False)

def rects_to_heat(rects_pct, hx, hy):
    g = np.zeros((hy, hx))
    for x, y, w, h in rects_pct:
        x1, y1 = x + w, y + h
        gx0, gx1 = int(x / (100 / hx)), int(x1 / (100 / hx))
        gy0, gy1 = int(y / (100 / hy)), int(y1 / (100 / hy))
        g[gy0 : gy1 + 1, gx0 : gx1 + 1] = 1
    g = gaussian_filter(g, SIGMA)
    if g.max() > 0:
        g /= g.max()
    return g

def main(img_path, anno_path):
    img = Image.open(img_path)
    W, H = img.size
    rects = load_rects(anno_path)

    if SHOW_HEATMAPS:
        fig = plt.figure(figsize=(10, 6), dpi=120)
        gs  = fig.add_gridspec(2, 4, width_ratios=[3, 1, 1, 1])
        ax_img = fig.add_subplot(gs[:, 0])
    else:
        fig, ax_img = plt.subplots(figsize=(4, 6), dpi=120)

    # --- poster with boxes ---
    ax_img.imshow(img)
    ax_img.set_title("Poster with Bounding Boxes", fontsize=9)
    draw_boxes(ax_img, rects, W, H)
    ax_img.axis("off")

    # --- optional heat-maps ---
    if SHOW_HEATMAPS:
        hx, hy = GRID
        for idx, lab in enumerate(LABELS):
            g = rects_to_heat(rects[lab], hx, hy)
            row, col = divmod(idx, 3)
            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(g, origin="upper", aspect="auto")
            ax.set_title(lab, fontsize=7)
            ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img",  required=True, help="Poster image file (PNG/JPG)")
    ap.add_argument("--anno", required=True, help="Matching Label-Studio JSON")
    args = ap.parse_args()
    main(args.img, args.anno)
