#!/usr/bin/env python
# viz_image_deco_heat.py  —  aggregate "Image" / "Decoration" polygons to a 12×21 heat-map
# -------------------------------------------------------------------------------
import json, glob, pathlib, argparse, numpy as np, matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

# ── config ─────────────────────────────────────────────────────────────
FOLDER = r"E:\SIA_works\PosterDatabase\Label_Studio\annotations_split_3"
LABELS = {"image", "decoration"}          # lower-case match
HX, HY = 12, 21
SIGMA  = 1.0                              # blur in grid cells
OUT_PNG   = "validation_heat.png"
OUT_TXT   = "validation_heat.txt"
# ----------------------------------------------------------------------

def raster_polygons(w, h, polys):
    """polys: list of list[(x%,y%)]  →  boolean mask (H,W)"""
    mask = Image.new("1", (w, h), 0)
    d    = ImageDraw.Draw(mask)
    for poly in polys:
        # convert % to px
        pts = [(x/100*w, y/100*h) for x,y in poly]
        d.polygon(pts, fill=1)
    return np.array(mask, dtype="float32")

def downsample(arr, hx, hy):
    H, W = arr.shape
    g = np.zeros((hy, hx))
    for gy in range(hy):
        for gx in range(hx):
            y0,y1 = int(gy*H/hy), int((gy+1)*H/hy)
            x0,x1 = int(gx*W/hx), int((gx+1)*W/hx)
            g[gy, gx] = arr[y0:y1, x0:x1].mean()
    return g

def process_file(jf):
    js = json.load(open(jf, encoding="utf-8"))
    # dive to results (single-task, nested, etc.)
    results = (js.get("result") or
               js.get("annotation", {}).get("result") or
               js.get("annotations", [{}])[0].get("result", []))
    polys = []
    origW = origH = None
    for r in results:
        lab = r["value"].get("polygonlabels",[""])[0].lower()
        if lab not in LABELS: continue
        pts = r["value"]["points"]          # list of [x%,y%]
        polys.append([(x,y) for x,y in pts])
        # grab size once
        origW = r["original_width"]
        origH = r["original_height"]
    if not polys: return None
    mask = raster_polygons(origW, origH, polys)
    return downsample(mask, HX, HY)

def main():
    acc = np.zeros((HY, HX))
    posters = 0
    for jf in glob.glob(str(pathlib.Path(FOLDER) / "*.json")):
        g = process_file(jf)
        if g is not None:
            acc += g
            posters += 1
    if posters == 0:
        print("No Image/Decoration polygons found"); return
    heat = gaussian_filter(acc / posters, sigma=SIGMA)
    if heat.max() > 0: heat /= heat.max()

    # save visual
    plt.figure(figsize=(3,5), dpi=150)
    plt.imshow(heat, origin='upper', cmap='hot', vmin=0, vmax=1)
    plt.title(f"Image / Decoration heat ({posters} posters)")
    plt.xticks([]); plt.yticks([])
    plt.colorbar(fraction=0.045, pad=0.03)
    plt.tight_layout(); plt.savefig(OUT_PNG, bbox_inches='tight')
    print(f"★ Heat-map PNG → {OUT_PNG}")

    # write 252-float line
    flat = " ".join(f"{v:.1f}" for v in heat.flatten())
    pathlib.Path(OUT_TXT).write_text(flat, encoding="utf-8")
    print(f"★ occ_heat line (252 floats) → {OUT_TXT}")

if __name__ == "__main__":
    main()
