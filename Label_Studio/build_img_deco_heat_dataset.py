#!/usr/bin/env python
# build_img_deco_heat_dataset.py  –  write image + decoration grids to JSONL
# ------------------------------------------------------------------------
import argparse, json, glob, pathlib, numpy as np
from scipy.ndimage import gaussian_filter

# editable defaults -------------------------------------------------------
SRC_DIR = r"E:\SIA_works\PosterDatabase\Label_Studio\annotations_split_3"
DST_FILE = "validation_heat.jsonl"
HX, HY   = 12, 21        # grid resolution   (override with --grid 15 26)
SIGMA    = 1.0           # blur in grid cells
# -------------------------------------------------------------------------

LABELS = {"image": "image_heat",
          "decoration": "decoration_heat"}      # LS label → tag in JSONL

# ── helpers ───────────────────────────────────────────────────────────────
def rects_to_grid(rects):
    """list[(x%,y%,w%,h%)] → HY×HX grid (0–1)"""
    g = np.zeros((HY, HX))
    for x,y,w,h in rects:
        gx0 = int(x        / (100/HX))
        gy0 = int(y        / (100/HY))
        gx1 = int((x+w)    / (100/HX))
        gy1 = int((y+h)    / (100/HY))
        g[gy0:gy1+1, gx0:gx1+1] = 1
    if SIGMA:
        g = gaussian_filter(g, SIGMA)
    if g.max()>0:
        g /= g.max()
    return g

def extract_boxes(task):
    """Return dict label→list[(x%,y%,w%,h%)]"""
    res = (task.get("annotation",{}).get("result") or
           task.get("result") or
           task.get("annotations",[{}])[0].get("result", []))

    buckets = {lab: [] for lab in LABELS}
    for r in res:
        # rectangles
        if "rectanglelabels" in r["value"]:
            lab = r["value"]["rectanglelabels"][0].lower()
            if lab in buckets:
                v=r["value"]; buckets[lab].append((v["x"],v["y"],v["width"],v["height"]))
        # polygons
        elif "polygonlabels" in r["value"]:
            lab = r["value"]["polygonlabels"][0].lower()
            if lab in buckets:
                pts = r["value"]["points"]
                xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
                x0,y0=min(xs),min(ys)
                buckets[lab].append((x0,y0,max(xs)-x0,max(ys)-y0))
    return buckets

# ── main writer ───────────────────────────────────────────────────────────
def main(src, dst):
    rows = 0
    with open(dst, "w", encoding="utf-8") as out:
        for jf in glob.glob(str(pathlib.Path(src) / "*.json")):
            task   = json.load(open(jf, encoding="utf-8"))
            buckets = extract_boxes(task)

            user_lines = ["FRAME_PCT 100 100"]
            for lab, tag in LABELS.items():
                grid = rects_to_grid(buckets[lab])
                flat = " ".join(f"{v:.1f}" for v in grid.flatten())
                user_lines.append(f"{tag} {flat}")

            out.write(json.dumps({
                "messages":[
                  {"role":"system",
                   "content":"<IMAGE_HEAT> Predict image & decoration layout."},
                  {"role":"user","content":"\n".join(user_lines)},
                  {"role":"assistant","content":""}
                ]}, ensure_ascii=False)+"\n")
            rows += 1
    print(f"Wrote {rows} posters → {dst}")

# ── CLI entry ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DIR,
                    help="Folder with per-poster JSONs")
    ap.add_argument("--dst", default=DST_FILE,
                    help="Output JSONL")
    ap.add_argument("--grid", nargs=2, type=int, metavar=("HX","HY"),
                    help="Grid cols rows (default 12 21)")
    ap.add_argument("--sigma", type=float, help="Blur sigma (default 1.0)")
    args = ap.parse_args()

    if args.grid: HX, HY = args.grid
    if args.sigma is not None: globals()["SIGMA"] = args.sigma
    main(args.src, args.dst)
