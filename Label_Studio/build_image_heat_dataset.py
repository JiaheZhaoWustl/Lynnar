#!/usr/bin/env python
# Build a JSONL file of 12×21 heat-maps for Image/Decoration labels.
# ------------------------------------------------------------------
import argparse, json, glob, pathlib, numpy as np
from scipy.ndimage import gaussian_filter

LABELS = {"image", "decoration"}          # lower-case
HX, HY  = 12, 21                          # grid resolution
SIGMA   = 1.0                             # blur in grid cells

def rects_to_grid(rects):
    """list of (x%,y%,w%,h%) → blurred HY×HX grid"""
    g = np.zeros((HY, HX), float)
    for x, y, w, h in rects:
        gx0 = int(x         / (100/HX))
        gy0 = int(y         / (100/HY))
        gx1 = int((x+w)     / (100/HX))
        gy1 = int((y+h)     / (100/HY))
        g[gy0:gy1+1, gx0:gx1+1] = 1
    if SIGMA:
        g = gaussian_filter(g, sigma=SIGMA)
    if g.max() > 0:
        g /= g.max()
    return g

def extract_rects(task):
    """Return list of (x%,y%,w%,h%) for Image / Decoration."""
    # dig down to the result list
    if "annotation" in task:
        results = task["annotation"]["result"]
    elif "result" in task:
        results = task["result"]
    else:
        results = task.get("annotations", [{}])[0].get("result", [])

    rects = []
    for r in results:
        # 1) Rectangle tasks
        if "rectanglelabels" in r["value"]:
            lab = r["value"]["rectanglelabels"][0].lower()
            if lab in LABELS:
                v = r["value"]
                rects.append((v["x"], v["y"], v["width"], v["height"]))
        # 2) Polygon tasks  ← NEW
        elif "polygonlabels" in r["value"]:
            lab = r["value"]["polygonlabels"][0].lower()
            if lab in LABELS:
                pts = r["value"]["points"]            # list of [x%, y%]
                xs  = [p[0] for p in pts]
                ys  = [p[1] for p in pts]
                x0, y0 = min(xs), min(ys)
                w,  h  = max(xs)-x0, max(ys)-y0
                rects.append((x0, y0, w, h))
    return rects


def main(src_dir, dst_jsonl):
    rows = 0
    with open(dst_jsonl, "w", encoding="utf-8") as out:
        for jf in glob.glob(str(pathlib.Path(src_dir) / "*.json")):
            task  = json.load(open(jf, encoding="utf-8"))
            rects = extract_rects(task)

            grid  = rects_to_grid(rects)
            flat  = " ".join(f"{v:.1f}" for v in grid.flatten())

            # JSONL row – system+user only (assistant left blank)
            out.write(json.dumps({
                "messages":[
                    {"role":"system",
                     "content":"<IMAGE_HEAT> Predict layout of images/decoration."},
                    {"role":"user",
                     "content":f"FRAME_PCT 100 100\nimage_deco_heat {flat}"},
                    {"role":"assistant","content":""}
                ]
            }, ensure_ascii=False)+"\n")
            rows += 1
    print(f"Wrote {rows} posters → {dst_jsonl}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"E:\SIA_works\PosterDatabase\Label_Studio\annotations_split_2",
                    help="Folder of per-poster JSON files")
    ap.add_argument("--dst", default="image_heat.jsonl",
                    help="Output JSONL filename")
    ap.add_argument("--grid", nargs=2, type=int, metavar=("HX","HY"),
                    default=[HX, HY], help="Grid cols rows (default 12 21)")
    args = ap.parse_args()
    HX, HY = args.grid
    main(args.src, args.dst)
