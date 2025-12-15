#!/usr/bin/env python
"""
Visualise the 12×21 heat-maps inside a <LAYOUT_HEAT> prompt.
Usage:
    python viz_heat_from_prompt.py poster_prompt.txt
"""
import sys, re, numpy as np, matplotlib.pyplot as plt

LABELS = [
    "title_heat",
    "location_heat",
    "time_heat",
    "host_organization_heat",
    "call-to-action_purpose_heat",
    "text_descriptions_details_heat",
]

HX, HY = 12, 21                # grid size

def read_grids(path):
    grids = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            tag = parts[0].lower()
            if tag not in LABELS:
                continue
            nums = list(map(float, parts[1:]))
            grids[tag] = np.array(nums).reshape(HY, HX)
    return grids

def main(txt):
    grids = read_grids(txt)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes = axes.flatten()
    pretty = {
        "title_heat":"Title", "location_heat":"Location", "time_heat":"Time",
        "host_organization_heat":"Host/org",
        "call-to-action_purpose_heat":"CTA",
        "text_descriptions_details_heat":"Details"
    }
    for ax, tag in zip(axes, LABELS):
        g = grids.get(tag, np.zeros((HY, HX)))
        ax.imshow(g, origin="upper", aspect="auto")
        ax.set_title(pretty[tag], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python viz_heat_from_prompt.py prompt.txt")
        sys.exit(1)
    main(sys.argv[1])
