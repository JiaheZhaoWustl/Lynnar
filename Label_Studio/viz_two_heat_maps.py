#!/usr/bin/env python
# viz_two_heat_maps.py
import json, numpy as np, matplotlib.pyplot as plt, collections, pathlib, sys
HX=12; HY=21
acc=collections.defaultdict(lambda: np.zeros((HY,HX)))
rows=0
for line in open(sys.argv[1],encoding="utf-8"):
    msg=next(m["content"] for m in json.loads(line)["messages"] if m["role"]=="user")
    for ln in msg.splitlines():
        parts=ln.split()
        if len(parts)<2 or parts[0]=="FRAME_PCT": continue
        tag=parts[0]; vals=list(map(float,parts[1:]))
        acc[tag]+=np.array(vals).reshape(HY,HX); rows+=1
# average
for k in acc: acc[k]/=rows
fig,axs=plt.subplots(1,len(acc),figsize=(3*len(acc),5),dpi=120)
if len(acc)==1: axs=[axs]
for ax,(tag,g) in zip(axs,acc.items()):
    im=ax.imshow(g,cmap="hot",origin="upper"); ax.set_title(tag); ax.axis("off")
    plt.colorbar(im,ax=ax,fraction=.04,pad=.03)
plt.tight_layout(); plt.savefig("avg_img_deco_heat.png"); print("saved → avg_img_deco_heat.png")
