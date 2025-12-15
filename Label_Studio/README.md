# `Label_Studio/` (dataset + visualization tooling)

This folder contains scripts for turning Label Studio bounding-box annotations into **category heatmaps** (including the 12×21 grid format used by the plugin workflow), plus visualization utilities.

## Setup

```bash
pip install -r requirements.txt
```

## Common entrypoints

- `viz_aggregate_heat.py`: render a 6-category heatmap grid (12×21 per category)
- `separated_bounding_boxes/HeatMapVisualization.py`: per-category heatmap visualization
- `separated_bounding_boxes/AllInOneVisualization.py`: combined heatmap visualization

## Notes

- Large annotation exports and generated images are ignored via `.gitignore`.


