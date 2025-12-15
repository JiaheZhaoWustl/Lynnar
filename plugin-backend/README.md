# `plugin-backend/` (Flask API for the Figma plugin)

This folder hosts a small Flask server used by the Figma plugin to:

- **Predict heatmaps** from a selected Figma frame (`POST /predict`)
- **Chat / prompt refinement** for the “Prompt Refinement” UI (`POST /chat`, `POST /chat/refine`)
- **(Optional) Generate an image** from a prompt (`POST /generate-image`)

## Setup

1. Create a virtual environment (recommended) and install deps:

```bash
pip install -r requirements.txt
```

2. Create a `.env` at repo root (or in this folder) with:

```bash
OPENAI_API_KEY=...
```

Tip: you can copy `../env.example` to `.env`.

3. Run the server:

```bash
python server.py
```

The server listens on `http://localhost:5000`.

## Notes

- The Figma plugin calls `http://localhost:5000/predict` from `heatmap-plugin-test/code.ts`.
- `primer.jsonl` is required for the layout/heatmap prediction pipeline.
- For more chat-specific details, see `README_CHAT.md`.


