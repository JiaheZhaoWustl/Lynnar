## `heatmap-plugin-test/` (Figma plugin)

This folder contains the Figma plugin prototype UI + controller code.

- `manifest.json`: plugin config
- `ui.html`: plugin UI
- `code.ts`: main-thread controller logic (compiled to `code.js`)

## Dev workflow

1. Install dependencies:

```bash
npm install
```

2. Compile TypeScript:

```bash
npm run build
```

Or keep it running while you iterate:

```bash
npm run watch
```

3. Start the backend (required for heatmap prediction):

- See `../plugin-backend/README.md`

## Load the plugin in Figma (dev)

Follow the Figma plugin quickstart guide:
`https://www.figma.com/plugin-docs/plugin-quickstart-guide/`

When selecting the plugin folder, choose this directory: `heatmap-plugin-test/`.

