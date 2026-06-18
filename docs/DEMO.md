# 30–60 s demo video (portfolio)

Record a short screen capture showing:

1. **Band landscape** (`docs/screenshots/band_landscape.png` or Streamlit live)
2. **Terminal** with `train` metrics (accuracy, R²)
3. **Audio** — play `outputs/sub01_portfolio.wav` (QuickTime / VLC)

## Suggested script (~45 s)

- 0–10 s: “EEG from OpenNeuro ds004148 — eyes closed, eyes open, cognitive task segments”
- 10–25 s: show the color band landscape changing across task segments
- 25–35 s: play the WAV — tempo shifts with band profile
- 35–45 s: confusion matrix screenshot + “task classifier ~61% hold-out accuracy”

## Tools (macOS)

```bash
# QuickTime → File → New Screen Recording
# or regenerate the bundled demo:
python scripts/build_demo_video.py
```

Upload to YouTube (unlisted) or Loom and link in README — or use the bundled file:

```markdown
## Demo video
[Watch 45s demo](docs/demo.mp4) · regenerate: `python scripts/build_demo_video.py`
```
