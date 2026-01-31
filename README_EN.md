# Easy PNGTuber

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

A tool for easily creating expression difference images for PNGTubers.

📝 [Usage Guide (Japanese, note.com)](https://note.com/rotejin/n/n106abaaa3957)

[日本語](README.md)

---

## Features

- Select eyes and mouth from different sources, auto-generate 4 patterns
- Split & align from 2x2 expression sheets
- High-precision alignment using AKAZE / ORB feature matching
- Overlay display for comparing differences while drawing masks
- Japanese file path support
- Simple GUI (PySide6)

---

## Quick Start

First, you need to prepare an **expression sheet** (a 2x2 image with expression variations) — see [Step 2](#step-2-create-an-expression-sheet).

### Step 1: Install the Tools

```bash
# Download or clone the repository
git clone https://github.com/rotejin/EasyPNGTuber.git
cd EasyPNGTuber

# Install dependencies (virtual environment created automatically)
uv sync
```

> If [uv](https://docs.astral.sh/uv/) is not installed: `pip install uv` or see [official site](https://docs.astral.sh/uv/getting-started/installation/)

### Step 2: Create an Expression Sheet

1. **Prepare a character image**
   - Get a single image of the character you want to make into a PNGTuber

2. **Create a 2x2 image with Grid Tiler**
   ```bash
   uv run python grid_tiler.py
   ```
   - Drag & drop your image
   - Click "Save Tiled Image" to export the 2x2 image

3. **Generate expressions with AI**
   - Use [Google AI Studio](https://aistudio.google.com/)'s image generation AI (Nano Banana)
   - Upload the 2x2 image and generate expression variations with this prompt:

   <details>
   <summary>Show prompt (click to expand)</summary>

   ```yaml
   expression_sheet:
     task: "edit_reference_image"
     format:
       grid: "2x2"
       preserve: "exact_pixel_position"

     critical_rule:
       source: "use provided reference image as base"
       maintain:
         - original_art_style
         - original_character_design
         - original_face_angle
         - original_head_tilt
         - original_color_palette
         - original_lighting
         - original_line_weight
       do_not_change:
         - head_position
         - face_outline
         - hair
         - background
         - overall_composition

     editable_elements:
       - eyelids_only
       - mouth_only

     parts_definition:
       eyes:
         open: "natural relaxed open state"
         closed: "gentle blink, relaxed eyelids"
       mouth:
         closed: "lips together, neutral"
         open: "natural speaking, showing inside"

     panels:
       top_left:
         action: "keep_unchanged"

       top_right:
         eyes: "open"
         mouth: "open"

       bottom_left:
         eyes: "closed"
         mouth: "closed"

       bottom_right:
         eyes: "closed"
         mouth: "open"
   ```

   </details>

### Step 3: Finalize with Parts Mixer

```bash
uv run python parts_mixer.py
```

Load the generated expression sheet into Parts Mixer, combine eye and mouth parts, and export 4 patterns for your PNGTuber.

---

## Parts Mixer Usage

Select eye and mouth parts from different source images and auto-generate 4 patterns (eyes ON/OFF × mouth ON/OFF).

Useful when AI-generated expression sheets have some parts that didn't turn out as expected.

```bash
uv run python parts_mixer.py
```

1. Drag & drop an expression sheet
2. Click "Split & Align"
3. Select Base image / Eye source / Mouth source (if base has closed eyes & mouth, select open eyes for eye source, open mouth for mouth source)
4. Draw eye region mask on Eye canvas
5. Draw mouth region mask on Mouth canvas
6. 4 patterns appear in preview
7. Click "Save 4 Patterns" to export

### Output Files

- `{original_name}_eyeOFF_mouthOFF.png` - Eyes OFF, Mouth OFF
- `{original_name}_eyeON_mouthOFF.png` - Eyes ON, Mouth OFF
- `{original_name}_eyeOFF_mouthON.png` - Eyes OFF, Mouth ON
- `{original_name}_eyeON_mouthON.png` - Eyes ON, Mouth ON

---

## Sample Images

The `sample/` folder contains sample images for testing.

- `tomari_sample.png` - Sample 2x2 expression sheet

---

## Troubleshooting

### Tool won't start

- Verify Python 3.10+ is installed
- Delete `.venv` folder and re-run `uv sync`

### Image won't load

- Supported formats: PNG, JPG, BMP, WebP
- Japanese file names are supported

### Alignment fails

- May fail if image differences are too large
- Rotations beyond ±30 degrees are not supported

---

## Technical Specifications

### Alignment Algorithm

- **AKAZE feature matching** (primary)
- **ORB** (fallback)
- **RANSAC** for affine transformation estimation

### Constraint Parameters

| Parameter | Value |
|-----------|-------|
| Max rotation | ±30 degrees |
| Scale range | 0.8x ~ 1.2x |
| Success score threshold | 0.6 or higher |

---

## Recommended Environment

| Item | Requirement |
|------|-------------|
| Python | 3.10 or higher |
| OS | Windows / macOS / Linux |
| Package Manager | [uv](https://docs.astral.sh/uv/) recommended |

---

## File Structure

```
EasyPNGTuber/
├── parts_mixer.py        # Main tool: Parts composition
├── grid_tiler.py         # Image tiling
├── mask_composer.py      # Mask composition
├── simple_aligner_app.py # Image alignment
├── aligner.py            # Alignment engine
├── compositor.py         # Image compositing engine
├── cv2_utils.py          # OpenCV utilities
├── mask_canvas.py        # Mask canvas UI
├── preview_widget.py     # Preview UI
├── gemini_prompt.txt     # AI prompt
├── pyproject.toml        # Dependency definitions
└── sample/               # Sample images
```

---

## License

[MIT License](LICENSE)

Copyright (c) 2026 rotejin
