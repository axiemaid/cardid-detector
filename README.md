# cardid-detector

Card identity detector using **template matching** — not OCR.

## Why template matching?

Previous approach (cardimg-identifier) used OCR (Tesseract.js) to read card codes. It was fragile:
- Manga halftone dots blended into text
- Required hand-patched confusion tables (`Cle→null`, `SU0→S06`)
- Any new art style could break it

Template matching is dumb and robust: normalize both images to the same size, compare pixels, get a similarity score. No text reading needed.

## Pipeline

1. **Edge detect** (Sobel operator) → find card boundaries
2. **Crop** → just the card, no sleeve/background
3. **Resize** → standard 343×480 (2.5:3.5 card ratio)
4. **Compare** → MSE + SSIM against reference images
5. **Match** → highest similarity above threshold wins

## Usage

### Add a reference image

```bash
# Manually cropped scan (preferred — you control quality)
node scripts/add-reference.cjs /path/to/scan.png C01

# Auto edge-crop (let the detector find card bounds)
node scripts/add-reference.cjs /path/to/photo.jpg C01 --auto-crop
```

### List references

```bash
node scripts/list-references.cjs
```

### Match a card

```bash
# Pretty output
node scripts/match.cjs /path/to/unknown.jpg

# JSON output
node scripts/match.cjs /path/to/unknown.jpg --json

# Custom reference directory
node scripts/match.cjs /path/to/unknown.jpg --references /custom/refs

# Show top 10 matches
node scripts/match.cjs /path/to/unknown.jpg --top 10
```

### Scoring

- **Score** = 30% MSE-based + 70% SSIM (structural similarity)
- **Confident match:** score > 0.8 AND margin > 0.05 over second best
- **Possible match:** score > 0.6 (needs better reference or manual check)
- **No match:** score < 0.6

## Reference images

Quality of references = quality of matching. Rules:
- One image per card code (C01–C34, S01–S08)
- Use real scans or high-quality photos
- Manually crop to just the card (no sleeves, no background)
- Saved as PNG in `references/` directory
- Named by code: `C01.png`, `C02.png`, etc.

## Structure

```
cardid-detector/
  lib/match.js          # Core matching engine
  scripts/
    match.cjs           # CLI: match unknown image
    add-reference.cjs   # CLI: add verified reference
    list-references.cjs # CLI: list all references
  references/           # Reference card images (C01.png, etc.)
  docs/                 # Documentation
```

## Integration (later)

Once standalone matching is proven, integrate into cardimg explorer uploader as a pre-upload validation gate: "Is this a known Hyper Battle card? Yes → allow. No → reject."
