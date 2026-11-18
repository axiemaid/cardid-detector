# cardid-detector

Card identity detector using **perceptual hashing** — not OCR.

Live demo: deployed on Vercel (static site, no backend).

## Why perceptual hashing?

Previous approach (cardimg-identifier) used OCR (Tesseract.js) to read card codes. It was fragile:
- Manga halftone dots blended into text
- Required hand-patched confusion tables (`Cle→null`, `SU0→S06`)
- Any new art style could break it

Perceptual hashing is dumb and robust: normalize both images to the same size, compute a DCT-based hash, compare Hamming distance. No text reading needed.

## Pipeline

1. **Edge detect** (Sobel operator) → find card boundaries
2. **Crop** → just the card, no sleeve/background
3. **Resize** → standard 343×480 (2.5:3.5 card ratio)
4. **Hash** → DCT-based pHash (64-bit)
5. **Match** → Hamming distance + color histogram + SSIM

## Scoring

- **Score** = 60% pHash (Hamming) + 25% color histogram + 15% SSIM
- **Confident match:** Hamming ≤6 AND score margin ≥0.05 over second best
- **Likely match:** Hamming ≤12 AND histogram >0.9
- **No match:** Hamming >18 OR score <0.75 OR insufficient margin

Since only 16 of 42 cards are in the DB, "no confident match" may simply mean the card hasn't been added yet.

## Local Development

### Add a reference image

```bash
# Manually cropped scan (preferred — you control quality)
node scripts/add-reference.cjs /path/to/scan.png C01
```

### Build the reference database

```bash
# Precompute hashes, histograms, and SSIM thumbnails into public/refs.json
node scripts/build-refs.cjs
```

### Match a card (CLI)

```bash
node scripts/match.cjs /path/to/unknown.jpg
node scripts/match.cjs /path/to/unknown.jpg --json
node scripts/match.cjs /path/to/unknown.jpg --top 10
```

### List references

```bash
node scripts/list-references.cjs
```

## Deployment (Vercel)

This is a fully static site. No server component.

1. Add a reference scan → `node scripts/add-reference.cjs scan.png C01`
2. Rebuild the DB → `node scripts/build-refs.cjs`
3. Commit and push → Vercel auto-deploys

The `vercel.json` config:
- **Build command:** `node scripts/build-refs.cjs`
- **Output directory:** `public/`
- **Framework:** None (static files)

All image processing (crop, hash, match) runs client-side in a Web Worker. The browser loads `refs.json` (precomputed hashes + histograms) but never processes reference images at runtime.

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
  lib/
    imageio.js         # Isomorphic image I/O (sharp in Node, Canvas in browser)
    crop.js            # OpenCV card boundary detection
    phash.js           # DCT-based perceptual hashing (pure math)
    match.js           # Scoring + matching orchestration
  scripts/
    match.cjs           # CLI: match unknown image
    add-reference.cjs   # CLI: add verified reference
    list-references.cjs # CLI: list all references
    build-refs.cjs      # CLI: precompute public/refs.json
  public/
    index.html          # Static frontend
    app.js              # UI logic
    worker.js           # Web Worker (crop + hash + match)
    refs.json           # Precomputed reference DB (generated)
  references/           # Reference card images (C01.png, etc.)
  test-images/          # Test images for development
  vercel.json           # Vercel static deployment config
```

## Integration

Once standalone matching is proven, integrate into cardimg explorer uploader as a pre-upload validation gate: "Is this a known Hyper Battle card? Yes → allow. No → reject."
