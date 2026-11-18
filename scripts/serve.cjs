#!/usr/bin/env node
// scripts/serve.cjs — Local web UI for card identification
// Usage: node serve.cjs [--port 3025]

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { matchCard, loadReferences } = require('../lib/match');

const PORT = parseInt(process.argv.find(a => a.startsWith('--port='))?.split('=')[1] || 3025);
const REF_DIR = path.join(__dirname, '..', 'references');

const app = express();
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 } // 50MB
});

app.use(express.static(path.join(__dirname, '..', 'public')));

// Cache references in memory
let refCache = null;
async function getRefs() {
  if (!refCache) {
    refCache = await loadReferences(REF_DIR);
  }
  return refCache;
}

// Reload refs endpoint
app.post('/api/refs/reload', async (req, res) => {
  refCache = null;
  const refs = await getRefs();
  res.json({ count: refs.length });
});

// List refs
app.get('/api/refs', async (req, res) => {
  try {
    const refs = await getRefs();
    res.json({ count: refs.length, codes: refs.map(r => r.code) });
  } catch (e) {
    res.json({ count: 0, codes: [] });
  }
});

// Crop preview endpoint — returns cropped image as JPEG
app.post('/api/crop', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No image provided' });

    const tmpPath = path.join(require('os').tmpdir(), `cardid-crop-${Date.now()}.jpg`);
    fs.writeFileSync(tmpPath, req.file.buffer);

    const { cropToCardFace } = require('../lib/crop');
    const result = await cropToCardFace(tmpPath);

    if (result.detected && result.cropped) {
      const croppedBuf = await result.cropped.jpeg({ quality: 85 }).toBuffer();
      try { fs.unlinkSync(tmpPath); } catch {}
      res.set('Content-Type', 'image/jpeg');
      res.send(croppedBuf);
    } else if (result.reason === 'no_valid_card') {
      try { fs.unlinkSync(tmpPath); } catch {}
      res.status(422).json({ error: 'No card detected. Please retake the photo with the card flat on a dark surface.' });
    } else {
      // No crop needed — return original as JPEG
      const sharp = require('sharp');
      const origBuf = await sharp(tmpPath).jpeg({ quality: 85 }).toBuffer();
      try { fs.unlinkSync(tmpPath); } catch {}
      res.set('Content-Type', 'image/jpeg');
      res.send(origBuf);
    }
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Match endpoint
app.post('/api/match', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No image provided' });

    // Write to temp file for sharp to read
    const tmpPath = path.join(require('os').tmpdir(), `cardid-${Date.now()}.jpg`);
    fs.writeFileSync(tmpPath, req.file.buffer);

    const { normalizeImage, computeHistogramSimilarity, computeSSIM } = require('../lib/match');
    const { hammingDistance } = require('../lib/phash');

    const inputNorm = await normalizeImage(tmpPath);

    // Check if crop failed with retake reason
    if (inputNorm.cropReason === 'no_valid_card') {
      try { fs.unlinkSync(tmpPath); } catch {}
      return res.status(422).json({ error: 'No card detected. Please retake the photo with the card flat on a dark surface.' });
    }

    const refs = await getRefs();

    // Group by code, take best (lowest Hamming) per code
    const byCode = {};
    for (const ref of refs) {
      const hamming = hammingDistance(inputNorm.hash, ref.hash);
      const hist = computeHistogramSimilarity(inputNorm.rgb, ref.rgb);
      const ssim = computeSSIM(inputNorm.gray, ref.gray, inputNorm.width, inputNorm.height);
      const phashScore = Math.max(0, 1 - hamming / 24);
      const score = (phashScore * 0.6) + (hist * 0.25) + (ssim * 0.15);

      const entry = {
        code: ref.code,
        hamming,
        hist: Math.round(hist * 1000) / 1000,
        ssim: Math.round(ssim * 1000) / 1000,
        phashScore: Math.round(phashScore * 1000) / 1000,
        score: Math.round(score * 1000) / 1000,
        source: ref.filename
      };

      if (!byCode[ref.code] || hamming < byCode[ref.code].hamming) {
        byCode[ref.code] = entry;
      }
    }

    const results = Object.values(byCode);
    results.sort((a, b) => b.score - a.score);

    // Cleanup temp file
    try { fs.unlinkSync(tmpPath); } catch {}

    res.json({ results, inputMeta: inputNorm.meta });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`\n  🎴 Card ID Detector running at http://localhost:${PORT}\n`);
  getRefs().then(refs => {
    const codes = [...new Set(refs.map(r => r.code))];
    console.log(`  Reference database: ${refs.length} images, ${codes.length} cards`);
    if (codes.length > 0) {
      console.log(`  Codes: ${codes.join(', ')}\n`);
    } else {
      console.log('  ⚠ No references found. Drop images in references/ folder\n');
    }
  });
});
