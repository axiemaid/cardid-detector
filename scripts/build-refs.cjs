#!/usr/bin/env node
// scripts/build-refs.cjs — Precompute reference DB into public/refs.json
// Run: node scripts/build-refs.cjs
// Output: public/refs.json with { code, hashHex, histogram, grayThumb } per card.
// The browser loads this JSON instead of processing reference images at runtime.

const fs = require('fs');
const path = require('path');
const { normalizeImage, parseCode, precomputeHistogram } = require('../lib/match');
const { hashToHex } = require('../lib/phash');

const REF_DIR = path.join(__dirname, '..', 'references');
const OUT_DIR = path.join(__dirname, '..', 'public');
const OUT_FILE = path.join(OUT_DIR, 'refs.json');

async function main() {
  // Check if refs.json already exists and is up-to-date
  if (fs.existsSync(OUT_FILE)) {
    const refMtime = fs.statSync(OUT_FILE).mtimeMs;
    let newestRef = 0;
    for (const f of await fs.promises.readdir(REF_DIR)) {
      const ext = path.extname(f).toLowerCase();
      if (['.png', '.jpg', '.jpeg', '.webp', '.bmp'].includes(ext)) {
        newestRef = Math.max(newestRef, fs.statSync(path.join(REF_DIR, f)).mtimeMs);
      }
    }
    if (refMtime > newestRef) {
      const stats = JSON.parse(fs.readFileSync(OUT_FILE, 'utf8'));
      console.log(`\n  refs.json up to date (${stats.count} cards). Skipping rebuild.\n  To force: delete public/refs.json\n`);
      return;
    }
  }

  const files = await fs.promises.readdir(REF_DIR);
  const imageFiles = files.filter(f => {
    const ext = path.extname(f).toLowerCase();
    return ['.png', '.jpg', '.jpeg', '.webp', '.bmp'].includes(ext);
  });

  console.log(`\n  Building refs.json from ${imageFiles.length} images...\n`);

  const byCode = {};

  for (const file of imageFiles) {
    const code = parseCode(file);
    const fullPath = path.join(REF_DIR, file);
    process.stdout.write(`  ${code} ← ${file} ... `);
    
    try {
      const norm = await normalizeImage(fullPath);
      const hashHex = hashToHex(norm.hash);
      
      // Store histogram as Uint8Array (RGB bins, 8 per channel = 512 bins)
      // Actually store the raw RGB pixels at STD_WIDTH × STD_HEIGHT for histogram
      // That's 343*480*3 = ~494KB per card — too much for JSON.
      // Instead, precompute the histogram bins and store those.
      const hist = precomputeHistogram(norm.rgb);      
      // Store grayscale thumbnail for SSIM (STD_WIDTH × STD_HEIGHT = 164,640 bytes)
      // As array of numbers in JSON — ~490KB per card. Still big.
      // Store as base64 of the Uint8Array buffer.
      const grayThumb = Buffer.from(norm.gray).toString('base64');
      
      if (!byCode[code] || true) {
        // For multiple refs of same code, keep the first (best quality scan)
        if (!byCode[code]) {
          byCode[code] = { code, hashHex, histogram: hist, grayThumb };
          console.log('✓');
        } else {
          // Multiple refs — store hash only (use best hamming at runtime)
          byCode[code].altHashes = byCode[code].altHashes || [];
          byCode[code].altHashes.push(hashHex);
          console.log('✓ (additional ref)');
        }
      }
    } catch (err) {
      console.log(`✗ ${err.message}`);
    }
  }

  const codes = Object.keys(byCode).sort();
  const entries = codes.map(c => byCode[c]);

  // Ensure public/ exists
  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

  fs.writeFileSync(OUT_FILE, JSON.stringify({ 
    version: 1,
    count: entries.length,
    cards: entries 
  }));

  const sizeKB = Math.round(fs.statSync(OUT_FILE).size / 1024);
  console.log(`\n  ✓ Wrote ${OUT_FILE}`);
  console.log(`    ${entries.length} cards, ${sizeKB} KB`);
  console.log(`    Codes: ${codes.join(', ')}\n`);
}

main().catch(err => { console.error('Error:', err); process.exit(1); });
