#!/usr/bin/env node
// scripts/add-reference.cjs — Add a verified scan as a reference image
// Usage: node add-reference.cjs <image> <code> [--auto-crop]
// Saves a processed (edge-cropped, resized) copy to references/

const { normalizeImage } = require('../lib/match');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

async function main() {
  const args = process.argv.slice(2);
  const image = args[0];
  const code = args[1]?.toUpperCase();
  const autoCrop = args.includes('--auto-crop');

  if (!image || !code) {
    console.error('Usage: node add-reference.cjs <image> <code> [--auto-crop]');
    console.error('  <code> = card code, e.g. C01, S03');
    console.error('  --auto-crop = edge detect and crop to card bounds before saving');
    console.error('\nWithout --auto-crop, saves the raw image as-is (for manually cropped scans).');
    process.exit(1);
  }

  const refDir = path.join(__dirname, '..', 'references');
  if (!fs.existsSync(refDir)) fs.mkdirSync(refDir, { recursive: true });

  const outPath = path.join(refDir, `${code}.png`);

  if (autoCrop) {
    const norm = await normalizeImage(image);
    // Save the normalized grayscale raw as PNG
    await sharp(norm.data, { raw: { width: norm.width, height: norm.height, channels: 1 } })
      .png()
      .toFile(outPath);
    console.log(`✅ Saved auto-cropped reference: ${outPath}`);
    console.log(`   Bounds: L=${norm.bounds.left} T=${norm.bounds.top} R=${norm.bounds.right} B=${norm.bounds.bottom}`);
  } else {
    // Save raw image, converted to PNG
    await sharp(image).removeAlpha().png().toFile(outPath);
    console.log(`✅ Saved raw reference: ${outPath}`);
    console.log(`   (Will be auto-cropped at match time)`);
  }

  // List all references
  const refs = fs.readdirSync(refDir).filter(f => /\.(png|jpg|jpeg|webp)$/i.test(f));
  console.log(`\n   Total references: ${refs.length}`);
}

main().catch(err => { console.error('Error:', err.message); process.exit(1); });
