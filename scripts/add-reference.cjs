#!/usr/bin/env node
// scripts/add-reference.cjs — Add a verified scan as a reference image
// Usage: node add-reference.cjs <image> <code>
// Saves to references/<code>.png (or <code>_2.png, <code>_3.png if exists)

const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

async function main() {
  const args = process.argv.slice(2);
  const image = args[0];
  const code = args[1]?.toUpperCase();

  if (!image || !code) {
    console.error('Usage: node add-reference.cjs <image> <code>');
    console.error('  <code> = card code, e.g. C01, S03');
    console.error('  Multiple references per code supported: C01.png, C01_2.png, C01_3.png');
    process.exit(1);
  }

  const refDir = path.join(__dirname, '..', 'references');
  if (!fs.existsSync(refDir)) fs.mkdirSync(refDir, { recursive: true });

  // Find next available filename
  let outPath = path.join(refDir, `${code}.png`);
  let n = 2;
  while (fs.existsSync(outPath)) {
    outPath = path.join(refDir, `${code}_${n}.png`);
    n++;
  }

  // Save as PNG
  await sharp(image).removeAlpha().png().toFile(outPath);
  console.log(`✅ Saved reference: ${path.basename(outPath)}`);
  console.log(`   Code: ${code}`);

  // List all references for this code
  const all = fs.readdirSync(refDir).filter(f => /\.(png|jpg|jpeg|webp)$/i.test(f));
  const forCode = all.filter(f => {
    const ext = path.extname(f);
    const base = path.basename(f, ext);
    const m = base.match(/^([A-Za-z]+\d+)[-_\s]?\d*$/);
    return m && m[1].toUpperCase() === code;
  });
  console.log(`   References for ${code}: ${forCode.length}`);

  // List all unique codes
  const codes = new Set();
  for (const f of all) {
    const ext = path.extname(f);
    const base = path.basename(f, ext);
    const m = base.match(/^([A-Za-z]+\d+)[-_\s]?\d*$/);
    if (m) codes.add(m[1].toUpperCase());
  }
  console.log(`\n   Total: ${all.length} images, ${codes.size} cards`);
  console.log(`   Codes: ${[...codes].sort().join(', ')}`);
}

main().catch(err => { console.error('Error:', err.message); process.exit(1); });
