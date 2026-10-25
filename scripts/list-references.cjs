#!/usr/bin/env node
// scripts/list-references.cjs — List all reference cards
// Usage: node list-references.cjs

const fs = require('fs');
const path = require('path');

const refDir = path.join(__dirname, '..', 'references');

if (!fs.existsSync(refDir) || fs.readdirSync(refDir).length === 0) {
  console.log('\n  No reference images found.');
  console.log('  Add some with: node add-reference.cjs <image> <code>\n');
  process.exit(0);
}

const files = fs.readdirSync(refDir).filter(f => /\.(png|jpg|jpeg|webp|bmp)$/i.test(f));
files.sort();

// Group by code
const byCode = {};
for (const f of files) {
  const ext = path.extname(f);
  const base = path.basename(f, ext);
  const m = base.match(/^([A-Za-z]+\d+)[-_\s]?\d*$/);
  const code = m ? m[1].toUpperCase() : base.toUpperCase();
  if (!byCode[code]) byCode[code] = [];
  byCode[code].push({ file: f, size: fs.statSync(path.join(refDir, f)).size });
}

const codes = Object.keys(byCode).sort();

console.log(`\n  Reference database: ${files.length} images, ${codes.length} cards\n`);

for (const code of codes) {
  const refs = byCode[code];
  const sizes = refs.map(r => `${(r.size / 1024).toFixed(0)}KB`).join(' + ');
  const count = refs.length > 1 ? ` (${refs.length} refs)` : '';
  console.log(`    ${code.padEnd(8)} ${sizes.padEnd(20)}${count}`);
}
console.log('');
