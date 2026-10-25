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

console.log(`\n  Reference images (${files.length}):\n`);
for (const f of files) {
  const stat = fs.statSync(path.join(refDir, f));
  const kb = (stat.size / 1024).toFixed(1);
  const code = path.basename(f, path.extname(f));
  console.log(`    ${code.padEnd(8)} ${kb.padStart(8)} KB    ${f}`);
}
console.log('');
