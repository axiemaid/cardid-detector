#!/usr/bin/env node
// scripts/match.cjs — Match an image against reference cards
// Usage: node match.cjs <image> [--references <dir>] [--json] [--top N]

const { matchCard } = require('../lib/match');
const { hashToBinary } = require('../lib/phash');
const path = require('path');

async function main() {
  const args = process.argv.slice(2);
  const image = args[0];
  const refFlagIdx = args.indexOf('--references');
  let refDir;
  if (refFlagIdx >= 0 && args[refFlagIdx + 1]) {
    refDir = args[refFlagIdx + 1];
  } else {
    refDir = path.join(__dirname, '..', 'references');
  }
  const jsonOut = args.includes('--json');
  const topIdx = args.indexOf('--top');
  const topN = topIdx >= 0 ? parseInt(args[topIdx + 1]) : 5;

  if (!image) {
    console.error('Usage: node match.cjs <image> [--references <dir>] [--json] [--top N]');
    process.exit(1);
  }

  try {
    const { results, inputMeta } = await matchCard(image, refDir);

    if (jsonOut) {
      console.log(JSON.stringify({
        input: image,
        meta: { width: inputMeta.width, height: inputMeta.height, format: inputMeta.format },
        topN: results.slice(0, topN)
      }, null, 2));
      return;
    }

    console.log(`\n  Input: ${image}`);
    console.log(`  Original: ${inputMeta.width}×${inputMeta.height} ${inputMeta.format}`);
    console.log(`  References: ${results.length} loaded\n`);

    if (results.length === 0) {
      console.log('  ⚠ No reference images found.');
      console.log('  Add some with: node add-reference.cjs <image> <code>');
      return;
    }

    console.log(`  Top ${Math.min(topN, results.length)} matches:\n`);
    console.log('  ┌────────┬──────────┬────────┬────────┬────────┬────────┐');
    console.log('  │ Code   │   Score  │ Hamming│  Hist  │  SSIM  │ pHash  │');
    console.log('  ├────────┼──────────┼────────┼────────┼────────┼────────┤');

    for (const r of results.slice(0, topN)) {
      const bar = '█'.repeat(Math.round(r.score * 10)).padEnd(10, '░');
      console.log(`  │ ${r.code.padEnd(6)} │ ${bar} │ ${String(r.hamming).padStart(6)} │ ${r.hist.toFixed(3)} │ ${r.ssim.toFixed(3)} │ ${r.phashScore.toFixed(3)} │`);
    }

    console.log('  └────────┴──────────┴────────┴────────┴────────┴────────┘');

    const best = results[0];
    const second = results[1];
    const margin = (second?.hamming || 99) - best.hamming;

    // Verdict based primarily on Hamming distance
    let verdict;
    if (best.hamming <= 6) {
      verdict = `✅ Match: ${best.code} (Hamming ${best.hamming}, confident)`;
    } else if (best.hamming <= 12 && best.hist > 0.9) {
      verdict = `🟡 Likely: ${best.code} (Hamming ${best.hamming}, hist ${best.hist.toFixed(3)})`;
    } else {
      verdict = `❌ No confident match (best: ${best.code} @ Hamming ${best.hamming})`;
    }

    console.log(`\n  ${verdict}\n`);
  } catch (err) {
    console.error('Error:', err.message);
    process.exit(1);
  }
}

main();
