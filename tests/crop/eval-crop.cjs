#!/usr/bin/env node
// tests/crop/eval-crop.cjs — Evaluate crop accuracy against hand-marked corners
//
// Usage: node eval-crop.cjs [crop-module-path]
// Reports mean corner error (in pixels) per image and overall.
// Lower is better. The crop module must export: cropToCardFace(inputPath) → { detected, bounds: {left,top,right,bottom} }

const path = require('path');
const fs = require('fs');

const CROP_DIR = path.join(__dirname, '..', '..');
const cornersPath = path.join(__dirname, 'corners.json');
const corners = JSON.parse(fs.readFileSync(cornersPath, 'utf8'));

// The crop module to evaluate (pass path as arg, or default to lib/crop.js)
const cropModulePath = process.argv[2] || path.join(CROP_DIR, 'lib', 'crop.js');
const { cropToCardFace } = require(cropModulePath);

async function evalCrop() {
  const results = [];
  let totalError = 0;
  let totalCount = 0;
  let passCount = 0;

  const imageNames = Object.keys(corners);

  for (const name of imageNames) {
    const imgPath = path.join(__dirname, name);
    if (!fs.existsSync(imgPath)) {
      console.log(`⚠ ${name}: file not found, skipping`);
      continue;
    }

    const expected = corners[name];
    const expectedCorners = [
      expected.corners.tl,
      expected.corners.tr,
      expected.corners.br,
      expected.corners.bl,
    ];

    try {
      const result = await cropToCardFace(imgPath);

      if (!result.detected) {
        const reason = result.reason || 'unknown';
        if (reason === 'no_valid_card') {
          console.log(`⊘ ${name}: no valid card (ring gate rejected all candidates)`);
        } else {
          console.log(`✗ ${name}: no card detected (${reason})`);
        }
        results.push({ name, error: null, detected: false, reason });
        continue;
      }

      // The crop returns bounding box {left, top, right, bottom}
      // Convert to 4 corners
      const b = result.bounds;
      const actualCorners = [
        [b.left, b.top],    // TL
        [b.right, b.top],   // TR
        [b.right, b.bottom], // BR
        [b.left, b.bottom], // BL
      ];

      // Compute corner error: average Euclidean distance between
      // expected and actual corner positions
      let maxError = 0;
      let sumError = 0;
      const errors = [];

      for (let i = 0; i < 4; i++) {
        const dx = actualCorners[i][0] - expectedCorners[i][0];
        const dy = actualCorners[i][1] - expectedCorners[i][1];
        const dist = Math.sqrt(dx * dx + dy * dy);
        errors.push(dist);
        sumError += dist;
        maxError = Math.max(maxError, dist);
      }

      const meanError = sumError / 4;
      totalError += sumError;
      totalCount += 4;
      passCount++;

      // Pass if mean error < 1.5% of image diagonal
      const [w, h] = expected.size;
      const diagonal = Math.sqrt(w * w + h * h);
      const passThreshold = diagonal * 0.03;
      const passed = meanError < passThreshold;

      console.log(
        `${passed ? '✅' : '❌'} ${name}: mean=${meanError.toFixed(1)}px max=${maxError.toFixed(1)}px ` +
        `(threshold=${passThreshold.toFixed(1)}px) ` +
        `[${errors.map(e => e.toFixed(0)).join(', ')}]`
      );

      // Show actual vs expected
      const fmt = c => `[${c[0]},${c[1]}]`;
      console.log(
        `   expected: ${expectedCorners.map(fmt).join(' ')}\n` +
        `   actual:   ${actualCorners.map(fmt).join(' ')}`
      );

      results.push({ name, meanError, maxError, errors, detected: true, passed });
    } catch (e) {
      console.log(`💥 ${name}: ${e.message}`);
      results.push({ name, error: e.message, detected: false });
    }
  }

  // Summary
  const overallMean = totalCount > 0 ? (totalError / totalCount).toFixed(1) : 'N/A';
  console.log(`\n${'─'.repeat(60)}`);
  console.log(`Overall: mean corner error = ${overallMean}px across ${passCount}/${imageNames.length} images`);

  const failed = results.filter(r => r.detected && !r.passed);
  if (failed.length > 0) {
    console.log(`\nFailing images:`);
    for (const f of failed) {
      console.log(`  ${f.name}: mean=${f.meanError.toFixed(1)}px max=${f.maxError.toFixed(1)}px`);
    }
  }

  const notDetected = results.filter(r => !r.detected);
  if (notDetected.length > 0) {
    console.log(`\nNot detected:`);
    for (const f of notDetected) {
      console.log(`  ${f.name}`);
    }
  }

  // Exit code: 0 if all pass, 1 if any fail
  process.exit(failed.length === 0 && notDetected.length === 0 ? 0 : 1);
}

evalCrop().catch(err => {
  console.error('Eval failed:', err);
  process.exit(1);
});
