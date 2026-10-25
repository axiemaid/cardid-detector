// lib/match.js — Card identity detector via perceptual hashing + histogram
// Pipeline: resize → grayscale → pHash (DCT) → compare hashes (Hamming distance)
//           + color histogram as secondary signal

const sharp = require('sharp');
const path = require('path');
const fs = require('fs');
const { promisify } = require('util');
const readdir = promisify(fs.readdir);
const { phash, hammingDistance } = require('./phash');

// Standard comparison dimensions
const STD_WIDTH = 343;
const STD_HEIGHT = 480;
const PHASH_SIZE = 32;

/**
 * Normalize an image: resize to standard dimensions, return RGB + grayscale.
 * No edge cropping — pHash is robust to minor border differences.
 * If the image has a sleeve/background, the DCT low frequencies still capture
 * the card's dominant structure.
 */
async function normalizeImage(inputPath) {
  const meta = await sharp(inputPath).metadata();

  // Resize to standard dimensions, keep RGB
  const rgb = await sharp(inputPath)
    .resize(STD_WIDTH, STD_HEIGHT, { fit: 'fill' })
    .removeAlpha()
    .raw()
    .toBuffer();

  // Create grayscale for pHash (downscaled to 32x32)
  const gray32 = await sharp(inputPath)
    .resize(PHASH_SIZE, PHASH_SIZE, { fit: 'fill' })
    .removeAlpha()
    .grayscale()
    .raw()
    .toBuffer();

  // Also create full-res grayscale for SSIM if needed
  const grayFull = new Uint8Array(STD_WIDTH * STD_HEIGHT);
  for (let i = 0; i < grayFull.length; i++) {
    const r = rgb[i * 3];
    const g = rgb[i * 3 + 1];
    const b = rgb[i * 3 + 2];
    grayFull[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }

  // Compute pHash
  const hash = phash(gray32, PHASH_SIZE, PHASH_SIZE);

  return {
    rgb,
    gray: grayFull,
    gray32,
    hash,
    width: STD_WIDTH,
    height: STD_HEIGHT,
    meta
  };
}

/**
 * Compute color histogram similarity (Bhattacharyya coefficient).
 * 8 bins per channel → 512 bins total.
 */
function computeHistogramSimilarity(a, b) {
  const bins = 8;
  const shift = 5;
  const numBins = bins * bins * bins;
  const histA = new Float64Array(numBins);
  const histB = new Float64Array(numBins);

  const pixels = a.length / 3;
  for (let i = 0; i < pixels; i++) {
    histA[(a[i * 3] >> shift) * bins * bins + (a[i * 3 + 1] >> shift) * bins + (a[i * 3 + 2] >> shift)]++;
    histB[(b[i * 3] >> shift) * bins * bins + (b[i * 3 + 1] >> shift) * bins + (b[i * 3 + 2] >> shift)]++;
  }

  let bc = 0;
  for (let i = 0; i < numBins; i++) {
    bc += Math.sqrt((histA[i] / pixels) * (histB[i] / pixels));
  }
  return bc;
}

/**
 * Compute SSIM between two grayscale images (8x8 window).
 */
function computeSSIM(a, b, width, height) {
  const windowSize = 8;
  const c1 = (0.01 * 255) ** 2;
  const c2 = (0.03 * 255) ** 2;
  const c3 = c2 / 2;

  let totalSSIM = 0;
  let windowCount = 0;

  for (let wy = 0; wy + windowSize <= height; wy += windowSize) {
    for (let wx = 0; wx + windowSize <= width; wx += windowSize) {
      let sumA = 0, sumB = 0, sumAA = 0, sumBB = 0, sumAB = 0;
      const n = windowSize * windowSize;

      for (let y = 0; y < windowSize; y++) {
        for (let x = 0; x < windowSize; x++) {
          const idx = (wy + y) * width + (wx + x);
          const av = a[idx];
          const bv = b[idx];
          sumA += av;
          sumB += bv;
          sumAA += av * av;
          sumBB += bv * bv;
          sumAB += av * bv;
        }
      }

      const muA = sumA / n;
      const muB = sumB / n;
      const varA = sumAA / n - muA * muA;
      const varB = sumBB / n - muB * muB;
      const covAB = sumAB / n - muA * muB;

      if (varA < 1 && varB < 1) {
        const meanDiff = Math.abs(muA - muB);
        totalSSIM += meanDiff < 5 ? 1.0 : Math.max(0, 1 - meanDiff / 255);
        windowCount++;
        continue;
      }

      const ssim = ((2 * muA * muB + c1) * (2 * covAB + c3)) /
                   ((muA * muA + muB * muB + c1) * (varA + varB + c2));
      totalSSIM += ssim;
      windowCount++;
    }
  }

  return windowCount > 0 ? totalSSIM / windowCount : 0;
}

/**
 * Load all reference images from a directory.
 */
async function loadReferences(refDir) {
  const files = await readdir(refDir);
  const refs = [];

  for (const file of files) {
    const ext = path.extname(file).toLowerCase();
    if (!['.png', '.jpg', '.jpeg', '.webp', '.bmp'].includes(ext)) continue;

    const code = path.basename(file, ext);
    const fullPath = path.join(refDir, file);
    const normalized = await normalizeImage(fullPath);
    refs.push({ code, ...normalized, source: fullPath });
  }

  return refs;
}

/**
 * Match an input image against all references.
 * 
 * pHash Hamming distance is the primary signal:
 *   0-6 bits  → very likely same card
 *   7-12 bits → possibly same card (needs secondary check)
 *   13+ bits  → different card
 * 
 * Histogram similarity is secondary:
 *   >0.95  → same color distribution
 *   0.85-0.95 → similar colors
 *   <0.85  → different colors
 * 
 * Returns sorted results with all metrics.
 */
async function matchCard(inputPath, refDir) {
  const inputNorm = await normalizeImage(inputPath);
  const refs = await loadReferences(refDir);

  const results = refs.map(ref => {
    const hamming = hammingDistance(inputNorm.hash, ref.hash);
    const hist = computeHistogramSimilarity(inputNorm.rgb, ref.rgb);
    const ssim = computeSSIM(inputNorm.gray, ref.gray, inputNorm.width, inputNorm.height);

    // Convert Hamming distance to a 0-1 similarity score
    // 0 bits = 1.0, 32+ bits = 0.0 (64-bit hash, but typical range is 0-30)
    const phashScore = Math.max(0, 1 - hamming / 24);

    // Histogram is already 0-1
    // SSIM is 0-1

    // Final score: pHash is most discriminating, histogram is secondary
    const score = (phashScore * 0.6) + (hist * 0.25) + (ssim * 0.15);

    return {
      code: ref.code,
      hamming,
      hist: Math.round(hist * 1000) / 1000,
      ssim: Math.round(ssim * 1000) / 1000,
      phashScore: Math.round(phashScore * 1000) / 1000,
      score: Math.round(score * 1000) / 1000
    };
  });

  results.sort((a, b) => b.score - a.score);
  return { results, inputMeta: inputNorm.meta };
}

module.exports = {
  normalizeImage,
  computeHistogramSimilarity,
  computeSSIM,
  loadReferences,
  matchCard,
  STD_WIDTH,
  STD_HEIGHT
};
