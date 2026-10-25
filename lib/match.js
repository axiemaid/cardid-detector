// lib/match.js — Card identity detector via perceptual hashing + histogram
// Pipeline: resize → grayscale → pHash (DCT) → compare hashes (Hamming distance)
//           + color histogram as secondary signal

const sharp = require('sharp');
const path = require('path');
const fs = require('fs');
const { promisify } = require('util');
const readdir = promisify(fs.readdir);
const { phash, hammingDistance } = require('./phash');
const { cropToCardFace } = require('./crop');

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

  // Try cropping to card face (brown border inner edge)
  let processPath = inputPath;
  let cropped = false;
  let cropReason = null;
  try {
    const cropResult = await cropToCardFace(inputPath);
    if (cropResult.detected && cropResult.cropped) {
      // Write cropped image to temp file for sharp to read
      const tmp = require('os').tmpdir() + '/cardid-crop-' + Date.now() + '.png';
      await cropResult.cropped.png().toFile(tmp);
      processPath = tmp;
      cropped = true;
    } else if (cropResult.reason === 'no_valid_card') {
      cropReason = 'no_valid_card';
    }
  } catch (e) {
    // Cropping failed — use original image
  }

  const meta2 = cropped ? await sharp(processPath).metadata() : meta;

  // Resize to standard dimensions, keep RGB
  const rgb = await sharp(processPath)
    .resize(STD_WIDTH, STD_HEIGHT, { fit: 'fill' })
    .removeAlpha()
    .raw()
    .toBuffer();

  // Create grayscale for pHash (downscaled to 32x32)
  const gray32 = await sharp(processPath)
    .resize(PHASH_SIZE, PHASH_SIZE, { fit: 'fill' })
    .removeAlpha()
    .grayscale()
    .raw()
    .toBuffer();

  // Full-res grayscale for SSIM
  const grayFull = new Uint8Array(STD_WIDTH * STD_HEIGHT);
  for (let i = 0; i < grayFull.length; i++) {
    grayFull[i] = Math.round(0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2]);
  }

  // Compute pHash
  const hash = phash(gray32, PHASH_SIZE, PHASH_SIZE);

  // Cleanup temp file
  if (cropped) {
    try { require('fs').unlinkSync(processPath); } catch {}
  }

  return {
    rgb,
    gray: grayFull,
    gray32,
    hash,
    width: STD_WIDTH,
    height: STD_HEIGHT,
    meta: meta2,
    cropped,
    cropReason
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
function parseCode(filename) {
  const ext = path.extname(filename);
  const base = path.basename(filename, ext);
  // Strip _2, -3, (1), etc. suffixes — code is the part before the last separator+number
  const m = base.match(/^([A-Za-z]+\d+)[-_\s]?\d*$/);
  return m ? m[1].toUpperCase() : base.toUpperCase();
}

async function loadReferences(refDir) {
  const files = await readdir(refDir);
  const refs = [];

  for (const file of files) {
    const ext = path.extname(file).toLowerCase();
    if (!['.png', '.jpg', '.jpeg', '.webp', '.bmp'].includes(ext)) continue;

    const code = parseCode(file);
    const fullPath = path.join(refDir, file);
    const normalized = await normalizeImage(fullPath);
    refs.push({ code, ...normalized, source: fullPath, filename: file });
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
