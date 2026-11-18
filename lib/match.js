// lib/match.js — Card identity detector via perceptual hashing + histogram + SSIM
// Isomorphic: uses imageio adapter for all image I/O, no direct sharp/canvas.
// Algorithm and weights unchanged: pHash 60% / histogram 25% / SSIM 15%.

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

// Scoring weights (unchanged)
const W_PHASH = 0.6;
const W_HIST = 0.25;
const W_SSIM = 0.15;

// Verdict thresholds (tunable constants)
const CONFIDENT_HAMMING = 6;     // ≤6 bits → confident match
const LIKELY_HAMMING = 12;      // ≤12 + good hist → likely match
const NO_MATCH_HAMMING = 18;     // >18 → no match regardless
const SCORE_MARGIN = 0.05;      // top1 must beat top2 by this much
const MIN_CONFIDENT_SCORE = 0.75; // absolute floor for confident match

/**
 * Normalize an image for matching:
 *   - try cropping to card face (OpenCV contour detection)
 *   - resize to STD_WIDTH × STD_HEIGHT (RGB)
 *   - compute 32×32 grayscale for pHash
 *   - compute full-res grayscale for SSIM
 *
 * Input: either a file path (Node) or a Blob/File (browser).
 * The imageio adapter handles the difference.
 */
async function normalizeImage(input) {
  const imageio = require('./imageio');

  // Load the raw image
  let img = await imageio.loadImage(input);
  const meta = { width: img.width, height: img.height, format: 'raw', channels: img.channels };

  // Try cropping to card face (brown border inner edge)
  let cropped = false;
  let cropReason = null;
  try {
    const cropResult = await cropToCardFace(input);
    if (cropResult.detected && cropResult.cropped) {
      // cropToCardFace in Node returns a sharp instance;
      // in browser it returns { data, width, height, channels }
      if (cropResult.cropped && cropResult.cropped.data) {
        img = cropResult.cropped;
      } else if (typeof cropResult.cropped === 'object' && cropResult.cropped.toBuffer) {
        // sharp instance — load raw pixels
        const sharp = require('sharp');
        const buf = await cropResult.cropped.removeAlpha().raw().toBuffer({ resolveWithObject: true });
        img = { data: buf.data, width: buf.info.width, height: buf.info.height, channels: buf.info.channels };
      }
      cropped = true;
    } else if (cropResult.reason === 'no_valid_card') {
      cropReason = 'no_valid_card';
    }
  } catch (e) {
    // Cropping failed — use original image
  }

  // Resize to standard dimensions (RGB)
  const rgbImg = await imageio.resizePixels(img, STD_WIDTH, STD_HEIGHT);
  const rgb = imageio.toRGB(rgbImg);

  // Create 32×32 grayscale for pHash
  const gray32Img = await imageio.resizePixels(img, PHASH_SIZE, PHASH_SIZE);
  const gray32 = imageio.toGrayscale(gray32Img);

  // Full-res grayscale for SSIM (from the STD_WIDTH×STD_HEIGHT RGB)
  const grayFull = new Uint8Array(STD_WIDTH * STD_HEIGHT);
  for (let i = 0; i < grayFull.length; i++) {
    grayFull[i] = Math.round(0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2]);
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
    meta,
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
 * Compute SSIM between two grayscale images (8×8 window).
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
 * Parse card code from filename.
 */
function parseCode(filename) {
  const ext = path.extname(filename);
  const base = path.basename(filename, ext);
  const m = base.match(/^([A-Za-z]+\d+)[-_\s]?\d*$/);
  return m ? m[1].toUpperCase() : base.toUpperCase();
}

/**
 * Load all reference images from a directory (Node only).
 * In browser, use loadReferencesFromJSON instead.
 */
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
 * Match an input image against all references (runtime path — used by CLI).
 */
async function matchCard(inputPath, refDir) {
  const inputNorm = await normalizeImage(inputPath);
  const refs = await loadReferences(refDir);

  const byCode = {};
  for (const ref of refs) {
    const hamming = hammingDistance(inputNorm.hash, ref.hash);
    const hist = computeHistogramSimilarity(inputNorm.rgb, ref.rgb);
    const ssim = computeSSIM(inputNorm.gray, ref.gray, inputNorm.width, inputNorm.height);
    const phashScore = Math.max(0, 1 - hamming / 24);
    const score = (phashScore * W_PHASH) + (hist * W_HIST) + (ssim * W_SSIM);

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

/**
 * Precompute histogram from raw RGB pixels (for build-refs).
 * Returns Float32Array of 512 normalized probabilities.
 */
function precomputeHistogram(rgb) {
  const bins = 8;
  const shift = 5;
  const numBins = bins * bins * bins;
  const hist = new Float32Array(numBins);
  const pixels = rgb.length / 3;

  for (let i = 0; i < pixels; i++) {
    hist[(rgb[i * 3] >> shift) * bins * bins + (rgb[i * 3 + 1] >> shift) * bins + (rgb[i * 3 + 2] >> shift)]++;
  }

  for (let i = 0; i < numBins; i++) {
    hist[i] /= pixels;
  }

  return Array.from(hist);
}

/**
 * Compute Bhattacharyya coefficient between input RGB pixels and a
 * precomputed histogram (array of 512 probabilities).
 */
function histSimilarityPrecomputed(rgb, refHist) {
  const bins = 8;
  const shift = 5;
  const numBins = bins * bins * bins;
  const histA = new Float64Array(numBins);
  const pixels = rgb.length / 3;

  for (let i = 0; i < pixels; i++) {
    histA[(rgb[i * 3] >> shift) * bins * bins + (rgb[i * 3 + 1] >> shift) * bins + (rgb[i * 3 + 2] >> shift)]++;
  }

  let bc = 0;
  for (let i = 0; i < numBins; i++) {
    bc += Math.sqrt((histA[i] / pixels) * refHist[i]);
  }
  return bc;
}

/**
 * Match pre-normalized input against a pre-built reference DB (browser path).
 * refsData = array of { code, hashHex, histogram, grayThumb }
 * inputNorm = result of normalizeImage()
 *
 * histogram = array of 512 probabilities (precomputed in build-refs)
 * grayThumb = base64-encoded Uint8Array of STD_WIDTH × STD_HEIGHT grayscale
 */
function matchAgainstDB(inputNorm, refsData) {
  const { hexToHash } = require('./phash');

  // Decode grayThumb from base64 once per ref
  const decodedRefs = refsData.map(ref => {
    let grayThumb;
    if (typeof ref.grayThumb === 'string') {
      // Base64 → Uint8Array
      const binary = typeof atob !== 'undefined'
        ? atob(ref.grayThumb)
        : Buffer.from(ref.grayThumb, 'base64').toString('binary');
      grayThumb = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) grayThumb[i] = binary.charCodeAt(i);
    } else {
      grayThumb = ref.grayThumb; // already decoded
    }
    return { ...ref, grayThumb };
  });

  const byCode = {};
  for (const ref of decodedRefs) {
    const refHash = hexToHash(ref.hashHex);
    const hamming = hammingDistance(inputNorm.hash, refHash);
    const hist = histSimilarityPrecomputed(inputNorm.rgb, ref.histogram);
    const ssim = computeSSIM(inputNorm.gray, ref.grayThumb, inputNorm.width, inputNorm.height);
    const phashScore = Math.max(0, 1 - hamming / 24);
    const score = (phashScore * W_PHASH) + (hist * W_HIST) + (ssim * W_SSIM);

    const entry = {
      code: ref.code,
      hamming,
      hist: Math.round(hist * 1000) / 1000,
      ssim: Math.round(ssim * 1000) / 1000,
      phashScore: Math.round(phashScore * 1000) / 1000,
      score: Math.round(score * 1000) / 1000,
    };

    if (!byCode[ref.code] || hamming < byCode[ref.code].hamming) {
      byCode[ref.code] = entry;
    }
  }

  const results = Object.values(byCode);
  results.sort((a, b) => b.score - a.score);
  return results;
}

/**
 * Determine verdict from match results.
 * Returns { verdict, code, hamming, score, margin }
 */
function getVerdict(results) {
  if (!results.length) {
    return { verdict: 'no_match', code: null, hamming: null, score: 0, margin: 0 };
  }

  const best = results[0];
  const second = results[1];
  const margin = second ? (best.score - second.score) : 1.0;

  // No match: hamming too high, or score too low
  if (best.hamming > NO_MATCH_HAMMING || best.score < MIN_CONFIDENT_SCORE) {
    return { verdict: 'no_match', code: best.code, hamming: best.hamming, score: best.score, margin };
  }

  // Confident: low hamming AND good margin
  if (best.hamming <= CONFIDENT_HAMMING && margin >= SCORE_MARGIN) {
    return { verdict: 'match', code: best.code, hamming: best.hamming, score: best.score, margin };
  }

  // Likely: moderate hamming with good histogram
  if (best.hamming <= LIKELY_HAMMING && best.hist > 0.9) {
    return { verdict: 'likely', code: best.code, hamming: best.hamming, score: best.score, margin };
  }

  // Everything else → no confident match
  return { verdict: 'no_match', code: best.code, hamming: best.hamming, score: best.score, margin };
}

module.exports = {
  normalizeImage,
  computeHistogramSimilarity,
  histSimilarityPrecomputed,
  precomputeHistogram,
  computeSSIM,
  loadReferences,
  matchCard,
  matchAgainstDB,
  getVerdict,
  parseCode,
  STD_WIDTH,
  STD_HEIGHT,
  W_PHASH,
  W_HIST,
  W_SSIM,
  CONFIDENT_HAMMING,
  LIKELY_HAMMING,
  NO_MATCH_HAMMING,
  SCORE_MARGIN,
  MIN_CONFIDENT_SCORE,
};
