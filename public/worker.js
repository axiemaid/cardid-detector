// worker.js — Web Worker for card detection + matching
// Runs OpenCV.js (crop) + pHash + histogram + SSIM in background thread.
// Communicates with app.js via postMessage.

importScripts('cv.js');          // OpenCV (lazy — see init)
// phash.js and match.js are loaded via importScripts below

// We can't use ES modules in workers easily with importScripts,
// so we inline the pure-math parts. The image I/O is canvas-based here.

// ── pHash (pure math, copied from lib/phash.js) ───────────
const STD_SIZE = 32;
const HASH_SIZE = 8;

function dct1d(input) {
  const n = input.length;
  const output = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += input[i] * Math.cos((Math.PI / n) * (i + 0.5) * k);
    }
    output[k] = sum;
  }
  return output;
}

function dct2d(data, width, height) {
  const temp = new Float64Array(width * height);
  for (let y = 0; y < height; y++) {
    const row = new Float64Array(width);
    for (let x = 0; x < width; x++) row[x] = data[y * width + x];
    const dctRow = dct1d(row);
    for (let x = 0; x < width; x++) temp[y * width + x] = dctRow[x];
  }
  const result = new Float64Array(width * height);
  for (let x = 0; x < width; x++) {
    const col = new Float64Array(height);
    for (let y = 0; y < height; y++) col[y] = temp[y * width + x];
    const dctCol = dct1d(col);
    for (let y = 0; y < height; y++) result[y * width + x] = dctCol[y];
  }
  return result;
}

function phash(grayData, width, height) {
  const dct = dct2d(grayData, width, height);
  const values = [];
  for (let y = 0; y < HASH_SIZE; y++)
    for (let x = 0; x < HASH_SIZE; x++)
      values.push(dct[y * width + x]);
  const sorted = values.slice(1).sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  let hash = 0n;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > median) hash |= (1n << BigInt(i));
  }
  return hash;
}

function hammingDistance(a, b) {
  let x = a ^ b;
  let count = 0;
  while (x > 0n) { count += Number(x & 1n); x >>= 1n; }
  return count;
}

function hexToHash(hex) { return BigInt('0x' + hex); }

// ── Image helpers (browser worker) ────────────────────────

function resizeGrayscale(data, width, height, targetW, targetH) {
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const imgData = new ImageData(new Uint8ClampedArray(data), width, height);
  // If data is grayscale, expand to RGBA
  const rgba = new Uint8ClampedArray(width * height * 4);
  const ch = data.length / (width * height);
  if (ch === 1) {
    for (let i = 0; i < width * height; i++) {
      rgba[i * 4] = data[i]; rgba[i * 4 + 1] = data[i];
      rgba[i * 4 + 2] = data[i]; rgba[i * 4 + 3] = 255;
    }
    imgData = new ImageData(rgba, width, height);
  }
  ctx.putImageData(imgData, 0, 0);
  const dst = new OffscreenCanvas(targetW, targetH);
  const dctx = dst.getContext('2d', { willReadFrequently: true });
  dctx.drawImage(canvas, 0, 0, targetW, targetH);
  const result = dctx.getImageData(0, 0, targetW, targetH);
  // Convert to grayscale
  const gray = new Uint8Array(targetW * targetH);
  for (let i = 0; i < gray.length; i++) {
    gray[i] = Math.round(0.299 * result.data[i * 4] + 0.587 * result.data[i * 4 + 1] + 0.114 * result.data[i * 4 + 2]);
  }
  return gray;
}

function resizeRGB(data, width, height, channels, targetW, targetH) {
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const rgba = new Uint8ClampedArray(width * height * 4);
  if (channels === 4) {
    rgba.set(data);
  } else {
    for (let i = 0; i < width * height; i++) {
      rgba[i * 4] = data[i * 3];
      rgba[i * 4 + 1] = data[i * 3 + 1];
      rgba[i * 4 + 2] = data[i * 3 + 2];
      rgba[i * 4 + 3] = 255;
    }
  }
  ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
  const dst = new OffscreenCanvas(targetW, targetH);
  const dctx = dst.getContext('2d', { willReadFrequently: true });
  dctx.drawImage(canvas, 0, 0, targetW, targetH);
  const result = dctx.getImageData(0, 0, targetW, targetH);
  // Return RGB Uint8Array
  const rgb = new Uint8Array(targetW * targetH * 3);
  for (let i = 0; i < targetW * targetH; i++) {
    rgb[i * 3] = result.data[i * 4];
    rgb[i * 3 + 1] = result.data[i * 4 + 1];
    rgb[i * 3 + 2] = result.data[i * 4 + 2];
  }
  return rgb;
}

// ── SSIM (from lib/match.js) ──────────────────────────────
function computeSSIM(a, b, width, height) {
  const windowSize = 8;
  const c1 = (0.01 * 255) ** 2;
  const c2 = (0.03 * 255) ** 2;
  const c3 = c2 / 2;
  let totalSSIM = 0, windowCount = 0;
  for (let wy = 0; wy + windowSize <= height; wy += windowSize) {
    for (let wx = 0; wx + windowSize <= width; wx += windowSize) {
      let sumA = 0, sumB = 0, sumAA = 0, sumBB = 0, sumAB = 0;
      const n = windowSize * windowSize;
      for (let y = 0; y < windowSize; y++) {
        for (let x = 0; x < windowSize; x++) {
          const idx = (wy + y) * width + (wx + x);
          const av = a[idx], bv = b[idx];
          sumA += av; sumB += bv;
          sumAA += av * av; sumBB += bv * bv;
          sumAB += av * bv;
        }
      }
      const muA = sumA / n, muB = sumB / n;
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

// ── Histogram (precomputed comparison) ─────────────────────
function histSimilarityPrecomputed(rgb, refHist) {
  const bins = 8, shift = 5;
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

// ── Verdict thresholds ────────────────────────────────────
const CONFIDENT_HAMMING = 6;
const LIKELY_HAMMING = 12;
const NO_MATCH_HAMMING = 18;
const SCORE_MARGIN = 0.05;
const MIN_CONFIDENT_SCORE = 0.75;
const W_PHASH = 0.6, W_HIST = 0.25, W_SSIM = 0.15;

const STD_WIDTH = 343, STD_HEIGHT = 480, PHASH_SIZE = 32;

// ── Main message handler ───────────────────────────────────
self.onmessage = function(e) {
  const { type, ...data } = e.data;
  if (type === 'match') {
    doMatch(data);
  }
};

async function doMatch({ imageData, width, height, refs }) {
  try {
    // Step 1: Downscale large images for processing
    postProgress('Preparing image...');

    // The imageData is RGBA from canvas
    let channels = 4;
    let srcData = imageData;
    let srcW = width;
    let srcH = height;

    // Downscale if very large (max 1024px on longest side)
    const maxDim = 1024;
    if (srcW > maxDim || srcH > maxDim) {
      const scale = maxDim / Math.max(srcW, srcH);
      const newW = Math.round(srcW * scale);
      const newH = Math.round(srcH * scale);
      const canvas = new OffscreenCanvas(srcW, srcH);
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.putImageData(new ImageData(new Uint8ClampedArray(srcData), srcW, srcH), 0, 0);
      const dst = new OffscreenCanvas(newW, newH);
      const dctx = dst.getContext('2d', { willReadFrequently: true });
      dctx.drawImage(canvas, 0, 0, newW, newH);
      const resized = dctx.getImageData(0, 0, newW, newH);
      srcData = resized.data;
      srcW = newW;
      srcH = newH;
    }

    // Step 2: Try OpenCV crop
    let cropData = null;
    let cropW = srcW, cropH = srcH;

    // For now, skip OpenCV in worker (it needs separate loading)
    // Just use the full image. OpenCV crop will be added later.
    // TODO: Load OpenCV.js in worker

    postStepDone('Image prepared');

    // Step 3: Resize to standard dimensions
    postProgress('Normalizing...');

    const rgb = resizeRGB(srcData, srcW, srcH, channels, STD_WIDTH, STD_HEIGHT);

    // 32×32 grayscale for pHash
    const gray32Canvas = new OffscreenCanvas(srcW, srcH);
    const g32ctx = gray32Canvas.getContext('2d', { willReadFrequently: true });
    g32ctx.putImageData(new ImageData(new Uint8ClampedArray(srcData), srcW, srcH), 0, 0);
    const g32dst = new OffscreenCanvas(PHASH_SIZE, PHASH_SIZE);
    const g32dctx = g32dst.getContext('2d', { willReadFrequently: true });
    g32dctx.drawImage(gray32Canvas, 0, 0, PHASH_SIZE, PHASH_SIZE);
    const g32result = g32dctx.getImageData(0, 0, PHASH_SIZE, PHASH_SIZE);
    const gray32 = new Uint8Array(PHASH_SIZE * PHASH_SIZE);
    for (let i = 0; i < gray32.length; i++) {
      gray32[i] = Math.round(0.299 * g32result.data[i * 4] + 0.587 * g32result.data[i * 4 + 1] + 0.114 * g32result.data[i * 4 + 2]);
    }

    // Full-res grayscale for SSIM
    const grayFull = new Uint8Array(STD_WIDTH * STD_HEIGHT);
    for (let i = 0; i < grayFull.length; i++) {
      grayFull[i] = Math.round(0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2]);
    }

    // Step 4: Compute pHash
    postProgress('Computing perceptual hash...');
    const hash = phash(gray32, PHASH_SIZE, PHASH_SIZE);

    postStepDone('Hash computed');

    // Step 5: Match against refs
    postProgress('Matching against database...');

    // Decode grayThumb from base64
    const decodedRefs = refs.map(ref => {
      let grayThumb;
      const binary = atob(ref.grayThumb);
      grayThumb = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) grayThumb[i] = binary.charCodeAt(i);
      return { ...ref, grayThumb };
    });

    const byCode = {};
    for (const ref of decodedRefs) {
      const refHash = hexToHash(ref.hashHex);
      const hamming = hammingDistance(hash, refHash);
      const hist = histSimilarityPrecomputed(rgb, ref.histogram);
      const ssim = computeSSIM(grayFull, ref.grayThumb, STD_WIDTH, STD_HEIGHT);
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

    postStepDone('Matching complete');

    // Verdict
    let verdict;
    if (!results.length) {
      verdict = { verdict: 'no_match', code: null, hamming: null, score: 0, margin: 0 };
    } else {
      const best = results[0];
      const second = results[1];
      const margin = second ? (best.score - second.score) : 1.0;

      if (best.hamming > NO_MATCH_HAMMING || best.score < MIN_CONFIDENT_SCORE) {
        verdict = { verdict: 'no_match', code: best.code, hamming: best.hamming, score: best.score, margin };
      } else if (best.hamming <= CONFIDENT_HAMMING && margin >= SCORE_MARGIN) {
        verdict = { verdict: 'match', code: best.code, hamming: best.hamming, score: best.score, margin };
      } else if (best.hamming <= LIKELY_HAMMING && best.hist > 0.9) {
        verdict = { verdict: 'likely', code: best.code, hamming: best.hamming, score: best.score, margin };
      } else {
        verdict = { verdict: 'no_match', code: best.code, hamming: best.hamming, score: best.score, margin };
      }
    }

    self.postMessage({ type: 'result', results, verdict });
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message });
  }
}

function postProgress(msg) {
  self.postMessage({ type: 'progress', message: msg });
}

function postStepDone(msg) {
  self.postMessage({ type: 'step_done', message: msg });
}
