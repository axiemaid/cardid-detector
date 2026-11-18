// lib/phash.js — Perceptual hash for card identification
// DCT-based pHash: 32×32 grayscale → 2D DCT → top-left 8×8 → median → 64-bit hash.
// Pure math, no I/O — isomorphic by default.

const STD_SIZE = 32;
const HASH_SIZE = 8;

/**
 * Compute 1D DCT (DCT-II).
 */
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

/**
 * Compute 2D DCT by applying 1D DCT to rows then columns.
 */
function dct2d(data, width, height) {
  // DCT on rows
  const temp = new Float64Array(width * height);
  for (let y = 0; y < height; y++) {
    const row = new Float64Array(width);
    for (let x = 0; x < width; x++) row[x] = data[y * width + x];
    const dctRow = dct1d(row);
    for (let x = 0; x < width; x++) temp[y * width + x] = dctRow[x];
  }

  // DCT on columns
  const result = new Float64Array(width * height);
  for (let x = 0; x < width; x++) {
    const col = new Float64Array(height);
    for (let y = 0; y < height; y++) col[y] = temp[y * width + x];
    const dctCol = dct1d(col);
    for (let y = 0; y < height; y++) result[y * width + x] = dctCol[y];
  }

  return result;
}

/**
 * Compute perceptual hash of a grayscale Uint8Array.
 * Returns a 64-bit hash as a BigInt.
 */
function phash(grayData, width, height) {
  const dct = dct2d(grayData, width, height);

  const values = [];
  for (let y = 0; y < HASH_SIZE; y++) {
    for (let x = 0; x < HASH_SIZE; x++) {
      values.push(dct[y * width + x]);
    }
  }

  // Median excluding DC term
  const sorted = values.slice(1).sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];

  let hash = 0n;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > median) {
      hash |= (1n << BigInt(i));
    }
  }

  return hash;
}

/**
 * Hamming distance between two BigInt hashes.
 */
function hammingDistance(a, b) {
  let x = a ^ b;
  let count = 0;
  while (x > 0n) {
    count += Number(x & 1n);
    x >>= 1n;
  }
  return count;
}

/**
 * Convert hash to hex string (for JSON serialization in refs.json).
 */
function hashToHex(hash) {
  return hash.toString(16);
}

/**
 * Parse hex string back to BigInt hash.
 */
function hexToHash(hex) {
  return BigInt('0x' + hex);
}

/**
 * Convert hash to 64-char binary string for debugging.
 */
function hashToBinary(hash) {
  return hash.toString(2).padStart(64, '0');
}

// Support both CommonJS (Node) and ES module (bundler/browser) environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    dct1d, dct2d, phash, hammingDistance,
    hashToHex, hexToHash, hashToBinary,
    STD_SIZE, HASH_SIZE
  };
}
