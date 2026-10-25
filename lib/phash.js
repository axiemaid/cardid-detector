// lib/phash.js — Perceptual hash for card identification
// Based on DCT-based pHash: resize to 32x32 grayscale, apply DCT,
// take top-left 8x8 (low frequencies), compute median, build 64-bit hash.
// Two images of the same card will have nearly identical hashes.

const STD_SIZE = 32;
const HASH_SIZE = 8;

/**
 * Compute 1D DCT of a sequence.
 * Uses the standard DCT-II formula.
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
 * Compute perceptual hash of a grayscale image.
 * Returns a 64-bit hash as a BigInt.
 */
function phash(grayData, width, height) {
  // Extract top-left HASH_SIZE × HASH_SIZE of the DCT
  // (low-frequency components capture overall structure)
  const dct = dct2d(grayData, width, height);

  const values = [];
  for (let y = 0; y < HASH_SIZE; y++) {
    for (let x = 0; x < HASH_SIZE; x++) {
      values.push(dct[y * width + x]);
    }
  }

  // Compute median (excluding the DC term which is just brightness)
  const sorted = values.slice(1).sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];

  // Build hash: bit = 1 if value > median, 0 otherwise
  let hash = 0n;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > median) {
      hash |= (1n << BigInt(i));
    }
  }

  return hash;
}

/**
 * Compute Hamming distance between two BigInt hashes.
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
 * Convert hash to a 64-char binary string for debugging.
 */
function hashToBinary(hash) {
  return hash.toString(2).padStart(64, '0');
}

module.exports = {
  dct1d,
  dct2d,
  phash,
  hammingDistance,
  hashToBinary,
  STD_SIZE,
  HASH_SIZE
};
