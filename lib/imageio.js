// lib/imageio.js — Isomorphic image I/O adapter
//
// Provides a unified interface for loading images and extracting raw pixel
// data. Node uses sharp; browser uses Canvas/ImageData. The rest of lib/
// calls these functions and never touches sharp or canvas directly.
//
// All functions return { data: Uint8Array (RGB or RGBA), width, height, channels }

const isBrowser = typeof window !== 'undefined' && typeof document !== 'undefined';

/**
 * Load an image from a file path (Node) or Blob/File (browser).
 * Returns { data, width, height, channels } in raw pixel form.
 */
async function loadImage(input) {
  if (isBrowser) {
    return loadFromBlob(input);
  } else {
    return loadFromPath(input);
  }
}

// ── Browser path ──────────────────────────────────────────

async function loadFromBlob(blob) {
  const bitmap = await createImageBitmap(blob, { imageOrientation: 'from-image' });
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return {
    data: imageData.data,      // Uint8ClampedArray RGBA
    width: imageData.width,
    height: imageData.height,
    channels: 4
  };
}

// ── Node path ─────────────────────────────────────────────

let _sharp = null;
function getSharp() {
  if (!_sharp) _sharp = require('sharp');
  return _sharp;
}

async function loadFromPath(filePath) {
  const sharp = getSharp();
  const { data, info } = await sharp(filePath)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return {
    data,
    width: info.width,
    height: info.height,
    channels: info.channels
  };
}

/**
 * Resize image data to target dimensions, return raw RGB pixels.
 * Works on the { data, width, height, channels } from loadImage.
 */
async function resizePixels(img, targetW, targetH) {
  if (isBrowser) {
    return resizeBrowser(img, targetW, targetH);
  } else {
    return resizeNode(img, targetW, targetH);
  }
}

async function resizeBrowser(img, targetW, targetH) {
  // Put data into a canvas, then scale-draw into a smaller canvas
  const srcCanvas = new OffscreenCanvas(img.width, img.height);
  const srcCtx = srcCanvas.getContext('2d', { willReadFrequently: true });
  const imageData = new ImageData(
    new Uint8ClampedArray(img.data),
    img.width,
    img.height
  );
  srcCtx.putImageData(imageData, 0, 0);

  const dstCanvas = new OffscreenCanvas(targetW, targetH);
  const dstCtx = dstCanvas.getContext('2d', { willReadFrequently: true });
  dstCtx.drawImage(srcCanvas, 0, 0, targetW, targetH);
  const result = dstCtx.getImageData(0, 0, targetW, targetH);
  return {
    data: new Uint8Array(result.data),  // RGBA
    width: targetW,
    height: targetH,
    channels: 4
  };
}

async function resizeNode(img, targetW, targetH) {
  const sharp = getSharp();
  // Reconstruct a sharp from raw buffer
  const rawImg = sharp(img.data, {
    raw: { width: img.width, height: img.height, channels: img.channels }
  });
  const { data, info } = await rawImg
    .resize(targetW, targetH, { fit: 'fill' })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return {
    data,
    width: info.width,
    height: info.height,
    channels: info.channels
  };
}

/**
 * Extract a sub-region (crop) from image data.
 * Returns { data, width, height, channels }.
 */
async function extractRegion(img, left, top, width, height) {
  if (isBrowser) {
    return extractBrowser(img, left, top, width, height);
  } else {
    return extractNode(img, left, top, width, height);
  }
}

async function extractBrowser(img, left, top, width, height) {
  const srcCanvas = new OffscreenCanvas(img.width, img.height);
  const srcCtx = srcCanvas.getContext('2d', { willReadFrequently: true });
  const imageData = new ImageData(
    new Uint8ClampedArray(img.data),
    img.width,
    img.height
  );
  srcCtx.putImageData(imageData, 0, 0);

  const dstCanvas = new OffscreenCanvas(width, height);
  const dstCtx = dstCanvas.getContext('2d', { willReadFrequently: true });
  dstCtx.drawImage(srcCanvas, left, top, width, height, 0, 0, width, height);
  const result = dstCtx.getImageData(0, 0, width, height);
  return {
    data: new Uint8Array(result.data),
    width,
    height,
    channels: 4
  };
}

async function extractNode(img, left, top, width, height) {
  const sharp = getSharp();
  const rawImg = sharp(img.data, {
    raw: { width: img.width, height: img.height, channels: img.channels }
  });
  const { data, info } = await rawImg
    .extract({ left, top, width, height })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return {
    data,
    width: info.width,
    height: info.height,
    channels: info.channels
  };
}

/**
 * Convert RGBA pixel data to grayscale Uint8Array (1 channel, same dimensions).
 */
function toGrayscale(img) {
  const pixels = img.width * img.height;
  const gray = new Uint8Array(pixels);
  const data = img.data;
  const ch = img.channels;
  for (let i = 0; i < pixels; i++) {
    const idx = i * ch;
    gray[i] = Math.round(0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2]);
  }
  return gray;
}

/**
 * Convert RGBA pixel data to RGB Uint8Array (drop alpha if present).
 */
function toRGB(img) {
  if (img.channels === 3) return img.data;
  const pixels = img.width * img.height;
  const rgb = new Uint8Array(pixels * 3);
  const data = img.data;
  for (let i = 0; i < pixels; i++) {
    rgb[i * 3] = data[i * 4];
    rgb[i * 3 + 1] = data[i * 4 + 1];
    rgb[i * 3 + 2] = data[i * 4 + 2];
  }
  return rgb;
}

module.exports = {
  isBrowser,
  loadImage,
  resizePixels,
  extractRegion,
  toGrayscale,
  toRGB,
};
