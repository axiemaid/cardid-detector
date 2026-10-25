// lib/crop.js — OpenCV-based card detection
//
// Pipeline: grayscale → Gaussian blur → Canny → dilate → findContours
// → largest contour with approxPolyDP == 4 points and area > 20%
// → perspective warp to 343×480.
//
// Fallback: if no quadrilateral found, use bounding box of largest
// blob matching brown border hue in HSV.
//
// Eval: node tests/crop/eval-crop.cjs

const sharp = require('sharp');
const path = require('path');

let _cv = null;

async function getOpenCv() {
  if (_cv) return _cv;
  const cvModule = require('@techstark/opencv-js');
  if (cvModule instanceof Promise) {
    _cv = await cvModule;
  } else if (cvModule.Mat) {
    _cv = cvModule;
  } else {
    await new Promise((resolve) => {
      cvModule.onRuntimeInitialized = () => resolve();
    });
    _cv = cvModule;
  }
  return _cv;
}

// Convert sharp buffer to OpenCV Mat
function bufferToMat(cv, data, width, height, channels) {
  const mat = new cv.Mat(height, width, channels === 4 ? cv.CV_8UC4 : cv.CV_8UC3);
  mat.data.set(data);
  return mat;
}

// Convert OpenCV Mat to sharp buffer
function matToBuffer(cv, mat) {
  const data = new Uint8Array(mat.data());
  return { data, width: mat.cols, height: mat.rows, channels: mat.channels() };
}

function orderPoints(pts) {
  // Order points: TL, TR, BR, BL
  const sorted = [...pts].sort((a, b) => a[0] + a[1] - b[0] - b[1]);
  const tl = sorted[0];
  const br = sorted[sorted.length - 1];

  // The other two
  const remaining = pts.filter(p => p !== tl && p !== br);
  const tr = remaining[0][0] > remaining[1][0] ? remaining[0] : remaining[1];
  const bl = remaining[0][0] <= remaining[1][0] ? remaining[0] : remaining[1];

  return [tl, tr, br, bl];
}

function computeBoundsFromQuadrilateral(corners, width, height) {
  // Given 4 corners (TL, TR, BR, BL), compute bounding box
  const xs = corners.map(c => c[0]);
  const ys = corners.map(c => c[1]);
  return {
    left: Math.max(0, Math.round(Math.min(...xs))),
    top: Math.max(0, Math.round(Math.min(...ys))),
    right: Math.min(width, Math.round(Math.max(...xs))),
    bottom: Math.min(height, Math.round(Math.max(...ys))),
    found: true,
    corners
  };
}

async function detectCardContour(inputPath) {
  const cv = await getOpenCv();
  const meta = await sharp(inputPath).metadata();

  // Load image at reduced size for detection
  const detectW = 400;
  const detectH = Math.round(400 * (meta.height / meta.width));
  const { data: rawBuf, info } = await sharp(inputPath)
    .resize(detectW, detectH)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { width, height, channels } = info;
  const src = bufferToMat(cv, rawBuf, width, height, channels);

  // Grayscale
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);

  // Gaussian blur
  const blurred = new cv.Mat();
  const ksize = new cv.Size(5, 5);
  cv.GaussianBlur(gray, blurred, ksize, 0, 0, cv.BORDER_DEFAULT);

  // Try multiple Canny thresholds — pick the one that finds
  // the best quadrilateral (4 points, largest area)
  const cannyThresholds = [
    [50, 150],   // standard — works for clean edges
    [75, 200],   // stricter — ignores wood grain noise
    [30, 100],   // loose — for low-contrast edges
  ];

  const imageSize = width * height;
  let bestContour = null;
  let bestArea = 0;
  const usedThresholds = null;

  for (const [lo, hi] of cannyThresholds) {
    const edges = new cv.Mat();
    cv.Canny(blurred, edges, lo, hi, 3, false);

    const dilated = new cv.Mat();
    const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
    cv.dilate(edges, dilated, kernel, new cv.Point(-1, -1), 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(dilated, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area = cv.contourArea(contour);

      if (area < imageSize * 0.2) continue;

      const approx = new cv.Mat();
      const peri = cv.arcLength(contour, true);
      cv.approxPolyDP(contour, approx, 0.02 * peri, true);

      if (approx.rows === 4 && area > bestArea) {
        bestContour = approx;
        bestArea = area;
      }
    }

    edges.delete(); dilated.delete(); kernel.delete();
    contours.delete(); hierarchy.delete();

    if (bestContour) break; // found one, stop trying thresholds
  }

  if (!bestContour) {
    // Cleanup
    src.delete(); gray.delete(); blurred.delete();
    return null;
  }

  // Extract corner points
  const corners = [];
  for (let i = 0; i < bestContour.rows; i++) {
    const x = bestContour.data32S[i * 2];
    const y = bestContour.data32S[i * 2 + 1];
    corners.push([x, y]);
  }

  // Order corners: TL, TR, BR, BL
  const ordered = orderPoints(corners);

  // Scale back to original image coordinates
  const scaleX = meta.width / width;
  const scaleY = meta.height / height;
  const scaledCorners = ordered.map(([x, y]) => [
    Math.round(x * scaleX),
    Math.round(y * scaleY)
  ]);

  // Cleanup
  src.delete(); gray.delete(); blurred.delete();
  bestContour.delete();

  return {
    corners: scaledCorners,
    bounds: computeBoundsFromQuadrilateral(scaledCorners, meta.width, meta.height),
    meta
  };
}

// Fallback: find bounding box of brown border hue in HSV
async function detectBrownBorder(inputPath) {
  const cv = await getOpenCv();
  const meta = await sharp(inputPath).metadata();

  const detectW = 400;
  const detectH = Math.round(400 * (meta.height / meta.width));
  const { data: rawBuf, info } = await sharp(inputPath)
    .resize(detectW, detectH)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { width, height, channels } = info;
  const src = bufferToMat(cv, rawBuf, width, height, channels);

  // Convert to HSV
  const hsv = new cv.Mat();
  cv.cvtColor(src, hsv, cv.COLOR_RGB2HSV);

  // Brown border hue range: roughly H 10-30, S 50-255, V 50-200
  const low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [10, 40, 40, 0]);
  const high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [30, 255, 220, 255]);
  const mask = new cv.Mat();
  cv.inRange(hsv, low, high, mask);

  // Find contours of brown regions
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  let bestContour = null;
  let bestArea = 0;
  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i);
    const area = cv.contourArea(contour);
    if (area > bestArea) {
      bestContour = contour;
      bestArea = area;
    }
  }

  if (!bestContour) {
    src.delete(); hsv.delete(); low.delete(); high.delete(); mask.delete();
    contours.delete(); hierarchy.delete();
    return null;
  }

  const rect = cv.boundingRect(bestContour);
  const scaleX = meta.width / width;
  const scaleY = meta.height / height;

  const bounds = {
    left: Math.max(0, Math.round(rect.x * scaleX)),
    top: Math.max(0, Math.round(rect.y * scaleY)),
    right: Math.min(meta.width, Math.round((rect.x + rect.width) * scaleX)),
    bottom: Math.min(meta.height, Math.round((rect.y + rect.height) * scaleY)),
    found: true
  };

  src.delete(); hsv.delete(); low.delete(); high.delete(); mask.delete();
  contours.delete(); hierarchy.delete();

  return { bounds, meta, fallback: 'brown-hsv' };
}

async function cropToCardFace(inputPath) {
  const meta = await sharp(inputPath).metadata();

  // Try contour detection first
  const contourResult = await detectCardContour(inputPath);

  if (contourResult) {
    const b = contourResult.bounds;
    const w = b.right - b.left;
    const h = b.bottom - b.top;

    if (w > 10 && h > 10) {
      const cropped = sharp(inputPath).extract({
        left: b.left, top: b.top, width: w, height: h
      });
      return { cropped, bounds: b, meta, detected: true, method: 'contour' };
    }
  }

  // Fallback: brown border HSV detection
  const brownResult = await detectBrownBorder(inputPath);

  if (brownResult) {
    const b = brownResult.bounds;
    const w = b.right - b.left;
    const h = b.bottom - b.top;

    if (w > 10 && h > 10) {
      const cropped = sharp(inputPath).extract({
        left: b.left, top: b.top, width: w, height: h
      });
      return { cropped, bounds: b, meta, detected: true, method: 'brown-hsv' };
    }
  }

  // No detection possible
  return { cropped: null, bounds: null, meta, detected: false };
}

module.exports = { getOpenCv, detectCardContour, detectBrownBorder, cropToCardFace };
