// lib/crop.js — OpenCV-based card detection with candidate scoring
//
// Pipeline: grayscale → Gaussian blur → Canny (multi-threshold) → dilate
// → findContours (RETR_LIST) → collect all 4-point contours as candidates
// → score each by: aspect ratio, corner angles, side length similarity,
//   brown ring presence → pick best → bounding box → crop.
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

function bufferToMat(cv, data, width, height, channels) {
  const mat = new cv.Mat(height, width, channels === 4 ? cv.CV_8UC4 : cv.CV_8UC3);
  mat.data.set(data);
  return mat;
}

function orderPoints(pts) {
  const sorted = [...pts].sort((a, b) => a[0] + a[1] - b[0] - b[1]);
  const tl = sorted[0];
  const br = sorted[sorted.length - 1];
  const remaining = pts.filter(p => p !== tl && p !== br);
  const tr = remaining[0][0] > remaining[1][0] ? remaining[0] : remaining[1];
  const bl = remaining[0][0] <= remaining[1][0] ? remaining[0] : remaining[1];
  return [tl, tr, br, bl];
}

function computeBoundsFromQuadrilateral(corners, width, height) {
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

// Distance between two points
function dist(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return Math.sqrt(dx * dx + dy * dy);
}

// Angle at corner B between vectors BA and BC
function angleAt(a, b, c) {
  const v1x = a[0] - b[0], v1y = a[1] - b[1];
  const v2x = c[0] - b[0], v2y = c[1] - b[1];
  const dot = v1x * v2x + v1y * v2y;
  const mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
  const mag2 = Math.sqrt(v2x * v2x + v2y * v2y);
  if (mag1 === 0 || mag2 === 0) return 0;
  const cosA = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
  return Math.acos(cosA) * 180 / Math.PI;
}

// Target card aspect ratio (2.5 : 3.5)
const TARGET_RATIO = 2.5 / 3.5; // ≈ 0.714
const RATIO_TOLERANCE = 0.15; // reject candidates > 15% off target

// Score a quad candidate [TL, TR, BR, BL]
function scoreQuad(quad) {
  const [tl, tr, br, bl] = quad;

  // Side lengths
  const top = dist(tl, tr);
  const right = dist(tr, br);
  const bottom = dist(br, bl);
  const left = dist(bl, tl);

  const avgW = (top + bottom) / 2;
  const avgH = (left + right) / 2;

  // 1. Aspect ratio vs 2.5:3.5 (target = 0.714)
  if (avgH === 0) return { score: 0, ratio: 0, ratioScore: 0, angleScore: 0, sideScore: 0, angles: [] };
  const ratio = avgW / avgH;
  const ratioOff = Math.abs(ratio - TARGET_RATIO) / TARGET_RATIO;
  // Hard gate: reject candidates more than 15% off target ratio
  if (ratioOff > RATIO_TOLERANCE) return { score: 0, ratio, ratioScore: 0, angleScore: 0, sideScore: 0, angles: [] };
  const ratioScore = 1 - Math.min(1, ratioOff);

  // 2. Corner angles near 90°
  const angles = [
    angleAt(bl, tl, tr),  // TL
    angleAt(tl, tr, br),  // TR
    angleAt(tr, br, bl),  // BR
    angleAt(br, bl, tl),  // BL
  ];
  const angleScore = angles.reduce((s, a) => s + (1 - Math.min(1, Math.abs(a - 90) / 45)), 0) / 4;

  // 3. Opposite side length similarity
  const topBotDiff = Math.abs(top - bottom) / Math.max(top, bottom, 1);
  const leftRightDiff = Math.abs(left - right) / Math.max(left, right, 1);
  const sideScore = (1 - topBotDiff + 1 - leftRightDiff) / 2;

  // Combined score (brown ring check added separately after warp)
  const score = ratioScore * 0.35 + angleScore * 0.35 + sideScore * 0.30;

  return { score, ratio, ratioScore, angleScore, sideScore, angles };
}

// Check if a thin band just inside the quad's border has brown-ring hue.
// We crop the bounding box from the source Mat, then check border pixels.
function brownRingCheck(cv, src, quad, width, height) {
  // Get bounding box of quad
  const xs = quad.map(c => c[0]);
  const ys = quad.map(c => c[1]);
  const x0 = Math.max(0, Math.round(Math.min(...xs)));
  const y0 = Math.max(0, Math.round(Math.min(...ys)));
  const x1 = Math.min(width - 1, Math.round(Math.max(...xs)));
  const y1 = Math.min(height - 1, Math.round(Math.max(...ys)));
  const bw = x1 - x0;
  const bh = y1 - y0;
  if (bw < 10 || bh < 10) return 0;

  // Extract region
  const region = src.roi(new cv.Rect(x0, y0, bw, bh));

  // Convert to HSV
  const hsv = new cv.Mat();
  cv.cvtColor(region, hsv, cv.COLOR_RGB2HSV);

  // Brown hue range: H 10-30, S 40-255, V 40-220
  const low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [10, 40, 40, 0]);
  const high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [30, 255, 220, 255]);
  const mask = new cv.Mat();
  cv.inRange(hsv, low, high, mask);

  // Sample a thin band (5px) just inside each edge of the bounding box
  const band = 5;
  let brownCount = 0;
  let totalCount = 0;

  // Top band
  for (let y = 0; y < Math.min(band, bh); y++) {
    for (let x = 0; x < bw; x++) {
      totalCount++;
      if (mask.data[y * bw + x]) brownCount++;
    }
  }
  // Bottom band
  for (let y = Math.max(0, bh - band); y < bh; y++) {
    for (let x = 0; x < bw; x++) {
      totalCount++;
      if (mask.data[y * bw + x]) brownCount++;
    }
  }
  // Left band
  for (let y = 0; y < bh; y++) {
    for (let x = 0; x < Math.min(band, bw); x++) {
      totalCount++;
      if (mask.data[y * bw + x]) brownCount++;
    }
  }
  // Right band
  for (let y = 0; y < bh; y++) {
    for (let x = Math.max(0, bw - band); x < bw; x++) {
      totalCount++;
      if (mask.data[y * bw + x]) brownCount++;
    }
  }

  region.delete(); hsv.delete(); low.delete(); high.delete(); mask.delete();

  return totalCount > 0 ? brownCount / totalCount : 0;
}

async function detectCardContour(inputPath, { debug = false } = {}) {
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

  // Grayscale
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);

  // Gaussian blur
  const blurred = new cv.Mat();
  cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);

  const cannyThresholds = [
    [50, 150],
    [75, 200],
    [30, 100],
  ];

  const imageSize = width * height;
  const candidates = [];

  // Phase 1: Standard approxPolyDP on all contours
  for (const [lo, hi] of cannyThresholds) {
    const edges = new cv.Mat();
    cv.Canny(blurred, edges, lo, hi, 3, false);

    const dilated = new cv.Mat();
    const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
    cv.dilate(edges, dilated, kernel, new cv.Point(-1, -1), 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(dilated, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area = cv.contourArea(contour);
      if (area < imageSize * 0.2) continue;

      const approx = new cv.Mat();
      const peri = cv.arcLength(contour, true);
      cv.approxPolyDP(contour, approx, 0.02 * peri, true);

      if (approx.rows !== 4) { approx.delete(); continue; }

      const pts = [];
      for (let j = 0; j < 4; j++) {
        pts.push([approx.data32S[j * 2], approx.data32S[j * 2 + 1]]);
      }
      approx.delete();

      const ordered = orderPoints(pts);
      const metrics = scoreQuad(ordered);
      if (metrics.score < 0.1) continue;

      candidates.push({
        corners: ordered, area, areaPct: area / imageSize,
        canny: [lo, hi], contourIdx: i, method: 'approx', ...metrics
      });
    }

    edges.delete(); dilated.delete(); kernel.delete();
    contours.delete(); hierarchy.delete();
  }

  // Phase 2: Hull fallback is after ring gate (below) — only triggers
  // if no standard candidates pass the ring gate.

  // Brown ring check for each candidate
  for (const c of candidates) {
    c.ringScore = brownRingCheck(cv, src, c.corners, width, height);
    // Final score: geometric score weighted by ring presence
    // Ring score 0-1: fraction of border pixels that are brown
    // If ringScore < 0.05, heavily penalize (likely not a card)
    const ringBonus = Math.min(1, c.ringScore / 0.15); // saturate at 15% brown
    c.finalScore = c.score * 0.6 + ringBonus * 0.4;
  }

  // Ring gate: reject candidates with ringScore < RING_GATE
  // These don't have a brown border → not a card (e.g. sleeve edge, image border)
  const RING_GATE = 0.03;
  const passed = candidates.filter(c => c.ringScore >= RING_GATE);

  // Sort by final score (descending) for debug display
  candidates.sort((a, b) => b.finalScore - a.finalScore);
  passed.sort((a, b) => b.finalScore - a.finalScore);

  if (debug) {
    console.log(`  ${candidates.length} candidates found (${passed.length} passed ring gate @ ${RING_GATE}):`);
    for (const c of candidates.slice(0, 5)) {
      const fmt = p => `[${p[0]},${p[1]}]`;
      const gate = c.ringScore >= RING_GATE ? '✓' : '✗';
      console.log(
        `    ${gate} score=${c.finalScore.toFixed(3)} geo=${c.score.toFixed(3)} ring=${c.ringScore.toFixed(3)} ` +
        `area=${(c.areaPct * 100).toFixed(1)}% canny=${c.canny.join(',')} ` +
        `corners=${c.corners.map(fmt).join(' ')}`
      );
    }
  }

  if (passed.length === 0) {
    // Phase 2: Hull fallback — glare may have broken the card contour
    // so approxPolyDP never yielded 4 points. Try convex hull + increasing
    // epsilon on the largest contours from each Canny threshold.
    if (debug) console.log('  Standard candidates all failed ring gate — trying hull fallback...');

    for (const [lo, hi] of cannyThresholds) {
      const edges = new cv.Mat();
      cv.Canny(blurred, edges, lo, hi, 3, false);

      const dilated = new cv.Mat();
      const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
      cv.dilate(edges, dilated, kernel, new cv.Point(-1, -1), 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());

      const contours = new cv.MatVector();
      const hierarchy = new cv.Mat();
      cv.findContours(dilated, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

      for (let i = 0; i < contours.size(); i++) {
        const contour = contours.get(i);
        const area = cv.contourArea(contour);
        if (area < imageSize * 0.2) continue;

        let quadPts = null;
        let usedMethod = 'none';

        // Convex hull
        const hull = new cv.Mat();
        cv.convexHull(contour, hull, false, true);

        if (hull.rows >= 4) {
          const hullPeri = cv.arcLength(hull, true);
          for (const eps of [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15]) {
            const hullApprox = new cv.Mat();
            cv.approxPolyDP(hull, hullApprox, eps * hullPeri, true);
            if (hullApprox.rows === 4) {
              quadPts = [];
              for (let j = 0; j < 4; j++) {
                quadPts.push([hullApprox.data32S[j * 2], hullApprox.data32S[j * 2 + 1]]);
              }
              usedMethod = 'hull';
              hullApprox.delete();
              break;
            }
            hullApprox.delete();
          }
        }

        // minAreaRect fallback
        if (!quadPts) {
          const rect = cv.minAreaRect(contour);
          const boxPts = cv.boxPoints(rect);
          if (boxPts && boxPts.rows === 4) {
            quadPts = [];
            for (let j = 0; j < 4; j++) {
              quadPts.push([Math.round(boxPts.data32F[j * 2]), Math.round(boxPts.data32F[j * 2 + 1])]);
            }
            usedMethod = 'minAreaRect';
          }
        }

        hull.delete();

        if (!quadPts) continue;

        const ordered = orderPoints(quadPts);
        const metrics = scoreQuad(ordered);
        if (metrics.score < 0.1) continue;

        // Ring check on hull candidate
        const ringScore = brownRingCheck(cv, src, ordered, width, height);
        if (ringScore < RING_GATE) continue; // still no brown ring — skip

        const ringBonus = Math.min(1, ringScore / 0.15);
        const finalScore = metrics.score * 0.6 + ringBonus * 0.4;

        passed.push({
          corners: ordered, area, areaPct: area / imageSize,
          canny: [lo, hi], contourIdx: i, method: usedMethod,
          ...metrics, ringScore, finalScore
        });

        if (debug) {
          const fmt = p => `[${p[0]},${p[1]}]`;
          console.log(`    ✓ hull score=${finalScore.toFixed(3)} geo=${metrics.score.toFixed(3)} ring=${ringScore.toFixed(3)} method=${usedMethod} corners=${ordered.map(fmt).join(' ')}`);
        }
      }

      edges.delete(); dilated.delete(); kernel.delete();
      contours.delete(); hierarchy.delete();
      if (passed.length > 0) break;
    }
  }

  if (passed.length === 0) {
    src.delete(); gray.delete(); blurred.delete();
    return { ok: false, reason: 'no_valid_card' };
  }

  // Select the outermost candidate: the one whose bounding box contains
  // all others. Score as tiebreaker when no single candidate contains all.
  // Exclude candidates that touch the image border (within BORDER_MARGIN px).
  const BORDER_MARGIN = 3;
  function bbox(c) {
    const xs = c.corners.map(p => p[0]);
    const ys = c.corners.map(p => p[1]);
    return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
  }
  function touchesBorder(c) {
    const b = bbox(c);
    return b.minX < BORDER_MARGIN || b.minY < BORDER_MARGIN ||
           b.maxX > width - BORDER_MARGIN || b.maxY > height - BORDER_MARGIN;
  }
  function contains(a, b) {
    const ba = bbox(a), bb = bbox(b);
    return bb.minX >= ba.minX && bb.maxX <= ba.maxX && bb.minY >= ba.minY && bb.maxY <= ba.maxY;
  }
  // Filter out border-touching candidates for outermost selection
  const nonBorder = passed.filter(c => !touchesBorder(c));
  const pool = nonBorder.length > 0 ? nonBorder : passed;
  let best = pool[0];
  for (const c of pool) {
    if (contains(c, best)) best = c; // c contains current best → c is outer
  }

  if (debug) {
    const fmt = p => `[${p[0]},${p[1]}]`;
    console.log(`  → selected: score=${best.finalScore.toFixed(3)} ring=${best.ringScore.toFixed(3)} corners=${best.corners.map(fmt).join(' ')} (outermost)`);
  }
  const scaleX = meta.width / width;
  const scaleY = meta.height / height;
  const scaledCorners = best.corners.map(([x, y]) => [
    Math.round(x * scaleX),
    Math.round(y * scaleY)
  ]);

  // Refine corners at full resolution
  const refinedCorners = await refineCorners(inputPath, scaledCorners, meta.width, meta.height, debug);

  src.delete(); gray.delete(); blurred.delete();

  return {
    corners: refinedCorners,
    bounds: computeBoundsFromQuadrilateral(refinedCorners, meta.width, meta.height),
    meta,
    candidates: candidates.slice(0, 5),
    rawCorners: scaledCorners
  };
}

// Refine corners at full resolution by searching for the strongest gradient
// in a small window around each corner. For each corner, look along the two
// adjacent edges and find the point of maximum gradient magnitude.
async function refineCorners(inputPath, corners, fullW, fullH, debug) {
  const cv = await getOpenCv();
  const { data: rawBuf, info } = await sharp(inputPath)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const { width, height, channels } = info;
  const src = bufferToMat(cv, rawBuf, width, height, channels);
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);

  // Compute Sobel gradients
  const gx = new cv.Mat();
  const gy = new cv.Mat();
  cv.Sobel(gray, gx, cv.CV_32F, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT);
  cv.Sobel(gray, gy, cv.CV_32F, 0, 1, 3, 1, 0, cv.BORDER_DEFAULT);

  // Gradient magnitude
  const mag = new cv.Mat();
  cv.magnitude(gx, gy, mag);

  // Window size: proportional to the detection-to-full-res scale factor
  // (detection was at 400px width). This keeps refinement sub-pixel —
  // just cleaning up quantization error, not wandering into internal art.
  const detectScale = width / 400;
  const windowSize = Math.max(2, Math.round(detectScale));

  // Compute centroid of the 4 corners (card center)
  const cx0 = Math.round(corners.reduce((s, c) => s + c[0], 0) / 4);
  const cy0 = Math.round(corners.reduce((s, c) => s + c[1], 0) / 4);

  const refined = corners.map(([cx, cy]) => {
    const x0 = Math.max(0, cx - windowSize);
    const x1 = Math.min(width - 1, cx + windowSize);
    const y0 = Math.max(0, cy - windowSize);
    const y1 = Math.min(height - 1, cy + windowSize);

    // Direction from center to this corner (outward)
    const dirX = cx - cx0;
    const dirY = cy - cy0;
    const dirLen = Math.sqrt(dirX * dirX + dirY * dirY) || 1;
    const ux = dirX / dirLen;
    const uy = dirY / dirLen;

    let bestX = cx, bestY = cy;
    let bestScore = -1;

    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const m = mag.data32F[y * width + x];
        // Only consider pixels in the outward half of the window
        const dx = x - cx;
        const dy = y - cy;
        const dot = dx * ux + dy * uy; // > 0 means outward
        if (dot < -windowSize * 0.3) continue; // allow slight inward movement
        // Weight by gradient magnitude, with preference for outward direction
        const score = m * (1 + 0.3 * Math.max(0, dot / windowSize));
        if (score > bestScore) {
          bestScore = score;
          bestX = x;
          bestY = y;
        }
      }
    }

    return [bestX, bestY];
  });

  if (debug) {
    for (let i = 0; i < corners.length; i++) {
      const dx = refined[i][0] - corners[i][0];
      const dy = refined[i][1] - corners[i][1];
      console.log(`    refine corner ${i}: [${corners[i][0]},${corners[i][1]}] → [${refined[i][0]},${refined[i][1]}] (Δ${dx},${dy})`);
    }
  }

  src.delete(); gray.delete(); gx.delete(); gy.delete(); mag.delete();
  return refined;
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

  const hsv = new cv.Mat();
  cv.cvtColor(src, hsv, cv.COLOR_RGB2HSV);

  const low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [10, 40, 40, 0]);
  const high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [30, 255, 220, 255]);
  const mask = new cv.Mat();
  cv.inRange(hsv, low, high, mask);

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

  const contourResult = await detectCardContour(inputPath);

  // Ring gate rejected all candidates — ask for retake
  if (contourResult && contourResult.ok === false) {
    return { cropped: null, bounds: null, meta, detected: false, reason: contourResult.reason };
  }

  if (contourResult && contourResult.bounds) {
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

  return { cropped: null, bounds: null, meta, detected: false };
}

module.exports = {
  getOpenCv, detectCardContour, detectBrownBorder, cropToCardFace,
  orderPoints, scoreQuad, brownRingCheck, angleAt, dist,
  TARGET_RATIO, RATIO_TOLERANCE
};
