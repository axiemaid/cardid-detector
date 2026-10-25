#!/usr/bin/env node
// tests/crop/debug-crop.cjs — Save intermediate stages for a single image
// Usage: node debug-crop.cjs <image-path>
// Outputs: /tmp/debug-*.png

const sharp = require('sharp');
const path = require('path');

async function main() {
  const inputPath = process.argv[2];
  if (!inputPath) {
    console.error('Usage: node debug-crop.cjs <image-path>');
    process.exit(1);
  }

  const { getOpenCv } = require('../../lib/crop');
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
  console.log('Image:', meta.width + 'x' + meta.height, '→ detect:', width + 'x' + height);

  const src = new cv.Mat(height, width, cv.CV_8UC3);
  src.data.set(rawBuf);

  // Stage 1: Grayscale
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
  await sharp(Buffer.from(gray.data), { raw: { width, height, channels: 1 } })
    .png().toFile('/tmp/debug-1-gray.png');
  console.log('Saved: /tmp/debug-1-gray.png');

  // Stage 2: Gaussian blur
  const blurred = new cv.Mat();
  cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
  await sharp(Buffer.from(blurred.data), { raw: { width, height, channels: 1 } })
    .png().toFile('/tmp/debug-2-blurred.png');
  console.log('Saved: /tmp/debug-2-blurred.png');

  // Stage 3: Canny edges (all thresholds)
  const thresholds = [[50, 150], [75, 200], [30, 100]];
  for (const [lo, hi] of thresholds) {
    const edges = new cv.Mat();
    cv.Canny(blurred, edges, lo, hi, 3, false);
    await sharp(Buffer.from(edges.data), { raw: { width, height, channels: 1 } })
      .png().toFile(`/tmp/debug-3-canny-${lo}-${hi}.png`);
    console.log(`Saved: /tmp/debug-3-canny-${lo}-${hi}.png`);

    // Stage 3b: Dilated
    const dilated = new cv.Mat();
    const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
    cv.dilate(edges, dilated, kernel, new cv.Point(-1, -1), 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());

    // Stage 4: All contours found
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(dilated, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    // Draw all contours on a copy of the original
    const allContoursMat = src.clone();
    const contourInfo = [];
    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area = cv.contourArea(contour);
      const peri = cv.arcLength(contour, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(contour, approx, 0.02 * peri, true);

      // Color: red for 4-point, blue for others
      const color = approx.rows === 4 ? [0, 0, 255, 255] : [255, 0, 0, 255];
      cv.drawContours(allContoursMat, contours, i, color, 2);

      contourInfo.push({
        idx: i,
        area: Math.round(area),
        pctOfImage: (area / (width * height) * 100).toFixed(1),
        approxPoints: approx.rows
      });

      approx.delete();
    }

    await sharp(Buffer.from(allContoursMat.data), { raw: { width, height, channels: 3 } })
      .png().toFile(`/tmp/debug-4-contours-${lo}-${hi}.png`);
    console.log(`Saved: /tmp/debug-4-contours-${lo}-${hi}.png (${contours.size()} contours)`);
    contourInfo.sort((a, b) => b.area - a.area);
    for (const c of contourInfo.slice(0, 5)) {
      console.log(`  contour #${c.idx}: area=${c.area} (${c.pctOfImage}%), approx=${c.approxPoints} pts`);
    }

    // Stage 5: Chosen quadrilateral overlaid
    const imageSize = width * height;
    let bestContour = null;
    let bestArea = 0;
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

    const chosenMat = src.clone();
    if (bestContour) {
      // Draw filled polygon
      const contourVec = new cv.MatVector();
      contourVec.push_back(bestContour);
      cv.fillPoly(chosenMat, contourVec, [0, 255, 0, 50]);
      // Draw outline
      cv.drawContours(chosenMat, contourVec, 0, [0, 255, 0, 255], 3);
      contourVec.delete();

      // Extract and log corners
      const corners = [];
      for (let i = 0; i < bestContour.rows; i++) {
        const x = bestContour.data32S[i * 2];
        const y = bestContour.data32S[i * 2 + 1];
        corners.push([x, y]);
      }
      console.log(`  Chosen quad (${lo}-${hi}): area=${Math.round(bestArea)} (${(bestArea/imageSize*100).toFixed(1)}%), corners=${JSON.stringify(corners)}`);

      // Stage 6: Final crop (bounding box of quadrilateral)
      const xs = corners.map(c => c[0]);
      const ys = corners.map(c => c[1]);
      const left = Math.round(Math.min(...xs));
      const top = Math.round(Math.min(...ys));
      const right = Math.round(Math.max(...xs));
      const bottom = Math.round(Math.max(...ys));

      const scaleX = meta.width / width;
      const scaleY = meta.height / height;

      const origLeft = Math.round(left * scaleX);
      const origTop = Math.round(top * scaleY);
      const origRight = Math.round(right * scaleX);
      const origBottom = Math.round(bottom * scaleY);

      const cropWidth = origRight - origLeft;
      const cropHeight = origBottom - origTop;

      if (cropWidth > 0 && cropHeight > 0) {
        const cropped = await sharp(inputPath).extract({
          left: origLeft, top: origTop, width: cropWidth, height: cropHeight
        }).png().toFile(`/tmp/debug-6-crop-${lo}-${hi}.png`);
        console.log(`Saved: /tmp/debug-6-crop-${lo}-${hi}.png (${cropWidth}x${cropHeight})`);
      }
    } else {
      console.log(`  No quadrilateral found for threshold ${lo}-${hi}`);

      // Show the largest contour regardless of point count
      let largestContour = null;
      let largestArea = 0;
      for (let i = 0; i < contours.size(); i++) {
        const contour = contours.get(i);
        const area = cv.contourArea(contour);
        if (area > largestArea) {
          largestContour = contour;
          largestArea = area;
        }
      }
      if (largestContour) {
        const approx = new cv.Mat();
        const peri = cv.arcLength(largestContour, true);
        // Try different epsilon values
        for (const eps of [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]) {
          cv.approxPolyDP(largestContour, approx, eps * peri, true);
          console.log(`    eps=${eps}: ${approx.rows} points`);
        }
        approx.delete();
      }
    }

    edges.delete(); dilated.delete(); kernel.delete();
    contours.delete(); hierarchy.delete(); allContoursMat.delete();
    chosenMat.delete();
  }

  // Cleanup
  src.delete(); gray.delete(); blurred.delete();
  console.log('\nDone. Check /tmp/debug-*.png');
}

main().catch(e => { console.error(e); process.exit(1); });
