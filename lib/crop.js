// lib/crop.js — Color-targeted cropping for Hyper Battle cards
// The card has a consistent structure:
//   - Outer beige/map pattern border
//   - Dark brown border ring inside that
//   - Card face (artwork, stats, text) inside the brown border
// We crop to the dark brown border's inner edge — turning the beige
// outer border into "background" and keeping just the card face.

const sharp = require('sharp');

// Dark brown border color range (from sampling real cards)
// These are the RGB values we see in the brown border ring
const BROWN_MIN = { r: 40, g: 10, b: 5 };
const BROWN_MAX = { r: 160, g: 80, b: 50 };

// Beige/map border color range (outermost)
const BEIGE_MIN = { r: 180, g: 150, b: 80 };
const BEIGE_MAX = { r: 240, g: 210, b: 160 };

/**
 * Check if a pixel is in the brown border range.
 */
function isBrown(r, g, b) {
  return r >= BROWN_MIN.r && r <= BROWN_MAX.r &&
         g >= BROWN_MIN.g && g <= BROWN_MAX.g &&
         b >= BROWN_MIN.b && b <= BROWN_MAX.b;
}

/**
 * Check if a pixel is in the beige/map border range.
 */
function isBeige(r, g, b) {
  return r >= BEIGE_MIN.r && r <= BEIGE_MAX.r &&
         g >= BEIGE_MIN.g && g <= BEIGE_MAX.g &&
         b >= BEIGE_MIN.b && b <= BEIGE_MAX.b;
}

/**
 * Find the inner boundary of the dark brown border.
 * Scans from each edge inward:
 *   1. Skip non-card pixels (black, beige, etc.)
 *   2. Find the brown border band
 *   3. Return the inner edge of the brown band
 * 
 * Returns { left, top, right, bottom } in source pixels.
 */
function findCardFaceBounds(rgbData, width, height, channels) {
  const sampleStep = 2; // sample every 2px for speed
  
  // --- Find left boundary ---
  // Scan left→right across multiple rows, find where brown band ends
  const leftVotes = [];
  for (let y = Math.floor(height * 0.15); y < height * 0.85; y += Math.floor(height * 0.05)) {
    let foundBrown = false;
    let brownEnd = -1;
    for (let x = 0; x < width * 0.3; x += sampleStep) {
      const idx = (y * width + x) * channels;
      const r = rgbData[idx];
      const g = rgbData[idx + 1];
      const b = rgbData[idx + 2];
      
      if (isBrown(r, g, b)) {
        foundBrown = true;
      } else if (foundBrown && !isBrown(r, g, b) && !isBeige(r, g, b)) {
        // We were in brown, now we're past it (not beige either) → card face
        brownEnd = x;
        break;
      }
    }
    if (brownEnd > 0) leftVotes.push(brownEnd);
  }
  
  // --- Find right boundary ---
  const rightVotes = [];
  for (let y = Math.floor(height * 0.15); y < height * 0.85; y += Math.floor(height * 0.05)) {
    let foundBrown = false;
    let brownStart = -1;
    for (let x = width - 1; x > width * 0.7; x -= sampleStep) {
      const idx = (y * width + x) * channels;
      const r = rgbData[idx];
      const g = rgbData[idx + 1];
      const b = rgbData[idx + 2];
      
      if (isBrown(r, g, b)) {
        foundBrown = true;
      } else if (foundBrown && !isBrown(r, g, b) && !isBeige(r, g, b)) {
        brownStart = x;
        break;
      }
    }
    if (brownStart > 0) rightVotes.push(brownStart);
  }
  
  // --- Find top boundary ---
  const topVotes = [];
  for (let x = Math.floor(width * 0.15); x < width * 0.85; x += Math.floor(width * 0.05)) {
    let foundBrown = false;
    let brownEnd = -1;
    for (let y = 0; y < height * 0.3; y += sampleStep) {
      const idx = (y * width + x) * channels;
      const r = rgbData[idx];
      const g = rgbData[idx + 1];
      const b = rgbData[idx + 2];
      
      if (isBrown(r, g, b)) {
        foundBrown = true;
      } else if (foundBrown && !isBrown(r, g, b) && !isBeige(r, g, b)) {
        brownEnd = y;
        break;
      }
    }
    if (brownEnd > 0) topVotes.push(brownEnd);
  }
  
  // --- Find bottom boundary ---
  const bottomVotes = [];
  for (let x = Math.floor(width * 0.15); x < width * 0.85; x += Math.floor(width * 0.05)) {
    let foundBrown = false;
    let brownStart = -1;
    for (let y = height - 1; y > height * 0.7; y -= sampleStep) {
      const idx = (y * width + x) * channels;
      const r = rgbData[idx];
      const g = rgbData[idx + 1];
      const b = rgbData[idx + 2];
      
      if (isBrown(r, g, b)) {
        foundBrown = true;
      } else if (foundBrown && !isBrown(r, g, b) && !isBeige(r, g, b)) {
        brownStart = y;
        break;
      }
    }
    if (brownStart > 0) bottomVotes.push(brownStart);
  }
  
  // Take median of votes (robust to outliers)
  const median = (arr) => {
    if (arr.length === 0) return -1;
    const sorted = arr.sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  };
  
  const left = median(leftVotes);
  const top = median(topVotes);
  const right = median(rightVotes);
  const bottom = median(bottomVotes);
  
  // Fallback: if we didn't find brown borders, use full image
  const result = {
    left: left > 0 ? left : 0,
    top: top > 0 ? top : 0,
    right: right > 0 ? right : width - 1,
    bottom: bottom > 0 ? bottom : height - 1,
    found: left > 0 && top > 0 && right > 0 && bottom > 0
  };
  
  // Sanity check: cropped region should be at least 40% of original
  const minW = width * 0.4;
  const minH = height * 0.4;
  if (result.right - result.left < minW || result.bottom - result.top < minH) {
    return { left: 0, top: 0, right: width, bottom: height, found: false };
  }
  
  return result;
}

/**
 * Crop a card image to the inner brown border.
 * Returns the cropped image as a sharp pipeline.
 */
async function cropToCardFace(inputPath) {
  const meta = await sharp(inputPath).metadata();
  
  // Downscale for border detection (400px wide for detail)
  const detectW = 400;
  const detectH = Math.round(400 * (meta.height / meta.width));
  const detectBuf = await sharp(inputPath)
    .resize(detectW, detectH)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  
  const bounds = findCardFaceBounds(detectBuf.data, detectW, detectH, detectBuf.info.channels);
  
  if (!bounds.found) {
    // No brown border detected — return original, let pHash handle it
    return { cropped: null, bounds, meta };
  }
  
  // Scale bounds back to original coordinates
  const scaleX = meta.width / detectW;
  const scaleY = meta.height / detectH;
  
  const origBounds = {
    left: Math.max(0, Math.round(bounds.left * scaleX)),
    top: Math.max(0, Math.round(bounds.top * scaleY)),
    right: Math.min(meta.width, Math.round(bounds.right * scaleX)),
    bottom: Math.min(meta.height, Math.round(bounds.bottom * scaleY))
  };
  
  const width = origBounds.right - origBounds.left;
  const height = origBounds.bottom - origBounds.top;
  
  const cropped = sharp(inputPath).extract({
    left: origBounds.left,
    top: origBounds.top,
    width,
    height
  });
  
  return { cropped, bounds: origBounds, meta, detected: true };
}

module.exports = {
  isBrown,
  isBeige,
  findCardFaceBounds,
  cropToCardFace,
  BROWN_MIN,
  BROWN_MAX,
  BEIGE_MIN,
  BEIGE_MAX
};
