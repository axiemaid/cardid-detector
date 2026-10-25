// lib/crop.js — Beige background removal (when present)
//
// If the image has a beige outer border (table background in phone photos),
// crop it off. If not (clean scan with black/dark background), skip
// cropping — pHash handles those fine as-is.
//
// The brown border ring stays in the crop — it's consistent across
// ALL cards, so pHash treats it as part of the card's fingerprint.

const sharp = require('sharp');

function isBeige(r, g, b) {
  const lum = 0.299 * r + 0.587 * g + 0.114 * b;
  const rg = r - g;
  const gb = g - b;
  return lum > 150 && rg > 12 && gb > 18;
}

function buildBeigeProfile(rgbData, width, height, channels, edge, scanLen) {
  const numSamples = 30;
  const profile = new Float64Array(scanLen);

  const samples = [];
  for (let s = 0; s < numSamples; s++) {
    samples.push(Math.floor((edge === 'left' || edge === 'right' ? height : width) * (0.05 + 0.9 * s / (numSamples - 1))));
  }

  for (const s of samples) {
    for (let i = 0; i < scanLen; i++) {
      let x, y;
      switch (edge) {
        case 'left':   x = i;           y = s;            break;
        case 'right':  x = width-1-i;   y = s;            break;
        case 'top':    x = s;           y = i;            break;
        case 'bottom': x = s;           y = height-1-i;   break;
      }
      const idx = (y * width + x) * channels;
      if (isBeige(rgbData[idx], rgbData[idx + 1], rgbData[idx + 2])) profile[i]++;
    }
  }

  for (let i = 0; i < scanLen; i++) profile[i] /= numSamples;
  return profile;
}

function findBeigeEnd(profile, threshold) {
  let seenBeige = false;

  for (let i = 0; i < profile.length; i++) {
    if (profile[i] > threshold) {
      seenBeige = true;
    } else if (seenBeige) {
      return i;
    }
  }

  return -1;
}

function findCardFaceBounds(rgbData, width, height, channels) {
  const scanDepth = 0.45;
  const scanW = Math.floor(width * scanDepth);
  const scanH = Math.floor(height * scanDepth);
  const threshold = 0.35;

  const leftProfile = buildBeigeProfile(rgbData, width, height, channels, 'left', scanW);
  const rightProfile = buildBeigeProfile(rgbData, width, height, channels, 'right', scanW);
  const topProfile = buildBeigeProfile(rgbData, width, height, channels, 'top', scanH);
  const bottomProfile = buildBeigeProfile(rgbData, width, height, channels, 'bottom', scanH);

  // Check if edges have significant beige (more than 25% at the very edge)
  const edgeSample = Math.max(3, Math.floor(scanW * 0.1));
  const edgeBeige = (profile, n) => {
    let sum = 0;
    for (let i = 0; i < Math.min(n, profile.length); i++) sum += profile[i];
    return sum / Math.min(n, profile.length);
  };

  const leftEdgeBeige = edgeBeige(leftProfile, edgeSample);
  const rightEdgeBeige = edgeBeige(rightProfile, edgeSample);
  const topEdgeBeige = edgeBeige(topProfile, edgeSample);
  const bottomEdgeBeige = edgeBeige(bottomProfile, edgeSample);

  // If no edge has significant beige, skip cropping
  if (leftEdgeBeige < 0.25 && rightEdgeBeige < 0.25 &&
      topEdgeBeige < 0.25 && bottomEdgeBeige < 0.25) {
    return { left: 0, top: 0, right: width, bottom: height, found: false, reason: 'no-beige' };
  }

  const leftEnd = findBeigeEnd(leftProfile, threshold);
  const rightEnd = findBeigeEnd(rightProfile, threshold);
  const topEnd = findBeigeEnd(topProfile, threshold);
  const bottomEnd = findBeigeEnd(bottomProfile, threshold);

  const left = leftEnd > 0 ? leftEnd : 0;
  const top = topEnd > 0 ? topEnd : 0;
  const right = rightEnd > 0 ? (width - 1 - rightEnd) : width - 1;
  const bottom = bottomEnd > 0 ? (height - 1 - bottomEnd) : height - 1;

  const result = {
    left,
    top,
    right,
    bottom,
    found: right > left && bottom > top
  };

  const minW = width * 0.3;
  const minH = height * 0.3;
  if (result.right - result.left < minW || result.bottom - result.top < minH) {
    return { left: 0, top: 0, right: width, bottom: height, found: false, reason: 'too-small' };
  }

  return result;
}

async function cropToCardFace(inputPath) {
  const meta = await sharp(inputPath).metadata();

  const detectW = 400;
  const detectH = Math.round(400 * (meta.height / meta.width));
  const detectBuf = await sharp(inputPath)
    .resize(detectW, detectH)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const bounds = findCardFaceBounds(detectBuf.data, detectW, detectH, detectBuf.info.channels);

  if (!bounds.found) {
    return { cropped: null, bounds, meta, detected: false };
  }

  const scaleX = meta.width / detectW;
  const scaleY = meta.height / detectH;

  const origBounds = {
    left: Math.max(0, Math.round(bounds.left * scaleX)),
    top: Math.max(0, Math.round(bounds.top * scaleY)),
    right: Math.min(meta.width, Math.round(bounds.right * scaleX)),
    bottom: Math.min(meta.height, Math.round(bounds.bottom * scaleY))
  };

  const w = origBounds.right - origBounds.left;
  const h = origBounds.bottom - origBounds.top;

  const cropped = sharp(inputPath).extract({
    left: origBounds.left, top: origBounds.top, width: w, height: h
  });

  return { cropped, bounds: origBounds, meta, detected: true };
}

module.exports = { isBeige, findCardFaceBounds, cropToCardFace };
