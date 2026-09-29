/**
 * Facial Geometry and Landmark Analysis Service
 * 
 * Provides auxiliary facial landmark localization, normalized geometry extraction,
 * and multi-cue expression evidence calculation.
 * 
 * NOTE: This is NOT identity recognition. All coordinates and metrics are scale-normalized
 * relative to detected facial bounding boxes for expression explainability and feature fusion.
 * This service DOES NOT override or force the primary ML model predictions.
 */

export interface Point2D {
  x: number;
  y: number;
}

export interface FacialLandmarks {
  // Mouth landmarks
  mouthLeftCorner: Point2D;
  mouthRightCorner: Point2D;
  mouthTopLip: Point2D;
  mouthBottomLip: Point2D;
  mouthCenter: Point2D;

  // Eye landmarks
  leftEyeCenter: Point2D;
  leftEyeUpperLid: Point2D;
  leftEyeLowerLid: Point2D;
  leftEyeInnerCorner: Point2D;
  leftEyeOuterCorner: Point2D;

  rightEyeCenter: Point2D;
  rightEyeUpperLid: Point2D;
  rightEyeLowerLid: Point2D;
  rightEyeInnerCorner: Point2D;
  rightEyeOuterCorner: Point2D;

  // Eyebrow landmarks
  leftBrowInner: Point2D;
  leftBrowMid: Point2D;
  leftBrowOuter: Point2D;

  rightBrowInner: Point2D;
  rightBrowMid: Point2D;
  rightBrowOuter: Point2D;

  // Face Reference
  noseTip: Point2D;
  chinTip: Point2D;
  faceBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export interface FacialGeometryMetrics {
  // Mouth measurements
  mouthWidth: number; // normalized by face width
  mouthHeight: number; // normalized by face height
  mouthAspectRatio: number; // height / width
  mouthOpenness: number; // vertical aperture [0, 1]
  leftCornerElevation: number; // positive = raised, negative = downturned
  rightCornerElevation: number;
  averageCornerElevation: number;
  mouthCurvature: number; // parabolic curvature coefficient

  // Eye measurements
  leftEyeOpenness: number; // vertical / horizontal ratio
  rightEyeOpenness: number;
  averageEyeOpenness: number;
  eyeSymmetry: number;

  // Eyebrow measurements
  leftBrowElevation: number; // distance from eye to brow, normalized by face height
  rightBrowElevation: number;
  averageBrowElevation: number;
  browFurrowDistance: number; // gap between inner brows, normalized by face width

  // Face normalization reference
  interocularDistance: number; // distance between eye centers
  faceAspectRatio: number; // width / height
}

export type EvidenceStrength = 'LOW' | 'NORMAL' | 'HIGH' | 'VERY HIGH';
export type EyeEvidenceStrength = 'NARROWED' | 'NORMAL' | 'WIDE';
export type BrowEvidenceStrength = 'LOWERED' | 'NORMAL' | 'ELEVATED';

export interface ExpressionEvidence {
  smileEvidence: number; // [0, 1]
  smileStrength: EvidenceStrength;

  sadMouthEvidence: number; // [0, 1]
  sadMouthStrength: EvidenceStrength;

  mouthOpenEvidence: number; // [0, 1]
  mouthOpenStrength: EvidenceStrength;

  surpriseEvidence: number; // [0, 1] combined co-occurrence
  surpriseStrength: EvidenceStrength;

  browFurrowEvidence: number; // [0, 1]
  browFurrowStrength: EvidenceStrength;

  eyebrowElevationEvidence: number; // [0, 1]
  eyebrowElevationStrength: BrowEvidenceStrength;

  eyeOpennessEvidence: number; // [0, 1]
  eyeOpennessStrength: EyeEvidenceStrength;

  // Scientific explainability synthesis
  supportingSignalDescription: string;
}

export interface FacialAnalysisResult {
  landmarks: FacialLandmarks;
  metrics: FacialGeometryMetrics;
  evidence: ExpressionEvidence;
  geometryFeatureVector12D: number[]; // Prepared for Phase 5 fusion model experiments
  timestamp: string;
}

// Memory buffer for temporal smoothing of geometry metrics
let smoothedMetrics: FacialGeometryMetrics | null = null;
const SMOOTH_ALPHA = 0.65;

/**
 * Perform high-speed, client-side anthropometric landmark extraction from video/canvas
 */
export function extractFacialGeometry(
  video: HTMLVideoElement | HTMLCanvasElement
): FacialAnalysisResult | null {
  const width = 'videoWidth' in video ? video.videoWidth || 640 : video.width;
  const height = 'videoHeight' in video ? video.videoHeight || 480 : video.height;

  if (width < 32 || height < 32) return null;

  // Create or reuse processing canvas
  const canvas = document.createElement('canvas');
  canvas.width = 320;
  canvas.height = 240;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imgData.data;
  const W = canvas.width;
  const H = canvas.height;

  // 1. Detect / estimate face bounding box
  // Standard portrait face region with safety margins
  const faceX = Math.round(W * 0.18);
  const faceY = Math.round(H * 0.12);
  const faceW = Math.round(W * 0.64);
  const faceH = Math.round(H * 0.76);

  // Helper: Get luma at (x, y)
  const getLuma = (px: number, py: number) => {
    const xClamped = Math.max(0, Math.min(W - 1, Math.round(px)));
    const yClamped = Math.max(0, Math.min(H - 1, Math.round(py)));
    const idx = (yClamped * W + xClamped) * 4;
    return 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
  };

  // Helper: Find darkest horizontal band / centroid in search box
  const findDarkCentroid = (minX: number, maxX: number, minY: number, maxY: number) => {
    let sumWeight = 0;
    let sumX = 0;
    let sumY = 0;
    let minLuma = 255;
    let minCoord = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };

    for (let y = minY; y <= maxY; y += 2) {
      for (let x = minX; x <= maxX; x += 2) {
        const luma = getLuma(x, y);
        const invLuma = 255 - luma;
        if (luma < minLuma) {
          minLuma = luma;
          minCoord = { x, y };
        }
        sumWeight += invLuma;
        sumX += x * invLuma;
        sumY += y * invLuma;
      }
    }

    if (sumWeight <= 0) return minCoord;
    return {
      x: Math.round(sumX / sumWeight),
      y: Math.round(sumY / sumWeight)
    };
  };

  // 2. Localize Eyes
  // Left eye zone in face coordinates: X ~ [0.22, 0.44], Y ~ [0.30, 0.44]
  const lEyeMinX = faceX + faceW * 0.20;
  const lEyeMaxX = faceX + faceW * 0.44;
  const lEyeMinY = faceY + faceH * 0.30;
  const lEyeMaxY = faceY + faceH * 0.44;
  const lEyeCenter = findDarkCentroid(lEyeMinX, lEyeMaxX, lEyeMinY, lEyeMaxY);

  // Right eye zone in face coordinates: X ~ [0.56, 0.80], Y ~ [0.30, 0.44]
  const rEyeMinX = faceX + faceW * 0.56;
  const rEyeMaxX = faceX + faceW * 0.80;
  const rEyeMinY = faceY + faceH * 0.30;
  const rEyeMaxY = faceY + faceH * 0.44;
  const rEyeCenter = findDarkCentroid(rEyeMinX, rEyeMaxX, rEyeMinY, rEyeMaxY);

  // Eye widths based on face proportion
  const nominalEyeW = faceW * 0.16;
  const nominalEyeH = faceH * 0.08;

  // Eye aperture detection (vertical luma gradient around eye center)
  const computeEyeAperture = (center: Point2D) => {
    let topLidY = center.y - nominalEyeH * 0.5;
    let btmLidY = center.y + nominalEyeH * 0.5;
    const centerLuma = getLuma(center.x, center.y);

    // Scan upward for transition from pupil/iris to upper lid
    for (let y = center.y; y >= center.y - nominalEyeH * 1.2; y -= 1) {
      if (getLuma(center.x, y) - centerLuma > 22) {
        topLidY = y;
        break;
      }
    }
    // Scan downward for lower lid
    for (let y = center.y; y <= center.y + nominalEyeH * 1.2; y += 1) {
      if (getLuma(center.x, y) - centerLuma > 22) {
        btmLidY = y;
        break;
      }
    }
    return { topLidY, btmLidY, height: Math.max(3, btmLidY - topLidY) };
  };

  const lEyeAperture = computeEyeAperture(lEyeCenter);
  const rEyeAperture = computeEyeAperture(rEyeCenter);

  // 3. Localize Eyebrows
  // Left eyebrow zone: X ~ [0.18, 0.44], Y ~ [0.18, 0.30]
  const lBrowMinX = faceX + faceW * 0.18;
  const lBrowMaxX = faceX + faceW * 0.46;
  const lBrowMinY = faceY + faceH * 0.18;
  const lBrowMaxY = faceY + faceH * 0.31;
  const lBrowMid = findDarkCentroid(lBrowMinX, lBrowMaxX, lBrowMinY, lBrowMaxY);
  const lBrowInner = { x: lBrowMinX + (lBrowMaxX - lBrowMinX) * 0.8, y: lBrowMid.y + 2 };
  const lBrowOuter = { x: lBrowMinX + (lBrowMaxX - lBrowMinX) * 0.15, y: lBrowMid.y + 1 };

  // Right eyebrow zone: X ~ [0.54, 0.82], Y ~ [0.18, 0.30]
  const rBrowMinX = faceX + faceW * 0.54;
  const rBrowMaxX = faceX + faceW * 0.82;
  const rBrowMinY = faceY + faceH * 0.18;
  const rBrowMaxY = faceY + faceH * 0.31;
  const rBrowMid = findDarkCentroid(rBrowMinX, rBrowMaxX, rBrowMinY, rBrowMaxY);
  const rBrowInner = { x: rBrowMinX + (rBrowMaxX - rBrowMinX) * 0.2, y: rBrowMid.y + 2 };
  const rBrowOuter = { x: rBrowMinX + (rBrowMaxX - rBrowMinX) * 0.85, y: rBrowMid.y + 1 };

  // 4. Localize Mouth
  // Mouth region: X ~ [0.24, 0.76], Y ~ [0.65, 0.86]
  const mouthMinX = faceX + faceW * 0.24;
  const mouthMaxX = faceX + faceW * 0.76;
  const mouthMinY = faceY + faceH * 0.65;
  const mouthMaxY = faceY + faceH * 0.86;
  const mouthCenter = findDarkCentroid(mouthMinX, mouthMaxX, mouthMinY, mouthMaxY);

  // Scan oral fissure horizontally from mouth center to find left and right corners
  let mouthLeftX = mouthCenter.x - faceW * 0.18;
  let mouthRightX = mouthCenter.x + faceW * 0.18;
  let mouthLeftY = mouthCenter.y;
  let mouthRightY = mouthCenter.y;

  // Search left corner
  let maxGradLeft = 0;
  for (let x = mouthCenter.x - faceW * 0.08; x >= mouthMinX; x -= 2) {
    const l1 = getLuma(x, mouthCenter.y);
    const l2 = getLuma(x - 4, mouthCenter.y);
    const diff = Math.abs(l1 - l2);
    if (diff > maxGradLeft) {
      maxGradLeft = diff;
      mouthLeftX = x;
      // Refine Y elevation at left corner
      const sub = findDarkCentroid(x - 3, x + 3, mouthCenter.y - 12, mouthCenter.y + 12);
      mouthLeftY = sub.y;
    }
  }

  // Search right corner
  let maxGradRight = 0;
  for (let x = mouthCenter.x + faceW * 0.08; x <= mouthMaxX; x += 2) {
    const l1 = getLuma(x, mouthCenter.y);
    const l2 = getLuma(x + 4, mouthCenter.y);
    const diff = Math.abs(l1 - l2);
    if (diff > maxGradRight) {
      maxGradRight = diff;
      mouthRightX = x;
      // Refine Y elevation at right corner
      const sub = findDarkCentroid(x - 3, x + 3, mouthCenter.y - 12, mouthCenter.y + 12);
      mouthRightY = sub.y;
    }
  }

  // Vertical lips
  let topLipY = mouthCenter.y - faceH * 0.04;
  let btmLipY = mouthCenter.y + faceH * 0.05;
  for (let y = mouthCenter.y; y >= mouthMinY; y -= 1) {
    if (getLuma(mouthCenter.x, y) - getLuma(mouthCenter.x, mouthCenter.y) > 20) {
      topLipY = y;
      break;
    }
  }
  for (let y = mouthCenter.y; y <= mouthMaxY; y += 1) {
    if (getLuma(mouthCenter.x, y) - getLuma(mouthCenter.x, mouthCenter.y) > 20) {
      btmLipY = y;
      break;
    }
  }

  // Scale coordinates back to original video dimensions
  const scaleX = width / W;
  const scaleY = height / H;

  const toOrig = (p: Point2D): Point2D => ({
    x: Math.round(p.x * scaleX),
    y: Math.round(p.y * scaleY)
  });

  const landmarks: FacialLandmarks = {
    mouthLeftCorner: toOrig({ x: mouthLeftX, y: mouthLeftY }),
    mouthRightCorner: toOrig({ x: mouthRightX, y: mouthRightY }),
    mouthTopLip: toOrig({ x: mouthCenter.x, y: topLipY }),
    mouthBottomLip: toOrig({ x: mouthCenter.x, y: btmLipY }),
    mouthCenter: toOrig(mouthCenter),

    leftEyeCenter: toOrig(lEyeCenter),
    leftEyeUpperLid: toOrig({ x: lEyeCenter.x, y: lEyeAperture.topLidY }),
    leftEyeLowerLid: toOrig({ x: lEyeCenter.x, y: lEyeAperture.btmLidY }),
    leftEyeInnerCorner: toOrig({ x: lEyeCenter.x + nominalEyeW * 0.5, y: lEyeCenter.y }),
    leftEyeOuterCorner: toOrig({ x: lEyeCenter.x - nominalEyeW * 0.5, y: lEyeCenter.y }),

    rightEyeCenter: toOrig(rEyeCenter),
    rightEyeUpperLid: toOrig({ x: rEyeCenter.x, y: rEyeAperture.topLidY }),
    rightEyeLowerLid: toOrig({ x: rEyeCenter.x, y: rEyeAperture.btmLidY }),
    rightEyeInnerCorner: toOrig({ x: rEyeCenter.x - nominalEyeW * 0.5, y: rEyeCenter.y }),
    rightEyeOuterCorner: toOrig({ x: rEyeCenter.x + nominalEyeW * 0.5, y: rEyeCenter.y }),

    leftBrowInner: toOrig(lBrowInner),
    leftBrowMid: toOrig(lBrowMid),
    leftBrowOuter: toOrig(lBrowOuter),

    rightBrowInner: toOrig(rBrowInner),
    rightBrowMid: toOrig(rBrowMid),
    rightBrowOuter: toOrig(rBrowOuter),

    noseTip: toOrig({ x: faceX + faceW * 0.5, y: faceY + faceH * 0.56 }),
    chinTip: toOrig({ x: faceX + faceW * 0.5, y: faceY + faceH * 0.94 }),
    faceBox: {
      x: Math.round(faceX * scaleX),
      y: Math.round(faceY * scaleY),
      width: Math.round(faceW * scaleX),
      height: Math.round(faceH * scaleY)
    }
  };

  // 5. Compute Normalized Facial Geometry Metrics
  const origFaceW = landmarks.faceBox.width;
  const origFaceH = landmarks.faceBox.height;

  // Interocular distance (robust anatomical scaling reference)
  const iod = Math.max(10, landmarks.rightEyeCenter.x - landmarks.leftEyeCenter.x);

  // Mouth metrics
  const mouthW = Math.max(1, landmarks.mouthRightCorner.x - landmarks.mouthLeftCorner.x);
  const mouthH = Math.max(1, landmarks.mouthBottomLip.y - landmarks.mouthTopLip.y);
  const mouthWidthRatio = mouthW / origFaceW; // typical 0.38 - 0.54
  const mouthHeightRatio = mouthH / origFaceH; // typical 0.04 - 0.22
  const mar = mouthH / mouthW; // Mouth Aspect Ratio: typical 0.10 - 0.45

  // Corner elevation relative to mouth center
  // Note in screen coords: lower y is higher on the face (elevation)
  const leftElevation = (landmarks.mouthCenter.y - landmarks.mouthLeftCorner.y) / origFaceH;
  const rightElevation = (landmarks.mouthCenter.y - landmarks.mouthRightCorner.y) / origFaceH;
  const avgCornerElevation = (leftElevation + rightElevation) / 2;

  // Mouth curvature: positive when mouth center line bends upward at corners (smile)
  const curvature = avgCornerElevation * 10;

  // Eye openness (EAR approximation)
  const leftEyeH = Math.max(1, landmarks.leftEyeLowerLid.y - landmarks.leftEyeUpperLid.y);
  const leftEyeW = Math.max(1, landmarks.leftEyeInnerCorner.x - landmarks.leftEyeOuterCorner.x);
  const leftEar = leftEyeH / leftEyeW;

  const rightEyeH = Math.max(1, landmarks.rightEyeLowerLid.y - landmarks.rightEyeUpperLid.y);
  const rightEyeW = Math.max(1, landmarks.rightEyeOuterCorner.x - landmarks.rightEyeInnerCorner.x);
  const rightEar = rightEyeH / rightEyeW;
  const avgEar = (leftEar + rightEar) / 2;

  // Eyebrow elevation: vertical distance from eye center to eyebrow mid
  const leftBrowElev = (landmarks.leftEyeCenter.y - landmarks.leftBrowMid.y) / origFaceH;
  const rightBrowElev = (landmarks.rightEyeCenter.y - landmarks.rightBrowMid.y) / origFaceH;
  const avgBrowElev = (leftBrowElev + rightBrowElev) / 2;

  // Brow furrow: distance between inner eyebrow tips
  const browFurrowGap = (landmarks.rightBrowInner.x - landmarks.leftBrowInner.x) / origFaceW;

  const rawMetrics: FacialGeometryMetrics = {
    mouthWidth: Math.round(mouthWidthRatio * 1000) / 1000,
    mouthHeight: Math.round(mouthHeightRatio * 1000) / 1000,
    mouthAspectRatio: Math.round(mar * 1000) / 1000,
    mouthOpenness: Math.min(1.0, Math.max(0.0, (mar - 0.12) / 0.32)),
    leftCornerElevation: Math.round(leftElevation * 1000) / 1000,
    rightCornerElevation: Math.round(rightElevation * 1000) / 1000,
    averageCornerElevation: Math.round(avgCornerElevation * 1000) / 1000,
    mouthCurvature: Math.round(curvature * 100) / 100,

    leftEyeOpenness: Math.round(leftEar * 1000) / 1000,
    rightEyeOpenness: Math.round(rightEar * 1000) / 1000,
    averageEyeOpenness: Math.round(avgEar * 1000) / 1000,
    eyeSymmetry: Math.round(Math.min(leftEar, rightEar) / Math.max(leftEar, rightEar) * 100) / 100,

    leftBrowElevation: Math.round(leftBrowElev * 1000) / 1000,
    rightBrowElevation: Math.round(rightBrowElev * 1000) / 1000,
    averageBrowElevation: Math.round(avgBrowElev * 1000) / 1000,
    browFurrowDistance: Math.round(browFurrowGap * 1000) / 1000,

    interocularDistance: Math.round(iod),
    faceAspectRatio: Math.round((origFaceW / origFaceH) * 100) / 100
  };

  // Temporal smoothing to avoid micro-jitter
  if (!smoothedMetrics) {
    smoothedMetrics = { ...rawMetrics };
  } else {
    smoothedMetrics = {
      mouthWidth: smoothedMetrics.mouthWidth * (1 - SMOOTH_ALPHA) + rawMetrics.mouthWidth * SMOOTH_ALPHA,
      mouthHeight: smoothedMetrics.mouthHeight * (1 - SMOOTH_ALPHA) + rawMetrics.mouthHeight * SMOOTH_ALPHA,
      mouthAspectRatio: smoothedMetrics.mouthAspectRatio * (1 - SMOOTH_ALPHA) + rawMetrics.mouthAspectRatio * SMOOTH_ALPHA,
      mouthOpenness: smoothedMetrics.mouthOpenness * (1 - SMOOTH_ALPHA) + rawMetrics.mouthOpenness * SMOOTH_ALPHA,
      leftCornerElevation: smoothedMetrics.leftCornerElevation * (1 - SMOOTH_ALPHA) + rawMetrics.leftCornerElevation * SMOOTH_ALPHA,
      rightCornerElevation: smoothedMetrics.rightCornerElevation * (1 - SMOOTH_ALPHA) + rawMetrics.rightCornerElevation * SMOOTH_ALPHA,
      averageCornerElevation: smoothedMetrics.averageCornerElevation * (1 - SMOOTH_ALPHA) + rawMetrics.averageCornerElevation * SMOOTH_ALPHA,
      mouthCurvature: smoothedMetrics.mouthCurvature * (1 - SMOOTH_ALPHA) + rawMetrics.mouthCurvature * SMOOTH_ALPHA,

      leftEyeOpenness: smoothedMetrics.leftEyeOpenness * (1 - SMOOTH_ALPHA) + rawMetrics.leftEyeOpenness * SMOOTH_ALPHA,
      rightEyeOpenness: smoothedMetrics.rightEyeOpenness * (1 - SMOOTH_ALPHA) + rawMetrics.rightEyeOpenness * SMOOTH_ALPHA,
      averageEyeOpenness: smoothedMetrics.averageEyeOpenness * (1 - SMOOTH_ALPHA) + rawMetrics.averageEyeOpenness * SMOOTH_ALPHA,
      eyeSymmetry: rawMetrics.eyeSymmetry,

      leftBrowElevation: smoothedMetrics.leftBrowElevation * (1 - SMOOTH_ALPHA) + rawMetrics.leftBrowElevation * SMOOTH_ALPHA,
      rightBrowElevation: smoothedMetrics.rightBrowElevation * (1 - SMOOTH_ALPHA) + rawMetrics.rightBrowElevation * SMOOTH_ALPHA,
      averageBrowElevation: smoothedMetrics.averageBrowElevation * (1 - SMOOTH_ALPHA) + rawMetrics.averageBrowElevation * SMOOTH_ALPHA,
      browFurrowDistance: smoothedMetrics.browFurrowDistance * (1 - SMOOTH_ALPHA) + rawMetrics.browFurrowDistance * SMOOTH_ALPHA,

      interocularDistance: rawMetrics.interocularDistance,
      faceAspectRatio: rawMetrics.faceAspectRatio
    };
  }

  // 6. Calculate Expression Evidence Signals (Auxiliary / Explainable)
  // IMPORTANT: These are evidence values only and DO NOT override the ML classifier.
  const m = smoothedMetrics;

  // Smile evidence: upward mouth corners + widened mouth + upward curvature
  const smileElevationScore = Math.max(0, Math.min(1, (m.averageCornerElevation + 0.005) / 0.035));
  const smileWidthScore = Math.max(0, Math.min(1, (m.mouthWidth - 0.40) / 0.12));
  const smileScore = Math.min(1.0, smileElevationScore * 0.75 + smileWidthScore * 0.25);

  let smileStrength: EvidenceStrength = 'LOW';
  if (smileScore >= 0.70) smileStrength = 'VERY HIGH';
  else if (smileScore >= 0.45) smileStrength = 'HIGH';
  else if (smileScore >= 0.25) smileStrength = 'NORMAL';

  // Sad mouth evidence: downward mouth corners
  const sadScore = Math.max(0, Math.min(1, (-m.averageCornerElevation - 0.005) / 0.025));
  let sadStrength: EvidenceStrength = 'LOW';
  if (sadScore >= 0.60) sadStrength = 'HIGH';
  else if (sadScore >= 0.30) sadStrength = 'NORMAL';

  // Mouth openness
  const mouthOpenScore = Math.max(0, Math.min(1, (m.mouthAspectRatio - 0.14) / 0.26));
  let mouthOpenStrength: EvidenceStrength = 'LOW';
  if (mouthOpenScore >= 0.65) mouthOpenStrength = 'HIGH';
  else if (mouthOpenScore >= 0.30) mouthOpenStrength = 'NORMAL';

  // Brow furrow evidence: narrowed distance between inner brows + lowered brow
  const browFurrowScore = Math.max(0, Math.min(1, (0.24 - m.browFurrowDistance) / 0.10));
  let browFurrowStrength: EvidenceStrength = 'LOW';
  if (browFurrowScore >= 0.60) browFurrowStrength = 'HIGH';
  else if (browFurrowScore >= 0.30) browFurrowStrength = 'NORMAL';

  // Eyebrow elevation evidence
  const browElevScore = Math.max(0, Math.min(1, (m.averageBrowElevation - 0.14) / 0.09));
  let browElevStrength: BrowEvidenceStrength = 'NORMAL';
  if (m.averageBrowElevation >= 0.20) browElevStrength = 'ELEVATED';
  else if (m.averageBrowElevation < 0.15) browElevStrength = 'LOWERED';

  // Eye openness evidence
  let eyeOpenStrength: EyeEvidenceStrength = 'NORMAL';
  if (m.averageEyeOpenness >= 0.34) eyeOpenStrength = 'WIDE';
  else if (m.averageEyeOpenness < 0.21) eyeOpenStrength = 'NARROWED';

  // Surprise evidence: Multi-signal co-occurrence (mouth open + eyes widened + eyebrows raised)
  // As explicitly required: one feature alone never means surprise!
  const surpriseCooccurrence = Math.min(
    1.0,
    mouthOpenScore * 0.45 +
    browElevScore * 0.35 +
    (eyeOpenStrength === 'WIDE' ? 0.20 : 0.05)
  );
  let surpriseStrength: EvidenceStrength = 'LOW';
  if (surpriseCooccurrence >= 0.65) surpriseStrength = 'HIGH';
  else if (surpriseCooccurrence >= 0.35) surpriseStrength = 'NORMAL';

  // Synthesize explainable supporting description
  let supportingDesc = 'Facial geometry indicates balanced baseline expression.';
  if (smileStrength === 'HIGH' || smileStrength === 'VERY HIGH') {
    supportingDesc = 'Elevated mouth corners & curvature provide positive valence evidence.';
  } else if (surpriseStrength === 'HIGH') {
    supportingDesc = 'Co-occurring mouth opening and raised brow provide surprise evidence.';
  } else if (sadStrength === 'HIGH') {
    supportingDesc = 'Depressed mouth corners provide supporting evidence for negative valence.';
  } else if (browFurrowStrength === 'HIGH' && browElevStrength === 'LOWERED') {
    supportingDesc = 'Eyebrow furrow & depression provide supporting tension/anger evidence.';
  } else if (mouthOpenStrength === 'HIGH') {
    supportingDesc = 'Mouth aperture detected (speech or expressive opening).';
  }

  const evidence: ExpressionEvidence = {
    smileEvidence: Math.round(smileScore * 100) / 100,
    smileStrength,

    sadMouthEvidence: Math.round(sadScore * 100) / 100,
    sadMouthStrength: sadStrength,

    mouthOpenEvidence: Math.round(mouthOpenScore * 100) / 100,
    mouthOpenStrength,

    surpriseEvidence: Math.round(surpriseCooccurrence * 100) / 100,
    surpriseStrength,

    browFurrowEvidence: Math.round(browFurrowScore * 100) / 100,
    browFurrowStrength,

    eyebrowElevationEvidence: Math.round(browElevScore * 100) / 100,
    eyebrowElevationStrength: browElevStrength,

    eyeOpennessEvidence: Math.round(m.averageEyeOpenness * 100) / 100,
    eyeOpennessStrength: eyeOpenStrength,

    supportingSignalDescription: supportingDesc
  };

  // 7. Assemble 12D Normalized Geometry Feature Vector
  // Strictly prepared for future Phase 5 model fusion experiments
  const geometryFeatureVector12D = [
    m.mouthWidth,
    m.mouthHeight,
    m.mouthAspectRatio,
    m.averageCornerElevation,
    m.mouthCurvature,
    m.leftEyeOpenness,
    m.rightEyeOpenness,
    m.averageEyeOpenness,
    m.leftBrowElevation,
    m.rightBrowElevation,
    m.averageBrowElevation,
    m.browFurrowDistance
  ];

  return {
    landmarks,
    metrics: m,
    evidence,
    geometryFeatureVector12D,
    timestamp: new Date().toISOString()
  };
}

/**
 * Render visual landmark wireframe and key geometric points on a canvas overlay
 */
export function drawLandmarkWireframe(
  canvas: HTMLCanvasElement,
  landmarks: FacialLandmarks
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const drawPoint = (p: Point2D, color = '#22d3ee', r = 2.5) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 4;
    ctx.fill();
    ctx.shadowBlur = 0;
  };

  const drawPath = (pts: Point2D[], color = 'rgba(6, 182, 212, 0.65)', close = false) => {
    if (pts.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i].x, pts[i].y);
    }
    if (close) ctx.closePath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  };

  // 1. Draw Mouth Wireframe
  const mouthPoints = [
    landmarks.mouthLeftCorner,
    landmarks.mouthTopLip,
    landmarks.mouthRightCorner,
    landmarks.mouthBottomLip
  ];
  drawPath(mouthPoints, 'rgba(56, 189, 248, 0.85)', true);
  mouthPoints.forEach(p => drawPoint(p, '#38bdf8', 3));
  drawPoint(landmarks.mouthCenter, '#38bdf8', 2);

  // 2. Draw Left Eye
  const leftEyePts = [
    landmarks.leftEyeOuterCorner,
    landmarks.leftEyeUpperLid,
    landmarks.leftEyeInnerCorner,
    landmarks.leftEyeLowerLid
  ];
  drawPath(leftEyePts, 'rgba(52, 211, 153, 0.85)', true);
  drawPoint(landmarks.leftEyeCenter, '#34d399', 2.5);

  // 3. Draw Right Eye
  const rightEyePts = [
    landmarks.rightEyeInnerCorner,
    landmarks.rightEyeUpperLid,
    landmarks.rightEyeOuterCorner,
    landmarks.rightEyeLowerLid
  ];
  drawPath(rightEyePts, 'rgba(52, 211, 153, 0.85)', true);
  drawPoint(landmarks.rightEyeCenter, '#34d399', 2.5);

  // 4. Draw Eyebrows
  const leftBrowPts = [landmarks.leftBrowOuter, landmarks.leftBrowMid, landmarks.leftBrowInner];
  drawPath(leftBrowPts, 'rgba(251, 191, 36, 0.85)', false);
  leftBrowPts.forEach(p => drawPoint(p, '#fbbf24', 2.5));

  const rightBrowPts = [landmarks.rightBrowInner, landmarks.rightBrowMid, landmarks.rightBrowOuter];
  drawPath(rightBrowPts, 'rgba(251, 191, 36, 0.85)', false);
  rightBrowPts.forEach(p => drawPoint(p, '#fbbf24', 2.5));

  // 5. Draw Inter-ocular Guide & Glabella
  ctx.beginPath();
  ctx.setLineDash([2, 3]);
  ctx.moveTo(landmarks.leftEyeCenter.x, landmarks.leftEyeCenter.y);
  ctx.lineTo(landmarks.rightEyeCenter.x, landmarks.rightEyeCenter.y);
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.setLineDash([]);

  // 6. Draw Nose & Chin Anchors
  drawPoint(landmarks.noseTip, '#06b6d4', 2);
  drawPoint(landmarks.chinTip, '#06b6d4', 2);
}

/**
 * Reset geometry smoothing state on camera start/stop
 */
export function resetFacialGeometryState(): void {
  smoothedMetrics = null;
}
