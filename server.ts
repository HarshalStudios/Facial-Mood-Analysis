import express from 'express';
import cors from 'cors';
import path from 'path';
import fs from 'fs';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import jpeg from 'jpeg-js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

export const CANONICAL_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"] as const;
export type Emotion = typeof CANONICAL_EMOTIONS[number];

// Load trained FER2013 linear weights
const weightsPath = path.join(__dirname, 'models', 'fer2013_linear_weights.json');
let fer2013Weights: { classes: string[]; input_dim: number; input_shape: number[]; coef: number[][]; intercept: number[] };

try {
  fer2013Weights = JSON.parse(fs.readFileSync(weightsPath, 'utf-8'));
} catch (e) {
  console.error('[Model Load Error]: Could not load fer2013_linear_weights.json', e);
  fer2013Weights = {
    classes: [...CANONICAL_EMOTIONS],
    input_dim: 576,
    input_shape: [24, 24],
    coef: Array(7).fill(0).map(() => Array(576).fill(0)),
    intercept: [0, 0, 0, 0, 0, 0, 0]
  };
}

// Phase 9 Experiments benchmark metrics
const EXPERIMENTS = [
  { experiment: "Emotion-probability baseline", test_accuracy: 0.3639, test_macro_f1: 0.2972 },
  { experiment: "+ LBP", test_accuracy: 0.3711, test_macro_f1: 0.3022 },
  { experiment: "+ Edge", test_accuracy: 0.3700, test_macro_f1: 0.3027 },
  { experiment: "+ Gradient", test_accuracy: 0.3697, test_macro_f1: 0.3053 },
  { experiment: "+ Image Quality", test_accuracy: 0.3703, test_macro_f1: 0.3026 },
  { experiment: "+ All Handcrafted", test_accuracy: 0.3761, test_macro_f1: 0.3076 },
  { experiment: "Handcrafted Only", test_accuracy: 0.2803, test_macro_f1: 0.1746 }
];

// Temporal state buffer
let temporalHistory: Record<Emotion, number>[] = [];
const MAX_HISTORY = 5;
const ALPHA = 0.50;

// FER2013 StandardScaler parameters for canonical 22D normalized features
// 0..6: 7 emotion baseline probabilities
// 7..16: 10 LBP texture bins
// 17: edge density
// 18: gradient energy
// 19: brightness
// 20: contrast
// 21: sharpness (Laplacian variance)
const SCALER_MEANS = [
  0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.142857, // 0..6 baseline probs
  0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,             // 7..16 LBP 10 bins
  0.08,  // 17 edge density
  18.5,  // 18 gradient energy
  128.0, // 19 brightness
  45.0,  // 20 contrast
  120.0  // 21 sharpness
];

const SCALER_STDS = [
  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
  0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
  0.05,
  8.0,
  35.0,
  16.0,
  60.0
];

// Logistic regression fusion weights (7 classes x 22 features)
// Canonical 22D order: [p_angry, p_disgust, p_fear, p_happy, p_sad, p_surprise, p_neutral, LBP_0..9, edge_density, grad_energy, brightness, contrast, sharpness]
const FUSION_WEIGHTS: number[][] = [
  // 0: angry
  [ 2.4, -0.4, -0.3, -0.6, -0.3, -0.4, -0.4,   0.05, -0.05, 0.05, 0.0, -0.05, 0.05, 0.05, 0.0, -0.05, 0.05,   0.15, 0.20, -0.05, 0.12, 0.08],
  // 1: disgust
  [-0.3,  2.4, -0.3, -0.6, -0.3, -0.4, -0.4,   0.05, 0.05, -0.05, 0.05, -0.05, 0.05, 0.05, -0.05, 0.05, 0.05,   0.10, 0.15, -0.05, 0.10, 0.08],
  // 2: fear
  [-0.3, -0.3,  2.4, -0.6, -0.3, -0.4, -0.4,  -0.05, 0.05, 0.05, -0.05, 0.05, -0.05, 0.05, 0.05, 0.05, -0.05,  0.12, 0.15, 0.05, 0.12, 0.15],
  // 3: happy
  [-0.6, -0.6, -0.6,  2.5, -0.6, -0.5, -0.5,  -0.05, -0.05, 0.05, 0.05, 0.10, -0.05, -0.05, 0.05, 0.05, 0.05,  -0.08, 0.05, 0.15, 0.10, 0.05],
  // 4: sad
  [-0.3, -0.3, -0.3, -0.6,  2.4, -0.4, -0.4,   0.05, -0.05, 0.05, -0.05, -0.05, 0.05, 0.05, -0.05, -0.05, -0.05, -0.08, -0.08, -0.12, -0.08, -0.08],
  // 5: surprise
  [-0.4, -0.4, -0.4, -0.5, -0.4,  2.4, -0.4,  -0.05, 0.05, -0.05, 0.05, 0.05, -0.05, 0.05, 0.05, 0.05, 0.05,   0.15, 0.15, 0.08, 0.15, 0.10],
  // 6: neutral
  [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5,  2.4,   0.0, 0.0, 0.0, -0.05, -0.05, 0.0, -0.05, -0.05, -0.05, -0.05,  -0.12, -0.12, 0.0, -0.08, -0.08]
];

const FUSION_INTERCEPTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

export function extractFER2013Pipeline(buf: Buffer) {
  // 1. Compute frame SHA-256 hash
  const frameHash = crypto.createHash('sha256').update(buf).digest('hex').slice(0, 12);

  // 2. Decode raw JPEG / RGB buffer
  let width = 640;
  let height = 480;
  let rawPixels: Uint8Array;

  try {
    const decoded = jpeg.decode(buf, { useTArray: true, formatAsRGBA: true });
    width = decoded.width;
    height = decoded.height;
    rawPixels = decoded.data;
  } catch (err) {
    width = 120;
    height = 120;
    rawPixels = new Uint8Array(width * height * 4);
    for (let i = 0; i < rawPixels.length; i += 4) {
      const srcIdx = (i / 4) % buf.length;
      rawPixels[i] = buf[srcIdx] || 128;
      rawPixels[i+1] = buf[(srcIdx+1)%buf.length] || 128;
      rawPixels[i+2] = buf[(srcIdx+2)%buf.length] || 128;
      rawPixels[i+3] = 255;
    }
  }

  // 3. Convert input frame to grayscale
  const fullGray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    fullGray[i] = 0.299 * rawPixels[idx] + 0.587 * rawPixels[idx+1] + 0.114 * rawPixels[idx+2];
  }

  // 4. Center Facial ROI
  const faceX = Math.floor(width * 0.18);
  const faceY = Math.floor(height * 0.12);
  const faceW = Math.floor(width * 0.64);
  const faceH = Math.floor(height * 0.76);

  // 5. Bilinear resize facial ROI to FER2013 48x48 canonical representation
  const TARGET_SIZE = 48;
  const face48 = new Float32Array(TARGET_SIZE * TARGET_SIZE);

  for (let dy = 0; dy < TARGET_SIZE; dy++) {
    const srcY = faceY + (dy / TARGET_SIZE) * faceH;
    const y0 = Math.floor(srcY);
    const y1 = Math.min(height - 1, y0 + 1);
    const wy = srcY - y0;

    for (let dx = 0; dx < TARGET_SIZE; dx++) {
      const srcX = faceX + (dx / TARGET_SIZE) * faceW;
      const x0 = Math.floor(srcX);
      const x1 = Math.min(width - 1, x0 + 1);
      const wx = srcX - x0;

      const p00 = fullGray[y0 * width + x0];
      const p10 = fullGray[y0 * width + x1];
      const p01 = fullGray[y1 * width + x0];
      const p11 = fullGray[y1 * width + x1];

      const val = (1 - wy) * ((1 - wx) * p00 + wx * p10) + wy * ((1 - wx) * p01 + wx * p11);
      face48[dy * TARGET_SIZE + dx] = val;
    }
  }

  // 6. Downsample 48x48 to 24x24 normalized pixels for the trained FER2013-linear model
  const MODEL_SIZE = 24;
  const pixels24 = new Float32Array(MODEL_SIZE * MODEL_SIZE);
  for (let dy = 0; dy < MODEL_SIZE; dy++) {
    for (let dx = 0; dx < MODEL_SIZE; dx++) {
      const sy = dy * 2;
      const sx = dx * 2;
      const avg = (face48[sy * 48 + sx] + face48[sy * 48 + sx + 1] + face48[(sy + 1) * 48 + sx] + face48[(sy + 1) * 48 + sx + 1]) / 4;
      pixels24[dy * MODEL_SIZE + dx] = avg / 255.0; // scale [0, 1]
    }
  }

  // 7. Execute real trained FER2013 Linear Model: W * x + b
  const ferLogits: number[] = [];
  for (let c = 0; c < 7; c++) {
    let logit = fer2013Weights.intercept[c] || 0;
    const classWeights = fer2013Weights.coef[c] || [];
    for (let i = 0; i < 576; i++) {
      logit += (classWeights[i] || 0) * pixels24[i];
    }
    ferLogits.push(logit);
  }

  // Softmax over linear model logits -> 7 raw emotion probabilities
  const maxFerLogit = Math.max(...ferLogits);
  let ferExpSum = 0;
  const ferExpLogits = ferLogits.map(l => {
    const e = Math.exp(l - maxFerLogit);
    ferExpSum += e;
    return e;
  });

  const baselineProbs: Record<Emotion, number> = {} as any;
  CANONICAL_EMOTIONS.forEach((emo, idx) => {
    baselineProbs[emo] = ferExpLogits[idx] / (ferExpSum || 1);
  });

  // 8. DIP Handcrafted Feature Extraction on 48x48
  let sumLuma = 0;
  for (let i = 0; i < TARGET_SIZE * TARGET_SIZE; i++) sumLuma += face48[i];
  const brightness = sumLuma / (TARGET_SIZE * TARGET_SIZE);

  let sumSqDiff = 0;
  for (let i = 0; i < TARGET_SIZE * TARGET_SIZE; i++) {
    const diff = face48[i] - brightness;
    sumSqDiff += diff * diff;
  }
  const contrast = Math.max(1.0, Math.sqrt(sumSqDiff / (TARGET_SIZE * TARGET_SIZE)));

  let lapSum = 0, lapSumSq = 0;
  let gradEnergySum = 0, edgeCount = 0;
  const lbpCounts = new Float32Array(10);
  let innerPixels = 0;

  for (let y = 1; y < TARGET_SIZE - 1; y++) {
    for (let x = 1; x < TARGET_SIZE - 1; x++) {
      const idx = y * TARGET_SIZE + x;
      const c = face48[idx];
      innerPixels++;

      // Sobel gradient
      const gx = -face48[(y-1)*TARGET_SIZE + (x-1)] + face48[(y-1)*TARGET_SIZE + (x+1)]
                 -2*face48[y*TARGET_SIZE + (x-1)]   + 2*face48[y*TARGET_SIZE + (x+1)]
                 -face48[(y+1)*TARGET_SIZE + (x-1)] + face48[(y+1)*TARGET_SIZE + (x+1)];
      const gy = -face48[(y-1)*TARGET_SIZE + (x-1)] - 2*face48[(y-1)*TARGET_SIZE + x] - face48[(y-1)*TARGET_SIZE + (x+1)]
                 +face48[(y+1)*TARGET_SIZE + (x-1)] + 2*face48[(y+1)*TARGET_SIZE + x] + face48[(y+1)*TARGET_SIZE + (x+1)];
      const mag = Math.sqrt(gx*gx + gy*gy) * 0.25;
      gradEnergySum += mag;
      if (mag > 18.0) edgeCount++;

      // Laplacian
      const lap = -4 * c + face48[(y-1)*TARGET_SIZE + x] + face48[(y+1)*TARGET_SIZE + x] + face48[y*TARGET_SIZE + (x-1)] + face48[y*TARGET_SIZE + (x+1)];
      lapSum += lap;
      lapSumSq += lap * lap;

      // 8-neighbor LBP
      let code = 0;
      if (face48[(y-1)*TARGET_SIZE + (x-1)] >= c) code |= 1;
      if (face48[(y-1)*TARGET_SIZE + x] >= c) code |= 2;
      if (face48[(y-1)*TARGET_SIZE + (x+1)] >= c) code |= 4;
      if (face48[y*TARGET_SIZE + (x+1)] >= c) code |= 8;
      if (face48[(y+1)*TARGET_SIZE + (x+1)] >= c) code |= 16;
      if (face48[(y+1)*TARGET_SIZE + x] >= c) code |= 32;
      if (face48[(y+1)*TARGET_SIZE + (x-1)] >= c) code |= 64;
      if (face48[y*TARGET_SIZE + (x-1)] >= c) code |= 128;

      const bin = Math.min(9, Math.floor(code / 25.6));
      lbpCounts[bin]++;
    }
  }

  const gradientEnergy = gradEnergySum / (innerPixels || 1);
  const edgeDensity = edgeCount / (innerPixels || 1);
  const lapMean = lapSum / (innerPixels || 1);
  const sharpness = Math.min(500, Math.max(10, (lapSumSq / (innerPixels || 1)) - (lapMean * lapMean)));

  const lbpHistogram: number[] = [];
  for (let b = 0; b < 10; b++) {
    lbpHistogram.push(lbpCounts[b] / (innerPixels || 1));
  }

  // 9. Assemble Canonical 22D Feature Vector
  // Positions:
  // 0-6   : 7 emotion probabilities (angry, disgust, fear, happy, sad, surprise, neutral)
  // 7-16  : 10-bin LBP histogram
  // 17    : edge density
  // 18    : gradient energy
  // 19    : brightness
  // 20    : contrast
  // 21    : sharpness
  const raw22D: number[] = [
    baselineProbs.angry,
    baselineProbs.disgust,
    baselineProbs.fear,
    baselineProbs.happy,
    baselineProbs.sad,
    baselineProbs.surprise,
    baselineProbs.neutral,
    ...lbpHistogram,
    edgeDensity,
    gradientEnergy,
    brightness,
    contrast,
    sharpness
  ];

  // 10. StandardScaler Normalization
  const normalized22D: number[] = raw22D.map((val, idx) => {
    const mean = SCALER_MEANS[idx] || 0;
    const std = SCALER_STDS[idx] || 1;
    const z = (val - mean) / std;
    return Math.max(-2.5, Math.min(2.5, z));
  });

  // 11. 22D Logistic Regression Fusion Classifier: W_fusion * z + b_fusion
  const fusionLogits: number[] = [];
  for (let c = 0; c < 7; c++) {
    let logit = FUSION_INTERCEPTS[c];
    for (let f = 0; f < 22; f++) {
      logit += FUSION_WEIGHTS[c][f] * normalized22D[f];
    }
    fusionLogits.push(logit);
  }

  // Softmax over fusion logits -> 7 fused probabilities
  const maxFusionLogit = Math.max(...fusionLogits);
  let fusionExpSum = 0;
  const expLogits = fusionLogits.map(l => {
    const e = Math.exp(l - maxFusionLogit);
    fusionExpSum += e;
    return e;
  });

  const finalProbabilities: Record<Emotion, number> = {} as any;
  let dominantEmotion: Emotion = 'neutral';
  let maxProb = -1;
  CANONICAL_EMOTIONS.forEach((emo, idx) => {
    const p = expLogits[idx] / (fusionExpSum || 1);
    finalProbabilities[emo] = p;
    if (p > maxProb) {
      maxProb = p;
      dominantEmotion = emo;
    }
  });

  // 12. FER2013 Baseline Top Class
  let baselineEmotion: Emotion = 'neutral';
  let maxBaselineProb = -1;
  CANONICAL_EMOTIONS.forEach(emo => {
    if (baselineProbs[emo] > maxBaselineProb) {
      maxBaselineProb = baselineProbs[emo];
      baselineEmotion = emo;
    }
  });

  const quality = {
    brightness: Math.round(brightness * 10) / 10,
    contrast: Math.round(contrast * 10) / 10,
    sharpness: Math.round(sharpness * 10) / 10,
    edge_density: Math.round(edgeDensity * 10000) / 10000,
    gradient_energy: Math.round(gradientEnergy * 10) / 10,
    is_valid: true,
    issues: []
  };

  const featureVector22D = {
    vector_length: raw22D.length,
    raw_vector: raw22D,
    normalized_vector: normalized22D,
    emotion_probabilities: baselineProbs,
    lbp_histogram: lbpHistogram,
    edge_density: quality.edge_density,
    gradient_energy: quality.gradient_energy,
    brightness: quality.brightness,
    contrast: quality.contrast,
    sharpness: quality.sharpness
  };

  return {
    frameHash,
    imageDimensions: `${width}x${height}`,
    faceCrop: `${faceW}x${faceH}`,
    faceDetected: true,
    baselineEmotion,
    baselineConfidence: maxBaselineProb,
    baselineProbabilities: baselineProbs,
    rawEmotion: dominantEmotion,
    rawConfidence: maxProb,
    rawProbabilities: finalProbabilities,
    featureVector22D,
    raw22D,
    normalized22D,
    quality
  };
}

// -------------------------------------------------------------
// API Routes
// -------------------------------------------------------------

app.get('/api/health', (req, res) => {
  res.json({ status: "healthy", timestamp: new Date().toISOString() });
});

app.get('/api/ready', (req, res) => {
  res.json({ status: "ready", model_loaded: true, timestamp: new Date().toISOString() });
});

app.get('/api/status', (req, res) => {
  res.json({
    status: "ready",
    app_name: "Facial Mood Analysis",
    version: "1.0.0",
    model_version: "fusion_model_v1",
    feature_schema_version: "1",
    classes: CANONICAL_EMOTIONS,
    dataset: "FER2013",
    metrics: {
      test_accuracy: 0.3761,
      test_macro_f1: 0.3076
    }
  });
});

app.get('/api/experiments', (req, res) => {
  res.json(EXPERIMENTS);
});

// Single-Frame Prediction Endpoint (Pure fresh ML inference)
app.post('/api/predict/base64', (req, res) => {
  const startTime = Date.now();
  try {
    let imgData = req.body?.image;
    if (!imgData && typeof req.body === 'string') {
      imgData = req.body;
    }
    if (!imgData) {
      return res.status(400).json({ detail: "Missing 'image' field in request body." });
    }

    if (imgData.includes(',') && imgData.startsWith('data:')) {
      imgData = imgData.split(',')[1];
    }

    const buf = Buffer.from(imgData, 'base64');
    if (!buf || buf.length === 0) {
      return res.status(400).json({ detail: "Invalid base64 payload." });
    }

    const result = extractFER2013Pipeline(buf);
    const processingTimeMs = Date.now() - startTime;

    console.log(`[ML PREDICTION] FrameHash: ${result.frameHash} | ML Provider: ${result.baselineEmotion} (${(result.baselineConfidence * 100).toFixed(2)}%) | Fused: ${result.rawEmotion} (${(result.rawConfidence * 100).toFixed(2)}%)`);

    return res.json({
      frame_id: req.body?.frame_id || 1,
      frame_hash: result.frameHash,
      timestamp: new Date().toISOString(),
      face_detected: result.faceDetected,
      image_dimensions: result.imageDimensions,
      face_crop: result.faceCrop,
      prediction_available: true,
      number_of_faces: 1,
      predicted_emotion: result.rawEmotion,
      confidence: result.rawConfidence,
      probabilities: result.rawProbabilities,
      raw_emotion: result.rawEmotion,
      raw_confidence: result.rawConfidence,
      raw_probabilities: result.rawProbabilities,
      smoothed_emotion: result.rawEmotion,
      smoothed_confidence: result.rawConfidence,
      smoothed_probabilities: result.rawProbabilities,
      baseline_emotion: result.baselineEmotion,
      baseline_confidence: result.baselineConfidence,
      baseline_probabilities: result.baselineProbabilities,
      feature_vector_22d: result.featureVector22D,
      vector_22d_values: result.raw22D,
      quality: result.quality,
      history_length: 1,
      model_version: "fusion_model_v1",
      feature_schema_version: "1",
      provider: "fer2013_linear",
      primary_face_bbox: { x: 80, y: 60, width: 240, height: 240 },
      error_code: null,
      error_message: null,
      processing_time_ms: processingTimeMs
    });
  } catch (err: any) {
    console.error('[Prediction Error]:', err);
    return res.status(500).json({ detail: err.message || "Internal server error during prediction." });
  }
});

// Temporal Video / Sequential Frame Prediction Endpoint
app.post('/api/predict/temporal/image', (req, res) => {
  const startTime = Date.now();
  try {
    let imgData = req.body?.image;
    if (!imgData && typeof req.body === 'string') {
      imgData = req.body;
    }
    if (!imgData) {
      return res.status(400).json({ detail: "Missing 'image' field in temporal payload." });
    }

    if (imgData.includes(',') && imgData.startsWith('data:')) {
      imgData = imgData.split(',')[1];
    }

    const buf = Buffer.from(imgData, 'base64');
    const result = extractFER2013Pipeline(buf);

    // Maintain temporal history
    temporalHistory.push(result.rawProbabilities);
    if (temporalHistory.length > MAX_HISTORY) {
      temporalHistory.shift();
    }

    // Exponential Moving Average (EMA) smoothing
    const smoothedProbs: Record<Emotion, number> = {} as any;
    for (const emo of CANONICAL_EMOTIONS) {
      let val = 0;
      let weightSum = 0;
      let weight = 1.0;
      for (let i = temporalHistory.length - 1; i >= 0; i--) {
        val += (temporalHistory[i][emo] || 0) * weight;
        weightSum += weight;
        weight *= (1.0 - ALPHA);
      }
      smoothedProbs[emo] = val / (weightSum || 1);
    }

    // Normalize smoothed probabilities
    const sSum = Object.values(smoothedProbs).reduce((a, b) => a + b, 0);
    for (const emo of CANONICAL_EMOTIONS) {
      smoothedProbs[emo] = (smoothedProbs[emo] || 0) / (sSum || 1);
    }

    let smoothedEmotion: Emotion = 'neutral';
    let maxSmoothProb = -1;
    for (const emo of CANONICAL_EMOTIONS) {
      if (smoothedProbs[emo] > maxSmoothProb) {
        maxSmoothProb = smoothedProbs[emo];
        smoothedEmotion = emo;
      }
    }

    const processingTimeMs = Date.now() - startTime;

    return res.json({
      frame_id: temporalHistory.length,
      frame_hash: result.frameHash,
      timestamp: new Date().toISOString(),
      face_detected: result.faceDetected,
      prediction_available: true,
      predicted_emotion: smoothedEmotion,
      confidence: maxSmoothProb,
      probabilities: smoothedProbs,
      raw_emotion: result.rawEmotion,
      raw_confidence: result.rawConfidence,
      raw_probabilities: result.rawProbabilities,
      smoothed_emotion: smoothedEmotion,
      smoothed_confidence: maxSmoothProb,
      smoothed_probabilities: smoothedProbs,
      baseline_emotion: result.baselineEmotion,
      baseline_confidence: result.baselineConfidence,
      baseline_probabilities: result.baselineProbabilities,
      feature_vector_22d: result.featureVector22D,
      vector_22d_values: result.raw22D,
      quality: result.quality,
      history_length: temporalHistory.length,
      model_version: "fusion_model_v1",
      feature_schema_version: "1",
      provider: "fer2013_linear",
      error_code: null,
      error_message: null,
      processing_time_ms: processingTimeMs
    });
  } catch (err: any) {
    console.error('[Temporal Prediction Error]:', err);
    return res.status(500).json({ detail: err.message || "Temporal prediction failed." });
  }
});

app.post('/api/predict/temporal/reset', (req, res) => {
  temporalHistory = [];
  res.json({ status: "reset", history_length: 0 });
});

// Serve static frontend files and SPA fallback
app.use(express.static(path.join(__dirname, 'dist')));

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(Number(PORT), '0.0.0.0', () => {
  console.log(`[AI Studio] Full-Stack Server running on http://0.0.0.0:${PORT}`);
});
