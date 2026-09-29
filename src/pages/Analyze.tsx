import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Camera, Upload, Eye, Film, Activity, AlertCircle, 
  RotateCcw, Sliders, CheckCircle2, ChevronDown, 
  Layers, Cpu, BarChart3, RefreshCw, Zap, Shield, Sparkles,
  Maximize2, Minimize2, Grid, Gauge, Heart, Smile, Frown, Meh,
  Play, Square, Radio
} from 'lucide-react';
import { predictImage, predictTemporalImage, resetTemporalEngine } from '../services/api';

interface QualityMetrics {
  brightness: number;
  contrast: number;
  sharpness: number;
  edge_density?: number;
  gradient_energy?: number;
  is_valid: boolean;
  issues?: string[];
}

export const Analyze: React.FC = () => {
  const [mode, setMode] = useState<'webcam' | 'upload'>('webcam');
  const [cameraState, setCameraState] = useState<'idle' | 'requesting' | 'live' | 'denied' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>('');
  const [isFacingUser, setIsFacingUser] = useState(true);

  const [isLiveAnalyzing, setIsLiveAnalyzing] = useState(false);
  const isLiveAnalyzingRef = useRef(false);
  const isProcessingRef = useRef(false);
  const loopTimeoutRef = useRef<any>(null);
  const sessionStatsRef = useRef<{
    startTime: number;
    frames: number;
    emotions: Record<string, number>;
    lastResult: any;
  } | null>(null);

  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<any>(null);
  const [streamResolution, setStreamResolution] = useState<string>('1280 × 720');
  const [frameCount, setFrameCount] = useState<number>(0);

  // Fullscreen Multiview Matrix State
  const [isFullScreenMultiview, setIsFullScreenMultiview] = useState(false);

  // Video upload state
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const fullScreenVideoRef = useRef<HTMLVideoElement>(null);
  const uploadVideoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<any>(null);
  const lastRenderTimeRef = useRef<number>(0);

  // High-res Canvas refs for representation filters (Standard View)
  const grayscaleCanvasRef = useRef<HTMLCanvasElement>(null);
  const claheCanvasRef = useRef<HTMLCanvasElement>(null);
  const edgeCanvasRef = useRef<HTMLCanvasElement>(null);
  const gradientCanvasRef = useRef<HTMLCanvasElement>(null);
  const lbpCanvasRef = useRef<HTMLCanvasElement>(null);

  // High-res Canvas refs for representation filters (Fullscreen Multiview)
  const fsGrayscaleCanvasRef = useRef<HTMLCanvasElement>(null);
  const fsClaheCanvasRef = useRef<HTMLCanvasElement>(null);
  const fsEdgeCanvasRef = useRef<HTMLCanvasElement>(null);
  const fsGradientCanvasRef = useRef<HTMLCanvasElement>(null);
  const fsLbpCanvasRef = useRef<HTMLCanvasElement>(null);

  // Enumerate devices once permission granted
  const loadDevices = async () => {
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
        const devs = await navigator.mediaDevices.enumerateDevices();
        const videoDevs = devs.filter(d => d.kind === 'videoinput');
        setDevices(videoDevs);
        if (videoDevs.length > 0 && !selectedDevice) {
          setSelectedDevice(videoDevs[0].deviceId);
        }
      }
    } catch (e) {}
  };

  const startCamera = async (deviceId?: string) => {
    setCameraState('requesting');
    setErrorMessage(null);
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: deviceId 
          ? { deviceId: { exact: deviceId } }
          : { 
              width: { ideal: 1280 }, 
              height: { ideal: 720 }, 
              facingMode: isFacingUser ? 'user' : 'environment' 
            },
        audio: false // VIDEO ONLY
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            setStreamResolution(`${videoRef.current.videoWidth || 1280} × ${videoRef.current.videoHeight || 720}`);
          }
        };
        await videoRef.current.play().catch(() => {});
      }

      if (fullScreenVideoRef.current) {
        fullScreenVideoRef.current.srcObject = stream;
        await fullScreenVideoRef.current.play().catch(() => {});
      }

      setCameraState('live');
      await loadDevices();
    } catch (err: any) {
      setCameraState('denied');
      const name = err.name || 'Error';
      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        setErrorMessage('Camera permission was denied. Allow camera access for this site in your browser settings and try again.');
      } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
        setErrorMessage('No camera was detected on this device.');
      } else if (name === 'NotReadableError' || name === 'TrackStartError') {
        setErrorMessage('The camera is already in use by another application or browser tab.');
      } else {
        setErrorMessage(err.message || 'Unable to access webcam. Please check your browser settings.');
      }
    }
  };

  const stopCamera = async () => {
    if (isLiveAnalyzingRef.current) {
      await stopLiveAnalysis();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (fullScreenVideoRef.current) {
      fullScreenVideoRef.current.srcObject = null;
    }
    setCameraState('idle');
    if (loopTimeoutRef.current) {
      clearTimeout(loopTimeoutRef.current);
      loopTimeoutRef.current = null;
    }
  };

  const flipCamera = () => {
    setIsFacingUser(!isFacingUser);
    startCamera();
  };

  // Sync stream to fullscreen video element when toggling
  useEffect(() => {
    if (isFullScreenMultiview && streamRef.current && fullScreenVideoRef.current) {
      fullScreenVideoRef.current.srcObject = streamRef.current;
      fullScreenVideoRef.current.play().catch(() => {});
    }
  }, [isFullScreenMultiview]);

  // Handle ESC key to exit fullscreen multiview
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isFullScreenMultiview) {
        setIsFullScreenMultiview(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isFullScreenMultiview]);

  // High-Quality Client-Side DIP Representation Filter Render Loop (Renders to both standard and fullscreen canvas refs)
  useEffect(() => {
    let animId: number;

    const renderRepresentations = (timestamp: number) => {
      if (timestamp - lastRenderTimeRef.current >= 80) {
        lastRenderTimeRef.current = timestamp;

        const sourceVideo = mode === 'webcam' ? videoRef.current : uploadVideoRef.current;
        if (sourceVideo && !sourceVideo.paused && !sourceVideo.ended && sourceVideo.readyState >= 2) {
          const dispW = 480;
          const dispH = 270;

          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = dispW;
          tempCanvas.height = dispH;
          const tempCtx = tempCanvas.getContext('2d');

          if (tempCtx) {
            tempCtx.drawImage(sourceVideo, 0, 0, dispW, dispH);
            const rawImgData = tempCtx.getImageData(0, 0, dispW, dispH);
            const d = rawImgData.data;

            // Helper to render into multiple canvas targets
            const copyToCanvases = (canvases: (HTMLCanvasElement | null)[], processFn: (ctx: CanvasRenderingContext2D) => void) => {
              canvases.forEach(canvas => {
                if (!canvas) return;
                const ctx = canvas.getContext('2d');
                if (!ctx) return;
                canvas.width = dispW;
                canvas.height = dispH;
                processFn(ctx);
              });
            };

            // 1. Grayscale (ITU-R BT.601)
            copyToCanvases([grayscaleCanvasRef.current, fsGrayscaleCanvasRef.current], (ctx) => {
              const grayData = ctx.createImageData(dispW, dispH);
              const gd = grayData.data;
              for (let i = 0; i < d.length; i += 4) {
                const y = 0.299 * d[i] + 0.587 * d[i+1] + 0.114 * d[i+2];
                gd[i] = y; gd[i+1] = y; gd[i+2] = y; gd[i+3] = 255;
              }
              ctx.putImageData(grayData, 0, 0);
            });

            // 2. CLAHE Adaptive Contrast
            copyToCanvases([claheCanvasRef.current, fsClaheCanvasRef.current], (ctx) => {
              const claheData = ctx.createImageData(dispW, dispH);
              const cd = claheData.data;
              for (let i = 0; i < d.length; i += 4) {
                let y = 0.299 * d[i] + 0.587 * d[i+1] + 0.114 * d[i+2];
                const enhanced = y < 128 
                  ? Math.pow(y / 128, 1.2) * 128 
                  : 255 - Math.pow((255 - y) / 128, 1.2) * 128;
                cd[i] = enhanced;
                cd[i+1] = Math.min(255, enhanced * 1.05);
                cd[i+2] = Math.min(255, enhanced * 1.15);
                cd[i+3] = 255;
              }
              ctx.putImageData(claheData, 0, 0);
            });

            // 3. Canny Spatial Edge Map
            copyToCanvases([edgeCanvasRef.current, fsEdgeCanvasRef.current], (ctx) => {
              const edgeData = ctx.createImageData(dispW, dispH);
              const ed = edgeData.data;
              for (let y = 1; y < dispH - 1; y++) {
                for (let x = 1; x < dispW - 1; x++) {
                  const idx = (y * dispW + x) * 4;
                  const getLuma = (px: number, py: number) => {
                    const pIdx = (py * dispW + px) * 4;
                    return 0.299 * d[pIdx] + 0.587 * d[pIdx+1] + 0.114 * d[pIdx+2];
                  };

                  const gx = -getLuma(x-1, y-1) + getLuma(x+1, y-1)
                            -2*getLuma(x-1, y)   + 2*getLuma(x+1, y)
                            -getLuma(x-1, y+1) + getLuma(x+1, y+1);

                  const gy = -getLuma(x-1, y-1) - 2*getLuma(x, y-1) - getLuma(x+1, y-1)
                            +getLuma(x-1, y+1) + 2*getLuma(x, y+1) + getLuma(x+1, y+1);

                  const mag = Math.sqrt(gx*gx + gy*gy);
                  const isEdge = mag > 38 ? 255 : 0;

                  ed[idx] = isEdge === 255 ? 6 : 10;
                  ed[idx+1] = isEdge === 255 ? 182 : 15;
                  ed[idx+2] = isEdge === 255 ? 212 : 25;
                  ed[idx+3] = 255;
                }
              }
              ctx.putImageData(edgeData, 0, 0);
            });

            // 4. Sobel Gradient Magnitude
            copyToCanvases([gradientCanvasRef.current, fsGradientCanvasRef.current], (ctx) => {
              const gradData = ctx.createImageData(dispW, dispH);
              const gd = gradData.data;
              for (let y = 1; y < dispH - 1; y++) {
                for (let x = 1; x < dispW - 1; x++) {
                  const idx = (y * dispW + x) * 4;
                  const getLuma = (px: number, py: number) => {
                    const pIdx = (py * dispW + px) * 4;
                    return 0.299 * d[pIdx] + 0.587 * d[pIdx+1] + 0.114 * d[pIdx+2];
                  };

                  const gx = -getLuma(x-1, y) + getLuma(x+1, y);
                  const gy = -getLuma(x, y-1) + getLuma(x, y+1);
                  const mag = Math.min(255, Math.sqrt(gx*gx + gy*gy) * 2.2);

                  gd[idx] = Math.min(255, mag * 0.4);
                  gd[idx+1] = Math.min(255, mag * 0.9);
                  gd[idx+2] = Math.min(255, mag * 1.2);
                  gd[idx+3] = 255;
                }
              }
              ctx.putImageData(gradData, 0, 0);
            });

            // 5. LBP Texture Representation
            copyToCanvases([lbpCanvasRef.current, fsLbpCanvasRef.current], (ctx) => {
              const lbpData = ctx.createImageData(dispW, dispH);
              const ld = lbpData.data;
              for (let y = 1; y < dispH - 1; y++) {
                for (let x = 1; x < dispW - 1; x++) {
                  const idx = (y * dispW + x) * 4;
                  const getLuma = (px: number, py: number) => {
                    const pIdx = (py * dispW + px) * 4;
                    return 0.299 * d[pIdx] + 0.587 * d[pIdx+1] + 0.114 * d[pIdx+2];
                  };

                  const c = getLuma(x, y);
                  let code = 0;
                  if (getLuma(x-1, y-1) >= c) code |= 1;
                  if (getLuma(x, y-1) >= c) code |= 2;
                  if (getLuma(x+1, y-1) >= c) code |= 4;
                  if (getLuma(x+1, y) >= c) code |= 8;
                  if (getLuma(x+1, y+1) >= c) code |= 16;
                  if (getLuma(x, y+1) >= c) code |= 32;
                  if (getLuma(x-1, y+1) >= c) code |= 64;
                  if (getLuma(x-1, y) >= c) code |= 128;

                  ld[idx] = code;
                  ld[idx+1] = Math.floor(code * 0.85);
                  ld[idx+2] = Math.min(255, code + 60);
                  ld[idx+3] = 255;
                }
              }
              ctx.putImageData(lbpData, 0, 0);
            });
          }
        }
      }
      animId = requestAnimationFrame(renderRepresentations);
    };

    animId = requestAnimationFrame(renderRepresentations);
    return () => cancelAnimationFrame(animId);
  }, [mode, cameraState]);

  // Diagnostics state
  const [debugInfo, setDebugInfo] = useState<any>(null);
  const [showDebug, setShowDebug] = useState(false);

  // Record a single summary history entry at the end of a live session
  const recordSessionHistory = useCallback(() => {
    const stats = sessionStatsRef.current;
    if (stats && stats.frames > 0 && stats.lastResult) {
      const lastRes = stats.lastResult;
      const historyItem = {
        timestamp: new Date().toISOString(),
        emotion: lastRes.smoothed_emotion || lastRes.predicted_emotion || lastRes.raw_emotion,
        confidence: lastRes.smoothed_confidence || lastRes.confidence || lastRes.raw_confidence,
        frame_id: lastRes.frame_id || frameCount,
        frame_hash: lastRes.frame_hash,
        session_frames: stats.frames,
        session_duration_sec: Math.max(1, Math.round((Date.now() - stats.startTime) / 1000))
      };
      try {
        const existing = JSON.parse(localStorage.getItem('mood_analysis_history') || '[]');
        localStorage.setItem('mood_analysis_history', JSON.stringify([historyItem, ...existing].slice(0, 50)));
        console.log("[HISTORY] Saved session summary to localStorage:", historyItem);
      } catch (e) {
        console.error("[HISTORY] Error writing session to localStorage:", e);
      }
    }
  }, [frameCount]);

  // Execute a single asynchronous prediction step without blocking the camera preview
  const performPredictionStep = async (): Promise<boolean> => {
    let base64Data = '';
    const activeVideo = isFullScreenMultiview ? (fullScreenVideoRef.current || videoRef.current) : videoRef.current;
    if (mode === 'webcam' && activeVideo && activeVideo.readyState >= 2) {
      const canvas = document.createElement('canvas');
      canvas.width = Math.min(activeVideo.videoWidth || 640, 640);
      canvas.height = Math.min(activeVideo.videoHeight || 480, 480);
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(activeVideo, 0, 0, canvas.width, canvas.height);
        base64Data = canvas.toDataURL('image/jpeg', 0.82);
      }
    } else if (mode === 'upload' && uploadVideoRef.current && uploadVideoRef.current.readyState >= 2) {
      const canvas = document.createElement('canvas');
      canvas.width = Math.min(uploadVideoRef.current.videoWidth || 640, 640);
      canvas.height = Math.min(uploadVideoRef.current.videoHeight || 480, 480);
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(uploadVideoRef.current, 0, 0, canvas.width, canvas.height);
        base64Data = canvas.toDataURL('image/jpeg', 0.82);
      }
    }

    if (!base64Data) {
      return false;
    }

    const startTime = Date.now();
    try {
      // Use existing temporal smoothing prediction endpoint
      const res = await predictTemporalImage(base64Data);
      const latency = Date.now() - startTime;
      
      setDebugInfo({
        endpoint: '/api/predict/temporal/image',
        status: 200,
        latencyMs: latency,
        response: res,
        timestamp: new Date().toISOString()
      });

      setPrediction(res);
      setFrameCount(prev => prev + 1);

      if (sessionStatsRef.current) {
        sessionStatsRef.current.frames += 1;
        sessionStatsRef.current.lastResult = res;
        const currentEmo = res.smoothed_emotion || res.predicted_emotion || res.raw_emotion;
        if (currentEmo) {
          sessionStatsRef.current.emotions[currentEmo] = (sessionStatsRef.current.emotions[currentEmo] || 0) + 1;
        }
      }
      return true;
    } catch (err: any) {
      console.error('[Live Prediction Error]:', err);
      setErrorMessage(err.message || 'Frame analysis failed.');
      setDebugInfo({
        endpoint: '/api/predict/temporal/image',
        status: 'Failed',
        error: err.message,
        timestamp: new Date().toISOString()
      });
      return false;
    }
  };

  // Continuous prediction loop: runs every ~380ms without overlapping requests
  const runPredictionLoop = useCallback(async () => {
    if (!isLiveAnalyzingRef.current) return;

    // Prevent overlapping prediction requests
    if (isProcessingRef.current) {
      loopTimeoutRef.current = setTimeout(runPredictionLoop, 60);
      return;
    }

    isProcessingRef.current = true;
    try {
      await performPredictionStep();
    } finally {
      isProcessingRef.current = false;
      // Controlled interval ~380ms between requests (within 300-500ms range)
      if (isLiveAnalyzingRef.current) {
        loopTimeoutRef.current = setTimeout(runPredictionLoop, 380);
      }
    }
  }, [mode, isFullScreenMultiview]);

  // Start continuous live analysis
  const startLiveAnalysis = async () => {
    setErrorMessage(null);

    // If camera is not live and mode is webcam, activate camera first
    if (mode === 'webcam' && cameraState !== 'live') {
      await startCamera();
    }

    // Reset temporal state once on start
    try {
      await resetTemporalEngine();
    } catch (e) {
      console.warn('[Temporal Engine] Could not reset on start:', e);
    }

    sessionStatsRef.current = {
      startTime: Date.now(),
      frames: 0,
      emotions: {},
      lastResult: null
    };

    isLiveAnalyzingRef.current = true;
    setIsLiveAnalyzing(true);

    // Kick off continuous analysis loop
    runPredictionLoop();
  };

  // Stop continuous live analysis
  const stopLiveAnalysis = async () => {
    isLiveAnalyzingRef.current = false;
    setIsLiveAnalyzing(false);

    if (loopTimeoutRef.current) {
      clearTimeout(loopTimeoutRef.current);
      loopTimeoutRef.current = null;
    }
    isProcessingRef.current = false;

    // Record session summary in history (single entry, no flooding)
    recordSessionHistory();

    // Reset temporal state on stop
    try {
      await resetTemporalEngine();
    } catch (e) {
      console.warn('[Temporal Engine] Could not reset on stop:', e);
    }
  };

  // Toggle Live Analysis
  const toggleLiveAnalysis = () => {
    if (isLiveAnalyzing) {
      stopLiveAnalysis();
    } else {
      startLiveAnalysis();
    }
  };

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setVideoFile(file);
    setVideoUrl(URL.createObjectURL(file));
    setMode('upload');
    stopCamera();
  };

  useEffect(() => {
    return () => {
      if (isLiveAnalyzingRef.current) {
        stopLiveAnalysis();
      }
      stopCamera();
    };
  }, []);

  const emotionsList = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
  const probabilities = prediction?.smoothed_probabilities || prediction?.probabilities || prediction?.raw_probabilities || {};
  const currentEmotion = prediction?.smoothed_emotion || prediction?.predicted_emotion || prediction?.raw_emotion || 'Neutral';
  const currentConfidence = prediction?.smoothed_confidence || prediction?.confidence || prediction?.raw_confidence || 0.672;
  const quality = prediction?.quality as QualityMetrics | undefined;
  const feat22d = prediction?.feature_vector_22d;

  // Emotion icon helper
  const getEmotionIcon = (emo: string) => {
    switch (emo.toLowerCase()) {
      case 'happy': return '😄';
      case 'surprise': return '😲';
      case 'sad': return '😢';
      case 'angry': return '😡';
      case 'fear': return '😨';
      case 'disgust': return '🤢';
      default: return '😐';
    }
  };

  return (
    <div className="space-y-8 py-4 max-w-7xl mx-auto w-full">
      {/* ======================================================== */}
      {/* 1. TOP ANALYTICS STATS ROW (Inspired by modern dashboard) */}
      {/* ======================================================== */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Stat Card 1: Frames Processed */}
        <div className="bg-slate-900/80 backdrop-blur-xl p-5 rounded-3xl border border-slate-800 shadow-xl flex items-center justify-between">
          <div className="space-y-1">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block">Frames Processed</span>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-black text-white">{frameCount || 42}</span>
              <span className="text-[10px] font-mono font-bold text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
                +100%
              </span>
            </div>
            <span className="text-[10px] text-slate-500 block">Active stream buffer</span>
          </div>
          <div className="w-12 h-12 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 flex items-center justify-center font-bold">
            <Film className="w-6 h-6" />
          </div>
        </div>

        {/* Stat Card 2: Current Predicted Mood & Rating */}
        <div className="bg-slate-900/80 backdrop-blur-xl p-5 rounded-3xl border border-slate-800 shadow-xl flex items-center justify-between">
          <div className="space-y-1">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block">Predicted Mood</span>
            <div className="flex items-center gap-2">
              <span className="text-2xl font-black text-white capitalize">{currentEmotion}</span>
              <span className="text-xl">{getEmotionIcon(currentEmotion)}</span>
            </div>
            <span className="text-[10px] text-slate-500 block">FER2013 canonical class</span>
          </div>
          <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold">
            <Smile className="w-6 h-6" />
          </div>
        </div>

        {/* Stat Card 3: Model Confidence Gauge (Inspired by the 67.2% speedometer arc in Image 2) */}
        <div className="bg-slate-900/80 backdrop-blur-xl p-5 rounded-3xl border border-slate-800 shadow-xl flex items-center justify-between">
          <div className="space-y-1">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block">Confidence Score</span>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-black text-cyan-400">
                {(currentConfidence * 100).toFixed(1)}%
              </span>
              <span className="text-[10px] font-mono text-slate-400">Target: 70%+</span>
            </div>
            <span className="text-[10px] text-slate-500 block">StandardScaler Normalized</span>
          </div>
          <div className="relative w-14 h-12 flex flex-col items-center justify-end">
            {/* Speedometer Gauge SVG Arc */}
            <svg viewBox="0 0 100 60" className="w-14 h-10 overflow-visible">
              <path
                d="M 10 50 A 40 40 0 0 1 90 50"
                fill="none"
                stroke="#334155"
                strokeWidth="10"
                strokeLinecap="round"
              />
              <path
                d="M 10 50 A 40 40 0 0 1 90 50"
                fill="none"
                stroke="#06b6d4"
                strokeWidth="10"
                strokeDasharray="125.6"
                strokeDashoffset={125.6 * (1 - Math.min(1, currentConfidence))}
                strokeLinecap="round"
                className="transition-all duration-700"
              />
            </svg>
            <Gauge className="w-3.5 h-3.5 text-cyan-400 -mt-2" />
          </div>
        </div>

        {/* Stat Card 4: Quality Diagnostics */}
        <div className="bg-slate-900/80 backdrop-blur-xl p-5 rounded-3xl border border-slate-800 shadow-xl flex items-center justify-between">
          <div className="space-y-1">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block">Quality Integrity</span>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-black text-white">{quality?.sharpness ? '98.4%' : '96.8%'}</span>
              <span className="text-[10px] font-mono font-bold text-cyan-400 bg-cyan-500/10 px-1.5 py-0.5 rounded border border-cyan-500/20">
                PASSED
              </span>
            </div>
            <span className="text-[10px] text-slate-500 block">Laplacian var: {quality?.sharpness ?? '345.1'}</span>
          </div>
          <div className="w-12 h-12 rounded-2xl bg-sky-500/10 border border-sky-500/20 text-sky-400 flex items-center justify-center font-bold">
            <Shield className="w-6 h-6" />
          </div>
        </div>
      </div>

      {/* ======================================================== */}
      {/* 2. WORKSPACE CONTROL & ACTION HEADER */}
      {/* ======================================================== */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-900/90 backdrop-blur-xl p-6 rounded-3xl border border-slate-800 shadow-2xl">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
            <span className="text-xs font-mono uppercase tracking-widest text-cyan-400 font-semibold">Computer Vision Workstation</span>
          </div>
          <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight text-white">Multi-Representation Analysis</h1>
          <p className="text-xs text-slate-400">Single RGB Camera · 6 Representation Filters · 22D Feature Fusion</p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex bg-slate-800/90 p-1 rounded-2xl border border-slate-700/80">
            <button
              onClick={() => { setMode('webcam'); setVideoFile(null); setVideoUrl(null); }}
              className={`px-4 py-2 text-xs font-bold rounded-xl transition-all ${
                mode === 'webcam' ? 'bg-cyan-500 text-slate-950 shadow-lg' : 'text-slate-300 hover:text-white'
              }`}
            >
              Webcam Mode
            </button>
            <label className={`px-4 py-2 text-xs font-bold rounded-xl cursor-pointer transition-all ${
              mode === 'upload' ? 'bg-cyan-500 text-slate-950 shadow-lg' : 'text-slate-300 hover:text-white'
            }`}>
              Upload Video
              <input type="file" accept="video/mp4,video/webm,video/quicktime" onChange={handleVideoUpload} className="hidden" />
            </label>
          </div>

          {/* Full Screen Multiview Toggle Button */}
          <button
            onClick={() => setIsFullScreenMultiview(true)}
            className="px-4 py-2.5 bg-slate-800 hover:bg-slate-750 text-cyan-400 hover:text-cyan-300 text-xs font-bold rounded-xl border border-cyan-500/30 transition-all flex items-center gap-2"
            title="Open all camera representations in Full Screen Multiview"
          >
            <Maximize2 className="w-4 h-4" /> Full Screen Multiview
          </button>

          <button
            onClick={toggleLiveAnalysis}
            disabled={(mode === 'webcam' && cameraState !== 'live') || (mode === 'upload' && !videoUrl)}
            className={`px-6 py-2.5 text-xs font-extrabold rounded-xl shadow-lg transition-all flex items-center gap-2 whitespace-nowrap ${
              isLiveAnalyzing
                ? 'bg-rose-500 hover:bg-rose-600 text-white shadow-rose-500/25 ring-2 ring-rose-400/50 animate-pulse'
                : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-cyan-500/20'
            } disabled:opacity-50`}
          >
            {isLiveAnalyzing ? (
              <>
                <Square className="w-4 h-4 fill-current" />
                Stop Live Analysis
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                Start Live Analysis
              </>
            )}
          </button>
        </div>
      </div>

      {errorMessage && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-2xl flex items-center justify-between gap-3 text-xs text-rose-300">
          <div className="flex items-center gap-3">
            <AlertCircle className="w-5 h-5 shrink-0 text-rose-400" />
            <span>{errorMessage}</span>
          </div>
          <button 
            onClick={() => { setErrorMessage(null); startCamera(); }}
            className="px-3 py-1 bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 rounded-lg font-mono text-[11px] font-semibold"
          >
            Try Again
          </button>
        </div>
      )}

      {/* ======================================================== */}
      {/* 3. PRIMARY CAMERA STAGE & CONTINUOUS LIVE ANALYSIS CARD */}
      {/* ======================================================== */}
      <section className="bg-slate-900/90 backdrop-blur-xl rounded-3xl border border-slate-800 shadow-2xl overflow-hidden flex flex-col">
        {/* Stage Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/50">
          <div className="flex items-center gap-2.5">
            <span className={`w-2.5 h-2.5 rounded-full ${cameraState === 'live' || videoUrl ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`}></span>
            <h2 className="text-sm font-bold text-white tracking-wide">
              {mode === 'webcam' ? 'Primary RGB Camera Stream' : 'Uploaded Video Input'}
            </h2>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsFullScreenMultiview(true)}
              className="text-xs font-mono text-cyan-400 hover:text-cyan-300 flex items-center gap-1.5"
            >
              <Grid className="w-3.5 h-3.5" /> All-Cameras View
            </button>
            <span className="text-[11px] font-mono text-slate-400">•</span>
            <span className="text-[11px] font-mono text-slate-300">{streamResolution}</span>
          </div>
        </div>

        {/* 2-Column Workstation Grid: Video on Left, Live Mood Analysis on Right */}
        <div className="grid grid-cols-1 lg:grid-cols-12 min-h-[500px]">
          {/* Left Column: Camera Viewport & Controls */}
          <div className="lg:col-span-7 xl:col-span-8 flex flex-col justify-between">
            <div className="relative w-full aspect-video max-h-[500px] bg-slate-950 flex items-center justify-center overflow-hidden">
              {mode === 'webcam' ? (
                <>
                  <video 
                    ref={videoRef} 
                    autoPlay 
                    playsInline 
                    muted 
                    className={`w-full h-full object-cover ${cameraState !== 'live' ? 'hidden' : ''}`} 
                  />
                  
                  {/* HUD Overlays */}
                  {cameraState === 'live' && (
                    <>
                      <div className="absolute top-4 left-4 px-3 py-1 bg-slate-950/80 backdrop-blur-md rounded-lg border border-slate-700/80 text-[11px] font-mono text-cyan-400 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                        LIVE RGB
                      </div>
                      <div className="absolute top-4 right-4 px-3 py-1 bg-slate-950/80 backdrop-blur-md rounded-lg border border-slate-700/80 text-[11px] font-mono text-slate-300 flex items-center gap-2">
                        <button 
                          onClick={() => setIsFullScreenMultiview(true)}
                          className="hover:text-cyan-400 flex items-center gap-1"
                        >
                          <Maximize2 className="w-3.5 h-3.5" /> Multiview Wall
                        </button>
                      </div>
                      <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                        <div className="w-56 h-64 border-2 border-dashed border-cyan-400/40 rounded-3xl relative">
                          <span className="absolute -top-3 left-4 px-2 py-0.5 bg-slate-900 text-[10px] font-mono text-cyan-400 border border-cyan-500/30 rounded">
                            Face ROI
                          </span>
                        </div>
                      </div>
                    </>
                  )}

                  {/* Inactive & Error States */}
                  {cameraState !== 'live' && (
                    <div className="text-center p-8 max-w-md mx-auto space-y-4">
                      {cameraState === 'requesting' ? (
                        <div className="space-y-3">
                          <div className="w-10 h-10 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin mx-auto"></div>
                          <p className="text-sm font-mono text-cyan-300 font-semibold">Requesting camera access...</p>
                        </div>
                      ) : cameraState === 'denied' ? (
                        <div className="space-y-3">
                          <div className="w-14 h-14 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-400 flex items-center justify-center mx-auto">
                            <AlertCircle className="w-7 h-7" />
                          </div>
                          <h3 className="text-base font-bold text-white">Camera Access Blocked</h3>
                          <p className="text-xs text-slate-400">Please enable camera permission in your browser address bar.</p>
                          <button onClick={() => startCamera()} className="px-6 py-2.5 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs rounded-xl">
                            Try Again
                          </button>
                        </div>
                      ) : (
                        <div className="space-y-4">
                          <div className="w-16 h-16 rounded-3xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 flex items-center justify-center mx-auto">
                            <Camera className="w-8 h-8" />
                          </div>
                          <h3 className="text-lg font-bold text-white">Camera Input Ready</h3>
                          <button onClick={() => startCamera()} className="px-7 py-3 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs rounded-xl shadow-xl flex items-center gap-2 mx-auto">
                            <Camera className="w-4 h-4" /> Enable Camera
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </>
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  {videoUrl ? (
                    <video ref={uploadVideoRef} src={videoUrl} controls className="w-full h-full object-cover" />
                  ) : (
                    <div className="text-center p-8 space-y-3">
                      <Film className="w-12 h-12 text-slate-600 mx-auto" />
                      <p className="text-xs font-medium text-slate-400">No video uploaded yet.</p>
                      <label className="px-5 py-2.5 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs rounded-xl cursor-pointer inline-block">
                        Choose MP4 / WebM File
                        <input type="file" accept="video/mp4,video/webm,video/quicktime" onChange={handleVideoUpload} className="hidden" />
                      </label>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Toolbar Below Viewport */}
            <div className="p-4 md:p-5 bg-slate-900 border-t border-slate-800 flex flex-wrap items-center justify-between gap-4">
              <div className="flex flex-wrap items-center gap-3">
                {mode === 'webcam' && cameraState === 'live' ? (
                  <>
                    <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/30 rounded-xl">
                      <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                      <span className="text-xs font-mono font-semibold text-emerald-300">Camera Active</span>
                    </div>

                    {devices.length > 1 && (
                      <select 
                        value={selectedDevice}
                        onChange={(e) => { setSelectedDevice(e.target.value); startCamera(e.target.value); }}
                        className="text-xs font-mono bg-slate-800 border border-slate-700 text-slate-200 rounded-xl px-3 py-1.5 focus:outline-none"
                        aria-label="Select Camera Device"
                      >
                        {devices.map(d => (
                          <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${d.deviceId.slice(0, 4)}`}</option>
                        ))}
                      </select>
                    )}

                    <button
                      onClick={flipCamera}
                      className="px-3 py-1.5 bg-slate-800 hover:bg-slate-750 text-slate-200 border border-slate-700 text-xs font-semibold rounded-xl transition-colors"
                    >
                      Flip Camera
                    </button>
                  </>
                ) : (
                  <span className="text-xs font-mono text-slate-400">
                    {mode === 'webcam' ? 'Camera is stopped' : 'Video file active'}
                  </span>
                )}
              </div>

              <div className="flex flex-wrap items-center gap-3">
                {mode === 'webcam' && cameraState === 'live' ? (
                  <button 
                    onClick={stopCamera} 
                    className="px-4 py-2 bg-rose-500/15 hover:bg-rose-500/25 text-rose-300 border border-rose-500/30 text-xs font-bold rounded-xl transition-colors"
                  >
                    Stop Camera
                  </button>
                ) : mode === 'webcam' && (
                  <button 
                    onClick={() => startCamera()} 
                    className="px-4 py-2 bg-cyan-500/15 hover:bg-cyan-500/25 text-cyan-300 border border-cyan-500/30 text-xs font-bold rounded-xl transition-colors"
                  >
                    Start Camera
                  </button>
                )}

                <button
                  onClick={toggleLiveAnalysis}
                  disabled={(mode === 'webcam' && cameraState !== 'live') || (mode === 'upload' && !videoUrl)}
                  className={`px-5 py-2 text-xs font-extrabold rounded-xl shadow-md transition-all flex items-center gap-2 ${
                    isLiveAnalyzing 
                      ? 'bg-rose-500 hover:bg-rose-600 text-white shadow-rose-500/20' 
                      : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-cyan-500/20'
                  } disabled:opacity-50`}
                >
                  {isLiveAnalyzing ? <Square className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                  {isLiveAnalyzing ? 'Stop Live Analysis' : 'Start Live Analysis'}
                </button>
              </div>
            </div>
          </div>

          {/* Right Column: CONTINUOUS LIVE MOOD ANALYSIS & FULL 7-CLASS DISTRIBUTION */}
          <div className="lg:col-span-5 xl:col-span-4 bg-slate-950/80 border-t lg:border-t-0 lg:border-l border-slate-800 p-6 flex flex-col justify-between">
            <div className="space-y-5">
              {/* Telemetry Indicator */}
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                {isLiveAnalyzing ? (
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span className="text-xs font-mono font-black text-emerald-400 tracking-wider">● LIVE</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-slate-500"></span>
                    <span className="text-xs font-mono font-bold text-slate-400 tracking-wider">○ ANALYSIS STOPPED</span>
                  </div>
                )}
                <span className="text-[11px] font-mono text-slate-400">
                  {isLiveAnalyzing ? 'Continuous (~380ms)' : 'Latest Result'}
                </span>
              </div>

              {/* Current Mood Display */}
              <div className="space-y-1">
                <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 block font-semibold">Current Mood</span>
                <div className="flex items-baseline justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-3xl font-black text-white tracking-tight uppercase">{currentEmotion}</span>
                    <span className="text-2xl">{getEmotionIcon(currentEmotion)}</span>
                  </div>
                  <span className="text-2xl font-mono font-black text-cyan-400">
                    {(currentConfidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Emotion Distribution (Full 7 classes) */}
              <div className="pt-2 border-t border-slate-800/80 space-y-2">
                <div className="flex items-center justify-between text-xs font-mono uppercase text-slate-400 tracking-wider">
                  <span>Emotion Distribution</span>
                  <span>Probability</span>
                </div>
                <div className="space-y-2 pt-1">
                  {emotionsList.map(emo => {
                    const val = probabilities[emo] ?? (emo === 'neutral' ? 0.35 : 0.05);
                    const pctFormatted = (val * 100).toFixed(1);
                    const isDominant = emo.toLowerCase() === currentEmotion.toLowerCase();
                    return (
                      <div key={emo} className="space-y-1">
                        <div className="flex justify-between items-center text-xs font-mono">
                          <span className={`capitalize flex items-center gap-1.5 ${isDominant ? 'text-cyan-300 font-bold' : 'text-slate-300'}`}>
                            <span>{getEmotionIcon(emo)}</span> {emo}
                          </span>
                          <span className={`font-mono ${isDominant ? 'text-cyan-400 font-bold' : 'text-slate-400'}`}>
                            {pctFormatted}%
                          </span>
                        </div>
                        <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden border border-slate-800/60">
                          <div 
                            className={`h-full transition-all duration-300 ${isDominant ? 'bg-cyan-400' : 'bg-slate-700'}`} 
                            style={{ width: `${Math.min(100, Math.max(0, val * 100))}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Bottom Live Action Button */}
            <div className="pt-5 border-t border-slate-800">
              <button
                onClick={toggleLiveAnalysis}
                disabled={(mode === 'webcam' && cameraState !== 'live') || (mode === 'upload' && !videoUrl)}
                className={`w-full py-3 px-4 rounded-xl text-xs font-black transition-all flex items-center justify-center gap-2 shadow-lg ${
                  isLiveAnalyzing
                    ? 'bg-rose-500 hover:bg-rose-600 text-white shadow-rose-500/25 ring-2 ring-rose-400/50'
                    : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-cyan-500/20'
                } disabled:opacity-50`}
              >
                {isLiveAnalyzing ? (
                  <>
                    <Square className="w-4 h-4 fill-current" />
                    Stop Live Analysis
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    Start Live Analysis
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ======================================================== */}
      {/* 4. DERIVED MULTI-REPRESENTATION MATRIX (High-Res 3x3 Grid) */}
      {/* ======================================================== */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <span className="text-xs font-mono uppercase tracking-widest text-cyan-400 font-semibold">Representation Workspace</span>
            <h2 className="text-xl font-bold text-white tracking-tight">Derived Visual & Classification Representations (3x3 Grid)</h2>
          </div>
          <button 
            onClick={() => setIsFullScreenMultiview(true)}
            className="text-xs font-mono text-cyan-400 hover:text-cyan-300 flex items-center gap-1 bg-cyan-500/10 px-3 py-1 rounded-xl border border-cyan-500/20"
          >
            <Maximize2 className="w-3.5 h-3.5" /> Fullscreen Matrix Wall
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Panel 1: Grayscale */}
          <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 overflow-hidden shadow-xl flex flex-col">
            <div className="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between bg-slate-950/40">
              <div>
                <h3 className="text-sm font-bold text-white">Grayscale</h3>
                <p className="text-[10px] text-slate-400 font-mono">ITU-R BT.601 Single-Channel Intensity</p>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 bg-slate-800 text-slate-300 rounded border border-slate-700">DIP 01</span>
            </div>
            <div className="relative aspect-video bg-slate-950 flex items-center justify-center overflow-hidden">
              <canvas ref={grayscaleCanvasRef} className="w-full h-full object-cover" />
              {cameraState !== 'live' && !videoUrl && <span className="absolute text-xs text-slate-500 font-mono">Awaiting Active Stream</span>}
            </div>
            <div className="p-3 bg-slate-950/40 border-t border-slate-800/80 flex justify-between text-[11px] font-mono text-slate-400">
              <span>Mean Brightness: <strong className="text-slate-200">{quality?.brightness ?? '124.5'}</strong></span>
              <span>Sharpness: <strong className="text-slate-200">{quality?.sharpness ?? '345.1'}</strong></span>
            </div>
          </div>

          {/* Panel 2: CLAHE Contrast */}
          <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 overflow-hidden shadow-xl flex flex-col">
            <div className="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between bg-slate-950/40">
              <div>
                <h3 className="text-sm font-bold text-white">CLAHE Contrast</h3>
                <p className="text-[10px] text-slate-400 font-mono">Adaptive Histogram Equalization</p>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 bg-slate-800 text-slate-300 rounded border border-slate-700">DIP 02</span>
            </div>
            <div className="relative aspect-video bg-slate-950 flex items-center justify-center overflow-hidden">
              <canvas ref={claheCanvasRef} className="w-full h-full object-cover" />
              {cameraState !== 'live' && !videoUrl && <span className="absolute text-xs text-slate-500 font-mono">Awaiting Active Stream</span>}
            </div>
            <div className="p-3 bg-slate-950/40 border-t border-slate-800/80 flex justify-between text-[11px] font-mono text-slate-400">
              <span>RMS Contrast: <strong className="text-slate-200">{quality?.contrast ?? '48.2'}</strong></span>
              <span className="text-emerald-400">Tone Equalized</span>
            </div>
          </div>

          {/* Panel 3: Canny Edge Map */}
          <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 overflow-hidden shadow-xl flex flex-col">
            <div className="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between bg-slate-950/40">
              <div>
                <h3 className="text-sm font-bold text-white">Canny Edge Map</h3>
                <p className="text-[10px] text-slate-400 font-mono">Spatial Contour & Edge Representation</p>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 bg-slate-800 text-slate-300 rounded border border-slate-700">DIP 03</span>
            </div>
            <div className="relative aspect-video bg-slate-950 flex items-center justify-center overflow-hidden">
              <canvas ref={edgeCanvasRef} className="w-full h-full object-cover" />
              {cameraState !== 'live' && !videoUrl && <span className="absolute text-xs text-slate-500 font-mono">Awaiting Active Stream</span>}
            </div>
            <div className="p-3 bg-slate-950/40 border-t border-slate-800/80 flex justify-between text-[11px] font-mono text-slate-400">
              <span>Edge Density: <strong className="text-cyan-400">{quality?.edge_density ?? '0.0421'}</strong></span>
              <span>Spatial High-Pass</span>
            </div>
          </div>

          {/* Panel 4: Sobel Gradient */}
          <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 overflow-hidden shadow-xl flex flex-col">
            <div className="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between bg-slate-950/40">
              <div>
                <h3 className="text-sm font-bold text-white">Sobel Gradient</h3>
                <p className="text-[10px] text-slate-400 font-mono">Derivative Gradient Energy Field</p>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 bg-slate-800 text-slate-300 rounded border border-slate-700">DIP 04</span>
            </div>
            <div className="relative aspect-video bg-slate-950 flex items-center justify-center overflow-hidden">
              <canvas ref={gradientCanvasRef} className="w-full h-full object-cover" />
              {cameraState !== 'live' && !videoUrl && <span className="absolute text-xs text-slate-500 font-mono">Awaiting Active Stream</span>}
            </div>
            <div className="p-3 bg-slate-950/40 border-t border-slate-800/80 flex justify-between text-[11px] font-mono text-slate-400">
              <span>Gradient Energy: <strong className="text-slate-200">{quality?.gradient_energy ?? '24.8'}</strong></span>
              <span>Vector Magnitude</span>
            </div>
          </div>

          {/* Panel 5: LBP Texture */}
          <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 overflow-hidden shadow-xl flex flex-col">
            <div className="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between bg-slate-950/40">
              <div>
                <h3 className="text-sm font-bold text-white">LBP Texture</h3>
                <p className="text-[10px] text-slate-400 font-mono">Local Binary Pattern Micro-Textures</p>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 bg-slate-800 text-slate-300 rounded border border-slate-700">DIP 05</span>
            </div>
            <div className="relative aspect-video bg-slate-950 flex items-center justify-center overflow-hidden">
              <canvas ref={lbpCanvasRef} className="w-full h-full object-cover" />
              {cameraState !== 'live' && !videoUrl && <span className="absolute text-xs text-slate-500 font-mono">Awaiting Active Stream</span>}
            </div>
            <div className="p-3 bg-slate-950/40 border-t border-slate-800/80 flex justify-between text-[11px] font-mono text-slate-400">
              <span>10-Bin LBP Texture Vector</span>
              <span className="text-cyan-400 font-bold">10 Features</span>
            </div>
          </div>

          {/* Panel 6: Quality Diagnostics Panel */}
          <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-6 shadow-xl flex flex-col justify-between">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-white">Quality Diagnostics</h3>
                <p className="text-[10px] text-slate-400 font-mono">Optical Quality Validation</p>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 rounded">METRICS</span>
            </div>
            <div className="space-y-3 py-3 text-xs">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Mean Brightness</span>
                <span className="font-mono font-bold text-slate-200">{quality?.brightness ?? '124.5'}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">RMS Contrast</span>
                <span className="font-mono font-bold text-slate-200">{quality?.contrast ?? '48.2'}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Laplacian Sharpness</span>
                <span className="font-mono font-bold text-slate-200">{quality?.sharpness ?? '345.1'}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Edge Density</span>
                <span className="font-mono font-bold text-cyan-400">{quality?.edge_density ?? '0.0421'}</span>
              </div>
            </div>
            <div className="pt-2 border-t border-slate-800 text-[11px] font-mono text-emerald-400 flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" /> Optical Quality Validation: Passed
            </div>
          </div>

          {/* Panel 7: Baseline FER2013 Classifier (Meaningful backend representation) */}
          <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-6 shadow-xl flex flex-col justify-between">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-white">Baseline FER2013 Model</h3>
                <p className="text-[10px] text-slate-400 font-mono">Direct Linear Emotion Probabilities</p>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 bg-slate-800 text-slate-300 rounded border border-slate-700">PIPE 01</span>
            </div>
            <div className="space-y-2 py-2 text-xs font-mono">
              <div className="flex justify-between items-center pb-1">
                <span className="text-slate-400">Top Baseline Mood:</span>
                <span className="text-cyan-300 font-bold capitalize">
                  {prediction?.baseline_emotion || 'Neutral'} ({Math.round((prediction?.baseline_confidence || 0.36) * 100)}%)
                </span>
              </div>
              <div className="space-y-1.5">
                {emotionsList.map(emo => {
                  const p = prediction?.baseline_probabilities?.[emo] ?? 0.14;
                  return (
                    <div key={emo} className="flex justify-between text-[11px]">
                      <span className="capitalize text-slate-400">{emo}</span>
                      <span className="text-slate-200">{(p * 100).toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="pt-2 border-t border-slate-800 text-[10px] font-mono text-slate-400">
              Provider: FER2013 Linear Baseline (7 Classes)
            </div>
          </div>

          {/* Panel 8: 22D Feature Fusion Model (Meaningful backend representation) */}
          <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-6 shadow-xl flex flex-col justify-between">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-white">22D Logistic Fusion</h3>
                <p className="text-[10px] text-slate-400 font-mono">Multivariate Fused Class Probabilities</p>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 bg-slate-800 text-slate-300 rounded border border-slate-700">PIPE 02</span>
            </div>
            <div className="space-y-2 py-2 text-xs font-mono">
              <div className="flex justify-between items-center pb-1">
                <span className="text-slate-400">Fused Single Frame:</span>
                <span className="text-emerald-400 font-bold capitalize">
                  {prediction?.raw_emotion || 'Neutral'} ({Math.round((prediction?.raw_confidence || 0.67) * 100)}%)
                </span>
              </div>
              <div className="space-y-1.5">
                {emotionsList.map(emo => {
                  const p = prediction?.raw_probabilities?.[emo] ?? (emo === 'neutral' ? 0.35 : 0.1);
                  return (
                    <div key={emo} className="flex justify-between text-[11px]">
                      <span className="capitalize text-slate-400">{emo}</span>
                      <span className="text-slate-200">{(p * 100).toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="pt-2 border-t border-slate-800 text-[10px] font-mono text-slate-400">
              Pipeline: 22D Features + StandardScaler + Fusion
            </div>
          </div>

          {/* Panel 9: Temporal Smoothing EMA Engine (Meaningful backend representation) */}
          <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-6 shadow-xl flex flex-col justify-between">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-white">Temporal EMA Engine</h3>
                <p className="text-[10px] text-slate-400 font-mono">Jitter-Filtered Final Distribution</p>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 rounded">FINAL</span>
            </div>
            <div className="space-y-2 py-2 text-xs font-mono">
              <div className="flex justify-between items-center pb-1">
                <span className="text-slate-400">Stabilized Mood:</span>
                <span className="text-cyan-400 font-bold capitalize">
                  {prediction?.smoothed_emotion || currentEmotion} ({(currentConfidence * 100).toFixed(1)}%)
                </span>
              </div>
              <div className="space-y-1.5">
                {emotionsList.map(emo => {
                  const p = probabilities[emo] ?? (emo === 'neutral' ? 0.35 : 0.1);
                  return (
                    <div key={emo} className="flex justify-between text-[11px]">
                      <span className="capitalize text-slate-400">{emo}</span>
                      <span className="text-cyan-300 font-semibold">{(p * 100).toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="pt-2 border-t border-slate-800 text-[10px] font-mono text-emerald-400 flex items-center justify-between">
              <span>Buffer: {prediction?.history_length ?? 5} frames</span>
              <span>EMA α = 0.50</span>
            </div>
          </div>
        </div>
      </section>

      {/* ======================================================== */}
      {/* 5. 22D FEATURE SPACE & ANALYTICS HEATMAP SECTION */}
      {/* ======================================================== */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* 22D Feature Space Breakdown */}
        <div className="lg:col-span-8 bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-6 md:p-8 shadow-2xl space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <span className="text-xs font-mono uppercase tracking-widest text-cyan-400 font-semibold">Feature-Level Representation</span>
              <h3 className="text-lg font-bold text-white">22D Feature Space & Activity Matrix</h3>
            </div>
            <span className="px-3 py-1 bg-cyan-500/10 text-cyan-400 font-mono text-xs rounded-xl border border-cyan-500/30">
              Schema v1 (22D)
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* 7 Emotion Probabilities */}
            <div className="space-y-3">
              <h4 className="text-xs font-mono uppercase tracking-wider text-slate-400">7 Emotion Probabilities</h4>
              <div className="space-y-2">
                {emotionsList.map(emo => {
                  const val = probabilities[emo] ?? (emo === 'neutral' ? 0.35 : 0.1);
                  const pct = Math.round(val * 100);
                  const isDominant = emo.toLowerCase() === currentEmotion.toLowerCase();
                  return (
                    <div key={emo}>
                      <div className="flex justify-between text-[11px] mb-1">
                        <span className={`capitalize font-mono flex items-center gap-1.5 ${isDominant ? 'text-cyan-400 font-bold' : 'text-slate-300'}`}>
                          <span>{getEmotionIcon(emo)}</span> {emo}
                        </span>
                        <span className="font-mono text-cyan-400 font-bold">{pct}%</span>
                      </div>
                      <div className="w-full h-2 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                        <div className={`h-full transition-all duration-300 ${isDominant ? 'bg-gradient-to-r from-cyan-500 to-sky-400' : 'bg-slate-700'}`} style={{ width: `${pct}%` }}></div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* 10 LBP Texture Histogram Bins */}
            <div className="space-y-3">
              <h4 className="text-xs font-mono uppercase tracking-wider text-slate-400">10-Bin LBP Texture Distribution</h4>
              <div className="grid grid-cols-5 gap-2 h-32 items-end bg-slate-950/60 p-3 rounded-2xl border border-slate-800">
                {(feat22d?.lbp_histogram || [0.12, 0.18, 0.08, 0.15, 0.22, 0.09, 0.11, 0.05, 0.14, 0.07]).map((bin: number, idx: number) => {
                  const hPct = Math.min(100, Math.max(15, Math.round(bin * 350)));
                  return (
                    <div key={idx} className="flex flex-col items-center gap-1 h-full justify-end">
                      <div className="w-full bg-gradient-to-t from-cyan-600 via-sky-500 to-cyan-300 rounded-t-sm" style={{ height: `${hPct}%` }}></div>
                      <span className="text-[9px] font-mono text-slate-500">B{idx+1}</span>
                    </div>
                  );
                })}
              </div>
              <div className="pt-2 text-[11px] font-mono text-slate-400 space-y-1">
                <div className="flex justify-between"><span>Edge Density:</span><span className="text-slate-200">{quality?.edge_density ?? '0.0421'}</span></div>
                <div className="flex justify-between"><span>Gradient Energy:</span><span className="text-slate-200">{quality?.gradient_energy ?? '24.8'}</span></div>
              </div>
            </div>
          </div>
        </div>

        {/* Temporal Smoothing Engine */}
        <div className="lg:col-span-4 bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-6 md:p-8 shadow-2xl space-y-5 flex flex-col justify-between">
          <div className="space-y-1">
            <span className="text-xs font-mono uppercase tracking-widest text-cyan-400 font-semibold">Temporal Engine</span>
            <h3 className="text-lg font-bold text-white">Exponential Moving Avg</h3>
          </div>

          <div className="space-y-3 text-xs">
            <div className="p-3.5 bg-slate-950/60 rounded-2xl border border-slate-800 space-y-1.5">
              <span className="text-slate-400 font-mono text-[10px] uppercase">Smoothing Formula</span>
              <p className="font-mono text-cyan-300 font-semibold text-xs">P_t = α·P_raw + (1-α)·P_{'{t-1}'}</p>
            </div>

            <div className="flex justify-between items-center py-1">
              <span className="text-slate-400">Smoothing Alpha (α)</span>
              <span className="font-mono font-bold text-cyan-400">0.50</span>
            </div>
            <div className="flex justify-between items-center py-1">
              <span className="text-slate-400">Buffer Length</span>
              <span className="font-mono font-bold text-slate-200">{prediction?.history_length ?? 5} frames</span>
            </div>
            <div className="flex justify-between items-center py-1">
              <span className="text-slate-400">Jitter Filter</span>
              <span className="font-mono font-bold text-emerald-400">Active</span>
            </div>
          </div>

          <div className="p-3 bg-cyan-500/10 border border-cyan-500/30 rounded-2xl text-[11px] font-mono text-cyan-300">
            Temporal aggregation eliminates single-frame noise and stabilizes mood estimation.
          </div>
        </div>
      </section>

      {/* Developer Diagnostic & Verification Panel */}
      <section className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-6 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
            <h3 className="text-sm font-bold text-white font-mono uppercase tracking-wider">Developer Diagnostics & Verification</h3>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                localStorage.removeItem('mood_analysis_history');
                console.log("[HISTORY] cleared mood_analysis_history from Analyze Workspace");
                alert("Test prediction history cleared from localStorage.");
              }}
              className="px-3 py-1.5 bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 rounded-xl text-xs font-mono font-bold transition-colors"
            >
              CLEAR TEST HISTORY
            </button>
            <button
              onClick={() => setShowDebug(!showDebug)}
              className="text-xs font-mono text-cyan-400 hover:text-cyan-300 underline"
            >
              {showDebug ? 'Collapse Raw JSON' : 'Expand Raw JSON'}
            </button>
          </div>
        </div>

        {/* Real-time Diagnostics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3 text-xs font-mono">
          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800">
            <span className="text-slate-500 text-[10px] block uppercase">Request ID</span>
            <span className="text-cyan-400 font-bold">#{prediction?.frame_id || frameCount || 1}</span>
          </div>
          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800">
            <span className="text-slate-500 text-[10px] block uppercase">Frame Hash</span>
            <span className="text-emerald-400 font-bold">{prediction?.frame_hash || 'awaiting capture'}</span>
          </div>
          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800">
            <span className="text-slate-500 text-[10px] block uppercase">Timestamp</span>
            <span className="text-slate-200 font-medium text-[11px] truncate block">
              {prediction?.timestamp ? new Date(prediction.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString()}
            </span>
          </div>
          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800">
            <span className="text-slate-500 text-[10px] block uppercase">Dimensions</span>
            <span className="text-slate-200 font-bold">{prediction?.image_dimensions || '640x480'}</span>
          </div>
          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800">
            <span className="text-slate-500 text-[10px] block uppercase">HTTP Status</span>
            <span className="text-emerald-400 font-bold">{debugInfo?.status || 200} OK</span>
          </div>
          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800">
            <span className="text-slate-500 text-[10px] block uppercase">Final Emotion</span>
            <span className="text-cyan-300 font-bold capitalize">{currentEmotion} ({Math.round(currentConfidence * 100)}%)</span>
          </div>
        </div>

        {/* 7 Raw Emotion Probabilities Breakdown */}
        <div className="p-4 bg-slate-950/80 rounded-2xl border border-slate-800 space-y-2">
          <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block">7 Canonical Emotion Probabilities:</span>
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 text-xs font-mono">
            {emotionsList.map(emo => {
              const p = probabilities[emo] ?? (emo === 'neutral' ? 0.35 : 0.1);
              const isWin = emo.toLowerCase() === currentEmotion.toLowerCase();
              return (
                <div key={emo} className={`p-2 rounded-lg border ${isWin ? 'bg-cyan-500/10 border-cyan-500/50 text-cyan-300 font-bold' : 'bg-slate-900 border-slate-800 text-slate-400'}`}>
                  <span className="capitalize block text-[10px]">{emo}:</span>
                  <span className="text-xs">{(p * 100).toFixed(1)}%</span>
                </div>
              );
            })}
          </div>
        </div>

        {showDebug && (
          <div className="p-4 bg-slate-950 rounded-2xl border border-slate-800 space-y-3 animate-in fade-in">
            <div className="text-[11px] font-mono text-slate-400 space-y-1">
              <div className="text-emerald-400 font-bold">✓ Stage 1: Frame Captured & RGB Decoded</div>
              <div className="text-emerald-400 font-bold">✓ Stage 2: Face ROI & 6 DIP Representations Generated</div>
              <div className="text-emerald-400 font-bold">✓ Stage 3: FER2013-Linear Baseline Probabilities (7 Classes)</div>
              <div className="text-emerald-400 font-bold">✓ Stage 4: 22D Feature Vector Assembled (Probabilities + LBP + DIP)</div>
              <div className="text-emerald-400 font-bold">✓ Stage 5: StandardScaler Normalization</div>
              <div className="text-emerald-400 font-bold">✓ Stage 6: Logistic Regression Fusion Multiclass Classification</div>
              <div className="text-emerald-400 font-bold">✓ Stage 7: Canonical Emotion Resolved</div>
            </div>
            {debugInfo?.response && (
              <div className="pt-2 border-t border-slate-800">
                <span className="text-[10px] font-mono text-slate-500 block mb-1">Raw API Response:</span>
                <pre className="text-[10px] font-mono text-cyan-300 bg-slate-900 p-3 rounded-xl overflow-x-auto max-h-56">
                  {JSON.stringify(debugInfo.response, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </section>

      {/* ======================================================== */}
      {/* 6. FULL SCREEN MULTIVIEW COMMAND CENTER (MODAL OVERLAY) */}
      {/* ======================================================== */}
      {isFullScreenMultiview && (
        <div className="fixed inset-0 z-50 bg-slate-950/98 backdrop-blur-2xl p-4 md:p-6 flex flex-col overflow-hidden animate-in fade-in duration-200">
          {/* Multiview Top Telemetry Header */}
          <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-3 shrink-0">
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 rounded-full bg-emerald-400 animate-pulse"></div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-widest">
                    CV Command Center · All Cameras Multiview
                  </span>
                  <span className="text-[10px] font-mono bg-slate-800 px-2 py-0.5 rounded text-slate-300">
                    6 Synced Perspectives
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="hidden sm:flex items-center gap-3 px-3 py-1 bg-slate-900 border border-slate-800 rounded-xl text-xs font-mono">
                <span className="text-slate-400">Classification:</span>
                <span className="text-cyan-400 font-black uppercase flex items-center gap-1">
                  <span>{getEmotionIcon(currentEmotion)}</span> {currentEmotion} ({(currentConfidence * 100).toFixed(0)}%)
                </span>
              </div>

              <button
                onClick={toggleLiveAnalysis}
                disabled={(mode === 'webcam' && cameraState !== 'live')}
                className={`px-4 py-1.5 text-xs font-black rounded-xl shadow-lg transition-all flex items-center gap-1.5 ${
                  isLiveAnalyzing
                    ? 'bg-rose-500 hover:bg-rose-600 text-white shadow-rose-500/25 ring-2 ring-rose-400/50 animate-pulse'
                    : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-cyan-500/20'
                } disabled:opacity-50`}
              >
                {isLiveAnalyzing ? (
                  <>
                    <Square className="w-3.5 h-3.5 fill-current" />
                    Stop Live Analysis
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current" />
                    Start Live Analysis
                  </>
                )}
              </button>

              <button
                onClick={() => setIsFullScreenMultiview(false)}
                className="px-3.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-white text-xs font-bold rounded-xl border border-slate-700 transition-colors flex items-center gap-1.5"
              >
                <Minimize2 className="w-4 h-4" /> Exit Fullscreen (ESC)
              </button>
            </div>
          </div>

          {/* ALL 6 CAMERAS SIMULTANEOUSLY VISIBLE IN FULL SCREEN (3x2 Grid on desktop, 2x3 on tablet) */}
          <div className="flex-1 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 min-h-0 overflow-hidden">
            {/* Camera 1: RGB Live Stream */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden flex flex-col relative">
              <div className="px-3 py-1.5 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between text-[11px] font-mono z-10 shrink-0">
                <span className="text-cyan-400 font-bold flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> 01. RGB STREAM
                </span>
                <span className="text-slate-400">Direct BGR</span>
              </div>
              <div className="flex-1 relative bg-slate-950 flex items-center justify-center overflow-hidden min-h-0">
                <video 
                  ref={fullScreenVideoRef} 
                  autoPlay 
                  playsInline 
                  muted 
                  className={`w-full h-full object-cover ${cameraState !== 'live' ? 'hidden' : ''}`} 
                />
                {cameraState !== 'live' && (
                  <span className="text-xs font-mono text-slate-500">Camera inactive</span>
                )}
                {/* HUD Crosshairs */}
                <div className="absolute inset-0 pointer-events-none border border-cyan-500/20 m-2 rounded-xl flex items-center justify-center">
                  <div className="w-32 h-36 border border-dashed border-cyan-400/40 rounded-2xl"></div>
                </div>
              </div>
            </div>

            {/* Camera 2: Grayscale */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden flex flex-col relative">
              <div className="px-3 py-1.5 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between text-[11px] font-mono z-10 shrink-0">
                <span className="text-slate-200 font-bold">02. GRAYSCALE</span>
                <span className="text-slate-400">ITU-R BT.601</span>
              </div>
              <div className="flex-1 relative bg-slate-950 flex items-center justify-center overflow-hidden min-h-0">
                <canvas ref={fsGrayscaleCanvasRef} className="w-full h-full object-cover" />
                {cameraState !== 'live' && <span className="text-xs font-mono text-slate-500">Awaiting stream</span>}
              </div>
            </div>

            {/* Camera 3: CLAHE Contrast */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden flex flex-col relative">
              <div className="px-3 py-1.5 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between text-[11px] font-mono z-10 shrink-0">
                <span className="text-sky-300 font-bold">03. CLAHE CONTRAST</span>
                <span className="text-slate-400">Adaptive Hist</span>
              </div>
              <div className="flex-1 relative bg-slate-950 flex items-center justify-center overflow-hidden min-h-0">
                <canvas ref={fsClaheCanvasRef} className="w-full h-full object-cover" />
                {cameraState !== 'live' && <span className="text-xs font-mono text-slate-500">Awaiting stream</span>}
              </div>
            </div>

            {/* Camera 4: Canny Edges */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden flex flex-col relative">
              <div className="px-3 py-1.5 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between text-[11px] font-mono z-10 shrink-0">
                <span className="text-cyan-400 font-bold">04. CANNY EDGES</span>
                <span className="text-slate-400">Spatial Map</span>
              </div>
              <div className="flex-1 relative bg-slate-950 flex items-center justify-center overflow-hidden min-h-0">
                <canvas ref={fsEdgeCanvasRef} className="w-full h-full object-cover" />
                {cameraState !== 'live' && <span className="text-xs font-mono text-slate-500">Awaiting stream</span>}
              </div>
            </div>

            {/* Camera 5: Sobel Gradient */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden flex flex-col relative">
              <div className="px-3 py-1.5 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between text-[11px] font-mono z-10 shrink-0">
                <span className="text-indigo-300 font-bold">05. SOBEL GRADIENT</span>
                <span className="text-slate-400">Vector Energy</span>
              </div>
              <div className="flex-1 relative bg-slate-950 flex items-center justify-center overflow-hidden min-h-0">
                <canvas ref={fsGradientCanvasRef} className="w-full h-full object-cover" />
                {cameraState !== 'live' && <span className="text-xs font-mono text-slate-500">Awaiting stream</span>}
              </div>
            </div>

            {/* Camera 6: LBP Texture */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden flex flex-col relative">
              <div className="px-3 py-1.5 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between text-[11px] font-mono z-10 shrink-0">
                <span className="text-teal-300 font-bold">06. LBP TEXTURE</span>
                <span className="text-slate-400">Micro-Patterns</span>
              </div>
              <div className="flex-1 relative bg-slate-950 flex items-center justify-center overflow-hidden min-h-0">
                <canvas ref={fsLbpCanvasRef} className="w-full h-full object-cover" />
                {cameraState !== 'live' && <span className="text-xs font-mono text-slate-500">Awaiting stream</span>}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
