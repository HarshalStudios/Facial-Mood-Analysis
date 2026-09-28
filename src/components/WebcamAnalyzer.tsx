import React, { useState, useRef, useEffect } from 'react';
import { Camera, Upload, Play, Square, RefreshCw, AlertCircle, CheckCircle2, Sliders, Eye } from 'lucide-react';

interface PredictionData {
  frame_id: number;
  timestamp: string;
  face_detected: boolean;
  prediction_available: boolean;
  predicted_emotion: string;
  confidence: number;
  probabilities: Record<string, number>;
  processing_time_ms: number;
  primary_face_bbox?: { x: number; y: number; width: number; height: number };
  quality?: {
    brightness: number;
    contrast: number;
    sharpness: number;
    is_valid: boolean;
  };
}

export const WebcamAnalyzer: React.FC = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<'webcam' | 'upload'>('webcam');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [autoPoll, setAutoPoll] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const timerRef = useRef<any>(null);

  const startWebcam = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsStreaming(true);
      }
    } catch (err: any) {
      setError('Unable to access webcam. Please check browser permissions or switch to image upload mode.');
      setIsStreaming(false);
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      setAutoPoll(false);
    }
  };

  const captureAndPredict = async () => {
    if (!videoRef.current && !selectedImage) return;
    setLoading(true);

    try {
      let base64Data = '';
      if (mode === 'webcam' && videoRef.current) {
        const canvas = document.createElement('canvas');
        canvas.width = videoRef.current.videoWidth || 640;
        canvas.height = videoRef.current.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
          base64Data = canvas.toDataURL('image/jpeg', 0.85);
        }
      } else if (selectedImage) {
        base64Data = selectedImage;
      }

      const res = await fetch('/api/predict/temporal/image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Data })
      });

      if (!res.ok) throw new Error('Prediction API failed');
      const data = await res.json();
      setPrediction({
        ...data,
        predicted_emotion: data.smoothed_emotion || data.raw_emotion,
        confidence: data.smoothed_confidence || data.raw_confidence,
        probabilities: data.smoothed_probabilities || data.raw_probabilities
      });
    } catch (err: any) {
      setError(err.message || 'Error communicating with prediction backend');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (autoPoll && isStreaming) {
      timerRef.current = setInterval(() => {
        captureAndPredict();
      }, 1500);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [autoPoll, isStreaming]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      setSelectedImage(event.target?.result as string);
      setMode('upload');
      stopWebcam();
    };
    reader.readAsDataURL(file);
  };

  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* Left Column: Input Stream / Viewport */}
      <div className="lg:col-span-7 bg-white rounded-xl border border-slate-200 p-6 shadow-sm flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-base font-bold text-slate-900">Live Video & Image Viewport</h2>
            <p className="text-xs text-slate-500">Real-time RGB frame capture and multi-representation processing</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => { setMode('webcam'); setSelectedImage(null); }}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                mode === 'webcam' ? 'bg-cyan-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              Webcam Feed
            </button>
            <label className={`px-3 py-1.5 text-xs font-medium rounded-lg cursor-pointer transition-colors ${
              mode === 'upload' ? 'bg-cyan-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}>
              Upload Image
              <input type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
            </label>
          </div>
        </div>

        {/* Viewport Screen */}
        <div className="relative aspect-video bg-slate-900 rounded-lg overflow-hidden flex items-center justify-center border border-slate-800">
          {mode === 'webcam' ? (
            <>
              <video ref={videoRef} className={`w-full h-full object-cover ${!isStreaming ? 'hidden' : ''}`} playsInline muted />
              {!isStreaming && (
                <div className="text-center p-6">
                  <Camera className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                  <p className="text-sm font-medium text-slate-300">Webcam feed is currently inactive</p>
                  <p className="text-xs text-slate-500 mt-1 mb-4">Click start below to initialize RGB camera capture</p>
                  <button
                    onClick={startWebcam}
                    className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white text-xs font-semibold rounded-lg transition-colors inline-flex items-center gap-2"
                  >
                    <Play className="w-4 h-4" /> Start Webcam Stream
                  </button>
                </div>
              )}
            </>
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              {selectedImage ? (
                <img src={selectedImage} alt="Uploaded sample" className="max-h-full max-w-full object-contain" />
              ) : (
                <div className="text-center p-6">
                  <Upload className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                  <p className="text-sm font-medium text-slate-300">No image uploaded</p>
                  <p className="text-xs text-slate-500 mt-1">Select an image file to analyze facial mood</p>
                </div>
              )}
            </div>
          )}

          {/* Bounding box overlay if detected */}
          {prediction?.face_detected && prediction.primary_face_bbox && (
            <div
              className="absolute border-2 border-cyan-400 bg-cyan-400/10 pointer-events-none transition-all duration-200"
              style={{
                left: `${(prediction.primary_face_bbox.x / 640) * 100}%`,
                top: `${(prediction.primary_face_bbox.y / 480) * 100}%`,
                width: `${(prediction.primary_face_bbox.width / 640) * 100}%`,
                height: `${(prediction.primary_face_bbox.height / 480) * 100}%`
              }}
            >
              <span className="absolute -top-6 left-0 bg-cyan-600 text-white text-[10px] font-mono px-1.5 py-0.5 rounded">
                FACE DETECTED ({Math.round(prediction.confidence * 100)}%)
              </span>
            </div>
          )}
        </div>

        {/* Controls Toolbar */}
        <div className="mt-4 flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-2">
            {mode === 'webcam' && isStreaming && (
              <>
                <button
                  onClick={stopWebcam}
                  className="px-3.5 py-2 bg-rose-600 hover:bg-rose-700 text-white text-xs font-semibold rounded-lg transition-colors inline-flex items-center gap-1.5"
                >
                  <Square className="w-3.5 h-3.5" /> Stop Stream
                </button>
                <button
                  onClick={() => setAutoPoll(!autoPoll)}
                  className={`px-3.5 py-2 text-xs font-semibold rounded-lg transition-colors inline-flex items-center gap-1.5 ${
                    autoPoll ? 'bg-amber-600 hover:bg-amber-700 text-white' : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
                  }`}
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${autoPoll ? 'animate-spin' : ''}`} />
                  {autoPoll ? 'Auto-Polling Active' : 'Enable Auto-Poll'}
                </button>
              </>
            )}
            {(mode === 'upload' || (mode === 'webcam' && isStreaming)) && (
              <button
                onClick={captureAndPredict}
                disabled={loading}
                className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white text-xs font-semibold rounded-lg transition-colors inline-flex items-center gap-1.5 disabled:opacity-50"
              >
                <Eye className="w-3.5 h-3.5" /> {loading ? 'Analyzing...' : 'Run Single Frame Prediction'}
              </button>
            )}
          </div>

          {prediction && (
            <div className="text-xs font-mono text-slate-500">
              Latency: {prediction.processing_time_ms}ms | Frame #{prediction.frame_id}
            </div>
          )}
        </div>

        {error && (
          <div className="mt-3 p-3 bg-rose-50 border border-rose-200 rounded-lg flex items-center gap-2 text-xs text-rose-700">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Right Column: Prediction & Probability Breakdown */}
      <div className="lg:col-span-5 space-y-6">
        {/* Dominant Emotion Card */}
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
          <h3 className="text-xs font-mono uppercase tracking-wider text-slate-400 mb-2">Dominant Expression</h3>
          {prediction ? (
            <div className="flex items-center justify-between">
              <div>
                <span className="text-3xl font-bold tracking-tight text-slate-900 capitalize block">
                  {prediction.predicted_emotion}
                </span>
                <span className="text-xs text-slate-500 mt-1 block">
                  Confidence: <strong className="font-mono text-slate-700">{(prediction.confidence * 100).toFixed(1)}%</strong>
                </span>
              </div>
              <div className="w-16 h-16 rounded-2xl bg-cyan-50 border border-cyan-200 flex items-center justify-center text-cyan-600 font-mono text-xl font-bold shadow-sm">
                {Math.round(prediction.confidence * 100)}%
              </div>
            </div>
          ) : (
            <div className="py-6 text-center text-slate-400 text-sm">
              Run prediction or start stream to view real-time mood analysis
            </div>
          )}
        </div>

        {/* 7 Canonical Emotion Probability Distribution */}
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
          <h3 className="text-xs font-mono uppercase tracking-wider text-slate-400 mb-4">Class Probability Distribution</h3>
          {prediction && prediction.probabilities ? (
            <div className="space-y-3">
              {Object.entries(prediction.probabilities).map(([emotion, prob]) => {
                const percentage = Math.round(prob * 100);
                const isDominant = emotion === prediction.predicted_emotion;
                return (
                  <div key={emotion}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className={`capitalize font-medium ${isDominant ? 'text-cyan-700 font-bold' : 'text-slate-600'}`}>
                        {emotion}
                      </span>
                      <span className="font-mono text-slate-700">{percentage}%</span>
                    </div>
                    <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className={`h-full transition-all duration-300 ${isDominant ? 'bg-cyan-600' : 'bg-slate-400'}`}
                        style={{ width: `${percentage}%` }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="py-8 text-center text-slate-400 text-sm">
              Probabilities will render here upon successful inference.
            </div>
          )}
        </div>

        {/* Quality Metrics */}
        {prediction?.quality && (
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
            <h3 className="text-xs font-mono uppercase tracking-wider text-slate-400 mb-3">Image Quality Telemetry</h3>
            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                <span className="text-[10px] font-mono text-slate-400 block uppercase">Brightness</span>
                <span className="text-sm font-mono font-bold text-slate-800">{prediction.quality.brightness.toFixed(1)}</span>
              </div>
              <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                <span className="text-[10px] font-mono text-slate-400 block uppercase">Contrast</span>
                <span className="text-sm font-mono font-bold text-slate-800">{prediction.quality.contrast.toFixed(1)}</span>
              </div>
              <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                <span className="text-[10px] font-mono text-slate-400 block uppercase">Sharpness</span>
                <span className="text-sm font-mono font-bold text-slate-800">{prediction.quality.sharpness.toFixed(1)}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
