const API_BASE = (import.meta.env.VITE_API_URL || '') + '/api';

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function checkReady() {
  const res = await fetch(`${API_BASE}/ready`);
  return res.json();
}

export async function getStatus() {
  const res = await fetch(`${API_BASE}/status`);
  return res.json();
}

export async function getExperiments() {
  const res = await fetch(`${API_BASE}/experiments`);
  return res.json();
}

export async function predictImage(base64Image: string) {
  const res = await fetch(`${API_BASE}/predict/base64`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: base64Image })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Prediction failed');
  }
  return res.json();
}

export async function predictTemporalImage(base64Image: string) {
  const res = await fetch(`${API_BASE}/predict/temporal/image`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: base64Image })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Temporal prediction failed');
  }
  return res.json();
}

export async function resetTemporalEngine() {
  const res = await fetch(`${API_BASE}/predict/temporal/reset`, { method: 'POST' });
  return res.json();
}
