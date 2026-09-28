# Facial Mood Analysis — Multi-Representation Real-Time System

Multi-Representation Based Real-Time Facial Mood Analysis System Using a Single RGB Camera.

## Architecture

```
Browser (React + Vite)
       ↓
Express Proxy Layer (server.ts)
       ↓
FastAPI Backend (Python / Uvicorn on port 8000)
       ↓
Face Detection → Representation Engine → 22D Feature Extraction → Fusion Model
```

---

## Local Development Setup

### Requirements

- Node.js (v18+)
- npm
- Python (v3.10+)

### 1. Backend Setup (Python)

Create and activate a virtual environment, then install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the FastAPI backend server:

```bash
python run.py
```

### 2. Frontend & Proxy Setup (Node.js)

In a separate terminal window, install Node.js dependencies and start the development server:

```bash
npm install
npm run dev
```

The application will be accessible at `http://localhost:3000`.

---

## Environment Configuration

Copy `.env.example` to `.env` and configure your API URL if running the backend on a custom host/port:

```env
VITE_API_URL=http://localhost:8000
```

---

## Production Build & Deployment

To install dependencies and build the React frontend for production:

```bash
npm ci
npm run build
```

The compiled static assets are generated in `dist/`.

To start the production server:

```bash
npm start
```
