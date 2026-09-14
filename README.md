# CycloneWatch — AI-Powered Tropical Cyclone Tracker

**Smart India Hackathon 2026 Submission · PS70**  
*AI/ML-based system for identification, classification, and prediction of tropical cyclone patterns using multi-source satellite data.*

---

## 🌐 Live Deployments
| Service | URL | Status |
|---------|-----|--------|
| Frontend Dashboard | [sih-26-one.vercel.app](https://sih-26-one.vercel.app/) | ✅ Live |
| Backend API (Render) | [sih26-o6nv.onrender.com/health](https://sih26-o6nv.onrender.com/health) | ✅ Live |
| API Docs (Swagger) | [sih26-o6nv.onrender.com/docs](https://sih26-o6nv.onrender.com/docs) | ✅ Live |

---

## 🌪️ What Is CycloneWatch?

CycloneWatch is an end-to-end AI-driven meteorological tracking system addressing the **interpretation gap** in traditional weather forecasting.

Traditional Numerical Weather Prediction (NWP) physics models are highly accurate but slow to respond to:
- **Rapid Intensification (RI)** events (wind speed increasing ≥ 30 knots in 24h)
- **Anomalous low-latitude storm formations** (like Cyclone Ockhi 2017)

CycloneWatch applies deep convolutional neural networks directly to infrared and water-vapor satellite imagery to:
1. **Automatically classify** the structural morphology of any tropical storm in real-time
2. **Detect dangerous transitions** (like banding → eye formation) hours before NWP models
3. **Visualise historical lifecycle data** across 7 major North Indian Ocean cyclones

---

## 📂 Repository Structure

```
SIH26/
├── backend/           # FastAPI server, SQLite DB, ML inference endpoints
│   ├── app/
│   │   ├── api/       # Route handlers: /classify, /predict, /replay, /metrics
│   │   ├── core/      # Config, settings
│   │   ├── db/        # SQLAlchemy async engine & session management
│   │   ├── models/    # DB ORM models
│   │   ├── schemas/   # Pydantic request/response schemas
│   │   └── services/  # classify_service.py, ml_adapter.py
│   ├── scripts/       # seed_db.py, precompute_replay.py
│   └── cyclonewatch.db # SQLite database (precomputed replay data)
│
├── ml/                # PyTorch ML pipeline
│   ├── src/
│   │   ├── model.py   # CycloneCNN + CycloneTemporalModel (CNN+GRU)
│   │   ├── train.py   # Training loop with class-weighted CE loss
│   │   ├── dataset.py # CycloneDataset PyTorch Dataset class
│   │   └── evaluate.py
│   ├── inference.py   # Public predict_frame() / predict_sequence() API
│   ├── checkpoints/   # model.pt (trained weights)
│   └── configs/       # model_config.json, evaluation_metrics.json
│
├── data/              # Satellite data pipeline
│   ├── raw/           # Downloaded GridSat-B1 NetCDF files
│   ├── normalized/    # Per-frame NPZ tensors (output of standardize_data.py)
│   ├── ground_truth/  # IBTrACS best-track CSVs + ground_truth_labels.csv
│   ├── training_manifest.csv
│   └── metadata.csv
│
├── frontend/          # React + TypeScript + Leaflet SPA
│   └── src/
│       ├── App.tsx               # Root layout (70:30 map:metrics split)
│       ├── store/useCycloneStore.ts # Zustand state manager
│       ├── data/cyclones.ts       # Cyclone metadata & pattern taxonomy
│       └── components/Dashboard/
│           ├── SatellitePanel.tsx  # Leaflet map with overlays & live feed
│           ├── MetricsPanel.tsx    # Right-side metrics: intensity, impact
│           ├── EvidenceDrawer.tsx  # Source provenance slide-out
│           ├── Timeline.tsx        # Historical frame scrubber (IST)
│           └── LeafletMap.tsx      # Map component with trajectory layers
│
├── docs/              # Technical documentation
│   ├── future_implementation.md  # Comprehensive roadmap (THIS FILE LINKS HERE)
│   ├── taxonomy.md               # The 5-class morphology taxonomy
│   ├── metrics_explained.md      # Dashboard metric definitions
│   ├── model_explained.md        # ML architecture deep-dive
│   ├── api_contract.md           # Full API schema contract
│   └── ...
│
├── PROJECT_EXPLAINER.md  # Non-technical full project explainer (read this first)
└── README.md             # This file
```

---

## 🚀 Local Development Setup

> **Prerequisite:** Python 3.11+, Node.js 18+, and Git installed.

### Step 1 — Clone the repo
```bash
git clone <repo-url>
cd SIH26
```

### Step 2 — Backend Setup (FastAPI Server)

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install all Python dependencies
pip install -r requirements.txt
```

**Configure environment variables:**
```bash
# Copy the example environment file
cp .env.example .env
```

Open `.env` and set the following key variables (defaults work for local SQLite):
```env
# The default uses SQLite — no Postgres required locally
DATABASE_URL=sqlite+aiosqlite:///./cyclonewatch.db
DATABASE_SYNC_URL=sqlite:///./cyclonewatch.db

# Keep ML in stub mode until you have a trained checkpoint
ML_FORCE_STUB=false

CORS_ORIGINS=http://localhost:5173
DEBUG=true
```

**Seed the database and precompute replay data:**
```bash
# Run from the backend/ directory
python -m scripts.seed_db

# Then precompute ML replay for all historical events
python -m scripts.precompute_replay
```

**Start the API server:**
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

API is now live at: `http://127.0.0.1:8000`  
Swagger docs at: `http://127.0.0.1:8000/docs`

---

### Step 3 — Frontend Setup (React Dashboard)

Open a new terminal:
```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Dashboard is now live at: `http://localhost:5173`

---

### Step 4 — (Optional) ML Training

To train or retrain the model yourself:
```bash
cd <repo-root>

# Activate the backend venv (ML package lives in the repo root)
cd backend && .venv\Scripts\activate && cd ..

# Run the training loop (requires data/normalized/ frames to be present)
python -m ml.src.train
```

Training config: 200 epochs, Adam optimizer, class-weighted cross-entropy loss, `ReduceLROnPlateau` scheduler.  
Checkpoint is saved to `ml/checkpoints/model.pt`.

---

### Step 5 — (Optional) Docker Compose (Full Stack)

```bash
# From the backend/ directory
docker-compose up --build
```

This spins up the FastAPI server + PostgreSQL container together.

---

## 📊 Current Model Performance

| Metric | Value |
|--------|-------|
| Validation samples | 60 frames |
| Pattern classification accuracy | **78.3%** |
| Center tracking MAE (T+12h) | **255.5 km** |
| Center tracking Median Error | 239.7 km |
| F1 — Eye | 1.00 |
| F1 — Banding | 0.79 |
| F1 — Curved Band | 0.55 |
| F1 — Shear-Affected | 0.80 |
| F1 — Disorganized | 0.88 |
| Inference speed | ~12 ms/frame (CPU) |

> Training set: 7 cyclones, 423 total labeled frames.  
> Baseline persistence T+12 MAE: 255 km (model is at baseline — upgrading with ConvLSTM in Phase 2).

---

## 🌊 Training Cyclones

| Event | Year | Basin | Peak Wind | Category | IMD Gap Case |
|-------|------|-------|-----------|----------|--------------|
| Biparjoy | 2023 | Arabian Sea | 165 km/h | VSCS | ✅ 24h early |
| Amphan | 2020 | Bay of Bengal | 240 km/h | SUCS | ✅ 18h early |
| Fani | 2019 | Bay of Bengal | 213 km/h | ESCS | ✅ 12h recurvature |
| Tauktae | 2021 | Arabian Sea | 185 km/h | ESCS | ✅ 30h early RI |
| Ockhi | 2017 | Arabian Sea | 165 km/h | VSCS | ✅ **CRITICAL — 36h late** |
| Hudhud | 2014 | Bay of Bengal | 185 km/h | ESCS | ✅ 24h early |
| Phailin | 2013 | Bay of Bengal | 215 km/h | ESCS | ✅ Structural validation |

---

## 🏗️ Architecture Overview

```
Satellite Data (INSAT/GridSat-B1)
        ↓
  standardize_data.py
        ↓
  NPZ tensors [C, H, W]
        ↓
  CycloneCNN → pattern label + center lat/lon
        ↓
  FastAPI backend (seed_db + precompute_replay)
        ↓
  SQLite (cyclonewatch.db)
        ↓
  React frontend (useCycloneStore → Zustand)
        ↓
  SatellitePanel (Leaflet map) + MetricsPanel
```

---

## 🔮 Future Roadmap

**Immediate Next Steps (Phase 2):**
1. **ISRO MOSDAC Integration** — 1km INSAT-3DR imagery (16× spatial resolution upgrade)
2. **ConvLSTM Temporal Forecasting** — Replace persistence fallback with learned T+12 / T+24 / T+48 predictions
3. **Expanded Training Pipeline** — 30–50 cyclone events from EU ECMWF + ISRO archives

**Vision (Phase 4–5):**
4. **M+G+S Multi-Task Impact Engine** — Predict Ground Damage Types & Severity alongside morphology
5. **3D CesiumJS Globe UI** — Replace Leaflet 2D map with WebGL-powered 3D Earth

📖 **Full detailed plan:** [docs/future_implementation.md](docs/future_implementation.md)

---

## 📖 Documentation Map

| Topic | Document |
|-------|---------|
| Non-technical full project explainer | [PROJECT_EXPLAINER.md](PROJECT_EXPLAINER.md) |
| Future roadmap & implementation plan | [docs/future_implementation.md](docs/future_implementation.md) |
| The 5-class pattern taxonomy | [docs/taxonomy.md](docs/taxonomy.md) |
| Dashboard metric definitions | [docs/metrics_explained.md](docs/metrics_explained.md) |
| ML model architecture deep-dive | [docs/model_explained.md](docs/model_explained.md) |
| Full API schema contract | [docs/api_contract.md](docs/api_contract.md) |
| Ockhi disaster case study | [docs/ockhi_analysis.md](docs/ockhi_analysis.md) |
| Backend setup (technical) | [backend/README.md](backend/README.md) |
| ML pipeline setup | [ml/README.md](ml/README.md) |
| Frontend component guide | [frontend/README.md](frontend/README.md) |
| Backend (non-technical) | [backend/EXPLAINER.md](backend/EXPLAINER.md) |
| ML brain (non-technical) | [ml/EXPLAINER.md](ml/EXPLAINER.md) |
| Data pipeline (non-technical) | [data/EXPLAINER.md](data/EXPLAINER.md) |
