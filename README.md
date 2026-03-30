# Dragon Ball Z — Kamehameha Web App

Real-time pose detection in the browser using MediaPipe JS + WebRTC.  
Flask serves the page; all AI processing runs client-side (no server GPU needed).

## Local setup

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## File structure

```
project/
├── app.py
├── requirements.txt
├── Procfile
├── templates/
│   └── index.html
└── static/
    ├── main.js
    └── assets/
        ├── energy.mp4        ← copy from your original assets/
        └── kamehameha.mp4    ← copy from your original assets/
```

> **Note:** `pose_landmarker_heavy.task` is loaded automatically from  
> Google's CDN — you no longer need to ship it yourself.

## Deploy to Render (free tier)

1. Push this folder to a GitHub repo.
2. Go to https://render.com → **New Web Service** → connect your repo.
3. Set:
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app`
4. Click **Deploy**.

> Webcam access requires HTTPS — Render provides this automatically.

## Deploy to Railway

1. Push to GitHub.
2. Go to https://railway.app → **New Project → Deploy from GitHub repo**.
3. Railway auto-detects the Procfile — no extra config needed.

## How it works

| Layer | Technology |
|---|---|
| Camera capture | `navigator.mediaDevices.getUserMedia` |
| Pose detection | MediaPipe Pose (WASM, runs in browser) |
| Rendering | HTML5 Canvas (`globalCompositeOperation: 'lighter'` = additive blend) |
| Server | Flask (only serves HTML + static files) |

## Pose states

| State | Trigger |
|---|---|
| **CHARGING** | Wrists close together, elbows bent |
| **FIRING** | Wrists close + at least one arm extended (>135°) |
