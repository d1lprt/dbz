/**
 * Dragon Ball Z — Kamehameha
 * MediaPipe Pose (JS/WASM) + Canvas overlay
 */

import { PoseLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ── DOM refs ───────────────────────────────────────────────────────────────────

const canvas     = document.getElementById('canvas');
const ctx        = canvas.getContext('2d');
const webcam     = document.getElementById('webcam');
const energyVid  = document.getElementById('energy');
const kameVid    = document.getElementById('kame');
const loadingEl  = document.getElementById('loading');
const loadingMsg = document.getElementById('loading-msg');
const badge      = document.getElementById('badge');

// ── State ──────────────────────────────────────────────────────────────────────

let prevState  = 'IDLE';
let idleFrames = 0;
let smoothHx   = -1;
let smoothHy   = -1;
const SMOOTH   = 0.5;

// ── Logging ────────────────────────────────────────────────────────────────────

function log(msg) {
  console.log('[DBZ]', msg);
  loadingMsg.textContent = msg;
}

// ── Geometry ───────────────────────────────────────────────────────────────────

function angle3D(a, b, c) {
  const v1 = [a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0)];
  const v2 = [c.x - b.x, c.y - b.y, (c.z ?? 0) - (b.z ?? 0)];
  const dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
  const m1  = Math.hypot(...v1);
  const m2  = Math.hypot(...v2);
  if (!m1 || !m2) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot / (m1 * m2)))) * (180 / Math.PI);
}

// ── Badge ──────────────────────────────────────────────────────────────────────

function setBadge(state) {
  if (state === 'IDLE') {
    badge.classList.remove('show', 'CHARGING', 'FIRING');
  } else {
    badge.textContent = state;
    badge.classList.add('show', state);
    badge.classList.remove(state === 'CHARGING' ? 'FIRING' : 'CHARGING');
  }
}

// ── Wait for video ready ───────────────────────────────────────────────────────

function waitForVideo(video) {
  return new Promise((resolve, reject) => {
    if (video.readyState >= 2) { resolve(); return; }

    const timeout = setTimeout(() => reject(new Error('Camera timed out after 15s')), 15000);

    const onReady = () => {
      clearTimeout(timeout);
      video.removeEventListener('loadeddata',  onReady);
      video.removeEventListener('canplay',     onReady);
      video.removeEventListener('playing',     onReady);
      resolve();
    };

    video.addEventListener('loadeddata', onReady);
    video.addEventListener('canplay',    onReady);
    video.addEventListener('playing',    onReady);
  });
}

// ── Main ───────────────────────────────────────────────────────────────────────

async function init() {
  // 1. Load WASM + model
  log('Downloading pose model… (first load ~25 MB)');
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );

  const landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/' +
        'pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
      delegate: 'GPU',
    },
    runningMode:                'VIDEO',
    numPoses:                   1,
    minPoseDetectionConfidence: 0.7,
    minPosePresenceConfidence:  0.7,
    minTrackingConfidence:      0.7,
  });

  log('Model ready. Requesting camera…');

  // 2. Camera
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
  } catch (e) {
    throw new Error(`Camera access denied: ${e.message}`);
  }

  log('Camera granted. Starting video…');
  webcam.srcObject = stream;

  // Force play (required in some browsers)
  try { await webcam.play(); } catch (_) {}

  await waitForVideo(webcam);

  log('Video ready. Starting…');

  // 3. Size canvas
  canvas.width  = webcam.videoWidth  || 1280;
  canvas.height = webcam.videoHeight || 720;

  log('done');
  loadingEl.classList.add('hidden');

  // 4. Render loop
  function loop(now) {
    render(landmarker, now);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

// ── Render ─────────────────────────────────────────────────────────────────────

function render(landmarker, timestampMs) {
  const W = canvas.width;
  const H = canvas.height;

  // Skip if video not producing frames yet
  if (webcam.readyState < 2) return;

  // Draw mirrored webcam
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(webcam, -W, 0, W, H);
  ctx.restore();

  // Darken
  ctx.fillStyle = 'rgba(0,0,0,0.5)';
  ctx.fillRect(0, 0, W, H);

  // Pose detection
  let result;
  try {
    result = landmarker.detectForVideo(webcam, timestampMs);
  } catch (_) {
    return;
  }

  let currentState = 'IDLE';
  let hx = -1, hy = -1;

  if (result.landmarks?.length > 0) {
    const lm = result.landmarks[0];

    const lShoulder = lm[11]; const rShoulder = lm[12];
    const lElbow    = lm[13]; const rElbow    = lm[14];
    const lWrist    = lm[15]; const rWrist    = lm[16];

    const lVis = lWrist.visibility ?? 0;
    const rVis = rWrist.visibility ?? 0;

    if (lVis > 0.4 || rVis > 0.4) {
      const wristDist   = Math.hypot(lWrist.x - rWrist.x, lWrist.y - rWrist.y);
      const distThresh  = prevState === 'FIRING' ? 0.5 : 0.35;

      if (wristDist < distThresh) {
        const lAngle      = angle3D(lShoulder, lElbow, lWrist);
        const rAngle      = angle3D(rShoulder, rElbow, rWrist);
        const angleThresh = prevState === 'FIRING' ? 120 : 135;

        if (Math.max(lAngle, rAngle) > angleThresh) {
          currentState = 'FIRING';
          const w = lVis > rVis ? lWrist : rWrist;
          hx = W - w.x * W;
          hy = w.y * H;
        } else {
          currentState = 'CHARGING';
          hx = W - ((lWrist.x + rWrist.x) / 2) * W;
          hy = ((lWrist.y + rWrist.y) / 2) * H;
        }
      }
    }
  }

  // Idle hysteresis
  if (currentState === 'IDLE' && prevState !== 'IDLE') {
    idleFrames++;
    if (idleFrames < 5) currentState = prevState;
  } else {
    idleFrames = 0;
  }

  // Smooth position
  if (hx !== -1) {
    if (smoothHx === -1) { smoothHx = hx; smoothHy = hy; }
    else {
      smoothHx = (1 - SMOOTH) * smoothHx + SMOOTH * hx;
      smoothHy = (1 - SMOOTH) * smoothHy + SMOOTH * hy;
    }
  } else if (currentState === 'IDLE') {
    smoothHx = -1; smoothHy = -1;
  }

  // State transitions
  if (currentState === 'CHARGING' && prevState !== 'CHARGING') {
    energyVid.currentTime = 0;
    energyVid.play().catch(() => {});
    kameVid.pause();
  }
  if (currentState === 'FIRING' && prevState !== 'FIRING') {
    kameVid.currentTime = 0;
    kameVid.play().catch(() => {});
    energyVid.pause();
  }
  if (currentState === 'IDLE') {
    energyVid.pause();
    kameVid.pause();
  }

  if (currentState !== prevState) setBadge(currentState);
  prevState = currentState;

  // Draw effect overlays
  if (smoothHx === -1) return;

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';

  if (currentState === 'CHARGING' && energyVid.readyState >= 2) {
    const size = H * 0.35;
    ctx.drawImage(energyVid, smoothHx - size / 2, smoothHy - size / 2, size, size);

  } else if (currentState === 'FIRING' && kameVid.readyState >= 2) {
    const origW = kameVid.videoWidth  || 1;
    const origH = kameVid.videoHeight || 1;
    const beamH = H * 0.75;
    const beamW = beamH * (origW / origH);
    ctx.drawImage(kameVid, smoothHx, smoothHy - beamH / 2, beamW, beamH);
  }

  ctx.restore();
}

// ── Boot ───────────────────────────────────────────────────────────────────────

init().catch(err => {
  console.error('[DBZ] Fatal:', err);
  loadingMsg.style.color = '#ff6b6b';
  loadingMsg.textContent = `Error: ${err.message}`;
});
