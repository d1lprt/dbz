/**
 * Dragon Ball Z — Kamehameha
 * MediaPipe Pose (JS/WASM) + Canvas overlay
 *
 * State machine:  IDLE → CHARGING → FIRING
 *
 * Pose rules (mirrors the original Python logic):
 *   CHARGING : wrists close together + elbows partially bent
 *   FIRING   : wrists still close + at least one arm fully extended
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

// ── Badge helper ───────────────────────────────────────────────────────────────

function setBadge(state) {
  if (state === 'IDLE') {
    badge.classList.remove('show', 'CHARGING', 'FIRING');
  } else {
    badge.textContent = state;
    badge.classList.add('show', state);
    badge.classList.remove(state === 'CHARGING' ? 'FIRING' : 'CHARGING');
  }
}

// ── Main ───────────────────────────────────────────────────────────────────────

async function init() {
  // 1. Load WASM fileset + model
  loadingMsg.textContent = 'Downloading pose model… (one-time, ~25 MB)';

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
    runningMode:                  'VIDEO',
    numPoses:                     1,
    minPoseDetectionConfidence:   0.7,
    minPosePresenceConfidence:    0.7,
    minTrackingConfidence:        0.7,
  });

  // 2. Start webcam
  loadingMsg.textContent = 'Requesting camera access…';
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
  });
  webcam.srcObject = stream;
  await new Promise(r => webcam.addEventListener('loadeddata', r, { once: true }));

  canvas.width  = webcam.videoWidth;
  canvas.height = webcam.videoHeight;

  loadingEl.classList.add('hidden');

  // 3. Start render loop
  let lastT = -1;
  function loop(now) {
    // Only re-run detection when there's a new video frame
    if (webcam.currentTime !== lastT) {
      lastT = webcam.currentTime;
      render(landmarker, now);
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

// ── Render ─────────────────────────────────────────────────────────────────────

function render(landmarker, timestampMs) {
  const W = canvas.width;
  const H = canvas.height;

  // ── Draw mirrored webcam ──────────────────────────────────────────────────
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(webcam, -W, 0, W, H);
  ctx.restore();

  // ── Darken (replicates cv2.convertScaleAbs alpha=0.5) ────────────────────
  ctx.fillStyle = 'rgba(0,0,0,0.5)';
  ctx.fillRect(0, 0, W, H);

  // ── Pose detection ────────────────────────────────────────────────────────
  const result = landmarker.detectForVideo(webcam, timestampMs);

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
      const wristDist  = Math.hypot(lWrist.x - rWrist.x, lWrist.y - rWrist.y);
      const distThresh = prevState === 'FIRING' ? 0.5 : 0.35;

      if (wristDist < distThresh) {
        const lAngle     = angle3D(lShoulder, lElbow, lWrist);
        const rAngle     = angle3D(rShoulder, rElbow, rWrist);
        const angleThresh = prevState === 'FIRING' ? 120 : 135;

        if (Math.max(lAngle, rAngle) > angleThresh) {
          // Arm(s) extended → FIRING
          currentState = 'FIRING';
          const w  = lVis > rVis ? lWrist : rWrist;
          hx = W - w.x * W;   // flip x for mirrored display
          hy = w.y * H;
        } else {
          // Arms bent, wrists together → CHARGING
          currentState = 'CHARGING';
          hx = W - ((lWrist.x + rWrist.x) / 2) * W;
          hy = ((lWrist.y + rWrist.y) / 2) * H;
        }
      }
    }
  }

  // ── Idle hysteresis (5-frame grace period) ────────────────────────────────
  if (currentState === 'IDLE' && prevState !== 'IDLE') {
    idleFrames++;
    if (idleFrames < 5) currentState = prevState;
  } else {
    idleFrames = 0;
  }

  // ── Smooth position ───────────────────────────────────────────────────────
  if (hx !== -1) {
    if (smoothHx === -1) {
      smoothHx = hx; smoothHy = hy;
    } else {
      smoothHx = (1 - SMOOTH) * smoothHx + SMOOTH * hx;
      smoothHy = (1 - SMOOTH) * smoothHy + SMOOTH * hy;
    }
  } else if (currentState === 'IDLE') {
    smoothHx = -1; smoothHy = -1;
  }

  // ── State transitions ─────────────────────────────────────────────────────
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

  // ── Draw effect overlays (additive blend = cv2.add) ───────────────────────
  if (smoothHx === -1) return;

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';

  if (currentState === 'CHARGING' && energyVid.readyState >= 2) {
    const size = H * 0.35;
    ctx.drawImage(energyVid, smoothHx - size / 2, smoothHy - size / 2, size, size);

  } else if (currentState === 'FIRING' && kameVid.readyState >= 2) {
    const origW  = kameVid.videoWidth  || 1;
    const origH  = kameVid.videoHeight || 1;
    const beamH  = H * 0.75;
    const beamW  = beamH * (origW / origH);
    // Left edge of beam starts at the wrist, extends to the right
    ctx.drawImage(kameVid, smoothHx, smoothHy - beamH / 2, beamW, beamH);
  }

  ctx.restore();
}

// ── Boot ───────────────────────────────────────────────────────────────────────

init().catch(err => {
  console.error(err);
  loadingMsg.textContent = `Error: ${err.message}`;
});
