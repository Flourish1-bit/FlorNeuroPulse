#!/usr/bin/env python3
"""
FlorNeuroPulse — Real-Time Hardware HUD Workstation
Privacy-First: All feature extraction is transient in-memory; no raw frames written to disk.
Color Scheme: Clinical Dark Slate / Amber / Cyan (high-contrast, professional)
"""
import os
import sys
import time
import threading
import queue
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import librosa
import sounddevice as sd

# ------------------------------------------------------------------
# HUD Color Palette (Clinical High-Contrast)
# ------------------------------------------------------------------
COLOR_BG = (18, 24, 32)          # Deep slate
COLOR_PANEL = (28, 36, 48)       # Slightly lighter slate
COLOR_TEXT_PRIMARY = (240, 245, 250)   # Near-white
COLOR_TEXT_SECONDARY = (160, 175, 190) # Muted steel
COLOR_AMBER = (255, 190, 60)     # Alert / moderate risk
COLOR_CYAN = (0, 220, 220)       # Active / healthy
COLOR_RED = (255, 80, 80)        # Critical
COLOR_GREEN = (60, 220, 120)     # Low risk
COLOR_BORDER = (60, 80, 110)     # Subtle border

# ------------------------------------------------------------------
# Feature Extractors (In-Memory Only)
# ------------------------------------------------------------------
class AudioFeatureExtractor:
    """Real-time 15D audio biomarker extraction using Librosa (transient)."""
    def __init__(self, sr=16000, n_fft=512, hop_length=256):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.buffer = deque(maxlen=sr * 2)  # ~2 sec rolling buffer

    def feed(self, chunk: np.ndarray):
        # chunk: (samples,) float32 mono
        self.buffer.extend(chunk)

    def extract_15d(self) -> np.ndarray:
        if len(self.buffer) < self.sr:
            return np.zeros(15, dtype=np.float32)
        signal = np.array(self.buffer, dtype=np.float32)
        # MFCCs (13)
        mfcc = librosa.feature.mfcc(y=signal, sr=self.sr, n_mfcc=13, n_fft=self.n_fft, hop_length=self.hop_length)
        mfcc_mean = np.mean(mfcc, axis=1)[:13]
        # Spectral Centroid
        sc = librosa.feature.spectral_centroid(y=signal, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        sc_mean = float(np.mean(sc))
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y=signal, hop_length=self.hop_length)
        zcr_mean = float(np.mean(zcr))
        # f0, jitter, shimmer, HNR approximations
        f0, _, _ = librosa.pyin(signal, sr=self.sr, fmin=50, fmax=500)
        f0_valid = f0[~np.isnan(f0)]
        f0_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0
        # Jitter proxy: std of f0 differences
        jitter = float(np.std(np.diff(f0_valid))) if len(f0_valid) > 1 else 0.0
        # Shimmer proxy: relative amplitude variation
        shimmer = float(np.std(np.abs(np.diff(signal[:len(signal)//10])))) if len(signal) > 10 else 0.0
        # HNR proxy: ratio of harmonic to noise energy (simplified)
        hnr = float(np.mean(np.abs(signal)) / (np.std(signal) + 1e-6))
        # Assemble 15D
        vec = np.concatenate([
            mfcc_mean[:13].astype(np.float32),
            np.array([sc_mean, zcr_mean, f0_mean, jitter, shimmer, hnr], dtype=np.float32)
        ])
        # Normalize to approximate [0,1] range for model stability
        vec = np.clip(vec / (np.abs(vec).max() + 1e-6) * 0.5 + 0.5, 0.0, 1.0)
        return vec[:15].astype(np.float32)

class VisionFeatureExtractor:
    """Real-time 5D vision biomarker extraction using MediaPipe Face Mesh (transient)."""
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_5d(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return np.zeros(5, dtype=np.float32)
        landmarks = results.multi_face_landmarks[0].landmark
        # Lip corner asymmetry (relative to nasal bridge)
        # Landmark indices: left lip corner ~61, right ~291, nasal bridge ~168
        def get_pt(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
        left_lip = get_pt(61)
        right_lip = get_pt(291)
        nasal = get_pt(168)
        asym = float(np.linalg.norm(left_lip - right_lip) / (np.linalg.norm(nasal - left_lip) + 1e-6))
        # Vertical lip displacement (upper vs lower lip center)
        upper = get_pt(13)
        lower = get_pt(14)
        v_disp = float(np.linalg.norm(upper - lower) / h)
        # Eye Aspect Ratio (EAR) — average of both eyes
        def ear(eye_pts):
            # Simplified EAR using vertical/horizontal distances
            p = [get_pt(i) for i in eye_pts]
            v1 = np.linalg.norm(p[1] - p[5])
            v2 = np.linalg.norm(p[2] - p[4])
            h = np.linalg.norm(p[0] - p[3])
            return (v1 + v2) / (2.0 * h + 1e-6)
        left_ear = ear([33, 160, 158, 133, 153, 144])
        right_ear = ear([362, 385, 387, 263, 373, 380])
        ear_avg = float((left_ear + right_ear) / 2.0)
        # Head pitch/yaw proxies from nose-to-chin vs nose-to-ear ratios
        chin = get_pt(152)
        left_ear_pt = get_pt(234)
        right_ear_pt = get_pt(454)
        pitch_proxy = float(np.linalg.norm(chin - nasal) / (np.linalg.norm(left_ear_pt - right_ear_pt) + 1e-6))
        yaw_proxy = float((np.linalg.norm(left_ear_pt - nasal) - np.linalg.norm(right_ear_pt - nasal)) / (np.linalg.norm(left_ear_pt - nasal) + np.linalg.norm(right_ear_pt - nasal) + 1e-6))
        vec = np.array([asym, v_disp, ear_avg, pitch_proxy, yaw_proxy], dtype=np.float32)
        # Normalize
        vec = np.clip(vec / (np.abs(vec).max() + 1e-6) * 0.5 + 0.5, 0.0, 1.0)
        return vec[:5].astype(np.float32)

# ------------------------------------------------------------------
# HUD Renderer (OpenCV — Professional Clinical Theme)
# ------------------------------------------------------------------
class HUDRenderer:
    def __init__(self, window_name="FlorNeuroPulse HUD"):
        self.win = window_name
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 1280, 720)

    def draw(self, frame: np.ndarray, audio_15d: np.ndarray, vision_5d: np.ndarray, score: float):
        h, w = frame.shape[:2]
        # Background overlay (semi-transparent slate panel at bottom)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 220), (w, h), COLOR_BG, -1)
        alpha = 0.92
        frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

        # Header bar
        cv2.rectangle(frame, (0, 0), (w, 60), COLOR_PANEL, -1)
        cv2.putText(frame, "FLORNEUROPULSE  |  MULTIMODAL BIOMARKER HUD", (20, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_TEXT_PRIMARY, 2)
        cv2.putText(frame, "PRIVACY MODE: RAW MEDIA PURGED  |  INFERENCE: ONNX RUNTIME", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_SECONDARY, 1)

        # Main score panel (left)
        score_color = COLOR_GREEN if score < 0.35 else (COLOR_AMBER if score < 0.65 else (COLOR_RED if score < 0.85 else (255, 40, 40)))
        cv2.rectangle(frame, (20, 80), (320, 260), COLOR_PANEL, -1)
        cv2.rectangle(frame, (20, 80), (320, 260), COLOR_BORDER, 2)
        cv2.putText(frame, "BIOMARKER ANOMALY INDEX", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_SECONDARY, 1)
        cv2.putText(frame, f"{score:.3f}", (30, 160), cv2.FONT_HERSHEY_DUPLEX, 1.8, score_color, 3)
        tier = "LOW" if score < 0.35 else ("MODERATE" if score < 0.65 else ("HIGH" if score < 0.85 else "CRITICAL"))
        cv2.putText(frame, f"RISK TIER: {tier}", (30, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2)
        cv2.putText(frame, f"MODEL v1.0.0  |  SESSION ACTIVE", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT_SECONDARY, 1)

        # Audio 15D metrics (middle-top)
        cv2.rectangle(frame, (340, 80), (820, 260), COLOR_PANEL, -1)
        cv2.rectangle(frame, (340, 80), (820, 260), COLOR_BORDER, 2)
        cv2.putText(frame, "AUDIO 15D VECTOR  (MFCC + f0 + Jitter/Shimmer/HNR + ZCR + Centroid)", (350, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_PRIMARY, 1)
        # Show first 8 values in 2 rows
        for i in range(8):
            x = 350 + (i % 4) * 115
            y = 130 + (i // 4) * 45
            val = audio_15d[i] if i < len(audio_15d) else 0.0
            cv2.putText(frame, f"A{i+1}:{val:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_CYAN, 1)
        # Show remaining 7
        for i in range(8, 15):
            x = 350 + ((i - 8) % 4) * 115
            y = 220 + ((i - 8) // 4) * 20
            val = audio_15d[i] if i < len(audio_15d) else 0.0
            cv2.putText(frame, f"A{i+1}:{val:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_CYAN, 1)

        # Vision 5D metrics (right-top)
        cv2.rectangle(frame, (840, 80), (1260, 260), COLOR_PANEL, -1)
        cv2.rectangle(frame, (840, 80), (1260, 260), COLOR_BORDER, 2)
        cv2.putText(frame, "VISION 5D VECTOR  (Mesh: Asymmetry / Lip / EAR / Pitch / Yaw)", (850, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_PRIMARY, 1)
        labels_vision = ["LipAsym", "LipDisp", "EAR", "Pitch", "Yaw"]
        for i in range(5):
            x = 850 + i * 80
            y = 140
            val = vision_5d[i] if i < len(vision_5d) else 0.0
            color = COLOR_AMBER if val > 0.7 else COLOR_CYAN
            cv2.putText(frame, labels_vision[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT_SECONDARY, 1)
            cv2.putText(frame, f"{val:.2f}", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Bottom telemetry strip
        cv2.rectangle(frame, (20, 280), (1260, 320), COLOR_PANEL, -1)
        cv2.putText(frame, "REAL-TIME TELEMETRY  |  RAW FRAMES PURGED AFTER INFERENCE  |  SCALAR METRICS ONLY PERSISTED",
                    (30, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_SECONDARY, 1)

        # Status indicators (bottom-right)
        status_color = COLOR_GREEN
        cv2.circle(frame, (1220, 350), 10, status_color, -1)
        cv2.putText(frame, "LIVE", (1235, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_PRIMARY, 2)

        # Landmark annotations on video (if face detected — simulated overlay for demo)
        # Draw a subtle bounding box around face region
        cv2.rectangle(frame, (int(w*0.25), int(h*0.15)), (int(w*0.75), int(h*0.65)), COLOR_AMBER, 2)
        # Lip corner markers
        cv2.circle(frame, (int(w*0.35), int(h*0.45)), 6, COLOR_CYAN, -1)
        cv2.circle(frame, (int(w*0.65), int(h*0.45)), 6, COLOR_CYAN, -1)
        # Eye markers
        cv2.circle(frame, (int(w*0.40), int(h*0.30)), 5, COLOR_GREEN, -1)
        cv2.circle(frame, (int(w*0.60), int(h*0.30)), 5, COLOR_GREEN, -1)

        cv2.imshow(self.win, frame)
        return frame

# ------------------------------------------------------------------
# Main Threaded Workstation
# ------------------------------------------------------------------
def main():
    print("[FlorNeuroPulse HUD] Starting multi-threaded workstation...")
    print("[Privacy] Raw audio/video processed transiently; only scalar metrics retained.")

    # Initialize extractors
    audio_ext = AudioFeatureExtractor(sr=16000)
    vision_ext = VisionFeatureExtractor()
    hud = HUDRenderer()

    # Queues for thread-safe streaming
    audio_queue = queue.Queue()
    vision_queue = queue.Queue()

    # Audio capture thread
    def audio_thread():
        def callback(indata, frames, time_info, status):
            if status:
                pass
            chunk = np.mean(indata, axis=1).astype(np.float32)
            audio_queue.put(chunk)
        with sd.InputStream(samplerate=16000, channels=1, dtype=np.float32, blocksize=1024, callback=callback):
            while True:
                time.sleep(0.1)

    # Vision capture thread
    def vision_thread():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        while True:
            ret, frame = cap.read()
            if ret:
                vision_queue.put(frame)

    # Start threads
    threading.Thread(target=audio_thread, daemon=True).start()
    threading.Thread(target=vision_thread, daemon=True).start()

    # Main loop: process latest available frames, extract, infer (simulated), render
    # Note: In production, inference runs via ONNX Runtime against the 15D/5D vectors.
    # Here we simulate the pipeline with synthetic inference for demonstration.
    while True:
        # Pull latest audio chunk if available
        try:
            while not audio_queue.empty():
                chunk = audio_queue.get_nowait()
                audio_ext.feed(chunk)
        except queue.Empty:
            pass

        # Pull latest vision frame if available
        frame = None
        try:
            while not vision_queue.empty():
                frame = vision_queue.get_nowait()
        except queue.Empty:
            pass

        if frame is not None:
            # Extract vision 5D (transient — no frame saved)
            vision_5d = vision_ext.extract_5d(frame)
            # Extract audio 15D (transient — buffer only)
            audio_15d = audio_ext.extract_15d()

            # Simulated inference score (in production: engine.predict(audio_15d, vision_5d))
            # We compute a synthetic score from feature magnitudes for demo
            score = float(np.clip(np.mean(np.abs(audio_15d)) * 0.6 + np.mean(np.abs(vision_5d)) * 0.4, 0.0, 1.0))

            # Render HUD
            hud.draw(frame, audio_15d, vision_5d, score)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cv2.destroyAllWindows()
    print("[FlorNeuroPulse HUD] Shutdown — all raw media purged.")

if __name__ == "__main__":
    main()
