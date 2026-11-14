# 👁️ VisionVoice AI: Object Detection & Voice Assistant for Visually Impaired People

> **Hackathon Ready-Made Project**  
> An Artificial Intelligence and Computer Vision assistive system designed to empower visually impaired individuals by detecting surroundings through a camera, analyzing spatial proximity, and announcing obstacles in real-time through intelligent voice synthesis.

---

## 📌 1. Project Overview & Problem Statement

Visually impaired individuals face daily mobility challenges, navigating dynamic environments with physical obstacles, approaching vehicles, and misplaced everyday objects. Conventional walking canes only detect ground-level physical obstructions within arm's reach.

**VisionVoice AI** bridges this gap by acting as an intelligent visual companion:
- **Observes** the environment continuously using standard webcam or mobile camera.
- **Identifies** common objects (people, chairs, bottles, vehicles, doors, stairs) using a pre-trained YOLO AI model.
- **Calculates** spatial coordinates (Left, Directly Ahead, Right) and estimated proximity (Immediate collision hazard, Close, Moderate, Distant).
- **Announces** contextual auditory instructions through a natural speech engine without overwhelming the user (smart cooldown deduplication).
- **Navigates** safely by evaluating clear walking paths.

---

## 🛠️ 2. Technology Stack

| Technology | Role & Purpose | Key Capabilities |
|---|---|---|
| **Python 3.10+** | Core Programming Language | High performance, rich ML ecosystem |
| **OpenCV (`cv2`)** | Computer Vision & Video Ingestion | Live camera capture, CLAHE low-light enhancement, annotations |
| **YOLO (`ultralytics`)** | Deep Learning Object Detector | Real-time object localization on 80 COCO classes |
| **PyTorch (`torch`)** | Neural Network Runtime | Accelerates YOLO inference on CPU / GPU |
| **`pyttsx3`** | Native Text-to-Speech Engine | Offline, zero-latency OS speech synthesis via background thread |
| **Web Speech API** | Browser Audio Fallback | Cross-platform audible voice directly through client browser |
| **Streamlit** | Assistive User Interface | High-contrast accessible UI, controls, real-time matrix & test bench |

---

## 🏗️ 3. System Architecture & Modules

```mermaid
flowchart TD
    A["Camera Module<br/>(OpenCV VideoCapture / Test Scenes)"] --> B["Image Processing Module<br/>(Luminance Check & CLAHE Low-Light Boost)"]
    B --> C["Object Detection Module<br/>(YOLOv8 / COCO 80 Classes)"]
    C --> D["Spatial & Proximity Estimator<br/>(Centroid Zones & Bbox Ratio Depth)"]
    D --> E["Alert Module<br/>(Priority Ranking & Path Clearance)"]
    E --> F["Voice Output Module<br/>(Cooldown Deduplication & Threaded TTS)"]
    F --> G["User Interface Module<br/>(Streamlit Accessible Dashboard & Audio Feed)"]
```

### Module Breakdown:
1. **Camera Module (`camera_stream.py`)**:
   - Ingests frames from live webcams, IP cameras, or video files.
   - Built-in simulation generator with realistic test scenes for offline judging (TC01 - TC05).
   - Catches camera disconnections and missing devices gracefully without crashing (TC06).

2. **Image Processing Module (`image_processor.py`)**:
   - Measures perceptual luminance in lux (ITU-R BT.601).
   - Automatically detects low-light conditions (<50 lux).
   - Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the LAB L-channel and Gamma correction for night visibility (TC05).
   - Renders high-contrast bounding boxes with accessibility color codes (Red = Immediate danger, Orange = Close, Yellow = Moderate, Green = Safe).

3. **Object Detection Module (`detector.py`)**:
   - Uses YOLOv8 nano (`yolov8n.pt`) pre-trained on 80 COCO classes.
   - Filters out uncertain predictions using a customizable confidence threshold (TC10).
   - Calculates horizontal zones (`on your left`, `directly ahead`, `on your right`) and vertical zones (`upper level`, `eye level`, `ground level`).
   - Uses normalized bounding box height ratio ($h_{bbox}/h_{frame}$) to categorize proximity:
     - $\ge 55\%$: **Immediate** (< 1 meter collision risk)
     - $30\% - 55\%$: **Close** (1 - 2 meters)
     - $12\% - 30\%$: **Moderate** (2 - 4 meters)
     - $< 12\%$: **Distant** (> 4 meters, TC04)

4. **Alert Module (`alert_manager.py`)**:
   - Computes path clearance: identifies if center, left, or right paths are obstructed.
   - Dispatches emergency collision warnings ("HALT! Person directly in your path!").
   - Provides gentle directional navigation hints ("Obstacle ahead. Veer slightly right").

5. **Voice Output Module (`voice_assistant.py`)**:
   - Thread-safe background worker thread (`queue.Queue`) prevents speech synthesis from blocking the camera video stream or freezing the UI.
   - Formats natural language phrases (TC07).
   - **Cooldown & Deduplication Filter (TC08)**: suppresses repeat announcements of the same object within a 4-second window to prevent cognitive overload.
   - Native `pyttsx3` + Browser Web Speech API fallback.

6. **User Interface Module (`app.py`)**:
   - High-contrast accessibility theme with large buttons.
   - Start / Stop detection controls (TC09).
   - Real-time video preview with accessibility HUD.
   - Live spoken announcements transcript with timestamps.
   - Interactive Hackathon Test Bench running all 10 test cases in one click.

---

## 🧪 4. Hackathon Test Cases (TC01 - TC10) Matrix

All test cases specified in the project requirements are implemented and verified:

| Test Case | Scenario Description | Expected Output | Project Implementation | Status |
|:---:|---|---|---|:---:|
| **TC01** | Single object detection | Detect bottle | YOLO identifies single bottle with 90%+ confidence | **PASS** ✅ |
| **TC02** | Multiple objects | Detect person, chair, bottle | Multi-class simultaneous detection and spatial sorting | **PASS** ✅ |
| **TC03** | Empty scene | No false announcements | Zero false alarms; empty floor triggers 0 audio calls | **PASS** ✅ |
| **TC04** | Distant objects | Test detection performance | Detects small objects with height ratio < 0.12 tagged "Distant" | **PASS** ✅ |
| **TC05** | Low lighting | Handle reduced visibility | Auto-detects <50 lux, applies CLAHE boost to illuminate dark frames | **PASS** ✅ |
| **TC06** | Camera disconnected | Display camera error | Traps missing hardware cleanly; shows user warning banner without crash | **PASS** ✅ |
| **TC07** | Voice announcement | Speak object name | Generates phrase: *"Bottle detected directly ahead, close distance"* | **PASS** ✅ |
| **TC08** | Repeated objects | Avoid repeated announcements | Debounce cache suppresses identical announcements within cooldown | **PASS** ✅ |
| **TC09** | Stop button | Stop detection | Cleanly terminates capture loop and releases hardware memory | **PASS** ✅ |
| **TC10** | Uncertain prediction | Avoid unreliable announcements | Filters out predictions below confidence threshold (e.g. 0.50) | **PASS** ✅ |

---

## 🚀 5. Quick Start & Setup Instructions

### Option A: 1-Click Launch (Windows)
Double-click `run.bat` or run in PowerShell:
```powershell
.\run.ps1
```

### Option B: Manual Setup
1. **Clone or Navigate to the Directory**:
   ```bash
   cd C:\Users\hz0r1\.gemini\antigravity\scratch\ai-vision-voice-assistant
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Test Assets**:
   ```bash
   python create_test_assets.py
   ```

4. **Launch Streamlit Web App**:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

5. **Run Automated Test Suite (Command Line)**:
   ```bash
   python test_suite.py
   ```

---

## 🎓 6. Hackathon Viva & Examiner Questions Cheat Sheet

### Q1: What makes this solution specifically designed for visually impaired users?
- **Auditory spatial awareness**: Instead of just saying "Chair", it announces "Chair on your left, close distance".
- **Collision alert prioritization**: Objects directly in the walking trajectory are prioritized over background objects.
- **Anti-spam cooldown (TC08)**: Continuous video feeds detect objects 30 times a second. Without deduplication, the user would hear a chaotic loop of words. Our cooldown ensures each unique object is announced only once per configurable interval.
- **High-contrast accessible dashboard**: For low-vision individuals or assistive guides, the UI uses large bold contrast badges and high-visibility color boundaries.

### Q2: Why is YOLO preferred over traditional Haar Cascades or MobileNet-SSD?
- Haar cascades only detect specific rigid objects (e.g. frontal faces) and fail on rotated or varied perspectives.
- MobileNet-SSD has lower mAP (mean Average Precision) on complex multi-object overlapping scenes.
- YOLOv8 provides state-of-the-art mAP with high frames-per-second (30+ FPS on edge devices), identifying 80 standard COCO classes with robust occlusion tolerance.

### Q3: How do you achieve non-blocking voice output?
- Standard `pyttsx3.speak()` blocks execution until audio finishes speaking, which drops video frames and creates high latency.
- We implement a multi-threaded architecture with a Python `queue.Queue` and a dedicated background daemon worker thread. Incoming detections are enqueued instantaneously, allowing the video loop to run smoothly at full frame rate.
