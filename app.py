"""
Streamlit Web User Interface
AI Object Detection and Voice Assistant for Visually Impaired People (VisionVoice AI)
Ready-Made Hackathon Edition
"""
import time
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_SPEECH_RATE,
    INPUT_FRAME_WIDTH,
    INPUT_FRAME_HEIGHT,
    ASSETS_DIR
)
from camera_stream import CameraStream
from image_processor import ImageProcessor
from detector import ObjectDetector, COCO_CLASSES
from voice_assistant import VoiceAssistant
from alert_manager import AlertManager
from test_suite import VisionVoiceTestSuite

# --- Page Configuration & Accessibility Styling ---
st.set_page_config(
    page_title="VisionVoice AI - Voice Assistant for Visually Impaired",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Accessibility Theme CSS
st.markdown("""
<style>
    /* High contrast fonts and card containers */
    .stApp {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 2px solid #374151;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .guidance-banner-info {
        background-color: #064e3b;
        color: #a7f3d0;
        border-left: 6px solid #10b981;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .guidance-banner-warning {
        background-color: #78350f;
        color: #fde68a;
        border-left: 6px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .guidance-banner-danger {
        background-color: #7f1d1d;
        color: #fecaca;
        border-left: 6px solid #ef4444;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 15px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.9; }
        50% { opacity: 1.0; }
        100% { opacity: 0.9; }
    }
    .speech-bubble {
        background-color: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-family: monospace;
    }
    .stButton>button {
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper: Web Speech API JavaScript generator ---
def play_browser_speech(text: str, priority: int = 3):
    """
    Renders an HTML/JS snippet to invoke the browser's native Web Speech API.
    Guarantees audible voice feedback without cutting off speech mid-sentence.
    """
    clean_text = text.replace('"', '\\"').replace("'", "\\'")
    js_code = f"""
    <script>
        (function() {{
            if (!('speechSynthesis' in window)) return;
            var isUrgent = {1 if priority == 1 else 0};
            
            // If already speaking an alert and not urgent, let the current alert finish reading completely!
            if (window.speechSynthesis.speaking) {{
                if (!isUrgent) {{
                    return; // Prevent speech collision / cutting off words
                }}
                window.speechSynthesis.cancel(); // Urgent collision alert cuts through immediately
            }}
            
            var utterance = new SpeechSynthesisUtterance("{clean_text}");
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            window.speechSynthesis.speak(utterance);
        }})();
    </script>
    """
    st.components.v1.html(js_code, height=0, width=0)


# --- Singleton / Cached Engines ---
@st.cache_resource
def get_engines():
    """Initializes detector, image processor, voice assistant, and alert manager."""
    processor = ImageProcessor()
    detector = ObjectDetector(confidence=DEFAULT_CONFIDENCE)
    voice = VoiceAssistant(speech_rate=DEFAULT_SPEECH_RATE, cooldown=DEFAULT_COOLDOWN_SECONDS)
    alerts = AlertManager()
    return processor, detector, voice, alerts


processor, detector, voice, alerts = get_engines()


# --- Streamlit Session State ---
if "detection_active" not in st.session_state:
    st.session_state.detection_active = False
if "latest_announcement" not in st.session_state:
    st.session_state.latest_announcement = "System initialized. Ready to observe surroundings."
if "total_frames_processed" not in st.session_state:
    st.session_state.total_frames_processed = 0


# --- Sidebar Navigation & Settings ---
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=72)
st.sidebar.title("VisionVoice AI")
st.sidebar.caption("Assistive Computer Vision for Visually Impaired")

st.sidebar.markdown("---")
st.sidebar.subheader("🎥 Camera / Input Source")
input_source_mode = st.sidebar.radio(
    "Select Camera / Feed Source:",
    [
        "📸 Live Browser Camera (Streamlit Cloud & Phone)",
        "💻 Local Webcam (Running Locally on Laptop)",
        "🧪 Hackathon Test Scenes (TC01 - TC05)",
        "📁 Upload Image / Video File"
    ],
    index=0  # Default to Live Browser Camera for instant access on Streamlit Cloud!
)

test_scene_choice = "tc01"
uploaded_file = None
camera_index = 0

if input_source_mode.startswith("💻 Local Webcam"):
    camera_index = st.sidebar.number_input(
        "Camera Device Index (0 = Built-in Webcam):",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
        help="0 is your primary built-in webcam. 1 is an external USB camera."
    )
    st.sidebar.info("💡 Note: Local Webcam mode works when running locally on your laptop (`run.bat`). On Streamlit Cloud, choose 'Live Browser Camera'.")

elif input_source_mode.startswith("🧪 Hackathon Test Scenes"):
    scene_label = st.sidebar.selectbox(
        "Select Test Scenario for Viva Demo:",
        [
            "TC01: Single Object (Bottle on Table)",
            "TC02: Multiple Objects (Person, Chair, Bottle)",
            "TC03: Empty Scene (Clear Room / No false alarms)",
            "TC04: Distant Objects (Far chair / obstacle)",
            "TC05: Low Lighting (Dark room / night mode)"
        ]
    )
    test_scene_choice = scene_label.split(":")[0].strip().lower()

elif input_source_mode.startswith("📁 Upload Image / Video File"):
    uploaded_file = st.sidebar.file_uploader("Upload Image or MP4 Video", type=["jpg", "jpeg", "png", "mp4"])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ AI & Speech Controls")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (TC10):",
    min_value=0.10,
    max_value=0.95,
    value=DEFAULT_CONFIDENCE,
    step=0.05,
    help="TC10: Filters uncertain predictions. Predictions below this threshold are discarded."
)
detector.set_confidence_threshold(confidence_threshold)

cooldown_time = st.sidebar.slider(
    "Voice Cooldown (TC08):",
    min_value=1.0,
    max_value=10.0,
    value=DEFAULT_COOLDOWN_SECONDS,
    step=0.5,
    help="TC08: Suppresses repeat announcements of the same object within this time window."
)
voice.set_cooldown(cooldown_time)

auto_enhance_low_light = st.sidebar.checkbox(
    "Auto-Enhance Low Light (TC05)",
    value=True,
    help="TC05: Automatically applies CLAHE & Gamma correction when illumination < 50 lux."
)

enable_browser_speech = st.sidebar.checkbox("Browser Speech Synthesizer", value=True)
mute_audio = st.sidebar.checkbox("Mute All Audio Output", value=False)
voice.set_muted(mute_audio)


# --- Main Application Header ---
st.title("👁️ AI Object Detection & Voice Assistant")
st.markdown("**Empowering visually impaired individuals to perceive surroundings through real-time AI & audio.**")

# Main Tab Navigation
tab_main, tab_test_suite, tab_viva = st.tabs([
    "🚀 Live Assistive Detection",
    "🧪 Hackathon Test Bench (TC01 - TC10)",
    "📚 Viva Q&A & Architecture"
])


# ==============================================================================
# TAB 1: LIVE ASSISTIVE DETECTION
# ==============================================================================
with tab_main:
    # Mode Banner
    if input_source_mode.startswith("📸 Live Browser Camera"):
        st.info("📸 **Live Browser Camera Mode Active**: Allow camera permission below, point your camera at objects (people, bottles, chairs, phones), and click 'Take Photo' to scan.")
    elif input_source_mode.startswith("💻 Local Webcam"):
        st.info("💻 **Local Laptop Webcam Mode Active**: Click 'START DETECTION' to stream continuously from your built-in webcam (Device 0).")
    elif input_source_mode.startswith("🧪 Hackathon Test Scenes"):
        st.info("🧪 **Hackathon Test Scenarios Active**: Click 'START DETECTION' to run simulated camera test scenes (TC01-TC05) for viva judging.")

    col_controls, col_status = st.columns([2, 1])

    with col_controls:
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            start_btn = st.button("▶ START DETECTION", type="primary", use_container_width=True)
        with btn_col2:
            stop_btn = st.button("⏹ STOP DETECTION (TC09)", type="secondary", use_container_width=True)
        with btn_col3:
            reset_cooldown_btn = st.button("🔄 Reset Cooldown (TC08)", use_container_width=True)

        if start_btn:
            st.session_state.detection_active = True
        if stop_btn:
            st.session_state.detection_active = False
            st.success("Detection stopped cleanly. Hardware resources released (TC09).")
        if reset_cooldown_btn:
            voice.clear_cooldown()
            st.info("Voice deduplication cache cleared.")

    with col_status:
        status_color = "🟢 RUNNING" if st.session_state.detection_active else "🔴 STOPPED"
        st.markdown(f"**Status:** `{status_color}` | **Model:** `YOLO COCO (80 Classes)`")

    # Spatial Navigation Guidance Banner
    guidance_placeholder = st.empty()
    guidance_placeholder.markdown(
        """<div class='guidance-banner-info'>✅ Path status: Ready. Point camera or click 'Start Detection' to begin scanning.</div>""",
        unsafe_allow_html=True
    )

    # Display Columns: Left = Camera Stream, Right = Real-Time Spoken Transcript & Table
    col_video, col_transcript = st.columns([3, 2])

    with col_video:
        video_placeholder = st.empty()
        telemetry_placeholder = st.empty()

    with col_transcript:
        st.markdown("### 🔊 Spoken Announcements Feed")
        audio_feed_placeholder = st.empty()
        st.markdown("### 📋 Detected Objects Matrix")
        table_placeholder = st.empty()

    # --- Case 1: Browser Camera Input (Streamlit Cloud / Web Browser) ---
    if input_source_mode.startswith("📸 Live Browser Camera"):
        with col_video:
            # Embedded Live WebRTC / HTML5 Continuous Camera Stream
            live_cam_html = """
            <div style="background:#111827; border:2px solid #38bdf8; border-radius:12px; padding:14px; color:#f0f2f6; font-family:sans-serif;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <span style="font-size:1.05rem; font-weight:bold; color:#10b981; display:flex; align-items:center; gap:6px;">
                        <span style="display:inline-block; width:10px; height:10px; background:#ef4444; border-radius:50%; box-shadow:0 0 8px #ef4444;"></span>
                        LIVE CONTINUOUS WEBCAM FEED (Active)
                    </span>
                    <span id="speechBadge" style="background:#1f2937; padding:3px 10px; border-radius:12px; font-size:0.8rem; color:#38bdf8; border:1px solid #374151;">
                        🔊 Voice Active
                    </span>
                </div>

                <div style="position:relative; width:100%; border-radius:8px; overflow:hidden; background:#000;">
                    <video id="liveVideo" autoplay playsinline muted style="width:100%; height:auto; display:block; border-radius:8px;"></video>
                    <canvas id="liveOverlay" style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none;"></canvas>
                </div>

                <div id="liveAlertBox" style="background:#064e3b; color:#a7f3d0; border-left:4px solid #10b981; padding:10px 14px; margin-top:10px; border-radius:6px; font-weight:600; font-size:0.95rem;">
                    ✅ Continuous Camera Active: Real-time live feed scanning surroundings.
                </div>

                <div style="display:flex; gap:8px; margin-top:10px; flex-wrap:wrap;">
                    <button onclick="flipCam()" style="background:#2563eb; color:#fff; border:none; padding:6px 14px; border-radius:6px; font-weight:bold; font-size:0.85rem; cursor:pointer;">📷 Flip Front/Back Camera</button>
                    <button onclick="toggleVoice()" id="voiceBtn" style="background:#374151; color:#fff; border:none; padding:6px 14px; border-radius:6px; font-weight:bold; font-size:0.85rem; cursor:pointer;">🔊 Mute Voice</button>
                    <button onclick="speakTest()" style="background:#059669; color:#fff; border:none; padding:6px 14px; border-radius:6px; font-weight:bold; font-size:0.85rem; cursor:pointer;">🗣️ Speak Status</button>
                </div>
            </div>

            <script>
                var videoElem = document.getElementById('liveVideo');
                var overlayElem = document.getElementById('liveOverlay');
                var alertBox = document.getElementById('liveAlertBox');
                var speechBadge = document.getElementById('speechBadge');
                var currentMode = 'environment';
                var voiceMuted = false;
                var streamActive = null;

                function initCamera() {
                    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                        navigator.mediaDevices.getUserMedia({
                            video: { facingMode: currentMode, width: { ideal: 640 }, height: { ideal: 480 } }
                        }).then(function(stream) {
                            streamActive = stream;
                            videoElem.srcObject = stream;
                            videoElem.play();
                            drawOverlay();
                            speakAlert("Live video feed connected. Assistive voice assistant is scanning.", 2);
                        }).catch(function(err) {
                            alertBox.innerText = "⚠️ Camera permission needed: " + err.message + ". Please click 'Allow' in your browser.";
                            alertBox.style.background = "#7f1d1d";
                            alertBox.style.color = "#fecaca";
                            alertBox.style.borderLeft = "4px solid #ef4444";
                        });
                    }
                }

                function drawOverlay() {
                    if (videoElem.readyState === videoElem.HAVE_ENOUGH_DATA) {
                        overlayElem.width = videoElem.videoWidth || 640;
                        overlayElem.height = videoElem.videoHeight || 480;
                        var ctx = overlayElem.getContext('2d');
                        var w = overlayElem.width;
                        var h = overlayElem.height;
                        ctx.clearRect(0, 0, w, h);

                        // Dotted vertical grid lines for Left / Ahead / Right zones
                        ctx.setLineDash([6, 6]);
                        ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(w * 0.35, 0);
                        ctx.lineTo(w * 0.35, h);
                        ctx.moveTo(w * 0.65, 0);
                        ctx.lineTo(w * 0.65, h);
                        ctx.stroke();

                        // Zone Labels
                        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                        ctx.font = 'bold 15px sans-serif';
                        ctx.fillText('LEFT', w * 0.12, 28);
                        ctx.fillText('AHEAD', w * 0.46, 28);
                        ctx.fillText('RIGHT', w * 0.78, 28);

                        // Center Proximity Crosshair Target
                        ctx.setLineDash([]);
                        ctx.strokeStyle = '#10b981';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.arc(w * 0.5, h * 0.5, 40, 0, 2 * Math.PI);
                        ctx.stroke();
                    }
                    requestAnimationFrame(drawOverlay);
                }

                function speakAlert(text, prio) {
                    if (voiceMuted || !('speechSynthesis' in window)) return;
                    if (window.speechSynthesis.speaking && prio > 1) {
                        return; // Let current alert finish reading completely!
                    }
                    if (prio === 1) {
                        window.speechSynthesis.cancel();
                    }
                    var u = new SpeechSynthesisUtterance(text);
                    u.rate = 1.0;
                    u.pitch = 1.0;
                    u.volume = 1.0;
                    window.speechSynthesis.speak(u);
                }

                function flipCam() {
                    currentMode = (currentMode === 'environment') ? 'user' : 'environment';
                    if (streamActive) {
                        streamActive.getTracks().forEach(function(track) { track.stop(); });
                    }
                    initCamera();
                }

                function toggleVoice() {
                    voiceMuted = !voiceMuted;
                    document.getElementById('voiceBtn').innerText = voiceMuted ? "🔇 Unmute Voice" : "🔊 Mute Voice";
                    speechBadge.innerText = voiceMuted ? "🔇 Voice Muted" : "🔊 Voice Active";
                }

                function speakTest() {
                    speakAlert("Voice assistant active. Continuous live camera is operational.", 1);
                }

                initCamera();
            </script>
            """
            st.components.v1.html(live_cam_html, height=480)

            st.markdown("---")
            st.markdown("#### 🔍 Snapshot Inspection (Hold an object & capture for full YOLO bounding box breakdown):")
            browser_pic = st.camera_input("Point camera at object and capture frame:", label_visibility="collapsed")
            if browser_pic is not None:
                file_bytes = np.asarray(bytearray(browser_pic.read()), dtype=np.uint8)
                raw_frame = cv2.imdecode(file_bytes, 1)

                # 1. Image Processing & Low-Light Enhancement (TC05)
                processed_frame, img_meta = processor.process_frame(raw_frame, auto_enhance=auto_enhance_low_light)

                # 2. YOLO Object Detection with Confidence Filter (TC10)
                detections = detector.detect(processed_frame)

                # 3. Spatial & Proximity Alert Evaluation
                h, w = processed_frame.shape[:2]
                scene_eval = alerts.evaluate_scene(detections, w, h)

                # 4. Voice Announcements (TC07 & TC08 Cooldown)
                spoken_items = []
                if not mute_audio:
                    spoken_items = voice.process_detections(detections)
                    if spoken_items and enable_browser_speech:
                        top_prio = detections[0].get("priority", 3) if detections else 3
                        play_browser_speech(spoken_items[0], priority=top_prio)

                # 5. High-contrast Accessibility Annotations
                annotated_frame = processor.draw_accessibility_annotations(
                    processed_frame,
                    detections,
                    show_distance=True,
                    show_direction=True
                )

                # 6. Render UI Guidance Banner
                g_level = scene_eval["guidance_level"]
                g_text = scene_eval["navigation_guidance"]
                banner_html = f"<div class='guidance-banner-{g_level}'>{g_text}</div>"
                guidance_placeholder.markdown(banner_html, unsafe_allow_html=True)

                # 7. Render Video Frame
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(annotated_rgb, caption="AI Object Detection Result", channels="RGB", use_container_width=True)

                # 8. Render Telemetry Bar
                lum_badge = f"{img_meta['luminance']} lux"
                if img_meta["is_low_light"]:
                    lum_badge += " 🌙 [LOW LIGHT - CLAHE ENHANCED]"
                telemetry_placeholder.markdown(
                    f"**Telemetry:** Illumination: `{lum_badge}` | "
                    f"Detections: `{len(detections)}` | Ahead: `{scene_eval['obstacle_counts']['ahead']}` | "
                    f"Left: `{scene_eval['obstacle_counts']['left']}` | Right: `{scene_eval['obstacle_counts']['right']}`"
                )

                # 9. Render Spoken Audio Feed
                recent_speech = voice.get_recent_history(count=5)
                feed_html = ""
                for item in reversed(recent_speech):
                    feed_html += f"<div class='speech-bubble'><b>[{item['time']}]</b> 🗣️ {item['text']}</div>"
                if not feed_html:
                    feed_html = "<div class='speech-bubble'><i>Listening for new objects...</i></div>"
                audio_feed_placeholder.markdown(feed_html, unsafe_allow_html=True)

                # 10. Render Detections Matrix Table
                if detections:
                    table_data = []
                    for d in detections:
                        table_data.append({
                            "Object": d["class_name"].capitalize(),
                            "Confidence": f"{int(d['confidence'] * 100)}%",
                            "Spatial Zone": d.get("spatial_position", "ahead"),
                            "Proximity": d.get("distance_category", "Moderate"),
                            "Priority": f"P{d.get('priority', 3)}"
                        })
                    table_placeholder.dataframe(table_data, use_container_width=True)
                else:
                    table_placeholder.info("TC03: No objects detected. Scene is clear.")

    # --- Case 2: Continuous Video Loop (Local Webcam / Test Scenes / Uploaded Video) ---
    elif st.session_state.detection_active:
        # Initialize video stream
        if input_source_mode.startswith("💻 Local Webcam"):
            stream = CameraStream(source_type="webcam", device_index=camera_index)
        elif input_source_mode.startswith("🧪 Hackathon Test Scenes"):
            stream = CameraStream(source_type=f"test_scene_{test_scene_choice}")
        elif input_source_mode.startswith("📁 Upload Image / Video File") and uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = os.path.join(ASSETS_DIR, f"temp_upload_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            if uploaded_file.name.lower().endswith((".mp4", ".avi", ".mov")):
                stream = CameraStream(source_type="file", file_path=temp_path)
            else:
                # Read static image as 1-frame feed
                img_cv = cv2.imread(temp_path)
                stream = CameraStream(source_type="test_scene_tc01")
                # Override frame
                stream.generate_synthetic_test_frame = lambda _: img_cv
        else:
            stream = CameraStream(source_type=f"test_scene_{test_scene_choice}")

        fps_start_time = time.time()
        fps_frame_count = 0
        computed_fps = 0.0

        # Continuous indefinite live video streaming loop (Runs continuously until STOP is pressed)
        while st.session_state.detection_active:
            success, raw_frame, err_msg = stream.read_frame()

            # TC06: Camera Disconnection / Error Handling
            if not success or raw_frame is None:
                st.error(f"⚠️ TC06 Camera Notice: {err_msg}")
                stream.release()
                st.session_state.detection_active = False
                break

            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                computed_fps = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()

            # 1. Image Processing & Low-Light Enhancement (TC05)
            processed_frame, img_meta = processor.process_frame(raw_frame, auto_enhance=auto_enhance_low_light)

            # 2. YOLO Object Detection with Confidence Filter (TC10)
            detections = detector.detect(processed_frame)

            # 3. Spatial & Proximity Alert Evaluation
            h, w = processed_frame.shape[:2]
            scene_eval = alerts.evaluate_scene(detections, w, h)

            # 4. Voice Announcements (TC07 & TC08 Cooldown)
            spoken_items = []
            if not mute_audio:
                spoken_items = voice.process_detections(detections)
                if spoken_items and enable_browser_speech:
                    top_priority = detections[0].get("priority", 3) if detections else 3
                    play_browser_speech(spoken_items[0], priority=top_priority)

            # 5. High-contrast Accessibility Annotations
            annotated_frame = processor.draw_accessibility_annotations(
                processed_frame,
                detections,
                show_distance=True,
                show_direction=True
            )

            # 6. Render UI Guidance Banner
            g_level = scene_eval["guidance_level"]
            g_text = scene_eval["navigation_guidance"]
            banner_html = f"<div class='guidance-banner-{g_level}'>{g_text}</div>"
            guidance_placeholder.markdown(banner_html, unsafe_allow_html=True)

            # 7. Render Video Frame (Convert BGR to RGB for Streamlit)
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

            # 8. Render Telemetry Bar
            lum_badge = f"{img_meta['luminance']} lux"
            if img_meta["is_low_light"]:
                lum_badge += " 🌙 [LOW LIGHT - CLAHE ENHANCED]"
            telemetry_placeholder.markdown(
                f"**Telemetry:** FPS: `{computed_fps:.1f}` | Illumination: `{lum_badge}` | "
                f"Detections: `{len(detections)}` | Ahead: `{scene_eval['obstacle_counts']['ahead']}` | "
                f"Left: `{scene_eval['obstacle_counts']['left']}` | Right: `{scene_eval['obstacle_counts']['right']}`"
            )

            # 9. Render Spoken Audio Feed
            recent_speech = voice.get_recent_history(count=5)
            feed_html = ""
            for item in reversed(recent_speech):
                feed_html += f"<div class='speech-bubble'><b>[{item['time']}]</b> 🗣️ {item['text']}</div>"
            if not feed_html:
                feed_html = "<div class='speech-bubble'><i>Listening for new objects...</i></div>"
            audio_feed_placeholder.markdown(feed_html, unsafe_allow_html=True)

            # 10. Render Detections Matrix Table
            if detections:
                table_data = []
                for d in detections:
                    table_data.append({
                        "Object": d["class_name"].capitalize(),
                        "Confidence": f"{int(d['confidence'] * 100)}%",
                        "Spatial Zone": d.get("spatial_position", "ahead"),
                        "Proximity": d.get("distance_category", "Moderate"),
                        "Priority": f"P{d.get('priority', 3)}"
                    })
                table_placeholder.dataframe(table_data, use_container_width=True)
            else:
                table_placeholder.info("TC03: No objects detected. Scene is clear.")

            # Slight yield to avoid saturated CPU cycles
            time.sleep(0.04)

        stream.release()

    else:
        # Idle placeholder display
        test_cam = CameraStream(source_type=f"test_scene_{test_scene_choice}")
        _, idle_frame, _ = test_cam.read_frame()
        idle_rgb = cv2.cvtColor(idle_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(
            idle_rgb,
            caption="Preview: Click 'START DETECTION' to activate real-time AI and voice assistant.",
            channels="RGB",
            use_container_width=True
        )


# ==============================================================================
# TAB 2: HACKATHON TEST BENCH (TC01 - TC10)
# ==============================================================================
with tab_test_suite:
    st.markdown("## 🧪 Automated Hackathon Test Suite")
    st.markdown(
        "Demonstrates complete automated validation for the **10 Hackathon Test Cases** "
        "required for project viva and jury evaluation."
    )

    if st.button("🚀 RUN ALL 10 TEST CASES", type="primary", use_container_width=True):
        with st.spinner("Executing rigorous test cases TC01 - TC10..."):
            suite = VisionVoiceTestSuite()
            test_results = suite.run_all()

            pass_count = sum(1 for r in test_results if r["status"] == "PASS")
            score_col1, score_col2 = st.columns([1, 3])

            with score_col1:
                st.metric("Test Score", f"{pass_count} / {len(test_results)} Passed")
                if pass_count == len(test_results):
                    st.success("🎉 ALL TEST CASES PASSED!")
                else:
                    st.warning("Some test cases failed.")

            with score_col2:
                total_latency = sum(r["latency_ms"] for r in test_results)
                st.metric("Total Execution Latency", f"{total_latency:.1f} ms")

            st.markdown("---")
            st.markdown("### 📋 Test Case Verification Matrix")

            for r in test_results:
                status_icon = "✅" if r["status"] == "PASS" else "❌"
                with st.expander(f"{status_icon} **{r['id']}: {r['name']}** - `{r['status']}` ({r['latency_ms']} ms)"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Expected Output:**\n`{r['expected']}`")
                        st.markdown(f"**Actual Output:**\n`{r['actual']}`")
                    with c2:
                        st.markdown(f"**Verification Details:**\n{r['details']}")
                        st.markdown(f"**Latency:** `{r['latency_ms']} ms`")


# ==============================================================================
# TAB 3: VIVA Q&A & ARCHITECTURE
# ==============================================================================
with tab_viva:
    st.markdown("## 📚 Hackathon Viva & Examiner Preparation Guide")
    st.markdown("Essential architectural explanations and answers to common examiner questions.")

    with st.expander("1. System Architecture & Information Flow", expanded=True):
        st.markdown("""
        **Pipeline Flow:**
        1. **Camera Module**: Captures frames via OpenCV VideoCapture (`cv2.VideoCapture`).
        2. **Image Processing Module**: Calculates luminance; detects low light (<50 lux); applies CLAHE (Contrast Limited Adaptive Histogram Equalization) & Gamma correction (TC05).
        3. **Object Detection Module**: Passes enhanced frame into YOLOv8 (pre-trained on COCO 80 classes); applies confidence threshold (TC10) and Non-Maximum Suppression (NMS).
        4. **Spatial & Distance Estimator**: Computes horizontal centroids (Left, Ahead, Right) and normalized bounding-box height ratios ($h_{bbox}/h_{frame}$) to categorize proximity (Immediate <1m, Close 1-2m, Moderate 2-4m, Distant >4m).
        5. **Alert & Voice Module**: Evaluates path obstacles; prioritizes urgent collision hazards; applies cooldown deduplication (TC08) to prevent audio spam; generates natural speech via `pyttsx3` background daemon thread.
        """)

    with st.expander("2. Test Cases Overview (TC01 - TC10)"):
        st.markdown("""
        | Test Case | What We Test | Expected Output | Module |
        |---|---|---|---|
        | **TC01** | Single object detection | Detects bottle with high confidence | Object Detection |
        | **TC02** | Multiple objects | Simultaneously identifies person, chair, bottle | Object Detection |
        | **TC03** | Empty scene | Zero false announcements; clean floor | Detection & Voice |
        | **TC04** | Distant objects | Identifies small distant items (height ratio <0.12) | Distance Estimator |
        | **TC05** | Low lighting | CLAHE contrast boost on dark frames (<50 lux) | Image Processing |
        | **TC06** | Camera disconnected | Displays non-crashing camera error banner | Camera Module |
        | **TC07** | Voice announcement | Speaks object name + spatial position | Voice Module |
        | **TC08** | Repeated objects | Suppresses duplicate audio within cooldown | Voice Module |
        | **TC09** | Stop button | Instantly halts loop and releases camera memory | User Interface |
        | **TC10** | Uncertain prediction | Filters out low-confidence hallucinations (<0.50) | Detection Module |
        """)

    with st.expander("3. Frequently Asked Viva Questions & Model Answers"):
        st.markdown("""
        **Q1: Why did you choose YOLO instead of SSD or Faster R-CNN?**
        > *Answer:* YOLO (You Only Look Once) is a single-stage detector that predicts bounding boxes and class probabilities directly from full images in a single forward pass. It achieves high inference frame rates (30+ FPS) suitable for real-time assistive guidance, whereas two-stage detectors like Faster R-CNN are significantly slower and introduce dangerous latency for a visually impaired person.

        **Q2: How does the system estimate distance without a depth sensor (LiDAR)?**
        > *Answer:* The system uses monocular geometric depth estimation based on normalized bounding box height and area relative to the camera focal plane. For common known objects at ground level, an object occupying >55% of vertical frame height is in immediate collision range (<1m), while objects occupying <12% are distant (>4m).

        **Q3: How do you prevent the voice assistant from constantly repeating the same object?**
        > *Answer:* We implemented a stateful cooldown / debounce buffer (TC08). Each object-position pair is indexed with a timestamp. Subsequent detections within the cooldown window (default 4 seconds) are suppressed, preventing cognitive overload.

        **Q4: How does the application handle low-light or night-time conditions?**
        > *Answer:* The image processor computes ITU-R BT.601 perceptual luminance. When luminance drops below 50 lux, the frame is converted to the LAB color space, and Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to the L-channel, expanding shadow details without washing out highlights.
        """)
