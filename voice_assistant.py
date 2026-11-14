"""
Voice Output Module
Converts detected objects into audio announcements using pyttsx3 and Web Speech.
Features:
- Threaded non-blocking speech queue (prevents video frame drops)
- TC07: Natural voice synthesis ("Bottle detected directly ahead")
- TC08: Cooldown & deduplication buffer (prevents spamming repeated announcements)
- Multi-engine: pyttsx3 native OS speech + Browser Web Speech API fallback
- Real-time announcement history log
"""
import time
import queue
import threading
from config import DEFAULT_SPEECH_RATE, DEFAULT_COOLDOWN_SECONDS, SPEECH_QUEUE_MAX_SIZE


class VoiceAssistant:
    """
    Manages speech synthesis with deduplication and priority queueing.
    """

    def __init__(self, speech_rate: int = DEFAULT_SPEECH_RATE, cooldown: float = DEFAULT_COOLDOWN_SECONDS):
        self.speech_rate = speech_rate
        self.cooldown_seconds = cooldown
        self.is_muted = False

        # TC08: Cooldown tracking cache
        # Map: (class_name, spatial_zone) -> last_spoken_timestamp
        self.last_announced = {}

        # History log for UI transcript
        self.announcement_history = []
        self.max_history = 50

        # Background pyttsx3 worker thread & queue
        self.speech_queue = queue.Queue(maxsize=SPEECH_QUEUE_MAX_SIZE)
        self.tts_engine = None
        self.tts_available = False
        self._init_tts_engine()

        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _init_tts_engine(self):
        """
        Initializes pyttsx3 engine safely.
        """
        try:
            import pyttsx3
            # Initialize COM in thread if Windows requires it
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty("rate", self.speech_rate)
            # Try to pick a natural voice if available
            voices = self.tts_engine.getProperty("voices")
            if voices:
                self.tts_engine.setProperty("voice", voices[0].id)
            self.tts_available = True
            print("[VoiceAssistant] pyttsx3 TTS engine initialized successfully.")
        except Exception as e:
            print(f"[VoiceAssistant] Native pyttsx3 not available or audio driver missing: {e}")
            self.tts_available = False

    def _speech_worker(self):
        """
        Daemon thread worker that executes pyttsx3 speech sequentially.
        Ensures camera frame rate and UI never freeze during audio playback.
        """
        while True:
            try:
                item = self.speech_queue.get()
                if item is None:
                    break
                text, priority = item

                if not self.is_muted and self.tts_available and self.tts_engine:
                    try:
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                    except Exception as err:
                        print(f"[VoiceAssistant] Error during speech synthesis: {err}")
                        # Re-initialize engine if COM state corrupted
                        try:
                            import pyttsx3
                            self.tts_engine = pyttsx3.init()
                        except Exception:
                            pass

                self.speech_queue.task_done()
            except Exception as e:
                print(f"[VoiceAssistant] Speech worker exception: {e}")
                time.sleep(0.1)

    def set_muted(self, muted: bool):
        """Mutes or unmutes audio announcements."""
        self.is_muted = muted

    def set_cooldown(self, seconds: float):
        """TC08: Sets cooldown period between repeating identical object announcements."""
        self.cooldown_seconds = max(0.5, seconds)

    def should_announce(self, class_name: str, spatial_pos: str, priority: int = 3) -> bool:
        """
        TC08: Checks whether an announcement for this object should be made or suppressed.
        - If spoken within cooldown_seconds, returns False.
        - Urgent priority 1 obstacles have a shortened cooldown (50% of normal) for safety.
        """
        key = (class_name.lower().strip(), spatial_pos.lower().strip())
        now = time.time()

        cooldown = self.cooldown_seconds
        if priority == 1:
            cooldown = max(1.5, self.cooldown_seconds * 0.5)

        if key in self.last_announced:
            elapsed = now - self.last_announced[key]
            if elapsed < cooldown:
                return False  # Suppressed by cooldown (TC08)

        self.last_announced[key] = now
        return True

    def format_phrase(self, detection: dict) -> str:
        """
        TC07: Formats a natural, accessible audio phrase for the visually impaired.
        Examples:
        - "Caution! Person very close directly ahead!"
        - "Chair on your left, close distance."
        - "Bottle detected directly ahead."
        """
        class_name = detection["class_name"]
        spatial_pos = detection.get("spatial_position", "ahead")
        distance_cat = detection.get("distance_category", "Moderate")
        priority = detection.get("priority", 3)

        if priority == 1 or distance_cat == "Immediate":
            phrase = f"Caution! {class_name.capitalize()} very close {spatial_pos}!"
        elif distance_cat == "Close":
            phrase = f"{class_name.capitalize()} {spatial_pos}, close distance."
        elif distance_cat == "Distant":
            phrase = f"{class_name.capitalize()} detected {spatial_pos}, far away."
        else:
            phrase = f"{class_name.capitalize()} {spatial_pos}."

        return phrase

    def announce(self, text: str, priority: int = 3, force: bool = False) -> bool:
        """
        Dispatches announcement to speech queue and records in history log.
        Returns True if queued, False if queue is full or muted.
        """
        timestamp = time.strftime("%H:%M:%S")

        # Record in UI transcript history
        self.announcement_history.append({
            "time": timestamp,
            "text": text,
            "priority": priority
        })
        if len(self.announcement_history) > self.max_history:
            self.announcement_history.pop(0)

        if self.is_muted and not force:
            return False

        try:
            # Drop older low-priority queue items if queue is full
            if self.speech_queue.full() and priority <= 2:
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    pass

            self.speech_queue.put_nowait((text, priority))
            return True
        except queue.Full:
            return False

    def process_detections(self, detections: list[dict]) -> list[str]:
        """
        Core pipeline:
        1. Evaluates all current detections.
        2. Applies TC08 cooldown filtering.
        3. Formats TC07 natural language phrases.
        4. Dispatches the top announcements.
        Returns list of newly announced phrases in this cycle.
        """
        spoken_phrases = []

        for det in detections:
            c_name = det["class_name"]
            s_pos = det.get("spatial_position", "ahead")
            prio = det.get("priority", 3)

            # TC08: Check cooldown deduplication
            if self.should_announce(c_name, s_pos, prio):
                phrase = self.format_phrase(det)
                queued = self.announce(phrase, priority=prio)
                if queued:
                    spoken_phrases.append(phrase)

            # Only announce top 2 most urgent items per frame to prevent cognitive overload
            if len(spoken_phrases) >= 2:
                break

        return spoken_phrases

    def get_recent_history(self, count: int = 10) -> list[dict]:
        """Returns the most recent spoken announcements."""
        return self.announcement_history[-count:]

    def clear_cooldown(self):
        """Resets cooldown cache (useful for new scenes / test resets)."""
        self.last_announced.clear()
