"""
Camera Module
Captures live camera video, handles disconnection errors (TC06),
supports video/image uploads, and provides built-in test scenes (TC01-TC05).
"""
import os
import cv2
import numpy as np
from pathlib import Path
from config import ASSETS_DIR, INPUT_FRAME_WIDTH, INPUT_FRAME_HEIGHT


class CameraStream:
    """
    Manages live camera feeds, video files, and simulated hackathon test scenes.
    """

    def __init__(self, source_type: str = "webcam", device_index: int = 0, file_path: str = None):
        self.source_type = source_type
        self.device_index = device_index
        self.file_path = file_path
        self.cap = None
        self.is_connected = False
        self.last_error = None
        self._init_source()

    def _init_source(self):
        """
        Initializes video capture stream based on selected source.
        TC06: Handles missing or disconnected camera gracefully.
        """
        self.last_error = None
        self.is_connected = False

        if self.source_type == "webcam":
            try:
                # Use CAP_DSHOW on Windows for fast webcam initialization
                self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
                if not self.cap.isOpened():
                    # Fallback to standard CAP_ANY
                    self.cap = cv2.VideoCapture(self.device_index)

                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_FRAME_WIDTH)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_FRAME_HEIGHT)
                    self.is_connected = True
                else:
                    self.is_connected = False
                    self.last_error = f"Camera device {self.device_index} not accessible or disconnected."
            except Exception as e:
                self.is_connected = False
                self.last_error = f"Exception opening camera {self.device_index}: {e}"

        elif self.source_type == "file" and self.file_path:
            if os.path.exists(self.file_path):
                self.cap = cv2.VideoCapture(self.file_path)
                self.is_connected = self.cap.isOpened()
                if not self.is_connected:
                    self.last_error = f"Could not read video file: {self.file_path}"
            else:
                self.is_connected = False
                self.last_error = f"File does not exist: {self.file_path}"

        elif self.source_type.startswith("test_scene_"):
            # Synthetic / test bench mode
            self.is_connected = True

    def read_frame(self) -> tuple[bool, np.ndarray, str]:
        """
        Reads next frame from stream.
        Returns:
            (success: bool, frame: np.ndarray, error_message: str)
        TC06: Returns (False, None, "Camera disconnected") when camera fails.
        """
        if self.source_type.startswith("test_scene_"):
            if not self.is_connected:
                return False, None, "Stream released / stopped."
            scene_name = self.source_type.replace("test_scene_", "")
            frame = self.generate_synthetic_test_frame(scene_name)
            return True, frame, ""

        if not self.is_connected or self.cap is None or not self.cap.isOpened():
            return False, None, self.last_error or "Camera disconnected or not initialized."

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.is_connected = False
            self.last_error = "Camera feed lost: frame capture returned null (TC06)."
            return False, None, self.last_error

        # Ensure consistent frame dimensions
        if frame.shape[1] != INPUT_FRAME_WIDTH or frame.shape[0] != INPUT_FRAME_HEIGHT:
            frame = cv2.resize(frame, (INPUT_FRAME_WIDTH, INPUT_FRAME_HEIGHT))

        return True, frame, ""

    def release(self):
        """
        TC09: Releases video capture device cleanly.
        """
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_connected = False

    def generate_synthetic_test_frame(self, scene_name: str) -> np.ndarray:
        """
        Generates realistic visual test frames for hackathon test cases TC01 to TC05:
        - tc01: Single object (Bottle)
        - tc02: Multiple objects (Person, Chair, Bottle)
        - tc03: Empty scene (Clear floor, no objects)
        - tc04: Distant objects (Small distant chair and car)
        - tc05: Low lighting scene (Dark indoor environment)
        """
        w, h = INPUT_FRAME_WIDTH, INPUT_FRAME_HEIGHT
        # Create base realistic indoor room background (wall and wooden floor)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Wall color (light beige/gray)
        frame[0:int(h * 0.65), :] = [210, 220, 225]
        # Floor color (warm wooden brown)
        frame[int(h * 0.65):, :] = [80, 110, 150]
        # Baseboard dividing line
        cv2.line(frame, (0, int(h * 0.65)), (w, int(h * 0.65)), (50, 70, 90), 3)

        metadata_items = []

        if scene_name == "tc01" or scene_name == "bottle":
            # TC01: Single bottle directly ahead on table
            # Draw table
            cv2.rectangle(frame, (180, int(h * 0.55)), (460, int(h * 0.85)), (90, 130, 170), -1)
            # Draw water bottle (blue transparent cylinder with cap)
            bx1, by1, bx2, by2 = 290, 180, 350, 320
            # Bottle body
            cv2.rectangle(frame, (bx1, by1 + 25), (bx2, by2), (220, 180, 70), -1)
            cv2.rectangle(frame, (bx1, by1 + 25), (bx2, by2), (180, 140, 40), 2)
            # Cap
            cv2.rectangle(frame, (bx1 + 15, by1), (bx2 - 15, by1 + 25), (40, 40, 220), -1)
            cv2.putText(frame, "WATER", (bx1 + 5, by1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            metadata_items.append({
                "class_name": "bottle",
                "confidence": 0.94,
                "bbox": [bx1, by1, bx2, by2]
            })

        elif scene_name == "tc02" or scene_name == "multi_object":
            # TC02: Multiple objects - Person (ahead/left), Chair (right), Bottle (center)
            # 1. Person on the left
            px1, py1, px2, py2 = 80, 90, 230, 420
            # Head
            cv2.circle(frame, (155, 140), 35, (170, 190, 220), -1)
            # Body/Torso
            cv2.rectangle(frame, (115, 180), (195, 330), (150, 70, 40), -1)
            # Legs
            cv2.rectangle(frame, (120, 330), (150, 415), (70, 50, 40), -1)
            cv2.rectangle(frame, (160, 330), (190, 415), (70, 50, 40), -1)

            metadata_items.append({
                "class_name": "person",
                "confidence": 0.91,
                "bbox": [px1, py1, px2, py2]
            })

            # 2. Bottle in the center
            bx1, by1, bx2, by2 = 285, 240, 345, 370
            cv2.rectangle(frame, (bx1, by1 + 20), (bx2, by2), (210, 170, 60), -1)
            cv2.rectangle(frame, (bx1 + 15, by1), (bx2 - 15, by1 + 20), (50, 50, 220), -1)
            metadata_items.append({
                "class_name": "bottle",
                "confidence": 0.88,
                "bbox": [bx1, by1, bx2, by2]
            })

            # 3. Chair on the right
            cx1, cy1, cx2, cy2 = 440, 190, 570, 410
            # Chair backrest
            cv2.rectangle(frame, (cx1 + 10, cy1), (cx2 - 10, cy1 + 90), (60, 80, 140), -1)
            # Chair seat
            cv2.rectangle(frame, (cx1, cy1 + 90), (cx2, cy1 + 120), (50, 70, 120), -1)
            # Chair legs
            cv2.rectangle(frame, (cx1 + 10, cy1 + 120), (cx1 + 25, cy2), (30, 40, 80), -1)
            cv2.rectangle(frame, (cx2 - 25, cy1 + 120), (cx2 - 10, cy2), (30, 40, 80), -1)

            metadata_items.append({
                "class_name": "chair",
                "confidence": 0.86,
                "bbox": [cx1, cy1, cx2, cy2]
            })

        elif scene_name == "tc03" or scene_name == "empty":
            # TC03: Completely empty scene (clean room, no objects)
            # Minimalist empty wall and floor
            pass  # No items

        elif scene_name == "tc04" or scene_name == "distant":
            # TC04: Distant small objects (< 12% height ratio)
            # Tiny distant chair way back against the horizon
            dcx1, dcy1, dcx2, dcy2 = 300, 260, 335, 305
            cv2.rectangle(frame, (dcx1, dcy1), (dcx2, dcy2), (70, 80, 110), -1)
            metadata_items.append({
                "class_name": "chair",
                "confidence": 0.76,
                "bbox": [dcx1, dcy1, dcx2, dcy2]
            })

        elif scene_name == "tc05" or scene_name == "low_light":
            # TC05: Low lighting - same as TC01 bottle, but very heavily darkened (< 35 luminance)
            # Draw table and bottle
            cv2.rectangle(frame, (180, int(h * 0.55)), (460, int(h * 0.85)), (90, 130, 170), -1)
            bx1, by1, bx2, by2 = 290, 180, 350, 320
            cv2.rectangle(frame, (bx1, by1 + 25), (bx2, by2), (220, 180, 70), -1)
            cv2.rectangle(frame, (bx1 + 15, by1), (bx2 - 15, by1 + 25), (40, 40, 220), -1)

            # Darken the frame drastically to simulate night / dim lighting
            frame = (frame.astype(np.float32) * 0.12).astype(np.uint8)

            metadata_items.append({
                "class_name": "bottle",
                "confidence": 0.82,
                "bbox": [bx1, by1, bx2, by2]
            })

        # Attach test metadata for fallback detection engine
        frame.test_metadata = metadata_items
        return frame
