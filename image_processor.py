"""
Image Processing Module
Handles frame preprocessing, low-lighting enhancement (TC05),
and accessible high-contrast annotations.
"""
import cv2
import numpy as np
from config import (
    LOW_LIGHT_LUMINANCE_THRESHOLD,
    CLAHE_CLIP_LIMIT,
    CLAHE_GRID_SIZE,
    GAMMA_CORRECTION_FACTOR
)


class ImageProcessor:
    """
    Processes video frames for object detection and visual display.
    Features:
    - Luminance calculation
    - Automatic low-light enhancement (TC05)
    - High-contrast accessibility visual overlays
    """

    def __init__(self):
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_GRID_SIZE
        )

    def calculate_luminance(self, frame: np.ndarray) -> float:
        """
        Calculates average perceptual luminance of the frame (0.0 to 255.0).
        Uses standard ITU-R BT.601 weights: Y = 0.299*R + 0.587*G + 0.114*B
        """
        if frame is None or frame.size == 0:
            return 0.0

        if len(frame.shape) == 2:
            return float(np.mean(frame))

        # Convert to grayscale to obtain luminance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def is_low_light(self, frame: np.ndarray, threshold: float = LOW_LIGHT_LUMINANCE_THRESHOLD) -> bool:
        """
        TC05: Detects if the scene has low lighting conditions.
        """
        return self.calculate_luminance(frame) < threshold

    def apply_gamma_correction(self, image: np.ndarray, gamma: float = GAMMA_CORRECTION_FACTOR) -> np.ndarray:
        """
        Applies non-linear gamma correction to expand dark pixel intensities.
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        """
        TC05: Enhances dark/low-light images.
        Converts BGR to LAB color space, applies CLAHE to the Luminance (L) channel,
        applies Gamma correction, and converts back to BGR.
        This preserves natural color balance while illuminating shadows.
        """
        if frame is None or frame.size == 0:
            return frame

        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L-channel
        enhanced_l = self.clahe.apply(l_channel)

        # Merge channels back
        merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        # Apply subtle gamma correction for deeper shadows
        return self.apply_gamma_correction(enhanced_bgr, gamma=GAMMA_CORRECTION_FACTOR)

    def process_frame(self, frame: np.ndarray, auto_enhance: bool = True) -> tuple[np.ndarray, dict]:
        """
        Full pipeline for incoming camera frame:
        Returns processed frame and telemetry metadata (luminance, is_low_light, enhanced).
        """
        if frame is None:
            return frame, {"luminance": 0.0, "is_low_light": False, "enhanced": False}

        luminance = self.calculate_luminance(frame)
        low_light = luminance < LOW_LIGHT_LUMINANCE_THRESHOLD
        enhanced = False

        processed = frame.copy()
        if hasattr(frame, "test_metadata"):
            processed.test_metadata = frame.test_metadata

        if low_light and auto_enhance:
            enhanced_frame = self.enhance_low_light(processed)
            if hasattr(frame, "test_metadata"):
                enhanced_frame.test_metadata = frame.test_metadata
            processed = enhanced_frame
            enhanced = True

        metadata = {
            "luminance": round(luminance, 1),
            "is_low_light": low_light,
            "enhanced": enhanced,
            "width": frame.shape[1],
            "height": frame.shape[0]
        }
        return processed, metadata

    def draw_accessibility_annotations(
        self,
        frame: np.ndarray,
        detections: list[dict],
        show_distance: bool = True,
        show_direction: bool = True
    ) -> np.ndarray:
        """
        Draws high-contrast, easily visible bounding boxes and tags
        designed specifically for visually impaired demonstrations.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw guideline dividers for Left / Ahead / Right zones (subtle dotted lines)
        left_x = int(w * 0.35)
        right_x = int(w * 0.65)
        cv2.line(annotated, (left_x, 0), (left_x, h), (100, 100, 100), 1, cv2.LINE_AA)
        cv2.line(annotated, (right_x, 0), (right_x, h), (100, 100, 100), 1, cv2.LINE_AA)

        # Palette for priority colors:
        # High Priority / Danger (Close obstacle) = Bright Red / Orange
        # Medium Priority / Nearby = Yellow / Gold
        # Distant / Normal = Cyan / Bright Green
        for det in detections:
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            label = det["class_name"]
            conf = det["confidence"]
            distance_cat = det.get("distance_category", "Moderate")
            spatial_pos = det.get("spatial_position", "Ahead")
            priority = det.get("priority", 3)

            x1, y1, x2, y2 = map(int, bbox)

            if priority == 1 or distance_cat in ["Immediate", "Very Close"]:
                box_color = (0, 0, 255)       # Red (BGR)
                text_bg_color = (0, 0, 200)
            elif distance_cat == "Close":
                box_color = (0, 165, 255)     # Orange
                text_bg_color = (0, 140, 220)
            elif distance_cat == "Moderate":
                box_color = (0, 255, 255)     # Yellow
                text_bg_color = (0, 200, 200)
            else:
                box_color = (0, 255, 128)     # Bright Green / Cyan
                text_bg_color = (0, 180, 80)

            # Draw thick bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3, cv2.LINE_AA)

            # Construct display tag
            tag_parts = [f"{label.capitalize()} ({int(conf * 100)}%)"]
            if show_distance:
                tag_parts.append(f"[{distance_cat}]")
            if show_direction:
                tag_parts.append(f"@ {spatial_pos}")
            tag = " ".join(tag_parts)

            # Calculate text size for background badge
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(tag, font, font_scale, thickness)

            badge_y1 = max(0, y1 - text_h - 10)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = min(w, x1 + text_w + 12)

            # Draw tag background
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), text_bg_color, -1)
            # Draw white text with black drop-shadow for ultra-clarity
            cv2.putText(
                annotated,
                tag,
                (badge_x1 + 6, badge_y2 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        return annotated
