"""
Generates synthetic high-resolution test scene assets for Hackathon verification.
Saves images into test_assets/ for direct upload or inspection.
"""
import os
import cv2
from camera_stream import CameraStream
from config import ASSETS_DIR


def generate_all_assets():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    cam = CameraStream(source_type="test_scene_tc01")

    scenes = {
        "tc01_bottle.jpg": "tc01",
        "tc02_multi_object.jpg": "tc02",
        "tc03_empty.jpg": "tc03",
        "tc04_distant.jpg": "tc04",
        "tc05_low_light.jpg": "tc05",
    }

    for filename, scene_code in scenes.items():
        frame = cam.generate_synthetic_test_frame(scene_code)
        out_path = os.path.join(ASSETS_DIR, filename)
        cv2.imwrite(out_path, frame)
        print(f"[Assets] Generated test image: {out_path}")


if __name__ == "__main__":
    generate_all_assets()
