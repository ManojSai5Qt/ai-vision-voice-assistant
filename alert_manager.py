"""
Alert Module
Manages proximity evaluation, collision warning triggers, crowd aggregation,
and intelligent spatial navigation advice for visually impaired users.
"""
from config import LEFT_ZONE_BOUNDARY, RIGHT_ZONE_BOUNDARY

# Specific obstacle categories for real-life environments
GROUND_COLLISION_HAZARDS = {
    "person", "chair", "couch", "bicycle", "motorcycle", "car", "bus", "truck",
    "dog", "bench", "fire hydrant", "potted plant", "stairs", "suitcase", "trash can"
}

HANDHELD_DESK_OBJECTS = {
    "bottle", "cup", "cell phone", "laptop", "mouse", "keyboard", "book", "clock",
    "remote", "apple", "banana", "sandwich"
}


class AlertManager:
    """
    Evaluates obstacles, crowd density, and walking path clearance
    with ground-corridor geometry and intelligent audio prioritization.
    """

    def __init__(self):
        self.last_crowd_alert_time = 0.0

    def evaluate_scene(self, detections: list[dict], frame_width: int, frame_height: int,
                       mode: str = "street_walking") -> dict:
        """
        Analyzes the full set of detections to compute:
        - Critical collision alert (if immediate threat in walking corridor)
        - Crowd density assessment (aggregates multiple people into a calm summary)
        - Path status (Clear / Obstructed ahead / Left clearer / Right clearer)
        - Recommended action / navigation guidance
        """
        center_obstacles = []
        left_obstacles = []
        right_obstacles = []
        immediate_threats = []
        people_in_path = []

        for det in detections:
            s_pos = det.get("spatial_position", "")
            dist = det.get("distance_category", "Moderate")
            c_name = det.get("class_name", "object").lower()
            bbox = det.get("bbox", [0, 0, frame_width, frame_height])
            bottom_y = bbox[3] if len(bbox) >= 4 else frame_height

            # Count people for crowd summarizer
            if c_name == "person":
                people_in_path.append(det)

            # Ground Footpath Corridor: obstacle must reach lower 55% of the frame to be a trip hazard
            is_ground_obstacle = (bottom_y / max(1, frame_height)) >= 0.40

            if dist == "Immediate" and is_ground_obstacle:
                immediate_threats.append(det)

            if "ahead" in s_pos:
                center_obstacles.append(det)
            elif "left" in s_pos:
                left_obstacles.append(det)
            elif "right" in s_pos:
                right_obstacles.append(det)

        # Crowd Density Evaluation
        is_crowded = len(people_in_path) >= 3
        crowd_count = len(people_in_path)

        # Determine path status
        has_critical_danger = len(immediate_threats) > 0
        center_blocked = any(d.get("distance_category") in ["Immediate", "Close"] for d in center_obstacles)
        left_blocked = any(d.get("distance_category") in ["Immediate", "Close"] for d in left_obstacles)
        right_blocked = any(d.get("distance_category") in ["Immediate", "Close"] for d in right_obstacles)

        navigation_guidance = "Path ahead is clear."
        guidance_level = "info"  # "info", "warning", "danger"

        if has_critical_danger:
            threat_name = immediate_threats[0]["class_name"]
            navigation_guidance = f"HALT! {threat_name.capitalize()} directly in your walking path!"
            guidance_level = "danger"
        elif is_crowded:
            # Intelligent crowd summarization
            if not right_blocked and left_blocked:
                navigation_guidance = f"Crowded ahead ({crowd_count} people). Path clearer on your right."
            elif not left_blocked and right_blocked:
                navigation_guidance = f"Crowded ahead ({crowd_count} people). Path clearer on your left."
            else:
                navigation_guidance = f"Crowded area ahead ({crowd_count} people). Proceed with short steps."
            guidance_level = "warning"
        elif center_blocked:
            if not right_blocked:
                navigation_guidance = "Obstacle ahead. Veer slightly right."
                guidance_level = "warning"
            elif not left_blocked:
                navigation_guidance = "Obstacle ahead. Veer slightly left."
                guidance_level = "warning"
            else:
                navigation_guidance = "Path ahead congested. Slow down."
                guidance_level = "warning"
        elif left_blocked and not right_blocked:
            navigation_guidance = "Obstacle on left. Keep right."
            guidance_level = "info"
        elif right_blocked and not left_blocked:
            navigation_guidance = "Obstacle on right. Keep left."
            guidance_level = "info"

        return {
            "critical_danger": has_critical_danger,
            "immediate_threats": [t["class_name"] for t in immediate_threats],
            "center_blocked": center_blocked,
            "left_blocked": left_blocked,
            "right_blocked": right_blocked,
            "is_crowded": is_crowded,
            "crowd_count": crowd_count,
            "navigation_guidance": navigation_guidance,
            "guidance_level": guidance_level,
            "obstacle_counts": {
                "ahead": len(center_obstacles),
                "left": len(left_obstacles),
                "right": len(right_obstacles),
            }
        }
