"""
Alert Module
Manages proximity evaluation, collision warning triggers,
and spatial navigation advice for visually impaired users.
"""
from config import LEFT_ZONE_BOUNDARY, RIGHT_ZONE_BOUNDARY


class AlertManager:
    """
    Evaluates obstacles and creates high-priority collision warnings
    and navigation recommendations.
    """

    def __init__(self):
        pass

    def evaluate_scene(self, detections: list[dict], frame_width: int, frame_height: int) -> dict:
        """
        Analyzes the full set of detections to compute:
        - Critical collision alert (if obstacle is immediate)
        - Path status (Clear / Obstructed ahead / Left blocked / Right blocked)
        - Recommended action / navigation guidance
        """
        center_obstacles = []
        left_obstacles = []
        right_obstacles = []
        immediate_threats = []

        for det in detections:
            s_pos = det.get("spatial_position", "")
            dist = det.get("distance_category", "Moderate")
            c_name = det.get("class_name", "object")

            if dist == "Immediate":
                immediate_threats.append(det)

            if "ahead" in s_pos:
                center_obstacles.append(det)
            elif "left" in s_pos:
                left_obstacles.append(det)
            elif "right" in s_pos:
                right_obstacles.append(det)

        # Determine path status
        has_critical_danger = len(immediate_threats) > 0
        center_blocked = any(d.get("distance_category") in ["Immediate", "Close"] for d in center_obstacles)
        left_blocked = any(d.get("distance_category") in ["Immediate", "Close"] for d in left_obstacles)
        right_blocked = any(d.get("distance_category") in ["Immediate", "Close"] for d in right_obstacles)

        navigation_guidance = "Path ahead is clear."
        guidance_level = "info"  # "info", "warning", "danger"

        if has_critical_danger:
            threat_name = immediate_threats[0]["class_name"]
            navigation_guidance = f"HALT! {threat_name.capitalize()} directly in your path!"
            guidance_level = "danger"
        elif center_blocked:
            if not right_blocked:
                navigation_guidance = "Obstacle directly ahead. Veer slightly right."
                guidance_level = "warning"
            elif not left_blocked:
                navigation_guidance = "Obstacle directly ahead. Veer slightly left."
                guidance_level = "warning"
            else:
                navigation_guidance = "Path ahead is congested. Proceed with caution."
                guidance_level = "warning"
        elif left_blocked and not right_blocked:
            navigation_guidance = "Obstacle on your left. Keep right."
            guidance_level = "info"
        elif right_blocked and not left_blocked:
            navigation_guidance = "Obstacle on your right. Keep left."
            guidance_level = "info"

        return {
            "critical_danger": has_critical_danger,
            "immediate_threats": [t["class_name"] for t in immediate_threats],
            "center_blocked": center_blocked,
            "left_blocked": left_blocked,
            "right_blocked": right_blocked,
            "navigation_guidance": navigation_guidance,
            "guidance_level": guidance_level,
            "obstacle_counts": {
                "ahead": len(center_obstacles),
                "left": len(left_obstacles),
                "right": len(right_obstacles),
            }
        }
