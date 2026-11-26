
from datetime import datetime


# Example restricted zone polygon.
# Coordinates are normalized between 0 and 1.
# Replace these values with coordinates appropriate
# to your actual camera view before real deployment.

RESTRICTED_ZONE = [
    (0.60, 0.30),
    (0.95, 0.30),
    (0.95, 0.95),
    (0.60, 0.95)
]


def point_inside_polygon(point, polygon):
    """Return True when a point lies inside a polygon."""

    x, y = point
    inside = False

    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if (yi > y) != (yj > y):
            intersection = (
                (xj - xi) * (y - yi) /
                (yj - yi) + xi
            )

            if x < intersection:
                inside = not inside

        j = i

    return inside


def detect_zone_violations(detections, image_width, image_height):
    """Identify person detections inside the restricted zone."""

    events = []

    for detection in detections:

        if detection.get("class", "").lower() != "person":
            continue

        box = detection.get("box")

        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = box

        # Use bottom-centre of bounding box
        # as approximate worker foot position.
        foot_x = ((x1 + x2) / 2) / image_width
        foot_y = y2 / image_height

        if point_inside_polygon(
            (foot_x, foot_y),
            RESTRICTED_ZONE
        ):
            events.append({
                "type": "restricted_zone_entry",
                "severity": "high",
                "detected_class": "Person",
                "position": {
                    "x": round(foot_x, 3),
                    "y": round(foot_y, 3)
                },
                "timestamp": datetime.now().isoformat()
            })

    return events
