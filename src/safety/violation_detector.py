
from datetime import datetime


PPE_REQUIREMENTS = {
    "helmet": ["helmet", "no_helmet"],
    "safety_vest": ["vest", "no_vest"],
    "gloves": ["gloves", "no_gloves"],
    "boots": ["boots", "no_boots"],
    "goggles": ["goggles", "no_goggle"],
}


def analyze_detections(detections):
    """
    Analyze model detections and report explicit
    missing-PPE detections.

    This is a demo-level detector, not a verified
    person-to-PPE association system.
    """

    detected_classes = {
        item["class"].lower()
        for item in detections
    }

    violations = []

    missing_classes = {
        "no_helmet": "helmet",
        "no_vest": "safety_vest",
        "no_gloves": "gloves",
        "no_boots": "boots",
        "no_goggle": "goggles",
    }

    for class_name, equipment in missing_classes.items():
        if class_name in detected_classes:
            violations.append({
                "type": "missing_ppe",
                "equipment": equipment,
                "detected_class": class_name,
                "severity": "high",
                "timestamp": datetime.now().isoformat()
            })

    return violations


if __name__ == "__main__":
    sample = [
        {"class": "Person"},
        {"class": "no_helmet"},
        {"class": "vest"}
    ]

    print(analyze_detections(sample))
