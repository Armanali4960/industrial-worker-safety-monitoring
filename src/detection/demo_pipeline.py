
from pathlib import Path
import sys
import json
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from src.safety.violation_detector import analyze_detections
from src.safety.zone_detector import detect_zone_violations
from src.safety.gpio_simulator import simulate_machine_lock


MODEL = ROOT / "runs/detect/runs/detect/ppe_full_v1/weights/best.pt"

INPUT = ROOT / "output/demo/input"
OUTPUT = ROOT / "output/demo/detections"
EVENTS = ROOT / "output/demo/events"

OUTPUT.mkdir(parents=True, exist_ok=True)
EVENTS.mkdir(parents=True, exist_ok=True)


def main():

    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")

    print("Loading YOLO model...")
    model = YOLO(str(MODEL))

    all_events = []

    images = sorted(
        p for p in INPUT.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    if not images:
        raise RuntimeError("No input images found")

    for image in images:

        print(f"\nProcessing: {image.name}")

        result = model.predict(
            source=str(image),
            imgsz=512,
            conf=0.25,
            device=0,
            verbose=False
        )[0]

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls.item())
                xyxy = box.xyxy[0].tolist()

                detections.append({
                    "class": model.names[class_id],
                    "confidence": round(
                        float(box.conf.item()), 3
                    ),
                    "box": [
                        round(float(v), 2)
                        for v in xyxy
                    ]
                })

        # 1. PPE violations
        ppe_violations = analyze_detections(detections)

        # 2. Restricted-zone violations
        height, width = result.orig_shape

        zone_violations = detect_zone_violations(
            detections,
            width,
            height
        )

        # 3. Combine safety events
        all_violations = (
            ppe_violations + zone_violations
        )

        # 4. Software-only GPIO simulation
        gpio_event = simulate_machine_lock(
            all_violations
        )

        # 5. Save annotated image
        annotated = result.plot()

        output_path = OUTPUT / image.name

        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(
                f"Could not save: {output_path}"
            )

        # 6. Collect report
        all_events.append({
            "image": image.name,
            "detections": detections,
            "ppe_violations": ppe_violations,
            "zone_violations": zone_violations,
            "violations": all_violations,
            "gpio_simulation": gpio_event
        })

        print("Detections:", len(detections))
        print("PPE violations:", len(ppe_violations))
        print("Zone violations:", len(zone_violations))
        print("GPIO lock requested:",
              gpio_event["lock_requested"])

    # 7. Save JSON report
    event_path = EVENTS / "demo_report.json"

    event_path.write_text(
        json.dumps(all_events, indent=2)
    )

    print("\n========== COMPLETE SAFETY DEMO ==========")
    print("Images processed:", len(all_events))
    print("Total detections:",
          sum(len(x["detections"]) for x in all_events))
    print("PPE violations:",
          sum(len(x["ppe_violations"]) for x in all_events))
    print("Zone violations:",
          sum(len(x["zone_violations"]) for x in all_events))
    print("Annotated images:", OUTPUT)
    print("JSON report:", event_path)


if __name__ == "__main__":
    main()

