
from ultralytics import YOLO
from pathlib import Path
import cv2
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]

MODEL = ROOT / "runs/detect/runs/detect/ppe_full_v1/weights/best.pt"
INPUT = ROOT / "output/demo/input"
OUTPUT = ROOT / "output/demo/detections"
EVENTS = ROOT / "output/demo/events"

OUTPUT.mkdir(parents=True, exist_ok=True)
EVENTS.mkdir(parents=True, exist_ok=True)

CONFIDENCE = 0.25

def main():
    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")

    model = YOLO(str(MODEL))
    events = []

    images = list(INPUT.glob("*.jpg"))

    if not images:
        raise RuntimeError("No demo images found")

    for image in images:
        results = model.predict(
            source=str(image),
            conf=CONFIDENCE,
            imgsz=512,
            device=0,
            verbose=False
        )

        result = results[0]
        annotated = result.plot()

        output_file = OUTPUT / image.name
        cv2.imwrite(str(output_file), annotated)

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                detection = {
                    "class": model.names[class_id],
                    "confidence": round(confidence, 3),
                    "box": box.xyxy[0].tolist()
                }

                detections.append(detection)

        events.append({
            "image": image.name,
            "timestamp": datetime.now().isoformat(),
            "detections": detections
        })

        print(
            f"{image.name}: "
            f"{len(detections)} detections"
        )

    event_file = EVENTS / "ppe_events.json"

    event_file.write_text(
        json.dumps(events, indent=2)
    )

    print("\nDEMO INFERENCE COMPLETED")
    print("Annotated images:", OUTPUT)
    print("Events:", event_file)


if __name__ == "__main__":
    main()

