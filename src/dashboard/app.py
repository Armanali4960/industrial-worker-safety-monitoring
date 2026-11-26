from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash
from pathlib import Path
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from datetime import datetime
import json
import cv2
import uuid
import sys
import subprocess
import os

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "runs/detect/runs/detect/ppe_full_v1/weights/best.pt"
UPLOAD_DIR = ROOT / "output/uploads"
IMAGE_DIR = ROOT / "output/demo/detections"
VIDEO_DIR = ROOT / "output/demo/videos"
EVENT_FILE = ROOT / "output/demo/events/demo_report.json"

for folder in [UPLOAD_DIR, IMAGE_DIR, VIDEO_DIR, EVENT_FILE.parent]:
    folder.mkdir(parents=True, exist_ok=True)

ALLOWED = {"jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"}
MAX_SIZE = 300 * 1024 * 1024

app = Flask(__name__)
app.secret_key = "industrial-safety-local-demo"
app.config["MAX_CONTENT_LENGTH"] = MAX_SIZE

model = None
latest_event = None


def get_model():
    global model
    if model is None:
        model = YOLO(str(MODEL_PATH))
    return model


def load_events():
    if not EVENT_FILE.exists():
        return []
    try:
        data = json.loads(EVENT_FILE.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_event(event):
    events = load_events()
    events.insert(0, event)
    EVENT_FILE.write_text(json.dumps(events, indent=2))


def is_violation_class(name):
    return name.lower() in {
        "no_helmet", "no_vest", "no_gloves",
        "no_boots", "no_goggle"
    }



def worker_records(detections, track_ids=None):
    people = [d for d in detections if d["class"].lower() == "person"]
    workers = []

    ppe_classes = {
        "helmet": ("helmet", "no_helmet"),
        "vest": ("vest", "no_vest"),
        "gloves": ("gloves", "no_gloves"),
        "boots": ("boots", "no_boots"),
        "goggles": ("goggles", "no_goggle")
    }

    for index, person in enumerate(people):
        x1, y1, x2, y2 = person["box"]
        matched = []

        ppe_names = {name for pair in ppe_classes.values() for name in pair}

        for item in detections:
            cname = item["class"].lower()
            if cname not in ppe_names:
                continue

            a, b, c, d = item["box"]
            cx, cy = (a + c) / 2, (b + d) / 2
            candidates = []

            for person_index, candidate in enumerate(people):
                px1, py1, px2, py2 = candidate["box"]

                if px1 <= cx <= px2 and py1 <= cy <= py2:
                    ix1, iy1 = max(px1, a), max(py1, b)
                    ix2, iy2 = min(px2, c), min(py2, d)
                    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

                    person_area = (px2 - px1) * (py2 - py1)
                    ppe_area = (c - a) * (d - b)
                    union = person_area + ppe_area - intersection
                    score = intersection / ppe_area if ppe_area > 0 else 0
                    candidates.append((score, -person_area, person_index))

            if candidates and max(candidates)[2] == index:
                matched.append(cname)

        ppe_status = {}
        missing = []
        detected = []

        for category, (positive, negative) in ppe_classes.items():
            if negative in matched:
                ppe_status[category] = "MISSING"
                missing.append(category)
            elif positive in matched:
                ppe_status[category] = "DETECTED"
                detected.append(category)
            else:
                ppe_status[category] = "UNCERTAIN"

        if missing:
            status = "VIOLATION"
        elif "UNCERTAIN" in ppe_status.values():
            status = "NEEDS REVIEW"
        else:
            status = "NO DETECTED VIOLATION"

        worker_id = (
            str(track_ids[index]) if track_ids and index < len(track_ids)
            and track_ids[index] is not None
            else f"IMG-{index + 1:02d}"
        )

        workers.append({
            "worker_id": worker_id,
            "name": "",
            "status": status,
            "ppe_status": ppe_status,
            "missing_ppe": missing,
            "detected_ppe": detected,
            "zone_status": "Not individually verified"
        })

    return workers


def extract_detections(result, detector):
    detections = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls.item())
        detections.append({
            "class": detector.names[class_id],
            "confidence": round(float(box.conf.item()), 3),
            "box": [round(float(v), 2) for v in box.xyxy[0].tolist()]
        })
    return detections


def process_image(path, unique_name):
    detector = get_model()
    result = detector.predict(
        source=str(path), imgsz=512, conf=0.25,
        device=0, verbose=False
    )[0]

    detections = extract_detections(result, detector)
    workers = worker_records(detections)

    annotated = result.plot()

    # Draw worker status labels around person detections.
    people = [d for d in detections if d["class"].lower() == "person"]
    for i, person in enumerate(people):
        x1, y1, x2, y2 = map(int, person["box"])
        worker = workers[i]
        bad = worker["status"] == "VIOLATION"
        color = (0, 0, 255) if bad else (0, 200, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            annotated, f'{worker["worker_id"]} - {worker["status"]}',
            (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, color, 2
        )

    output_name = unique_name.rsplit(".", 1)[0] + ".jpg"
    cv2.imwrite(str(IMAGE_DIR / output_name), annotated)

    ppe = [d for d in detections if is_violation_class(d["class"])]
    event = {
        "id": uuid.uuid4().hex[:10],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": path.name,
        "media_type": "image",
        "image": output_name,
        "video": "",
        "detections": detections,
        "workers": workers,
        "ppe_violations": ppe,
        "zone_violations": [],
        "violations": ppe,
        "alert": bool(ppe),
        "note": "PPE-to-worker association is approximate. Zone status is not individually verified."
    }
    save_event(event)
    return event


def process_video(path, unique_name):
    detector = get_model()
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 20

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_name = unique_name.rsplit(".", 1)[0] + "_annotated.mp4"
    output_path = VIDEO_DIR / output_name
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    worker_summary = {}
    violation_frames = 0
    frame_count = 0
    max_frames = 600

    ppe_categories = ["helmet", "vest", "gloves", "boots", "goggles"]

    try:
        while frame_count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            results = detector.track(
                source=frame, persist=True,
                tracker="src/dashboard/bytetrack_custom.yaml",
                imgsz=512, conf=0.25, device=0, verbose=False
            )
            result = results[0]
            annotated = result.plot()

            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()

                detections = extract_detections(result, detector)
                person_ids = [
                    int(track_id)
                    for cls_id, track_id in zip(classes, ids)
                    if detector.names[cls_id].lower() == "person"
                ]

                frame_workers = worker_records(detections, person_ids)

                for worker in frame_workers:
                    wid = str(worker["worker_id"])
                    record = worker_summary.setdefault(wid, {
                        "worker_id": f"Worker {wid}",
                        "name": "",
                        "status": "NEEDS REVIEW",
                        "ppe_status": {},
                        "missing_ppe": [],
                        "detected_ppe": [],
                        "frames_seen": 0,
                        "_present": {c: 0 for c in ppe_categories},
                        "_missing": {c: 0 for c in ppe_categories}
                    })

                    record["frames_seen"] += 1

                    for category, value in worker["ppe_status"].items():
                        if value == "DETECTED":
                            record["_present"][category] += 1
                        elif value == "MISSING":
                            record["_missing"][category] += 1

                    record["ppe_status"] = {}
                    record["missing_ppe"] = []
                    record["detected_ppe"] = []

                    for category in ppe_categories:
                        present = record["_present"][category]
                        missing = record["_missing"][category]

                        if missing >= 2:
                            value = "MISSING"
                            record["missing_ppe"].append(category)
                        elif present >= 2 and missing == 0:
                            value = "DETECTED"
                            record["detected_ppe"].append(category)
                        else:
                            value = "UNCERTAIN"

                        record["ppe_status"][category] = value

                    if record["missing_ppe"]:
                        record["status"] = "VIOLATION"
                    elif all(
                        value == "DETECTED"
                        for value in record["ppe_status"].values()
                    ):
                        record["status"] = "NO DETECTED VIOLATION"
                    else:
                        record["status"] = "NEEDS REVIEW"

                    if record["status"] == "VIOLATION":
                        violation_frames += 1

                for cls_id, coords, track_id in zip(
                    classes,
                    result.boxes.xyxy.cpu().tolist(),
                    ids
                ):
                    if detector.names[cls_id].lower() != "person":
                        continue

                    x1, y1, x2, y2 = map(int, coords)
                    record = worker_summary.get(str(track_id), {})
                    status = record.get("status", "NEEDS REVIEW")
                    color = (0, 0, 255) if status == "VIOLATION" else (0, 165, 255)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(
                        annotated, f"Worker {track_id}: {status}",
                        (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2
                    )

            writer.write(annotated)
            frame_count += 1

    finally:
        cap.release()
        writer.release()

    if frame_count == 0:
        raise RuntimeError("No video frames were processed")

    playback_name = output_name
    h264_path = output_path.with_name(output_path.stem + "_firefox.mp4")

    ffmpeg_env = os.environ.copy()
    ffmpeg_env["LD_LIBRARY_PATH"] = (
        "/tmp/ffmpeg-test/root/usr/lib/aarch64-linux-gnu:"
        + ffmpeg_env.get("LD_LIBRARY_PATH", "")
    )

    try:
        subprocess.run(
            [
                "/tmp/ffmpeg-test/root/usr/bin/ffmpeg",
                "-y", "-hwaccel", "none",
                "-i", str(output_path),
                "-an", "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(h264_path)
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=ffmpeg_env
        )
        playback_name = h264_path.name
        print("H.264 conversion successful:", playback_name)
    except Exception as e:
        print("H.264 conversion failed; using original video:", e)

    workers = list(worker_summary.values())
    for worker in workers:
        worker.pop("_present", None)
        worker.pop("_missing", None)
    event = {
        "id": uuid.uuid4().hex[:10],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": path.name,
        "media_type": "video",
        "image": "",
        "video": playback_name,
        "detections": [],
        "workers": workers,
        "ppe_violations": [],
        "zone_violations": [],
        "violations": [],
        "alert": any(w["status"] == "VIOLATION" for w in workers),
        "frames_processed": frame_count,
        "note": "Worker IDs are video tracker IDs. PPE association is approximate. Video processing is limited to 600 frames."
    }
    save_event(event)
    return event


@app.route("/", methods=["GET", "POST"])
def index():
    global latest_event
    if request.method == "POST":
        uploaded = request.files.get("media")
        if not uploaded or not uploaded.filename:
            flash("Choose an image or video first.")
            return redirect(url_for("index"))

        safe_name = secure_filename(uploaded.filename)
        ext = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
        if ext not in ALLOWED:
            flash("Supported: JPG, JPEG, PNG, MP4, AVI, MOV, MKV.")
            return redirect(url_for("index"))

        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        input_path = UPLOAD_DIR / unique_name
        uploaded.save(str(input_path))

        try:
            if ext in {"jpg", "jpeg", "png"}:
                event = process_image(input_path, unique_name)
            else:
                event = process_video(input_path, unique_name)
            latest_event = event
            flash(f"Processing complete. Record ID: {event['id']}")
        except Exception as error:
            app.logger.exception("Processing failed")
            flash(f"Processing failed: {error}")
        finally:
            if input_path.exists():
                input_path.unlink()

        return redirect(url_for("index"))

    events = [latest_event] if latest_event else []
    return render_template(
        "index.html",
        events=events,
        total_files=len(events),
        total_workers=sum(len(e.get("workers", [])) for e in events),
        total_alerts=sum(1 for e in events if e.get("alert"))
    )


@app.route("/history")
def history():
    return render_template("history.html", events=load_events())


@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory(IMAGE_DIR, filename)


@app.route("/videos/<path:filename>")
def videos(filename):
    return send_from_directory(VIDEO_DIR, filename)


@app.route("/api/events")
def api_events():
    return jsonify(load_events())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
