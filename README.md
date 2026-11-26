# Industrial Worker Safety Monitoring System

An edge-AI-based industrial safety monitoring project designed to detect workers, identify personal protective equipment (PPE), track workers across video frames, and support safety monitoring through a web dashboard.

Developed and tested on an NVIDIA Jetson Orin platform.

> **Academic prototype:** Detection results require human review. This project is not a certified or autonomous workplace safety system.

## Project Overview

The system uses computer vision and a YOLO object-detection model to analyse images and video, identify people and PPE-related classes, and present analysis through a Flask dashboard.

## Key Features

- Detects PPE-related classes: helmets, gloves, safety vests, boots, and goggles.
- Includes missing-PPE classes for helmets, goggles, gloves, boots, and vests.
- Uses YOLO tracking with a custom ByteTrack configuration to associate detections across video frames.
- Aggregates PPE observations over multiple frames to estimate worker safety status.
- Includes polygon-based zone-detection logic.
- Provides a Flask web dashboard and an analysis-history page.
- Generates annotated video output, with H.264 conversion for browser playback compatibility.
- Includes a GPIO simulator for demonstrating a possible hardware-alert workflow.

## Technology Stack

- Python, Ultralytics YOLO, OpenCV
- ByteTrack
- Flask, HTML, CSS
- NVIDIA Jetson Orin
- DeepStream environment exploration and basic pipeline testing

**Implementation note:** The current application uses the YOLO-based detection and tracking workflow. A complete DeepStream YOLO inference integration and physical GPIO integration are not claimed as completed features.

## Model Information

The trained model contains 12 classes:

| Class ID | Class |
|---:|---|
| 0 | helmet |
| 1 | gloves |
| 2 | vest |
| 3 | boots |
| 4 | goggles |
| 5 | Person |
| 6 | no_helmet |
| 7 | no_goggle |
| 8 | no_gloves |
| 9 | no_boots |
| 10 | forklift |
| 11 | no_vest |

### Recorded evaluation metrics

| Metric | Result |
|---|---:|
| Precision | 78.65% |
| Recall | 59.62% |
| mAP@50 | 64.80% |
| mAP@50–95 | 41.66% |

Real-world performance may vary with lighting, camera angle, distance, occlusion, and object size.

## Dataset

The training work used the **Construction-PPE** dataset in YOLO format. The recorded dataset layout contained 800 training images and 200 validation images.

The dataset and trained model weights are not included in this repository. Check the dataset's original license and terms before redistributing it.

## Repository Structure

```text
industrial_safety_monitor/
├── config/
│   ├── project.yaml
│   └── zones.yaml
├── src/
│   ├── dashboard/
│   │   ├── app.py
│   │   ├── bytetrack_custom.yaml
│   │   └── templates/
│   │       ├── history.html
│   │       └── index.html
│   ├── detection/
│   │   ├── demo_pipeline.py
│   │   └── ppe_image_detector.py
│   ├── safety/
│   │   ├── gpio_simulator.py
│   │   ├── violation_detector.py
│   │   └── zone_detector.py
│   └── test_deepstream.py
├── .gitignore
└── requirements.txt
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Armanali4960/industrial-worker-safety-monitoring.git
cd industrial-worker-safety-monitoring
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Some computer-vision and GPU dependencies may require platform-specific installation compatible with the JetPack, CUDA, PyTorch, and TensorRT versions on the target device.

### 4. Add the trained model

Model weights are excluded from Git because they can be large. Place the compatible trained weights at the path expected by `src/dashboard/app.py`, or update the model path in the application.

### 5. Run the dashboard

On the configured Jetson environment:

```bash
python src/dashboard/app.py
```

Open the local address printed by Flask. The usual local development address is:

```text
http://127.0.0.1:5000
```

## System Workflow

1. A user submits an image or video through the dashboard.
2. YOLO detects people, PPE items, and missing-PPE classes.
3. The tracking workflow associates detections across video frames.
4. Safety logic aggregates PPE observations for each tracked worker.
5. Zone logic can evaluate person locations against configured zones.
6. The dashboard presents the analysis and stores history records.
7. Annotated video output can be reviewed in the browser.

## Limitations and Future Work

- Improve recall, especially for small or occluded PPE objects.
- Improve PPE-to-worker association when workers overlap.
- Evaluate on additional unseen industrial environments and camera positions.
- Improve worker re-identification and zone-entry/exit handling.
- Integrate and validate a complete DeepStream inference pipeline.
- Connect physical alert hardware after GPIO integration and electrical safety testing.
- Add automated tests and measured edge-device latency.

## Screenshots and Demo

Add screenshots of the dashboard, analysis history, and annotated detections under a `docs/images/` folder, then link them here. For example:

```markdown
![Dashboard screenshot](docs/images/dashboard.png)
![Detection result](docs/images/detection-result.png)
```

## Safety Notice

This is an educational prototype, not a replacement for certified PPE inspection, workplace safety procedures, or human supervision. False detections and missed detections are possible.

## Author

**Armanali Mulani**

GitHub: [@Armanali4960](https://github.com/Armanali4960)

## License

No license has been specified. Until a license is added, standard copyright applies and reuse permissions are not automatically granted.
