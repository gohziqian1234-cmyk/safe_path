# SafePath AI

SafePath AI is a browser-camera prototype for preventive home-safety monitoring.
It uses a small YOLO model to detect a person plus common household hazard
proxies, checks whether both are inside a walking danger zone, gives a browser
voice warning, and records the high-risk event.

## What works in this MVP

- Browser webcam support locally and on Streamlit Community Cloud
- Person and common-object detection using `yolo11n.pt`
- Visible trapezoid walking/danger zone
- Conservative `HIGH` risk rule: person in zone + hazard in zone
- Browser voice warnings with an 8-second default cooldown
- Live Streamlit status and temporary event history
- Standalone OpenCV mode for testing the AI on a Windows laptop

The default hazard proxies are backpack, bottle, chair, handbag, sports ball, and
suitcase. A pretrained general-purpose model is **not** a reliable cable or fall
detector. Use `models/cable.pt` only after training and validating a custom model.

## Run locally on Windows

Python 3.11 or newer is recommended. In PowerShell, open this project folder and
run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open `http://localhost:8501`, click **START** inside the camera panel, and
allow camera access. Close Zoom or Teams first if another app is using the camera.

The first launch downloads `yolo11n.pt`. Later runs reuse the local model file.

## Use the public app

Open the deployed Streamlit URL in Chrome or Edge, click **START**, and select
**Allow** when the browser asks for camera permission. The hosted page uses HTTPS,
which browsers require for webcam access.

If the stream cannot connect on a restricted office or school network, try a
normal home/mobile connection. Some restricted networks block WebRTC traffic.

## Test detection without Streamlit

The standalone Windows camera mode is still available:

```powershell
.\.venv\Scripts\python.exe detection.py
```

Press `Q` in the camera window to stop. Useful options:

```powershell
python detection.py --camera 1 --confidence 0.50 --no-voice
```

## Run the automated tests

```powershell
python -m unittest discover -s tests -v
```

The risk and monitor-state tests do not need a camera or downloaded AI weights.

## Project structure

```text
safepath-ai/
|-- app.py               Streamlit dashboard and browser WebRTC camera
|-- web_monitor.py       Thread-safe browser-frame processing and alerts
|-- detection.py         YOLO adapter, annotations, and desktop camera mode
|-- risk_engine.py       Pure danger-zone decision logic
|-- voice_alert.py       Desktop-mode speech and cooldown
|-- event_store.py       CSV and high-risk snapshot records
|-- requirements.txt
|-- packages.txt         Linux OpenCV libraries for Streamlit Cloud
|-- models/              Put custom weights here later
|-- videos/              Optional demo clips
|-- outputs/             Runtime events and snapshots (not committed)
`-- tests/
```

## Demo claim to use

> Our MVP uses real-time object detection and spatial danger-zone analysis.
> Future versions will add a validated cable model, fall/pose analysis, and
> trajectory prediction.

This is a prototype, not a certified medical or emergency-alert device. The
hosted app's runtime records are temporary and reset when the cloud app restarts.
