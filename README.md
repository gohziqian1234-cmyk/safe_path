# SafePath AI

SafePath AI is a local-first laptop prototype for preventive home-safety monitoring.
It uses a webcam and a small YOLO model to detect a person plus common household
hazard proxies, checks whether both are inside a walking danger zone, speaks one
warning, and records the event locally.

## What works in this MVP

- Live webcam detection on the laptop
- Person and common-object detection using `yolo11n.pt`
- Visible trapezoid walking/danger zone
- Conservative `HIGH` risk rule: person in zone + hazard in zone
- Non-blocking voice warning with an 8-second default cooldown
- Streamlit dashboard with live status
- Local CSV event history and annotated snapshots
- Standalone OpenCV mode for testing the AI before the dashboard

The default hazard proxies are backpack, bottle, chair, handbag, sports ball, and
suitcase. A pretrained general-purpose model is **not** a reliable cable or fall
detector. Use `models/cable.pt` only after training and validating a custom model.

## Windows setup

Python 3.11 is recommended. In PowerShell, open this project folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use the virtual-environment Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the complete dashboard

On this prepared laptop, double-click `start_safepath.bat`. You can also run:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Then:

1. Allow camera access for desktop apps in Windows privacy settings if asked.
2. Press **Start monitoring**.
3. Stand in the visible walking zone with a backpack, bottle, or suitcase.
4. Confirm that the risk changes to `HIGH`, one voice warning plays, and an event
   appears in the table.
5. Press **Stop** before closing the browser tab.

The first launch downloads `yolo11n.pt`. Later runs reuse the local model file.
If the camera cannot open, close Zoom/Teams or change the camera index to `1`.

## Test detection without Streamlit

```powershell
.\.venv\Scripts\python.exe detection.py
```

Press `Q` in the camera window to stop. Useful options:

```powershell
python detection.py --camera 1 --confidence 0.50 --no-voice
```

Test the speaker separately:

```powershell
python voice_alert.py
```

If Windows speech synthesis is unavailable, SafePath uses the native Windows
warning beep and prints the speech error in the terminal.

## Run the automated risk tests

```powershell
python -m unittest discover -s tests -v
```

These tests do not need a camera or downloaded AI weights.

## Project structure

```text
safepath-ai/
├── app.py               Streamlit dashboard and live monitoring loop
├── detection.py         YOLO adapter, annotations, and desktop camera mode
├── risk_engine.py       Pure danger-zone decision logic
├── voice_alert.py       Non-blocking speech and cooldown
├── event_store.py       Local CSV and snapshot records
├── requirements.txt
├── models/              Put custom weights here later
├── videos/              Optional demo clips
├── outputs/             Runtime events and snapshots (not committed)
└── tests/
```

## Demo claim to use

> Our MVP uses local real-time object detection and spatial danger-zone analysis.
> Future versions will add a validated cable model, fall/pose analysis, and
> trajectory prediction.

This is a prototype, not a certified medical or emergency-alert device. Keep local
CSV records during the MVP; Huawei Cloud synchronization can be added later once
the team selects a service and supplies credentials.
