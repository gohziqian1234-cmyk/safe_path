# SafePath AI

SafePath AI is a browser-camera prototype for preventive home-safety monitoring.
It runs YOLO object detection on CPU, checks whether a person and a configured
hazard proxy are both inside a walking danger zone, gives a browser voice
warning, and records the high-risk event.

## Current capabilities

- Low-latency cloud camera with front/rear preference and automatic fallback
- Optional WebRTC mode with front/rear preference and device selection
- Secret-backed TURN support for restrictive networks
- YOLO11n and YOLO11s model choices plus a custom-weights path
- CPU inference sizes from 320 to 640; 416 is the cloud default
- Model runtime caching and one-time warm-up before camera frames arrive
- Pre-inference frame downscaling with boxes restored to display coordinates
- Live average AI-latency metric for benchmarking on the actual host
- Visible trapezoid walking/danger zone and conservative HIGH-risk rule
- Browser voice warnings, event history, and a standalone OpenCV desktop mode

The default COCO hazard proxies are backpack, bottle, chair, handbag, sports
ball, and suitcase. These are demonstration proxies, not real cable, spill, or
fall classes. This prototype is not a certified medical or emergency-alert
device.

## Performance changes

The cloud camera captures a 640x480 JPEG at quality 0.68. It waits at least
200 ms between frames and does not send another frame until Streamlit has
finished the previous rerun, so CPU work cannot create an unbounded browser
queue. Before YOLO, a 640x480 frame is reduced to 416x312. Detection boxes are
then scaled back to the original 640x480 frame for risk evaluation and display.

Local Windows CPU measurements from the same synthetic 640x480 input:

| Model and inference path | First camera frame | Steady average |
|---|---:|---:|
| YOLO11n, old 640 path without warm-up | 8,829 ms | 165 ms |
| YOLO11n, warmed and downscaled to 416 | 145 ms | 102 ms |
| YOLO11s, warmed and downscaled to 416 | 212 ms | 186 ms |

YOLO11s was about 82% slower than optimized YOLO11n in this CPU test. YOLO11n
therefore remains the Streamlit Community Cloud default. Host speed varies, so
the dashboard reports **Average AI latency** from the current Streamlit Cloud
process; switch models while monitoring to compare the real deployment.

## Run locally on Windows

Python 3.11 or newer is recommended:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`, select a camera direction, switch on
**Start monitoring**, and allow camera access. The first model load includes a
warm-up pass; later reruns reuse the cached model runtime.

## Camera modes

### Cloud-compatible low-latency

This is the public-app default. SafePath's small built-in Streamlit component
uses `getUserMedia`, requests the selected `facingMode` exactly, and falls back
to the other camera or the browser default when that device does not exist. It
sends compressed JPEG frames through Streamlit's existing secure connection and
does not depend on peer-to-peer ICE negotiation.

### WebRTC real-time

WebRTC has the lowest transport latency when ICE succeeds. The sidebar sends
`facingMode: environment` or `facingMode: user` as the preferred video
constraint. The WebRTC panel also exposes **SELECT DEVICE** when the browser
allows direct device selection.

The dashboard shows Ready, Connecting, Connected, or a ten-second failure
message. If WebRTC cannot connect, switch back to cloud-compatible mode.

## Configure TURN securely

Do not commit TURN usernames or credentials. Copy the shape from
`.streamlit/secrets.toml.example` into the deployed app's Streamlit **Secrets**
panel:

```toml
[turn]
urls = [
  "turn:YOUR_TURN_HOST:443?transport=udp",
  "turn:YOUR_TURN_HOST:443?transport=tcp",
  "turns:YOUR_TURN_HOST:443?transport=tcp",
]
username = "YOUR_EPHEMERAL_USERNAME"
credential = "YOUR_EPHEMERAL_CREDENTIAL"
```

Metered, Twilio Network Traversal, or a private coturn server can provide these
values. Without all three fields, SafePath deliberately uses STUN only and
shows a warning; the cloud-compatible camera remains available.

## Detection settings

- **YOLO11n:** fastest and the default for free CPU hosting.
- **YOLO11s:** better general COCO accuracy, but noticeably slower.
- **Custom weights path:** intended for a validated `models/cable.pt` or a
  future multi-hazard model.
- **Inference image size:** 416 is the speed/accuracy default; 320 is faster,
  while 512 and 640 can help small objects at higher CPU cost.
- **Confidence:** remains 0.40 because `videos/` currently has no labeled hazard
  footage. Lowering it without validation would simply trade false negatives
  for unknown false-positive rates.

## Minimal custom cable/spill/fall dataset plan

1. Collect 500-1,500 diverse images or sampled video frames across different
   homes, rooms, floors, lighting, camera heights, cable colors, and device
   types. Include at least as many negative scenes as hazardous scenes.
2. Label cable/wire and spill regions with consistent bounding boxes. For falls,
   use a separate `fallen_person` class only if the visual definition is
   consistent; otherwise train a pose/temporal fall model rather than pretending
   one still image proves a fall.
3. Split by household or recording session, not by neighboring video frames:
   approximately 70% train, 20% validation, and 10% held-out test. Double-review
   10-20% of labels and resolve disagreements.
4. Fine-tune on a GPU workstation or Colab, not Streamlit Cloud:

   ```bash
   yolo detect train data=safepath.yaml model=yolo11s.pt imgsz=640 \
     epochs=80 patience=15 batch=8
   ```

5. Evaluate class precision, recall, mAP, and—more importantly—false alarms per
   hour and missed hazards on unseen homes. Add representative labeled clips to
   `videos/`, sweep confidence thresholds, and choose the threshold from those
   results rather than intuition.
6. Place the accepted weights at `models/cable.pt`, select **Custom weights
   path**, and re-run both automated tests and the held-out video evaluation.

## Standalone desktop mode

The OpenCV CLI remains available and keeps its 640 inference-size default:

```powershell
.\.venv\Scripts\python.exe detection.py
```

Useful options:

```powershell
python detection.py --camera 1 --model yolo11s.pt --imgsz 416 \
  --confidence 0.40 --no-voice
```

Press `Q` in the camera window to stop.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover danger-zone logic, monitor state and latency, frame decoding and
resizing, restored box coordinates, one-time model warm-up, shared runtime use,
camera facing/JPEG backpressure, and secret-backed TURN configuration.

## Project structure

```text
safepath-ai/
|-- app.py                         Streamlit dashboard and mode controls
|-- safepath_camera/               Low-latency getUserMedia component
|   `-- frontend/                  Camera, facingMode, JPEG, backpressure
|-- camera_feed.py                 Frame decoding and inference downscaling
|-- detection.py                   Cached YOLO runtime and desktop camera mode
|-- web_monitor.py                 Thread-safe state, latency, and alerts
|-- rtc_config.py                  STUN plus optional secret-backed TURN
|-- risk_engine.py                 Pure danger-zone decision logic
|-- event_store.py                 CSV and high-risk snapshots
|-- .streamlit/secrets.toml.example
|-- models/                        Future validated custom weights
|-- videos/                        Future labeled validation footage
`-- tests/
```
