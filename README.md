# SafePath AI

SafePath AI is a browser-camera prototype for preventive home-safety monitoring.
It runs YOLO object detection on CPU, checks whether a person and a configured
hazard proxy are both inside a walking danger zone, gives a browser voice
warning, and records the high-risk event.

## Current capabilities

- Primary WebRTC camera with front/rear preference and device selection
- Latest-frame asynchronous inference: slow AI drops frames instead of queuing
- Smooth current-frame video with bounding boxes from the latest completed result
- Static or Twilio-ephemeral secret-backed TURN support
- Explicit low-bandwidth snapshot fallback, labeled as not real-time
- YOLO11n and YOLO11s model choices plus a custom-weights path
- CPU inference sizes from 320 to 640; WebRTC defaults to 320 and snapshots to 416
- Model runtime caching and one-time warm-up before camera frames arrive
- Pre-inference frame downscaling with boxes restored to display coordinates
- Live AI, pipeline, annotation-age, dropped-frame, and ICE diagnostics
- Visible trapezoid walking/danger zone and conservative HIGH-risk rule
- Browser voice warnings, event history, and a standalone OpenCV desktop mode

The default COCO hazard proxies are backpack, bottle, chair, handbag, sports
ball, and suitcase. These are demonstration proxies, not real cable, spill, or
fall classes. This prototype is not a certified medical or emergency-alert
device.

## Performance changes

The snapshot fallback captures a 640x480 JPEG at quality 0.68. It waits at least
200 ms between frames and does not send another frame until Streamlit has
finished the previous rerun, so CPU work cannot create an unbounded browser
queue. Before YOLO, a 640x480 frame is reduced to 416x312. Detection boxes are
then scaled back to the original 640x480 frame for risk evaluation and display.

The remaining Streamlit request/rerender delay is architectural. SafePath does
not describe this path as smooth video. In WebRTC mode, the current frame is
returned immediately. At most one YOLO job runs in the background; every frame
that arrives while that job is busy is counted and dropped from AI analysis.
The moving video therefore never waits behind an inference backlog, while the
latest completed assessment is drawn over current frames.

The WebRTC camera requests 15 FPS (20 maximum) and defaults to 320px inference
so the free-tier CPU keeps capacity for video decode/encode. The snapshot
fallback retains the 416px default because it has no continuous video workload.

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

On the deployed Community Cloud app, the 200 ms setting sustained 20 processed
YOLO11n/416 frames in a steady 10-second window (about 2 FPS) with 43 ms average
AI inference. The remaining cadence is Streamlit rerun/transport overhead, not
model inference; backpressure intentionally favors current frames over a queue.

## Run locally on Windows

Python 3.11 or newer is recommended:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`, select a camera direction, press **START** in the
WebRTC panel, and allow camera access. Without TURN secrets, SafePath initially
selects the snapshot fallback; you can still select WebRTC for local testing.
The first model load includes a warm-up pass; later reruns reuse the cached
model runtime.

## Camera modes

### WebRTC real-time

WebRTC is the only continuous real-time path. It becomes the initial sidebar
selection automatically when the resolved ICE configuration contains a TURN
relay. The current video frame passes through independently from YOLO; the
detector accepts one latest frame at a time and drops analysis frames while
busy.

The sidebar sends
`facingMode: environment` or `facingMode: user` as the preferred video
constraint. The WebRTC panel also exposes **SELECT DEVICE** when the browser
allows direct device selection.

The dashboard shows ready, connecting, playing, timeout, and ICE-failure states.
Open **Debug: latency** in the sidebar to see frontend playing/signalling,
server peer connection, ICE connection/gathering, signaling state, incoming /
analyzed / dropped frame counts, callback time, capture-to-result latency, and
annotation age.

### Low-bandwidth snapshot fallback (not real-time)

SafePath's small built-in Streamlit component uses `getUserMedia`, requests the
selected `facingMode` exactly, and falls back to the other camera or the browser
default when that device does not exist. It sends compressed JPEG snapshots
through Streamlit's existing secure connection and does not depend on ICE.
The sidebar debug panel reports a browser-capture-to-server-response estimate.

## Configure TURN securely

Do not commit TURN usernames or credentials. Copy one of the provider shapes
from `.streamlit/secrets.toml.example` into the deployed app's Streamlit
**Secrets** panel.

For Metered, OpenRelay, or coturn static credentials:

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

For Twilio Network Traversal:

```toml
[twilio]
account_sid = "ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
auth_token = "YOUR_TWILIO_AUTH_TOKEN"
```

SafePath sends the SID/token only from the Streamlit server to Twilio's token
endpoint, receives one-hour ephemeral ICE credentials, and caches them for 55
minutes. The browser receives only the temporary TURN username and credential,
not the Twilio account token.

Restart the deployed app after saving secrets. Select WebRTC, press **START**,
then confirm all of the following in **Debug: latency**:

- `frontend_playing` is `true`
- `peer_connection` is `connected`
- `ice_connection` is `connected` or `completed`
- `ice_gathering` is `complete`

Test again from a phone with Wi-Fi disabled. That cellular test cannot be
automated by the repository and is the meaningful proof that relay traversal
works. Without complete credentials SafePath deliberately uses STUN only,
selects the snapshot fallback initially, and shows a warning.

## Detection settings

- **YOLO11n:** fastest and the default for free CPU hosting.
- **YOLO11s:** better general COCO accuracy, but noticeably slower.
- **Custom weights path:** intended for a validated `models/cable.pt` or a
  future multi-hazard model.
- **Inference image size:** WebRTC defaults to 320 to protect the live video
  path; snapshots default to 416. Sizes 512 and 640 can help small objects at
  substantially higher CPU cost.
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

The tests cover danger-zone logic, monitor state and latency, explicit
latest-frame dropping, non-blocking WebRTC output, peer/ICE diagnostics, frame
decoding and resizing, restored box coordinates, one-time model warm-up, shared
runtime use, camera facing/JPEG backpressure and capture timestamps, static
TURN configuration, and Twilio ephemeral-token parsing.

## Project structure

```text
safepath-ai/
|-- app.py                         Streamlit dashboard and mode controls
|-- safepath_camera/               Low-latency getUserMedia component
|   `-- frontend/                  Camera, facingMode, JPEG, backpressure
|-- camera_feed.py                 Frame decoding and inference downscaling
|-- detection.py                   Cached YOLO runtime and desktop camera mode
|-- web_monitor.py                 Thread-safe state, latency, and alerts
|-- rtc_config.py                  STUN, static TURN, and Twilio token exchange
|-- webrtc_diagnostics.py          Safe peer/ICE state inspection
|-- risk_engine.py                 Pure danger-zone decision logic
|-- event_store.py                 CSV and high-risk snapshots
|-- .streamlit/secrets.toml.example
|-- models/                        Future validated custom weights
|-- videos/                        Future labeled validation footage
`-- tests/
```
