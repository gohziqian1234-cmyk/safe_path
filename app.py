"""Streamlit dashboard for the SafePath AI browser-camera monitor."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit_webrtc import webrtc_streamer

from camera_feed import decode_camera_frame
from detection import LocalDetector, ModelRuntime
from event_store import EventStore
from risk_engine import DEFAULT_HAZARD_LABELS
from rtc_config import build_rtc_configuration, has_turn_relay
from safepath_camera import camera_capture
from web_monitor import BrowserMonitor, MonitorSnapshot


APP_DIRECTORY = Path(__file__).parent
EVENT_STORE = EventStore(APP_DIRECTORY / "outputs")
CLOUD_CAMERA_INTERVAL_MS = 300
CLOUD_CAMERA_WIDTH = 640
CLOUD_CAMERA_HEIGHT = 480
CLOUD_CAMERA_JPEG_QUALITY = 0.68

CAMERA_MODES = (
    "Cloud-compatible low-latency (recommended)",
    "WebRTC real-time (advanced)",
)
CAMERA_FACING_OPTIONS = {
    "Rear / environment camera": "environment",
    "Front / selfie camera": "user",
}
MODEL_OPTIONS = {
    "YOLO11n · fastest (recommended for cloud)": "yolo11n.pt",
    "YOLO11s · more accurate, slower": "yolo11s.pt",
    "Custom weights path": None,
}


st.set_page_config(
    page_title="SafePath AI",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.18);
        padding: 0.85rem;
        border-radius: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def resolve_model_path(model_name: str) -> str:
    """Prefer project-local weights, otherwise let Ultralytics download them."""

    local_candidate = APP_DIRECTORY / model_name
    return str(local_candidate) if local_candidate.exists() else model_name


@st.cache_resource(show_spinner=False)
def load_model_runtime(model_name: str, inference_size: int) -> ModelRuntime:
    """Load and warm each model/size pair once for the whole app process."""

    return ModelRuntime(
        model_path=resolve_model_path(model_name),
        inference_size=inference_size,
        warm_up=True,
    )


def get_detector(
    runtime: ModelRuntime,
    model_name: str,
    confidence: float,
    hazard_labels: tuple[str, ...],
) -> LocalDetector:
    """Create lightweight per-session settings around a shared cached model."""

    detector_key = (
        id(runtime),
        model_name,
        confidence,
        hazard_labels,
    )
    if st.session_state.get("detector_key") != detector_key:
        st.session_state.local_detector = LocalDetector(
            model_path=resolve_model_path(model_name),
            confidence=confidence,
            hazard_labels=hazard_labels,
            inference_size=runtime.inference_size,
            model_runtime=runtime,
        )
        st.session_state.detector_key = detector_key
    return st.session_state.local_detector


def get_monitor(
    detector: LocalDetector,
    monitor_key: tuple[object, ...],
    cooldown_seconds: float,
) -> BrowserMonitor:
    if st.session_state.get("monitor_key") != monitor_key:
        st.session_state.browser_monitor = BrowserMonitor(
            detector=detector,
            event_store=EVENT_STORE,
            cooldown_seconds=cooldown_seconds,
        )
        st.session_state.monitor_key = monitor_key
    return st.session_state.browser_monitor


def read_turn_settings() -> Mapping[str, object] | None:
    """Read optional TURN credentials without requiring a local secrets file."""

    try:
        settings = st.secrets.to_dict().get("turn")
    except StreamlitSecretNotFoundError:
        return None
    return settings if isinstance(settings, Mapping) else None


def render_status(snapshot: MonitorSnapshot, playing: bool) -> None:
    assessment = snapshot.assessment
    system_value = "Monitoring" if playing else "Ready"
    risk_value = assessment.risk_level if assessment else "LOW"
    person_value = "Yes" if assessment and assessment.person_detected else "No"
    hazard_value = assessment.latest_hazard if assessment else "None"
    latency_value = (
        f"{snapshot.average_processing_ms:.0f} ms"
        if snapshot.average_processing_ms
        else "Warming up"
    )

    st.metric("System", system_value)
    st.metric("Risk level", risk_value)
    st.metric("Person detected", person_value)
    st.metric("Latest hazard", hazard_value)
    st.metric("Frames analyzed", snapshot.processed_frames)
    st.metric("Average AI latency", latency_value)


def render_events() -> None:
    st.subheader("Recent high-risk events")
    rows = EVENT_STORE.read_recent(limit=10)
    if not rows:
        st.info("No high-risk events recorded yet.")
        return

    table = pd.DataFrame(rows)
    visible_columns = ["timestamp", "hazard", "risk", "warning_issued", "snapshot"]
    st.dataframe(
        table[visible_columns],
        use_container_width=True,
        hide_index=True,
    )


def speak_in_browser(message: str) -> None:
    """Use the visitor's browser speech engine instead of the cloud speaker."""

    encoded_message = json.dumps(message)
    components.html(
        f"""
        <script>
        const message = {encoded_message};
        if ("speechSynthesis" in window) {{
            window.speechSynthesis.cancel();
            const warning = new SpeechSynthesisUtterance(message);
            warning.rate = 0.9;
            warning.volume = 1.0;
            window.speechSynthesis.speak(warning);
        }}
        </script>
        """,
        height=0,
    )


@st.fragment(run_every=0.5)
def live_status(
    monitor: BrowserMonitor,
    voice_enabled: bool,
    *,
    playing: bool = False,
    webrtc_context=None,
    start_hint: str = "Start the camera above and allow camera access.",
) -> None:
    snapshot = monitor.snapshot()
    if webrtc_context is not None:
        playing = bool(webrtc_context.state.playing)

    if snapshot.last_error:
        st.error(f"Detection error: {snapshot.last_error}")
    elif playing and snapshot.processed_frames == 0:
        st.info("Camera connected. Waiting for the first AI-processed frame...")

    status_column, explanation_column = st.columns([1, 2.2])
    with status_column:
        render_status(snapshot, playing)
    with explanation_column:
        st.markdown(
            "The green trapezoid is the walking danger zone. A **HIGH** risk "
            "event requires both a person and a configured hazard to have their "
            "bottom-center points inside that zone."
        )
        if snapshot.assessment and snapshot.assessment.risk_level == "HIGH":
            st.error(snapshot.assessment.warning_message, icon="⚠️")
        elif playing:
            st.success("Monitoring is active. No high-risk condition is detected.")
        else:
            st.info(start_hint)

    warning_key = (id(monitor), snapshot.warning_sequence)
    if (
        snapshot.warning_sequence
        and st.session_state.get("last_warning_key") != warning_key
    ):
        st.session_state.last_warning_key = warning_key
        st.toast(snapshot.warning_message, icon="⚠️")
        if voice_enabled:
            speak_in_browser(snapshot.warning_message)

    render_events()


@st.fragment(run_every=1.0)
def render_webrtc_connection_status(
    webrtc_context,
    *,
    turn_enabled: bool,
    status_key: str,
) -> None:
    """Expose WebRTC signalling state and a clear cloud-camera fallback."""

    state = webrtc_context.state
    started_key = f"webrtc_started_at_{status_key}"

    if state.playing:
        st.session_state.pop(started_key, None)
        st.success("WebRTC connection: connected")
    elif state.signalling:
        started_at = st.session_state.setdefault(started_key, time.monotonic())
        elapsed = time.monotonic() - started_at
        if elapsed >= 10:
            st.error(
                "WebRTC connection failed to establish. Select "
                "**Cloud-compatible low-latency** in the sidebar."
            )
        else:
            st.info(f"WebRTC connection: connecting ({elapsed:.0f}s)")
    else:
        st.session_state.pop(started_key, None)
        st.info("WebRTC connection: ready — press START below.")

    if turn_enabled:
        st.caption("TURN relay: configured from Streamlit secrets.")
    else:
        st.warning(
            "TURN relay is not configured. WebRTC may fail on mobile carriers, "
            "school Wi-Fi, or symmetric NAT; cloud-compatible mode will still work."
        )


st.title("🛡️ SafePath AI")
st.caption("AI-powered preventive home-safety monitor using your browser camera")

with st.sidebar:
    st.header("Monitoring settings")
    camera_mode = st.selectbox(
        "Camera connection",
        options=CAMERA_MODES,
        help=(
            "Cloud-compatible mode uses compressed JPEG frames with server "
            "backpressure. WebRTC has lower transport latency but needs TURN on "
            "some networks."
        ),
    )
    camera_facing_label = st.selectbox(
        "Preferred camera",
        options=tuple(CAMERA_FACING_OPTIONS),
        help=(
            "Rear camera is requested exactly first. If the device has no rear "
            "camera, SafePath falls back to the front camera."
        ),
    )
    camera_facing_mode = CAMERA_FACING_OPTIONS[camera_facing_label]

    model_profile = st.selectbox(
        "Detection model",
        options=tuple(MODEL_OPTIONS),
        help=(
            "YOLO11n is the free-tier default. YOLO11s improves general-object "
            "accuracy but uses more CPU. Custom cable/fall weights can be entered "
            "below."
        ),
    )
    selected_model = MODEL_OPTIONS[model_profile]
    if selected_model is None:
        model_name = st.text_input(
            "Custom model path",
            value="models/cable.pt",
            help="Add validated custom weights to this path before selecting it.",
        )
    else:
        model_name = selected_model

    inference_size = st.select_slider(
        "Inference image size",
        options=(320, 416, 512, 640),
        value=416,
        help=(
            "416 is the CPU-friendly default. Higher sizes can improve small-object "
            "recall but increase latency."
        ),
    )
    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help=(
            "0.40 remains the baseline until labeled hazard footage is added to "
            "videos/ for precision/recall tuning."
        ),
    )
    selected_hazards = st.multiselect(
        "Household hazard proxies",
        options=sorted(DEFAULT_HAZARD_LABELS),
        default=sorted(DEFAULT_HAZARD_LABELS),
    )
    voice_enabled = st.toggle("Browser voice warnings", value=True)
    cooldown_seconds = st.slider(
        "Warning cooldown (seconds)",
        min_value=3,
        max_value=30,
        value=8,
    )

st.caption(
    "The selected model is loaded, cached, and warmed once. The first launch can "
    "still take a minute while Ultralytics downloads model weights."
)

if not selected_hazards:
    st.error("Select at least one household hazard proxy in the sidebar.")
    st.stop()

try:
    with st.spinner("Loading and warming the AI model..."):
        normalized_model_name = model_name.strip() or "yolo11n.pt"
        hazard_labels = tuple(sorted(selected_hazards))
        runtime = load_model_runtime(
            normalized_model_name,
            int(inference_size),
        )
        detector = get_detector(
            runtime,
            normalized_model_name,
            float(confidence),
            hazard_labels,
        )
except Exception as error:
    st.error(f"SafePath could not load the AI model: {error}")
    st.stop()

monitor_key = (
    id(detector),
    float(cooldown_seconds),
)
monitor = get_monitor(detector, monitor_key, float(cooldown_seconds))

if camera_mode == CAMERA_MODES[0]:
    st.subheader("Cloud-compatible low-latency camera")
    st.caption(
        f"Compressed {CLOUD_CAMERA_WIDTH}×{CLOUD_CAMERA_HEIGHT} JPEG frames, "
        f"{CLOUD_CAMERA_INTERVAL_MS} ms minimum capture interval, server "
        f"backpressure, and {inference_size}px CPU inference."
    )
    cloud_camera_active = st.toggle(
        "Start monitoring",
        key="cloud_camera_active",
    )

    if cloud_camera_active:
        camera_acknowledgement = (
            st.session_state.get("camera_acknowledgement", 0) + 1
        )
        st.session_state.camera_acknowledgement = camera_acknowledgement
        capture = camera_capture(
            interval_ms=CLOUD_CAMERA_INTERVAL_MS,
            width=CLOUD_CAMERA_WIDTH,
            height=CLOUD_CAMERA_HEIGHT,
            jpeg_quality=CLOUD_CAMERA_JPEG_QUALITY,
            facing_mode=camera_facing_mode,
            acknowledgement=camera_acknowledgement,
            key=f"safepath-cloud-camera-{camera_facing_mode}",
        )
        if capture is None or capture.status == "starting":
            st.info("Starting the camera. Choose **Allow** if prompted.")
        elif capture.status == "error":
            st.error(f"Camera error: {capture.error}")
        else:
            if (
                capture.active_facing_mode
                and capture.active_facing_mode != camera_facing_mode
            ):
                st.warning(
                    f"{camera_facing_label} was unavailable; the browser selected "
                    f"the {capture.active_facing_mode} camera instead."
                )

            if capture.image is not None:
                try:
                    frame_digest, frame = decode_camera_frame(capture.image)
                    frame_key = (id(monitor), frame_digest)
                    if st.session_state.get("cloud_frame_key") != frame_key:
                        st.session_state.cloud_annotated_frame = (
                            monitor.process_image(frame)
                        )
                        st.session_state.cloud_frame_key = frame_key

                    annotated_frame = st.session_state.get(
                        "cloud_annotated_frame"
                    )
                    if annotated_frame is not None:
                        st.image(
                            annotated_frame,
                            channels="BGR",
                            caption="Live AI-analyzed camera frame",
                            use_container_width=True,
                        )
                except Exception as error:
                    st.error(f"Could not process the browser frame: {error}")
    else:
        st.info("Switch on **Start monitoring** to open your camera.")

    live_status(
        monitor,
        voice_enabled,
        playing=cloud_camera_active,
        start_hint="Switch on **Start monitoring** above and allow camera access.",
    )
else:
    st.subheader("WebRTC real-time camera")
    st.caption(
        f"Preferred camera: **{camera_facing_label}**. If the browser ignores the "
        "preference, use **SELECT DEVICE** in the camera panel."
    )

    try:
        rtc_configuration = build_rtc_configuration(read_turn_settings())
        turn_configuration_error = ""
    except ValueError as error:
        rtc_configuration = build_rtc_configuration()
        turn_configuration_error = str(error)

    turn_enabled = has_turn_relay(rtc_configuration)
    if turn_configuration_error:
        st.error(f"TURN secrets are incomplete: {turn_configuration_error}")

    webrtc_context = webrtc_streamer(
        key=f"safepath-webrtc-{camera_facing_mode}",
        video_frame_callback=monitor.process_video_frame,
        media_stream_constraints={
            "video": {
                "facingMode": {"ideal": camera_facing_mode},
                "width": {"ideal": CLOUD_CAMERA_WIDTH},
                "height": {"ideal": CLOUD_CAMERA_HEIGHT},
            },
            "audio": False,
        },
        rtc_configuration=rtc_configuration,
        async_processing=True,
    )

    render_webrtc_connection_status(
        webrtc_context,
        turn_enabled=turn_enabled,
        status_key=camera_facing_mode,
    )
    live_status(
        monitor,
        voice_enabled,
        webrtc_context=webrtc_context,
        start_hint="Press **START** in the WebRTC panel and allow camera access.",
    )

st.divider()
st.caption(
    "Privacy: live frames are processed by this Streamlit app and are not kept. "
    "Only a high-risk event snapshot is stored temporarily; cloud storage resets "
    "when the app restarts. This prototype is not a certified emergency-alert device."
)
st.caption(
    "Accuracy limitation: COCO objects such as bags, bottles, chairs, and suitcases "
    "are only hazard proxies. Reliable cable, spill, and fall detection requires a "
    "separately labeled, trained, and validated dataset."
)
