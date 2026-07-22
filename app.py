"""Streamlit dashboard for the SafePath AI browser-camera monitor."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer

from detection import LocalDetector
from event_store import EventStore
from risk_engine import DEFAULT_HAZARD_LABELS
from web_monitor import BrowserMonitor, MonitorSnapshot


APP_DIRECTORY = Path(__file__).parent
EVENT_STORE = EventStore(APP_DIRECTORY / "outputs")
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
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


@st.cache_resource(show_spinner=False)
def load_detector(
    model_name: str,
    confidence: float,
    hazard_labels: tuple[str, ...],
) -> LocalDetector:
    local_candidate = APP_DIRECTORY / model_name
    model_path = str(local_candidate) if local_candidate.exists() else model_name
    return LocalDetector(
        model_path=model_path,
        confidence=confidence,
        hazard_labels=hazard_labels,
    )


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


def render_status(snapshot: MonitorSnapshot, playing: bool) -> None:
    assessment = snapshot.assessment
    system_value = "Monitoring" if playing else "Ready"
    risk_value = assessment.risk_level if assessment else "LOW"
    person_value = "Yes" if assessment and assessment.person_detected else "No"
    hazard_value = assessment.latest_hazard if assessment else "None"

    st.metric("System", system_value)
    st.metric("Risk level", risk_value)
    st.metric("Person detected", person_value)
    st.metric("Latest hazard", hazard_value)
    st.metric("Frames analyzed", snapshot.processed_frames)


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
def live_status(monitor: BrowserMonitor, webrtc_context, voice_enabled: bool) -> None:
    snapshot = monitor.snapshot()
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
            st.info("Click **START** in the camera panel and allow camera access.")

    warning_key = (id(monitor), snapshot.warning_sequence)
    if snapshot.warning_sequence and st.session_state.get("last_warning_key") != warning_key:
        st.session_state.last_warning_key = warning_key
        st.toast(snapshot.warning_message, icon="⚠️")
        if voice_enabled:
            speak_in_browser(snapshot.warning_message)

    render_events()


st.title("🛡️ SafePath AI")
st.caption("AI-powered preventive home-safety monitor using your browser camera")

with st.sidebar:
    st.header("Monitoring settings")
    model_name = st.text_input(
        "YOLO model",
        value="yolo11n.pt",
        help="Use models/cable.pt here after training a custom cable detector.",
    )
    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
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
    "The first launch can take a minute while the small YOLO model loads. "
    "Click START below, then choose **Allow** when your browser asks for camera access."
)

if not selected_hazards:
    st.error("Select at least one household hazard proxy in the sidebar.")
    st.stop()

try:
    with st.spinner("Loading the AI model..."):
        normalized_model_name = model_name.strip() or "yolo11n.pt"
        hazard_labels = tuple(sorted(selected_hazards))
        detector = load_detector(
            normalized_model_name,
            float(confidence),
            hazard_labels,
        )
except Exception as error:
    st.error(f"SafePath could not load the AI model: {error}")
    st.stop()

monitor_key = (
    normalized_model_name,
    float(confidence),
    hazard_labels,
    float(cooldown_seconds),
)
monitor = get_monitor(detector, monitor_key, float(cooldown_seconds))

webrtc_context = webrtc_streamer(
    key="safepath-browser-camera",
    video_frame_callback=monitor.process_video_frame,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTC_CONFIGURATION,
    async_processing=True,
)

live_status(monitor, webrtc_context, voice_enabled)

st.divider()
st.caption(
    "Privacy: live frames are processed by this Streamlit app and are not kept. "
    "Only a high-risk event snapshot is stored temporarily; cloud storage resets "
    "when the app restarts. This prototype is not a certified emergency-alert device."
)
st.caption(
    "MVP limitation: the pretrained model detects common object proxies such as bags, "
    "bottles, chairs, and suitcases. Reliable loose-cable and fall detection requires "
    "a separately trained and validated model."
)
