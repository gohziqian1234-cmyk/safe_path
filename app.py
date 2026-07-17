"""Streamlit dashboard for the SafePath AI local safety monitor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from detection import LocalDetector, open_camera
from event_store import EventStore
from risk_engine import DEFAULT_HAZARD_LABELS
from voice_alert import VoiceAlert


APP_DIRECTORY = Path(__file__).parent
EVENT_STORE = EventStore(APP_DIRECTORY / "outputs")

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


def initialize_state(cooldown_seconds: float, voice_enabled: bool) -> None:
    defaults = {
        "monitoring": False,
        "camera": None,
        "detector": None,
        "latest_assessment": None,
        "latest_warning_issued": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    alert_settings = (cooldown_seconds, voice_enabled)
    if st.session_state.get("alert_settings") != alert_settings:
        previous_alert = st.session_state.get("voice_alert")
        if previous_alert:
            previous_alert.close()
        st.session_state.voice_alert = VoiceAlert(
            cooldown_seconds=cooldown_seconds,
            enabled=voice_enabled,
        )
        st.session_state.alert_settings = alert_settings


def stop_monitoring() -> None:
    camera = st.session_state.get("camera")
    if camera is not None:
        camera.release()
    st.session_state.camera = None
    st.session_state.monitoring = False


def render_status(assessment, warning_issued: bool) -> None:
    if assessment is None:
        system_value = "Ready"
        risk_value = "LOW"
        person_value = "No"
        hazard_value = "None"
    else:
        system_value = "Monitoring"
        risk_value = assessment.risk_level
        person_value = "Yes" if assessment.person_detected else "No"
        hazard_value = assessment.latest_hazard

    st.metric("System", system_value)
    st.metric("Risk level", risk_value)
    st.metric("Person detected", person_value)
    st.metric("Latest hazard", hazard_value)
    st.metric("Warning issued this frame", "Yes" if warning_issued else "No")


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


@st.fragment(run_every=0.20)
def live_monitor() -> None:
    camera = st.session_state.camera
    detector = st.session_state.detector
    success, frame = camera.read()

    if not success:
        st.error("The camera stopped returning frames. Stop monitoring and reconnect it.")
        return

    annotated_frame, assessment = detector.process_frame(frame)
    warning_issued = False
    if assessment.risk_level == "HIGH":
        warning_issued = st.session_state.voice_alert.trigger(
            assessment.warning_message
        )
        if warning_issued:
            EVENT_STORE.record(
                assessment,
                annotated_frame=annotated_frame,
                warning_issued=True,
            )
            st.toast(assessment.warning_message, icon="⚠️")

    st.session_state.latest_assessment = assessment
    st.session_state.latest_warning_issued = warning_issued

    video_column, status_column = st.columns([2.2, 1])
    with video_column:
        st.image(
            annotated_frame,
            channels="BGR",
            use_container_width=True,
            caption="Local processing — frames are not uploaded",
        )
    with status_column:
        render_status(assessment, warning_issued)

    render_events()


st.title("🛡️ SafePath AI")
st.caption("AI-powered preventive home safety guardian — local laptop MVP")

with st.sidebar:
    st.header("Monitoring settings")
    settings_disabled = st.session_state.get("monitoring", False)
    camera_index = st.number_input(
        "Camera index",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        disabled=settings_disabled,
    )
    model_name = st.text_input(
        "YOLO model",
        value="yolo11n.pt",
        help="Use models/cable.pt here after training a custom cable detector.",
        disabled=settings_disabled,
    )
    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
        disabled=settings_disabled,
    )
    selected_hazards = st.multiselect(
        "Household hazard proxies",
        options=sorted(DEFAULT_HAZARD_LABELS),
        default=sorted(DEFAULT_HAZARD_LABELS),
        disabled=settings_disabled,
    )
    voice_enabled = st.toggle(
        "Voice warnings",
        value=True,
        disabled=settings_disabled,
    )
    cooldown_seconds = st.slider(
        "Warning cooldown (seconds)",
        min_value=3,
        max_value=30,
        value=8,
        disabled=settings_disabled,
    )

initialize_state(float(cooldown_seconds), voice_enabled)

start_column, stop_column, note_column = st.columns([1, 1, 4])
with start_column:
    start_clicked = st.button(
        "▶ Start monitoring",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.monitoring,
    )
with stop_column:
    stop_clicked = st.button(
        "■ Stop",
        use_container_width=True,
        disabled=not st.session_state.monitoring,
    )
with note_column:
    st.caption(
        "First launch downloads the small YOLO model. Allow camera access when prompted."
    )

if start_clicked:
    if not selected_hazards:
        st.error("Select at least one household hazard proxy.")
    else:
        try:
            with st.spinner("Loading the local AI model and opening the camera…"):
                st.session_state.detector = load_detector(
                    model_name.strip() or "yolo11n.pt",
                    float(confidence),
                    tuple(sorted(selected_hazards)),
                )
                st.session_state.camera = open_camera(int(camera_index))
            st.session_state.monitoring = True
            st.rerun()
        except Exception as error:
            stop_monitoring()
            st.error(f"SafePath could not start: {error}")

if stop_clicked:
    stop_monitoring()
    st.rerun()

if st.session_state.monitoring:
    live_monitor()
else:
    feed_column, status_column = st.columns([2.2, 1])
    with feed_column:
        st.info("Press **Start monitoring** to open the webcam and begin local detection.")
        st.markdown(
            "The green trapezoid is the walking danger zone. A **HIGH** risk event "
            "requires both a person and a configured hazard to have their bottom-center "
            "point inside that zone."
        )
    with status_column:
        render_status(st.session_state.latest_assessment, False)
    render_events()

st.divider()
st.caption(
    "MVP limitation: the pretrained model detects common object proxies such as bags, "
    "bottles, chairs, and suitcases. Reliable loose-cable and fall detection requires "
    "a separately trained and validated model."
)
