"""Safe read-only diagnostics for streamlit-webrtc peer connections."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WebRtcDiagnostics:
    playing: bool = False
    signalling: bool = False
    connection_state: str = "not-created"
    ice_connection_state: str = "not-created"
    ice_gathering_state: str = "not-created"
    signaling_state: str = "not-created"


def inspect_webrtc_context(context) -> WebRtcDiagnostics:
    """Read frontend and server-side peer state without failing the UI."""

    if context is None:
        return WebRtcDiagnostics()

    state = getattr(context, "state", None)
    playing = bool(getattr(state, "playing", False))
    signalling = bool(getattr(state, "signalling", False))

    worker = None
    worker_getter = getattr(context, "_get_worker", None)
    if callable(worker_getter):
        try:
            worker = worker_getter()
        except Exception:
            worker = None

    peer_connection = getattr(worker, "pc", None)

    def peer_state(attribute: str) -> str:
        value = getattr(peer_connection, attribute, None)
        return str(value) if value else "not-created"

    return WebRtcDiagnostics(
        playing=playing,
        signalling=signalling,
        connection_state=peer_state("connectionState"),
        ice_connection_state=peer_state("iceConnectionState"),
        ice_gathering_state=peer_state("iceGatheringState"),
        signaling_state=peer_state("signalingState"),
    )
