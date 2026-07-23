"""WebRTC ICE configuration with optional secret-backed TURN credentials."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


DEFAULT_STUN_URLS = (
    "stun:stun.l.google.com:19302",
    "stun:stun1.l.google.com:19302",
)


def _normalize_urls(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def build_rtc_configuration(
    turn_settings: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build STUN configuration and append a TURN relay when secrets exist."""

    ice_servers: list[dict[str, object]] = [
        {"urls": list(DEFAULT_STUN_URLS)},
    ]

    if turn_settings:
        urls = _normalize_urls(turn_settings.get("urls"))
        username = str(turn_settings.get("username") or "").strip()
        credential = str(turn_settings.get("credential") or "").strip()
        supplied_values = bool(urls or username or credential)
        complete = bool(urls and username and credential)
        if supplied_values and not complete:
            raise ValueError(
                "TURN secrets require urls, username, and credential."
            )
        if complete:
            ice_servers.append(
                {
                    "urls": urls,
                    "username": username,
                    "credential": credential,
                }
            )

    return {
        "iceServers": ice_servers,
        "iceCandidatePoolSize": 10,
    }


def has_turn_relay(configuration: Mapping[str, object]) -> bool:
    """Return whether an ICE configuration contains a TURN/TURNS URL."""

    for server in configuration.get("iceServers", []):
        if not isinstance(server, Mapping):
            continue
        for url in _normalize_urls(server.get("urls")):
            if url.startswith(("turn:", "turns:")):
                return True
    return False
