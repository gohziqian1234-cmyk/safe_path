"""WebRTC ICE configuration with optional secret-backed TURN credentials."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_STUN_URLS = (
    "stun:stun.l.google.com:19302",
    "stun:stun1.l.google.com:19302",
)
TWILIO_TOKEN_TTL_SECONDS = 3600


def _normalize_urls(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _normalize_ice_server(server: object) -> dict[str, object] | None:
    if isinstance(server, Mapping):
        urls = _normalize_urls(server.get("urls") or server.get("url"))
        username = str(server.get("username") or "").strip()
        credential = str(server.get("credential") or "").strip()
    else:
        urls = _normalize_urls(getattr(server, "urls", None))
        username = str(getattr(server, "username", "") or "").strip()
        credential = str(getattr(server, "credential", "") or "").strip()

    if not urls:
        return None
    normalized: dict[str, object] = {"urls": urls}
    if username:
        normalized["username"] = username
    if credential:
        normalized["credential"] = credential
    return normalized


def build_rtc_configuration(
    turn_settings: Mapping[str, object] | None = None,
    additional_ice_servers: Sequence[object] | None = None,
) -> dict[str, object]:
    """Build STUN plus static or short-lived TURN relay configuration."""

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

    for server in additional_ice_servers or ():
        normalized = _normalize_ice_server(server)
        if normalized and normalized not in ice_servers:
            ice_servers.append(normalized)

    return {
        "iceServers": ice_servers,
        "iceCandidatePoolSize": 10,
    }


def has_turn_relay(configuration: Mapping[str, object]) -> bool:
    """Return whether an ICE configuration contains a TURN/TURNS URL."""

    for server in configuration.get("iceServers", []):
        if not isinstance(server, Mapping):
            continue
        for url in _normalize_urls(server.get("urls") or server.get("url")):
            if url.startswith(("turn:", "turns:")):
                return True
    return False


def ice_url_schemes(configuration: Mapping[str, object]) -> tuple[str, ...]:
    """Return the configured ICE URI schemes without exposing credentials."""

    schemes: set[str] = set()
    for server in configuration.get("iceServers", []):
        normalized = _normalize_ice_server(server)
        if normalized is None:
            continue
        for url in _normalize_urls(normalized["urls"]):
            scheme, separator, _rest = url.partition(":")
            if separator:
                schemes.add(scheme.casefold())
    return tuple(sorted(schemes))


def fetch_twilio_ice_servers(
    account_sid: str,
    auth_token: str,
    *,
    ttl_seconds: int = TWILIO_TOKEN_TTL_SECONDS,
    timeout_seconds: float = 10.0,
    opener: Callable[..., object] = urlopen,
) -> list[dict[str, object]]:
    """Request short-lived STUN/TURN credentials from Twilio server-side."""

    account_sid = account_sid.strip()
    auth_token = auth_token.strip()
    if not account_sid or not auth_token:
        raise ValueError("Twilio secrets require account_sid and auth_token.")

    endpoint = (
        "https://api.twilio.com/2010-04-01/Accounts/"
        f"{account_sid}/Tokens.json"
    )
    basic_token = base64.b64encode(
        f"{account_sid}:{auth_token}".encode("utf-8")
    ).decode("ascii")
    request = Request(
        endpoint,
        data=urlencode(
            {"Ttl": max(60, min(86400, int(ttl_seconds)))}
        ).encode("ascii"),
        headers={
            "Authorization": f"Basic {basic_token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "SafePath-AI/1.0",
        },
        method="POST",
    )

    try:
        with opener(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        raise RuntimeError(
            "Twilio could not issue temporary TURN credentials."
        ) from error

    raw_servers = payload.get("ice_servers")
    if not isinstance(raw_servers, list):
        raise RuntimeError("Twilio returned no ICE server list.")

    servers = [
        normalized
        for item in raw_servers
        if (normalized := _normalize_ice_server(item)) is not None
    ]
    if not has_turn_relay({"iceServers": servers}):
        raise RuntimeError("Twilio returned no TURN relay URLs.")
    return servers
