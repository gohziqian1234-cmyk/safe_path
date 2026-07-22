"""WebRTC network configuration for local and cloud SafePath sessions."""

from __future__ import annotations

import base64
import hashlib
import hmac
import time


PUBLIC_TURN_HOST = "staticauth.openrelay.metered.ca"
PUBLIC_TURN_SECRET = "openrelayprojectsecret"
TURN_CREDENTIAL_LIFETIME_SECONDS = 24 * 60 * 60


def create_temporary_turn_credentials(
    *,
    now: float | None = None,
    lifetime_seconds: int = TURN_CREDENTIAL_LIFETIME_SECONDS,
    user_id: str = "safepath",
) -> tuple[str, str]:
    """Create coturn REST credentials for Metered's public static-auth relay."""

    issued_at = time.time() if now is None else float(now)
    expires_at = int(issued_at + max(60, int(lifetime_seconds)))
    username = f"{expires_at}:{user_id}"
    digest = hmac.new(
        PUBLIC_TURN_SECRET.encode("utf-8"),
        username.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    credential = base64.b64encode(digest).decode("ascii")
    return username, credential


def build_rtc_configuration(*, now: float | None = None) -> dict[str, object]:
    """Return STUN plus TURN routes suitable for Streamlit Community Cloud."""

    username, credential = create_temporary_turn_credentials(now=now)
    return {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": [
                    f"turn:{PUBLIC_TURN_HOST}:80",
                    f"turn:{PUBLIC_TURN_HOST}:443",
                    f"turn:{PUBLIC_TURN_HOST}:443?transport=tcp",
                    f"turns:{PUBLIC_TURN_HOST}:443?transport=tcp",
                ],
                "username": username,
                "credential": credential,
            },
        ],
        "iceCandidatePoolSize": 10,
    }
