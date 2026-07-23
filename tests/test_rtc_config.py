import unittest
import base64
import json

from rtc_config import (
    DEFAULT_STUN_URLS,
    build_rtc_configuration,
    fetch_twilio_ice_servers,
    has_turn_relay,
    ice_url_schemes,
)


class RtcConfigurationTests(unittest.TestCase):
    def test_stun_only_configuration_is_safe_without_secrets(self):
        configuration = build_rtc_configuration()

        self.assertEqual(
            configuration["iceServers"][0]["urls"],
            list(DEFAULT_STUN_URLS),
        )
        self.assertFalse(has_turn_relay(configuration))

    def test_turn_credentials_are_added_from_secrets(self):
        configuration = build_rtc_configuration(
            {
                "urls": [
                    "turn:relay.example.com:443?transport=udp",
                    "turns:relay.example.com:443?transport=tcp",
                ],
                "username": "temporary-user",
                "credential": "temporary-secret",
            }
        )

        turn_server = configuration["iceServers"][1]
        self.assertEqual(turn_server["username"], "temporary-user")
        self.assertEqual(turn_server["credential"], "temporary-secret")
        self.assertTrue(has_turn_relay(configuration))
        self.assertEqual(
            ice_url_schemes(configuration),
            ("stun", "turn", "turns"),
        )

    def test_incomplete_turn_secrets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "urls, username, and credential"):
            build_rtc_configuration(
                {
                    "urls": ["turn:relay.example.com:443"],
                    "username": "missing-credential",
                }
            )

    def test_twilio_ephemeral_servers_are_requested_server_side(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "ice_servers": [
                            {"urls": "stun:global.stun.twilio.com:3478"},
                            {
                                "urls": (
                                    "turn:global.turn.twilio.com:443"
                                    "?transport=tcp"
                                ),
                                "username": "ephemeral-user",
                                "credential": "ephemeral-password",
                            },
                        ]
                    }
                ).encode("utf-8")

        def fake_opener(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse()

        servers = fetch_twilio_ice_servers(
            "AC123",
            "auth-secret",
            opener=fake_opener,
        )

        expected_basic = base64.b64encode(
            b"AC123:auth-secret"
        ).decode("ascii")
        self.assertEqual(
            captured["request"].get_header("Authorization"),
            f"Basic {expected_basic}",
        )
        self.assertIn(b"Ttl=3600", captured["request"].data)
        configuration = build_rtc_configuration(
            additional_ice_servers=servers
        )
        self.assertTrue(has_turn_relay(configuration))


if __name__ == "__main__":
    unittest.main()
