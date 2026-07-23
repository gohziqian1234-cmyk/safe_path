import unittest

from rtc_config import (
    DEFAULT_STUN_URLS,
    build_rtc_configuration,
    has_turn_relay,
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

    def test_incomplete_turn_secrets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "urls, username, and credential"):
            build_rtc_configuration(
                {
                    "urls": ["turn:relay.example.com:443"],
                    "username": "missing-credential",
                }
            )


if __name__ == "__main__":
    unittest.main()
