import base64
import hashlib
import hmac
import unittest

from rtc_config import (
    PUBLIC_TURN_HOST,
    PUBLIC_TURN_SECRET,
    build_rtc_configuration,
    create_temporary_turn_credentials,
)


class RtcConfigurationTests(unittest.TestCase):
    def test_temporary_credentials_follow_coturn_rest_format(self):
        username, credential = create_temporary_turn_credentials(
            now=1_000,
            lifetime_seconds=3_600,
            user_id="test-user",
        )

        expected_username = "4600:test-user"
        expected_digest = hmac.new(
            PUBLIC_TURN_SECRET.encode("utf-8"),
            expected_username.encode("utf-8"),
            hashlib.sha1,
        ).digest()

        self.assertEqual(username, expected_username)
        self.assertEqual(credential, base64.b64encode(expected_digest).decode("ascii"))

    def test_cloud_configuration_contains_stun_and_turn_routes(self):
        configuration = build_rtc_configuration(now=1_000)
        ice_servers = configuration["iceServers"]

        self.assertIn("stun:stun.l.google.com:19302", ice_servers[0]["urls"])
        self.assertTrue(
            any(
                url.startswith(f"turn:{PUBLIC_TURN_HOST}")
                for url in ice_servers[1]["urls"]
            )
        )
        self.assertTrue(
            any(
                url.startswith(f"turns:{PUBLIC_TURN_HOST}")
                for url in ice_servers[1]["urls"]
            )
        )
        self.assertEqual(ice_servers[1]["username"], "87400:safepath")
        self.assertTrue(ice_servers[1]["credential"])


if __name__ == "__main__":
    unittest.main()
