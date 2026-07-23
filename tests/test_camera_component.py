import unittest
from pathlib import Path


class CameraComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frontend = (
            Path(__file__).parents[1]
            / "safepath_camera"
            / "frontend"
            / "main.js"
        )
        cls.source = frontend.read_text(encoding="utf-8")

    def test_requests_exact_facing_mode_with_fallback(self):
        self.assertIn('facingMode: { exact: facingMode }', self.source)
        self.assertIn(
            'args.facingMode === "environment" ? "user" : "environment"',
            self.source,
        )

    def test_uses_compressed_jpeg_and_server_backpressure(self):
        self.assertIn('toDataURL("image/jpeg"', self.source)
        self.assertIn("waitingForServer", self.source)


if __name__ == "__main__":
    unittest.main()
