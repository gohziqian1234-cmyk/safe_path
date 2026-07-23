import unittest
from pathlib import Path


class CameraComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frontend_directory = (
            Path(__file__).parents[1]
            / "safepath_camera"
            / "frontend"
        )
        cls.source = (
            cls.frontend_directory / "main.js"
        ).read_text(encoding="utf-8")
        cls.styles = (
            cls.frontend_directory / "style.css"
        ).read_text(encoding="utf-8")

    def test_requests_exact_facing_mode_with_fallback(self):
        self.assertIn('facingMode: { exact: facingMode }', self.source)
        self.assertIn(
            'args.facingMode === "environment" ? "user" : "environment"',
            self.source,
        )

    def test_uses_compressed_jpeg_and_server_backpressure(self):
        self.assertIn('toDataURL("image/jpeg"', self.source)
        self.assertIn("waitingForServer", self.source)
        self.assertIn("event.detail.args", self.source)
        self.assertIn("capturedAtEpochMs = Date.now()", self.source)

    def test_keeps_a_visible_local_preview_and_draws_server_results(self):
        self.assertIn("function drawOverlay(overlay)", self.source)
        self.assertIn("overlayRevision", self.source)
        self.assertNotIn('sendStatus("starting")', self.source)
        self.assertIn("#preview-shell", self.styles)
        self.assertIn("#camera-status", self.styles)
        self.assertNotIn("height: 0", self.styles)


if __name__ == "__main__":
    unittest.main()
