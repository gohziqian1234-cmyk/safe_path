import unittest
from io import BytesIO

import cv2
import numpy as np

from camera_feed import decode_camera_frame


class CameraFeedTests(unittest.TestCase):
    def test_decodes_png_as_bgr_and_returns_stable_digest(self):
        original = np.zeros((3, 4, 3), dtype=np.uint8)
        original[1, 2] = (11, 22, 33)
        ok, encoded = cv2.imencode(".png", original)
        self.assertTrue(ok)

        digest_one, decoded = decode_camera_frame(BytesIO(encoded.tobytes()))
        digest_two, _ = decode_camera_frame(BytesIO(encoded.tobytes()))

        self.assertEqual(decoded.shape, original.shape)
        self.assertTrue(np.array_equal(decoded, original))
        self.assertEqual(digest_one, digest_two)

    def test_rejects_empty_frame(self):
        with self.assertRaisesRegex(ValueError, "empty camera frame"):
            decode_camera_frame(BytesIO())

    def test_rejects_invalid_frame(self):
        with self.assertRaisesRegex(ValueError, "could not be decoded"):
            decode_camera_frame(BytesIO(b"not an image"))


if __name__ == "__main__":
    unittest.main()
