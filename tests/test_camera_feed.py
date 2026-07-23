import unittest
from io import BytesIO

import cv2
import numpy as np

from camera_feed import decode_camera_frame, resize_for_inference


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

    def test_decodes_compressed_jpeg(self):
        original = np.full((20, 30, 3), 120, dtype=np.uint8)
        ok, encoded = cv2.imencode(
            ".jpg",
            original,
            [cv2.IMWRITE_JPEG_QUALITY, 68],
        )
        self.assertTrue(ok)

        _digest, decoded = decode_camera_frame(BytesIO(encoded.tobytes()))

        self.assertEqual(decoded.shape, original.shape)

    def test_downscales_longest_side_and_returns_coordinate_scales(self):
        original = np.zeros((480, 640, 3), dtype=np.uint8)

        resized, x_scale, y_scale = resize_for_inference(original, 416)

        self.assertEqual(resized.shape, (312, 416, 3))
        self.assertAlmostEqual(x_scale, 640 / 416)
        self.assertAlmostEqual(y_scale, 480 / 312)

    def test_small_frame_is_not_reallocated(self):
        original = np.zeros((240, 320, 3), dtype=np.uint8)

        resized, x_scale, y_scale = resize_for_inference(original, 416)

        self.assertIs(resized, original)
        self.assertEqual((x_scale, y_scale), (1.0, 1.0))

    def test_rejects_empty_frame(self):
        with self.assertRaisesRegex(ValueError, "empty camera frame"):
            decode_camera_frame(BytesIO())

    def test_rejects_invalid_frame(self):
        with self.assertRaisesRegex(ValueError, "could not be decoded"):
            decode_camera_frame(BytesIO(b"not an image"))


if __name__ == "__main__":
    unittest.main()
