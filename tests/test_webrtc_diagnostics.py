import unittest

from webrtc_diagnostics import inspect_webrtc_context


class FakeState:
    playing = True
    signalling = False


class FakePeerConnection:
    connectionState = "connected"
    iceConnectionState = "completed"
    iceGatheringState = "complete"
    signalingState = "stable"


class FakeWorker:
    pc = FakePeerConnection()


class FakeContext:
    state = FakeState()

    def _get_worker(self):
        return FakeWorker()


class WebRtcDiagnosticsTests(unittest.TestCase):
    def test_reads_frontend_and_server_peer_states(self):
        diagnostics = inspect_webrtc_context(FakeContext())

        self.assertTrue(diagnostics.playing)
        self.assertFalse(diagnostics.signalling)
        self.assertEqual(diagnostics.connection_state, "connected")
        self.assertEqual(diagnostics.ice_connection_state, "completed")
        self.assertEqual(diagnostics.ice_gathering_state, "complete")
        self.assertEqual(diagnostics.signaling_state, "stable")

    def test_missing_worker_is_reported_without_error(self):
        class ContextWithoutWorker:
            state = FakeState()

        diagnostics = inspect_webrtc_context(ContextWithoutWorker())

        self.assertEqual(diagnostics.connection_state, "not-created")
        self.assertEqual(diagnostics.ice_connection_state, "not-created")


if __name__ == "__main__":
    unittest.main()
