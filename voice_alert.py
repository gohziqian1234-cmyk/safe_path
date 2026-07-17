"""Non-blocking, cooldown-controlled voice warnings."""

from __future__ import annotations

import threading
import time


class VoiceAlert:
    """Speak warnings without freezing the camera processing loop."""

    def __init__(self, cooldown_seconds: float = 8.0, enabled: bool = True) -> None:
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.enabled = enabled
        self._last_alert_at = float("-inf")
        self._speaking = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def trigger(self, message: str) -> bool:
        """Start one warning and return whether it was accepted."""

        if not self.enabled or not message:
            return False

        now = time.monotonic()
        with self._lock:
            if self._speaking or now - self._last_alert_at < self.cooldown_seconds:
                return False
            self._speaking = True
            self._last_alert_at = now

        self._thread = threading.Thread(
            target=self._speak,
            args=(message,),
            daemon=True,
            name="safepath-voice-alert",
        )
        self._thread.start()
        return True

    def _speak(self, message: str) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.setProperty("volume", 1.0)
            engine.say(message)
            engine.runAndWait()
            engine.stop()
        except Exception as error:
            print(f"SafePath voice warning failed: {error}")
            try:
                import winsound

                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                print("SafePath used the Windows warning-beep fallback.")
            except Exception as fallback_error:
                print(f"SafePath warning-beep fallback failed: {fallback_error}")
        finally:
            with self._lock:
                self._speaking = False

    def close(self) -> None:
        """Wait briefly for an active warning before shutdown."""

        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)


def speak_warning(
    message: str = "Warning. An obstacle is blocking the walking path.",
) -> None:
    alert = VoiceAlert(cooldown_seconds=0)
    if alert.trigger(message):
        alert.close()


if __name__ == "__main__":
    speak_warning()
