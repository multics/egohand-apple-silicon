"""A one-slot camera reader. An unread frame is replaced, never queued."""

from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class Frame:
    index: int
    monotonic_ns: int
    unix_ns: int
    image: object


class LatestCapture:
    def __init__(self, capture):
        self.capture = capture
        self.condition = threading.Condition()
        self.frame = None
        self.dropped = 0
        self.count = 0
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._read, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _read(self):
        try:
            while not self.closed:
                ok, image = self.capture.read()
                if not ok:
                    raise RuntimeError("Camera frame read failed")
                frame = Frame(self.count, time.monotonic_ns(), time.time_ns(), image)
                with self.condition:
                    if self.frame is not None:
                        self.dropped += 1
                    self.frame = frame
                    self.count += 1
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def next(self, timeout=10):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.frame is not None or self.error or self.closed, timeout
            )
            if not ready:
                raise TimeoutError("No camera frame within the timeout")
            if self.error:
                raise self.error
            frame, self.frame = self.frame, None
            return frame

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout=2)
        self.capture.release()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("Camera reader did not stop")
