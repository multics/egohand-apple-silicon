"""Check that slow inference cannot build a camera backlog."""

import threading
import unittest
from egohand_live.capture import LatestCapture


class ControlledCamera:
    def __init__(self):
        self.step = threading.Semaphore(0)
        self.index = 0
        self.closed = False

    def read(self):
        self.step.acquire()
        if self.closed:
            return False, None
        self.index += 1
        return True, self.index

    def release(self):
        self.closed = True
        self.step.release()


class CaptureTests(unittest.TestCase):
    def test_only_latest_frame_is_kept(self):
        camera = ControlledCamera()
        reader = LatestCapture(camera).start()
        try:
            camera.step.release(3)
            with reader.condition:
                self.assertTrue(reader.condition.wait_for(lambda: reader.count == 3, 2))
            frame = reader.next()
            self.assertEqual(frame.index, 2)
            self.assertEqual(frame.image, 3)
            self.assertEqual(reader.dropped, 2)
            self.assertGreater(frame.unix_ns, 0)
        finally:
            reader.close()
        self.assertFalse(reader.thread.is_alive())

    def test_read_failure_is_reported(self):
        camera = ControlledCamera()
        reader = LatestCapture(camera).start()
        camera.release()
        try:
            with self.assertRaisesRegex(RuntimeError, "read failed"):
                reader.next()
        finally:
            reader.close()

    def test_stalled_camera_has_timeout(self):
        camera = ControlledCamera()
        reader = LatestCapture(camera).start()
        try:
            with self.assertRaises(TimeoutError):
                reader.next(timeout=0.01)
        finally:
            reader.close()


if __name__ == "__main__":
    unittest.main()
