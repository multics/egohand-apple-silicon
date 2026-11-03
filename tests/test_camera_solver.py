"""Check calibrated translation recovery across the explicit CPU boundary."""

from pathlib import Path
import sys
import unittest

import torch
from egohand_live.backend import solve_camera_on_cpu

SOURCE = Path(__file__).resolve().parents[1] / "assets/egopost"


@unittest.skipUnless(SOURCE.is_dir(), "Fetch the pinned EgoPost source first")
class CameraSolverTests(unittest.TestCase):
    def test_known_translation_and_missing_points(self):
        sys.path.insert(0, str(SOURCE))
        devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                joints = (
                    torch.linspace(-0.05, 0.05, 75, device=device)
                    .reshape(1, 25, 3)
                    .repeat(2, 1, 1)
                )
                translation = torch.tensor([0.04, -0.02, 0.8], device=device)
                camera = torch.tensor(
                    [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]],
                    device=device,
                ).expand(2, 3, 3)
                points = joints + translation
                pixels = points[..., :2] / points[..., 2:] * 1000 + torch.tensor(
                    [320.0, 240.0], device=device
                )
                mask = torch.ones(2, 25, device=device, dtype=torch.bool)
                mask[1] = False
                result = solve_camera_on_cpu(
                    joints,
                    pixels,
                    translation[2].expand(2),
                    camera,
                    point_mask=mask,
                    damping=0,
                )
                self.assertEqual(result.translation_xy.device.type, device)
                self.assertTrue(result.valid[0].item())
                self.assertFalse(result.valid[1].item())
                self.assertEqual(result.num_points.cpu().tolist(), [25, 0])
                torch.testing.assert_close(
                    result.translation_xy[0], translation[:2], atol=1e-6, rtol=1e-5
                )
                self.assertTrue(torch.isfinite(result.translation_xy).all().item())


if __name__ == "__main__":
    unittest.main()
