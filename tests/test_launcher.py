"""Check the public launcher from a different working directory."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def launch(self, *args):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkout with spaces"
            (root / ".venv/bin").mkdir(parents=True)
            launcher = root / "run_demo.sh"
            shutil.copy2(Path(__file__).resolve().parents[1] / "run_demo.sh", launcher)
            cli = root / ".venv/bin/egohand-live"
            cli.write_text(
                '#!/bin/bash\nprintf "%s\\0" "$PWD" "$@"\n'
            )
            cli.chmod(0o755)
            result = subprocess.run(
                [str(launcher), *args], cwd=directory, check=True, capture_output=True
            )
            values = result.stdout.decode().split("\0")[:-1]
            self.assertEqual(Path(values[0]).resolve(), root.resolve())
            return values[1:]

    def test_default_uses_checkout_resources_and_estimated_camera(self):
        args = self.launch("--camera", "2")
        self.assertIn("--estimate-intrinsics", args)
        self.assertEqual(args[args.index("--source") + 1], "assets/egopost")
        self.assertEqual(args[-2:], ["--camera", "2"])
        self.assertEqual(args.count("--output"), 1)

    def test_measured_calibration_does_not_conflict_with_estimated_default(self):
        args = self.launch("--calibration", "calibration/my camera.json")
        self.assertNotIn("--estimate-intrinsics", args)
        self.assertEqual(args[-1], "calibration/my camera.json")

    def test_equals_options_and_explicit_output(self):
        args = self.launch("--calibration=calibration/camera.json", "--output=runs/test")
        self.assertNotIn("--estimate-intrinsics", args)
        self.assertNotIn("--output", args)
        self.assertIn("--output=runs/test", args)

    def test_explicit_estimate_and_output_are_not_duplicated(self):
        args = self.launch("--estimate-intrinsics", "--output", "runs/test")
        self.assertEqual(args.count("--estimate-intrinsics"), 1)
        self.assertEqual(args.count("--output"), 1)


if __name__ == "__main__":
    unittest.main()
