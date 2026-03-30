import os
import tempfile
import unittest

from visual_zendo import (
    DEFAULT_ARTIFACTS_DIR,
    PIL_AVAILABLE,
    LithicArrayEnv,
    Shape,
    Arrangement,
    WorldAxis,
)


class LithicArrayEnvTests(unittest.TestCase):
    def test_default_artifacts_dir_is_module_local(self):
        env = LithicArrayEnv()
        self.assertEqual(env.artifacts_dir, os.path.abspath(DEFAULT_ARTIFACTS_DIR))

    def test_high_alias_maps_to_hard_world(self):
        env = LithicArrayEnv(world="high")
        self.assertEqual(env.world, WorldAxis.HARD)

    @unittest.skipUnless(PIL_AVAILABLE, "Pillow is required for image rendering tests")
    def test_hard_world_returns_image_metadata(self):
        arrangement = Arrangement(
            shapes=[
                Shape(color="red", size="large", type_="triangle"),
                Shape(color="blue", size="small", type_="square"),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            env = LithicArrayEnv(world=WorldAxis.HARD, artifacts_dir=tmpdir)
            result = env._format_arrangement(arrangement, label=True, filename="sample.png")

            expected_path = os.path.join(tmpdir, "sample.png")
            self.assertEqual(result["representation_type"], "image")
            self.assertEqual(result["representation"], expected_path)
            self.assertEqual(result["image_path"], expected_path)
            self.assertEqual(result["label"], "Quartz")
            self.assertTrue(os.path.exists(expected_path))


if __name__ == "__main__":
    unittest.main()
