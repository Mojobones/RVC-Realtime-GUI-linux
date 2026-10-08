import os
import tempfile
import unittest

from engine.assets import RMVPE_PATH, missing_assets


class MissingAssetsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.assets = self.temp.name

    def tearDown(self):
        self.temp.cleanup()

    def touch(self, *parts):
        path = os.path.join(self.assets, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").close()

    def test_empty_assets_folder_lists_everything(self):
        self.assertEqual(
            missing_assets(assets_dir=self.assets),
            [
                "assets/hubert_base/config.json",
                "assets/hubert_base/pytorch_model.bin",
                "assets/rmvpe/rmvpe.pt",
            ],
        )

    def test_rmvpe_is_only_required_for_rmvpe(self):
        self.touch("hubert_base", "config.json")
        self.touch("hubert_base", "pytorch_model.bin")

        self.assertEqual(missing_assets("fcpe", self.assets), [])
        self.assertEqual(missing_assets("pm", self.assets), [])
        self.assertEqual(missing_assets("rmvpe", self.assets), ["assets/rmvpe/rmvpe.pt"])

    def test_safetensors_weights_are_accepted(self):
        self.touch("hubert_base", "config.json")
        self.touch("hubert_base", "model.safetensors")
        self.touch("rmvpe", "rmvpe.pt")

        self.assertEqual(missing_assets(assets_dir=self.assets), [])

    def test_rmvpe_path_does_not_depend_on_the_working_directory(self):
        self.assertTrue(os.path.isabs(RMVPE_PATH))
        self.assertTrue(RMVPE_PATH.endswith(os.path.join("assets", "rmvpe", "rmvpe.pt")))


if __name__ == "__main__":
    unittest.main()
