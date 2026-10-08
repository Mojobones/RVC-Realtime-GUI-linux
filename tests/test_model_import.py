import os
import shutil
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from tools.model_import import ModelImportError, import_models, plan_import
from tools.model_registry import discover_models


class ModelImportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.downloads = self.root / "Downloads"
        self.downloads.mkdir()
        self.models = self.root / "models"

    def tearDown(self):
        self.temp.cleanup()

    def make(self, name, content=b"weights"):
        path = self.downloads / name
        path.write_bytes(content)
        return str(path)

    def test_single_model_takes_every_dropped_index(self):
        names = import_models(
            [self.make("Voice.pth"), self.make("added_IVF256_Voice.index")], self.models
        )

        self.assertEqual(names, ["Voice"])
        entries = discover_models(self.models)
        self.assertEqual([entry.name for entry in entries], ["Voice"])
        self.assertEqual(entries[0].index_path.name, "added_IVF256_Voice.index")
        self.assertEqual(entries[0].model_path.read_bytes(), b"weights")

    def test_several_models_get_the_index_that_names_them(self):
        plan = plan_import(
            [
                self.make("Alto.pth"),
                self.make("Bass.PTH"),
                self.make("added_Bass_v2.index"),
            ]
        )

        self.assertEqual(
            [(model.name, [index.name for index in indexes]) for model, indexes in plan],
            [("Alto.pth", []), ("Bass.PTH", ["added_Bass_v2.index"])],
        )

    def test_imported_folder_uses_normal_permissions(self):
        previous = os.umask(0o022)
        try:
            import_models([self.make("Voice.pth")], self.models)
        finally:
            os.umask(previous)

        self.assertEqual((self.models / "Voice").stat().st_mode & 0o777, 0o755)

    def test_existing_folder_name_gets_a_suffix(self):
        (self.models / "Voice").mkdir(parents=True)

        names = import_models([self.make("Voice.pth")], self.models)

        self.assertEqual(names, ["Voice (2)"])
        self.assertTrue((self.models / "Voice (2)" / "Voice.pth").is_file())

    def test_requires_a_pth_file(self):
        with self.assertRaises(ModelImportError) as raised:
            import_models([self.make("added_Voice.index")], self.models)

        self.assertEqual(raised.exception.code, "import_no_model_file")
        self.assertFalse(self.models.exists())

    def test_missing_file_is_reported(self):
        with self.assertRaises(ModelImportError) as raised:
            plan_import([str(self.downloads / "gone.pth")])

        self.assertEqual(raised.exception.code, "import_failed")

    def test_failed_copy_leaves_no_partial_model(self):
        files = [self.make("Voice.pth"), self.make("added_Voice.index")]
        real_copy = shutil.copy2

        def copy_then_fail(source, destination):
            if str(source).endswith(".index"):
                raise OSError("No space left on device")
            return real_copy(source, destination)

        with mock.patch("tools.model_import.shutil.copy2", side_effect=copy_then_fail):
            with self.assertRaises(ModelImportError) as raised:
                import_models(files, self.models)

        self.assertIn("No space left", raised.exception.message)
        self.assertEqual(list(self.models.iterdir()), [])


class HiddenFolderTest(unittest.TestCase):
    def test_hidden_folders_are_not_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            staging = Path(temp_dir) / ".import-abc123"
            staging.mkdir()
            (staging / "Voice.pth").touch()

            self.assertEqual(discover_models(temp_dir), [])


if __name__ == "__main__":
    unittest.main()
