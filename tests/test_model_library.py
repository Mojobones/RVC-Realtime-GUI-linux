import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from engine import core
from engine.settings import EngineSettings, load_model_settings, save_model_settings
from tools import model_registry
from tools.model_registry import (
    TrashError,
    discover_models,
    model_name_problem,
    move_to_trash,
    rename_model_folder,
)


class MoveToTrashTest(unittest.TestCase):
    def test_uses_gio_trash(self):
        done = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(model_registry.shutil, "which", return_value="/usr/bin/gio"), \
             mock.patch.object(model_registry.subprocess, "run", return_value=done) as run:
            move_to_trash(Path("/models/Voice"))
        self.assertEqual(run.call_args.args[0], ["/usr/bin/gio", "trash", "--", "/models/Voice"])

    def test_reports_failures_and_a_missing_gio(self):
        failed = subprocess.CompletedProcess([], 1, "", "Unable to find or create trash directory")
        with mock.patch.object(model_registry.shutil, "which", return_value="/usr/bin/gio"), \
             mock.patch.object(model_registry.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(TrashError, "trash directory"):
                move_to_trash(Path("/models/Voice"))
        with mock.patch.object(model_registry.shutil, "which", return_value=None):
            with self.assertRaisesRegex(TrashError, "gio"):
                move_to_trash(Path("/models/Voice"))


class ModelFolders(unittest.TestCase):
    """models/ with Alto, Bass and Tenor, and an engine without devices."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ("Alto", "Bass", "Tenor"):
            (self.root / name).mkdir()
            (self.root / name / f"{name}.pth").write_bytes(b"x" * 10)
        save_model_settings(str(self.root / "Alto"), EngineSettings(pitch=4.0))
        patcher = mock.patch.object(core, "MODELS_ROOT", str(self.root))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.temp.cleanup)
        save = mock.patch.object(core, "save_settings")
        self.saved = save.start()
        self.addCleanup(save.stop)

    def engine(self, selected="Bass"):
        engine = core.RealtimeEngine.__new__(core.RealtimeEngine)
        engine.settings = EngineSettings(model_name=selected)
        engine.devices = SimpleNamespace(
            resolve=lambda value, kind: value, resolve_monitor=lambda value: value
        )
        engine.gpu_options = [{"id": "auto", "label": None}]
        engine.pending_model_settings_name = None
        engine.model_settings_save_due = 0.0
        engine.running = False
        engine.function = "vc"
        engine.events = []
        engine.emit = lambda event, data: engine.events.append((event, data))
        engine.state = lambda: {}
        engine.refresh_models()
        return engine

    def trash(self, path):
        # Stand-in for gio: the folder leaves models/ (hidden folders are not models).
        os.rename(path, self.root / f".trashed-{Path(path).name}")


class DeleteModelTest(ModelFolders):
    def test_deleting_another_model_keeps_the_selection(self):
        engine = self.engine(selected="Bass")
        with mock.patch.object(core, "move_to_trash", side_effect=self.trash) as trash:
            engine.delete_model("Tenor")
        self.assertEqual(trash.call_args.args[0], (self.root / "Tenor").resolve())
        self.assertEqual([model.name for model in engine.models], ["Alto", "Bass"])
        self.assertEqual(engine.settings.model_name, "Bass")
        self.assertIn(("status", {"code": "model_deleted", "name": "Tenor"}), engine.events)

    def test_deleting_the_selected_model_selects_another_with_its_settings(self):
        engine = self.engine(selected="Bass")
        with mock.patch.object(core, "move_to_trash", side_effect=self.trash):
            engine.delete_model("Bass")
        self.assertEqual(engine.settings.model_name, "Alto")
        self.assertEqual(engine.settings.pitch, 4.0)
        self.saved.assert_called()

    def test_pending_settings_are_saved_first_and_not_into_a_dead_folder(self):
        engine = self.engine(selected="Bass")
        engine.settings.pitch = 7.0
        engine.schedule_model_settings_save()
        order = []
        real_save = core.save_model_settings

        def record_save(directory, settings):
            order.append(("save", Path(directory).name))
            real_save(directory, settings)

        def record_trash(path):
            order.append(("trash", Path(path).name))
            self.trash(path)

        with mock.patch.object(core, "save_model_settings", side_effect=record_save), \
             mock.patch.object(core, "move_to_trash", side_effect=record_trash):
            engine.delete_model("Bass")
        self.assertEqual(order, [("save", "Bass"), ("trash", "Bass")])
        self.assertFalse((self.root / "Bass").exists())

    def test_the_converting_model_cannot_be_deleted(self):
        engine = self.engine(selected="Bass")
        engine.running = True
        with mock.patch.object(core, "move_to_trash") as trash:
            with self.assertRaises(core.EngineError) as raised:
                engine.delete_model("Bass")
        self.assertEqual(raised.exception.code, "model_in_use")
        trash.assert_not_called()

    def test_failures_and_unknown_models_are_reported(self):
        engine = self.engine()
        with self.assertRaises(core.EngineError) as raised:
            engine.delete_model("Nobody")
        self.assertEqual(raised.exception.code, "no_model")
        with mock.patch.object(core, "move_to_trash", side_effect=TrashError("no trash")):
            with self.assertRaises(core.EngineError) as raised:
                engine.delete_model("Alto")
        self.assertEqual(raised.exception.code, "delete_failed")
        self.assertTrue((self.root / "Alto").exists())


class RenameModelTest(ModelFolders):

    def test_renaming_keeps_settings_and_the_selection(self):
        engine = self.engine(selected="Alto")
        engine.rename_model("Alto", "Alto-final_200e_14200s")
        self.assertEqual([m.name for m in engine.models], ["Alto-final_200e_14200s", "Bass", "Tenor"])
        self.assertEqual(engine.settings.model_name, "Alto-final_200e_14200s")
        self.assertEqual(engine.settings.pitch, 4.0)
        self.assertTrue((self.root / "Alto-final_200e_14200s" / "Alto.pth").is_file())
        self.assertFalse((self.root / "Alto").exists())
        self.saved.assert_called()
        self.assertIn(
            ("status", {"code": "model_renamed", "name": "Alto", "new_name": "Alto-final_200e_14200s"}),
            engine.events,
        )

    def test_pending_settings_land_in_the_renamed_folder(self):
        engine = self.engine(selected="Bass")
        engine.settings.pitch = 9.5
        engine.schedule_model_settings_save()
        engine.rename_model("Bass", "Baritone")
        self.assertFalse((self.root / "Bass").exists())
        self.assertEqual(load_model_settings(str(self.root / "Baritone"))["pitch"], 9.5)

    def test_case_only_rename(self):
        engine = self.engine()
        engine.rename_model("Tenor", "TENOR")
        self.assertEqual([m.name for m in engine.models], ["Alto", "Bass", "TENOR"])

    def test_rejected_names(self):
        engine = self.engine()
        cases = {
            "Bass": "model_name_taken",
            "bass": "model_name_taken",
            "": "bad_model_name",
            " Alto2": "bad_model_name",
            ".hidden": "bad_model_name",
            "a/b": "bad_model_name",
            "what?": "bad_model_name",
        }
        for new_name, code in cases.items():
            with self.subTest(new_name=new_name):
                with self.assertRaises(core.EngineError) as raised:
                    engine.rename_model("Alto", new_name)
                self.assertEqual(raised.exception.code, code)
        self.assertTrue((self.root / "Alto").is_dir())

    def test_the_converting_model_cannot_be_renamed(self):
        engine = self.engine(selected="Bass")
        engine.running = True
        with self.assertRaises(core.EngineError) as raised:
            engine.rename_model("Bass", "Baritone")
        self.assertEqual(raised.exception.code, "model_in_use")
        engine.rename_model("Tenor", "Countertenor")  # other models are fine

    def test_name_rules(self):
        self.assertIsNone(model_name_problem("Sakura-shortstream-50m-1.6_200e_14200s"))
        self.assertIsNone(model_name_problem("さくら 声"))
        self.assertEqual(model_name_problem("trailing."), "edges")
        self.assertEqual(model_name_problem("x" * 201), "too_long")
        self.assertEqual(model_name_problem("tab\there"), "characters")

    def test_folder_rename_refuses_to_overwrite(self):
        with self.assertRaises(FileExistsError):
            rename_model_folder(self.root / "Alto", "Bass")
        self.assertTrue((self.root / "Alto").is_dir())


class RegistryMetadataTest(unittest.TestCase):
    def test_size_and_date_of_the_model_file(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "Voice").mkdir()
            model = Path(temp) / "Voice" / "Voice.pth"
            model.write_bytes(b"x" * 1234)
            os.utime(model, (1_700_000_000, 1_700_000_000))
            (entry,) = discover_models(temp)
        self.assertEqual((entry.size_bytes, entry.modified), (1234, 1_700_000_000))


if __name__ == "__main__":
    unittest.main()
