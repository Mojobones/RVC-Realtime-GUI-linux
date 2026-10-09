import os
import shutil
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from engine import core
from engine.settings import (
    RESETTABLE_MODEL_SETTING_KEYS,
    EngineSettings,
    load_model_settings,
    save_model_settings,
)
from tools import pitch_match
from tools.pitch_match import PitchHistogram, recommended_shift


def seconds_of(hz, seconds):
    return np.full(int(seconds * pitch_match.FRAMES_PER_SECOND), float(hz))


class RecommendedShiftTest(unittest.TestCase):
    def test_moves_the_voice_median_onto_the_source_median(self):
        # The user's measured voice against the Sakura and Fuufie datasets.
        self.assertEqual(recommended_shift(114.0, 212.0), 10.7)
        self.assertEqual(recommended_shift(114.0, 220.0), 11.4)
        self.assertEqual(recommended_shift(220.0, 110.0), -12.0)

    def test_unknown_sides_give_no_recommendation(self):
        self.assertIsNone(recommended_shift(None, 212.0))
        self.assertIsNone(recommended_shift(114.0, 0.0))


class PitchHistogramTest(unittest.TestCase):
    def test_median_is_accurate_to_a_few_cents(self):
        histogram = PitchHistogram()
        histogram.add(seconds_of(212.0, 30))
        self.assertAlmostEqual(12 * np.log2(histogram.median_hz() / 212.0), 0.0, delta=0.06)

    def test_median_of_a_spread_of_pitches(self):
        rng = np.random.default_rng(0)
        f0 = 150.0 * 2 ** (rng.normal(0, 3, 6000) / 12)
        histogram = PitchHistogram()
        histogram.add(f0)
        self.assertAlmostEqual(
            12 * np.log2(histogram.median_hz() / np.median(f0)), 0.0, delta=0.06
        )

    def test_unvoiced_and_out_of_range_frames_are_ignored(self):
        histogram = PitchHistogram()
        histogram.add([0.0, 0.0, np.nan, 5.0, 5000.0])
        self.assertEqual(histogram.voiced_seconds, 0.0)
        self.assertIsNone(histogram.median_hz())
        self.assertFalse(histogram.changed)

    def test_summary_waits_for_enough_speech(self):
        histogram = PitchHistogram()
        histogram.add(seconds_of(120.0, pitch_match.MIN_VOICED_SECONDS / 2))
        self.assertIsNone(histogram.summary()["median_hz"])
        histogram.add(seconds_of(120.0, pitch_match.MIN_VOICED_SECONDS))
        self.assertAlmostEqual(histogram.summary()["median_hz"], 120.0, delta=0.5)

    def test_recent_speech_outweighs_old_speech(self):
        histogram = PitchHistogram(memory_seconds=60)
        histogram.add(seconds_of(100.0, 60))
        for _ in range(60):
            histogram.add(seconds_of(200.0, 1))
        self.assertAlmostEqual(histogram.voiced_seconds, 60.0, places=6)
        self.assertGreater(histogram.median_hz(), 190.0)

    def test_save_and_load_round_trip(self):
        histogram = PitchHistogram()
        histogram.add(seconds_of(130.0, 25))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "voice_pitch.json")
            histogram.save(path)
            self.assertFalse(histogram.changed)
            loaded = PitchHistogram.load(path)
            self.assertAlmostEqual(loaded.median_hz(), histogram.median_hz(), places=3)
            self.assertAlmostEqual(loaded.voiced_seconds, 25.0, places=2)
            with open(path, "w", encoding="utf-8") as broken:
                broken.write("{nope")
            self.assertEqual(PitchHistogram.load(path).voiced_seconds, 0.0)
        self.assertEqual(PitchHistogram.load("/nonexistent/voice.json").voiced_seconds, 0.0)


def tone_detector(calls):
    """A stand-in for RMVPE that tells the two test tones apart."""

    def detect(audio):
        calls.append(audio.shape[0])
        frames = audio[: audio.shape[0] // 160 * 160].reshape(-1, 160)
        loud = np.sqrt((frames**2).mean(axis=1)) > 0.1
        # Which of the two test tones each 10 ms frame projects onto most.
        n = np.arange(160) / pitch_match.ANALYSIS_RATE
        power = [np.abs(frames @ np.exp(-2j * np.pi * hz * n)) for hz in (200, 250)]
        return np.where(loud, np.where(power[1] > power[0], 250.0, 200.0), 0.0)

    return detect


def write_tone(path, hz, seconds, silence=0.0, rate=44100):
    t = np.arange(int(rate * (seconds + silence))) / rate
    audio = 0.5 * np.sin(2 * np.pi * hz * t)
    audio[t >= seconds] = 0.0
    sf.write(path, audio.astype(np.float32), rate)


class SourceFilesTest(unittest.TestCase):
    def test_folders_contribute_the_audio_files_directly_inside(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("b.FLAC", "a.wav", "notes.txt"):
                open(os.path.join(directory, name), "wb").close()
            os.makedirs(os.path.join(directory, "validation"))
            open(os.path.join(directory, "validation", "other_voice.wav"), "wb").close()
            single = os.path.join(directory, "notes.txt")

            self.assertEqual(
                pitch_match.source_files([directory, single]),
                [os.path.join(directory, "a.wav"), os.path.join(directory, "b.FLAC"), single],
            )
            with self.assertRaises(FileNotFoundError):
                pitch_match.source_files([os.path.join(directory, "gone.wav")])


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class MeasureSourcePitchTest(unittest.TestCase):
    def test_every_frame_of_every_file_counts(self):
        calls, pauses = [], []
        with tempfile.TemporaryDirectory() as directory:
            # 4 s at 200 Hz (+2 s silence) and 14 s at 250 Hz: median 250.
            write_tone(os.path.join(directory, "one.wav"), 200, 4, silence=2)
            write_tone(os.path.join(directory, "two.flac"), 250, 14)
            median, seconds = pitch_match.measure_source_pitch(
                shutil.which("ffmpeg"), [directory], tone_detector(calls),
                pause=lambda: pauses.append(1),
            )
        self.assertEqual(median, 250.0)
        self.assertAlmostEqual(seconds, 18.0, delta=0.3)
        piece = int(pitch_match.ANALYSIS_PIECE_SECONDS * pitch_match.ANALYSIS_RATE)
        # One piece for the 6 s file, two for the 14 s one, all the same shape.
        self.assertEqual(calls, [piece] * 3)
        self.assertEqual(len(pauses), 3)

    def test_undecodable_file_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "broken.wav")
            with open(path, "wb") as broken:
                broken.write(b"not audio at all")
            with self.assertRaises(ValueError):
                pitch_match.measure_source_pitch(shutil.which("ffmpeg"), [path], tone_detector([]))

    def test_silence_has_no_median(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "quiet.wav")
            write_tone(path, 200, 0, silence=3)
            result = pitch_match.measure_source_pitch(shutil.which("ffmpeg"), [path], tone_detector([]))
        self.assertEqual(result, (None, 0.0))


class FinishAnalysisTest(unittest.TestCase):
    """Applying a finished background analysis on the command thread."""

    def engine(self, directory):
        from types import SimpleNamespace
        from pathlib import Path

        engine = core.RealtimeEngine.__new__(core.RealtimeEngine)
        engine.settings = EngineSettings(model_name="A", pitch=11.0)
        engine.models_by_name = {
            name: SimpleNamespace(name=name, directory=Path(directory) / name) for name in "AB"
        }
        for name in "AB":
            os.makedirs(os.path.join(directory, name))
        engine.devices = SimpleNamespace(inputs={}, outputs={})
        engine.pending_model_settings_name = None
        engine.model_settings_save_due = 0.0
        engine.running = False
        engine.events = []
        engine.emit = lambda event, data: engine.events.append((event, data))
        engine.state = lambda: {}
        return engine

    def job(self, model, result=None, error=None):
        from types import SimpleNamespace

        done = SimpleNamespace(is_alive=lambda: False)
        return {"model": model, "files": [], "result": result, "error": error, "thread": done}

    def test_result_is_stored_with_the_selected_model(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            engine.pitch_analysis = self.job("A", result=(211.87, 1200.0))
            engine.finish_pitch_analysis()
            self.assertEqual(engine.settings.source_pitch_hz, 211.87)
            saved = load_model_settings(os.path.join(directory, "A"))
            self.assertEqual(saved["source_pitch_hz"], 211.87)
            self.assertEqual(saved["pitch"], 11.0)
        self.assertIsNone(engine.pitch_analysis)
        self.assertIn(("status", {"code": "pitch_analyzed", "name": "A", "median_hz": 211.87, "seconds": 1200.0}), engine.events)

    def test_result_follows_the_analysed_model_after_a_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            save_model_settings(os.path.join(directory, "B"), EngineSettings(pitch=5.0))
            engine.pitch_analysis = self.job("B", result=(150.0, 60.0))
            engine.finish_pitch_analysis()
            self.assertEqual(engine.settings.source_pitch_hz, 0.0)
            saved = load_model_settings(os.path.join(directory, "B"))
            self.assertEqual((saved["source_pitch_hz"], saved["pitch"]), (150.0, 5.0))

    def test_failures_become_error_events(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            engine.pitch_analysis = self.job("A", result=(None, 0.0))
            engine.finish_pitch_analysis()
            engine.pitch_analysis = self.job("A", error=ValueError("bad file"))
            engine.finish_pitch_analysis()
        codes = [data["code"] for event, data in engine.events if event == "error"]
        self.assertEqual(codes, ["no_voice_found", "pitch_analysis_failed"])

    def test_running_job_is_left_alone(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            job = self.job("A", result=(200.0, 30.0))
            job["thread"] = SimpleNamespace(is_alive=lambda: True)
            engine.pitch_analysis = job
            engine.finish_pitch_analysis()
        self.assertIs(engine.pitch_analysis, job)


class ModelSettingTest(unittest.TestCase):
    def test_source_pitch_is_stored_with_the_model_and_survives_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            save_model_settings(directory, EngineSettings(pitch=11.0, source_pitch_hz=212.0))
            self.assertEqual(load_model_settings(directory)["source_pitch_hz"], 212.0)
            self.assertEqual(load_model_settings(None)["source_pitch_hz"], 0.0)
        self.assertNotIn("source_pitch_hz", RESETTABLE_MODEL_SETTING_KEYS)
        self.assertIn("pitch", RESETTABLE_MODEL_SETTING_KEYS)


class LiveVoicePitchTest(unittest.TestCase):
    def engine(self):
        from tests.test_engine_pipeline import passthrough_engine

        engine = passthrough_engine(block_time=0.1)
        engine.function = "vc"
        engine.settings.hold_context = False
        engine.settings.rms_mix_rate = 1.0
        engine.voice_pitch = PitchHistogram()
        engine.file_audio_source = None
        engine.settings.input_source = "microphone"
        shift = engine.block_frame_16k // 160
        engine.rvc.input_f0 = np.full(shift, 115.0)
        engine.rvc.infer = lambda *args: torch.zeros(
            engine.splicer.input_length + engine.block_frame
        )
        return engine, shift

    def test_microphone_pitch_frames_are_counted(self):
        engine, shift = self.engine()
        block = np.zeros((engine.block_frame, 1), dtype=np.float32)
        for _ in range(3):
            engine.audio_callback(block.copy(), engine.block_frame, None, None)
        self.assertAlmostEqual(engine.voice_pitch.voiced_seconds, 3 * shift / 100)
        self.assertAlmostEqual(engine.voice_pitch.median_hz(), 115.0, delta=0.5)

    def test_file_input_is_not_the_users_voice(self):
        engine, _ = self.engine()
        engine.settings.input_source = "file"
        block = np.zeros((engine.block_frame, 1), dtype=np.float32)
        engine.audio_callback(block, engine.block_frame, None, None)
        self.assertEqual(engine.voice_pitch.voiced_seconds, 0.0)


class InputF0CaptureTest(unittest.TestCase):
    def test_newest_frames_are_unshifted_back_to_the_input_pitch(self):
        from infer import rtrvc

        rvc = rtrvc.RVC.__new__(rtrvc.RVC)
        rvc.device = torch.device("cpu")
        rvc.f0_mel_min = 1127 * np.log(1 + 50 / 700)
        rvc.f0_mel_max = 1127 * np.log(1 + 1100 / 700)
        detected = np.linspace(100, 160, 40) * 2 ** (11.0 / 12)
        rvc.get_f0_post(detected.copy())
        np.testing.assert_allclose(rvc.last_f0, detected)
        newest = rtrvc.newest_input_f0(rvc.last_f0, 10, 11.0)
        # infer() stores f0[3:-1] at the end of the pitch cache.
        np.testing.assert_allclose(newest, np.linspace(100, 160, 40)[3:-1][-10:])

if __name__ == "__main__":
    unittest.main()
