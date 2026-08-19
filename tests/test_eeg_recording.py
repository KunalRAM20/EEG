import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np

from src.processing.eeg_recording import (
    aggregate_candidate_events,
    assess_window_quality,
    load_recording,
    window_recording,
)


class EEGRecordingTests(unittest.TestCase):
    def test_csv_txt_npy_parsing_and_sampling_rate_requirement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "recording.csv"
            txt_path = root / "recording.txt"
            npy_path = root / "recording.npy"
            values = np.linspace(0.0, 1.0, 32)
            np.savetxt(csv_path, values, delimiter=",")
            np.savetxt(txt_path, values)
            np.save(npy_path, values)

            with self.assertRaisesRegex(ValueError, "sampling rate"):
                load_recording(csv_path)

            csv_recording = load_recording(csv_path, sampling_rate=128.0)
            txt_recording = load_recording(txt_path, sampling_rate=128.0)
            npy_recording = load_recording(npy_path, sampling_rate=128.0)

            self.assertEqual(csv_recording.sample_count, 32)
            self.assertEqual(txt_recording.sample_count, 32)
            self.assertEqual(npy_recording.sample_count, 32)
            self.assertEqual(csv_recording.channel_count, 1)
            self.assertEqual(txt_recording.channel_count, 1)
            self.assertEqual(npy_recording.channel_count, 1)

    def test_edf_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "roundtrip.edf"
            info = mne.create_info(["Fz"], 100.0, ch_types=["eeg"])
            raw = mne.io.RawArray(np.zeros((1, 100)), info, verbose="ERROR")
            raw.export(str(path), fmt="edf", overwrite=True)

            recording = load_recording(path)
            self.assertEqual(recording.channel_count, 1)
            self.assertEqual(recording.channel_names, ["Fz"])
            self.assertEqual(recording.metadata["format"], "edf")

    def test_bdf_round_trip_when_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "roundtrip.bdf"
            info = mne.create_info(["Fz"], 100.0, ch_types=["eeg"])
            raw = mne.io.RawArray(np.zeros((1, 100)), info, verbose="ERROR")
            try:
                raw.export(str(path), fmt="bdf", overwrite=True)
            except Exception as exc:
                self.skipTest(f"BDF export not available: {exc}")

            recording = load_recording(path)
            self.assertEqual(recording.channel_count, 1)
            self.assertEqual(recording.metadata["format"], "bdf")

    def test_multi_channel_requires_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi.npy"
            np.save(path, np.column_stack([np.arange(8.0), np.arange(8.0) * 2]))

            with self.assertRaisesRegex(ValueError, "channel names"):
                load_recording(path, sampling_rate=256.0)

            recording = load_recording(
                path, sampling_rate=256.0, channel_names="Fz,Cz")
            self.assertEqual(recording.channel_count, 2)
            self.assertEqual(recording.channel_names, ["Fz", "Cz"])

    def test_quality_and_abstention_checks(self):
        flat = np.zeros(128)
        quality = assess_window_quality(flat, 128.0)
        self.assertEqual(quality["status"], "unsuitable")
        self.assertIn("flat or constant signal", quality["issues"])

        noisy = np.sin(np.linspace(0, 50, 256)) + 0.5 * np.random.default_rng(0).normal(size=256)
        quality_noisy = assess_window_quality(noisy, 256.0)
        self.assertIn(quality_noisy["status"], {"suitable", "suitable_with_warnings"})

    def test_window_and_candidate_aggregation(self):
        signal = np.concatenate([
            np.zeros(100),
            np.ones(100),
            np.zeros(100),
            np.ones(100),
        ])
        windows = window_recording(signal, sampling_rate=100.0, window_seconds=1.0, overlap_seconds=0.5)
        self.assertGreaterEqual(len(windows), 3)
        for window in windows:
            window["channels_used"] = ["Fz"]
            window["seizure_probability"] = 0.9 if window["start_seconds"] >= 1.0 else 0.1
            window["quality"] = {"status": "suitable", "warnings": []}
        events = aggregate_candidate_events(windows, threshold=0.5)
        self.assertTrue(events)
        self.assertGreaterEqual(events[0]["supporting_windows"], 1)
        self.assertIn("duration_seconds", events[0])


if __name__ == "__main__":
    unittest.main()