"""Typed, clinically honest modality-processing contracts."""

from .eeg_recording import (EEGRecording, aggregate_candidate_events,
							assess_window_quality, load_recording,
							recording_hash, window_recording)
from .processors import ModalityResult, processing_manifest

__all__ = [
	"ModalityResult",
	"processing_manifest",
	"EEGRecording",
	"aggregate_candidate_events",
	"assess_window_quality",
	"load_recording",
	"recording_hash",
	"window_recording",
]
