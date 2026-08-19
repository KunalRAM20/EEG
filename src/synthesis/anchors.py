"""Anchor discovery over the locally materialized SeizeIT2 participants.

An *anchor* is one real annotated moment in one real recording: a seizure event
or a seizure-free background window. It fixes the participant, the run, the
second, and the recorded semiology, and every other part of a composite case is
derived from it. Nothing here invents a value.
"""
from __future__ import annotations

import csv
import dataclasses
import re
from pathlib import Path

import config
from data.dataset_registry import SEIZEIT2_ROOT, _is_real_edf, seizeit2_subjects

# Modality directory names inside sub-XXX/ses-01/, in the order we report them.
MODALITY_DIRS = {"eeg": "eeg", "ecg": "ecg", "emg": "emg", "mov": "mov"}

# Anchor windowing. A seizure anchor uses the annotated event; a background
# anchor uses a fixed slice of a seizure-free run.
BACKGROUND_WINDOW_SECONDS = 60.0
# Baseline used as the pre-ictal comparison for heart rate and movement. Taken
# well before onset so it is not contaminated by the ictal rise itself.
BASELINE_LEAD_SECONDS = 300.0
BASELINE_WINDOW_SECONDS = 60.0
# Impedance-artifact events are avoided when placing a background anchor.
ARTIFACT_EVENT = "impd"

_RUN_RE = re.compile(r"_run-(\d+)_")


@dataclasses.dataclass(frozen=True)
class Anchor:
    """One real annotated moment in one real SeizeIT2 recording."""

    subject_id: str
    run: int
    kind: str              # "seizure" | "background"
    onset_seconds: float
    duration_seconds: float
    event_type: str        # e.g. "sz_foc_ia_m_hyperkinetic", "bckg"
    lateralization: str    # "left" | "right" | "n/a"
    localization: str      # "temp" | "cen_par" | "un" | "n/a"
    vigilance: str         # "awake" | "asleep" | "n/a"
    recording_duration: float

    @property
    def is_seizure(self) -> bool:
        return self.kind == "seizure"

    @property
    def impaired_awareness(self) -> bool:
        """SeizeIT2 encodes impaired awareness as the ``_ia_`` infix."""
        return "_ia_" in self.event_type

    @property
    def motor(self) -> bool:
        """``_m_`` marks motor semiology; ``_nm`` marks non-motor."""
        return "_m_" in self.event_type

    @property
    def hyperkinetic(self) -> bool:
        return self.event_type.endswith("hyperkinetic")

    @property
    def label(self) -> str:
        return f"{self.subject_id} run-{self.run:02d} @ {self.onset_seconds:.0f}s"

    def baseline_window(self) -> tuple[float, float]:
        """Return (start, end) of a pre-anchor baseline, clamped to the run."""
        start = self.onset_seconds - BASELINE_LEAD_SECONDS
        if start < 0.0:
            # Too close to the start of the run; fall back to the earliest
            # stretch that does not overlap the anchor itself.
            start = 0.0
        end = min(start + BASELINE_WINDOW_SECONDS, self.onset_seconds)
        if end <= start:
            # Degenerate only if the anchor begins at t=0; use the tail instead.
            start = self.onset_seconds + self.duration_seconds
            end = min(start + BASELINE_WINDOW_SECONDS, self.recording_duration)
        return float(start), float(end)

    def as_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "run": self.run,
            "kind": self.kind,
            "onset_seconds": self.onset_seconds,
            "duration_seconds": self.duration_seconds,
            "event_type": self.event_type,
            "lateralization": self.lateralization,
            "localization": self.localization,
            "vigilance": self.vigilance,
            "recording_duration": self.recording_duration,
            "impaired_awareness": self.impaired_awareness,
            "motor": self.motor,
            "hyperkinetic": self.hyperkinetic,
            "label": self.label,
        }


def _run_number(path: Path) -> int | None:
    match = _RUN_RE.search(path.name)
    return int(match.group(1)) if match else None


def _session_root(subject_id: str) -> Path:
    return SEIZEIT2_ROOT / subject_id / "ses-01"


def modality_paths(subject_id: str, run: int) -> dict[str, Path]:
    """Return the materialized EDF for each modality of one run.

    Only modalities whose EDF is genuinely present locally are returned, so a
    caller can never silently analyze a git-annex pointer stub.
    """
    session = _session_root(subject_id)
    found: dict[str, Path] = {}
    for modality, directory in MODALITY_DIRS.items():
        folder = session / directory
        if not folder.is_dir():
            continue
        for path in folder.glob("*.edf"):
            if _run_number(path) == run and _is_real_edf(path):
                found[modality] = path
                break
    return found


def available_subjects() -> list[dict]:
    """Participants whose four modalities are fully materialized locally.

    Reuses the registry's completeness check so this module and the dataset
    review pages can never disagree about what is actually on disk.
    """
    return [subject for subject in seizeit2_subjects() if subject.get("complete")]


def subject_events(subject_id: str) -> list[dict]:
    """Every annotated event for a participant, tagged with its run number."""
    session = _session_root(subject_id)
    events: list[dict] = []
    for path in sorted((session / "eeg").glob("*_events.tsv")):
        run = _run_number(path)
        if run is None:
            continue
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
        except OSError:
            continue
        for row in rows:
            row = dict(row)
            row["run"] = run
            events.append(row)
    return events


def _as_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def _clean(value) -> str:
    text = str(value or "").strip()
    return text or "n/a"


def anchor_pool(subject_id: str) -> list[Anchor]:
    """Every usable anchor for a participant: seizures first, then background.

    Background anchors let a generated case land at a low severity tier; without
    them every composite would be ictal, which would misrepresent the system.
    """
    runs_present = {
        run for run in {
            _run_number(path)
            for path in (_session_root(subject_id) / "eeg").glob("*.edf")
            if _is_real_edf(path)
        } if run is not None
    }
    seizures: list[Anchor] = []
    backgrounds: list[Anchor] = []
    artifacts: dict[int, list[tuple[float, float]]] = {}

    for row in subject_events(subject_id):
        run = row["run"]
        if run not in runs_present:
            continue
        event_type = _clean(row.get("eventType"))
        onset = _as_float(row.get("onset"))
        duration = _as_float(row.get("duration"))
        recording_duration = _as_float(row.get("recordingDuration"))
        if event_type == ARTIFACT_EVENT:
            artifacts.setdefault(run, []).append((onset, onset + duration))
            continue
        if event_type.startswith("sz"):
            seizures.append(Anchor(
                subject_id=subject_id, run=run, kind="seizure",
                onset_seconds=onset, duration_seconds=duration,
                event_type=event_type,
                lateralization=_clean(row.get("lateralization")),
                localization=_clean(row.get("localization")),
                vigilance=_clean(row.get("vigilance")),
                recording_duration=recording_duration))
        elif event_type == "bckg":
            backgrounds.append(Anchor(
                subject_id=subject_id, run=run, kind="background",
                onset_seconds=onset, duration_seconds=duration,
                event_type=event_type,
                lateralization=_clean(row.get("lateralization")),
                localization=_clean(row.get("localization")),
                vigilance=_clean(row.get("vigilance")),
                recording_duration=recording_duration))

    # A background row spans a whole seizure-free run. Narrow it to a single
    # analysable window placed away from the run edges and any impedance
    # artifact, so the composite analyses a defensible stretch of signal.
    narrowed: list[Anchor] = []
    for anchor in backgrounds:
        window = _place_background_window(anchor, artifacts.get(anchor.run, []))
        if window is None:
            continue
        start, length = window
        narrowed.append(dataclasses.replace(
            anchor, onset_seconds=start, duration_seconds=length))

    return seizures + narrowed


def _place_background_window(
        anchor: Anchor,
        artifact_spans: list[tuple[float, float]],
) -> tuple[float, float] | None:
    """Choose a clean BACKGROUND_WINDOW_SECONDS slice inside a background run."""
    span_end = anchor.onset_seconds + anchor.duration_seconds
    # Skip the first and last few minutes: electrode settling and disconnection
    # dominate those stretches.
    margin = max(BASELINE_LEAD_SECONDS + BASELINE_WINDOW_SECONDS, 600.0)
    earliest = anchor.onset_seconds + margin
    latest = span_end - margin - BACKGROUND_WINDOW_SECONDS
    if latest <= earliest:
        return None

    # Deterministic sweep: step through candidate starts and take the first that
    # clears every impedance artifact. Deterministic keeps a seeded case stable.
    step = max(BACKGROUND_WINDOW_SECONDS, 30.0)
    candidate = earliest
    while candidate <= latest:
        window = (candidate, candidate + BACKGROUND_WINDOW_SECONDS)
        clean = all(
            window[1] <= start or window[0] >= end
            for start, end in artifact_spans
        )
        if clean:
            return float(candidate), float(BACKGROUND_WINDOW_SECONDS)
        candidate += step
    return None


def subject_profile(subject_id: str) -> dict:
    """Aggregate real monitoring statistics for one participant.

    These are counted from the annotation files, not estimated, and they drive
    the seizure-burden term in composite severity and the derived history text.
    """
    events = subject_events(subject_id)
    monitored_seconds = 0.0
    seen_runs: set[int] = set()
    seizure_count = 0
    awake = 0
    asleep = 0
    semiology: dict[str, int] = {}
    lateralizations: dict[str, int] = {}
    localizations: dict[str, int] = {}
    durations: list[float] = []

    for row in events:
        run = row["run"]
        if run not in seen_runs:
            seen_runs.add(run)
            monitored_seconds += _as_float(row.get("recordingDuration"))
        event_type = _clean(row.get("eventType"))
        if not event_type.startswith("sz"):
            continue
        seizure_count += 1
        durations.append(_as_float(row.get("duration")))
        semiology[event_type] = semiology.get(event_type, 0) + 1
        lateralization = _clean(row.get("lateralization"))
        lateralizations[lateralization] = lateralizations.get(lateralization, 0) + 1
        localization = _clean(row.get("localization"))
        localizations[localization] = localizations.get(localization, 0) + 1
        vigilance = _clean(row.get("vigilance"))
        if vigilance == "awake":
            awake += 1
        elif vigilance == "asleep":
            asleep += 1

    monitored_hours = monitored_seconds / 3600.0
    monitored_days = monitored_hours / 24.0
    return {
        "subject_id": subject_id,
        "runs": len(seen_runs),
        "monitored_hours": monitored_hours,
        "seizure_count": seizure_count,
        "seizures_per_24h": (
            seizure_count / monitored_days if monitored_days > 0 else 0.0),
        "mean_seizure_duration": (
            sum(durations) / len(durations) if durations else 0.0),
        "longest_seizure_duration": max(durations) if durations else 0.0,
        "awake_seizures": awake,
        "asleep_seizures": asleep,
        "semiology": semiology,
        "lateralizations": lateralizations,
        "localizations": localizations,
    }


def subject_sex(subject_id: str) -> str | None:
    """Biological sex as published in the SeizeIT2 participants table."""
    path = Path(config.DATA_DIR) / "seizeit2" / "participants.tsv"
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row.get("participant_id") == subject_id:
                    value = (row.get("sex") or "").strip().lower()
                    return {"m": "male", "f": "female"}.get(value) or None
    except OSError:
        return None
    return None
