"""Shared modality result contract.

Only the narrowly compatible Bonn/UCI numeric EEG path has a deployed model.
Every other uploaded modality is preserved for authorised clinician review and
must not acquire generated findings merely because a file was uploaded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Status = Literal["processed", "unsupported", "failed", "contextual-only"]


@dataclass(frozen=True)
class ModalityResult:
    modality: str
    status: Status
    input_provenance: dict[str, Any]
    quality_assessment: dict[str, Any] = field(default_factory=dict)
    extracted_findings: list[dict[str, Any]] = field(default_factory=list)
    model_name: str | None = None
    model_version: str | None = None
    preprocessing_version: str | None = None
    confidence: float | None = None
    uncertainty: str | None = None
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    machine_readable_evidence: dict[str, Any] = field(default_factory=dict)
    human_readable_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def processing_manifest(files, *, eeg_processed: bool = False) -> list[dict]:
    """Describe what actually affected output, without fabricating findings."""
    results = []
    seen = set()
    for item in files:
        modality = item["modality"]
        seen.add(modality)
        processed = modality == "EEG" and eeg_processed
        results.append(ModalityResult(
            modality=modality,
            status="processed" if processed else "contextual-only",
            input_provenance={
                "asset_id": item["id"],
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
                "mime_type": item["mime_type"],
            },
            quality_assessment={"status": "basic-file-validation-passed"},
            model_name="Bonn/UCI EEG classifier" if processed else None,
            uncertainty=None if processed else "No validated model is available.",
            limitations=[] if processed else [
                "Stored for authorised clinician review; not used by the model."
            ],
            human_readable_summary=(
                "Used by the compatible EEG model." if processed else
                "Saved for clinician review; automatic analysis is unsupported."
            ),
        ).to_dict())
    if eeg_processed and "EEG" not in seen:
        results.append(ModalityResult(
            modality="EEG",
            status="processed",
            input_provenance={"source": "registered numeric EEG"},
            quality_assessment={"status": "legacy fixed-window checks passed"},
            model_name="Bonn/UCI EEG classifier",
            limitations=["Single-channel fixed-length research-dataset model."],
            human_readable_summary="Used by the compatible EEG model.",
        ).to_dict())
    return results
