"""Conservative upload validation and immutable-file metadata."""
from __future__ import annotations

import hashlib
import os

ALLOWED = {
    "eeg_file": {".txt", ".csv", ".npy", ".edf", ".bdf"},
    "ecg_file": {".txt", ".csv", ".npy", ".edf", ".bdf"},
    "mri_file": {".nii", ".gz", ".dcm"},
    "xray_file": {".png", ".jpg", ".jpeg", ".dcm"},
    "medical_report": {".txt", ".pdf", ".doc", ".docx"},
    "imaging_data": {".nii", ".gz", ".dcm", ".png", ".jpg", ".jpeg"},
}

SIGNATURES = {
    ".pdf": (b"%PDF-",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",), ".jpeg": (b"\xff\xd8\xff",),
    ".npy": (b"\x93NUMPY",),
}


def validate_upload(storage, field: str, max_bytes: int) -> dict:
    """Validate name, size and known magic bytes without trusting MIME alone."""
    supplied = storage.filename or ""
    name = os.path.basename(supplied)
    if (not name or "/" in supplied or "\\" in supplied or
            name in {".", ".."}):
        raise ValueError("The uploaded filename is invalid.")
    suffix = os.path.splitext(name.lower())[1]
    if suffix not in ALLOWED.get(field, set()):
        raise ValueError(f"{name}: this file type is not supported.")
    stream = storage.stream
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(0)
    if size <= 0:
        raise ValueError(f"{name}: the file is empty.")
    if size > max_bytes:
        raise ValueError(f"{name}: the file exceeds the upload size limit.")
    head = stream.read(132)
    stream.seek(0)
    signatures = SIGNATURES.get(suffix)
    if signatures and not any(head.startswith(sig) for sig in signatures):
        raise ValueError(f"{name}: file contents do not match its extension.")
    if suffix == ".dcm" and not (len(head) >= 132 and head[128:132] == b"DICM"):
        raise ValueError(f"{name}: only DICOM files with a valid preamble are accepted.")
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    stream.seek(0)
    return {"original_name": name, "extension": suffix, "size_bytes": size,
            "sha256": digest.hexdigest(),
            "declared_mime_type": storage.mimetype or "application/octet-stream"}
