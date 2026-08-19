"""
SQLite persistence for the clinical decision-support web app.

Four tables mirror the PDR workflow:
  patients     - uploaded patient + attached EEG segment + multi-modal profile
  predictions  - model output (risk, severity, features, XAI) per patient
  reports      - the generated clinical report (draft -> validated)
  validations  - the doctor's review decision on a report
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
import unicodedata

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    age          INTEGER,
    sex          TEXT,
    eeg_source   TEXT,
    eeg_group    TEXT,
    segment_json TEXT NOT NULL,
    modality_json TEXT NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id     INTEGER NOT NULL REFERENCES patients(id),
    model_name     TEXT,
    seizure_prob   REAL,
    seizure_label  INTEGER,
    severity_level INTEGER,
    severity_label TEXT,
    severity_score REAL,
    severity_method TEXT,
    features_json  TEXT,
    xai_json       TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reports (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id),
    content       TEXT,
    status        TEXT DEFAULT 'draft',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS validations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id     INTEGER NOT NULL REFERENCES reports(id),
    doctor_name   TEXT,
    decision      TEXT,
    edited_content TEXT,
    notes         TEXT,
    validated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id     INTEGER NOT NULL REFERENCES patients(id),
    report_id     INTEGER REFERENCES reports(id),
    actor_name    TEXT NOT NULL,
    action        TEXT NOT NULL,
    field_changes TEXT,
    notes         TEXT,
    timestamp     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS case_files (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id     INTEGER NOT NULL REFERENCES patients(id),
    modality       TEXT NOT NULL,
    original_name  TEXT NOT NULL,
    stored_path    TEXT NOT NULL,
    mime_type      TEXT,
    size_bytes     INTEGER,
    source_dataset TEXT,
    sha256         TEXT,
    processing_status TEXT DEFAULT 'contextual-only',
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS file_deletion_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id  INTEGER NOT NULL,
    stored_path TEXT NOT NULL,
    last_error  TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(patient_id, stored_path)
);
"""


@contextmanager
def get_conn():
    """Yield one transactional connection and always release its file handle."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        prediction_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(predictions)")
        }
        if "severity_method" not in prediction_columns:
            conn.execute("ALTER TABLE predictions ADD COLUMN severity_method TEXT")
        file_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(case_files)")
        }
        if "sha256" not in file_columns:
            conn.execute("ALTER TABLE case_files ADD COLUMN sha256 TEXT")
        if "processing_status" not in file_columns:
            conn.execute(
                "ALTER TABLE case_files ADD COLUMN processing_status TEXT "
                "DEFAULT 'contextual-only'")


# --------------------------------------------------------------------------- #
# Patients
# --------------------------------------------------------------------------- #
def create_patient(name, age, sex, eeg_source, eeg_group, segment, modality, created_at=None) -> int:
    with get_conn() as conn:
        if created_at:
            cur = conn.execute(
                """INSERT INTO patients
                   (name, age, sex, eeg_source, eeg_group, segment_json, modality_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (name, age, sex, eeg_source, eeg_group,
                 json.dumps(list(map(float, segment))), json.dumps(modality), str(created_at)),
            )
        else:
            cur = conn.execute(
                """INSERT INTO patients
                   (name, age, sex, eeg_source, eeg_group, segment_json, modality_json)
                   VALUES (?,?,?,?,?,?,?)""",
                (name, age, sex, eeg_source, eeg_group,
                 json.dumps(list(map(float, segment))), json.dumps(modality)),
            )
        return cur.lastrowid


def get_or_create_patient(name, age, sex, eeg_source, eeg_group, segment, modality, created_at=None) -> tuple[int, bool]:
    """Search for existing patient by exact name. If exists, returns (patient_id, False).
    Otherwise creates a new patient row and returns (new_patient_id, True).
    """
    matches = find_patients_by_exact_name(name)
    if matches:
        return matches[0]["id"], False
    pid = create_patient(name, age, sex, eeg_source, eeg_group, segment, modality, created_at=created_at)
    return pid, True


def update_patient_created_at(patient_id: int, created_at: str):
    """Update patient creation timestamp to match the actual file timestamp."""
    with get_conn() as conn:
        conn.execute("UPDATE patients SET created_at=? WHERE id=?", (str(created_at), patient_id))


def get_patient(pid: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()


def list_patients():
    with get_conn() as conn:
        return conn.execute("""
            SELECT p.*,
                   r.id AS report_id,
                   r.status AS report_status,
                   v.doctor_name AS latest_doctor_name,
                   v.decision AS latest_decision,
                   v.validated_at AS latest_validated_at
            FROM patients p
            LEFT JOIN predictions pr ON pr.id = (
                SELECT pr2.id FROM predictions pr2 WHERE pr2.patient_id = p.id ORDER BY pr2.created_at DESC, pr2.id DESC LIMIT 1
            )
            LEFT JOIN reports r ON r.prediction_id = pr.id
            LEFT JOIN validations v ON v.id = (
                SELECT v2.id FROM validations v2 WHERE v2.report_id = r.id ORDER BY v2.validated_at DESC, v2.id DESC LIMIT 1
            )
            ORDER BY p.created_at DESC, p.id DESC
        """).fetchall()


def find_patients_by_exact_name(name: str):
    """Return every Unicode-aware, case-insensitive exact name match."""
    wanted = unicodedata.normalize("NFKC", str(name).strip()).casefold()
    if not wanted:
        return []
    return [
        row for row in list_patients()
        if unicodedata.normalize("NFKC", row["name"].strip()).casefold() == wanted
    ]


def _patient_delete_counts(conn, patient_id: int) -> dict:
    row = conn.execute(
        """SELECT
             (SELECT COUNT(*) FROM predictions WHERE patient_id=?) predictions,
             (SELECT COUNT(*) FROM reports r JOIN predictions p
                ON r.prediction_id=p.id WHERE p.patient_id=?) reports,
             (SELECT COUNT(*) FROM validations v JOIN reports r
                ON v.report_id=r.id JOIN predictions p
                ON r.prediction_id=p.id WHERE p.patient_id=?) validations,
             (SELECT COUNT(*) FROM case_files WHERE patient_id=?) files,
             (SELECT COUNT(*) FROM audit_logs WHERE patient_id=?) audit_logs""",
        (patient_id, patient_id, patient_id, patient_id, patient_id),
    ).fetchone()
    return dict(row)


def patient_delete_summary(patient_id: int):
    """Preview the exact database graph removed with a patient."""
    with get_conn() as conn:
        patient = conn.execute(
            "SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone()
        if patient is None:
            return None
        return {"patient": dict(patient),
                **_patient_delete_counts(conn, patient_id)}


def delete_patient(patient_id: int):
    """Delete one patient and every dependent row in a single transaction."""
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        patient = conn.execute(
            "SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone()
        if patient is None:
            return None
        result = {"patient": dict(patient),
                  **_patient_delete_counts(conn, patient_id)}
        paths = conn.execute(
            "SELECT stored_path FROM case_files WHERE patient_id=?",
            (patient_id,),
        ).fetchall()
        for row in paths:
            conn.execute(
                """INSERT OR IGNORE INTO file_deletion_queue
                   (patient_id, stored_path) VALUES (?,?)""",
                (patient_id, row["stored_path"]),
            )

        conn.execute(
            "DELETE FROM audit_logs WHERE patient_id=?", (patient_id,)
        )
        conn.execute(
            """DELETE FROM validations WHERE report_id IN (
                   SELECT r.id FROM reports r JOIN predictions p
                     ON r.prediction_id=p.id WHERE p.patient_id=?)""",
            (patient_id,),
        )
        conn.execute(
            """DELETE FROM reports WHERE prediction_id IN (
                   SELECT id FROM predictions WHERE patient_id=?)""",
            (patient_id,),
        )
        conn.execute("DELETE FROM predictions WHERE patient_id=?", (patient_id,))
        conn.execute("DELETE FROM case_files WHERE patient_id=?", (patient_id,))
        conn.execute("DELETE FROM patients WHERE id=?", (patient_id,))
        result["stored_paths"] = [row["stored_path"] for row in paths]
        return result


def complete_file_deletions(patient_id: int, stored_paths: list[str]):
    """Remove successfully erased files from the persistent retry queue."""
    if not stored_paths:
        return
    with get_conn() as conn:
        conn.executemany(
            "DELETE FROM file_deletion_queue WHERE patient_id=? AND stored_path=?",
            [(patient_id, path) for path in stored_paths],
        )


def record_file_deletion_failures(patient_id: int,
                                  failures: list[tuple[str, str]]):
    """Keep the error for files that need a later removal retry."""
    if not failures:
        return
    with get_conn() as conn:
        conn.executemany(
            """UPDATE file_deletion_queue SET last_error=?
               WHERE patient_id=? AND stored_path=?""",
            [(error, patient_id, path) for path, error in failures],
        )


def list_pending_file_deletions():
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM file_deletion_queue ORDER BY id").fetchall()


def create_case_file(patient_id: int, modality: str, original_name: str,
                     stored_path: str, mime_type: str | None, size_bytes: int,
                     source_dataset: str | None = None, sha256: str | None = None,
                     processing_status: str = "contextual-only") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO case_files
               (patient_id, modality, original_name, stored_path, mime_type,
                size_bytes, source_dataset, sha256, processing_status)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (patient_id, modality, original_name, stored_path, mime_type,
             int(size_bytes), source_dataset, sha256, processing_status),
        )
        return cur.lastrowid


def list_case_files(patient_id: int):
    with get_conn() as conn:
        return conn.execute(
            """SELECT * FROM case_files WHERE patient_id=?
               ORDER BY modality, id""", (patient_id,)).fetchall()


def get_case_file(file_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM case_files WHERE id=?", (file_id,)).fetchone()


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
def _insert_prediction(conn, patient_id, prediction: dict) -> int:
    sev = prediction["severity"]
    cur = conn.execute(
        """INSERT INTO predictions
           (patient_id, model_name, seizure_prob, seizure_label,
            severity_level, severity_label, severity_score,
            severity_method, features_json, xai_json)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (patient_id, prediction.get("model_name"),
         prediction["seizure_probability"], prediction["seizure_prediction"],
         sev["level"], sev["label"], sev["score"],
         prediction.get("severity_method") or sev.get("grading_basis") or sev.get("method") or "calibrated seizure-probability band",
         json.dumps(prediction.get("features", {})),
         json.dumps(prediction.get("xai", {}))),
    )
    return cur.lastrowid


def create_prediction(patient_id, prediction: dict) -> int:
    with get_conn() as conn:
        return _insert_prediction(conn, patient_id, prediction)


def create_prediction_and_report(patient_id: int, prediction: dict,
                                 content: str) -> tuple[int, int]:
    """Create a prediction and its report atomically."""
    with get_conn() as conn:
        prediction_id = _insert_prediction(conn, patient_id, prediction)
        cur = conn.execute(
            "INSERT INTO reports (prediction_id, content) VALUES (?,?)",
            (prediction_id, content),
        )
        return prediction_id, cur.lastrowid


def get_prediction(pred_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM predictions WHERE id=?", (pred_id,)).fetchone()


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def create_report(prediction_id, content) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO reports (prediction_id, content) VALUES (?,?)",
            (prediction_id, content))
        return cur.lastrowid


def get_report(report_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()


def get_report_bundle(report_id: int):
    """Return joined report + prediction + patient rows for rendering."""
    with get_conn() as conn:
        return conn.execute(
            """SELECT r.id AS report_id, r.content, r.status, r.created_at,
                      p.id AS prediction_id, p.model_name, p.seizure_prob,
                      p.seizure_label, p.severity_level, p.severity_label,
                      p.severity_score, p.features_json, p.xai_json,
                      p.severity_method,
                      pt.id AS patient_id, pt.name, pt.age, pt.sex,
                      pt.eeg_source, pt.eeg_group, pt.modality_json,
                      pt.created_at AS patient_created_at
               FROM reports r
               JOIN predictions p ON r.prediction_id = p.id
               JOIN patients pt ON p.patient_id = pt.id
               WHERE r.id=?""", (report_id,)).fetchone()


def update_report_status(report_id: int, status: str):
    with get_conn() as conn:
        conn.execute("UPDATE reports SET status=? WHERE id=?", (status, report_id))


def list_reports(include_legacy: bool = False):
    with get_conn() as conn:
        where = "" if include_legacy else "WHERE p.severity_method IS NOT NULL"
        return conn.execute(
            f"""SELECT r.id AS report_id, p.patient_id AS patient_id, r.status, r.created_at,
                      pt.name, p.severity_level, p.severity_label, p.seizure_prob,
                      p.severity_method
               FROM reports r
               JOIN predictions p ON r.prediction_id = p.id
               JOIN patients pt ON p.patient_id = pt.id
               {where}
               ORDER BY r.created_at DESC, r.id DESC""").fetchall()


# --------------------------------------------------------------------------- #
# Validations
# --------------------------------------------------------------------------- #
def create_validation(report_id, doctor_name, decision, edited_content, notes):
    status = {"approve": "approved", "modify": "modified",
              "reject": "rejected"}.get(decision, "reviewed")
    action_label = {"approve": "Approved", "modify": "Modified",
                    "reject": "Rejected"}.get(decision, "Updated")
    
    with get_conn() as conn:
        old_report = conn.execute(
            """SELECT r.*, p.patient_id FROM reports r
               JOIN predictions p ON r.prediction_id = p.id
               WHERE r.id=?""", (report_id,)
        ).fetchone()
        
        old_status = old_report["status"] if old_report else "draft"
        patient_id = old_report["patient_id"] if old_report else 0
        old_content = old_report["content"] if old_report else ""

        conn.execute(
            """INSERT INTO validations
               (report_id, doctor_name, decision, edited_content, notes)
               VALUES (?,?,?,?,?)""",
            (report_id, doctor_name, decision, edited_content, notes))
        conn.execute("UPDATE reports SET status=? WHERE id=?", (status, report_id))

        changes = [
            {"field": "Review Status", "old": old_status, "new": status}
        ]
        if edited_content and edited_content.strip() != old_content.strip():
            changes.append({
                "field": "Clinical Content",
                "old": (old_content[:50] + "...") if len(old_content) > 50 else old_content,
                "new": (edited_content[:50] + "...") if len(edited_content) > 50 else edited_content
            })
        if notes:
            changes.append({"field": "Clinical Notes", "old": "None", "new": notes})

        changes_json = json.dumps(changes)
        conn.execute(
            """INSERT INTO audit_logs (patient_id, report_id, actor_name, action, field_changes, notes)
               VALUES (?,?,?,?,?,?)""",
            (patient_id, report_id, doctor_name, action_label, changes_json, notes)
        )


def create_audit_log(patient_id: int, report_id: int | None, actor_name: str, action: str,
                     field_changes: list[dict] | dict | str | None, notes: str = "", timestamp: str | None = None):
    changes_str = json.dumps(field_changes) if isinstance(field_changes, (list, dict)) else (field_changes or "[]")
    with get_conn() as conn:
        if timestamp:
            conn.execute(
                """INSERT INTO audit_logs
                   (patient_id, report_id, actor_name, action, field_changes, notes, timestamp)
                   VALUES (?,?,?,?,?,?,?)""",
                (patient_id, report_id, actor_name, action, changes_str, notes, str(timestamp))
            )
        else:
            conn.execute(
                """INSERT INTO audit_logs
                   (patient_id, report_id, actor_name, action, field_changes, notes)
                   VALUES (?,?,?,?,?,?)""",
                (patient_id, report_id, actor_name, action, changes_str, notes)
            )


def list_audit_logs(patient_id: int | None = None, doctor_name: str | None = None,
                    action: str | None = None, start_date: str | None = None,
                    end_date: str | None = None):
    with get_conn() as conn:
        query = """
            SELECT a.*, p.name AS patient_name
            FROM audit_logs a
            JOIN patients p ON a.patient_id = p.id
            WHERE 1=1
        """
        params = []
        if patient_id:
            query += " AND (a.patient_id = ? OR CAST(a.patient_id AS TEXT) = ?)"
            params.extend([patient_id, str(patient_id)])
        if doctor_name:
            query += " AND LOWER(a.actor_name) LIKE LOWER(?)"
            params.append(f"%{doctor_name}%")
        if action and action.lower() != "all":
            query += " AND LOWER(a.action) = LOWER(?)"
            params.append(action)
        if start_date:
            query += " AND DATE(a.timestamp) >= DATE(?)"
            params.append(start_date)
        if end_date:
            query += " AND DATE(a.timestamp) <= DATE(?)"
            params.append(end_date)
        
        query += " ORDER BY a.timestamp DESC, a.id DESC"
        rows = conn.execute(query, params).fetchall()

        if not rows and not (doctor_name or action or start_date or end_date or patient_id):
            conn.execute("""
                INSERT INTO audit_logs (patient_id, report_id, actor_name, action, field_changes, notes, timestamp)
                SELECT pr.patient_id, v.report_id, v.doctor_name,
                       CASE WHEN LOWER(v.decision) = 'approve' THEN 'Approved'
                            WHEN LOWER(v.decision) = 'modify' THEN 'Modified'
                            WHEN LOWER(v.decision) = 'reject' THEN 'Rejected'
                            ELSE 'Updated' END,
                       json_array(json_object('field', 'Status', 'old', 'draft', 'new', v.decision)),
                       v.notes, v.validated_at
                FROM validations v
                JOIN reports r ON v.report_id = r.id
                JOIN predictions pr ON r.prediction_id = pr.id
            """)
            rows = conn.execute(query, params).fetchall()

        return [dict(r) for r in rows]


def get_validation(report_id: int):
    with get_conn() as conn:
        return conn.execute(
            """SELECT * FROM validations WHERE report_id=?
               ORDER BY validated_at DESC, id DESC LIMIT 1""",
            (report_id,)).fetchone()


def get_validation_by_id(validation_id: int):
    """Return a single validation record by its primary key ID."""
    with get_conn() as conn:
        return conn.execute(
            """SELECT * FROM validations WHERE id=?""",
            (validation_id,)).fetchone()


def get_report_validations(report_id: int):
    """Return all historical validations for a report ordered from newest to oldest."""
    with get_conn() as conn:
        return conn.execute(
            """SELECT * FROM validations WHERE report_id=?
               ORDER BY validated_at DESC, id DESC""",
            (report_id,)).fetchall()


def get_patient_validations(patient_id: int):
    """Return all historical doctor reviews/validations for a patient's reports."""
    with get_conn() as conn:
        return conn.execute(
            """SELECT v.*, r.id AS report_id, r.status AS report_status
               FROM validations v
               JOIN reports r ON v.report_id = r.id
               JOIN predictions p ON r.prediction_id = p.id
               WHERE p.patient_id = ?
               ORDER BY v.validated_at DESC, v.id DESC""",
            (patient_id,)).fetchall()


def dashboard_stats():
    with get_conn() as conn:
        stats = {
            "patients": conn.execute("SELECT COUNT(*) c FROM patients").fetchone()["c"],
            "predictions": conn.execute(
                "SELECT COUNT(*) c FROM predictions WHERE severity_method IS NOT NULL"
            ).fetchone()["c"],
            "reports": conn.execute(
                """SELECT COUNT(*) c FROM reports r JOIN predictions p
                     ON r.prediction_id=p.id
                     WHERE p.severity_method IS NOT NULL"""
            ).fetchone()["c"],
            "validated": conn.execute(
                """SELECT COUNT(*) c FROM reports r JOIN predictions p
                     ON r.prediction_id=p.id
                     WHERE r.status!='draft' AND p.severity_method IS NOT NULL"""
            ).fetchone()["c"],
            "legacy_reports": conn.execute(
                """SELECT COUNT(*) c FROM reports r JOIN predictions p
                     ON r.prediction_id=p.id WHERE p.severity_method IS NULL"""
            ).fetchone()["c"],
            "pending_file_deletions": conn.execute(
                "SELECT COUNT(*) c FROM file_deletion_queue"
            ).fetchone()["c"],
        }
    return stats
