"""Controller-owned SQLite state. Chain commitments remain the source of work."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from shared.config import TargetConfig
from shared.contracts import (
    Award,
    Commitment,
    Decision,
    EvaluationJob,
    EvaluationResult,
    SubmissionManifest,
)
from shared.datasets import DatasetEvidence, RoundContext
from shared.releases import WinnerRelease
from validator.similarity import Fingerprint, SimilarityComparison, compare, comparison_shortlist

_SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
 id TEXT PRIMARY KEY, model_id TEXT NOT NULL, revision TEXT NOT NULL,
 config TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS submissions (
 id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id),
 commitment TEXT NOT NULL, manifest TEXT NOT NULL, content_id TEXT NOT NULL,
 block INTEGER NOT NULL, position INTEGER NOT NULL, hotkey TEXT NOT NULL,
 status TEXT NOT NULL, reason TEXT, score REAL
);
CREATE INDEX IF NOT EXISTS queue ON submissions(profile_id,status,block,position,hotkey);
CREATE TABLE IF NOT EXISTS sessions (
 id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id),
 target_key TEXT NOT NULL, state TEXT NOT NULL, started_at REAL NOT NULL,
 deadline REAL NOT NULL, pod_id TEXT, request TEXT, finished_at REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS active_target ON sessions(target_key)
 WHERE state != 'finished';
CREATE TABLE IF NOT EXISTS session_members (
 session_id TEXT NOT NULL REFERENCES sessions(id),
 submission_id TEXT NOT NULL REFERENCES submissions(id),
 PRIMARY KEY(session_id,submission_id)
);
CREATE TABLE IF NOT EXISTS jobs (
 id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
 submission_id TEXT NOT NULL REFERENCES submissions(id), job TEXT NOT NULL,
 status TEXT NOT NULL, result TEXT, decision TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS active_job ON jobs(session_id) WHERE status='running';
CREATE TABLE IF NOT EXISTS incumbents (
 target_key TEXT PRIMARY KEY, submission_id TEXT NOT NULL, artifact TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS awards (
 sequence INTEGER PRIMARY KEY AUTOINCREMENT,
 submission_id TEXT NOT NULL UNIQUE, target_key TEXT NOT NULL, award TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS discoveries (
 id TEXT PRIMARY KEY, commitment TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
 reason TEXT, attempts INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS publications (
 submission_id TEXT PRIMARY KEY REFERENCES submissions(id), release TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fingerprints (
 scope TEXT NOT NULL, artifact TEXT NOT NULL, fingerprint TEXT NOT NULL,
 submission_id TEXT, PRIMARY KEY(scope,artifact)
);
CREATE TABLE IF NOT EXISTS similarity_verdicts (
 submission_id TEXT PRIMARY KEY, verdict TEXT NOT NULL
);
"""


def target_key(target: TargetConfig) -> str:
    return f"{target.model_id}@{target.revision}"


class State:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=10, isolation_level=None)
        path.chmod(0o600)
        self.connection.row_factory = sqlite3.Row
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version > 5:
            raise ValueError("State schema is newer than this controller")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(_SCHEMA)
        if "attempts" not in {
            r[1] for r in self.connection.execute("PRAGMA table_info(discoveries)")
        }:
            self.connection.execute(
                "ALTER TABLE discoveries ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0"
            )
        columns = {r[1] for r in self.connection.execute("PRAGMA table_info(sessions)")}
        for name, kind in (
            ("cutoff_block", "INTEGER"),
            ("sampling", "TEXT"),
            ("dataset_plan_digest", "TEXT"),
            ("seed_block", "INTEGER"),
        ):
            if name not in columns:
                self.connection.execute(f"ALTER TABLE sessions ADD COLUMN {name} {kind}")
        for table, column in (("submissions", "submitted_at"), ("discoveries", "seen_at")):
            if column not in {r[1] for r in self.connection.execute(f"PRAGMA table_info({table})")}:
                self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} REAL")
        with self.transaction() as conn:
            if version and version < 5:
                self._revalidate_legacy_intake(conn)
            conn.execute("PRAGMA user_version=5")

    def _revalidate_legacy_intake(self, conn: sqlite3.Connection) -> None:
        # Existing ready rows have no mandatory fingerprint evidence. Revalidate before rent.
        rows = conn.execute(
            "SELECT s.id,s.commitment FROM submissions s WHERE status IN ('ready','running') "
            "AND NOT EXISTS(SELECT 1 FROM similarity_verdicts v WHERE v.submission_id=s.id)"
        ).fetchall()
        if rows:
            # Claimed sessions may not have a running job yet. Stop those too.
            conn.execute("UPDATE sessions SET state='stopping' WHERE state!='finished'")
            conn.execute("UPDATE jobs SET status='abandoned' WHERE status='running'")
        observed = time.time()
        for row in rows:
            commitment = Commitment.model_validate_json(row["commitment"])
            submitted_at = commitment.submitted_at
            if submitted_at is None:
                discovered = conn.execute(
                    "SELECT seen_at FROM discoveries WHERE id=?", (row["id"],)
                ).fetchone()
                submitted_at = (
                    discovered[0] if discovered and discovered[0] is not None else observed
                )
            conn.execute(
                "INSERT INTO discoveries(id,commitment,status,seen_at) VALUES(?,?,'pending',?) "
                "ON CONFLICT(id) DO UPDATE SET status='pending'",
                (row["id"], row["commitment"], submitted_at),
            )
            conn.execute(
                "UPDATE submissions SET status='pending_similarity',submitted_at="
                "COALESCE(submitted_at,?) WHERE id=?",
                (submitted_at, row["id"]),
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self.connection.close()

    def register_target(self, target: TargetConfig) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO profiles(id,model_id,revision,config) VALUES(?,?,?,?)",
            (target.identity, target.model_id, target.revision, target.model_dump_json()),
        )

    def profile(self, profile_id: str) -> TargetConfig:
        row = self.connection.execute(
            "SELECT config FROM profiles WHERE id=?", (profile_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Unknown competition profile")
        return TargetConfig.model_validate_json(row[0])

    def known(self, submission_id: str) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM submissions WHERE id=?", (submission_id,)
            ).fetchone()
            is not None
        )

    def discover(
        self, commitments: tuple[Commitment, ...], finalized_block: int, now: float | None = None
    ) -> None:
        observed = time.time() if now is None else now
        with self.transaction() as conn:
            for commitment in commitments:
                conn.execute(
                    "INSERT OR IGNORE INTO discoveries(id,commitment,seen_at) VALUES(?,?,?)",
                    (commitment.identity, commitment.model_dump_json(), observed),
                )
            self.set_metadata("finalized_block", str(finalized_block))

    def pending_admissions(self, limit: int = 100) -> list[Commitment]:
        if not 1 <= limit <= 100:
            raise ValueError("Admission batch must contain between 1 and 100 commitments")
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT id,commitment FROM discoveries WHERE status='pending' "
                "ORDER BY attempts,rowid LIMIT ?",
                (limit,),
            ).fetchall()
            conn.executemany(
                "UPDATE discoveries SET attempts=attempts+1 WHERE id=?", [(row[0],) for row in rows]
            )
            return [Commitment.model_validate_json(row[1]) for row in rows]

    def resolve_admission(self, commitment: Commitment, status: str, reason: str = "") -> None:
        self.connection.execute(
            "UPDATE discoveries SET status=?,reason=? WHERE id=?",
            (status, reason, commitment.identity),
        )

    def admit(
        self,
        commitment: Commitment,
        manifest: SubmissionManifest,
        target: TargetConfig,
        now: float,
        *,
        fingerprint: Fingerprint,
        full_matches: dict[str, SimilarityComparison] | None = None,
    ) -> None:
        if (
            manifest.competition_id != target.identity
            or manifest.target_id != target.model_id
            or manifest.target_revision != target.revision
            or manifest.method not in target.methods
        ):
            raise ValueError("Submission does not match its competition")
        content_id = hashlib.sha256(
            json.dumps(
                [f.model_dump() for f in sorted(manifest.files, key=lambda f: f.path)],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        with self.transaction() as conn:
            self.register_target(target)
            existing = conn.execute(
                "SELECT status FROM submissions WHERE id=?", (commitment.identity,)
            ).fetchone()
            if existing and existing[0] != "pending_similarity":
                return
            duplicate = conn.execute(
                "SELECT 1 FROM submissions WHERE profile_id=? AND content_id=? AND id!=?",
                (target.identity, content_id, commitment.identity),
            ).fetchone()
            public_copy = (
                target.public_reference and commitment.artifact == target.public_reference.artifact
            )
            status, reason = (
                ("rejected", "duplicate_checkpoint")
                if duplicate or public_copy
                else ("ready", None)
            )
            verdict: dict[str, Any] = {
                "version": 1,
                "outcome": "pass",
                "exact_duplicate": False,
                "cosine": None,
                "relative_distance": None,
                "copied_fraction": None,
                "source_artifact": None,
                "checked_at": now,
                "policy": target.similarity.model_dump(mode="json") if target.similarity else None,
            }
            policy = target.similarity
            comparable: dict[str, SimilarityComparison] = {}
            for row in conn.execute(
                "SELECT artifact,fingerprint FROM fingerprints WHERE scope=? ORDER BY artifact",
                (fingerprint.scope,),
            ).fetchall():
                previous = Fingerprint.model_validate_json(row["fingerprint"])
                comparison = compare(fingerprint, previous)
                if not comparison.comparable:
                    continue
                comparable[row["artifact"]] = comparison
                if comparison.cosine is not None and (
                    verdict["cosine"] is None or comparison.cosine > verdict["cosine"]
                ):
                    verdict.update(comparison.model_dump(), source_artifact=row["artifact"])
                if comparison.exact_duplicate:
                    status, reason = "rejected", "duplicate_checkpoint"
                    verdict.update(
                        comparison.model_dump(),
                        outcome="duplicate",
                        source_artifact=row["artifact"],
                    )
                    break
                if policy is not None and policy.reject_near_copies:
                    confirmed = (full_matches or {}).get(row["artifact"])
                    if confirmed is None or not confirmed.full_comparison:
                        # Preparation fully checks the declared shortlist before admission.
                        continue
                    elements = sum(t.elements for t in confirmed.tensor_comparisons)
                    copied = sum(
                        t.elements
                        for t in confirmed.tensor_comparisons
                        if t.cosine is not None
                        and t.relative_distance is not None
                        and t.cosine >= policy.min_cosine
                        and t.relative_distance <= policy.max_relative_distance
                    )
                    if (
                        fingerprint.config_digest == previous.config_digest
                        and confirmed.comparable
                        and confirmed.cosine is not None
                        and confirmed.relative_distance is not None
                        and elements
                        and confirmed.cosine >= policy.min_cosine
                        and confirmed.relative_distance <= policy.max_relative_distance
                        and copied / elements >= policy.min_copied_fraction
                    ):
                        status, reason = "rejected", "near_duplicate_checkpoint"
                        verdict.update(
                            confirmed.model_dump(),
                            outcome="near_duplicate",
                            source_artifact=row["artifact"],
                        )
                        break
            if status == "ready" and policy is not None and policy.reject_near_copies:
                required = comparison_shortlist(comparable, manifest.provenance.parent_artifacts)
                if any(
                    (confirmation := (full_matches or {}).get(artifact)) is None
                    or not confirmation.full_comparison
                    for artifact in required
                ):
                    status, reason = "pending_similarity", "similarity_confirmation_missing"
                    verdict["outcome"] = "inconclusive"
            if status == "rejected" and verdict["outcome"] == "pass":
                verdict["outcome"] = "duplicate"
            discovered = conn.execute(
                "SELECT seen_at FROM discoveries WHERE id=?",
                (commitment.identity,),
            ).fetchone()
            submitted_at = commitment.submitted_at
            if submitted_at is None:
                submitted_at = discovered[0] if discovered and discovered[0] is not None else now
            submitted_at = min(float(submitted_at), now)
            conn.execute(
                "INSERT OR IGNORE INTO submissions(id,profile_id,commitment,manifest,content_id,"
                "block,position,hotkey,status,reason,score,submitted_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    commitment.identity,
                    target.identity,
                    commitment.model_dump_json(),
                    manifest.model_dump_json(),
                    content_id,
                    commitment.block,
                    commitment.extrinsic_index,
                    commitment.hotkey,
                    status,
                    reason,
                    None,
                    submitted_at,
                ),
            )
            if existing:
                conn.execute(
                    "UPDATE submissions SET status=?,reason=?,"
                    "submitted_at=COALESCE(submitted_at,?) "
                    "WHERE id=?",
                    (status, reason, submitted_at, commitment.identity),
                )
            conn.execute(
                "INSERT OR REPLACE INTO similarity_verdicts VALUES(?,?)",
                (commitment.identity, json.dumps(verdict, sort_keys=True)),
            )
            if status == "ready":
                conn.execute(
                    "INSERT OR IGNORE INTO fingerprints VALUES(?,?,?,?)",
                    (
                        fingerprint.scope,
                        commitment.artifact,
                        fingerprint.model_dump_json(),
                        commitment.identity,
                    ),
                )

    def ready_count(self, profile_id: str) -> int:
        return int(
            self.connection.execute(
                "SELECT count(*) FROM submissions WHERE profile_id=? AND status='ready'",
                (profile_id,),
            ).fetchone()[0]
        )

    def _oldest_ready(self, target: TargetConfig) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT s.* FROM submissions s JOIN profiles p ON p.id=s.profile_id "
            "WHERE p.model_id=? AND p.revision=? AND s.status='ready' "
            "ORDER BY s.block,s.position,s.hotkey,s.id LIMIT 1",
            (target.model_id, target.revision),
        ).fetchone()

    def target_ready_count(self, target: TargetConfig) -> int:
        return int(
            self.connection.execute(
                "SELECT count(*) FROM submissions s JOIN profiles p ON s.profile_id=p.id "
                "WHERE p.model_id=? AND p.revision=? AND s.status='ready'",
                (target.model_id, target.revision),
            ).fetchone()[0]
        )

    def claim_session(
        self,
        target: TargetConfig,
        now: float,
        schedule_seconds: float,
        early_depth: int,
        session_seconds: int,
    ) -> dict[str, Any] | None:
        with self.transaction() as conn:
            self.register_target(target)
            if conn.execute(
                "SELECT 1 FROM sessions WHERE target_key=? AND state!='finished'",
                (target_key(target),),
            ).fetchone():
                return None
            oldest = self._oldest_ready(target)
            if oldest is None or oldest["profile_id"] != target.identity:
                return None
            depth = self.target_ready_count(target)
            first = float(
                conn.execute(
                    "SELECT MIN(COALESCE(s.submitted_at,?)) FROM submissions s "
                    "JOIN profiles p ON p.id=s.profile_id "
                    "WHERE p.model_id=? AND p.revision=? AND s.status='ready'",
                    (now, target.model_id, target.revision),
                ).fetchone()[0]
            )
            if depth < early_depth and now - first < schedule_seconds:
                return None
            cutoff = None
            if target.datasets is not None:
                cursor = self.metadata("finalized_block")
                if not cursor or int(cursor) < oldest["block"]:
                    return None
                cutoff = int(cursor)
            session_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO sessions(id,profile_id,target_key,state,started_at,"
                "deadline,cutoff_block) "
                "VALUES(?,?,?,'active',?,?,?)",
                (
                    session_id,
                    target.identity,
                    target_key(target),
                    now,
                    now + session_seconds,
                    cutoff,
                ),
            )
            if cutoff is not None:
                conn.execute(
                    "INSERT INTO session_members SELECT ?,s.id FROM submissions s "
                    "JOIN profiles p ON p.id=s.profile_id "
                    "WHERE p.model_id=? AND p.revision=? AND s.status='ready' AND s.block<=?",
                    (session_id, target.model_id, target.revision, cutoff),
                )
            return dict(conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone())

    def active_sessions(self) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.connection.execute("SELECT * FROM sessions WHERE state!='finished'")
        ]

    def session(self, session_id: str) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown session")
        return dict(row)

    def set_request(self, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        with self.transaction() as conn:
            row = self.session(session_id)
            if row["request"] is None:
                encoded = json.dumps(request, sort_keys=True, separators=(",", ":"))
                conn.execute("UPDATE sessions SET request=? WHERE id=?", (encoded, session_id))
                return request
            return dict(json.loads(row["request"]))

    def set_pod(self, session_id: str, pod_id: str) -> None:
        with self.transaction() as conn:
            prior = self.session(session_id)
            if prior["pod_id"] not in {None, pod_id}:
                raise ValueError("Session already owns a different pod")
            conn.execute("UPDATE sessions SET pod_id=? WHERE id=?", (pod_id, session_id))

    def anchor_sampling(self, session_id: str, finalized_height: int) -> None:
        """Observe fresh chain entropy only after membership has already frozen."""
        with self.transaction() as conn:
            session = self.session(session_id)
            if session["seed_block"] is not None:
                return
            target = self.profile(session["profile_id"])
            if (
                target.datasets is None
                or session["cutoff_block"] is None
                or finalized_height < session["cutoff_block"]
                or session["state"] != "active"
                or session["pod_id"] is not None
            ):
                raise ValueError("Invalid post-freeze sampling anchor")
            conn.execute(
                "UPDATE sessions SET cutoff_block=?,seed_block=? WHERE id=?",
                (finalized_height, finalized_height + target.datasets.seed_block_delay, session_id),
            )

    def set_sampling(self, session_id: str, context: RoundContext, plan_digest: str) -> None:
        with self.transaction() as conn:
            session = self.session(session_id)
            target = self.profile(session["profile_id"])
            if (
                target.datasets is None
                or context.cutoff_block != session["cutoff_block"]
                or context.seed_block != context.cutoff_block + target.datasets.seed_block_delay
                or context.seed_block != session["seed_block"]
                or len(plan_digest) != 64
                or any(c not in "0123456789abcdef" for c in plan_digest)
            ):
                raise ValueError("Invalid dataset sampling context")
            encoded = context.model_dump_json()
            if session["sampling"] is not None:
                if (session["sampling"], session["dataset_plan_digest"]) != (encoded, plan_digest):
                    raise ValueError("Round sampling is immutable")
                return
            if session["pod_id"] is not None or session["state"] != "active":
                raise ValueError("Sampling must be frozen before provisioning")
            conn.execute(
                "UPDATE sessions SET sampling=?,dataset_plan_digest=? WHERE id=?",
                (encoded, plan_digest, session_id),
            )

    def request_stop(self, session_id: str) -> None:
        self.connection.execute(
            "UPDATE sessions SET state='stopping' WHERE id=? AND state!='finished'", (session_id,)
        )

    def finish_session(self, session_id: str, now: float) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE submissions SET status='ready' WHERE id IN "
                "(SELECT submission_id FROM jobs WHERE session_id=? AND status='running')",
                (session_id,),
            )
            conn.execute(
                "UPDATE jobs SET status='abandoned' WHERE session_id=? AND status='running'",
                (session_id,),
            )
            conn.execute(
                "UPDATE sessions SET state='finished',finished_at=? WHERE id=?", (now, session_id)
            )

    def next_submission(self, session_id: str) -> dict[str, Any] | None:
        session = self.session(session_id)
        if session["cutoff_block"] is None:
            row = self._oldest_ready(self.profile(session["profile_id"]))
        else:
            row = self.connection.execute(
                "SELECT s.* FROM submissions s JOIN session_members m ON s.id=m.submission_id "
                "WHERE m.session_id=? AND s.status='ready' "
                "ORDER BY s.block,s.position,s.hotkey,s.id LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None or row["profile_id"] != session["profile_id"]:
            return None
        return {**dict(row), "commitment": Commitment.model_validate_json(row["commitment"])}

    def incumbent(self, target: TargetConfig) -> str | None:
        row = self.connection.execute(
            "SELECT artifact FROM incumbents WHERE target_key=?", (target_key(target),)
        ).fetchone()
        return str(row[0]) if row else None

    def running_job(self, session_id: str) -> EvaluationJob | None:
        row = self.connection.execute(
            "SELECT job FROM jobs WHERE session_id=? AND status='running'", (session_id,)
        ).fetchone()
        return EvaluationJob.model_validate_json(row[0]) if row else None

    def create_job(
        self,
        session_id: str,
        *,
        candidate_url: str,
        now: float,
        public_url: str | None = None,
        incumbent_url: str | None = None,
        qualification_url: str | None = None,
        dataset_plan_url: str | None = None,
        expected_dataset_selection: DatasetEvidence | None = None,
    ) -> EvaluationJob:
        with self.transaction() as conn:
            running = self.running_job(session_id)
            if running:
                return running
            session = self.session(session_id)
            if session["state"] != "active" or now >= session["deadline"]:
                raise ValueError("Session is not active")
            submission = self.next_submission(session_id)
            if submission is None:
                raise ValueError("Queue is empty")
            target = self.profile(session["profile_id"])
            context = (
                RoundContext.model_validate_json(session["sampling"])
                if session["sampling"]
                else None
            )
            if target.datasets is not None and (
                context is None or dataset_plan_url is None or expected_dataset_selection is None
            ):
                raise ValueError("Dataset round sampling is not prepared")
            job = EvaluationJob(
                job_id=uuid.uuid4().hex,
                session_id=session_id,
                target=target,
                commitment=submission["commitment"],
                incumbent_artifact=self.incumbent(target),
                candidate_url=candidate_url,
                public_url=public_url,
                incumbent_url=incumbent_url,
                qualification_url=qualification_url,
                deadline=min(session["deadline"], now + target.policy.max_job_seconds),
                round_context=context,
                dataset_plan_digest=session["dataset_plan_digest"],
                dataset_plan_url=dataset_plan_url,
                expected_dataset_selection=expected_dataset_selection,
            )
            conn.execute(
                "INSERT INTO jobs(id,session_id,submission_id,job,status) "
                "VALUES(?,?,?,?,'running')",
                (job.job_id, session_id, job.commitment.identity, job.model_dump_json()),
            )
            conn.execute(
                "UPDATE submissions SET status='running' WHERE id=?", (job.commitment.identity,)
            )
            return job

    def complete_job(
        self,
        job: EvaluationJob,
        result: EvaluationResult,
        decision: Decision,
        now: float,
    ) -> None:
        with self.transaction() as conn:
            stored = conn.execute("SELECT * FROM jobs WHERE id=?", (job.job_id,)).fetchone()
            if stored is None or EvaluationJob.model_validate_json(stored["job"]) != job:
                raise ValueError("Unknown or stale job")
            if stored["status"] != "running":
                return
            if self.incumbent(job.target) != job.incumbent_artifact:
                raise ValueError("Job incumbent is stale")
            if (
                result.job_id != job.job_id
                or result.session_id != job.session_id
                or result.submission_id != job.commitment.identity
                or result.competition_id != job.target.identity
            ):
                raise ValueError("Result identity is stale")
            if decision.outcome == "won" and (
                now > job.deadline or self.session(job.session_id)["state"] != "active"
            ):
                raise ValueError("Cannot promote an expired or stopped job")
            status = "ready" if decision.outcome == "inconclusive" else decision.outcome
            conn.execute(
                "UPDATE jobs SET status=?,result=?,decision=? WHERE id=?",
                (
                    decision.outcome,
                    result.model_dump_json(),
                    decision.model_dump_json(),
                    job.job_id,
                ),
            )
            conn.execute(
                "UPDATE submissions SET status=?,reason=?,score=? WHERE id=?",
                (status, decision.reason, decision.score, job.commitment.identity),
            )
            if decision.outcome == "won":
                award = Award(
                    submission_id=job.commitment.identity,
                    competition_id=job.target.identity,
                    hotkey=job.commitment.hotkey,
                    uid=job.commitment.uid,
                    promoted_at=now,
                    expires_at=now + job.target.policy.award_seconds,
                    artifact=job.commitment.artifact,
                )
                conn.execute(
                    "INSERT INTO awards(submission_id,target_key,award) VALUES(?,?,?)",
                    (award.submission_id, target_key(job.target), award.model_dump_json()),
                )
                conn.execute(
                    "INSERT INTO incumbents VALUES(?,?,?) ON CONFLICT(target_key) DO UPDATE "
                    "SET submission_id=excluded.submission_id,artifact=excluded.artifact",
                    (target_key(job.target), award.submission_id, award.artifact),
                )

    def awards(self, targets: tuple[TargetConfig, ...] | None = None) -> list[Award]:
        rows = self.connection.execute(
            "SELECT target_key,award FROM awards ORDER BY sequence"
        ).fetchall()
        if targets is None:
            return [Award.model_validate_json(row["award"]) for row in rows]
        # An unchanged target/revision retains succession across benchmark-profile changes.
        # Each challenge still measures that incumbent on its own frozen profile.
        current = {target_key(t): t.identity for t in targets}
        return [
            Award.model_validate_json(r["award"]).model_copy(
                update={"competition_id": current[r["target_key"]]}
            )
            for r in rows
            if r["target_key"] in current
        ]

    def publications(self) -> dict[str, WinnerRelease]:
        return {
            row["submission_id"]: WinnerRelease.model_validate_json(row["release"])
            for row in self.connection.execute("SELECT * FROM publications")
        }

    def registered_profiles(self) -> dict[str, TargetConfig]:
        return {
            r["id"]: TargetConfig.model_validate_json(r["config"])
            for r in self.connection.execute("SELECT id,config FROM profiles")
        }

    def needs_similarity(self, submission_id: str) -> bool:
        row = self.connection.execute(
            "SELECT status FROM submissions WHERE id=?", (submission_id,)
        ).fetchone()
        return row is not None and row[0] == "pending_similarity"

    def fingerprint_bank(self) -> dict[str, Fingerprint]:
        return {
            r["artifact"]: Fingerprint.model_validate_json(r["fingerprint"])
            for r in self.connection.execute("SELECT artifact,fingerprint FROM fingerprints")
        }

    def comparison_sources(self) -> dict[str, TargetConfig]:
        return {
            Commitment.model_validate_json(
                r["commitment"]
            ).artifact: TargetConfig.model_validate_json(r["config"])
            for r in self.connection.execute(
                "SELECT s.commitment,p.config FROM submissions s JOIN profiles p "
                "ON p.id=s.profile_id WHERE s.status IN ('won','lost','ready','running')"
            )
        }

    def store_fingerprint(self, artifact: str, fingerprint: Fingerprint) -> None:
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT fingerprint FROM fingerprints WHERE scope=? AND artifact=?",
                (fingerprint.scope, artifact),
            ).fetchone()
            if existing and Fingerprint.model_validate_json(existing[0]) != fingerprint:
                raise ValueError("Fingerprint identity is immutable")
            conn.execute(
                "INSERT OR IGNORE INTO fingerprints VALUES(?,?,?,NULL)",
                (fingerprint.scope, artifact, fingerprint.model_dump_json()),
            )

    def similarity_verdict(self, submission_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT verdict FROM similarity_verdicts WHERE submission_id=?", (submission_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def record_publication(self, release: WinnerRelease) -> None:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT award FROM awards WHERE submission_id=?",
                (release.submission_id,),
            ).fetchone()
            if row is None or Award.model_validate_json(row[0]).artifact != release.source_artifact:
                raise ValueError("Only promoted matching winner artifacts may be published")
            previous = conn.execute(
                "SELECT release FROM publications WHERE submission_id=?",
                (release.submission_id,),
            ).fetchone()
            if previous:
                old = WinnerRelease.model_validate_json(previous[0])
                if old.model_dump(exclude={"published_at"}) != release.model_dump(
                    exclude={"published_at"}
                ):
                    raise ValueError("Winner publication identity is immutable")
                return
            conn.execute(
                "INSERT INTO publications VALUES(?,?)",
                (release.submission_id, release.model_dump_json()),
            )

    def submissions(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM submissions ORDER BY block,position,hotkey,id"
            )
        ]

    def evaluations(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM jobs WHERE decision IS NOT NULL ORDER BY rowid"
            )
        ]

    def metadata(self, key: str, default: str = "") -> str:
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else default

    def pending_profiles(self, enabled: tuple[TargetConfig, ...]) -> list[TargetConfig]:
        keys = {target_key(t) for t in enabled}
        rows = self.connection.execute(
            "SELECT p.config,MIN(s.block) AS first_block FROM profiles p JOIN submissions s "
            "ON p.id=s.profile_id WHERE s.status='ready' GROUP BY p.id ORDER BY first_block,p.id"
        )
        return [
            t for row in rows if target_key(t := TargetConfig.model_validate_json(row[0])) in keys
        ]

    def set_metadata(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO metadata VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
