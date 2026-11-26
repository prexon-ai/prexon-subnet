"""Always-on CPU orchestration, independent of dashboard availability."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from typing import TYPE_CHECKING, Any

from evaluation.config import DATASET_CACHE
from evaluation.datasets.sampling import select_round, validate_plan
from evaluation.lium_api import FAILED_POD_STATUSES, RentalRefused
from shared.artifacts import ArtifactStore
from shared.contracts import EvaluationResult, Qualification
from shared.datasets import RoundContext, RoundPlan
from shared.targets import TargetConfig
from shared.uploads import UploadPolicy, UploadTarget
from validator.chain import Chain, ChainSnapshot
from validator.competition import emission_weights, score_evidence
from validator.config import (
    BURN_UID,
    MAX_EVALUATION_ATTEMPTS,
    NETUID,
    NETWORK,
    PUBLIC_ARTIFACT_BASE_URL,
    PUBLIC_ARTIFACT_ROOT,
    PUBLISH_SECONDS,
    S3_ACCESS_KEY_ID,
    S3_SECRET_ACCESS_KEY,
    S3_SESSION_TOKEN,
    SESSION_MAX_COST,
    UPLOAD_POLICY_PATH,
)
from validator.dashboard import publish_snapshot
from validator.history import publish_history
from validator.providers import Provider
from validator.publication import publish_winners
from validator.scheduler import claim_due_sessions
from validator.state import State
from validator.submissions import admit_commitment

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from validator.admission import AdmissionService
    from validator.publication import PublicationService


class Controller:
    def __init__(
        self,
        targets: tuple[TargetConfig, ...],
        state: State,
        store: ArtifactStore,
        chain: Chain,
        provider: Provider | None,
        *,
        admission: AdmissionService | None = None,
        publications: PublicationService | None = None,
    ) -> None:
        self.targets, self.state, self.store = targets, state, store
        self.chain, self.provider = chain, provider
        self.admission, self.publications = admission, publications
        self.public_store = ArtifactStore(
            PUBLIC_ARTIFACT_ROOT,
            access_key=S3_ACCESS_KEY_ID,
            secret_key=S3_SECRET_ACCESS_KEY,
            session_token=S3_SESSION_TOKEN,
        )
        self.snapshot = ChainSnapshot(block=0, commitments=(), owners={})
        if targets:
            context = json.dumps({"network": NETWORK, "netuid": NETUID}, sort_keys=True)
            previous = state.metadata("chain_context")
            if previous and previous != context:
                raise ValueError("Use separate validator state for a different chain/subnet")
            state.set_metadata("chain_context", context)
        for target in targets:
            state.register_target(target)

    def _qualified(self, target: TargetConfig) -> bool:
        if target.qualification_artifact is None:
            return False
        data = self.store.read_bytes(f"objects/{target.qualification_artifact}", 1024**2)
        q = Qualification.model_validate_json(data)
        return (
            hashlib.sha256(data).hexdigest() == target.qualification_artifact
            and q.profile_id == target.qualification_identity
            and q.engine_version == target.engine_version
        )

    def tick(self, now: float) -> None:
        scan_ok = False
        self.state.touch_models(self.targets, now)
        if self.targets or self.state.metadata("chain_initialized"):
            try:
                cursor = self.state.metadata("finalized_block")
                self.snapshot = self.chain.scan(
                    int(cursor) if cursor else None, self.state.known_commitments()
                )
                self.state.discover(self.snapshot.commitments, self.snapshot.block, now)
                self.state.set_metadata("chain_initialized", "1")
                scan_ok = True
            except Exception as exc:
                log.error("chain_scan_failed error=%s", type(exc).__name__)
            if self.admission is not None:
                self.admission.pump(self.state, self.store, now)
            else:
                for commitment in self.state.pending_admissions():
                    try:
                        admit_commitment(self.state, self.store, commitment, now)
                    except Exception as exc:
                        log.error(
                            "admission_retry submission=%s error=%s",
                            commitment.identity,
                            type(exc).__name__,
                        )
        # Reconcile every recorded session, including sessions whose target was just disabled.
        for session in self.state.active_sessions():
            try:
                self._advance(session, now)
            except Exception as exc:
                log.error("session_retry session=%s error=%s", session["id"], type(exc).__name__)
        qualified = []
        enabled = {t.identity: t for t in self.targets}
        profiles = {
            **enabled,
            **{t.identity: t for t in self.state.pending_profiles(self.targets)},
        }
        upload_targets = []
        for profile in profiles.values():
            try:
                caught_up = self.snapshot.finalized_block in {None, self.snapshot.block}
                if self._qualified(profile):
                    if profile.identity in enabled:
                        upload_targets.append(
                            UploadTarget(
                                competition_id=profile.identity,
                                max_package_bytes=profile.policy.max_checkpoint_bytes + 2 * 1024**2,
                            )
                        )
                    if profile.datasets is None or scan_ok and caught_up:
                        qualified.append(profile)
            except Exception as exc:
                log.error(
                    "qualification_unavailable profile=%s error=%s",
                    profile.identity,
                    type(exc).__name__,
                )
        if self.provider is not None:
            # Scheduler uses the original per-target allocations and immutable queued profiles.
            for session in claim_due_sessions(
                self.state, self.targets, now, {t.identity for t in qualified}
            ):
                try:
                    self._advance(session, now)
                except Exception as exc:
                    log.error(
                        "session_start_retry session=%s error=%s", session["id"], type(exc).__name__
                    )
        owners: dict[int, str] = {}
        if scan_ok:
            try:
                owners = self.chain.current_owners()
            except Exception as exc:
                scan_ok = False
                log.error("ownership_refresh_failed error=%s", type(exc).__name__)
        if scan_ok or not self.targets:
            self._publish_upload_policy(now, owners, tuple(upload_targets) if scan_ok else ())
        weights = emission_weights(
            self.state.awards(self.targets),
            self.targets,
            owners,
            now,
            BURN_UID,
        )
        last = float(self.state.metadata("last_weight_publication", "0"))
        if scan_ok and now - last >= PUBLISH_SECONDS:
            try:
                self.chain.publish(weights)
                self.state.set_metadata("last_weight_publication", str(now))
            except Exception as exc:
                log.error("weight_publication_retry error=%s", type(exc).__name__)
        try:
            if self.publications is None:
                publish_winners(
                    self.state,
                    self.store,
                    self.public_store,
                    PUBLIC_ARTIFACT_BASE_URL,
                    now,
                )
            else:
                self.publications.pump(
                    self.state,
                    self.store,
                    self.public_store,
                    PUBLIC_ARTIFACT_BASE_URL,
                    now,
                )
        except Exception as exc:
            log.error("winner_release_retry error=%s", type(exc).__name__)
        # Rewards depend on who owns each UID. Without a fresh chain read the last
        # published files stay, rather than showing every reward as gone.
        if scan_ok or not self.targets:
            public = (self.state, self.public_store, self.targets, owners)
            try:
                publish_snapshot(*public, weights, now, self.snapshot.block)
                publish_history(*public, now, self.snapshot.block)
            except Exception as exc:
                log.error("dashboard_publication_retry error=%s", type(exc).__name__)

    def _publish_upload_policy(
        self,
        now: float,
        owners: dict[int, str],
        targets: tuple[UploadTarget, ...],
    ) -> None:
        policy = UploadPolicy(
            updated_at=now,
            network=NETWORK,
            netuid=NETUID,
            hotkeys=tuple(sorted(set(owners.values()))),
            targets=targets,
        )
        path = UPLOAD_POLICY_PATH
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as temp:
                temporary = temp.name
                temp.write(policy.model_dump_json() + "\n")
            os.chmod(temporary, 0o640)
            os.replace(temporary, path)
        except OSError as exc:
            log.error("upload_policy_publication_retry error=%s", type(exc).__name__)

    def stop_session(self, session: dict[str, Any], now: float) -> None:
        if self.provider is None:
            raise RuntimeError("A provider is required to reconcile recorded rentals")
        self.state.request_stop(session["id"])
        if session["pod_id"]:
            self.provider.delete(session["pod_id"])
        elif session["request"]:
            recover = getattr(self.provider, "recover", None)
            pod_id = recover(session["id"]) if recover else None
            if pod_id:
                self.state.set_pod(session["id"], pod_id)
                self.provider.delete(pod_id)
            else:
                hours = json.loads(session["request"]).get("termination_hours", 1)
                if now < session["deadline"] + hours * 3600 + 60:
                    return  # Keep ownership until an ambiguous rental's provider TTL has elapsed.
        self.state.finish_session(session["id"], now)

    def _advance(self, session: dict[str, Any], now: float) -> None:
        if self.provider is None:
            raise RuntimeError("Provider unavailable")
        target = self.state.profile(session["profile_id"])
        enabled = any(
            t.model_id == target.model_id and t.revision == target.revision for t in self.targets
        )
        if session["state"] == "stopping" or now >= session["deadline"] or not enabled:
            self.stop_session(session, now)
            return
        if not self._qualified(target):
            self.stop_session(session, now)
            return
        if target.datasets is not None and session["sampling"] is None:
            if session["seed_block"] is None:
                # Membership is already durable. The fresh head is observed AFTER that freeze,
                # so admissions/qualification cannot make the entropy block a known past block.
                self.state.anchor_sampling(session["id"], self.chain.finalized_height())
                session = self.state.session(session["id"])
            cutoff = session["cutoff_block"]
            if cutoff is None:
                raise ValueError("Dataset session has no frozen submission cutoff")
            seed_block = session["seed_block"]
            if self.snapshot.block < seed_block:
                return
            context = RoundContext(
                cutoff_block=cutoff,
                seed_block=seed_block,
                block_hash=self.chain.block_hash(seed_block),
                chain=NETWORK,
                netuid=NETUID,
            )
            plan = select_round(target, DATASET_CACHE, context.seed(target.identity))
            payload = json.dumps(
                plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
            digest = self.store.put_bytes(payload)
            self.state.set_sampling(session["id"], context, digest)
            session = self.state.session(session["id"])
        if session["request"] is None:
            request = self.provider.build_request(session, target)
            self.state.set_request(session["id"], request)
            session = self.state.session(session["id"])
        if session["pod_id"] is None:
            try:
                pod_id = self.provider.rent(session["id"], json.loads(session["request"]))
            except RentalRefused as exc:
                # Nothing was rented, so there is no pod to keep ownership of. The provider
                # replays a refusal for this identity; a later session retries with a new one.
                log.warning(
                    "rental_refused session=%s reason=%s", session["id"], exc.code or exc.status
                )
                self.state.finish_session(session["id"], now)
                return
            self.state.set_pod(session["id"], pod_id)
            session = self.state.session(session["id"])
        pod = self.provider.status(session["pod_id"])
        request = json.loads(session["request"])
        ceiling = request.get("max_price_per_gpu_hour", target.hardware.max_price_per_gpu_hour)
        prices = (
            float(price)
            for field in ("price", "pending_price")
            if (price := pod.get(field)) is not None
        )
        if any(
            not math.isfinite(price) or price < 0 or price > ceiling * target.hardware.gpu_count
            for price in prices
        ):
            self.stop_session(session, now)
            return
        spend = float(pod.get("spend_to_date", 0))
        if not math.isfinite(spend) or spend < 0 or spend >= SESSION_MAX_COST:
            self.stop_session(session, now)
            return
        status = str(pod.get("status", "")).lower()
        if status in FAILED_POD_STATUSES:
            self.stop_session(session, now)
            return
        if status not in {"running", "ready", "active"}:
            return
        job = self.state.running_job(session["id"])
        if job is not None:
            result_key = f"jobs/{job.job_id}/result.json"
            # A result is read even after the deadline: scoring checks when it finished.
            if self.store.exists(result_key):
                try:
                    result = EvaluationResult.model_validate_json(self.store.read_bytes(result_key))
                    decision = score_evidence(job, result)
                    self.store.write_bytes(
                        f"reports/{job.job_id}.json", result.model_dump_json().encode()
                    )
                    outcome = self.state.complete_job(
                        job,
                        result,
                        decision,
                        now,
                        max_attempts=MAX_EVALUATION_ATTEMPTS,
                    )
                except ValueError:
                    # Once the budget is spent, an unusable result is the same as none.
                    if now < job.deadline:
                        raise
                else:
                    if outcome == "stalled":
                        log.warning("submission_stalled submission=%s", job.commitment.identity)
                    if decision.outcome == "inconclusive":
                        self.stop_session(session, now)
                    return
            if now >= job.deadline:
                # The whole job budget passed without a result: one failed attempt.
                outcome = self.state.timeout_job(job, max_attempts=MAX_EVALUATION_ATTEMPTS)
                if outcome == "stalled":
                    log.warning("submission_stalled submission=%s", job.commitment.identity)
                self.stop_session(session, now)
                return
        else:
            submission = self.state.next_submission(session["id"])
            if submission is None:
                self.stop_session(session, now)
                return
            expiry = max(60, min(7 * 86400, int(session["deadline"] - now + 60)))
            incumbent = self.state.incumbent(target)
            public = target.public_reference.artifact if target.public_reference else None
            expected_selection = None
            if target.datasets is not None:
                payload = self.store.read_bytes(
                    f"objects/{session['dataset_plan_digest']}",
                    max_bytes=64 * 1024**2,
                )
                if hashlib.sha256(payload).hexdigest() != session["dataset_plan_digest"]:
                    raise ValueError("Stored dataset plan digest mismatch")
                plan = RoundPlan.model_validate_json(payload)
                context = RoundContext.model_validate_json(session["sampling"])
                validate_plan(target, plan, context.seed(target.identity))
                if plan.identity != session["dataset_plan_digest"]:
                    raise ValueError("Stored dataset plan identity mismatch")
                expected_selection = plan.evidence()
            job = self.state.create_job(
                session["id"],
                now=now,
                candidate_url=self.store.url(
                    f"objects/{submission['commitment'].artifact}", expires=expiry
                ),
                public_url=self.store.url(f"objects/{public}", expires=expiry) if public else None,
                incumbent_url=self.store.url(f"objects/{incumbent}", expires=expiry)
                if incumbent
                else None,
                qualification_url=self.store.url(
                    f"objects/{target.qualification_artifact}", expires=expiry
                ),
                dataset_plan_url=self.store.url(
                    f"objects/{session['dataset_plan_digest']}", expires=expiry
                )
                if session["dataset_plan_digest"]
                else None,
                expected_dataset_selection=expected_selection,
            )
        # This also repairs a crash after the SQLite job commit but before object upload.
        self.store.write_bytes(f"jobs/{job.job_id}/job.json", job.model_dump_json().encode())
        expiry = max(60, min(7 * 86400, int(job.deadline - now + 60)))
        self.provider.launch(
            session["pod_id"],
            job,
            self.store.url(f"jobs/{job.job_id}/job.json", expires=expiry),
            self.store.url(f"jobs/{job.job_id}/result.json", write=True, expires=expiry),
        )
