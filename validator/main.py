from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import signal
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from evaluation.config import LIUM_API_KEY
from shared.artifacts import ArtifactStore, extract_package
from shared.checkpoints import sha256_file
from shared.contracts import Qualification
from shared.targets import load_competitions
from validator.admission import AdmissionService
from validator.chain import BittensorChain
from validator.config import (
    ARTIFACT_ROOT,
    NETUID,
    NETWORK,
    POLL_SECONDS,
    PROVIDER,
    PUBLIC_ARTIFACT_BASE_URL,
    PUBLIC_ARTIFACT_ROOT,
    S3_ACCESS_KEY_ID,
    S3_SECRET_ACCESS_KEY,
    SESSION_MAX_COST,
    STATE_PATH,
    UPLOAD_STAGING_ROOT,
)
from validator.controller import Controller
from validator.providers import Provider
from validator.publication import PublicationService
from validator.state import State


def main() -> None:
    parser = argparse.ArgumentParser(description="Prexon validator controller")
    parser.add_argument("--once", action="store_true", help="Run one controller tick")
    parser.add_argument(
        "--qualification-profile",
        metavar="MODEL_ID",
        help="Print the qualification ID of a competition in competitions/",
    )
    parser.add_argument("--publish-qualification", type=Path)
    parser.add_argument(
        "--publish-reference",
        type=Path,
        metavar="PACKAGE",
        help="Store a public reference package built for a competition; prints its SHA-256",
    )
    parser.add_argument(
        "--check-config", action="store_true", help="List the competitions in competitions/"
    )
    parser.add_argument(
        "--requeue-stalled",
        action="store_true",
        help="Return submissions set aside after repeated failed evaluations to the queue",
    )
    args = parser.parse_args()
    competitions = load_competitions()
    # Only active competitions are evaluated.
    targets = tuple(c for c in competitions if c.active)
    if args.qualification_profile:
        target = next((c for c in competitions if c.model_id == args.qualification_profile), None)
        if target is None:
            parser.error(f"No competition for {args.qualification_profile} in competitions/")
        print(target.qualification_identity)
        return
    if args.publish_qualification:
        record = Qualification.model_validate_json(args.publish_qualification.read_bytes())
        if not any(
            t.qualification_identity == record.profile_id
            and t.engine_version == record.engine_version
            for t in competitions
        ):
            parser.error("Qualification does not cover a competition in competitions/")
        store = ArtifactStore(
            ARTIFACT_ROOT,
            access_key=S3_ACCESS_KEY_ID,
            secret_key=S3_SECRET_ACCESS_KEY,
        )
        print(store.put_file(args.publish_qualification))
        return
    if args.publish_reference:
        digest = sha256_file(args.publish_reference)
        limit = max((c.policy.max_checkpoint_bytes for c in competitions), default=0)
        # Unpacking checks the archive and its files the way admission will.
        with tempfile.TemporaryDirectory() as scratch:
            manifest = extract_package(
                args.publish_reference, Path(scratch) / "head", digest, limit
            )
        if not any(
            manifest.target_id == c.model_id
            and manifest.target_revision == c.revision
            and manifest.method in c.methods
            for c in competitions
        ):
            parser.error("Package is not built for a competition in competitions/")
        store = ArtifactStore(
            ARTIFACT_ROOT,
            access_key=S3_ACCESS_KEY_ID,
            secret_key=S3_SECRET_ACCESS_KEY,
        )
        print(store.put_file(args.publish_reference))
        return
    if args.check_config:
        print(
            json.dumps(
                {
                    "competitions": [
                        {"model_id": c.model_id, "active": c.active, "allocation": c.allocation}
                        for c in competitions
                    ]
                },
                indent=2,
            )
        )
        return
    os.umask(0o077)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.with_suffix(".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("Another controller owns this state file") from exc
        state = State(STATE_PATH)
        admission: AdmissionService | None = None
        publications: PublicationService | None = None
        controller: Controller | None = None
        try:
            if args.requeue_stalled:
                print(f"Requeued {state.requeue_stalled()} stalled submission(s)")
                return
            has_chain = bool(targets or state.metadata("chain_context"))
            if has_chain and NETUID == 0:
                raise ValueError("Set PREXON_NETUID in .env before enabling models")
            context = state.metadata("chain_context")
            if context and json.loads(context) != {"network": NETWORK, "netuid": NETUID}:
                raise ValueError(
                    "Retain the initialized chain/subnet settings when disabling targets"
                )
            if targets:
                roots = (UPLOAD_STAGING_ROOT, ARTIFACT_ROOT, PUBLIC_ARTIFACT_ROOT)
                if any(urlsplit(root).scheme != "s3" for root in roots):
                    raise ValueError("Live targets require three separate Hippius S3 stores")
                if not PUBLIC_ARTIFACT_BASE_URL:
                    raise ValueError("Live targets require a public Hippius release base URL")
            store = ArtifactStore(
                ARTIFACT_ROOT,
                access_key=S3_ACCESS_KEY_ID,
                secret_key=S3_SECRET_ACCESS_KEY,
            )
            provider: Provider | None = None
            if targets or state.active_sessions():
                if PROVIDER == "lium":
                    from validator.providers.lium import LiumProvider

                    if not store.bucket:
                        raise ValueError("Lium workers require S3-compatible artifact storage")
                    provider = LiumProvider(LIUM_API_KEY, SESSION_MAX_COST)
                else:
                    from validator.providers.local import LocalProvider

                    provider = LocalProvider()
            admission = None if args.once else AdmissionService()
            publications = None if args.once else PublicationService()
            controller = Controller(
                targets,
                state,
                store,
                BittensorChain(),
                provider,
                admission=admission,
                publications=publications,
            )
            stop = threading.Event()
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: stop.set())
            while not stop.is_set():
                controller.tick(time.time())
                if args.once:
                    break
                stop.wait(POLL_SECONDS)
        finally:
            if controller is not None and not args.once:
                for session in state.active_sessions():
                    try:
                        controller.stop_session(session, time.time())
                    except Exception as exc:
                        logging.error(
                            "shutdown_reconciliation_pending error=%s", type(exc).__name__
                        )
            try:
                if admission is not None:
                    admission.close()
                    admission.pump(state, store, targets, time.time())
                if publications is not None and controller is not None:
                    publications.close()
                    publications.pump(
                        state,
                        store,
                        controller.public_store,
                        PUBLIC_ARTIFACT_BASE_URL,
                        time.time(),
                    )
            finally:
                state.close()


if __name__ == "__main__":
    main()
