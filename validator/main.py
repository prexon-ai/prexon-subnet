from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from shared.artifacts import ArtifactStore
from shared.config import load_settings, read_environment
from shared.contracts import Qualification
from validator.admission import AdmissionService
from validator.chain import BittensorChain, Chain, FileChain
from validator.controller import Controller
from validator.providers import Provider
from validator.publication import PublicationService
from validator.state import State


def main() -> None:
    parser = argparse.ArgumentParser(description="Prexon validator controller")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--once", action="store_true", help="Run one controller tick")
    parser.add_argument("--qualification-profile", type=int, choices=[1, 2])
    parser.add_argument("--publish-qualification", type=Path)
    parser.add_argument(
        "--check-config", action="store_true", help="Validate configuration without external calls"
    )
    args = parser.parse_args()
    settings = load_settings("validator", args.env_file)
    if args.qualification_profile:
        target = next((t for t in settings.targets if t.slot == args.qualification_profile), None)
        if target is None:
            parser.error("Selected target slot is disabled")
        print(target.qualification_identity)
        return
    if args.publish_qualification:
        record = Qualification.model_validate_json(args.publish_qualification.read_bytes())
        if not any(
            t.qualification_identity == record.profile_id
            and t.engine_version == record.engine_version
            for t in settings.targets
        ):
            parser.error("Qualification does not cover a configured target profile")
        store = ArtifactStore(settings.artifact_root, environment=read_environment(args.env_file))
        print(store.put_file(args.publish_qualification))
        return
    if args.check_config:
        print(
            json.dumps(
                {
                    "targets": [
                        {"slot": t.slot, "model_id": t.model_id, "competition_id": t.identity}
                        for t in settings.targets
                    ]
                },
                indent=2,
            )
        )
        return
    os.umask(0o077)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings.state_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.state_path.with_suffix(".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("Another controller owns this state file") from exc
        state = State(settings.state_path)
        admission: AdmissionService | None = None
        publications: PublicationService | None = None
        controller: Controller | None = None
        try:
            has_chain = bool(settings.targets or state.metadata("chain_context"))
            if has_chain and settings.chain == "bittensor" and settings.netuid == 0:
                raise ValueError("Set PREXON_NETUID for the subnet before enabling models")
            context = state.metadata("chain_context")
            if context and json.loads(context) != {
                "network": settings.network,
                "netuid": settings.netuid,
            }:
                raise ValueError(
                    "Retain the initialized chain/subnet settings when disabling targets"
                )
            env = read_environment(args.env_file)
            if settings.targets and settings.chain == "bittensor":
                roots = (
                    settings.upload_staging_root,
                    settings.artifact_root,
                    settings.public_artifact_root,
                )
                if any(urlsplit(root).scheme != "s3" for root in roots):
                    raise ValueError("Live targets require three separate Hippius S3 stores")
                if not settings.public_artifact_base_url:
                    raise ValueError("Live targets require a public Hippius release base URL")
            store = ArtifactStore(settings.artifact_root, environment=env)
            provider: Provider | None = None
            if settings.targets or state.active_sessions():
                if settings.provider == "lium":
                    from validator.providers.lium import LiumProvider

                    if not store.bucket:
                        raise ValueError("Lium workers require S3-compatible artifact storage")
                    provider = LiumProvider(settings, env.get("LIUM_API_KEY", ""))
                else:
                    from validator.providers.local import LocalProvider

                    provider = LocalProvider(settings)
            chain: Chain = (
                BittensorChain(settings)
                if has_chain and settings.chain == "bittensor"
                else FileChain(settings.chain_file)
            )
            admission = None if args.once else AdmissionService()
            publications = None if args.once else PublicationService()
            controller = Controller(
                settings,
                state,
                store,
                chain,
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
                stop.wait(settings.poll_seconds)
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
                    admission.pump(state, store, settings, time.time())
                if publications is not None and controller is not None:
                    publications.close()
                    publications.pump(
                        state,
                        store,
                        controller.public_store,
                        settings.public_artifact_base_url,
                        time.time(),
                    )
            finally:
                state.close()


if __name__ == "__main__":
    main()
