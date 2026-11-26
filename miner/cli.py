from __future__ import annotations

import argparse
import json
from pathlib import Path

from miner.config import NETWORK, UPLOAD_API_URL, WALLET_HOTKEY, WALLET_NAME, WALLET_PATH
from miner.target import MinerTarget
from shared.artifacts import package_checkpoint
from shared.config import UPLOAD_MAX_BYTES
from shared.contracts import Provenance
from shared.targets import TargetConfig


def _common_options() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument("--api-url", help=f"API base URL (default: {UPLOAD_API_URL})")
    common.add_argument(
        "--network", choices=["finney", "test"], help=f"Bittensor network (default: {NETWORK})"
    )
    common.add_argument("--netuid", type=int, help="The Prexon subnet ID")
    common.add_argument("--wallet-name", help=f"Default: {WALLET_NAME}")
    common.add_argument("--wallet-hotkey", help=f"Default: {WALLET_HOTKEY}")
    common.add_argument("--wallet-path", help=f"Default: {WALLET_PATH}")
    common.add_argument("--max-upload-bytes", type=int, help="Compressed package byte limit")
    return common


def _target_options(command: argparse.ArgumentParser, *, identity: bool = True) -> None:
    command.add_argument("--profile", type=Path, help="Optional full target benchmark profile")
    command.add_argument(
        "--slot", type=int, choices=[1, 2], help="Slot recorded for --profile (default: 1)"
    )
    command.add_argument("--max-checkpoint-bytes", type=int, help="Local checkpoint byte limit")
    if identity:
        command.add_argument("--model-id")
        command.add_argument("--revision", help="Exact 40-character target checkpoint revision")
        command.add_argument("--competition-id", help="Published validator competition digest")


def _targets(args: argparse.Namespace) -> tuple[TargetConfig, ...]:
    """The target of --profile, with --model-id and --revision applied; none without it."""
    profile_path = getattr(args, "profile", None)
    if profile_path is None:
        return ()
    profile = json.loads(profile_path.read_text())
    if not isinstance(profile, dict):
        raise ValueError("Target profile must contain a JSON object")
    profile.setdefault("slot", getattr(args, "slot", None) or 1)
    profile.setdefault("allocation", 100)
    for field in ("model_id", "revision"):
        value = getattr(args, field, None)
        if value is not None:
            profile[field] = value
    return (TargetConfig.model_validate(profile),)


def _target(args: argparse.Namespace, targets: tuple[TargetConfig, ...]) -> MinerTarget:
    if targets:
        data = MinerTarget.from_profile(targets[0]).model_dump()
    else:
        if not args.model_id or not args.revision:
            raise ValueError("Provide --model-id and --revision, or --profile")
        data = {"model_id": args.model_id, "revision": args.revision}
    if args.competition_id is not None:
        data["competition_id"] = args.competition_id
    if args.max_checkpoint_bytes is not None:
        data["max_checkpoint_bytes"] = args.max_checkpoint_bytes
    return MinerTarget.model_validate(data)


def main() -> None:
    common = _common_options()
    parser = argparse.ArgumentParser(
        description="Prexon drafter training, packaging, and submission", parents=[common]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "package"):
        command = commands.add_parser(name, parents=[common])
        _target_options(command)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        if name == "train":
            command.add_argument("--recipe", type=Path, required=True)
            command.add_argument("--run-dir", type=Path, required=True)
            command.add_argument("--specforge-version", default="0.2.0")
        else:
            command.add_argument("--provenance", type=Path, required=True)
    send = commands.add_parser("submit", parents=[common])
    _target_options(send, identity=False)
    send.add_argument("package", type=Path)
    send.add_argument("--upload-only", action="store_true")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--checkpoint", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    listing = commands.add_parser("targets", parents=[common])
    _target_options(listing)
    args = parser.parse_args()
    if args.command == "inspect":
        from shared.checkpoints import checkpoint_schema

        schema = checkpoint_schema(args.checkpoint)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(schema.model_dump_json() + "\n")
        return
    try:
        targets = _targets(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.command == "targets":
        if any(
            getattr(args, option) is not None
            for option in (
                "profile",
                "model_id",
                "revision",
                "competition_id",
                "slot",
                "max_checkpoint_bytes",
            )
        ):
            try:
                selected = _target(args, targets)
            except (ValueError, OSError) as exc:
                parser.error(str(exc))
            record = selected.model_dump(mode="json")
            record["slot"] = args.slot or (targets[0].slot if targets else 1)
            print(json.dumps([record], indent=2))
            return
        print(
            json.dumps(
                [
                    {"slot": t.slot, "model_id": t.model_id, "competition_id": t.identity}
                    for t in targets
                ],
                indent=2,
            )
        )
        return
    if args.command == "submit":
        from miner.submit import submit

        print(
            submit(
                args.package,
                netuid=getattr(args, "netuid", 0),
                network=getattr(args, "network", NETWORK),
                api_url=getattr(args, "api_url", UPLOAD_API_URL),
                wallet_name=getattr(args, "wallet_name", WALLET_NAME),
                wallet_hotkey=getattr(args, "wallet_hotkey", WALLET_HOTKEY),
                wallet_path=getattr(args, "wallet_path", WALLET_PATH),
                targets=targets,
                max_upload_bytes=getattr(args, "max_upload_bytes", UPLOAD_MAX_BYTES),
                publish=not args.upload_only,
                max_checkpoint_bytes=args.max_checkpoint_bytes,
            )
        )
        return
    try:
        target = _target(args, targets)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.command == "train":
        from miner.train import train_and_export

        train_and_export(
            target,
            args.recipe,
            args.run_dir,
            args.checkpoint,
            args.output,
            specforge_version=args.specforge_version,
        )
    elif args.command == "package":
        if target.competition_id is None:
            parser.error("Packaging requires --competition-id or a full target profile")
        provenance = Provenance.model_validate_json(args.provenance.read_bytes())
        package_checkpoint(
            args.checkpoint,
            args.output,
            target_id=target.model_id,
            target_revision=target.revision,
            competition_id=target.competition_id,
            provenance=provenance,
            max_bytes=target.max_checkpoint_bytes,
        )


if __name__ == "__main__":
    main()
