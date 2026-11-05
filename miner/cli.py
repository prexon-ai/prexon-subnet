from __future__ import annotations

import argparse
import json
from pathlib import Path

from miner.config import MinerTarget
from shared.artifacts import package_checkpoint
from shared.config import Settings, TargetConfig, load_settings
from shared.contracts import Provenance


def _common_options() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument("--env-file", type=Path, help="Optional miner dotenv file")
    common.add_argument(
        "--api-url",
        help=f"API base URL (default: {Settings.model_fields['upload_api_url'].default})",
    )
    common.add_argument("--network", help="Bittensor network or RPC endpoint")
    common.add_argument("--netuid", type=int)
    common.add_argument("--wallet-name")
    common.add_argument("--wallet-hotkey")
    common.add_argument("--wallet-path")
    common.add_argument("--max-upload-bytes", type=int, help="Compressed package byte limit")
    return common


def _target_options(command: argparse.ArgumentParser, *, identity: bool = True) -> None:
    command.add_argument("--profile", type=Path, help="Optional full target benchmark profile")
    command.add_argument(
        "--slot", type=int, choices=[1, 2], help="Target slot in an optional dotenv"
    )
    command.add_argument("--max-checkpoint-bytes", type=int, help="Local checkpoint byte limit")
    if identity:
        command.add_argument("--model-id")
        command.add_argument("--revision", help="Exact 40-character target checkpoint revision")
        command.add_argument("--competition-id", help="Published validator competition digest")


def _settings(args: argparse.Namespace) -> Settings:
    env_file = getattr(args, "env_file", None)
    if env_file is not None and not env_file.is_file():
        raise ValueError(f"Miner dotenv file does not exist: {env_file}")
    direct = bool(
        getattr(args, "profile", None)
        or getattr(args, "model_id", None)
        or getattr(args, "revision", None)
    )
    overrides = {}
    for option, field in {
        "api_url": "upload_api_url",
        "network": "network",
        "netuid": "netuid",
        "wallet_name": "wallet_name",
        "wallet_hotkey": "wallet_hotkey",
        "wallet_path": "wallet_path",
        "max_upload_bytes": "upload_max_bytes",
    }.items():
        if hasattr(args, option):
            overrides[field] = getattr(args, option)
    settings = load_settings("miner", env_file, load_targets=not direct, overrides=overrides)
    profile_path = getattr(args, "profile", None)
    if profile_path is not None:
        profile = json.loads(profile_path.read_text())
        if not isinstance(profile, dict):
            raise ValueError("Target profile must contain a JSON object")
        profile.setdefault("slot", getattr(args, "slot", None) or 1)
        profile.setdefault("allocation", 100)
        for field, option in (("model_id", "model_id"), ("revision", "revision")):
            value = getattr(args, option, None)
            if value is not None:
                profile[field] = value
        settings = settings.model_copy(update={"targets": (TargetConfig.model_validate(profile),)})
    if args.command == "submit" and args.slot is not None:
        selected = tuple(target for target in settings.targets if target.slot == args.slot)
        if not selected:
            raise ValueError("Selected model slot is disabled")
        settings = settings.model_copy(update={"targets": selected})
    return settings


def _target(args: argparse.Namespace, settings: Settings) -> MinerTarget:
    if settings.targets:
        slot = args.slot or settings.targets[0].slot
        source = next((target for target in settings.targets if target.slot == slot), None)
        if source is None:
            raise ValueError("Selected model slot is disabled")
        data = MinerTarget.from_profile(source).model_dump()
    else:
        if not args.model_id or not args.revision:
            raise ValueError("Provide --model-id and --revision, or --profile / --env-file")
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
    targets = commands.add_parser("targets", parents=[common])
    _target_options(targets)
    args = parser.parse_args()
    if args.command == "inspect":
        from shared.checkpoints import checkpoint_schema

        schema = checkpoint_schema(args.checkpoint)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(schema.model_dump_json() + "\n")
        return
    try:
        settings = _settings(args)
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
                selected = _target(args, settings)
            except (ValueError, OSError) as exc:
                parser.error(str(exc))
            record = selected.model_dump(mode="json")
            record["slot"] = args.slot or (settings.targets[0].slot if settings.targets else 1)
            print(json.dumps([record], indent=2))
            return
        print(
            json.dumps(
                [
                    {"slot": t.slot, "model_id": t.model_id, "competition_id": t.identity}
                    for t in settings.targets
                ],
                indent=2,
            )
        )
        return
    if args.command == "submit":
        from miner.submit import submit

        if args.max_checkpoint_bytes is None:
            print(submit(settings, args.package, publish=not args.upload_only))
        else:
            print(
                submit(
                    settings,
                    args.package,
                    publish=not args.upload_only,
                    max_checkpoint_bytes=args.max_checkpoint_bytes,
                )
            )
        return
    try:
        target = _target(args, settings)
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
