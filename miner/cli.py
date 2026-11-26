from __future__ import annotations

import argparse
import json
from pathlib import Path

from miner.target import MinerTarget
from shared.artifacts import package_checkpoint
from shared.contracts import Provenance
from shared.targets import find_competition, load_competitions

# Defaults of the connection and wallet flags. These flags may come before or after the
# command, so their defaults are filled in after parsing.
DEFAULTS = {
    "api_url": "https://prexonai.com/api/v1",
    "network": "finney",
    "wallet_name": "default",
    "wallet_hotkey": "default",
    "wallet_path": "~/.bittensor/wallets",
    "max_upload_bytes": 4 * 1024**3 + 2 * 1024**2,  # Same as the API's UPLOAD_MAX_BYTES
}


def _common_options() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument("--api-url", help=f"API base URL (default: {DEFAULTS['api_url']})")
    common.add_argument(
        "--network", choices=["finney", "test"], help="Bittensor network (default: finney)"
    )
    common.add_argument("--netuid", type=int, help="The Prexon subnet ID (required to submit)")
    common.add_argument("--wallet-name", help="Default: default")
    common.add_argument("--wallet-hotkey", help="Default: default")
    common.add_argument("--wallet-path", help=f"Default: {DEFAULTS['wallet_path']}")
    common.add_argument(
        "--max-upload-bytes",
        type=int,
        help="Compressed package byte limit (default: 4 GiB + 2 MiB)",
    )
    return common


def main() -> None:
    common = _common_options()
    parser = argparse.ArgumentParser(
        description="Prexon drafter training, packaging, and submission", parents=[common]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "package"):
        command = commands.add_parser(name, parents=[common])
        command.add_argument(
            "--competition", required=True, metavar="MODEL_ID", help="A model in competitions/"
        )
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        if name == "train":
            command.add_argument("--recipe", type=Path, required=True)
            command.add_argument("--run-dir", type=Path, required=True)
            command.add_argument("--specforge-version", default="0.2.0")
        else:
            command.add_argument("--provenance", type=Path, required=True)
    send = commands.add_parser("submit", parents=[common])
    send.add_argument("package", type=Path)
    send.add_argument("--upload-only", action="store_true")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--checkpoint", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    commands.add_parser("competitions", help="List the competitions in competitions/")
    args = parser.parse_args()
    for name, value in DEFAULTS.items():
        vars(args).setdefault(name, value)
    if args.command == "inspect":
        from shared.checkpoints import checkpoint_schema

        schema = checkpoint_schema(args.checkpoint)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(schema.model_dump_json() + "\n")
        return
    if args.command == "competitions":
        print(
            json.dumps(
                [
                    {
                        "model_id": c.model_id,
                        "revision": c.revision,
                        "active": c.active,
                        "allocation": c.allocation,
                    }
                    for c in load_competitions()
                ],
                indent=2,
            )
        )
        return
    if args.command == "submit":
        from miner.submit import submit

        if not hasattr(args, "netuid"):
            parser.error("submit requires --netuid, the Prexon subnet ID")
        print(
            submit(
                args.package,
                netuid=args.netuid,
                network=args.network,
                api_url=args.api_url,
                wallet_name=args.wallet_name,
                wallet_hotkey=args.wallet_hotkey,
                wallet_path=args.wallet_path,
                targets=load_competitions(),
                max_upload_bytes=args.max_upload_bytes,
                publish=not args.upload_only,
            )
        )
        return
    try:
        target = MinerTarget.from_profile(find_competition(args.competition))
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
        assert target.competition_id is not None
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
