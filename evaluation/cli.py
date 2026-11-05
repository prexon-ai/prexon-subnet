"""Profile-based, single-head development evaluation; no production controller."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import uuid
from pathlib import Path

from evaluation.profile import EvaluationProfile, load_profile
from evaluation.runner import LocalEvaluationReport, evaluate
from shared.config import Settings, load_settings, read_environment
from shared.datasets import RoundPlan


def run_inline(
    profile: EvaluationProfile,
    execution_id: str | None = None,
    *,
    dataset_cache: Path | None = None,
    prepared_round: RoundPlan | None = None,
) -> LocalEvaluationReport:
    async def inline() -> LocalEvaluationReport:
        running = asyncio.create_task(
            evaluate(
                profile.target,
                profile.draft_head,
                profile.output_dir,
                public_checkpoint=profile.baseline_head,
                incumbent_checkpoint=profile.incumbent_head,
                target_path=profile.target_path,
                seed=profile.seed,
                port=profile.port,
                mode=profile.mode,
                dataset_cache=dataset_cache,
                prepared_round=prepared_round,
            )
        )
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, running.cancel)
        try:
            return await running
        finally:
            loop.remove_signal_handler(signal.SIGTERM)

    return asyncio.run(inline()).model_copy(update={"execution_id": execution_id})


def unfinished(output_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(output_dir.glob("*/*.lium.json"))
        if (data := json.loads(path.read_bytes()))["phase"] != "finished"
        or data["cleanup"] == "pending"
    ]


def execute(
    profile: EvaluationProfile, settings: Settings, output: Path, api_key: str
) -> LocalEvaluationReport:
    from evaluation.lium import evaluate_on_lium

    if profile.backend == "pod":
        return run_inline(
            profile.model_copy(update={"output_dir": output.with_suffix(".logs")}),
            dataset_cache=settings.dataset_cache,
        )

    def interrupt(signum: int, frame: object) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        return evaluate_on_lium(
            settings,
            profile.target,
            profile.draft_head,
            output,
            api_key=api_key,
            public_checkpoint=profile.baseline_head,
            incumbent_checkpoint=profile.incumbent_head,
            remote_target_path=profile.target_path,
            seed=profile.seed,
            port=profile.port,
            mode=profile.mode,
        )
    finally:
        signal.signal(signal.SIGTERM, previous)


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from evaluation.lium import cleanup_evaluation, locked, quote_evaluation, write_json

    if args.dry_run and args.cleanup_only:
        parser.error("--dry-run and --cleanup-only cannot be combined")
    try:
        profile = load_profile(args.profile)
        if args.mode:
            profile = profile.model_copy(update={"mode": args.mode})
        settings = load_settings("evaluation", args.env_file)
        overrides = {
            k: getattr(profile, k)
            for k in ("session_seconds", "session_max_cost")
            if getattr(profile, k) is not None
        }
        settings = settings.model_copy(update=overrides)
        api_key = read_environment(args.env_file).get("LIUM_API_KEY", "")
        if args.dry_run:
            if profile.backend != "lium":
                parser.error("--dry-run requires a Lium profile")
            print(
                json.dumps(
                    quote_evaluation(
                        settings,
                        profile.target,
                        profile.draft_head,
                        api_key=api_key,
                        public_checkpoint=profile.baseline_head,
                        incumbent_checkpoint=profile.incumbent_head,
                        remote_target_path=profile.target_path,
                        seed=profile.seed,
                        port=profile.port,
                        mode=profile.mode,
                    ),
                    indent=2,
                )
            )
            return
        with locked(profile.output_dir / "evaluation"):
            pending = unfinished(profile.output_dir)
            if args.cleanup_only:
                results = [cleanup_evaluation(settings, path, api_key=api_key) for path in pending]
                print(
                    "Rental reconciliation complete" if all(results) else "Rental cleanup pending"
                )
                if not all(results):
                    raise SystemExit(2)
                return
            if (
                len(pending) > 1
                or pending
                and (pending[0].parent.name != profile.mode or profile.backend != "lium")
            ):
                raise ValueError(
                    "Unfinished rentals exist; use --cleanup-only before starting this mode"
                )
            if pending:
                output = Path(json.loads(pending[0].read_bytes())["output"])
            else:
                output = profile.output_dir / profile.mode / f"{uuid.uuid4().hex}.json"
            report = execute(profile, settings, output, api_key)
            write_json(output, report.model_dump(mode="json"))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Development evaluation ({report.mode}): {report.assessment}")
    for name, metrics in report.head_metrics.items():
        acceptance = (
            f"{metrics.acceptance_rate:.1%}"
            if metrics.acceptance_rate is not None
            else "unavailable"
        )
        length = (
            f"{metrics.average_acceptance_length:.2f}"
            if metrics.average_acceptance_length is not None
            else "unavailable"
        )
        print(
            f"{name}: {metrics.throughput_tps:.3f} tokens/s; "
            f"p95 {metrics.p95_latency_seconds:.3f}s; "
            f"acceptance {acceptance}; acceptance length {length}"
        )
    if report.score is not None:
        print(
            f"Target-only speedup: {report.score:.3f}x; "
            f"available threshold: {report.threshold:.3f}x"
        )
    if report.error_code:
        print(f"Evaluation incomplete: {report.error_code}")
    print(f"Report: {output.resolve()}")
    if report.backend == "lium":
        print(f"Rental cleanup: {report.rental_cleanup}")
    if report.rental_cleanup == "pending":
        raise SystemExit(2)
    if report.assessment not in {"head_measured", "passes_local_checks"}:
        raise SystemExit(2 if report.assessment == "inconclusive" else 1)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "prepare-datasets":
        from evaluation.datasets.cli import main as prepare

        prepare(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(
        description="Standalone DFlash head and full comparison evaluation",
        epilog="Prepare datasets: prexon-evaluate prepare-datasets --config PATH --cache-dir PATH",
    )
    parser.add_argument(
        "--env-file", default=".env", help="Rental credentials and operational settings"
    )
    parser.add_argument("--profile", type=Path, required=True, help="Local evaluation JSON profile")
    parser.add_argument(
        "--mode", choices=["head", "full"], help="Override the profile mode (default: head)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Quote Lium without renting")
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Clean unfinished rentals in the profile output directory",
    )
    run(parser.parse_args(), parser)


if __name__ == "__main__":
    main()
