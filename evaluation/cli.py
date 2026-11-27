"""Single-head development evaluation from the command line; no production controller."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import uuid
from pathlib import Path

from evaluation.config import (
    DATASET_CACHE,
    EVALUATIONS_DIR,
    LIUM_API_KEY,
    SESSION_MAX_COST,
    SESSION_SECONDS,
)
from evaluation.runner import EvaluationRun, LocalEvaluationReport, evaluate
from shared.datasets import RoundPlan
from shared.targets import find_competition


def run_inline(
    evaluation: EvaluationRun,
    execution_id: str | None = None,
    *,
    dataset_cache: Path | None = None,
    prepared_round: RoundPlan | None = None,
) -> LocalEvaluationReport:
    async def inline() -> LocalEvaluationReport:
        running = asyncio.create_task(
            evaluate(
                evaluation.target,
                evaluation.draft_head,
                evaluation.output_dir,
                public_checkpoint=evaluation.public_head,
                incumbent_checkpoint=evaluation.incumbent_head,
                target_path=evaluation.target_path,
                seed=evaluation.seed,
                port=evaluation.port,
                mode=evaluation.mode,
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


def execute(evaluation: EvaluationRun, output: Path, api_key: str) -> LocalEvaluationReport:
    from evaluation.lium import evaluate_on_lium

    if evaluation.backend == "pod":
        return run_inline(
            evaluation.model_copy(update={"output_dir": output.with_suffix(".logs")}),
            dataset_cache=DATASET_CACHE,
        )

    def interrupt(signum: int, frame: object) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        return evaluate_on_lium(
            evaluation.target,
            evaluation.draft_head,
            output,
            api_key=api_key,
            public_checkpoint=evaluation.public_head,
            incumbent_checkpoint=evaluation.incumbent_head,
            remote_target_path=evaluation.target_path,
            seed=evaluation.seed,
            port=evaluation.port,
            mode=evaluation.mode,
            session_seconds=evaluation.session_seconds,
            session_max_cost=evaluation.session_max_cost,
        )
    finally:
        signal.signal(signal.SIGTERM, previous)


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from evaluation.lium import cleanup_evaluation, locked, quote_evaluation, write_json

    if args.dry_run and args.cleanup_only:
        parser.error("--dry-run and --cleanup-only cannot be combined")
    if not args.cleanup_only and (args.competition is None or args.head is None):
        parser.error("--competition and --head are required")
    try:
        api_key = LIUM_API_KEY
        evaluation = None
        if not args.cleanup_only:
            evaluation = EvaluationRun(
                target=find_competition(args.competition),
                draft_head=args.head.resolve(),
                public_head=args.public_head.resolve() if args.public_head else None,
                incumbent_head=args.incumbent_head.resolve() if args.incumbent_head else None,
                mode=args.mode,
                backend=args.backend,
                output_dir=EVALUATIONS_DIR,
                target_path=args.target_path,
                seed=args.seed,
                session_seconds=args.session_seconds,
                session_max_cost=args.session_max_cost,
            )
        if args.dry_run:
            assert evaluation is not None
            if evaluation.backend != "lium":
                parser.error("--dry-run quotes a Lium rental")
            print(
                json.dumps(
                    quote_evaluation(
                        evaluation.target,
                        evaluation.draft_head,
                        api_key=api_key,
                        public_checkpoint=evaluation.public_head,
                        incumbent_checkpoint=evaluation.incumbent_head,
                        remote_target_path=evaluation.target_path,
                        seed=evaluation.seed,
                        port=evaluation.port,
                        mode=evaluation.mode,
                        session_seconds=evaluation.session_seconds,
                        session_max_cost=evaluation.session_max_cost,
                    ),
                    indent=2,
                )
            )
            return
        with locked(EVALUATIONS_DIR / "evaluation"):
            pending = unfinished(EVALUATIONS_DIR)
            if args.cleanup_only:
                results = [cleanup_evaluation(path, api_key=api_key) for path in pending]
                print(
                    "Rental reconciliation complete" if all(results) else "Rental cleanup pending"
                )
                if not all(results):
                    raise SystemExit(2)
                return
            assert evaluation is not None
            if (
                len(pending) > 1
                or pending
                and (pending[0].parent.name != evaluation.mode or evaluation.backend != "lium")
            ):
                raise ValueError(
                    "Unfinished rentals exist; use --cleanup-only before starting this mode"
                )
            if pending:
                output = Path(json.loads(pending[0].read_bytes())["output"])
            else:
                output = EVALUATIONS_DIR / evaluation.mode / f"{uuid.uuid4().hex}.json"
            report = execute(evaluation, output, api_key)
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
        epilog="Prepare a prompt bank: prexon-evaluate prepare-datasets --competition MODEL_ID",
    )
    parser.add_argument("--competition", metavar="MODEL_ID", help="A model in competitions/")
    parser.add_argument(
        "--head", type=Path, help="The draft model to measure: config.json and safetensors"
    )
    parser.add_argument(
        "--mode",
        choices=["head", "full"],
        default="head",
        help="head: serving with the head only; full: also the target alone and the "
        "references given, with speedups and a score preview (default: head)",
    )
    parser.add_argument(
        "--public-head", type=Path, help="Full mode: the public reference to compare against"
    )
    parser.add_argument(
        "--incumbent-head", type=Path, help="Full mode: the current winner to compare against"
    )
    parser.add_argument(
        "--backend",
        choices=["lium", "pod"],
        default="lium",
        help="lium rents a pod; pod uses this GPU machine (default: lium)",
    )
    parser.add_argument(
        "--target-path",
        type=Path,
        help="A copy of the target model already on the GPU machine (absolute for Lium)",
    )
    parser.add_argument("--seed", default="42", help="Picks the round's prompts (default: 42)")
    parser.add_argument(
        "--session-seconds",
        type=int,
        help=f"Longest Lium rental (default: {SESSION_SECONDS})",
    )
    parser.add_argument(
        "--session-max-cost",
        type=float,
        help=f"Most a Lium rental may cost, in USD (default: {SESSION_MAX_COST})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Quote Lium without renting")
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help=f"Clean up unfinished rentals recorded in {EVALUATIONS_DIR}",
    )
    run(parser.parse_args(), parser)


if __name__ == "__main__":
    main()
