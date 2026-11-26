import json
from collections import Counter

import pytest

from evaluation.datasets.sampling import select_round
from tests.factories import make_checkpoint, target


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "return_dict": False}
        return [ord(c) for c in json.dumps(messages, sort_keys=True)]


def prepared_pool(tmp_path, workers=1):
    from evaluation.datasets.pool import prepare_pool
    from evaluation.datasets.sources import PreparationConfig

    config = PreparationConfig(
        model_id="org/target",
        revision="a" * 40,
        sources=[
            {"kind": kind, "category": category, "repo_id": repo, "revision": "b" * 40}
            for kind, category, repo in [
                ("wildchat", "chat", "allenai/WildChat-4.8M"),
                ("code", "code", "nvidia/OpenCodeInstruct"),
                ("math", "math", "nvidia/OpenMathReasoning"),
            ]
        ],
        buckets=[{"name": "all", "min_tokens": 1, "max_tokens": 512}],
    )

    def rows(source):
        for i in range(30):
            if source.kind == "wildchat":
                yield {
                    "conversation": [
                        {"role": "user", "content": f"chat question {i}"},
                        {"role": "assistant", "content": "answer must be excluded"},
                    ]
                }
            elif source.kind == "code":
                yield {"input": f"code question {i}", "output": "excluded"}
            else:
                for _ in range(3):
                    yield {"problem": f"math question {i}", "generated_solution": "excluded"}

    manifest = prepare_pool(
        config, tmp_path, Tokenizer(), rows, versions={"test": "1"}, workers=workers
    )
    return config, manifest


def dataset_target(manifest):
    from shared.targets import DatasetConfig, TargetConfig

    config = target().model_dump()
    config["model_id"] = "org/target"
    config["datasets"] = DatasetConfig(
        manifest=manifest.identity,
        min_unique_prompts=20,
        buckets=[{"name": "all", "min_tokens": 1, "max_tokens": 512}],
        mix={"chat": {"all": 40}, "code": {"all": 35}, "math": {"all": 25}},
    ).model_dump()
    for workload in config["workloads"]:
        workload.update(requests=20, prompts=[])
    return TargetConfig.model_validate(config)


def test_preparation_deduplicates_questions_and_excludes_stored_answers(tmp_path):
    config, manifest = prepared_pool(tmp_path)
    assert sum(shard.count for shard in manifest.shards) == 90
    assert {s.category: s.count for s in manifest.shards} == {
        "chat": 30,
        "code": 30,
        "math": 30,
    }
    contents = "".join(p.read_text() for p in (tmp_path / manifest.identity).glob("*.jsonl"))
    # The tokenizer fixture preserves all text: no stored completion enters the prompt.
    records = [json.loads(line) for line in contents.splitlines()]
    assert all("excluded" not in "".join(chr(i) for i in r["input_ids"]) for r in records)


def test_sampling_is_reproducible_distinct_and_stratified(tmp_path):
    from evaluation.datasets.sampling import select_round

    _, manifest = prepared_pool(tmp_path)
    config = dataset_target(manifest)
    first = select_round(config, tmp_path, "round-one")
    assert first == select_round(config, tmp_path, "round-one")
    assert first.identity != select_round(config, tmp_path, "round-two").identity
    assert len(first.prompts) == 20  # Same inputs at each concurrency.
    lookup = {p.identity: p for p in first.prompts}
    for workload in first.workloads:
        assert len(set(workload.prompt_ids)) == 20
        assert Counter(lookup[p].category for p in workload.prompt_ids) == {
            "chat": 8,
            "code": 7,
            "math": 5,
        }
    assert first.workloads[0].prompt_ids == first.workloads[1].prompt_ids


def test_sampling_rejects_insufficient_pool_and_changed_cache(tmp_path):
    from evaluation.datasets.sampling import select_round
    from shared.targets import TargetConfig

    _, manifest = prepared_pool(tmp_path)
    config = dataset_target(manifest)
    data = config.model_dump()
    for workload in data["workloads"]:
        workload["requests"] = 100
    with pytest.raises(ValueError, match="Insufficient"):
        select_round(TargetConfig.model_validate(data), tmp_path, "seed")
    shard = manifest.shards[0]
    path = tmp_path / manifest.identity / shard.data_file
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        select_round(config, tmp_path, "seed")


def test_wildchat_keeps_history_only_up_to_last_user_turn():
    from evaluation.datasets.sources import normalize

    messages = normalize(
        "wildchat",
        {
            "conversation": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "history"},
                {"role": "user", "content": "follow up"},
                {"role": "assistant", "content": "future answer"},
            ]
        },
    )
    assert [m["content"] for m in messages] == ["first", "history", "follow up"]
    assert normalize("code", {"input": "-"}) is None
    assert normalize("math", {"problem": "   "}) is None


def test_wildchat_unused_empty_completion_does_not_remove_valid_prompt():
    from evaluation.datasets.sources import normalize

    assert normalize(
        "wildchat",
        {
            "conversation": [
                {"role": "user", "content": "valid question"},
                {"role": "assistant", "content": ""},
            ]
        },
    ) == [{"role": "user", "content": "valid question"}]


def test_head_report_requires_dataset_evidence_even_when_timings_are_complete(tmp_path, pod):
    import asyncio

    from evaluation.runner import evaluate, summarize_report

    _, manifest = prepared_pool(tmp_path / "datasets")
    config = dataset_target(manifest)
    head, _, _ = pod
    report = asyncio.run(
        evaluate(
            config,
            head,
            tmp_path / "logs",
            mode="head",
            dataset_cache=tmp_path / "datasets",
        )
    )
    tampered = report.model_copy(update={"dataset_selection": None})
    assert summarize_report(config, tampered).status == "inconclusive"


def test_dataset_profiles_reject_tiny_samples_mixed_inputs_and_context_overflow(tmp_path):
    from shared.targets import TargetConfig

    _, manifest = prepared_pool(tmp_path)
    valid = dataset_target(manifest).model_dump()
    for update in ({"requests": 2}, {"prompts": ["manual"]}, {"output_tokens": 8000}):
        data = json.loads(json.dumps(valid))
        data["workloads"][0].update(update)
        with pytest.raises(ValueError):
            TargetConfig.model_validate(data)


def test_preparation_failure_never_publishes_partial_snapshot(tmp_path):
    from evaluation.datasets.pool import prepare_pool

    config, manifest = prepared_pool(tmp_path)

    def broken(_):
        yield {"problem": "one"}
        raise RuntimeError("download interrupted")

    with pytest.raises(RuntimeError):
        prepare_pool(config, tmp_path, Tokenizer(), broken, versions={"test": "2"})
    assert [p.name for p in tmp_path.iterdir()] == [manifest.identity]


@pytest.mark.parametrize("mode", ["head", "full"])
def test_measurement_uses_identical_frozen_tokens_for_every_arm_and_repeat(
    tmp_path, monkeypatch, pod, mode
):
    import asyncio
    from contextlib import asynccontextmanager

    import httpx

    from evaluation.runner import evaluate
    from evaluation.sglang import SGLangClient

    _, manifest = prepared_pool(tmp_path / "datasets")
    config = dataset_target(manifest)
    requests = []

    @asynccontextmanager
    async def server(config, draft, *args, **kwargs):
        def respond(request):
            if request.url.path == "/flush_cache":
                return httpx.Response(200, json={})
            body = json.loads(request.content)
            assert "text" not in body
            requests.append(tuple(body["input_ids"]))
            n = body["sampling_params"]["max_new_tokens"]
            return httpx.Response(
                200,
                json={
                    "output_ids": list(range(n)),
                    "meta_info": {"completion_tokens": n},
                },
            )

        async with httpx.AsyncClient(
            base_url="http://engine", transport=httpx.MockTransport(respond)
        ) as client:
            yield SGLangClient(client)

    monkeypatch.setattr("evaluation.measurement.launch_server", server)
    head, snapshot, _ = pod
    report = asyncio.run(
        evaluate(
            config,
            head,
            tmp_path / "logs",
            mode=mode,
            dataset_cache=tmp_path / "datasets",
        )
    )
    assert report.status == "complete", report.error_code
    selection = report.dataset_selection
    assert selection.manifest == manifest.identity
    assert selection.category_counts == {
        "low": {"chat": 8, "code": 7, "math": 5},
        "high": {"chat": 8, "code": 7, "math": 5},
    }
    assert len(set(requests)) == 20
    assert all(m.output_tokens == 160 for m in report.measurements)
    if mode == "head":
        assert report.score is None
        assert report.decision is None
        assert {m.arm for m in report.measurements} == {"candidate"}
    else:
        assert report.correctness_passed and report.reference_correctness_passed


def test_lium_inputs_freeze_dataset_plan_before_renting(tmp_path):
    from evaluation.lium import inputs

    _, manifest = prepared_pool(tmp_path / "datasets")
    config = dataset_target(manifest)
    request, paths = inputs(
        config,
        make_checkpoint(tmp_path / "head"),
        None,
        None,
        "local",
        30000,
        None,
        "head",
        dataset_cache=tmp_path / "datasets",
    )
    assert request["dataset_plan"]["manifest"] == manifest.identity
    assert len(request["dataset_plan"]["prompts"]) == 20
    with pytest.raises((OSError, ValueError)):
        inputs(
            config,
            paths["candidate"],
            None,
            None,
            "local",
            30000,
            None,
            "head",
            dataset_cache=tmp_path / "missing",
        )


def test_changed_selected_record_with_same_file_size_is_rejected(tmp_path):
    _, manifest = prepared_pool(tmp_path)
    target = dataset_target(manifest)
    shard = manifest.shards[0]
    path = tmp_path / manifest.identity / shard.data_file
    path.write_bytes(b"".join(b"!" + line[1:] for line in path.read_bytes().splitlines(True)))
    with pytest.raises(ValueError, match="record digest"):
        select_round(target, tmp_path, "seed")


def test_preparation_with_a_real_local_chat_tokenizer(tmp_path):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from evaluation.datasets.pool import prepare_pool
    from evaluation.datasets.sampling import read_prompts
    from evaluation.datasets.sources import PreparationConfig

    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"[UNK]": 0, "user": 1, ":": 2, "question": 3, "one": 4, "two": 5, "assistant": 6},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }} "
        "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
    )
    config = PreparationConfig(
        model_id="org/tokenizer",
        revision="a" * 40,
        sources=[{"kind": "math", "category": "math", "repo_id": "org/math", "revision": "b" * 40}],
        buckets=[{"name": "all", "min_tokens": 1, "max_tokens": 128}],
    )
    manifest = prepare_pool(
        config,
        tmp_path,
        tokenizer,
        lambda _: iter([{"problem": "question one"}, {"problem": "question two"}]),
    )
    prompts = read_prompts(tmp_path / manifest.identity, manifest.shards[0], [0, 1])
    assert [p.input_ids for p in prompts] == [(1, 2, 3, 4, 6, 2), (1, 2, 3, 5, 6, 2)]
    assert manifest.versions["transformers"] == transformers.__version__


def test_sources_stream_only_the_column_each_prompt_uses(monkeypatch):
    # The maths split is mostly worked solutions; reading only `problem` avoids ~30 GB.
    import sys
    from types import SimpleNamespace

    from evaluation.datasets.sources import DatasetSource, source_rows

    calls = []

    def load_dataset(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return iter([{"problem": "1 + 1?"}])

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    for kind, category, column in [
        ("wildchat", "chat", "conversation"),
        ("code", "code", "input"),
        ("math", "math", "problem"),
    ]:
        source = DatasetSource(
            kind=kind, category=category, repo_id=f"org/{kind}", revision="a" * 40, splits=("s",)
        )
        list(source_rows(source))
        assert calls[-1][1]["columns"] == [column]
        assert calls[-1][1]["streaming"] is True


def test_parallel_preparation_writes_the_same_bank_byte_for_byte(tmp_path, monkeypatch):
    # Small batches, so the run spans many batches spread over the workers.
    monkeypatch.setattr("evaluation.datasets.pool.BATCH", 7)
    _, one = prepared_pool(tmp_path / "one")
    _, many = prepared_pool(tmp_path / "many", workers=3)
    assert many == one
    for shard in one.shards:
        for name in (shard.data_file, shard.index_file):
            parallel = (tmp_path / "many" / many.identity / name).read_bytes()
            assert parallel == (tmp_path / "one" / one.identity / name).read_bytes()


def test_bank_recipe_comes_from_the_competition():
    from evaluation.datasets.sources import PreparationConfig, recipe
    from shared.targets import DatasetConfig, DatasetSource

    source = DatasetSource(kind="math", category="math", repo_id="org/math", revision="b" * 40)
    competition = target().model_copy(
        update={
            "datasets": DatasetConfig(
                manifest="c" * 64,
                sources=(source,),
                chat_template_kwargs={"reasoning_effort": "low"},
            )
        }
    )
    assert recipe(competition) == PreparationConfig(
        model_id=competition.model_id,
        revision=competition.revision,
        sources=(source,),
        chat_template_kwargs={"reasoning_effort": "low"},
    )
    with pytest.raises(ValueError, match="no dataset sources"):
        recipe(competition.model_copy(update={"datasets": DatasetConfig(manifest="c" * 64)}))


def test_prepare_datasets_builds_the_named_competitions_bank(tmp_path, monkeypatch, capsys):
    import sys
    from types import SimpleNamespace

    from evaluation.datasets import cli
    from evaluation.datasets.sources import recipe
    from shared.targets import DatasetSource
    from tests.factories import write_competitions

    source = DatasetSource(kind="code", category="code", repo_id="org/code", revision="b" * 40)
    competition = dataset_target(SimpleNamespace(identity="c" * 64))
    datasets = competition.datasets.model_copy(update={"sources": (source,)})
    competition = competition.model_copy(update={"datasets": datasets})
    write_competitions(competition)
    calls = []

    def prepare(config, cache, tokenizer, *, workers):
        calls.append((config, cache, workers))
        return SimpleNamespace(identity="d" * 64, shards=(), skipped_rows=0, duplicate_rows=0)

    monkeypatch.setattr(cli, "prepare_pool", prepare)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object())),
    )
    cli.main(
        ["--competition", competition.model_id, "--cache-dir", str(tmp_path), "--workers", "3"]
    )
    assert calls == [(recipe(competition), tmp_path, 3)]
    assert "d" * 64 in capsys.readouterr().out


def short_and_long_bank(tmp_path):
    """Chat has short and long prompts; code only short ones, like real coding questions."""
    from evaluation.datasets.pool import prepare_pool
    from evaluation.datasets.sources import PreparationConfig

    config = PreparationConfig(
        model_id="org/target",
        revision="a" * 40,
        sources=[
            {"kind": "wildchat", "category": "chat", "repo_id": "org/chat", "revision": "b" * 40},
            {"kind": "code", "category": "code", "repo_id": "org/code", "revision": "b" * 40},
        ],
        buckets=[
            {"name": "short", "min_tokens": 1, "max_tokens": 150},
            {"name": "long", "min_tokens": 151, "max_tokens": 1024},
        ],
    )

    def rows(source):
        for i in range(20):
            if source.kind == "wildchat":
                text = f"chat {i} " + ("x" * 200 if i % 2 else "")
                yield {"conversation": [{"role": "user", "content": text}]}
            else:
                yield {"input": f"code {i}"}

    return config, prepare_pool(config, tmp_path, Tokenizer(), rows, versions={"test": "1"})


def test_preparation_leaves_out_shelves_no_prompt_reaches(tmp_path):
    _, manifest = short_and_long_bank(tmp_path)
    shelves = {(s.category, s.bucket): s.count for s in manifest.shards}
    assert shelves == {("chat", "short"): 10, ("chat", "long"): 10, ("code", "short"): 20}


def test_round_takes_the_mix_share_of_each_category_and_length(tmp_path):
    from collections import Counter

    from evaluation.datasets.sampling import select_round, validate_selection
    from shared.targets import TargetConfig

    config, manifest = short_and_long_bank(tmp_path)
    competition = target().model_dump()
    competition["model_id"] = config.model_id
    competition["datasets"] = {
        "manifest": manifest.identity,
        "min_unique_prompts": 10,
        "buckets": [b.model_dump() for b in config.buckets],
        "mix": {"chat": {"short": 30, "long": 30}, "code": {"short": 40}},
    }
    for workload in competition["workloads"]:
        workload.update(requests=10, prompts=[])
    competition = TargetConfig.model_validate(competition)
    plan = select_round(competition, tmp_path, "seed")
    lookup = {p.identity: p for p in plan.prompts}
    for workload in plan.workloads:
        shelves = Counter((lookup[i].category, lookup[i].bucket) for i in workload.prompt_ids)
        assert shelves == {("chat", "short"): 3, ("chat", "long"): 3, ("code", "short"): 4}
    validate_selection(competition, plan.evidence(), seed="seed")
    datasets = competition.datasets.model_copy(update={"mix": {"code": {"long": 100}}})
    with pytest.raises(ValueError, match="code/long"):
        select_round(competition.model_copy(update={"datasets": datasets}), tmp_path, "seed")


def test_mix_is_percentages_adding_up_to_100():
    from shared.targets import DatasetConfig

    config = DatasetConfig(manifest="c" * 64)
    assert sum(share for lengths in config.mix.values() for share in lengths.values()) == 100
    with pytest.raises(ValueError, match="add up to 100"):
        DatasetConfig(manifest="c" * 64, mix={"chat": {"short": 0.4}, "code": {"short": 0.6}})
