# Config File and Secrets-Only `.env` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Superseded during execution (2026-11-25):** tasks 1–6 were carried out as written
> (`prexon.toml`), then the owner asked for a committed `validator/config.py` instead.
> The TOML loader, `prexon.example.toml` and the `--config` flags were replaced by
> `validator/config.py` plus `shared/settings_loader.py`; see the design doc's
> "Settings: `validator/config.py`" section. Everything else below still holds.

**Goal:** `.env` holds only the wallet and secrets. Every other setting used by the validator, the upload API and `prexon-evaluate` comes from `prexon.toml`.

**Architecture:**
- A new module `shared/config_file.py` reads `prexon.toml` and checks its sections and keys. It rejects old `PREXON_*` keys found in `.env` or in the process environment, then builds the same flat `Settings` object the code uses today.
- The target-building logic is shared with the miner's env loader, so competition IDs cannot drift between the two.
- The miner keeps its env loader unchanged.
- The upload API container mounts `prexon.toml` and gets only the S3 secrets.

**Tech Stack:** Python 3.12 `tomllib` (standard library), pydantic 2, python-dotenv, pytest, Docker Compose 2.40.

**Spec:** `docs/superpowers/specs/2026-11-25-config-file-and-validator-modes-design.md`.
- This plan covers its config sections: "Secrets and wallet", "Settings", "Commands" (`--config`, `--host`), "Docker: upload API only", "Migration" and "Docs and examples".
- **Not covered here:** light mode, the full-mode owner check, the top-level `mode` key, the `[light]` section, the light-mode example file, and the mode/network/netuid lines in `--check-config`. They get a separate plan. Until then the validator behaves as today, which is full mode.

## Global Constraints

- **`.env` keys:** `.env.example` contains exactly these keys: `PREXON_WALLET_NAME`, `PREXON_WALLET_HOTKEY`, `PREXON_WALLET_PATH`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_SESSION_TOKEN`, `LIUM_API_KEY`.
- **S3 endpoint and region:** `S3_ENDPOINT_URL` and `S3_REGION` stay accepted in the environment, but they leave `.env.example`.
- **TOML key names:** keys inside a section are the existing `Settings` field names, verbatim.
- **No overrides:** environment variables never override `prexon.toml`.
- **Competition IDs must not change:** the same target values must produce byte-identical `TargetConfig`s.
- **No new dependencies.**
- **CI stays green:**
  - `uv run ruff check .`
  - `uv run ruff format --check .`
  - `uv run mypy` (`disallow_untyped_defs`)
  - `uv run pytest`
  - `uv build`
  - `uv run prexon-validator --check-config` with no config file and no `.env`
  - `uv run prexon-miner targets`
  - `uv run prexon-evaluate --help`
- **Miner CLI unchanged:** `prexon-miner` behaves exactly as today, including `PREXON_MODEL_*` in its optional `--env-file`.
- **No secrets in committed files:** no secret ever appears in `prexon.toml` or in a committed example.
- **Commits:** do not commit or push. The owner commits when they ask. The working tree already holds the uncommitted Docker-removal change; build on top of it.

## Review Focus

1. **An old `.env` copied forward:** it has 55 keys, and some are present but empty (`PREXON_MODEL_2_ID=`, `PREXON_UPLOAD_STAGING_ROOT=`).
   - Every moved key is listed, including the empty ones.
   - The command exits with code 2 and a message, not a traceback, and never starts idle on defaults.
   - Tested in Task 1 (empty values) and Task 2 (subprocess exit code and stderr).
2. **`prexon.toml` missing when `docker compose up` runs:** Docker must not create it as a root-owned directory.
   - Compose refuses to start.
   - A directory at the config path gives a clear `ConfigError`, not `IsADirectoryError`.
   - Tested in Task 1 (directory path) and Task 4 (every bind sets `create_host_path: false`).
3. **Malformed TOML** (unclosed quote, duplicate key): a `ConfigError` names the file and the parser's line, and the command exits with code 2. Tested in Task 1.
4. **TOML integers where code wants floats** (`allocation = 20`, `poll_seconds = 30`): accepted, with the same competition ID as the miner env loader. Tested in Task 1 (the identity test uses an integer allocation).
5. **A miner's own `.env` holding `PREXON_MODEL_1_*`** in the folder where they run `prexon-evaluate`: evaluation still runs.
   - This is a deliberate narrowing of the spec: evaluation rejects only the moved keys it reads.
   - The existing `tests/evaluation/test_cli.py::test_public_cli_only_needs_profile_for_head_mode` keeps covering it. It must pass unmodified in Task 2.

---

### Task 1: The `prexon.toml` loader

**Files:**
- Create: `shared/config_file.py`
- Modify: `shared/config.py:309-371`. Extract `build_targets`; `load_settings` keeps its behaviour.
- Create: `tests/shared/test_config_file.py`
- Modify: `tests/shared/test_config.py`. Delete `test_two_allocations_and_profile_identity`, `test_api_loads_upload_settings_without_validator_or_target_configuration` and `test_evaluation_configuration_ignores_unrelated_validator_settings`; their behaviour moves to the new file.

**Interfaces:**
- Produces, in `shared/config.py`:
  - `build_targets(entries: Iterable[Mapping[str, Any]]) -> tuple[TargetConfig, ...]`. Each entry may hold `slot`, `model_id`, `revision`, `allocation`, `methods` and `profile`.
- Produces, in `shared/config_file.py`:
  - `class ConfigError(ValueError)`
  - `Role = Literal["validator", "api", "evaluation"]`
  - `SECTIONS: dict[str, tuple[str, ...]]`
  - `TARGET_KEYS: tuple[str, ...]`
  - `load_config(role: Role, config_path: str | Path | None = None, env_file: str | Path | None = ".env") -> Settings`

**`SECTIONS`, verbatim from the spec:**

```python
SECTIONS = {
    "chain": ("network", "netuid", "chain", "chain_file", "burn_uid", "weights_version"),
    "schedule": (
        "poll_seconds",
        "schedule_seconds",
        "early_queue_depth",
        "session_seconds",
        "session_max_cost",
        "session_retry_seconds",
        "max_evaluation_attempts",
        "publish_seconds",
    ),
    "storage": (
        "state_path",
        "artifact_root",
        "upload_staging_root",
        "public_artifact_root",
        "public_artifact_base_url",
        "dataset_cache",
        "upload_policy_path",
    ),
    "rental": (
        "provider",
        "lium_base_url",
        "lium_template_id",
        "lium_volume_id",
        "ssh_private_key",
        "ssh_public_key",
        "ssh_known_hosts",
    ),
    "api": (
        "api_host",
        "api_port",
        "uploads_enabled",
        "upload_state_path",
        "upload_policy_max_age_seconds",
        "upload_url_ttl_seconds",
        "upload_request_max_age_seconds",
        "upload_max_requests_per_hour",
        "upload_max_bytes",
    ),
}
TARGET_KEYS = ("slot", "model_id", "revision", "allocation", "methods", "profile")
```

**Fields each role reads:**
- **validator:** every `SECTIONS` key, `[[targets]]`, and the wallet from the environment.
- **api:** `SECTIONS["api"]`, plus `public_artifact_root`, `upload_staging_root` and `upload_policy_path`.
- **evaluation:** `SECTIONS["rental"]`, plus `session_seconds`, `session_max_cost` and `dataset_cache`.

**Rules for `load_config`:**

*Choosing the file*
- `config_path=None` means `prexon.toml` in the working directory, and a missing file there means all defaults.
- An explicit path that does not exist is a `ConfigError`.
- Any path that exists but is not a regular file is a `ConfigError`.

*Reading it*
- A `tomllib.TOMLDecodeError` becomes a `ConfigError` that starts with the path and keeps tomllib's line and column text.

*Structure checks (every role, the whole file)*
- Collect every problem, then raise one `ConfigError` listing them all.
- Unknown section: `[colours]: unknown section`.
- Unknown key: `[chain] colour: unknown key`.
- A known key in the wrong place, including at top level: `[chain] api_port: belongs in [api]`.
- Secret-looking keys in any section or at top level:
  - `s3_access_key_id`, `s3_secret_access_key`, `s3_session_token`, `lium_api_key`, and any key starting `wallet_`.
  - Message: `lium_api_key belongs in .env as LIUM_API_KEY`. Map `wallet_name/hotkey/path` to `PREXON_WALLET_NAME/HOTKEY/PATH`; any other `wallet_*` key just says `belongs in .env`.
- `targets` must be an array of tables (`[[targets]]`). A single `[targets]` table is an error that says to use `[[targets]]`.
- Unknown keys inside a target entry are errors.

*Moved keys in the environment*
- Read the merged environment with the existing `read_environment(env_file)`: the dotenv file plus the process environment.
- **Validator and api:** a `PREXON_<FIELD>` key is an error when `FIELD` is any `SECTIONS` key. So is any key matching `^PREXON_MODEL_[12]_`.
- **Evaluation:** only `PREXON_<FIELD>` keys for the fields evaluation reads.
- A key counts even when its value is empty.
- Message: `PREXON_NETUID moved to prexon.toml [chain] netuid`; for target keys, `PREXON_MODEL_1_ID moved to prexon.toml [[targets]]`.
- These errors are collected into the same list as the structure errors.
- Every other `PREXON_*` key is ignored, for example `PREXON_TARGET_CACHE` and `PREXON_UPLOAD_API_URL`.

*Building `Settings`*
- Flatten only the keys the role reads, and add the validator's wallet values from non-empty `PREXON_WALLET_*`.
- Call `Settings.model_validate({"role": role, "targets": ..., "env_file": env_file, **flat})`.
- Convert a pydantic `ValidationError` into a `ConfigError` with one line per error.
  - Name each field with its section: `[api] api_port: Input should be a valid integer...`.
  - An error with an empty location (the storage-separation validator) keeps its message as-is.
- For the validator only, targets come from `build_targets(data.get("targets", []))`. A `ValueError` or `ValidationError` from it becomes a `ConfigError` prefixed `[[targets]]`.

**Rules for `build_targets`** (same meaning as today's `PREXON_MODEL_*` loop):
- Skip an entry whose stripped `model_id` is empty.
- An enabled entry without `profile` raises `ValueError("Enabled target slot N requires a profile")`.
- Profile paths are read relative to the working directory.
- Merge `slot`, the stripped `model_id` and `revision` (default `""`) into the profile.
- `allocation`: use `float(value)` unless the value is `None` or `""`, which means 100. This way 0 still fails validation.
- `methods`: set it only when non-empty, as a list.
- Then:
  - validate every entry as a `TargetConfig`;
  - reject a repeated slot;
  - check that allocations sum to 100, keeping the existing message "Enabled target emission allocations must sum to 100";
  - check that models are distinct, keeping the existing message;
  - return the targets sorted by slot.
- `load_settings` converts its `PREXON_MODEL_{1,2}_*` values into these entries and calls `build_targets`. The env value `METHODS` is split on commas, as today.

- [ ] **Step 1: Write the failing tests in `tests/shared/test_config_file.py`**

  Use a `write(tmp_path, text) -> Path` helper for TOML files. Reuse `profile(tmp_path)` from `tests/shared/test_config.py` by importing it. Every call passes an explicit `env_file` (a tmp file or `None`), so the developer's `.env` is never read.

  | Test | Assertions |
  | --- | --- |
  | `test_sections_load_into_settings_fields` | File with `[chain] netuid = 7`, `network = "test"`, `[schedule] poll_seconds = 5`, `[storage] state_path = "x/state.sqlite3"`, `[rental] lium_template_id = "t"` and `[api] api_port = 9001`. The validator's settings hold `netuid == 7`, `network == "test"`, `poll_seconds == 5`, `state_path == Path("x/state.sqlite3")`, `lium_template_id == "t"` and `api_port == 9001`. |
  | `test_missing_default_file_means_defaults_and_missing_explicit_file_fails` | With `monkeypatch.chdir(tmp_path)`, `load_config("validator", None, None).model_dump() == Settings(role="validator").model_dump()`. `load_config("validator", tmp_path / "absent.toml", None)` raises `ConfigError` matching `absent.toml`. |
  | `test_directory_at_config_path_is_a_config_error` | `(tmp_path / "prexon.toml").mkdir()` plus chdir: `load_config("api", None, None)` raises `ConfigError`, not `IsADirectoryError`. |
  | `test_malformed_toml_names_the_file_and_line` | Text `[chain]\nnetwork = "finney\n`: raises `ConfigError` whose message contains the file name and `line`. |
  | `test_structure_errors_are_listed_together` | Parametrize over the three roles. One file with `[colours]`, `[chain] colour = 1`, `[chain] api_port = 1` and a top-level `poll_seconds = 1`. The message contains `[colours]: unknown section`, `[chain] colour: unknown key`, `[chain] api_port: belongs in [api]` and `poll_seconds: belongs in [schedule]`. |
  | `test_secrets_in_the_config_file_point_to_env` | `[rental] lium_api_key = "x"` gives `lium_api_key belongs in .env as LIUM_API_KEY`. Top-level `wallet_name = "x"` gives `belongs in .env as PREXON_WALLET_NAME`. `[storage] s3_secret_access_key = "x"` gives `S3_SECRET_ACCESS_KEY`. |
  | `test_targets_must_be_an_array_of_tables_with_known_keys` | `[targets]\nslot = 1` raises matching `\[\[targets\]\]`. `[[targets]]\nslot = 1\ncolour = 1` raises matching `colour` for role `"api"` too. |
  | `test_moved_env_keys_are_rejected_with_their_new_location` | Parametrize roles `validator` and `api`. The env file holds `PREXON_NETUID=5`, `PREXON_MODEL_2_ID=` (empty) and `PREXON_UPLOAD_STAGING_ROOT=` (empty); `monkeypatch.setenv("PREXON_API_PORT", "1")`. The message contains all four of: `PREXON_NETUID moved to prexon.toml [chain] netuid`, `PREXON_MODEL_2_ID moved to prexon.toml [[targets]]`, `PREXON_UPLOAD_STAGING_ROOT moved to prexon.toml [storage] upload_staging_root` and `PREXON_API_PORT moved to prexon.toml [api] api_port`. |
  | `test_evaluation_rejects_only_moved_keys_it_reads` | The env file holds `PREXON_MODEL_1_ID=x` and `PREXON_NETUID=5`, and loads fine for `"evaluation"`. Adding `PREXON_SESSION_SECONDS=60` raises matching `PREXON_SESSION_SECONDS moved to prexon.toml [schedule] session_seconds`. |
  | `test_wallet_and_unrelated_prexon_variables_come_from_env` | The env file holds `PREXON_WALLET_NAME=owner`, `PREXON_WALLET_HOTKEY=hot`, `PREXON_WALLET_PATH=/w`, `PREXON_TARGET_CACHE=/c` and `PREXON_UPLOAD_API_URL=http://x`. The validator loads with `wallet_name == "owner"`, `wallet_hotkey == "hot"`, `wallet_path == "/w"`, and `upload_api_url` keeps its default. |
  | `test_wrong_type_names_section_and_key` | `[api] api_port = "eighty"` raises matching `\[api\] api_port`. |
  | `test_roles_type_check_only_the_keys_they_read` | `[schedule] poll_seconds = "x"`, `[chain] netuid = "x"`, plus `[[targets]]` with `slot = 1`, `model_id = "m"`, `profile = "/missing"`. `load_config("api", ...)` gives `targets == ()`. `load_config("evaluation", ...)` gives `targets == ()`. The validator raises `ConfigError`. Also: `[api] uploads_enabled = true` with `[storage] upload_policy_path = "p.json"` gives the api role `uploads_enabled` true and `upload_policy_path == Path("p.json")`. |
  | `test_toml_targets_match_the_miner_env_loader_identity` | TOML `[[targets]]` for slot 1 (`allocation = 20`, an integer) and slot 2 (`allocation = 80`), with `revision = "a" * 40` and the shared profile. The same values as `PREXON_MODEL_{n}_*` in an env file go to `load_settings("miner", env)`. The `[t.identity for t in ...targets]` lists are equal, and the allocations are `[20.0, 80.0]`. Setting `LIUM_API_KEY` in the env leaves the identities unchanged. |
  | `test_target_rules` | Parametrize: allocations 20 and 70 → `100`; a repeated `slot = 1` → `slot`; an enabled entry without `profile` → `profile`; `allocation = 0` → `ConfigError`. Separately, an entry with `model_id = ""` and no profile loads as disabled. Entries written in the order slot 2 then slot 1 come back as `[1, 2]`. |

- [ ] **Step 2: Run the tests and watch them fail**

  Run: `uv run pytest tests/shared/test_config_file.py -q`
  Expected: collection error `ModuleNotFoundError: No module named 'shared.config_file'`.

- [ ] **Step 3: Implement `build_targets` in `shared/config.py`**

  Make `load_settings` use it, following the rules above.

- [ ] **Step 4: Implement `shared/config_file.py`**

  Follow the rules above.

- [ ] **Step 5: Delete the three env-based tests from `tests/shared/test_config.py`**

  Remove the `load_settings` import if it is now unused.

- [ ] **Step 6: Run the tests and watch them pass**

  Run: `uv run pytest tests/shared -q && uv run ruff check . && uv run mypy`
  Expected: all pass, with no lint or type errors.

---

### Task 2: Validator and evaluation commands read `prexon.toml`

**Files:**
- Modify: `validator/main.py:25-40` and `:89`
- Modify: `evaluation/cli.py:104` and `:202-204`
- Modify: `evaluation/measurement.py:116`. Drop the `PREXON_DATASET_CACHE` environment fallback and use `dataset_cache or Path("runtime/datasets")`, as `evaluation/lium.py:91` does. That variable is now a moved key.
- Test: `tests/integration/test_cli.py`, `tests/evaluation/test_lium.py:345`

**Interfaces:**
- Consumes: `load_config` and `ConfigError` from Task 1.

**Changes:**
- **Validator:**
  - Add `--config` (`type=Path`, default `None`, help "Settings file (default: prexon.toml if present)").
  - Call `load_config("validator", args.config, args.env_file)`.
  - On `ConfigError`, call `parser.error(str(exc))`. That gives exit code 2 with no traceback.
  - The netuid error becomes `"Set [chain] netuid in prexon.toml before enabling models"`.
- **Evaluation:**
  - Add the same `--config`.
  - Call `load_config("evaluation", args.config, args.env_file)`. The existing `except (ValueError, OSError)` already turns `ConfigError` into `parser.error`.
  - `--env-file` help becomes "Secrets file: LIUM_API_KEY".
- **`tests/evaluation/test_lium.py:345`:** monkeypatch `evaluation.cli.load_config` instead of `load_settings`.

- [ ] **Step 1: Rewrite the integration tests to use `prexon.toml`**

  In `tests/integration/test_cli.py`:
  - `test_idle_controller_runs_without_external_credentials` and `test_requeue_command_returns_set_aside_heads_to_the_queue` write `tmp_path / "prexon.toml"` with a `[storage]` section holding the same paths, and pass `--config`.
  - `test_qualification_cli_validates_uploads_and_prints_immutable_id` writes the `[[targets]]` entry (`slot = 1`, `model_id`, `revision`, `profile`) and `[storage] artifact_root`.
  - In `test_initialized_empty_controller_keeps_bittensor_for_clearing_weights`, monkeypatch `validator.main.load_config`.

  Add these tests:

  | Test | Assertions |
  | --- | --- |
  | `test_old_env_file_stops_the_validator_with_every_moved_key` | Run `validator.main --env-file <old.env> --once` with `cwd=tmp_path`. The old file holds `PREXON_NETUID=5`, `PREXON_MODEL_1_ID=` and `PREXON_POLL_SECONDS=30`. `returncode == 2`. Stderr contains the three `moved to prexon.toml` lines and no `Traceback`. `tmp_path / "runtime"` was not created. |
  | `test_check_config_runs_without_config_or_env_files` | Run `validator.main --check-config` in an empty `tmp_path`. `returncode == 0` and `json.loads(stdout) == {"targets": []}`. |

- [ ] **Step 2: Run them and watch them fail**

  Run: `uv run pytest tests/integration/test_cli.py -q`
  Expected: failures. argparse rejects `--config` as an unrecognized argument, and the old-env test gets exit code 0 or 1 instead of 2.

- [ ] **Step 3: Make the code changes listed above in the three files and the test monkeypatch**

- [ ] **Step 4: Run the tests and watch them pass**

  Run: `uv run pytest tests/integration tests/evaluation -q`
  Expected: PASS. `tests/evaluation/test_cli.py::test_public_cli_only_needs_profile_for_head_mode` passes without edits (Review Focus 5).

---

### Task 3: Upload API command and Docker

**Files:**
- Modify: `api/main.py`
- Modify: `shared/config.py`:
  - `api_port` default becomes `8080`.
  - Rename `load_settings` to `load_miner_settings(env_file: str | Path | None, *, load_targets: bool = True, overrides: dict[str, str | int] | None = None) -> Settings`. It fixes `role="miner"` and drops the api/evaluation field filtering, which no longer has callers.
- Modify: `miner/cli.py:9,62` to call `load_miner_settings`. Do the same in `tests/shared/test_config_file.py`.
- Modify: `docker/Dockerfile.api`, `docker/compose.yaml`
- Test: `tests/api/test_api.py`, `tests/shared/test_config.py` (the compose test)

**Interfaces:**
- Consumes: `load_config` and `ConfigError` from Task 1.
- Produces: `prexon-api --config PATH --env-file PATH --host HOST`. `--host` overrides `[api] api_host`; the port always comes from `[api] api_port`.

**Target files:**

- **`api/main.py`:**
  - `create_app(settings=None)` falls back to `load_config("api")`.
  - `main()` adds `--config` and `--host`, and maps `ConfigError` to `parser.error`.
  - It calls `uvicorn.run(..., host=args.host or settings.api_host, port=settings.api_port)`.
- **`docker/Dockerfile.api`:**
  - Delete the three `PREXON_*` `ENV` entries; keep `PATH` and `PYTHONUNBUFFERED`.
  - Create `/app/runtime/api` and `/app/runtime/intake-policy`, owned by `prexon`, instead of `/data/...`.
  - Change `EXPOSE 8000` to `EXPOSE 8080`.
  - Keep `ENTRYPOINT ["prexon-api"]` and add `CMD ["--host", "0.0.0.0"]`.
- **`docker/compose.yaml`, `api` service:**
  - Keep `build` and the required `user: "${API_CONTAINER_USER:?...}"`.
  - `environment`: only `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` and `S3_SESSION_TOKEN`, each `${NAME:-}`.
  - `ports: ["127.0.0.1:8080:8080"]`.
  - `restart: unless-stopped`.
  - `volumes` in long syntax, each `type: bind` with `bind: {create_host_path: false}`:

    | Source | Target | Access |
    | --- | --- | --- |
    | `../prexon.toml` | `/app/prexon.toml` | `read_only: true` |
    | `../runtime/intake-policy` | `/app/runtime/intake-policy` | `read_only: true` |
    | `../runtime/api` | `/app/runtime/api` | writable |

  - Keep comments short: why the API runs as the validator's user (the 0640 policy file), and why the binds never create host paths.

- [ ] **Step 1: Write the failing tests**

  In `tests/api/test_api.py`, add `test_main_uses_config_port_and_host_override`:
  - Monkeypatch `uvicorn.run` to record its `host` and `port` keywords, and patch `sys.argv`.
  - Config `[api] api_port = 9001`, run with `--host 0.0.0.0`: recorded `("0.0.0.0", 9001)`.
  - An empty config without `--host`: recorded `("127.0.0.1", 8080)`.

  In `tests/shared/test_config.py`, replace the compose test with `test_compose_api_mounts_config_and_policy_read_only_and_gets_only_s3_secrets`:
  - `set(services) == {"api"}`.
  - Build `{target: (resolved source, read_only)}` from the long-syntax volumes. Every volume has `bind["create_host_path"] is False`.
  - `/app/prexon.toml` maps to `(root / "prexon.toml", True)`.
  - Take the defaults `Settings.model_fields["upload_policy_path"].default` and `Settings.model_fields["upload_state_path"].default`, which are relative paths.
  - `PurePosixPath("/app") / policy.parent` maps to `(root / policy.parent, True)`, and `PurePosixPath("/app") / state.parent` maps to `(root / state.parent, False)`.
  - `set(api["environment"]) == {"S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_SESSION_TOKEN"}`.
  - `api["ports"] == ["127.0.0.1:8080:8080"]`.
  - `api["user"].startswith("${API_CONTAINER_USER:?")`.
  - `"PREXON_" not in (root / "docker/Dockerfile.api").read_text()`.

- [ ] **Step 2: Run the tests and watch them fail**

  Run: `uv run pytest tests/api/test_api.py tests/shared/test_config.py -q`
  Expected: the API test fails because `--config` is unrecognized; the compose test fails on the volumes and environment.

- [ ] **Step 3: Make the changes listed above**

  This covers `api/main.py`, the `shared/config.py` default and rename, `miner/cli.py`, `Dockerfile.api` and `compose.yaml`.

- [ ] **Step 4: Run the tests and watch them pass**

  Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy`
  Expected: all pass. No caller of `load_settings` remains: `grep -rn "load_settings" --include=*.py .` matches nothing outside `.venv`.

- [ ] **Step 5: Check the compose file and the image without touching the live API on port 8080**

  Run each item and confirm its result:
  - **Compose file:**
    - Command: `API_CONTAINER_USER="$(id -u):$(id -g)" docker compose --env-file .env.example -f docker/compose.yaml config --quiet`
    - Expected: exit 0.
  - **Missing `prexon.toml`:**
    - Copy the checkout without `.venv` into the scratchpad as `$COPY`, with no `prexon.toml` and with `runtime/intake-policy` and `runtime/api` created.
    - Command: `API_CONTAINER_USER="$(id -u):$(id -g)" docker compose -p prexon-plancheck --env-file $COPY/.env.example -f $COPY/docker/compose.yaml create`
    - Expected: it fails with "bind source path does not exist" (or similar), and `$COPY/prexon.toml` still does not exist.
    - `create` does not bind ports, so the live API on 8080 is untouched. Clean up with `docker compose -p prexon-plancheck -f $COPY/docker/compose.yaml down`.
  - **Image:**
    - Command: `docker build -f docker/Dockerfile.api -t prexon-api:plancheck .`
    - Then run it as UID 10001 on a spare host port (`-p 127.0.0.1:18080:8080`), with a temporary `prexon.toml` mounted at `/app/prexon.toml` and temporary runtime directories.
    - Expected: `curl -fsS http://127.0.0.1:18080/api/health` returns `{"status":"ok"}`.
    - Remove the test container afterwards.

---

### Task 4: The two example files

**Files:**
- Modify: `.env.example` (full rewrite)
- Create: `prexon.example.toml`
- Modify: `.gitignore`. Add `/prexon.toml` under `.env.example`'s block.
- Test: `tests/shared/test_config_file.py`

**Interfaces:**
- Consumes: `load_config`, `SECTIONS` and `TARGET_KEYS` from Task 1.

**`.env.example` content:** the seven keys, in three commented groups.
- **Validator wallet:** `PREXON_WALLET_NAME=default`, `PREXON_WALLET_HOTKEY=default`, `PREXON_WALLET_PATH=~/.bittensor/wallets`.
- **Hippius S3 key:** `S3_ACCESS_KEY_ID=`, `S3_SECRET_ACCESS_KEY=`, `S3_SESSION_TOKEN=`. One key for all three buckets, shared by the full validator and the upload API. It is created at https://console.hippius.com, and miners need none.
- **Lium:** `LIUM_API_KEY=`, for the full validator's rentals and `prexon-evaluate`.
- The header says: copy to `.env`; everything else is in `prexon.toml`.

**`prexon.example.toml` content:**
- **Header:** copy to `prexon.toml`; secrets and the wallet belong in `.env`; relative paths resolve against the working directory.
- **Sections:** every `SECTIONS` key at its `Settings` default, grouped `[chain]`, `[schedule]`, `[storage]`, `[rental]`, `[api]`.
- **Comments:** carry over the useful explanations from the old `.env.example`:
  - the session trigger;
  - the retry wait;
  - stalled attempts;
  - the three separate Hippius buckets, with commented `s3://prexon-uploads`, `s3://prexon-private`, `s3://prexon-public` and `https://s3.hippius.com/prexon-public` examples;
  - that the API keeps its state separate from the validator's.
- **`netuid = 0`:** commented "required on a live chain; 0 means not set".
- **Targets:** two `[[targets]]` entries, slots 1 and 2.
  - Both have `model_id = ""`, `revision = ""` and `methods = ["dflash2"]`.
  - Slot 1 has `allocation = 100` and `profile = "profiles/model-1.json"`. Slot 2 has `profile = "profiles/model-2.json"`.
  - The comment says an empty `model_id` disables a slot, and two enabled slots need allocations summing to 100.

- [ ] **Step 1: Write the failing tests in `tests/shared/test_config_file.py`**

  | Test | Assertions |
  | --- | --- |
  | `test_env_example_holds_only_wallet_and_secrets` | `set(dotenv_values(root / ".env.example")) == {the seven keys}`. `load_config("validator", None, root / ".env.example")` loads, with a chdir to an empty `tmp_path` first so no `prexon.toml` is read. |
  | `test_example_config_documents_every_setting_at_its_default` | `tomllib` parse of `root / "prexon.example.toml"`: `{s: set(v) for s, v in data.items() if s != "targets"} == {s: set(k) for s, k in SECTIONS.items()}`. Every target entry's keys equal `set(TARGET_KEYS)`, except slot 2, which may omit `allocation`. For each role, with a chdir to an empty `tmp_path`, `load_config(role, example, None).model_dump() == load_config(role, None, None).model_dump()`. |

- [ ] **Step 2: Run them and watch them fail**

  Run: `uv run pytest tests/shared/test_config_file.py -k example -q`
  Expected: FAIL. The old `.env.example` has about 55 keys, and `prexon.example.toml` does not exist.

- [ ] **Step 3: Write both files and the `.gitignore` line, as specified above**

- [ ] **Step 4: Run the tests and watch them pass**

  Run: `uv run pytest tests/shared -q`
  Expected: PASS.

---

### Task 5: Documentation

**Files:**
- Modify: `README.md`, `docs/validator.md`, `docs/docker.md`, `docs/api.md`, `docs/evaluation.md`, `docs/datasets.md`, `docs/architecture.md:35`, `docs/miner.md:113`, `docs/testing.md:35`

**What changes:**

- **`README.md`:**
  - Quick start copies both examples (`cp .env.example .env`, `cp prexon.example.toml prexon.toml`).
  - The local API URL becomes `http://127.0.0.1:8080/api/v1`.
  - The continuous-operation paragraph says the validator and upload API share `.env` for secrets and `prexon.toml` for settings.
- **`docs/validator.md`:**
  - "Configure the controller" copies both files, and says S3 and Lium keys plus the wallet go in `.env` while everything else goes in `prexon.toml`.
  - Every `PREXON_X=v` mention becomes the TOML key with its section. For example `[storage] state_path`, `[schedule] early_queue_depth = 10`, `[chain] chain = "file"` and `[rental] provider = "local"`.
  - "Start and monitor" says pm2 must start in the checkout so `.env`, `prexon.toml` and `runtime/` resolve.
  - Add a short migration note: an old `.env` stops startup with a list of where each key moved. Move each value to the named TOML key and keep only the seven keys in `.env`.
- **`docs/docker.md`:**
  - "Start the upload API" first creates `prexon.toml` (Compose refuses to start without it) and the two runtime folders.
  - The mount table gets three rows matching Task 3.
  - The storage snippet splits into the `.env` secrets and the `[storage]` roots plus `[api] uploads_enabled = true`.
  - Routing: the API listens on `127.0.0.1:8080`; the container gets `--host 0.0.0.0` from the image. Keep `upload_policy_path` and `upload_state_path` relative so the same file works on the host and in the container.
- **`docs/api.md`:**
  - The configuration table lists `[api]` keys and `[storage] upload_policy_path` with their defaults.
  - Line 9 becomes `[storage] public_artifact_root`.
  - Line 73 uses port 8080.
  - Line 88 becomes `[api] uploads_enabled = true`.
- **`docs/evaluation.md`:** Lium settings go under `[rental]` in `prexon.toml` and `LIUM_API_KEY` in `.env`. Mention `--config`.
- **`docs/datasets.md`:**
  - "dotenv-selected validator profiles" becomes "validator target profiles".
  - `PREXON_DATASET_CACHE` becomes `[storage] dataset_cache`.
- **`docs/architecture.md:35`:** "share S3 secrets in one host `.env` and settings in `prexon.toml`".
- **`docs/miner.md:113`:** the local dev URL uses port 8080.
- **`docs/testing.md:35`:** the compose check command stays. Add that it needs no `prexon.toml`, but `up` does.

- [ ] **Step 1: Make the edits**

- [ ] **Step 2: Check for leftovers**

  Run: `grep -rn "PREXON_" README.md docs/*.md`
  Expected: only these remain:
  - the `PREXON_WALLET_*` keys;
  - `PREXON_TARGET_CACHE`;
  - `PREXON_DATA_URL` (the website's setting);
  - `PREXON_GPU_JOB_PATH` and `PREXON_NETWORK_TESTS` (test switches);
  - the migration note's description of old keys.

  Run: `grep -rn "8000" README.md docs/*.md`
  Expected: no API port references.

---

### Task 6: Final checks

- [ ] **Step 1: Run every CI step locally from an empty working directory for the CLI checks**

  ```bash
  uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest && uv build
  cd "$(mktemp -d)" && uv run --project /root/workspace/prexon-subnet prexon-validator --check-config \
    && uv run --project /root/workspace/prexon-subnet prexon-miner targets \
    && uv run --project /root/workspace/prexon-subnet prexon-evaluate --help
  ```

  Expected: all pass, and `--check-config` prints `{"targets": []}`.

- [ ] **Step 2: Report to the owner**

  Include changed files, test counts, and the Task 3 container check result. Commit nothing.

---

### Task 7: Redeploy the live upload API (only after the owner says go)

The live `prexon-api` container runs image dad7880 with `PREXON_API_*` environment variables. Those keys would be rejected by the new code. This task moves it to the new config. It touches production, so ask first.

- [ ] **Step 1: Write `/root/workspace/prexon-subnet/prexon.toml`**

  It holds only `[api] api_port = 8080` and `uploads_enabled = false`. Make it readable by UID 10001.

- [ ] **Step 2: Build the new image and start it beside the old container**

  Start it under a new name on a spare port. It runs as UID 10001 (no validator runs here yet). Mount:
  - `prexon.toml` at `/app/prexon.toml:ro`;
  - `runtime/intake-policy` at `/app/runtime/intake-policy:ro`;
  - `runtime/api` at `/app/runtime/api`.

  Do not pass any `PREXON_*` environment variables. Check `/api/health`.

- [ ] **Step 3: Swap the containers**

  Stop and rename the old container to `prexon-api-old`. Start the new one as `prexon-api` on `127.0.0.1:8080` with `--restart unless-stopped`. Check `https://prexonai.com/api/health`. Caddy needs no change.

- [ ] **Step 4: Update the memory note**

  Record the new run command in `/root/.claude/projects/-root-workspace/memory/production-deployment.md`.
