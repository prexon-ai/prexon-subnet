# Config file, secrets-only `.env`, and light/full validator modes

> **Settings superseded 2026-11-26:** the `Settings` class, `load_settings`, startup
> validation and "moved to" errors described below were replaced by plain config modules;
> see `2026-11-26-config-modules-design.md`. The light/full modes part still stands.

Status: approved 2026-11-25; settings file changed by the owner during implementation
(see "Settings: `validator/config.py`"). On 2026-11-26 the owner moved the network and
netuid into `.env` and removed the file chain. Date: 2026-11-25.

## Intent

Two related changes to `prexon-subnet`:

1. **Settings move out of `.env`.** Today `.env.example` carries about 55 keys, of
   which only four are secrets. The owner wants `.env` to hold secrets and the
   wallet, and everything else (target models, chain, timings, storage, rental,
   upload API) in a readable, commented config file.
2. **Two validator modes.** Other validators should not need GPUs, Lium, S3 or the
   target profiles. A new **light** mode, the default, copies the weights the
   subnet owner hotkey has set on chain. **Full** mode is today's pipeline
   (admission, GPU evaluations, scoring, publication, upload policy) and is run
   only by the owner.

Light mode is the reason the config split matters: a light validator should need
only its network, netuid and wallet in `.env`.

## Decided with the owner

- `.env` holds secrets and the wallet. The wallet name was the owner's call. The
  hotkey and path join it: see "Calls made in drafting".
- `.env` also holds the chain: `PREXON_NETWORK` (`finney` or `test`, nothing else)
  and `PREXON_NETUID` (owner, 2026-11-26). The chain is always Bittensor; there is
  no local file chain.
- Other validators run light mode, and it is the default.
- Light mode copies the weights of the **subnet owner hotkey**, which is also the
  hotkey the owner validator signs with. Light validators find it on chain.
- Full mode is for the owner validator.
- The owner validator stays up, so light mode copies the owner's row verbatim.
  There is no handling for stale rows: no activity-cutoff check and no dropping
  of re-registered UIDs.
- Validators run with pm2, not Docker. Docker stays only for the upload API.
- Old `.env` files are not kept working. Third parties move to light mode and the
  owner migrates their own setup. Startup must stop and say where each moved key
  now belongs; it must never silently fall back to defaults.

## Goals

- `.env` contains only `PREXON_NETWORK`, `PREXON_NETUID`, `PREXON_WALLET_NAME`,
  `PREXON_WALLET_HOTKEY`, `PREXON_WALLET_PATH`, `S3_ACCESS_KEY_ID`,
  `S3_SECRET_ACCESS_KEY`, `S3_SESSION_TOKEN` and `LIUM_API_KEY`.
- All other validator, API and evaluation-CLI settings come from `validator/config.py`.
- `prexon-validator` defaults to light mode. A light validator needs the three
  wallet lines and `PREXON_NETUID` in `.env` (plus `PREXON_NETWORK=test` on the
  testnet).
- A light validator sets the same weights as the owner hotkey, keeps itself
  active, and never sets weights from any other source.

## Non-goals

- The miner CLI. It has no secrets, takes flags, and keeps its optional
  `--env-file` exactly as today (including `PREXON_MODEL_*` targets).
- The website repo's deploy files.
- A Docker image or compose service for the validator.
- Reloading config while running.
- Light validators checking the owner's evaluations.

## Anti-goals (failure even if it otherwise works)

- A light validator setting weights that did not come from the owner hotkey's
  current on-chain row: no zeros, no burn-only fallback, no locally computed
  weights.
- A secret appearing in `validator/config.py` or in any committed example.
- An upgraded validator running with defaults (for example netuid 0) because its
  old keys were ignored.
- Any change to competition IDs (the hash of each `TargetConfig`).

## Approaches considered

**Where non-secret settings live**

- **`validator/config.py` + secrets-only `.env` (picked by the owner during
  implementation).** A committed, commented Python module of settings that reads
  `.env` itself; every command takes its settings from it. It allows comments, real
  lists for targets, and paths anchored to the checkout.
- TOML file (`prexon.toml`): picked first and built, then replaced at the owner's
  request.
- YAML: adds a dependency and has type surprises (`no` becomes false).
- A second env-style file: least code, but stays flat `PREXON_MODEL_1_*` strings
  with no structure for targets.
- Config file with environment-variable overrides: rejected. It would keep two
  sources for every setting, which is what this change removes.

**Where light validators get weights**

- **The owner hotkey's on-chain weights (picked).** This is trustless, and it is
  exactly what consensus scores against.
- The `emissions` shares in the published `dashboard/snapshot.json`: trusts the
  storage host, can lag or be unreachable, and is not what the chain sees.
- Re-running evaluations: not light.

**How light mode is packaged**

- **A mode inside `prexon-validator` (picked).** One command, one image, one guide.
- A separate command: two entry points to document for no gain.

**How light validators know the owner**

- **Read `get_subnet_owner_hotkey(netuid)` on chain each interval (picked, owner's
  answer).** No setting, and it follows a hotkey swap.
- Ship a hotkey constant: needed only if the owner validator used a separate
  hotkey, which it does not.

## Design

### Secrets, wallet and chain: `.env`

Allowed keys are the nine in Goals. The validator reads the network, netuid and
wallet; the API and evaluation CLI ignore them. `S3_ENDPOINT_URL` and `S3_REGION` stay
accepted, because the store already forces them to the Hippius values, but they
leave `.env.example`. Secrets keep being read the way they are today, through
`read_environment` (S3 store, Lium provider, upload API, evaluation CLI). The
wallet keys keep their current names, so existing lines keep working.

For the validator, API and evaluation roles, any `PREXON_<FIELD>` key for a
setting that moved, and any `PREXON_MODEL_{1,2}_*` key, is an error if found in
the `.env` file **or** the process environment. The error names the new location,
for example `PREXON_POLL_SECONDS moved to validator/config.py POLL_SECONDS`, and
lists all offending keys at once. `PREXON_CHAIN` and `PREXON_CHAIN_FILE` are
reported as removed. Unrelated `PREXON_*` variables are not touched, such as
`PREXON_TARGET_CACHE`, which GPU workers read.

### Settings: `validator/config.py`

- **The file:** `validator/config.py`, committed. Operators edit the values in place
  and restart the validator and the API. Constants are the `Settings` field names in
  upper case (`BURN_UID`, `API_PORT`, `STATE_PATH`); targets are a `TARGETS` list of
  dicts. Its `load_settings(role, env_file)` reads `.env` (default: the checkout's
  `.env`) and returns that role's `Settings`. The validator, the upload API and
  `prexon-evaluate` call it; code below the entry points keeps using `Settings`.
- **Paths:** anchored to the checkout (`ROOT`, `RUNTIME = ROOT / "runtime"`), so the
  working directory does not matter. In the API container the checkout is `/app`.
- **Validation:** an upper-case name that is not a setting stops startup
  (`NETUD: unknown setting`), as does a secret written into the file
  (`LIUM_API_KEY belongs in .env`, `WALLET_NAME belongs in .env as
  PREXON_WALLET_NAME`, `NETUID belongs in .env as PREXON_NETUID`) or an unknown key in a target. A wrong type names the
  constant. Each role type-checks only the settings it reads.
- **Settings it holds:** chain (`BURN_UID`, `WEIGHTS_VERSION`), `TARGETS` (`slot`, `model_id`, `revision`,
  `allocation`, `methods`, `profile`), schedule, storage, GPU rentals and upload API,
  (`FIELDS` in `shared/settings_loader.py` lists them). `upload_api_url` stays
  miner-only.

**Targets.** Each `TARGETS` entry replaces the five `PREXON_MODEL_{n}_*` variables,
with the same meaning and checks: `profile` points at the JSON profile; `slot`,
`model_id`, `revision`, `allocation` and `methods` are merged into it; allocations of
enabled targets sum to 100; models are distinct; slots are 1 and 2 and not
repeated. An empty `model_id` disables an entry. The same values produce
byte-identical `TargetConfig`s as the miner's env loader, so competition IDs do not
change.

**Roles.** The API reads the upload-API settings plus `PUBLIC_ARTIFACT_ROOT`,
`UPLOAD_STAGING_ROOT` and `UPLOAD_POLICY_PATH`. The evaluation CLI reads the rental
settings, `SESSION_SECONDS`, `SESSION_MAX_COST` and `DATASET_CACHE`, and rejects only
moved `PREXON_*` keys for those (it also runs on miners' machines, whose own miner
`.env` may hold `PREXON_MODEL_*`). The validator reads everything. The light-mode
plan adds `MODE` and `COPY_INTERVAL_SECONDS`.

### Light mode

**Startup**
- Requires a non-zero netuid; otherwise exit with a message naming `PREXON_NETUID`
  in `.env`.
- Loads only the wallet. No database, artifact store, Lium, profiles or upload
  policy is opened, and no S3 or Lium secret is needed.
- If the wallet's hotkey **is** the subnet owner hotkey, exit with "the owner
  validator runs full mode".

**Every `copy_interval_seconds`**
1. Read the finalized block `b`.
2. Read `get_subnet_owner_hotkey(netuid, block=b)` again, so a hotkey swap is
   followed.
3. From the metagraph at `b`, find the owner's UID and our own UID. If either is
   missing, log why and skip this interval.
4. Read `weights(netuid, block=b)` and take the owner's row, keeping non-zero
   entries only. If it is empty, log it and skip. Never substitute anything.
5. Decide whether to send:
   - send if the owner's row differs from the last row this process sent, which
     is held in memory and is empty after a restart;
   - otherwise send if `blocks_since_last_update(netuid, our_uid)` is at least
     `tempo(netuid)`, to stay active;
   - never send while `blocks_since_last_update` is at most `weights_rate_limit`.
     The SDK only sends when it is strictly greater, so wait for a later
     interval.
   - skip, with the reason `owner_row_not_copyable`, when the chain would reject
     the row from a copier: fewer UIDs than `min_allowed_weights`, or a largest
     share above `max_weight_limit`. Mirror subtensor's own check. This happens
     at launch, when the owner's row is 100% on `burn_uid`, which is the owner's
     own UID. That is a self-weight for the owner, which is exempt from both
     limits, but not for a copier. Owner action: keep `min_allowed_weights = 1`
     and `max_weight_limit = 65535` on the subnet.
6. Send through the existing weight-publication path (`set_weights` with
   `version_key = weights_version`, waiting for inclusion and finalization). Use
   the same UIDs and the same proportions as the owner's row. On success, remember
   the row as sent.
7. Any chain error is logged and retried next interval; the loop never exits on
   one. Log one line per interval with the outcome (copied N UIDs at block `b`,
   or the skip reason).

**Stale rows.** The owner decided the owner validator stays up, so the owner's
row is copied as it is. The owner validator sets weights every `publish_seconds`
(600 by default). A miner re-registered between two owner updates can therefore
be paid by copiers until the owner's next update; this is accepted.

**Commit-reveal.** If the subnet has commit-reveal on, `set_weights` in SDK
10.2.1 commits and reveals by itself, and `weights()` returns revealed weights.
Copies then trail the owner by the reveal period. Nothing extra is built for it.

### Full mode

Behaviour is unchanged: today's controller. One addition (see "Calls made in
drafting"): full mode refuses to start unless the wallet's hotkey is the subnet owner hotkey. The check runs only when
the controller loop will touch the chain (today's `has_chain` condition). It does
not run for an idle install that has never had a chain context, which keeps
working without credentials (`test_idle_controller_runs_without_external_credentials`).
It also does not run for `--check-config`, `--qualification-profile`,
`--publish-qualification` or `--requeue-stalled`. Tests inject a chain double.

### Commands

- `prexon-validator`: `--env-file` (default: the checkout's `.env`). `--check-config` prints the mode, network and netuid, plus targets and
  competition IDs in full mode. It must keep passing in CI with no `.env`
  present (`uv run prexon-validator --check-config` is a CI step).
- `prexon-api`: `--env-file`, and a new `--host` that overrides `API_HOST`.
  Containers pass `--host 0.0.0.0`, so one config file works both on the host and
  in a container without environment overrides.
- `prexon-evaluate`: `--env-file`.
- `prexon-miner`: unchanged.

### Running the validator: pm2, no Docker

The owner decided validators run under pm2 from a repo checkout, and Docker stays
only for the upload API.

- **Remove** `docker/Dockerfile.validator` and the validator service in
  `docker/compose.yaml`.
- **Run from the checkout.** The validator runs from the checkout after
  `uv sync --frozen --extra validator`. The guide gives the pm2 commands; there is
  no ecosystem file.
  - Light: `pm2 start .venv/bin/prexon-validator --name prexon-validator --interpreter none`
  - Full: the same command plus `--kill-timeout 120000`
- **Working directory.** The pm2 command uses the checkout's `.venv`; settings,
  `.env` and `runtime/` paths are anchored to the checkout by `validator/config.py`.
- **Kill timeout (full mode).** On shutdown the full validator stops its active
  GPU sessions, which releases the Lium rentals. Compose gave it 120 seconds
  (`stop_grace_period`), but pm2's default kill timeout is 1.6 seconds. Without
  `--kill-timeout 120000`, pm2 would kill it mid-cleanup and rented GPUs could keep
  running. pm2 stops with SIGINT, which the validator already handles.
- **Wallet.** The validator reads the wallet from `PREXON_WALLET_PATH`
  (default `~/.bittensor/wallets`) directly.
- **Logs.** Log rotation is the operator's choice (`pm2-logrotate`); the guide
  mentions it.

### Docker: upload API only

- **API image:** remove the `PREXON_*` `ENV` lines from `docker/Dockerfile.api`,
  since they would now be rejected. Copy `validator/config.py`; its paths resolve
  under `/app/runtime`, owned by UID 10001. The default arguments include `--host 0.0.0.0`.
- **Compose:** `docker/compose.yaml` keeps only the `api` service, with no profile.
  It mounts `../validator/config.py` read-only at `/app/validator/config.py`, and
  `../runtime/intake-policy` (read-only) and `../runtime/api` under
  `/app/runtime`. The validator writes the policy file from the host into
  `runtime/intake-policy`, so the mount points at the same place. It passes only
  the S3 secrets and publishes `127.0.0.1:8080:8080`.
- **API port default:** the `api_port` default becomes 8080, matching the owner's
  deployment.
- **GPU worker image:** `docker/Dockerfile.worker` stays. It is not a service on
  the validator machine. It builds the image that rented Lium GPUs run for
  evaluations, referenced by digest in each target profile's `worker_image` and
  in the Lium template. Full mode cannot evaluate without it.

### Migration

- **This machine:** no validator runs here yet. The live `prexon-api` container
  is configured with `PREXON_API_*` environment variables. After this change,
  redeploy it with the checkout's `validator/config.py` mounted (`API_PORT = 8080`,
  `UPLOADS_ENABLED = False`, the defaults) and `--host 0.0.0.0`. The published port
  `127.0.0.1:8080` and the Caddy routes stay as they are.
- **Other validators:** their old `.env` stops startup with the list of moved
  keys. The validator guide leads with the light-mode setup: wallet lines,
  `PREXON_NETUID`, run.

### Docs and examples

- Update `README.md`, `docs/validator.md` (modes; light setup first; full setup
  for the owner), `docs/docker.md`, `docs/api.md` (the upload configuration table
  becomes config keys) and any env-variable references in `docs/evaluation.md`.
- Replace `.env.example` with the nine keys. `validator/config.py` is itself the
  commented reference for every setting.

### Tests

**Config**
- `validator/config.py` constants load into the right `Settings` fields, and the
  committed file loads for every role.
- An unknown constant, a secret in the file, and an unknown target key each give the
  specific error.
- A moved key in `.env` or the process environment gives an error naming its new
  location.
- Wallet values come from env.
- The miner still reads its env file as before.
- API and evaluation roles load only their keys.
- The same target values give the same competition ID as the current loader.

**Light mode** (fake Subtensor client)
- Copies the owner's row.
- Skips, sending nothing, when the owner is unregistered, the owner has no
  weights, or we are unregistered.
- Refuses to run on the owner hotkey.
- Follows an owner hotkey swap.
- Sends again after a tempo and does not send an unchanged row sooner.
- Respects the rate limit.
- A chain error does not stop the loop.

**Full mode**
- Refuses a non-owner hotkey.

**Commands and Docker**
- `prexon-validator --check-config` passes with no files.
- The compose test (`test_compose_validator_writes_only_to_its_data_volume` in
  `tests/shared/test_config.py`) is replaced by a test of the API-only compose
  file: config mounted read-only, policy directory read-only, only the S3 secrets
  passed. References in `README.md`, `docs/docker.md` and `docs/testing.md` are
  updated.

## Constraints

- Competition IDs must not change.
- No new dependencies (`tomllib` is in the standard library).
- CI must stay green: ruff, ruff format, mypy with `disallow_untyped_defs`,
  pytest, and the CLI smoke steps in `.github/workflows/ci.yml`.
- Bittensor SDK 10.2.1: `get_subnet_owner_hotkey`, `weights`,
  `blocks_since_last_update`, `tempo`, `weights_rate_limit` and `set_weights` all
  exist with the signatures this design relies on.

## Left to the builder

- Module and class names, and the shape of the chain interface for reading the
  owner's weights.
- Log and error wording, within the rules above.
- Layout and comments of `validator/config.py`.
- Whether light mode takes a process lock.

## Calls made in drafting (for the owner to check)

1. The wallet hotkey and path live in `.env` with the wallet name, because
   together they identify the signing key.
2. Config constants are the existing setting names in upper case (`API_PORT`).
3. Environment variables cannot override the config file. The API gets a `--host`
   flag for containers instead.
4. Full mode on a live chain refuses to start unless the wallet hotkey is the
   subnet owner hotkey.
5. Light mode checks every 300 seconds and refreshes its weights at least once per
   tempo.
6. The `api_port` default becomes 8080, and compose publishes `127.0.0.1:8080`.
7. The miner CLI is unchanged.
8. `NETUID` is 0 (not set) in the committed file until the owner sets it; light
   validators then need nothing but `.env`.
9. Light mode skips rows the chain would reject from a copier (too few UIDs, or a
   share above `max_weight_limit`) instead of sending them and failing.
10. `docker/Dockerfile.worker` stays, because rented GPUs need it for evaluations.
11. pm2 is documented as commands, with no ecosystem file, and full mode uses
    `--kill-timeout 120000` so shutdown can release GPU rentals.
