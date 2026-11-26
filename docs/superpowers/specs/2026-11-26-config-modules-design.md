# Config modules: values in `config.py`, secrets in `.env` sections

Status: approved by the owner 2026-11-26. Replaces the settings machinery of
`2026-11-25-config-file-and-validator-modes-design.md` (the `Settings` class,
`load_settings`, `shared/settings_loader.py`, startup validation and "moved to" errors).
Light/full validator modes stay a later plan.

## Intent

One plain configuration system the owner can read at a glance. The owner is the only
operator, so values are checked by tests, not at startup. Never over-engineer.

## Decided with the owner

- `.env` has sections; each machine fills only what it uses.
  - **Validator** (owner's server): `PREXON_NETWORK`, `PREXON_NETUID`,
    `PREXON_WALLET_NAME`, `PREXON_WALLET_HOTKEY`, `PREXON_WALLET_PATH`,
    `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` (the session token was removed later the
    same day: Hippius keys are long-lived).
  - **Evaluation** (validator and miners): `LIUM_API_KEY`, `LIUM_TEMPLATE_ID`,
    `LIUM_VOLUME_ID`, `SSH_PRIVATE_KEY`, `SSH_PUBLIC_KEY`.
- No settings class. Code imports values directly: `from validator.config import POLL_SECONDS`.
- Only config files read `.env`; every other module imports from a config file.
- No validation of config values. Tests check the committed values.
- Miner commands take flags and read no `.env` section of their own.

## Design

- `validator/config.py`: values for the validator and the upload API, plus the Validator
  section of `.env`. Plain constants and `ROOT` for paths. `.env` is read with
  `dotenv_values`, with the process environment on top (Docker passes the S3 keys as
  environment variables). Empty values fall back to the defaults.
- `evaluation/config.py` (new): Lium, SSH, the dataset cache and development-round
  defaults, plus the Evaluation section of `.env`. Used by `prexon-evaluate` and by the
  validator's GPU rentals.
- `UPLOAD_MAX_BYTES` (upload API, miner default, validator download cap) lives in
  `validator/config.py`, and the miner CLI repeats it as its default; a test keeps the two
  equal. (A `shared/config.py` for it was removed later the same day, as was
  `miner/config.py`: the miner takes every setting as a flag.)
- `shared/targets.py`: target profile models and `build_targets`, moved out of
  `shared/config.py` unchanged, so competition IDs do not change.
- The Lium client and SSH transport move from `shared/` to `evaluation/` and import
  `evaluation.config`. `LiumClient` takes the API key and the spend cap as arguments,
  because the validator's and the evaluator's budgets differ.
- `shared/chain.py` and `ArtifactStore` take plain arguments (network, netuid, wallet
  fields, S3 keys); the miner passes its flags and the validator passes its config.
- Values computed at startup are passed as arguments: the validator's targets, built once
  from `TARGETS` and their profile files, go to the controller and scheduler.
- Removed: `Settings`, `shared/settings_loader.py`, `read_environment`,
  `load_miner_settings`, "moved to" errors, every `--env-file` flag, `S3_ENDPOINT_URL`
  and `S3_REGION`.
- Kept: the startup checks in `validator/main.py` that the netuid is set and that enabled
  targets use the three Hippius stores. They protect the live subnet, and the local
  hash-only admission shortcut is safe only because of the second.

## Tests

- `tests/conftest.py` blanks every `.env` key in the environment before code is imported,
  so tests never read the owner's `.env` values.
- A `set_config` fixture changes a value in every module that imported it.
- Tests check the committed values: enabled allocations sum to 100, slots and models are
  distinct, the three buckets differ, the public URL is HTTPS, `.env.example` holds exactly
  the two sections' keys.
- A test checks that only the two config files read `.env`.

## Left to the builder

Names of the moved modules and how tests are reorganized.
