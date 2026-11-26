# Config modules implementation plan

**Goal:** values in `validator/config.py` and `evaluation/config.py`, secrets in `.env`
sections, no settings class. **Spec:**
`docs/superpowers/specs/2026-11-26-config-modules-design.md`. Executed natively, in
place on `main`, uncommitted. The suite stays green after every task.

## Tasks

1. **Config files and test guards.** Write `evaluation/config.py`; turn
   `validator/config.py` into plain values that read the Validator section; rewrite
   `.env.example` with the two sections. Add the conftest blanking, the `set_config`
   fixture, and tests for both config files (subprocess with environment values), the
   committed values and `.env.example`. The old `load_settings` stays until task 7.
2. **Target models.** Move the profile models and `build_targets` to `shared/targets.py`
   and update imports. `shared/config.py` keeps `Settings` for now.
3. **Shared plumbing.** Move `shared/lium.py` and `shared/ssh.py` to `evaluation/`, reading
   `evaluation.config`; `LiumClient(api_key, max_cost)`. `shared/chain.py` and
   `ArtifactStore` take plain arguments.
4. **Evaluator.** `evaluation/cli.py` and `evaluation/lium.py` read `evaluation.config`;
   the profile's session overrides become arguments. No validator imports.
5. **Validator.** `main`, `controller`, `chain`, `scheduler`, `admission`, `submissions`
   and `providers` import from `validator.config`; targets are passed as an argument.
6. **API and miner.** The API reads `validator.config`; the miner passes its flags as
   arguments and loses `--env-file`.
7. **Cleanup.** Delete `Settings`, `settings_loader.py`, `read_environment` and
   `load_miner_settings`; `shared/config.py` keeps `UPLOAD_MAX_BYTES`. Add the "only config
   files read .env" test. Update docs, Docker check, final review.
