# Competitions file implementation plan

**Spec:** `docs/superpowers/specs/2026-11-26-competitions-file-design.md`. Executed natively,
in place on `main`, uncommitted (on top of the uncommitted miner-flags and shared/config
removals). The suite stays green after every task; TDD for each behaviour change.

1. **Targets and the file.** `TargetConfig`: drop `slot`, add required `active`, add
   `competition_id` (= `model_id`), hashes leave out `active`/`allocation`. Add
   `load_competitions()`/`find_competition()`, committed `competitions.json` (`[]`),
   `docs/examples/competitions.json`; remove `build_targets`, `TARGETS`. Conftest points every
   test at an empty file in tmp. Validator `main` uses active competitions;
   `--qualification-profile MODEL_ID`; `--check-config` lists competitions.
2. **IDs in messages.** Public `competition_id` = model ID in the manifest, upload request and
   policy, and dashboard data; `EvaluationResult`, `Award` and the local report use
   `profile_id`. Update state, rewards, publication, dashboard, history; remove slot fields.
3. **Admission.** Map packages to active competitions by model ID; `pump`/`admit_commitment`
   take the active competitions.
4. **Miner.** `--competition`, `competitions` command, removed flags; `submit` checks the file.
5. **Evaluator.** Profiles name `"competition"`.
6. **Docs and guard.** Docs and examples; leftover check; full checks; final review.
7. **Folder (owner change).** `competitions/shared.json` plus one file per competition, merged on load; tests, examples and docs follow.
