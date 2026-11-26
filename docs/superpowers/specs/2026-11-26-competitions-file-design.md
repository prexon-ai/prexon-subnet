# Competitions file: one shared list, named by model ID

Status: approved by the owner 2026-11-26. Builds on
`2026-11-26-config-modules-design.md`. Later the same day the owner replaced the single file
with a folder; see "Folder layout" below.

## Intent

One committed list of competitions that the validator, miners and the evaluator all read,
where a competition is simply named by its target model. Miners stop copying 64-character
hashes, and reward or benchmark edits no longer rename a competition. Never over-engineer.

## Decided with the owner

- `competitions.json` at the repository root, committed. Miners get it with `git pull`.
- A competition's ID is its `model_id` (for example `Qwen/Qwen3-8B`).
- Every entry has an `active` field. The validator evaluates only active competitions.
- The file only grows: old models stay with `"active": false`; new competitions are added.
  A model appears once. A new revision of a model edits that entry's `revision` (a new
  winner line starts; heads for the old revision no longer count).
- No cap on the number of competitions, and the word "slot" is gone everywhere. File order
  is the display order. Allocations of active competitions add up to 100 (checked by tests).
- The dashboard is redesigned later; its data only loses the slot fields now.

## Design

- **The file:** a JSON list; each entry is a full target profile (`TargetConfig`): `model_id`,
  `revision`, `active`, `allocation`, `methods`, engine, worker image, hardware, workloads,
  policy, optional public reference, datasets and similarity, `qualification_artifact`.
  It replaces `TARGETS` in `validator/config.py` and the `profiles/` files. The committed
  file starts as `[]`; `docs/examples/competitions.json` shows an entry.
- **Loading:** `shared/targets.py` has `load_competitions()` (every entry, in file order) and
  `find_competition(model_id)`. The file is found next to the code, not in the working folder.
- **IDs:**
  - `competition_id` = `model_id`: public, in packages, upload requests, the upload policy and
    dashboard data.
  - `identity` (the profile hash): internal, for the database, jobs and results, and dataset
    seeds. It leaves out `active` and `allocation`, so toggling or re-weighting a competition
    keeps its profile.
  - `qualification_identity`: the same hash also without `qualification_artifact`; your
    qualification record names it.
  - Internal fields that hold the profile hash are named `profile_id` (`EvaluationResult`,
    `Award`, the local evaluation report), never `competition_id`.
- **Validator:** reads every competition and evaluates the active ones. Admission matches a
  package's competition (model ID) to the active competition, so a package keeps working after
  a benchmark edit; a package for an inactive competition waits. `--qualification-profile`
  takes a model ID; `--check-config` lists the competitions.
- **Upload API:** unchanged role; it does not read the file. Its open competitions stay the
  validator's live upload policy (active and qualified only).
- **Miner:** `train` and `package` take `--competition MODEL_ID` and read the revision,
  methods and size limit from the file; `--model-id`, `--revision`, `--competition-id`,
  `--profile`, `--slot` and `--max-checkpoint-bytes` go away. `submit` checks the package
  against the file. A `competitions` command lists the file.
- **Evaluator:** an evaluation profile names `"competition": "MODEL_ID"` instead of embedding
  a target.
- **GPU worker:** unchanged; its job file carries the target.

## Tests

- The committed `competitions.json` and the example both load; model IDs are unique; active
  allocations add up to 100.
- Every test reads an empty competitions file in its temporary folder unless it writes its own.
- A leftover check fails on "slot", `TARGETS`, `build_targets` and the removed miner flags.

## Folder layout (owner change, same day)

- `competitions/` replaces `competitions.json`. `competitions/shared.json` holds what every
  competition shares: methods, engine and worker image, Lium template, workloads, policy,
  similarity, the dataset mix (category weights, buckets, minimum prompts, seed delay) and
  serving settings (dtype, context length, memory fraction, draft tokens).
- Every other file is one competition with only its own settings: `model_id`, `revision`,
  `active`, `allocation`, `hardware`, `tensor_parallel_size`, `datasets.manifest` (the
  prepared pool is tokenized for that model), optional `public_reference` and
  `qualification_artifact`. Example: `docs/examples/competition.json`.
- Loading lays each competition file over `shared.json`; nested objects merge and a
  competition may override any shared setting (owner-approved). The merged result is the
  same `TargetConfig`, so IDs and the rest of the code are unchanged.
- Competitions are listed in file name order. Changing `shared.json` changes every
  competition's profile hash, so each needs a new qualification record.

