# Architecture

Prexon separates miner tooling, benchmark execution, validator authority, and
upload authorization. Miners and validators both use the `evaluation` library;
it depends on shared contracts and transports, not on validator or miner
implementations.

## Submission flow

1. The miner freezes a data-only checkpoint archive and signs an upload request.
2. The API verifies subnet membership, competition, size, freshness, replay and
   quota limits, then returns a short-lived Hippius PUT URL.
3. The miner uploads successfully before committing the archive hash on chain.
4. The validator discovers the finalized commitment, downloads that hotkey's
   staging object, verifies its hashes and qualified schema, and checks similarity
   on CPU. Accepted bytes are copied to validator-owned canonical storage.
5. The scheduler claims an eligible target session, freezes its evaluation inputs,
   rents matching hardware, and processes submissions in finalized order.
6. The controller validates worker evidence, promotes winners, publishes their
   exact evaluated packages, and submits emission weights.

A storage upload does not create a queue entry. Finalized chain commitments supply
identity and priority; only validated candidates become GPU-ready.

## Storage and authority

| Storage | Contents | Access |
| --- | --- | --- |
| Uploads bucket (private) | Hotkey-scoped unchecked uploads | API-issued PUT capabilities; validator reads |
| Private bucket | Verified packages, qualification, jobs, plans, results | Validator; workers receive scoped URLs |
| Public bucket | Verified promoted packages and the dashboard's data | Public HTTPS reads |
| Validator SQLite | Discoveries, fingerprints, sessions, jobs, awards, releases | Controller thread only |
| API SQLite | Upload nonce and quota state | API only |

The validator and trusted upload API share S3 settings in one host `.env`.
Compose limits the API's file mounts to the read-only upload policy directory and
its own replay database. Public data is read from the public bucket, never from
the validator's machine. Workers receive no wallet, rental credentials, or
database.

Signed object metadata binds the API-authorized competition to the archive
manifest. A digest under a different hotkey cannot claim another miner's upload.
Private fingerprints, losing heads' checkpoints, loss reasons and detailed losing
measurements stay private. Public views contain released winners and, for every
model, its queue and each evaluated submission with miner, time, result and
speedup.

## Scheduling and recovery

Each of the two optional target slots has its own queue, GPU requirements,
incumbent, and emission allocation. A session starts at 10 ready submissions or
six hours after the oldest unevaluated ready commitment. Original chain/discovery
age survives retries. A submission whose evaluations repeatedly end without a
verdict is set aside instead of blocking the queue, and sessions that reach no
verdict are retried with a growing wait. One durable session claim per target
prevents overlapping
rentals; different targets may run concurrently.

Profiles and dataset plans are immutable per job. Dataset membership freezes
before a fresh finalized sampling cutoff and later block entropy are recorded.
All candidates and references use the same selected token inputs. Late admissions
wait for the next round. See [datasets](datasets.md).

Background admission and publication workers perform CPU/storage work while the
controller continues polling. They receive copied inputs and return results;
SQLite finalization stays on the controller thread. A process lock prevents
multiple controllers owning one state file.

Rental IDs, requests, deadlines and cleanup status are persisted. Cleanup retries
retain ownership until deletion or provider expiry is established. Promotion,
public copying and chain weight publication are separate idempotent operations;
retries cannot duplicate awards or change their place in the reward order.

## Standalone evaluation

`prexon-evaluate` measures one candidate in `head` or `full` mode on Lium or an
existing pod. It uses the same measurement/scoring code through a separate
execution lifecycle. Its private rental journal and reports have no authority over
production queues, winners, or emissions. See [evaluation](evaluation.md).
