# Runbook

Operational procedures. What to check, what it means, and what to do about it.

## Running it

### Local, no database

The in-memory checkpointer needs nothing. The workflow runs, gates park, and
resumes work within one process.

```bash
make venv
awf demo --interactive
```

Caveat worth knowing before you try to demo anything else: the in-memory
checkpointer is per-process, so a run started by `awf demo --memory` is not
visible to a later `awf replay --memory`. For anything that needs a run to
survive the process that made it, use PostgreSQL.

### With PostgreSQL

```bash
docker compose up -d postgres
awf demo            # against the durable store
awf replay <run_id> # in a different process, and it works
```

### The API

```bash
docker compose up              # postgres + the built image
curl -s localhost:8000/health/ready | jq
open http://localhost:8000/docs
```

The image is multi-stage, runs as a non-root user, and has no build tooling in
the runtime layer. `docker compose up` is the whole setup.

## First things to check

| Check | Command | Healthy looks like |
| --- | --- | --- |
| Is the process up? | `GET /health/live` | `200` with `"status": "ok"` |
| Can it reach its dependencies? | `GET /health/ready` | `200`; `503` names what is missing |
| Is the graph wired? | `awf topology` | Seven nodes and the routing table |
| Is the checkpointer reachable? | `awf janitor --dry-run` | A non-zero `examined` count |
| Are gates flowing? | `GET /v1/approvals/stats` | `pending` moves |

`/health/live` and `/health/ready` are deliberately different. Liveness answers
"should this process be restarted"; readiness answers "can it serve a request".
Conflating them is how a database outage turns into a crash loop.

Neither is rate limited. A liveness probe that can be rate-limited is a liveness
probe that reports the process unhealthy because someone else is busy.

## Common situations

### A run is parked and nobody knows why

```bash
curl -s localhost:8000/v1/runs/$RUN | jq '{status, pending_approval}'
curl -s localhost:8000/v1/approvals/$APR | jq '{stage, reason, rationale, options}'
```

`reason` is the `EscalationReason`: `always` (irreversible stage),
`low_confidence`, `high_severity`, or `policy` (unapproved verdict, or the loop
ran out of iterations). `rationale` is the sentence shown to the human.

### The approval inbox is growing

```bash
curl -s localhost:8000/v1/approvals/stats | jq
```

`pending` rising without `resolved` means nobody is answering. Either drain it:

```bash
curl -sX POST localhost:8000/v1/approvals/sweep \
  -H 'content-type: application/json' \
  -d '{"decision":"reject","reviewer":"janitor","comment":"backlog sweep"}'
```

or turn the policy down (`AWF_HITL_ESCALATION_THRESHOLD`), which is a better
answer than approving a queue nobody read. Decide deliberately: a `reject` sweep
is a real decision recorded against every run it touches.

### Everything is 500ing

```bash
docker compose logs --tail=100 api
```

Then, in order:

1. **`persistence_error: the pool is not open yet`** — the app started before
   PostgreSQL was accepting connections. The API fails its boot deliberately
   rather than serving a half-initialised app. Fix the database, not the app.
2. **`ConfigurationError`** — an environment variable is wrong. The message
   names the offending field.
3. **`RateLimitExceeded`** (429) — `AWF_API_RATE_LIMIT_PER_MINUTE` is too low for
   your client, or something is looping.

### Postgres-backed runs cannot be read

Authoritative reads go to the checkpointer; the run registry is a rebuildable
projection, and `startup()` rebuilds it by asking the store which threads exist.
If a read fails, check the connection first — do not rebuild anything.

Three things follow from the registry being per-process, and all three were
actual bugs before they were written down:

- **A restarted process lists nothing until it has hydrated.** That is
  `startup()`'s job and it is bounded by `AWF_RECOVERY_MAX_RUNS` (default 1000).
  On a database with more live threads than that, the listing shows the most
  recent N and **every run can still be read, resumed and cancelled by id** — the
  bound costs you the listing, not access. Raise it if your listing matters.
- **`engine.registry_rehydrated unreadable=N`** in the boot log means N threads
  were found but their state could not be read. Those runs are registered as
  `pending`. They are not lost; they are not yet described.
- **A status outside the lifecycle vocabulary is rejected, not stored.** If you
  see a `ValueError: invalid run status` at boot, something wrote a graph status
  (`triaged`, `reviewing`) into a field whose vocabulary is the engine's
  (`waiting_human`, …). That is a bug in the writer, and it is worth chasing: an
  unregistered status is not in `TERMINAL_STATUSES`, so every poller waiting on
  that run waits forever.

### Disk is filling up

Checkpoints accumulate. The janitor removes whole thread histories older than
the retention window:

```bash
awf janitor --dry-run                 # what would go
awf janitor --retention-days 7        # do it
awf janitor --retention-days 7 --json | jq
```

Schedule it. `RetentionReport` separates `examined`, `stale`, `deleted` and
`undated`; `undated` is the one to watch — it counts threads whose age could not
be determined, and the janitor **keeps** them.

That is deliberate: the sweep fails closed. An undecidable age resolves to
"keep", never "delete", and is counted and logged so it is visible rather than
silent. A retention job that deletes on a parse error is a retention job that
eventually deletes the wrong thing.

### I need to know what a model actually did

```bash
awf replay $RUN --limit 50      # every checkpointed super-step
awf replay $RUN --index -1      # fork from the first one
```

Replay branches: the original history is never modified, so the comparison
between what happened and what would happen is honest. Same facility over HTTP:

```bash
curl -s localhost:8000/v1/threads/$RUN/history | jq
curl -sX POST localhost:8000/v1/threads/$RUN/replay \
  -H 'content-type: application/json' -d '{"reason":"checking a regression"}'
```

Replay takes a `checkpoint_id` and a `reason`, and forks as a **sibling** thread
rather than overwriting the original.

## Where the time goes

```bash
curl -s localhost:8000/v1/runs/$RUN/timings | jq '{items, total_ms, slowest_node}'
```

`total_ms` is the sum of node time; `slowest_node` names the one to look at
first. Note that node time is not the same as wall-clock — on this workflow the
human-in-the-loop machinery (park, checkpoint, ask, read back, resume) costs
more than the node work itself. `benchmarks/bench.py` measures that gap
explicitly as `gate_overhead`; the per-run `timings` endpoint shows only the
nodes.

## Verifying a change did not break quality

```bash
awf eval                       # full report, exits 1 if a threshold is missed
awf eval --gate invariants     # the offline gate CI uses
awf eval --json | jq .metrics
```

The `--gate` distinction matters when you are reading a number. Under
`--gate all` (the default) the suite includes `recall`, and the bundled `echo`
provider scores 0.150 on it — so the default gate *always fails out of the box*.
That is not a broken build; it is a null model doing what a null model does. A
real provider has to beat 0.150 for this to be worth anything.

Under `--gate invariants` the suite checks the properties that hold whatever
produced the report — no fabricated quotes, no invented paths, no approving over
your own critical finding — and the offline provider passes 20/20.

## Configuration reference

Everything is `AWF_`-prefixed. The ones worth knowing by heart:

| Variable | Default | Effect |
| --- | --- | --- |
| `AWF_ENVIRONMENT` | `development` | `production` disables `/docs` and raises log level. |
| `AWF_POSTGRES_ENABLED` | `false` | Turn on to persist checkpoints. |
| `AWF_POSTGRES_DSN` | — | Required when enabled. |
| `AWF_POSTGRES_SCHEMA` | `public` | Schema holding the checkpoint tables. Must be a bare SQL identifier. |
| `AWF_POSTGRES_AUTO_SETUP` | `true` | Create the schema and the tables on boot. Turn off if migrations are managed separately. |
| `AWF_RECOVERY_MAX_RUNS` | `1000` | Ceiling on runs rehydrated into the registry on boot. Bounds the listing, never access to a run by id. |
| `AWF_LLM_PROVIDER` | `echo` | `echo` (offline, deterministic) or `openai_compatible`. |
| `AWF_LLM_API_KEY` | — | Required for a real provider. |
| `AWF_HITL_ENABLED` | `true` | Master switch for human gates. |
| `AWF_HITL_ESCALATION_THRESHOLD` | `0.70` | Confidence below which a human is asked. |
| `AWF_MAX_ITERATIONS` | `6` | Feedback-loop budget before the run is escalated. |
| `AWF_STATE_RETENTION_DAYS` | `30` | Retention window for the janitor. |
| `AWF_API_RATE_LIMIT_PER_MINUTE` | `120` | Per-client request budget. |
| `AWF_LOG_FORMAT` | `console` | `json` for machine-readable logs. |

`Settings` **refuses** `awf_`-prefixed keyword arguments, so
`Settings(AWF_LOG_LEVEL="DEBUG")` is a loud `ConfigurationError` rather than a
setting that quietly stays at its default. Pass `log_level="DEBUG"`.

## Logs

`AWF_LOG_FORMAT=json` for machine-readable output; console format renders for a
terminal. Logs go to **stderr**, always — the CLI writes its result to stdout, so
`awf eval --json | jq` works and a log line never lands inside a JSON document.

Every log line carries a run id, node, and duration where one applies, so a slow
run can be traced without a debugger.

## Backups

The checkpoint store is the state. Everything else is derived:

- The run registry is a projection; losing it loses no runs. It is rebuilt on
  boot, so a restored database does not need a second restore step.
- Decision signatures are verifiable against a secret, so a restored database
  can be checked for tampering: `GET /v1/approvals/by-run/{run_id}/audit` reports
  `unverified` alongside `decisions`.

Back up PostgreSQL. If you back up nothing else, that is sufficient. Losing the
signing secret is different — it does not lose decisions, but it makes them
permanently unverifiable.
