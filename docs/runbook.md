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

## Behind a reverse proxy

If the API is not exposed directly, one setting decides whether the request
limiter works at all: `AWF_API_TRUST_FORWARDED_FOR`.

**Off by default, and it should stay off unless a proxy really is in front.**
The limiter keys on the address the server learned, not the one it was told.
`X-Forwarded-For` is written by the caller, so honouring it means the caller
chooses its own budget — a rotation of that header defeats the limit entirely.

Turn it on when the proxy overwrites the header rather than appending blindly,
and when you have exactly one trusted hop in front of the service. The rightmost
entry is the one used, so a client that pre-forges its own header is ignored; a
chain of two or more proxies needs a different arrangement, because a single
boolean cannot say how many hops to skip.

The consequence of leaving it off is a shared budget, not a broken one: every
request appears to come from the proxy, so the limit is effectively per-process.
That is usually the right trade, and it is why the setting is opt-in.

## First things to check

| Check | Command | Healthy looks like |
| --- | --- | --- |
| Is the process up? | `GET /health/live` | `200` with `"status": "ok"` |
| Can it reach its dependencies? | `GET /health/ready` | `200`; `503` names what is missing |
| Is the graph wired? | `awf topology` | Seven nodes and the routing table |
| Is the checkpointer reachable? | `awf janitor --dry-run` | A non-zero `examined` count |
| Are gates flowing? | `GET /v1/approvals/stats` | `pending` moves |

For approval counts, use `/v1/approvals/stats` and not the probe — see
[What the readiness probe's approval numbers mean](#what-the-readiness-probes-approval-numbers-mean).

`/health/live` and `/health/ready` are deliberately different. Liveness answers
"should this process be restarted"; readiness answers "can it serve a request".
Conflating them is how a database outage turns into a crash loop.

Neither is rate limited. A liveness probe that can be rate-limited is a liveness
probe that reports the process unhealthy because someone else is busy.

### What the readiness probe's approval numbers mean

`/health/ready` reports an `approvals` block, and it is **not** the same as
`/v1/approvals/stats`. Read the table before trusting it during an incident:

| Counter | `/health/ready` | `/v1/approvals/stats` |
| --- | --- | --- |
| `pending` | This process's parked runs | Every known run, read from the store |
| `by_stage` | Same, derived from the approval ids | Same, from the approval objects |
| `expired` | Always `0` | Real count |
| `resolved_pending` | Always `0` | Real count |

The two zeros are not "nothing is wrong". The probe counts from the run registry,
which records *that* a run is parked and *which* approval, and holds neither the
expiry date nor the decision log — so it reports zero rather than guess. Use
`/v1/approvals/stats` for those two numbers.

The registry is per-process, so a run parked by another replica is missing from
`pending` here. That is the same caveat the run count in the same payload already
carries.

Why the split: the probe is polled every few seconds whether or not anything is
wrong. Building the inbox to count it cost, against a real PostgreSQL, 181 ms per
probe at 25 parked runs and 5,788 ms at 800 — past the 5 s budget, which meant a
healthy instance reporting `503` and being dropped from the load balancer at
around 700 parked runs. Counting from the registry is 2.3 ms and flat. The
dashboard endpoint is read by a person, occasionally, and can afford the sweep.

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

### A deploy interrupted in-flight runs

Shutting the process down stops whatever it was carrying. Those runs are recorded
as **`interrupted`**, not `cancelled`, and the boot of the next process logs:

```
engine.shutdown_interrupted runs=3 run_ids=['a1b2…', 'c3d4…', 'e5f6…']
```

That line is the work that needs a human. Anything not named there completed,
parked on its own, or failed on its own terms.

**Recovering one is a replay, not a resume.** An interrupted run was stopped
between graph steps rather than waiting on an approval, so it has no pending
decision for `POST /v1/runs/{id}/resume` to apply — that returns `422` by design.
Find its last checkpoint and replay from it:

```bash
curl -s localhost:8000/v1/runs/a1b2... | jq '.history[-1].checkpoint_id'
curl -s -X POST localhost:8000/v1/threads/a1b2.../replay \
     -H 'content-type: application/json' \
     -d '{"checkpoint_id": "1f0e…"}'
```

A replay **forks**: the new work is appended to the same thread as a sibling of
the interrupted history, so what happened before is still there to compare
against. Runs waiting on a human come back as `waiting_human` and are resumed
the ordinary way.

Two supporting details, so the record is not a surprise. `interrupted` is
terminal — a poller stops waiting on it, because nothing will move it on its own.
And the WebSocket for an interrupted run closes rather than hanging, because
`run.interrupted` is a terminal event; before it existed, a subscriber to a
departing run's stream waited forever for an event that was never published.

The distinction matters more than it looks. `cancelled` means a person stopped
the work and it is recorded as their decision. A rolling deploy is not a person,
and an audit trail that cannot tell the two apart is not one.

### Disk is filling up

Checkpoints accumulate. The janitor removes whole thread histories older than
the retention window:

```bash
awf janitor --dry-run                 # what would go
awf janitor --retention-days 7        # do it
awf janitor --retention-days 7 --json | jq
```

Schedule it, and read the report before you trust it. `RetentionReport` separates
`examined`, `stale`, `deleted` and `undated`; `undated` is the one to watch — it
counts threads whose age could not be determined, and the janitor **keeps** them.

That is deliberate: the sweep fails closed. An undecidable age resolves to
"keep", never "delete", and is counted and logged so it is visible rather than
silent. A retention job that deletes on a parse error is a retention job that
eventually deletes the wrong thing.

#### Scheduling it

Three facts shape the schedule.

* **A pass is bounded, and `stale` is the truth about the store.** `--limit`
  (default 1000) caps how many threads one invocation deletes; `stale` reports
  the whole backlog, limited or not. A backlog larger than the limit drains
  across repeated passes — each pass deletes the next `--limit` — and a `stale`
  that keeps returning at the cap is the report telling you more remains.
* **The exit code is not the whole signal.** A store that refuses to open or
  list raises, the command exits 1, and the `error: PersistenceError` line on
  stderr is the alert. A store that lists *nothing* is legitimate (a fresh
  deployment) and exits 0 with `examined: 0` and a warning that the sweep saw
  no threads — so alert on `examined: 0` only when you know there should be
  threads, and on `stale` for the backlog.
* **A scheduled run needs the runtime's own configuration** — `AWF_POSTGRES_ENABLED=1`
  and `AWF_POSTGRES_DSN`. A `--memory` janitor would sweep only what that one
  process wrote, which on a schedule is nothing.

Run it with `--dry-run` the first day and read `stale` in the `--json` output,
then schedule the real thing.

**cron** — on the host that can reach the checkpointer. The cron environment
carries no `PATH`, so the binary path is explicit; the JSON line on stdout is
what a monitoring check should parse, not the exit code:

```bash
# crontab -e
17 3 * * * AWF_POSTGRES_ENABLED=1 \
           AWF_POSTGRES_DSN='postgresql://agentic:agentic@db:5432/agentic' \
           /opt/agentic-workflow/bin/awf janitor --json \
           >> /var/log/awf-janitor.log 2>&1
```

**systemd** — a oneshot unit on a timer, so output lands in the journal and
`Persistent=true` covers a host that was down at the scheduled time:

```ini
# /etc/systemd/system/awf-janitor.service
[Unit]
Description=Checkpoint retention pass
[Service]
Type=oneshot
Environment=AWF_POSTGRES_ENABLED=1
Environment=AWF_POSTGRES_DSN=postgresql://agentic:agentic@db:5432/agentic
ExecStart=/opt/agentic-workflow/bin/awf janitor --json
```

```ini
# /etc/systemd/system/awf-janitor.timer
[Unit]
Description=Daily checkpoint retention
[Timer]
OnCalendar=*-*-* 03:17:00
Persistent=true
[Install]
WantedBy=timers.target
```

**Kubernetes** — a CronJob using the runtime image, so it inherits the
environment the API runs with, and the pod's stdout is the log:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: awf-janitor
spec:
  schedule: "17 3 * * *"
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      backoffLimit: 0
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: janitor
              image: <the runtime image>
              command: ["awf", "janitor", "--json"]
              envFrom:
                - secretRef:
                    name: awf-postgres   # supplies AWF_POSTGRES_ENABLED / AWF_POSTGRES_DSN
```

Whichever schedule you choose, alert on `examined: 0` or a rising `stale` long
before the disk fills; the run's own exit code is not the signal.

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

### Total LLM spend

```bash
curl -s localhost:8000/metrics | grep awf_llm
```

The engine owns one LLM client per process and injects it into every run, so
the token counts on `/metrics` are the process-wide total since boot — the
number a budget dashboard wants, and the only number that can be reported
honestly. They are exported as monotonic counters (`awf_llm_calls_total`,
`awf_llm_prompt_tokens_total`, `awf_llm_completion_tokens_total`,
`awf_llm_cached_tokens_total`); the same numbers appear in the JSON object the
endpoint returns when it cannot render exposition text. A fresh process reads
all zeros: the client is built lazily, so empty really means "no calls yet".
Note what the aggregate is *not*: per-run. The engine attributes LLM usage to
each run at the source — it wraps the shared client per drive and records the
usage each completion actually reports — so a single run's spend is available
without counting log lines:

```bash
curl -s localhost:8000/v1/runs/$RUN_ID/usage
# {"run_id": "...", "usage": {"prompt_tokens": ..., "completion_tokens": ...,
#   "cached_tokens": ..., "calls": ..., "total_tokens": ...}}
```

The attributed totals accumulate on the run registry across every drive
(start and each resolve), so a parked run answers "what has it cost so far".
Because the attribution is per completed call rather than a process-counter
diff, concurrent runs are charged exactly — no double counting. Two caveats,
both deliberate: the registration lives in process memory (the authoritative
state is the checkpoint), so the numbers restart from empty with the process;
and the operation payloads themselves carry the spend — `POST /v1/runs` shows
the whole run when `auto_resolve` is set, or that drive's share when it
parks.

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
| `AWF_HITL_SIGNING_SECRET` | — | Key that signs decisions. Falls back to `AWF_API_AUTH_TOKEN`. |
| `AWF_MAX_ITERATIONS` | `6` | Feedback-loop budget before the run is escalated. |
| `AWF_STATE_RETENTION_DAYS` | `30` | Retention window for the janitor. |
| `AWF_API_RATE_LIMIT_PER_MINUTE` | `120` | Per-client request budget. |
| `AWF_API_TRUST_FORWARDED_FOR` | `false` | Key the limiter on `X-Forwarded-For`. Only behind a proxy that overwrites it. |
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

### The audit signing secret

Human decisions are signed with `AWF_HITL_SIGNING_SECRET`, falling back to
`AWF_API_AUTH_TOKEN` when it is unset. Generate a dedicated one with
`openssl rand -hex 32`.

`AWF_LLM_API_KEY` is **not** used for this, deliberately. Rotating a provider
credential at the provider's convenience would retroactively turn every
signature ever written into an unverifiable one — indistinguishable, in the
audit output, from a forged entry. If you are upgrading from a version that did
use it, set a dedicated secret and treat pre-upgrade signatures as unverifiable
from that point on; the decisions themselves are unaffected.

When `AWF_HITL_REQUIRE_SIGNATURE` is on and neither secret is set, decisions are
recorded **unsigned** and the API logs `hitl.signatures_unavailable` at boot.
That is the default out of the box, so if you are reading this on a real
deployment it is worth checking which of the three states you are in:

- `audit_verified: true` on approval responses — signatures are being produced
  and verified.
- `audit_verified: null` — no secret is configured. Anyone with write access to
  the database can alter the decision log undetectably.
- Setting `AWF_HITL_REQUIRE_SIGNATURE=false` — the same gap, declared on purpose.
