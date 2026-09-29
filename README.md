<div align="center">

# agentic-workflow

**A multi-agent AI workflow engine with human-in-the-loop, built on LangGraph.**

Cyclic agent graph · PostgreSQL checkpointing · pause/resume gates · graded evals

</div>

```mermaid
flowchart TB
    subgraph client["Clients"]
        UI["Review UI / CLI / CI"]
    end

    subgraph control["Control plane — FastAPI"]
        R1["/v1/runs<br/>create · read · report · timings<br/>cancel · resume"]
        R2["/v1/approvals<br/>inbox · diff · resolve<br/>audit · sweep"]
        R3["/v1/threads<br/>history · checkpoints · replay"]
        WS["/ws/runs/&#123;id&#125;<br/>/ws/events"]
        TH["rate limit · auth · error codes"]
    end

    subgraph engine["WorkflowEngine — sole owner of graph.astream"]
        CTX["AgentContext<br/>live deps, never checkpointed"]
    end

    subgraph wf["LangGraph StateGraph — checkpointer attached"]
        T["triage"]
        P["programmer"]
        V["reviewer"]
        R["router"]
        X["tester"]
        A["apply_patch"]
        REP["reporter"]
        T --> P --> V --> R
        R -->|"needs work"| P
        R -->|"verify"| X
        X --> R
        R -->|"approved"| A --> REP
    end

    subgraph hitl["Human in the loop"]
        POL["policy.py<br/><i>should</i> we ask? pure function"]
        G["gates.py<br/>interrupt · resume · sign"]
        SVC["service.py<br/>REST inbox · replay · audit"]
    end

    subgraph store["Persistence"]
        PG[("PostgreSQL<br/>checkpoints")]
        REG["run registry<br/><i>rebuildable projection</i>"]
        RET["retention janitor<br/>fails closed"]
    end

    subgraph llm["Providers"]
        ECHO["echo<br/><i>deterministic, offline</i>"]
        OAI["openai_compatible"]
    end

    subgraph quality["Quality"]
        DS["golden dataset<br/>20 grounded cases"]
        MET["native metrics<br/><i>deterministic, model-free</i>"]
        JDG["ragas / deepeval<br/><i>optional judges</i>"]
    end

    UI --> TH
    TH --> R1 & R2 & R3 & WS
    R1 & R2 & R3 --> engine
    WS -.events.-> UI
    engine --> CTX
    CTX --> wf
    wf <--> POL
    POL -.ask.-> G
    G -.park/resume.-> R2
    SVC --> G
    G <--> PG
    wf <--> PG
    engine <--> REG
    RET --> PG
    CTX --> llm
    wf -.reports.-> DS
    DS --> MET
    MET -.gate.-> CI["CI"]
    JDG -.optional.-> MET

    classDef gate fill:#fff4e6,stroke:#d97706,stroke-width:2px
    classDef store fill:#eef2ff,stroke:#4f46e5
    classDef prov fill:#f0fdf4,stroke:#16a34a
    class POL,G,SVC gate
    class PG,REG,RET store
    class ECHO,OAI prov
```

## The problem

An agent that can change code without a supervisor is a liability. The naive
arrangement — one prompt, one model, one shot — fails in three specific ways:

**It cannot be stopped.** There is no point between "the model decided" and "the
change landed" where a person can read a diff and say no. A stop button that is
a `KeyboardInterrupt` is not a stop button; it is a way to lose work.

**It forgets.** If the process dies, or a human answers an hour later, the work
so far is gone or has to be reassembled from logs. An agent that cannot be
resumed cannot be supervised by someone who is not sitting at the terminal.

**Nobody can tell whether it is any good.** "The model said it was fine" is not a
measurement, so changing a prompt is indistinguishable from changing the
weather.

Each is a missing mechanism rather than a missing capability.

## The solution

A cyclic LangGraph `StateGraph` where three agents hand work to each other
through a router, every super-step checkpointed to PostgreSQL, and four
explicit places where the graph interrupts and waits for a person.

| Requirement | How it is met |
| --- | --- |
| **Cycles and feedback** | `router` is the only node with conditional edges. `programmer → reviewer → router → programmer` closes a real loop, bounded by `max_iterations` and by a stall detector. |
| **Durable state** | One checkpoint per super-step. A parked run is a row, not a process — any replica can resume it. |
| **Time travel** | Every checkpoint is addressable. `POST /v1/threads/{id}/replay` forks a sibling run from any super-step; the original is never modified. |
| **Human gates** | Four `ask_human` sites. A rejection is a **verdict, not an exception** — it sets `verdict: rejected`, keeps the human's comment, and still publishes a report. |
| **Graded quality** | 7 deterministic, model-free metrics over a 20-case golden dataset, plus optional Ragas/DeepEval judges. |
| **Measurable** | A benchmark harness that reports its own CPU, its own caveats, and its own null-model floor. |

### The three decisions that shaped it

**A human rejection is a verdict, not an exception.** When a person rejects a
patch, `AgentNode.rejection_update()` converts the `ApprovalRejectedError` into
an ordinary state update at all four gate sites. Raising instead would produce a
`failed` run whose only record is a stack trace — discarding the comment, which
is the one thing the rejection was made to produce. `RunStatus` has `rejected`
as its own value, and a rejected run still publishes a report.

**The checkpointer is authoritative; the registry is a projection.** Every read
goes to the checkpoint store. The run registry exists so `GET /v1/runs` is a
query rather than a scan, and nothing depends on it being correct. It is why
cancellation — which happens *outside* the graph — is overlaid during
reconstruction rather than stored as a checkpoint.

**A parked run answers 202, not 200.** `waiting_human` is a first-class outcome
and `WorkflowEngine` is the only thing permitted to call `graph.astream`. A
report request on a parked run returns 202 with a `null` body, because the report
does not exist *yet* rather than never.

Full reasoning in [`docs/architecture.md`](docs/architecture.md).

## Quickstart

Three steps. No API key, no database.

```bash
git clone https://github.com/danielgarciaperal2001-glitch/agentic-workflow && cd agentic-workflow
make venv
source .venv/bin/activate && awf demo --interactive
```

`--interactive` prints each human gate — the diff, the agent's confidence, why
it escalated — and reads your verdict from the prompt. The run resumes from its
checkpoint each time. Unrecognised input re-asks rather than defaulting to
approve.

Drop `--interactive` to have every gate approved automatically and watch the
whole loop run unattended, or `make demo` if you would rather not activate the
virtualenv.

To run the whole stack with a real database instead:

```bash
docker compose up
```

That starts PostgreSQL, builds the image, and serves the API on
`localhost:8000` — `/docs` for the OpenAPI browser, `/health/ready` to confirm
the store is reachable.

## Architecture & tech stack

| Layer | Choice | Why |
| --- | --- | --- |
| Orchestration | **LangGraph** `StateGraph` | Cyclic graphs, interrupt/resume and checkpointing are built in, rather than reimplemented on top of a linear chain. |
| State | **pydantic 2** | Validated at every boundary; `extra="forbid"` turns a typo'd field into a 422 rather than a silently dropped value. |
| Persistence | **PostgreSQL** + `langgraph-checkpoint-postgres` | Durable resume and time travel. Authoritative over any in-process cache. |
| API | **FastAPI** + **uvicorn** | Async-native, typed, auto-documented. |
| Streaming | **WebSockets** | Per-run and broadcast event feeds with heartbeat and send timeout. |
| Validation | **pydantic-settings** | Every setting is `AWF_`-prefixed, read once, and `lru_cache`d. |
| Logging | **structlog** | JSON or console, one envelope, request/run correlation. |
| Quality | **pytest**, **ruff**, **mypy --strict** | 585 tests; the strict typing is the point, not the coverage number. |
| Evals | Native metrics + **Ragas** / **DeepEval** (optional) | Deterministic by default; judges when you have credentials. |

`src/` layout. Dependencies point one way: `domain/` imports nothing from the
project, and `graph/`, `api/`, `services/` and `evals/` all depend on it.

| Package | Owns |
| --- | --- |
| `domain/` | Schemas, graph state. Pydantic and nothing else. |
| `graph/` | The `StateGraph`, its seven nodes, the router. |
| `human/` | `policy` (should we ask), `gates` (interrupt plumbing), `service` (the REST inbox). |
| `persistence/` | Serializer, checkpointer factory, registry, retention janitor. |
| `services/` | `WorkflowEngine` — the sole owner of `graph.astream`. |
| `api/` | Wire models, error handlers, 22 REST operations, 2 WebSocket routes. |
| `evals/` | Dataset, 7 metrics, runner, report. |
| `llm/` | `LLMClient` protocol, the `echo` null model, the OpenAI-compatible client. |

### The graph

```
START → triage → programmer → reviewer → router ─┬→ apply_patch → reporter → END
                            ▲                   ├→ tester ─────┘
                            └───────────────────┘
```

`awf topology` prints the live table, including the full routing predicate set.

## Benchmarks

Produced by `python benchmarks/bench.py`. Not estimates — a real run on a real
machine, and the harness prints the CPU it ran on alongside every figure.

| Measurement | Median | Mean | p95 | Min | Max | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `graph_compile` | 0.182 ms | 0.374 | 1.158 | 0.122 | 1.158 | 5 |
| `end_to_end` | 66.970 ms | 66.644 | 82.096 | 55.423 | 82.096 | 15 |
| `checkpoint_write` | 5.375 ms | 5.375 | 5.375 | 5.375 | 5.375 | 1 |
| `end_to_end_5files` | 71.448 ms | 75.268 | 88.531 | 63.562 | 88.531 | 7 |
| `end_to_end_20files` | 96.704 ms | 103.676 | 116.603 | 90.412 | 116.603 | 7 |
| `gate_overhead` | 48.330 ms | 48.005 | 63.457 | 36.783 | 63.457 | 15 |
| `throughput` | 15.293 runs/s | — | — | — | — | 1 |
| `eval_suite` | 1.364 s | — | — | — | — | 1 |

| Node | Median ms | Share of node time |
| --- | ---: | ---: |
| `reviewer` | 4.829 | 25.9% |
| `reporter` | 4.827 | 25.9% |
| `tester` | 3.936 | 21.1% |
| `programmer` | 1.858 | 10.0% |
| `triage` | 1.806 | 9.7% |
| `apply_patch` | 0.965 | 5.2% |
| `router` | 0.419 | 2.2% |

Measured on CPython 3.14.7, AMD Athlon Silver 3050U, 2 CPUs, Linux
7.2.7-200.fc44.x86_64.

### Read these numbers for what they are

**The `echo` provider is a null model.** It finds defects by regular expression
and answers in microseconds by design. So `end_to_end` is not end-to-end latency
with a real model — it is **the orchestration overhead this engine adds around a
model call**, which is the part this project owns and the part worth optimising.
A real provider dominates a run by orders of magnitude.

**Checkpoints here are in-memory.** `checkpoint_write` is not a proxy for
PostgreSQL. Durable-store latency is a different measurement with different
numbers, and quoting this one as if it were the other is exactly the
substitution this project argues against.

**`checkpoint_write` is derived**, not directly timed: it is the end-to-end
median divided by the checkpoints one run produced. Isolating a single write
would mean instrumenting the store, which changes what is being measured.

**The interesting number is `gate_overhead`: 48 ms of a 67 ms run.** Node work
sums to ~18 ms; the other ~48 ms is spent *between* nodes — park, checkpoint,
ask, read the state back, resume. The human-in-the-loop machinery costs more
than the agents it coordinates. That is the price of being able to stop a run
and come back to it, and it is stated here rather than left for you to subtract
from the table.

## Evaluation

The golden dataset is 20 hand-checked cases, each with a seeded defect that
must be **present in the code it names** — the loader refuses to load a case
asserting a finding about code that does not contain it.

### Measured baseline — `echo`, 20 cases, gates=approve

| Metric | Mean | Min | Max | Failed | Counted |
| --- | ---: | ---: | ---: | ---: | ---: |
| `grounding` | 1.000 | 1.000 | 1.000 | 0 | 20 |
| `path_grounding` | 1.000 | 1.000 | 1.000 | 0 | 20 |
| `structure` | 1.000 | 1.000 | 1.000 | 0 | 20 |
| `self_consistency` | 1.000 | 1.000 | 1.000 | 0 | 20 |
| `precision` | 1.000 | 1.000 | 1.000 | 0 | 3 |
| `citation_coverage` | 1.000 | 1.000 | 1.000 | 0 | 3 |
| `recall` | **0.150** | 0.000 | 1.000 | 17 | 20 |

**What this means, stated plainly.** The invariant metrics are at 1.000: the
workflow never fabricated a quote, never invented a file path, never published
an inconsistent verdict, and never approved over its own critical finding. That
is a real result and it is what the design guarantees.

`recall` is **0.150** — the null model found 3 of 20 seeded defects. That is the
**floor a real provider has to beat**, not a quality claim. This workflow does
not, on its own, detect real bugs; it orchestrates a model that might. The
number is published because a baseline nobody has measured is not a baseline.

`precision` and `citation_coverage` are averaged over the 3 cases that made a
checkable claim. The other 17 produced "no defects found" — a null result, not
a false positive, and deliberately not scored as one.

### Why the gate is split

```bash
awf eval --gate invariants    # 20/20 — the offline gate CI runs
awf eval                      # 3/20  — includes recall; fails, correctly
```

The metrics partition by *what they measure*:

- **Invariants** — properties of the **workflow**. A report that quotes a line
  nobody wrote is broken as a review regardless of which model produced it, and
  a deterministic stub violates them exactly as well as a frontier model does.
- **Detection** (`recall`) — a property of the **model**.

A CI job gating on every metric while running the null provider would be red
forever, and a permanently red job is a job everyone learns to ignore. A job
gating on nothing would prove nothing. Gating the invariants is free, offline,
reproducible, and still catches the regression class that actually breaks this
system.

The native metrics check that a claim is *grounded*, not that it is *right*.
That needs a judge (`pip install 'agentic-workflow[eval,eval-deepeval]'`) or a
human.

## The API

22 REST operations and 2 WebSocket routes.

```bash
# Submit a review. Returns 202 — the run is parked, awaiting a human.
curl -sX POST localhost:8000/v1/runs -H 'content-type: application/json' -d '{
  "run_id": "r-1", "title": "Fix float drift in order totals",
  "language": "python",
  "description": "Totals drift by cents on large orders.",
  "files": [{"path": "checkout/total.py", "content": "def total(xs):\n    t = 0.0\n    for x in xs: t += x[\"price\"] * x[\"qty\"]\n    return t\n"}],
  "acceptance_criteria": ["Totals exact to two places."]
}'

curl -s localhost:8000/v1/approvals | jq '.items[] | {approval_id, stage}'
curl -s localhost:8000/v1/approvals/$APR/diff | jq '.files'
curl -sX POST localhost:8000/v1/approvals/$APR/resolve \
  -H 'content-type: application/json' \
  -d '{"decision":"approve","reviewer":"daniel","comment":"ok"}'

curl -s localhost:8000/v1/runs/r-1/report | jq
curl -s localhost:8000/v1/runs/r-1/timings | jq '{total_ms, slowest_node}'
```

WebSocket event feeds at `/ws/runs/{run_id}` and `/ws/events`.

## CLI

```bash
awf demo [--interactive]   # a full run, gates answered at the prompt
awf replay <run_id> --index -1   # time travel: fork from any checkpoint
awf topology               # nodes, edges, loops, routing table
awf eval [--gate ...] [--json]  # score the golden dataset
awf janitor [--dry-run]    # one checkpoint-retention pass
```

Exit codes: `0` ok, `1` not done, `2` bad input, `130` interrupted. Set
`AWF_CLI_TRACE=1` for a traceback.

Logs go to **stderr**; results to **stdout**. `awf eval --json | jq` works.

## Development

```bash
make venv          # virtualenv with the dev extra
make check         # ruff check + format --check + mypy --strict
make test          # the suite
make test-cov      # with coverage
make eval          # score the golden dataset
make bench         # the tables above
```

`make help` lists everything. A `.devcontainer/` definition is included for
VS Code; it brings its own Docker-in-Docker so `docker compose` works inside it.

CI runs four jobs: lint/format/mypy strict, the suite against a real PostgreSQL
on the three Python versions the package claims to support (3.11 is the
`requires-python` floor, so the floor is the one that gets tested), the golden
dataset gated on the invariants, and a buildx build that then **boots the image**
and waits for `/health/live` — a successful build and a bootable image are
different claims and only the second is useful.

## Configuration

Everything is `AWF_`-prefixed. See [`.env.example`](.env.example) and the table in
[`docs/runbook.md`](docs/runbook.md).

```bash
AWF_LLM_PROVIDER=openai_compatible
AWF_LLM_API_KEY=sk-...
AWF_POSTGRES_ENABLED=true
AWF_POSTGRES_DSN=postgresql://agentic:agentic@localhost:5432/agentic
AWF_POSTGRES_SCHEMA=awf_tenant_a
```

`Settings` **refuses** `awf_`-prefixed keyword arguments. `Settings(AWF_LOG_LEVEL="DEBUG")`
is a loud `ConfigurationError` with suggestions, not a setting that quietly stays
at its default.

`AWF_POSTGRES_SCHEMA` is applied as a `search_path` on every pooled connection,
because `AsyncPostgresSaver` has no schema parameter and its migrations are
unqualified `CREATE TABLE`. The field is pattern-constrained to a bare SQL
identifier — the backend parses `search_path` as SQL, so `awf_x, public` would
silently redirect every table in the application.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — layering, the graph, the
  checkpointer, and why each non-obvious choice was made.
- [`docs/hitl.md`](docs/hitl.md) — the gate lifecycle, the five escalation rules,
  and the failure modes.
- [`docs/runbook.md`](docs/runbook.md) — operating it, and what to do when.
- [`docs/security.md`](docs/security.md) — what the control plane can do, what
  is protected, what is deliberately open, and how to deploy it.

## What this does not do

- The `echo` provider is a null model. Its recall of 0.150 is a floor, not a
  claim.
- The native metrics check groundedness, not truth.
- The benchmark numbers are orchestration overhead with an offline provider, not
  end-to-end latency with a real one.
- No multi-tenancy. One `AWF_API_AUTH_TOKEN` guards the whole API — that is
  authentication, not authorisation.
- The retention sweep fails closed: an undeterminable age resolves to "keep", and
  is counted.

## License

MIT — see [`LICENSE`](LICENSE).
