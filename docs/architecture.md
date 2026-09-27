# Architecture

A multi-agent review workflow built on a LangGraph `StateGraph`, a PostgreSQL
checkpointer, and explicit human gates. This document explains how the pieces fit
together and, where a choice was not obvious, why it was made that way.

## The problem

An agent that can edit code is a liability without a supervisor. The naive
arrangement — one prompt, one model, one shot — fails in three specific ways:

1. **It cannot be stopped.** There is no point between "the model decided" and
   "the change landed" where a person can look at a diff and say no.
2. **It forgets.** If the process dies, or a human answers an hour later, the
   work so far is gone or has to be reconstructed from logs.
3. **Nobody can tell whether it is any good.** "The model said it was fine" is
   not a measurement, so a change to the prompts is indistinguishable from a
   change to the weather.

Each of those is a missing mechanism rather than a missing capability, and each
has a corresponding piece below.

## Layering

Dependencies point one way. `domain/` imports nothing from the project; `graph/`,
`api/` and `evals/` depend on it; it never depends on them.

```
        api/          graph/          evals/
           \              |              /
            \             |             /
             +---- services/ ---------+
                       |
              persistence/     human/      llm/
                       \          |          /
                        +--- domain/ ------+
```

| Package | Owns | May import |
| --- | --- | --- |
| `domain/` | `ReviewRequest`, `SourceFile`, `Finding`, `FinalReport`, `Verdict`, graph state. Pydantic and nothing else. | stdlib, pydantic |
| `config.py` | `Settings`, `load_settings` | pydantic-settings |
| `llm/` | The `LLMClient` protocol, the `echo` null model, the OpenAI-compatible client | domain, config |
| `graph/` | The `StateGraph`, its nodes, and the router | domain, llm, human, persistence |
| `human/` | `policy` (whether to ask), `gates` (interrupt plumbing), `service` (the REST inbox) | domain, config |
| `persistence/` | Serializer, checkpointer factory, run registry, retention janitor | domain, config |
| `services/` | `WorkflowEngine` (the owner of the graph), `runner` (one-shot use) | everything |
| `api/` | Wire models, error handlers, routers, the event hub | domain, services, human |
| `evals/` | Dataset, metrics, runners, report | domain, services, config |

The rule is not stylistic. `domain/` is where the meaning lives: if a
`Verdict` has to change, it has to change in one file, and nothing in
`domain/` should be able to force a review of `api/schemas.py` to find out.

### Two sets of schemas, on purpose

`domain/schemas.py` and `api/schemas.py` both define a `ReviewRequest`. They are
not a duplication to be refactored away — they answer different questions.

The domain model is what the workflow *means*: it has computed fields, it
validates cross-field invariants, and it is serialised into checkpoints. The API
model is what the *wire contract* allows: it forbids unknown fields so a
typo'd field name is a 422 rather than a silently dropped value, and it carries
the HTTP-only concerns like idempotency keys.

`api/schemas.py` converts outward with an explicit `to_domain()`. Explicit,
because an implicit conversion is where a field gets renamed and the mismatch is
discovered in production.

### Retrying is a property of the error, declared by the error

Every class in `errors.py` declares `retryable`, and `LLMClient.complete` reads
that flag — not the exception's class name. The distinction is the whole point:

A provider adapter is the only place that can tell a `503 overloaded` from a
`401 invalid api key`, because only it can see the status code. It says so by
constructing `ProviderHTTPError(..., retryable=True)` for the first and
`retryable=False` for the second, and the retry loop, the exponential backoff and
the `retryable` field an API client reads all follow from that one decision.

Keying the loop off a hardcoded tuple of exception types instead looks equivalent
and is not, in both directions at once. It cannot express "a 5xx is worth another
attempt" for a type it was not told about, so a transient gateway failure kills
the run on the first try; and it cannot express "this particular 401 is
permanent" for a type it *was* told about, so a bad key is announced to the caller
as retryable. Both were live here before the flag was honoured.

The `retryable` kwarg is therefore *consumed* by `ProviderError.__init__` rather
than left in the context dict. Left there it would be echoed into the error
message — so the human-readable text said `retryable=False` while the
machine-readable field said `true`, and a client that trusted either one was
misled by half of the contract.

### Redaction lives in the processor chain

`logging.scrub()` masks values whose *key name* says they are credentials, and it
is installed as a structlog processor rather than called at each log site. Both
halves of that choice are load-bearing.

In the chain, because the caller who forgets is exactly the case a redaction
exists for. The one person who cannot be trusted to remember at 2am is whoever
is writing an ad-hoc `log.info("...", config=settings.model_dump())`, and no
amount of documentation in the module docstring fixes that.

Walking sequences, because a payload carrying a list of files, findings or
per-node results is an ordinary shape, not an exotic one. A walk that descended
only into mappings left every secret inside a list reaching the log verbatim
while redaction appeared to be working — which is worse than no redaction,
because it is trusted.

## The graph

Seven nodes. `awf topology` prints the live table; this is the shape.

```
START → triage → programmer → reviewer → router ─┬→ apply_patch → reporter → END
                            ▲                   ├→ tester ─────┘
                            └───────────────────┘
```

| Node | Does | Fails over to |
| --- | --- | --- |
| `triage` | Turns the description into a task brief. | — |
| `programmer` | Produces a patch, or a rationale for not producing one. | — |
| `reviewer` | Produces a verdict and findings against the patch. | — |
| `router` | Decides the next node. The only place routing is decided. | — |
| `tester` | Runs the checks the brief named and reports what failed. | `router` |
| `apply_patch` | Applies the approved patch and records the files touched. | `router` |
| `reporter` | Publishes the final report. | — |

### Why the loops close on `router`

`router` is the only node with conditional edges, and both agents' loops pass
through it. The alternative — a static edge plus a conditional return inside each
node — scatters the transition rules across the graph until the topology is
unknowable without reading every node. Centralising them means the full set of
transitions is one function, `decide_route`, which is unit-tested directly
rather than inferred from a run.

### Iteration is bounded, not hoped for

`max_iterations` is enforced by `router`. A programmer/reviewer loop that does
not converge is a normal outcome, not an exception: the loop exits with a
`blocked` verdict and a report saying what it could not resolve. The alternative
— looping until the model agrees with itself — converts a failure into a hang.

## The checkpointer is the source of truth

Runs are persisted as LangGraph checkpoints, one per super-step, which is what
makes three things possible:

- **Resume.** A run parked on a gate is not a process holding a socket. It is a
  row. Answering the gate is a write, and the process that answers it need not be
  the one that started the run.
- **Time travel.** `GET /v1/threads/{run_id}/checkpoints` lists every
  super-step, newest first; `POST /v1/threads/{run_id}/replay` forks from any of
  them. Nothing already written is ever rewritten — LangGraph forks by handing
  the graph a `checkpoint_id`, so the branch is *appended to the same thread* as a
  sibling of the history it branched from. The thread therefore grows, and "the
  original history is untouched" would be the wrong claim to make about it; the
  one worth making is that every pre-fork checkpoint is still there, in order.
  That is what makes a before/after diff of a replay meaningful.
- **Crash recovery.** A process that dies mid-run leaves a consistent state, not
  a half-applied patch.

### Schema isolation, and why the setting had to change shape

`AWF_POSTGRES_SCHEMA` names the schema holding the checkpoint tables, and it is
applied as a `search_path` on every pooled connection.

It has to work that way because `AsyncPostgresSaver` **has no schema parameter**
and its migrations are unqualified `CREATE TABLE IF NOT EXISTS`. The setting was
read, validated, logged on every boot and on every ready event, and passed to
nothing: every deployment and every test wrote to `public`. A setting that looks
like it is working is worse than one that is obviously absent, because the things
that depend on it — multi-tenant isolation, test isolation — fail without failing
visibly.

Two details are load-bearing:

- **The name is pattern-constrained** to a bare unquoted SQL identifier. The
  backend parses `search_path` as SQL, so a value like `awf_a, public` would
  silently redirect every unqualified table in the application. The pattern is a
  security control, not a style rule, and it is also what makes interpolating the
  name into `CREATE SCHEMA IF NOT EXISTS` safe — PostgreSQL will not bind a
  parameter there, so the constraint and the interpolation have to agree.
- **The schema is created before the saver's migrations run.** A `search_path`
  naming a schema that does not exist does not create one; every unqualified
  `CREATE TABLE` would land in `public` anyway, which is the behaviour the change
  exists to remove.

The pool is also held rather than recovered from the saver by attribute lookup.
`getattr(saver.conn, "_pool", saver.conn)` was a guess, and a wrong one:
psycopg's `AsyncConnectionPool` has a `_pool` attribute that is its internal
`deque` of idle connections, not the pool. It happened to work only because the
code then fell through to `saver.conn.open`, so the deque was discarded — by luck,
in the one place that used it.

### The registry is a projection, not a record

`persistence/repository.py` keeps a run registry for listing and filtering. It
is explicitly **rebuildable**: every authoritative read goes to the checkpointer,
and `_outcome_from_snapshot` reconstructs a run's outcome from its checkpoints.
The registry exists so `GET /v1/runs` is a query rather than a scan, and nothing
depends on it being correct — if it is lost, the runs are still there.

This is why cancellation and rejection are *overlaid* during reconstruction
(`_OUTSIDE_OVERLAY_STATUSES`). A run that was cancelled while parked has no
checkpoint saying so, because the cancellation happened outside the graph; the
overlay is how that state is recovered from the audit log rather than inferred
from the absence of progress.

Being a projection is a claim about *reads*, and it is easy to state and hard to
keep. The registry is **per-process**, so the moment the projection is more than a
cache, the process boundary shows up in the code. Three things depend on this,
and all three were wrong before they were written down:

- **Hydration.** `RunRegistry.rebuild_from` existed and nothing called it, so a
  restarted process started with an empty registry and a full database.
  `startup()` now asks the checkpointer which threads exist
  (`list_thread_ids`) and rebuilds the projection. The bound is
  `AWF_RECOVERY_MAX_RUNS`, because a restart must not stall on a large
  `checkpoints` table, and a store that cannot be listed degrades the listing
  rather than failing the boot.
- **Hydration takes projections, not state.** The registry is handed the
  engine's own view of each run — `status`, `iteration`, `pending_approval` —
  never the raw checkpoint values. Copying `state["status"]` across looks
  equivalent and is not: that is the *graph's* status, whose vocabulary is
  different. A run parked on a human gate was rehydrated as `"triaged"`, which
  matched no status filter and was not in `TERMINAL_STATUSES`, so a poller would
  have waited forever on a run that had already finished. `RunRecord.__post_init__`
  now rejects a status outside the vocabulary, because a `Literal` annotation is a
  promise to a type checker and a `dataclass` enforces nothing.
- **Writes create before they update.** Every path that writes registry state
  goes through `WorkflowEngine._record`, which creates a missing record first.
  Without it, `registry.update` raised `RunNotFoundError` from inside the `except`
  block of `_drive` — so a genuine workflow error was replaced by a registry
  error before it could be logged, and the traceback named the registry and said
  nothing about what had actually gone wrong. The same missing record made
  cancelling a run owned by another replica impossible: the operator got a 404
  for a run that was demonstrably running.

`cancel()` therefore takes its existence check from `_aget_state`, the
authoritative read, and not from the registry. Cancelling is the one action a human
takes about a process they are not talking to, and "I cannot stop it" is the worst
answer that endpoint can give.

## Human in the loop

Three files, split by what they are responsible for, because the three
concerns change for completely different reasons:

| Module | Question it answers | Changes when |
| --- | --- | --- |
| `human/policy.py` | *Should* we ask a human? | The risk model changes. |
| `human/gates.py` | *How* does the graph park and resume? | LangGraph changes. |
| `human/service.py` | How does a person answer over HTTP? | The API changes. |

`policy.py` is a pure function from `(state, config)` to a decision, with no I/O.
That is what makes it testable without a graph, a database or a clock — and it is
where the interesting behaviour lives: asking a human about a trivially correct
patch is a defect, not a safety feature.

### Approval ids are deterministic

```
apr_{run_id}_{stage}_{iteration:02d}_{sha256(run_id|stage|iteration)[:8]}
```

Deterministic so a retry produces the same id and a double-click cannot create
two approvals for one gate. This is load-bearing rather than tidy: LangGraph
re-executes a node from the top after an interrupt, so the approval request is
built *twice* for one human decision. A random id would fail to match on the
second pass and the stale-answer guard would reject every legitimate decision.

Derived from the run, stage and iteration, so an approval id is enough to
locate the checkpoint it belongs to. The inverse `run_id_from_approval_id()`
parses from the *right* — stage names contain underscores and run ids are
alphabet-restricted not to, so the three trailing components are unambiguous.
The alternative, scanning every run's decision log, is O(runs) on the hot path
of every human decision.

### A rejection is a verdict, not an exception

This is the single most consequential decision in the HITL layer. When a human
rejects, `AgentNode.rejection_update()` converts the
`ApprovalRejectedError` into an ordinary state update — `verdict: rejected` — at
all four `ask_human` call sites.

The reasoning: rejection is the system working. A person looked at a diff and
said no, which is a verdict the reporter should publish and the next iteration
should act on. Raising an exception instead would produce a `failed` run whose
only record is a stack trace, discarding the human's comment — the one piece of
information the rejection was made to produce. `RunStatus` therefore has
`rejected` as its own value, distinct from `cancelled` and `failed`.

### `waiting_human` is a first-class outcome

`WorkflowEngine` is the only thing permitted to call `graph.astream`, and it
returns a `RunOutcome` with `status: waiting_human` when it parks. The API
answers **202 Accepted** for a parked read rather than 200 with a partial body,
because "still going" and "finished" must not be the same status code. A report
request on a parked run returns 202 with a `null` body rather than a 404, since
the report does not exist *yet* rather than never.

## The API

22 REST operations and 2 WebSocket routes, all under `/v1` except the health
probes.

| Group | Operations |
| --- | --- |
| Health | `GET /health/live`, `GET /health/ready` |
| Runs | create, list, read, report, timings, decisions, cancel, resume |
| Approvals | list, read, diff, resolve, replay, audit, stats, sweep |
| Threads | history, checkpoints, one checkpoint, replay |
| WebSocket | `/ws/runs/{run_id}`, `/ws/events` |

Three decisions that are not obvious from the route list:

**Health probes are unthrottled.** Rate limiting is applied per-router with
`include_router(..., dependencies=throttle)` on runs, approvals and threads
only. A liveness probe that can be rate-limited is a liveness probe that
reports the process unhealthy because someone else is busy.

**The event sink is transport, not lifecycle.** `create_app` *always* calls
`active.set_event_sink(hub.publish)`, including for an engine the app did not
build. A hub is per-app state; an injected engine carries whatever sink it was
constructed with, and an engine with no sink produces an app that completes the
WebSocket handshake and then emits nothing at all. That failure is silent, which
is what makes the unconditional wiring worth the one line.

**Unknown field → 422, content mismatch → 400.** `extra="forbid"` everywhere
means a typo'd field is rejected rather than dropped. The content-hash check is
a different thing — the payload is well-formed but describes different files than
it claims — and gets its own code so a client can tell "you sent nonsense" from
"you sent a stale request".

## Configuration

Every setting is an environment variable prefixed `AWF_`, from
`pydantic-settings`. `Settings` **refuses** `awf_`-prefixed keyword arguments:

```
Settings(log_level="DEBUG")           # fine
Settings(AWF_LOG_LEVEL="DEBUG")       # ConfigurationError, with suggestions
```

The second one is the error people actually make, and it fails silently
otherwise — an ignored `AWF_` kwarg leaves the default in place and the
mysterious behaviour starts somewhere else entirely. The refusal names the
prefixed variables that are set in the environment, because the near-miss is
almost always a real variable that was passed in the wrong place.

`load_settings` is `lru_cache`d, so the process reads its environment once.
Tests call `reset_settings_cache()`.

## What the evals can and cannot tell you

The native metrics are pure functions of `(source, findings, report)`. No model
judge, no network, no nondeterminism — the same three inputs always give the
same score, which is what makes them usable in CI.

They are split by what they measure, because conflating the two is how a quality
gate becomes decorative:

- **Invariants** (`grounding`, `path_grounding`, `precision`,
  `citation_coverage`, `structure`, `self_consistency`) are properties of the
  workflow. A report that quotes a line nobody wrote is broken as a review
  regardless of which model produced it.
- **Detection** (`recall`) is a property of the model.

`awf eval --gate invariants` gates the first group. This is not a weakening: it
is the only gate that can run offline, free, and reproducibly — and the recall
number is still reported next to it as the floor a real provider must beat.

### Two rules the metrics learned the hard way

**A null result is not a claim.** `is_claim(finding, source)` requires a file, a
line, or an identifier actually present in the submitted source. A report that
says "no defects found" is a good report, and scoring it as a false positive
punishes the most honest possible answer.

**Diff additions are new by definition.** A block tagged `diff`/`patch`/`unified`
has its `+` lines exempt from the quote check — they are the proposed change, so
they cannot already be in the file. Context and `-` lines are checked normally.
Without this rule every correct patch suggestion reads as a hallucination, and
the metric is worse than none.

## What this does not do

Stated plainly, because the gap between what a system claims and what it does is
where incidents come from.

- **The echo provider is a null model.** It finds defects by regular expression.
  Its measured recall of 0.150 on the golden set is a baseline, not a quality
  claim. A real provider must beat it to have added anything.
- **The native metrics cannot tell whether a finding is correct.** They check
  that a claim is grounded in the submitted code, not that the claim is *true*.
  That needs a judge model (`ragas`/`deepeval`, both optional) or a human.
- **Retention is a best-effort sweep.** It fails closed: a thread whose age
  cannot be determined is kept and counted, never deleted.
- **No multi-tenancy.** One `AWF_API_AUTH_TOKEN` guards the whole API. It is
  authentication, not authorisation.
