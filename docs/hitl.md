# Human in the loop

How a run stops, waits, and resumes — and why each of those is designed the way
it is.

## The shape of a gate

```
  node            checkpoint written
    │                    │
    │  ask_human(policy) │
    ▼                    ▼
  interrupt() ──────►  parked: status = waiting_human
                             │
                    approval_id known to the client
                             │
              POST /v1/approvals/{id}/resolve
                             │
                             ▼
                      graph resumes
```

Three things are true at the moment a run parks, and all three are what make
resumption possible later:

1. A checkpoint exists for the state *before* the gate.
2. The `approval_id` is a pure function of `(run_id, stage, iteration)`.
3. The decision is recorded with a signature and an operator name.

## Why the run is a row, not a process

A parked run is not a process holding a socket open. It is a checkpoint plus an
approval record. That is what allows:

- The answering process to be a different process from the one that started the
  run — an API replica, a CLI, a cron job that drains the inbox.
- A process to die while a run waits, and the run to survive.
- A gate to sit unanswered for a day without a connection, a timeout, or a
  restart.

The cost is that nothing is implicitly in flight, so the API has to be explicit
about parked state. A parked run answers **202**, not 200.

## The three decisions

### `approve`

Apply the proposal and continue. The decision, the operator and the comment are
appended to the audit log before the graph resumes, so an answer cannot be lost
by a crash in the middle of resuming.

### `edit`

A human changes something before it lands. At `patch_apply` the edit is the
patched content; at `patch_review` and `final_report` it is a replacement
report. An edit is recorded as an edit, distinct from an approval of modified
content — otherwise the audit log would say a human approved something they did
not actually see.

### Answering twice

A duplicate submission is not an error condition to be tolerated; it is a
decision someone already made being offered again, and the answer has to say so.
`POST /v1/runs/{run_id}/resume` and `POST /v1/approvals/{approval_id}/resolve`
both return `409 approval_already_resolved`, carrying the run id and the decision
already on record.

The distinction that matters is *already applied* versus *stale*. By the time a
duplicate arrives the run has parked on a different gate, so an id comparison
against the current gate reports a mismatch. That comparison is arithmetically
true and practically wrong: the decision was not too old, it was accepted while
it was current. A client told its approval expired fetches the new gate and
answers that too, and one approval becomes two. The check is therefore made
against the decision log, which holds the answer, and the graph's own
stale-answer guard stays in place for the separate question of whether a *new*
decision belongs to the gate the run is parked on.

### `reject`

This is the one that is easy to get wrong.

**A rejection is a verdict, not an exception.** `AgentNode.rejection_update()`
converts `ApprovalRejectedError` into an ordinary state update at all four
`ask_human` call sites, setting `verdict: rejected` and carrying the human's
comment forward.

Raising instead would be the obvious implementation and it would be wrong. A
person looked at a diff and said no — that is the system working. Turning it into
a `failed` run would replace the outcome with a stack trace and discard the
comment, which is the single piece of information the rejection was made to
produce. `RunStatus` therefore carries `rejected` as its own value, and
`run.rejected` as its own event, both distinct from `cancelled` and `failed`.

A rejected run still publishes a report. The reporter runs.

## Who gets asked, and when

`human/policy.py` is a pure function from `(stage, review, confidence,
iteration, max_iterations)` to a `GateDecision`. No I/O, no clock, no database —
which is what makes it testable without any of them, and it is where the
interesting behaviour lives.

It evaluates five rules **in order**, and the first match wins:

| # | Rule | Fires when |
| --- | --- | --- |
| 1 | `ALWAYS` | The stage is irreversible: `patch_apply` or `final_report`. Also `patch_apply` when `AWF_HITL_REQUIRE_APPROVAL_BEFORE_APPLY` is set. |
| 2 | `LOW_CONFIDENCE` | The agent's reported confidence is below `AWF_HITL_ESCALATION_THRESHOLD` (default 0.70). |
| 3 | `HIGH_SEVERITY` | A blocking finding at `high` or `critical` is still unresolved, on `patch_review`, `test_review` or `on_call`. |
| 4 | `POLICY` | The run is about to publish a `final_report` whose verdict is anything other than `approved`. |
| 5 | `POLICY` | The feedback loop is exhausted (`iteration >= max_iterations`) on `patch_review` or `on_call`. |

Otherwise the stage does not gate. A `GateDecision` also carries the
`EscalationReason` — `always`, `low_confidence`, `high_severity`, `policy`,
`disabled` — which is persisted and shown in the inbox, so an operator can see
*why* they are being asked and can tell a policy gate from a model that was
unsure.

Two properties of this design matter more than the rules themselves:

**Order is deliberate.** An explicit `ALWAYS` gate outranks the low-confidence
heuristic, because a policy gate is a deliberate operator decision and must not
be silently downgraded by a well-meaning shortcut.

**Not gating is the common case.** A run that never crosses a threshold finishes
with zero human interaction, which is what you want for batch processing. Asking
about a trivially correct patch is a defect, not a safety feature: a gate that
fires on everything trains people to approve without reading, and then the gate
is decorative for exactly the case it was built for.

Configure with `AWF_HITL_ENABLED`, `AWF_HITL_REQUIRE_APPROVAL_BEFORE_APPLY`,
`AWF_HITL_ESCALATION_THRESHOLD`, `AWF_HITL_ALLOW_EDIT`, `AWF_HITL_ALLOW_REJECT`
and `AWF_HITL_DEFAULT_TIMEOUT_SECONDS`.

> **`AWF_HITL_ESCALATION_THRESHOLD` is a confidence threshold, not a clock.**
> It answers "how unsure must the model be before a human is involved", not "how
> long before a pending gate escalates". Nothing in this system auto-approves a
> gate that has gone unanswered; an unanswered gate stays a queue item until
> someone deals with it.

## Answering a gate

### Over HTTP

```bash
# What is waiting?
curl -s localhost:8000/v1/approvals | jq '.items[] | {approval_id, stage, run_id}'

# Read it properly before answering.
curl -s localhost:8000/v1/approvals/$APR | jq

# The diff, if the gate has one.
curl -s localhost:8000/v1/approvals/$APR/diff | jq '.files'

# Answer.
curl -sX POST localhost:8000/v1/approvals/$APR/resolve \
  -H 'content-type: application/json' \
  -d '{"decision":"approve","reviewer":"daniel","comment":"looks right"}'
```

`POST /v1/approvals/sweep` answers everything in the inbox with one verdict —
for draining a queue unattended, not for reviewing work.

### From a terminal

```bash
awf demo --interactive     # answers each gate at the prompt
awf demo                  # approves every gate automatically
```

`--interactive` re-asks on input it does not recognise rather than defaulting to
approve. A prompt that assumes yes when it cannot parse the answer is a prompt
that approves things nobody read.

### Over a WebSocket

```
ws://localhost:8000/ws/runs/{run_id}    one run
ws://localhost:8000/ws/events          every run
```

Frames are `{"event": ..., "run_id": ..., "data": ...}`. The event sink is wired
unconditionally in `create_app`, including for an injected engine, so a connected
client receives events rather than completing a handshake and hearing nothing.

**The stream is lossy on purpose, and tells you when it is.** Each client gets a
bounded queue, and a client that reads slower than events arrive loses the
*oldest* ones rather than applying backpressure to a running workflow. When that
happens the next frame is a heartbeat carrying how many events were missed:

```json
{"event": "heartbeat", "dropped_since_last": 12}
```

Re-fetch the run over REST and carry on — the stream is notifications, the run
state is the source of truth. Do not use the per-event `seq` field to detect
this: it counts publications across *all* runs, so the gaps in it belong to
other runs, and because the loss is always the oldest event it lands at the head
of the queue where no gap appears at all.

It rides the `heartbeat` event name, so a client that ignores the field keeps
working unchanged, and it arrives as soon as the loss happens rather than on the
next idle tick — a client that fell behind is the one that never goes idle.

## Correlating a run with the work that caused it

`request_id` and `metadata` are the submitter's own fields, and they come back on
every read of the run under the same names they were sent:

```bash
curl -s -X POST localhost:8000/v1/runs -H 'content-type: application/json' -d '{
  "run_id": "pr-1042", "request_id": "PR-1042",
  "title": "Fix rounding in invoice totals",
  "files": [{"path": "checkout/total.py", "content": "..."}],
  "metadata": {"repo": "acme/checkout", "branch": "fix/rounding"}
}'
```

```json
{"run_id": "pr-1042", "request_id": "PR-1042", "status": "waiting_human",
 "metadata": {"repo": "acme/checkout", "branch": "fix/rounding"}}
```

They are read back from the checkpointed request, not from the run registry, so
they answer the same on every replica and survive a restart. The registry is
per-process and rebuilt at startup, and its own dict is the engine's annotation
bag — projecting that would hand the client keys it never sent.

`metadata` is bounded: at most 64 keys and 4096 bytes serialised, refused with
**422** past that. The bound exists because the dict is now returned on every
read — measured before it, a 19.5 MiB `metadata` was accepted and came back on
the following `GET`.

**Two ids, both called `request_id`.** The body field is the submitter's: a PR
number, a ticket, an invoice id, stored with the run and returned with it. The
`X-Request-ID` header identifies the *HTTP call* instead — it is echoed on the
response and appears on the run's log lines, but is not stored with the run. The
log plane joins `X-Request-ID` to `run_id` for the whole run; the API joins the
body `request_id` to the run for as long as the checkpoint lives. Neither
substitutes for the other.

## The audit log

Every decision is appended, never updated:

```bash
curl -s localhost:8000/v1/runs/$RUN/decisions | jq
curl -s localhost:8000/v1/approvals/by-run/$RUN/audit | jq
```

Each record carries the operator, the decision, the comment, a timestamp and a
signature. With `AWF_HITL_REQUIRE_SIGNATURE=true` and a configured secret, the
signature is an HMAC-SHA256 over the canonical decision, which lets a verifier
prove the record was not edited afterwards.

`GET /v1/approvals/by-run/{run_id}/audit` also reports `unverified` — the count
of records whose signature does not check out. A tampered log is a fact the
operator needs surfaced, not an exception to raise at read time.

**Replay is checked against the log, not the inbox.** `ApprovalService.replay`
compares the incoming decision against the recorded one, field by field
(`_DECISION_FIELDS`). `resolve` checks the log before the inbox, so answering a
gate that was already answered returns **409**, not 404 — the gate exists, it has
been dealt with, and telling the caller it does not exist sends them looking for
the wrong problem.

## Failure modes worth knowing

| Symptom | Cause | What to do |
| --- | --- | --- |
| `409` on resolve | Already answered, or the decision disagrees with the log. | Read `GET /v1/approvals/{id}` to see what was recorded. |
| `409` on cancel | The run is already in a final state. | Check `status` first; cancel is not idempotent against a finished run. |
| `404` on a checkpoint | The checkpoint id is not in that thread. | `GET /v1/threads/{run_id}/checkpoints` to list the valid ids. |
| `202` on a run read | The run is parked on a gate. | Answer the pending approval; the pending id is in the run detail. |
| `400` on submit | Content-hash mismatch: the payload describes different files than it claims. | Re-fetch and re-submit. |
| `422` on submit | An unknown field, or one that fails validation. | Check the field name; `extra="forbid"` means typos are rejected, not dropped. |

## When nothing is being answered

A parked run waits. That is the design — it is a row, not a process — but it
means a forgotten gate is a run that never finishes, so the backlog has to be
visible:

```bash
curl -s localhost:8000/v1/approvals/stats | jq
```

```json
{
  "pending": 3,
  "expired": 1,
  "resolved_pending": 0,
  "by_stage": {"final_report": 2, "patch_review": 1}
}
```

`expired` counts gates past `AWF_HITL_DEFAULT_TIMEOUT_SECONDS`. It is a report,
not a resolution: an expired gate is still pending and still waiting for a
person. Nothing in this system turns a timeout into an approval. `final_report`
and `patch_apply` gate unconditionally (rule 1), so a run that reaches them always
waits — and a gate that quietly approves itself is the one behaviour that would
make every other guarantee here worthless.

`resolved_pending` counts inbox entries whose decision is already in the log. A
non-zero value usually means two operators answered the same gate; the log
keeps the first and the audit view shows both.

These four counters come from reading every known run, which is why this endpoint
is the one to use. `/health/ready` reports the same shape but counts from
per-process state and always reports `expired` and `resolved_pending` as `0`,
because a probe polled every few seconds cannot afford the sweep. See
[the runbook](runbook.md#what-the-readiness-probes-approval-numbers-mean) for the
measured cost that forced the split.
