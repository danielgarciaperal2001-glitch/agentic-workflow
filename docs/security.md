# Security

The trust boundary of this system is a single question: **who is allowed to make
a human decision?** Everything else in this document follows from that.

The control plane can start a run, spend provider tokens, read and diff every
checkpoint, replay a thread, resolve an approval — approve, edit or reject — and
stream the events of every run on the box. Resolving an approval is the one that
matters: it is the step a person is supposed to be accountable for, and the
graph is built so a decision changes what the workflow does next. A control
plane whose approval endpoint is open is not an unauthenticated dashboard; it
is an unattended reviewer that writes its own audit trail.

So the deployment question is not "is this service hardened" but "who can reach
it, and who can prove they are who they say they are".

## The defaults, and what they cost

| Setting | Default | Why |
| --- | --- | --- |
| `AWF_API_AUTH_ENABLED` | `false` | `docker compose up` and `awf demo` must work with no configuration at all. |
| `AWF_API_AUTH_TOKEN` | — | Required when authentication is on. `Settings` refuses to boot with the flag set and no token, so "secure but broken" is not a reachable state. |
| `AWF_HITL_REQUIRE_SIGNATURE` | `true` | The audit log claims to be tamper-evident, so it says so loudly when it cannot be. |
| `AWF_HITL_SIGNING_SECRET` | — | Falls back to `AWF_API_AUTH_TOKEN`. Deliberately never to `AWF_LLM_API_KEY`. |
| `AWF_API_RATE_LIMIT_PER_MINUTE` | `120` | Bounds guessing, and bounds a runaway client. Not an authentication mechanism. |
| `AWF_LLM_TOKEN_BUDGET_PER_RUN` | `500000` | Bounds what one request can cost. The request budget above does not: it counts requests, and one request was measured from 6,472 tokens to 4,060,022. `0` disables it. |
| `AWF_API_TRUST_FORWARDED_FOR` | `false` | A caller can write `X-Forwarded-For`. Off unless a proxy overwrites it. |
| `AWF_API_DOCS_ENABLED` | `true`, removed in production | An API description is a map of the control plane. |

Authentication off by default is a convenience with a real cost, and the cost is
not hypothetical: on the default configuration, anyone who can reach the port
can approve a review. What makes the default defensible is that it is a
*local* default — a laptop, a `docker compose` stack, a demo — and that turning
it on is one environment variable rather than a code change.

The default deployment also cannot sign its decisions, because there is nothing
to sign with, so the application says so at boot:

```
hitl.signatures_unavailable  detail='hitl_require_signature is enabled but no
signing secret is available, so human decisions are recorded UNSIGNED. Set
AWF_HITL_SIGNING_SECRET (or api_auth_token) ...'
```

That line is the point. An operator who reads the configuration, sees
`hitl_require_signature: true`, and concludes the audit log is tamper-evident
would otherwise be wrong, and would find out during an incident.

## Turning it on

```bash
export AWF_API_AUTH_ENABLED=true
export AWF_API_AUTH_TOKEN="$(openssl rand -hex 32)"
export AWF_HITL_SIGNING_SECRET="$AWF_API_AUTH_TOKEN"   # optional, see below

curl -H "Authorization: Bearer $AWF_API_AUTH_TOKEN" http://localhost:8000/v1/runs
```

A refusal is a 401 with the challenge that says what to send next:

```
$ curl -i http://localhost:8000/v1/runs
HTTP/1.1 401 Unauthorized
www-authenticate: Bearer

{"error":{"code":"authentication_error", ...},"request_id":"req_1f2e..."}
```

Then, in rough order of how often each is missed:

- **Terminate TLS in front of it.** The token is a bearer credential: it is in
  every request, and the WebSocket token is in a URL, which proxies log. Plain
  HTTP is a token handed to the network.
- **Keep `/metrics` internal.** It is unauthenticated by design (below) and it
  exposes the run count, event counters and the process-wide token spend.
- **Set `AWF_HITL_SIGNING_SECRET`** to its own value, separate from the API
  token, if the audit log has to stay verifiable after the API token rotates.
  Sharing one secret means rotating the API token invalidates every signature
  ever written.
- **Leave `AWF_API_TRUST_FORWARDED_FOR` off** unless a proxy overwrites the
  header. See the runbook's
  [Behind a reverse proxy](runbook.md#behind-a-reverse-proxy) for what turning
  it on costs.
- **Set `AWF_API_CORS_ORIGINS`** to the dashboard's origin. The default is
  `http://localhost:3000`, which is a development convenience, not a policy.
- **Leave `AWF_ENVIRONMENT=production`.** It removes `/docs`, `/redoc` and
  `/openapi.json` at boot, whatever `api_docs_enabled` says.
- **Use a real PostgreSQL password.** The checkpointer holds the source code
  under review and every decision ever made. The compose file's credentials are
  development credentials.

## The token

One static bearer token, shared by every caller. It is compared with
`hmac.compare_digest` on both transports, so neither the length nor the prefix
of the token leaks through a refusal's timing — a control plane whose only
secret is a static token has no business being measurably wrong about it.

What that token is not:

- **Not an identity.** Every authenticated caller is the same principal. The
  audit log records what was decided and a signature over it, not who decided:
  there is no `alice` in the record, because the application never learns one.
- **Not revocable without a restart.** Settings are read once, at boot. Rotation
  is a new process.
- **Not scoped.** Possession is full access to every route in the table below.

If you need per-person attribution or revocation, the honest answer is that this
does not have it: put an authenticating proxy in front, and let it be the thing
that knows who the caller is.

## What is protected

Everything under `/v1`, authenticated at the router level, so a new endpoint
inherits it rather than depending on the author remembering:

| Router | What it can do |
| --- | --- |
| `/v1/runs` | start, list, inspect, resume, cancel; read report, decisions, timings and token usage |
| `/v1/approvals` | list, read, diff, resolve, replay, sweep expired; read a run's audit trail |
| `/v1/threads` | history, checkpoint state, replay, checkpoint diff |

`tests/api/test_auth_surface.py` walks the real route table and asserts that
every route is either in this table or in the exemption list below, so the two
cannot drift apart silently.

## What is deliberately not

Each of these is a claim about a deployment, not an omission:

| Route | Open because | What it costs |
| --- | --- | --- |
| `/health/live`, `/health/ready` | An orchestrator, a load balancer and a `docker healthcheck` cannot hold a token. A probe that needs credentials fails when they rotate, and reports the application unhealthy when it is not. | Readiness names its dependencies and counts pending approvals. Nothing else. |
| `/` | A pointer at the API. Refusing it makes an unauthenticated `curl` look like an outage. | The service name, its version, and the endpoint list. |
| `/metrics` | Prometheus authenticates by network, not by header. | The number of runs in memory, event counters (published, delivered, dropped, subscribers, runs observed) and process-wide token spend. Keep it on an internal network. |
| `/docs`, `/redoc`, `/openapi.json` | They exist only when `environment` is not `production`. | A complete map of the control plane. |
| `/ws/runs/{id}`, `/ws/events` | Not exempt: a socket authenticates in the handshake, in the query string, and is refused with a 403 before any data flows. `tests/api/test_ws_auth.py`. | The token is in a URL. |

The socket is the one with a caveat worth stating plainly. A browser's
`WebSocket` cannot set an `Authorization` header, so the token travels as
`?token=…`, and **query strings end up in proxy access logs**. Mitigations, in
order: TLS; tell the proxy not to log query strings on the WebSocket path; treat
the token as compromised and rotate it if a log has been shipped anywhere
uncontrolled. `ws.auth_failed` logs the *path* only — never the token — so the
application's own logs are not the problem.

The refusal is a close before accept, which the ASGI server renders as HTTP
403. The 1008 policy code and the `unauthorised` reason are what a client
library sees when the server passes them through. A denial response with a
`401` and a `WWW-Authenticate` header was implemented and measured first: the
client got the better status, and the server still wrote an `ERROR` line for
every refusal, because uvicorn marks a handshake finished only on accept or
close. A log that fills with errors because someone guessed a token is the
situation the refusal exists to prevent, and a browser cannot read the status
either way.

## Rate limiting is not authentication

The limiter bounds how fast one client can spend the control plane. Its key is a
verified token's digest when there is one, and otherwise the socket peer — the
only address the server learned rather than was told. Forwarded addresses are
consulted only when `api_trust_forwarded_for` is set, and then only from the
right, because a proxy appends to whatever the client sent.

The budget is per process and per client, so `N` replicas multiply it by `N`.
The token is reduced to a digest before it becomes part of a key, so a
rate-limit bucket — which outlives the request and can reach a log — holds no
key material.

## The audit log

`GET /v1/approvals/by-run/{run_id}/audit` re-derives every decision signature
rather than trusting the stored value, because the decision log lives inside the
graph state and inherits the trust level of whatever can write to the checkpoint
store. A signature that does not verify is reported as such: the entry carries
`verified: false`, the count comes back in `unverified`, and the endpoint
answers 409 so monitoring can alert on it. `verified: null` means no secret is
configured, so the decision could not be checked either way.

The signing secret is `AWF_HITL_SIGNING_SECRET`, falling back to
`AWF_API_AUTH_TOKEN`. The LLM credential is deliberately *not* a candidate: it
belongs to a third party, it is the most widely distributed secret in most
deployments, and tying the integrity of the audit log to whether an LLM
happens to be configured would be a strange coupling.

What a signature proves: the record has not been edited since it was written by
someone without the secret. What it does not prove: that the decision was wise,
that the signature came from the person who made the decision, or that the
checkpoint store was not writable at the time. It is tamper-*evidence*, and the
name says so.

## What an attacker is up against

| They try | What stops them |
| --- | --- |
| Reading runs, checkpoints, decisions | Every `/v1` route answers 401 without the token. |
| Approving a review | The same, and the audit trail shows the decision was made. |
| Guessing the token | Constant-time comparison on both transports, no hint in any body, and a per-client rate limit. |
| Timing their way to the token | `hmac.compare_digest` over HTTP and over the socket. |
| Brute-forcing through the socket | The same rate limiter keys on the peer address when there is no verified token, and a refused handshake is refused before any data flows. |
| Reading `/metrics` | Nothing. Keep it internal. |
| Reading the token from a log | The application never logs it. A reverse proxy logging query strings will. |
| Editing a decision after the fact | The signature, if a secret is configured. Without one, the boot log already said so. |
| Getting a browser to send the token cross-site | Nothing to steal: the token is a header, not a cookie, so there is no ambient credential for a cross-site request to ride on. A dashboard that keeps it in `localStorage` is exposed to any XSS on its own origin. |

## What this does not do

- **No per-user identity or authorization.** One token, one principal. See
  above.
- **No mTLS, no OIDC, no OAuth.** Static bearer only.
- **No secret rotation without a restart.** Settings are read once.
- **No rate limiting across processes.** The budget is per worker.
- **No encryption at rest.** Checkpoints hold the source under review and every
  decision; that is PostgreSQL's job and its backup's.
- **No audit of *who*.** Decisions are signed, not attributed.
- **No protection for the checkpoint store itself.** Anyone who can reach the
  database can rewrite state. The signatures are what make that detectable
  afterwards, which is a different guarantee from preventing it.
