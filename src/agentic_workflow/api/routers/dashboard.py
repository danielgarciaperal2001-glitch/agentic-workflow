"""Read-only dashboard for observing the workflow in real time."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> str:
    """Serve a minimal, read-only dashboard that streams events live."""
    return """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Workflow Dashboard</title>
    <style>
      :root {
        color-scheme: light dark;
        --bg: #0f172a;
        --fg: #e2e8f0;
        --card-bg: rgba(15, 23, 42, 0.6);
        --border: rgba(148, 163, 184, 0.4);
        --accent: #38bdf8;
      }
      body {
        margin: 0;
        padding: 1.5rem;
        background: var(--bg);
        color: var(--fg);
        font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      }
      h1 {
        margin: 0 0 1rem;
        font-size: 1.5rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
      }
      main {
        display: grid;
        gap: 1rem;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      }
      section {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 0.75rem;
        padding: 1rem;
        backdrop-filter: blur(8px);
        box-shadow: 0 20px 50px -25px rgba(56, 189, 248, 0.4);
      }
      h2 {
        margin: 0 0 0.75rem;
        font-size: 0.875rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.85;
      }
      ul {
        list-style: none;
        margin: 0;
        padding: 0;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        max-height: 20rem;
        overflow: auto;
      }
      li {
        padding: 0.5rem 0.75rem;
        border: 1px solid var(--border);
        border-radius: 0.5rem;
        background: rgba(15, 23, 42, 0.4);
        font-size: 0.875rem;
        line-height: 1.4;
        white-space: pre-wrap;
        word-break: break-word;
      }
      .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 0.75rem;
      }
      .stat {
        border: 1px solid var(--border);
        border-radius: 0.5rem;
        padding: 0.75rem;
        background: rgba(15, 23, 42, 0.4);
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
      }
      .stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.65;
      }
      .stat-value {
        font-size: 1.5rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
      }
      .muted {
        opacity: 0.65;
        font-size: 0.875rem;
      }
      footer {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
        font-size: 0.75rem;
        opacity: 0.65;
      }
      code {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      }
    </style>
  </head>
  <body>
    <h1>Workflow Dashboard</h1>
    <main>
      <section>
        <h2>Overview</h2>
        <div class="stat-grid">
          <div class="stat">
            <span class="stat-label">Runs registered</span>
            <span class="stat-value" id="runs-registered">0</span>
          </div>
          <div class="stat">
            <span class="stat-label">Events published</span>
            <span class="stat-value" id="events-published">0</span>
          </div>
          <div class="stat">
            <span class="stat-label">Events delivered</span>
            <span class="stat-value" id="events-delivered">0</span>
          </div>
          <div class="stat">
            <span class="stat-label">Events dropped</span>
            <span class="stat-value" id="events-dropped">0</span>
          </div>
          <div class="stat">
            <span class="stat-label">Subscribers</span>
            <span class="stat-value" id="subscribers">0</span>
          </div>
          <div class="stat">
            <span class="stat-label">Subscribers max</span>
            <span class="stat-value" id="subscribers-max">0</span>
          </div>
        </div>
      </section>
      <section>
        <h2>Refusals</h2>
        <ul id="refusals"></ul>
        <p id="refusals-empty" class="muted">No refusals recorded.</p>
      </section>
      <section>
        <h2>Active Runs</h2>
        <ul id="runs"></ul>
        <p id="runs-empty" class="muted">No runs found.</p>
      </section>
      <section>
        <h2>Pending Approvals</h2>
        <ul id="approvals"></ul>
        <p id="approvals-empty" class="muted">No pending approvals.</p>
      </section>
      <section>
        <h2>Live Events</h2>
        <ul id="events"></ul>
        <p id="events-empty" class="muted">Waiting for events...</p>
      </section>
    </main>
    <footer>
      Read-only. Connects to <code id="ws-url"></code> and polls REST endpoints periodically.
    </footer>
    <script>
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/stream`;
      document.getElementById('ws-url').textContent = wsUrl;
      const runsList = document.getElementById('runs');
      const runsEmpty = document.getElementById('runs-empty');
      const approvalsList = document.getElementById('approvals');
      const approvalsEmpty = document.getElementById('approvals-empty');
      const eventsList = document.getElementById('events');
      const eventsEmpty = document.getElementById('events-empty');
      const refusalsList = document.getElementById('refusals');
      const refusalsEmpty = document.getElementById('refusals-empty');

      function setStat(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value ?? '0';
      }

      function clearList(list, emptyEl) {
        list.innerHTML = '';
        if (emptyEl) emptyEl.style.display = 'block';
      }

      function appendItem(list, emptyEl, text) {
        if (emptyEl) emptyEl.style.display = 'none';
        const li = document.createElement('li');
        li.textContent = text;
        list.prepend(li);
        while (list.childElementCount > 200) {
          list.removeChild(list.lastElementChild);
        }
      }

      async function fetchJSON(url) {
        try {
          const res = await fetch(url);
          if (!res.ok) return null;
          return await res.json();
        } catch (err) {
          return null;
        }
      }

      async function refreshRuns() {
        const data = await fetchJSON('/v1/runs?limit=50&offset=0');
        runsList.innerHTML = '';
        if (!data || !Array.isArray(data.items) || data.items.length === 0) {
          runsEmpty.style.display = 'block';
          return;
        }
        runsEmpty.style.display = 'none';
        for (const run of data.items.slice(0, 50)) {
          const li = document.createElement('li');
          const parts = [run.run_id, run.status].filter(Boolean);
          if (run.title) parts.push(run.title);
          if (run.request_id) parts.push(run.request_id);
          if (run.started_at) parts.push(run.started_at);
          li.textContent = parts.join(' • ');
          runsList.appendChild(li);
        }
      }

      async function refreshApprovals() {
        const data = await fetchJSON('/v1/approvals/pending?limit=50&offset=0');
        approvalsList.innerHTML = '';
        if (!data || !Array.isArray(data.items) || data.items.length === 0) {
          approvalsEmpty.style.display = 'block';
          return;
        }
        approvalsEmpty.style.display = 'none';
        for (const ap of data.items.slice(0, 50)) {
          const li = document.createElement('li');
          const parts = [ap.approval_id, ap.run_id, ap.status].filter(Boolean);
          if (ap.question) parts.push(ap.question);
          if (ap.expires_at) parts.push(`expires ${ap.expires_at}`);
          li.textContent = parts.join(' • ');
          approvalsList.appendChild(li);
        }
      }

      function parseMetrics(text) {
        const result = { events: {}, refusals: {} };
        for (const line of text.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith('#')) continue;
          if (trimmed.startsWith('awf_metric{name=')) {
            const m = trimmed.match(/awf_metric\\{name="([^"]+)"\\}\\s+([^\\s]+)/);
            if (m) {
              result.events[m[1]] = Number(m[2]);
            }
            continue;
          }
          if (trimmed.startsWith('awf_runs_registered')) {
            const m = trimmed.match(/awf_runs_registered\\s+([^\\s]+)/);
            if (m) result.events['awf_runs_registered'] = Number(m[1]);
            continue;
          }
          if (trimmed.startsWith('awf_refusals_total')) {
            const m = trimmed.match(/awf_refusals_total\\{reason="([^"]+)"\\}\\s+([^\\s]+)/);
            if (m) {
              result.refusals[m[1]] = Number(m[2]);
            }
            continue;
          }
          if (trimmed.startsWith('awf_events_')) {
            const m = trimmed.match(/awf_events_([^\\s{]+)\\s+([^\\s]+)/);
            if (m) {
              result.events[`awf_events_${m[1]}`] = Number(m[2]);
            }
          }
        }
        return result;
      }

      async function refreshMetrics() {
        try {
          const res = await fetch('/metrics');
          if (!res.ok) return;
          const text = await res.text();
          const m = parseMetrics(text);
          const events = m.events;
          setStat('runs-registered', events.awf_runs_registered ?? events.runs ?? 0);
          setStat('events-published', events.awf_events_published ?? 0);
          setStat('events-delivered', events.awf_events_delivered ?? 0);
          setStat('events-dropped', events.awf_events_dropped ?? 0);
          setStat('subscribers', events.awf_events_subscribers ?? 0);
          setStat('subscribers-max', events.awf_events_subscribers_max ?? 0);
          refusalsList.innerHTML = '';
          const reasons = Object.keys(m.refusals).sort();
          if (reasons.length === 0) {
            refusalsEmpty.style.display = 'block';
          } else {
            refusalsEmpty.style.display = 'none';
            for (const r of reasons) {
              const li = document.createElement('li');
              li.textContent = `${r}: ${m.refusals[r]}`;
              refusalsList.appendChild(li);
            }
          }
        } catch (err) {
          // ignore
        }
      }

      function connect() {
        const ws = new WebSocket(wsUrl);
        ws.onmessage = (ev) => {
          try {
            const data = JSON.parse(ev.data);
            const event = data.event || 'unknown';
            const ts = new Date().toLocaleTimeString();
            appendItem(eventsList, eventsEmpty, `[${ts}] ${event} ${data.run_id ? `(run ${data.run_id})` : ''}`);
          } catch (err) {
            appendItem(eventsList, eventsEmpty, ev.data);
          }
        };
        ws.onclose = () => {
          setTimeout(connect, 2000);
        };
        ws.onerror = () => {
          ws.close();
        };
      }

      async function boot() {
        await Promise.all([refreshRuns(), refreshApprovals(), refreshMetrics()]);
        connect();
        setInterval(refreshRuns, 5000);
        setInterval(refreshApprovals, 5000);
        setInterval(refreshMetrics, 5000);
      }

      boot();
    </script>
  </body>
</html>
"""
