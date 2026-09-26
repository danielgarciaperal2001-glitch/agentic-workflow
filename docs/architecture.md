# Architecture: Multi-Agent AI Workflow Engine with Human-in-the-Loop

A production-grade system built around LangGraph's `StateGraph`, PostgreSQL-backed checkpointers for persistence, and explicit human gates for approval/modify/reject. 

## Overview
- Domain-first design (no dependencies on infra in `domain/`)
- Separation of wire models (`api/schemas.py`) from domain (`domain/schemas.py`) with `to_domain()`
- Deterministic approval IDs, explicit status model, transport-agnostic event sink
- HITL split: policy (should we ask?), gates (plumbing), service (REST-facing)
- Checkpointer-authoritative reads; registry is rebuildable

## Key Components
- Graph: Programmer → Reviewer → Tester with feedback loops
- Persistence: PostgreSQL checkpointer with resume, time-travel
- HITL: interrupt-based gates with deterministic approval IDs
- Evals: native, deterministic metrics + optional judge adapters
- API: FastAPI with rate limiting, audit trail, WebSocket events

See README for the Mermaid architecture diagram.
