# Feature: Queue Core

Date: 2026-06-12
Status: Draft
Owner: Open Superintelligence Lab

## One Sentence

A standalone queue contract that lets humans, agents, GPUs, CLIs, websites, and future databases exchange lab work without depending on one app.

## Decision

The queue is the core.

Everything else is an adapter:

- Public website
- Private lab dashboard
- CLI
- Local review server
- GPU worker
- Agent runner
- Future API
- Future database

The queue must work without Next.js, Supabase, GitHub Pages, a browser, a deploy, or a specific repo layout.

## Why

The lab already has queue-shaped pieces:

- local specs
- experiment result files
- public snapshots
- `/lab/queue`
- local review controls

But there is no single contract for creating work, claiming it, returning evidence, and reviewing it.

If that contract lives inside one UI, every future tool will reinvent the queue. The queue needs to be a small portable layer that other systems plug into.

## Scope

The queue should support these job types:

- `experiment`
- `review`
- `code`
- `write`
- `ops`

The type is secondary. The shared lifecycle is the real feature:

```text
draft -> ready -> claimed -> running -> submitted -> accepted
                                      -> rejected
                                      -> requeued
```

## Non-Goals

Do not put these in the queue core:

- Auth
- Payments
- Reputation
- Website rendering
- GPU scheduling
- Public write access
- Full project management

Build those around the queue later.

## Core Objects

### Job

One unit of work.

```yaml
id: arq-030-unetskip
kind: experiment
title: FIRE + U-Net sigmoid skips
status: ready
priority: 3
owner: vukrosic
created_at: "2026-06-12T00:00:00Z"
updated_at: "2026-06-12T00:00:00Z"
input:
  brief: Test whether sigmoid-gated U-Net skips improve Tiny1M3M validation loss.
expected_output:
  summary: Compare final validation loss against baseline.
  artifacts: [result.json, metrics.json]
constraints:
  gpu_vram_gb: 24
  max_hours: 2
```

Optional fields can include repo, branch, command, baseline, metric, acceptance checks, and public summary.

### Claim

An expiring lease on a job.

```yaml
job_id: arq-030-unetskip
lease_id: lease_20260612_001
worker_id: gpu-box-01
claimed_at: "2026-06-12T01:00:00Z"
expires_at: "2026-06-12T03:30:00Z"
heartbeat_at: "2026-06-12T01:30:00Z"
```

Claims must expire so a dead worker cannot block the queue.

### Result

The worker's report.

```json
{
  "job_id": "arq-030-unetskip",
  "lease_id": "lease_20260612_001",
  "worker_id": "gpu-box-01",
  "finished_at": "2026-06-12T03:00:00Z",
  "exit_status": "success",
  "summary": "Validation loss improved versus baseline.",
  "metrics": {
    "val_loss": 5.008125,
    "tokens_per_sec": 19509.88
  },
  "artifacts": [
    "results/arq-030-unetskip/run-001/result.json",
    "results/arq-030-unetskip/run-001/training.log"
  ]
}
```

Failed runs are still results. They should include the error, logs, and retry recommendation.

### Review

A decision on a job or result.

```yaml
target_type: result
target_id: arq-030-unetskip/run-001
reviewer_id: vukrosic
decision: approve
created_at: "2026-06-12T04:00:00Z"
note: Promote to public lab snapshot.
```

Allowed decisions:

- `approve`
- `request_changes`
- `reject`
- `keep`

### Event

Append-only record of state changes.

Examples:

- `job.created`
- `job.claimed`
- `claim.heartbeat`
- `result.submitted`
- `review.created`
- `job.requeued`

The event log is what lets files, APIs, dashboards, and databases stay compatible.

## Core Verbs

Every adapter should use the same verbs:

```text
create_job(payload) -> job
list_jobs(filter) -> job[]
get_job(id) -> job
update_job(id, patch) -> job
claim_job(id, worker) -> claim
heartbeat_claim(lease_id) -> claim
release_claim(lease_id, reason) -> event
submit_result(lease_id, payload) -> result
review_target(target_type, target_id, payload) -> review
export_snapshot(filter) -> snapshot
```

No UI should become the queue. UIs either call these verbs or read snapshots.

## File Adapter

First implementation should be file-backed:

```text
queue/
  jobs/
    arq-030-unetskip.yaml
  claims/
    lease_20260612_001.json
  events.jsonl
  reviews.jsonl
results/
  arq-030-unetskip/
    run-001/
      result.json
      metrics.json
      training.log
public/
  queue-snapshot.json
```

This layout is only the first adapter. A database can implement the same verbs later.

## Public Boundary

The public website reads a sanitized snapshot.

It does not read the working queue directly.

```json
{
  "generated_at": "2026-06-12T03:05:03Z",
  "jobs": [
    {
      "id": "arq-030-unetskip",
      "kind": "experiment",
      "title": "FIRE + U-Net sigmoid skips",
      "status": "done",
      "summary": "Validation loss improved versus baseline."
    }
  ],
  "results": []
}
```

Public snapshots exclude local paths, secrets, raw logs, internal notes, and machine details by default.

## Adapters

Producers create jobs:

- Human spec
- Agent-generated task
- Website submission
- Follow-up from a result

Workers claim jobs and submit results:

- CLI
- GPU box
- Agent runner
- Human contributor
- GitHub Action

Reviewers decide what graduates:

- Approve
- Request changes
- Reject
- Keep

Surfaces display or operate the queue:

- `/lab/queue`
- `/lab/my`
- terminal dashboard
- future authenticated app

## First Build

Build only the core and local file adapter.

Deliverables:

1. Schemas for job, claim, result, review, and event.
2. File-backed module implementing the core verbs.
3. CLI commands: `list`, `claim`, `heartbeat`, `submit`, `review`, `export`.
4. Public snapshot exporter.
5. Migration path from current `queue/*.yaml` and result files.

Do not build auth or public write access yet.

## Done

This feature is done when:

- A worker can list ready jobs without chat context.
- A worker can claim a job with an expiring lease.
- A worker can heartbeat or release the claim.
- A worker can submit success or failure as a structured result.
- A reviewer can approve, request changes, reject, or keep a job/result.
- The website can render queue state from a sanitized snapshot.
- The queue works without a web server, browser, deploy, Supabase, or Next.js.
- A future API or database can implement the same verbs without changing worker behavior.

## Deferred Questions

- What is the smallest identity model for outside contributors?
- Should first-pass reviews attach to jobs, results, or both?
- What review gate promotes a result into a public paper or leaderboard entry?
