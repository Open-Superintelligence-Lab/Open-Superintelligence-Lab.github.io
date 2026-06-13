# Queue Core

Standalone file-backed queue contract for Open Superintelligence Lab work.

This folder is intentionally independent from the Next.js app. The website, local review UI, GPU workers, agents, and future database/API layers should treat this as the queue contract or as an adapter to that contract.

## Commands

Run from the repo root:

```bash
node queue-core/src/cli.js init
node queue-core/src/cli.js list
node queue-core/src/cli.js claim arq-030-unetskip --worker gpu-box-01
node queue-core/src/cli.js heartbeat lease_20260612_001
node queue-core/src/cli.js submit lease_20260612_001 --file result.json
node queue-core/src/cli.js revise-result arq-030-unetskip vukrosic-20260602T073131Z --file result.json
node queue-core/src/cli.js review result arq-030-unetskip/run-001 --reviewer vukrosic --decision approve
node queue-core/src/cli.js export --public public/data/queue-snapshot.json
```

Use `--root <path>` to point the queue at a different data root.

## Migration

To import the current local research queue:

```bash
node queue-core/src/cli.js migrate \
  --source-queue ../llm-research-kit-scaling/queue \
  --source-results ../llm-research-kit-scaling/results
```

The migration reads simple queue YAML files and result JSON files, then writes normalized queue-core records under `queue-core/data/`.

## Folder Shape

```text
queue-core/
  schemas/
  src/
  data/
    jobs/
    claims/
    results/
    events.jsonl
    reviews.jsonl
```

`data/` is the first storage adapter, not a permanent platform decision. A database-backed adapter can implement the same verbs later.

Result updates are revisioned:

- the latest `result.json` is the current version
- older versions are copied into `revisions/<revision-id>.json`
- `revise-result` updates the current file without deleting history
