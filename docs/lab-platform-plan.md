# Lab platform plan (/lab)

Goal: opensuperintelligencelab.com/lab is the public window into the lab —
people see the main experiments live, and (stage 2) log in to track their own.

## Stage 1 — shipped
- `/lab` route: papers, per-experiment verdict badges (win/null/worse/diverged/pending),
  inline SVG val-loss curves. Dark theme matching the site.
- Data pipeline: the research repo (`llm-research-kit-scaling`) owns the truth.
  `tools/export_site_data.py` walks `token2science/papers/*/paper.json` +
  `figures/curves.json` and writes `public/data/lab/summary.json` here.
  Publishing = run exporter → commit JSON → deploy. No server needed (static export).

## Stage 2 — user accounts + own experiments (not built yet)
Constraint: GitHub Pages = static only. Everything dynamic must be client-side.
- **Auth**: Supabase (free tier) with GitHub OAuth, called from the browser.
- **Own experiments**: `experiments` table (user_id, name, config_json, baseline,
  result, status, curves_json). CRUD from a `/lab/mine` client page.
- **Main experiments stay file-based** (paper.json in the research repo is the
  source of truth; reviewable in git, agents write it). Supabase only holds
  community/user data — never lab results.
- **Submissions**: a user can submit an experiment proposal → lands in Supabase →
  lab triages → accepted ones get run on lab GPUs and graduate into a paper.json.

## Stage 3 — live runs
- The local `tools/status_dashboard.py` (SSH-polling live board) stays local/private.
  Public live view = exporter run on a cron that commits data snapshots
  (GitHub Action on the research repo, or the existing launchd runner-tick).

## Conventions
- Website never reaches into the research repo at build time; it only reads
  `public/data/lab/*.json` committed here. Keeps repos decoupled and deploys pure.
