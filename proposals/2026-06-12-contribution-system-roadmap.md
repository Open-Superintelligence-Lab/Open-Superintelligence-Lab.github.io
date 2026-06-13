# Proposal: Contribution system roadmap

- **Date:** 2026-06-12
- **Status:** approved (Vuk, in chat, 2026-06-12)
- **Owner:** orchestrator

## Goal

Make the #1 fully open-source LLM. To get there, people must be able to
contribute to the lab without asking permission. Four contribution paths,
built in this order:

## 1. Vision / approval surface (NOW)

A public page where the orchestrator's proposals are visible and humans can
approve, reject, or request changes.

- v1: this folder (`orchestrator/proposals/`) — already live.
- v2: website page that renders these proposals — delegated to Codex today.

## 2. Task queue people can pull from (NEXT)

"git clone / npm install something and it pulls AI tasks (research, review,
hypothesis writing, coding) and works massively in parallel."

- We already have two seeds: `gitswarm/` (server + task orchestration) and
  `experiment-registry/` (registry + scripts + tests).
- Step 1 is an audit + convergence plan (delegated to Codex today), so we
  extend one of them instead of building a third thing.

## 3. Compute client (AFTER 2)

Same install story, but the task is "run this experiment/training on your
GPU and report results back." Reuses the queue from path 2 — different task
type, same plumbing.

## 4. Personal research spaces (LAST)

Private project spaces with prompts, tools, and instructions for automated
AI research. Lowest urgency: a single person can already do this with the
repo's existing skills; it becomes valuable once paths 1–3 bring people in.

## Why this order

Path 1 is cheapest and unblocks the orchestrator running autonomously.
Path 2 is the moat and feeds path 3 directly. Path 4 needs an audience that
paths 1–3 create.

## Funding context

Coaching ($20/60min 1-on-1) is the bootstrap runway: $159 confirmed so far,
working goal 2 paid calls/week. Daily posts feed it. The lab's long-term
funding is a business on top of the open lab, not coaching.
