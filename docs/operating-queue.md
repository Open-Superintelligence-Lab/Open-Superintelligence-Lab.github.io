# Open Superintelligence Lab Operating Queue

Date: 2026-06-12

This note comes from the current `my-life` scan and the `open-superintelligence-lab-github-io` repo.

## What Exists

- `app/page.tsx` already carries the public mission statement.
- `app/lab/*` already exposes the public lab surface: Goals, Research, Ideas, Problems, Leaderboard, and Compute.
- `app/lab/my/page.tsx` already shows a private-looking workspace, but it still reads from public JSON and does not enforce access.
- `public/data/lab/*.json` is the current site data layer.
- `docs/lab-platform-plan.md` already describes the three-stage platform split.
- `coaching-offer.md` and the surrounding coaching pages already contain the funding offer.

## Scan Notes

- The public site is ahead of the platform: the homepage and lab shell exist, but the contribution system does not.
- The compute page explains the model, but not the attach / claim / report flow.
- The queue is only described in prose; there is no job schema yet.
- The human approval loop is still just an idea.
- The daily content funnel exists in fragments, not as one system.

## Current Priority Queue

### P0. Make the private side real

- Define what is public data and what requires login.
- Keep public state in `public/data/lab/*`.
- Move private projects, notes, prompts, submissions, and approvals behind a real auth boundary.
- Do not keep the private area as a visual-only split.

Acceptance:
- A reader can tell which data is public, which is private, and where each one lives.
- The current private workspace no longer depends on the same public JSON source as the public views.

### P1. Define the job queue contract

- Create one small job format for research, review, hypothesis writing, and coding.
- Include `title`, `input`, `owner`, `priority`, `status`, `expected_output`, and `result`.
- Add one canonical worker lifecycle: claim -> run -> upload -> mark done.
- Make it easy for a contributor to clone/install and start pulling tasks.

Acceptance:
- A worker can discover a task without manual guidance.
- A worker can write a result back in a predictable format.

### P2. Make compute contribution operational

- Turn the compute page into a concrete attach flow.
- Add machine attach, heartbeat, job pickup, artifact upload, and failure reporting.
- Make it clear what donated compute should do first.
- Keep the model simple enough that a non-core contributor can follow it.

Acceptance:
- The page explains how a machine joins the queue.
- The page explains what comes back from a completed job.

### P3. Add human review and vision approval

- Add approve / request changes / reject as first-class states.
- Attach a short human-readable vision summary to each major initiative.
- Keep the review surface simple enough that people can act on it quickly.

Acceptance:
- Ideas can be reviewed without reading internal chat history.
- Humans can see the lab direction and respond to it in one place.

### P4. Turn research output into daily distribution

- Publish one public post per day from a real result.
- Tie each post to a booking CTA for the 1:1 offer.
- Track leads, calls, and closes.
- Keep the sales engine separate from the public mission.

Acceptance:
- Every post comes from an actual result, not a filler prompt.
- Every post has a next step that can produce revenue.

### P5. Tighten the public control panel

- Make the homepage, `/lab`, and `/about` read like one system.
- Show public mission, public goals, active papers, and contribution paths in one clear flow.
- Add a "how to contribute" page if the current nav still scatters the story.

Acceptance:
- A new visitor can understand the lab in under a minute.
- A contributor can find the right path without asking for directions.

## Build Order

1. Data contracts.
2. Queue spec.
3. Private boundary.
4. Compute attach flow.
5. Human review flow.
6. Daily content / revenue loop.
7. Navigation and copy cleanup.

## Weekly Operating Rule

- Ship one visible site improvement.
- Ship one research artifact.
- Publish one post.
- Improve one revenue step.
