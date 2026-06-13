import { execFile } from 'child_process';
import { promisify } from 'util';
import { readFile, readdir } from 'fs/promises';
import { join } from 'path';
import { launchCodexWithText, RESEARCH_REPO_DIR } from '@/lib/codexLauncher';
import { getAutorunAgent, setAutorun } from '@/lib/autorun';

const execFileAsync = promisify(execFile);
const IDEAS_DIR = join(RESEARCH_REPO_DIR, 'autoresearch', 'ideas');
const RUNNER_PROMPT = join(RESEARCH_REPO_DIR, 'autoresearch', 'prompts', 'runner.md');
const BOX_JSON = join(RESEARCH_REPO_DIR, 'autoresearch', 'remote-box.json');

// One persistent runner agent owns the GPU queue while autorun is on. It reads
// runner.md, claims the WHOLE needs-run set, and drains it on the box's detached
// `arq` tmux — exactly the model the existing cron uses. We just re-invoke it
// from the UI poll instead of cron, gated by the autorun flag.
const RUNNER_SESSION = 'lab-autorun';

function field(md: string, key: string): string {
  const m = md.match(new RegExp(`^${key}:\\s*(.+)$`, 'm'));
  return m ? m[1].trim() : '';
}

async function needsRunCount(): Promise<number> {
  let dirs: string[];
  try {
    dirs = await readdir(IDEAS_DIR);
  } catch {
    return 0;
  }
  let n = 0;
  for (const dir of dirs) {
    try {
      const md = await readFile(join(IDEAS_DIR, dir, 'idea.md'), 'utf8');
      if (field(md, 'status') === 'needs-run') n += 1;
    } catch {
      /* skip */
    }
  }
  return n;
}

async function runnerAlive(): Promise<boolean> {
  try {
    await execFileAsync('tmux', ['has-session', '-t', RUNNER_SESSION], { timeout: 5_000 });
    return true;
  } catch {
    return false;
  }
}

async function killRunner(): Promise<void> {
  try {
    await execFileAsync('tmux', ['kill-session', '-t', RUNNER_SESSION], { timeout: 5_000 });
  } catch {
    /* already gone */
  }
}

// Build the runner-pass instruction: tell the agent to read runner.md + PIPELINE
// and run ONE pass, with the LIVE box pulled from remote-box.json (so a new Vast
// instance is picked up just by editing that file). The agent reads the full
// protocol from runner.md itself — we don't inline it.
async function runnerPrompt(): Promise<string> {
  let box: Record<string, string> = {};
  try {
    box = JSON.parse(await readFile(BOX_JSON, 'utf8'));
  } catch {
    /* runner.md §0 falls back to the latest results.json box */
  }
  const sshLine = box.ssh
    ? `Live box (rented): ${box.ssh} (-o StrictHostKeyChecking=accept-new). Repo on box: ${box.remote_repo ?? '/root/universe-lm'}. Python: ${box.remote_venv ?? '/venv/main'}/bin/python (export PATH=${box.remote_venv ?? '/venv/main'}/bin:$PATH). Hardware: ${box.hardware ?? 'see remote-box.json'}.`
    : 'No box in remote-box.json — follow runner.md §0 to recover the live box, or print NO BOX and stop.';

  return [
    `Read autoresearch/prompts/runner.md and autoresearch/PIPELINE.md and execute ONE full runner pass exactly as specified. You are in ${RESEARCH_REPO_DIR}.`,
    '',
    sshLine,
    '',
    "If a queue tmux session named `arq` is already live on the box, do NOT relaunch it — poll STATUS, pull finished logs, finalize evidence.md + status flips, then exit. One pass per invocation; no waiting loops longer than a few minutes. The runs live in the box's detached tmux, so this agent exiting does not stop them.",
  ].join('\n');
}

// Read or toggle autorun. POST with no `enabled` = read state + tick. The tick
// (re-)launches the single runner agent only when autorun is on, no runner is
// already alive, and there is actually needs-run work — so the UI poll keeps the
// queue draining without ever stacking two runners.
export async function POST(req: Request) {
  let body: { enabled?: unknown; agent?: unknown } = {};
  try {
    body = await req.json();
  } catch {
    /* state read + tick */
  }

  if (typeof body.enabled === 'boolean') {
    const agent = typeof body.agent === 'string' ? body.agent : 'minimax';
    await setAutorun(body.enabled ? agent : null);
    if (!body.enabled) {
      // Stop the loop: the runner self-exits per pass, but kill the session now
      // so the GPU queue visibly stops re-launching.
      await killRunner();
    }
  }

  const current = await getAutorunAgent();
  let launched = false;
  let alive = await runnerAlive();
  if (current && !alive && (await needsRunCount()) > 0) {
    const result = await launchCodexWithText(
      await runnerPrompt(),
      'lab-autorun',
      RESEARCH_REPO_DIR,
      RUNNER_SESSION,
      current,
      { headless: true }
    );
    launched = result.success;
    alive = result.success;
  }

  return Response.json(
    { success: true, enabled: current !== null, agent: current, runnerAlive: alive, launched },
    { status: 200 }
  );
}
