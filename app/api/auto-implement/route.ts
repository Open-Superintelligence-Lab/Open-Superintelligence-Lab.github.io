import { execFile } from 'child_process';
import { promisify } from 'util';
import { readFile, readdir } from 'fs/promises';
import { join } from 'path';
import { RESEARCH_REPO_DIR } from '@/lib/codexLauncher';
import {
  getAutoImplementAgent,
  setAutoImplement,
  MAX_PARALLEL_IMPLEMENTS,
} from '@/lib/autoimplement';

const execFileAsync = promisify(execFile);
const IDEAS_DIR = join(RESEARCH_REPO_DIR, 'autoresearch', 'ideas');
const IMPLEMENT_PREFIX = 'lab-implement-';

function field(md: string, key: string): string {
  const m = md.match(new RegExp(`^${key}:\\s*(.+)$`, 'm'));
  return m ? m[1].trim() : '';
}

type Idea = { slug: string; status: string; updated: string };

async function allIdeas(): Promise<Idea[]> {
  let dirs: string[];
  try {
    dirs = await readdir(IDEAS_DIR);
  } catch {
    return [];
  }
  const ideas: Idea[] = [];
  for (const dir of dirs) {
    try {
      const md = await readFile(join(IDEAS_DIR, dir, 'idea.md'), 'utf8');
      ideas.push({
        slug: field(md, 'id') || dir,
        status: field(md, 'status'),
        updated: field(md, 'updated'),
      });
    } catch {
      /* skip non-idea dirs */
    }
  }
  return ideas;
}

// Slugs that already have a live implement tmux session — our concurrency unit
// (the launcher creates the session synchronously, so this is reliable between
// ticks and prevents double-launching the same idea).
async function activeImplementSlugs(): Promise<Set<string>> {
  try {
    const { stdout } = await execFileAsync('tmux', ['ls', '-F', '#{session_name}'], {
      timeout: 10_000,
    });
    const slugs = stdout
      .split('\n')
      .map((s) => s.trim())
      .filter((s) => s.startsWith(IMPLEMENT_PREFIX))
      .map((s) => s.slice(IMPLEMENT_PREFIX.length));
    return new Set(slugs);
  } catch {
    return new Set();
  }
}

// Launch implements for Proposed (needs-taste) ideas, oldest first, up to the
// parallel cap. Returns the slugs it launched this tick.
async function tick(host: string, agent: string): Promise<string[]> {
  const ideas = await allIdeas();
  const active = await activeImplementSlugs();

  const proposed = ideas
    .filter((i) => i.status === 'needs-taste' && !active.has(i.slug))
    .sort((a, b) => a.updated.localeCompare(b.updated));

  let budget = MAX_PARALLEL_IMPLEMENTS - active.size;
  const launched: string[] = [];
  for (const idea of proposed) {
    if (budget <= 0) break;
    // Reuse the existing implement-idea endpoint (claims the idea, sets up the
    // done-curl, launches the agent). Fire-and-forget; failures just mean the
    // next tick retries this slug.
    fetch(`http://${host}/api/implement-idea/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slug: idea.slug, agent, headless: true }),
    }).catch(() => {
      /* retried next tick */
    });
    launched.push(idea.slug);
    budget -= 1;
  }
  return launched;
}

// Read or toggle auto-implement, and (whenever it's on) drive one tick. The UI
// calls this on load, on toggle, and on its poll — each call both reports state
// and keeps the implement pipeline filled up to the cap.
export async function POST(req: Request) {
  let body: { enabled?: unknown; agent?: unknown } = {};
  try {
    body = await req.json();
  } catch {
    /* state read + tick */
  }

  if (typeof body.enabled === 'boolean') {
    const agent = typeof body.agent === 'string' ? body.agent : 'minimax';
    await setAutoImplement(body.enabled ? agent : null);
  }

  const current = await getAutoImplementAgent();
  let launched: string[] = [];
  if (current) {
    const host = req.headers.get('host') ?? 'localhost:3001';
    launched = await tick(host, current);
  }

  return Response.json(
    {
      success: true,
      enabled: current !== null,
      agent: current,
      maxParallel: MAX_PARALLEL_IMPLEMENTS,
      launched,
    },
    { status: 200 }
  );
}
