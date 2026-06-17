import { execFile } from 'child_process';
import { promisify } from 'util';
import { readFile, readdir } from 'fs/promises';
import { join } from 'path';
import { launchCodexWithText, RESEARCH_REPO_DIR } from '@/lib/codexLauncher';

const execFileAsync = promisify(execFile);
const IDEAS_DIR = join(RESEARCH_REPO_DIR, 'autoresearch', 'ideas');
const FLIP_SH = join(RESEARCH_REPO_DIR, 'autoresearch', 'bin', 'flip.sh');
const TEMPLATE_PATH = join(RESEARCH_REPO_DIR, 'autoresearch', 'prompts', 'run-idea.md');

function field(md: string, key: string): string {
  const m = md.match(new RegExp(`^${key}:\\s*(.+)$`, 'm'));
  return m ? m[1].trim() : '';
}

type QueueIdea = { slug: string; status: string; updated: string };

async function requeueAfterFailure(slug: string) {
  try {
    await execFileAsync(FLIP_SH, [slug, 'needs-run', 'run-button', 'launch failed; requeued'], {
      cwd: RESEARCH_REPO_DIR,
      timeout: 15_000,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error('Failed to requeue after run-next failure:', message);
  }
}

async function queueIdeas(): Promise<QueueIdea[]> {
  let dirs: string[];
  try {
    dirs = await readdir(IDEAS_DIR);
  } catch {
    return [];
  }
  const ideas: QueueIdea[] = [];
  for (const dir of dirs) {
    try {
      const md = await readFile(join(IDEAS_DIR, dir, 'idea.md'), 'utf8');
      const status = field(md, 'status');
      if (status === 'needs-run' || status === 'running') {
        ideas.push({ slug: field(md, 'id') || dir, status, updated: field(md, 'updated') });
      }
    } catch {
      /* skip */
    }
  }
  ideas.sort((a, b) => a.updated.localeCompare(b.updated));
  return ideas;
}

export async function POST(req: Request) {
  let agent: string | undefined;
  let headless = true;
  try {
    const body = await req.json();
    agent = body.agent;
    if (typeof body.headless === 'boolean') headless = body.headless;
  } catch {
    agent = undefined;
  }

  const queue = await queueIdeas();
  const running = queue.find((idea) => idea.status === 'running');
  if (running) {
    return Response.json(
      { success: false, busy: true, slug: running.slug, error: `${running.slug} is already running` },
      { status: 200 }
    );
  }

  const next = queue.find((idea) => idea.status === 'needs-run') ?? null;
  if (!next) {
    return Response.json(
      { success: false, error: 'queue empty: no ideas at needs-run' },
      { status: 200 }
    );
  }
  const { slug } = next;

  // Claim it now so a second "Run next" click picks the following idea, not this one.
  try {
    await execFileAsync(FLIP_SH, [slug, 'running', 'run-button', 'claimed by Run-next'], {
      cwd: RESEARCH_REPO_DIR,
      timeout: 15_000,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return Response.json({ success: false, slug, error: `claim failed: ${message}` }, { status: 500 });
  }

  let template: string;
  try {
    template = await readFile(TEMPLATE_PATH, 'utf8');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    await requeueAfterFailure(slug);
    return Response.json({ success: false, slug, error: message }, { status: 500 });
  }

  const host = req.headers.get('host') ?? 'localhost:3000';
  const doneUrl = `http://${host}/api/run-done/`;
  const prompt = template.replaceAll('{{IDEA_SLUG}}', slug).replaceAll('{{DONE_URL}}', doneUrl);
  const session = `lab-run-${slug}`;

  // Headless safety net: curl run-done after the agent exits so a finished (or
  // crashed) run always finalizes — run-done flips a still-"running" idea to
  // needs-recode rather than leaving the GPU queue wedged. slug is validated.
  const onExit = `curl -s -X POST '${doneUrl}' -H 'Content-Type: application/json' -d '{"slug":"${slug}"}' >/dev/null 2>&1`;

  const result = await launchCodexWithText(prompt, 'lab-run', RESEARCH_REPO_DIR, session, agent, {
    headless,
    onExit,
  });

  if (result.success) {
    return Response.json(
      { success: true, slug, session: result.session, message: `Running ${slug} on the GPU box` },
      { status: 200 }
    );
  }

  console.error('Failed to launch run-next:', result.error);
  await requeueAfterFailure(slug);
  return Response.json({ success: false, slug, session: result.session, error: result.error }, { status: 500 });
}
