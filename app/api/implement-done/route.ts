import { execFile, spawn } from 'child_process';
import { promisify } from 'util';
import { readFile } from 'fs/promises';
import { join } from 'path';
import { RESEARCH_REPO_DIR } from '@/lib/codexLauncher';
import { getAutoImplementAgent } from '@/lib/autoimplement';

const execFileAsync = promisify(execFile);
const FLIP_SH = join(RESEARCH_REPO_DIR, 'autoresearch', 'bin', 'flip.sh');
const SESSION_PREFIX = 'lab-implement-';

function statusOf(md: string): string {
  const m = md.match(/^status:\s*(.+)$/m);
  return m ? m[1].trim() : '';
}

// Called by the implement Codex session as its final act. It (1) makes sure the
// idea actually landed at needs-run (the GPU queue), and (2) kills the session
// so it doesn't sit attached/idle. The kill is detached + delayed so this
// response reaches the curl before the session (which is running the curl) dies.
export async function POST(req: Request) {
  let slug = '';
  try {
    ({ slug } = await req.json());
  } catch {
    /* validation below */
  }

  if (!slug || !/^[a-z0-9][a-z0-9-]*$/i.test(slug)) {
    return Response.json({ success: false, error: 'invalid idea slug' }, { status: 400 });
  }

  const session = SESSION_PREFIX + slug;
  let finalized = '';

  // Defensive finalize: if the agent left it mid-flight at "implementing",
  // push it to needs-run. If it intentionally set needs-run or needs-review,
  // leave that verdict alone.
  try {
    const md = await readFile(
      join(RESEARCH_REPO_DIR, 'autoresearch', 'ideas', slug, 'idea.md'),
      'utf8'
    );
    const status = statusOf(md);
    finalized = status;
    if (status === 'implementing') {
      await execFileAsync(
        FLIP_SH,
        [slug, 'needs-run', 'done-button', 'auto-finalized on implement-done'],
        { cwd: RESEARCH_REPO_DIR, timeout: 15_000 }
      );
      finalized = 'needs-run';
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error('implement-done finalize failed:', message);
  }

  // Detached, delayed self-kill so the curl inside the session gets its reply.
  const killer = spawn(
    'sh',
    ['-c', `sleep 2; tmux kill-session -t ${session} 2>/dev/null`],
    { detached: true, stdio: 'ignore' }
  );
  killer.unref();

  // Auto-implement chain: this implement just finished, so a parallel slot is
  // freeing up. If auto-implement is on, nudge the tick to launch the next
  // Proposed idea — keeps the pipeline flowing even with no browser open.
  // Fire-and-forget; the kill above lands first because the tick is async.
  let autoImplement = false;
  try {
    if (await getAutoImplementAgent()) {
      autoImplement = true;
      const host = req.headers.get('host') ?? 'localhost:3001';
      fetch(`http://${host}/api/auto-implement/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      }).catch(() => {
        /* the browser poll tick will catch up */
      });
    }
  } catch {
    /* treat as off */
  }

  return Response.json(
    { success: true, slug, status: finalized, killed: session, autoImplement },
    { status: 200 }
  );
}
