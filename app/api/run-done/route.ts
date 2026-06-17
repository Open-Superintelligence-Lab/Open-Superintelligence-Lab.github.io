import { execFile, spawn } from 'child_process';
import { promisify } from 'util';
import { readFile } from 'fs/promises';
import { join } from 'path';
import { RESEARCH_REPO_DIR } from '@/lib/codexLauncher';

const execFileAsync = promisify(execFile);
const FLIP_SH = join(RESEARCH_REPO_DIR, 'autoresearch', 'bin', 'flip.sh');
const SESSION_PREFIX = 'lab-run-';

function statusOf(md: string): string {
  const m = md.match(/^status:\s*(.+)$/m);
  return m ? m[1].trim() : '';
}

// Called by the run Codex session as its final act. A good runner already wrote
// evidence.md and flipped to done; a failed runner should flip to needs-recode.
// If it exits while still marked running, surface that as a fixable failure
// instead of leaving the GPU queue wedged forever.
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

  try {
    const md = await readFile(
      join(RESEARCH_REPO_DIR, 'autoresearch', 'ideas', slug, 'idea.md'),
      'utf8'
    );
    const status = statusOf(md);
    finalized = status;
    if (status === 'running') {
      await execFileAsync(
        FLIP_SH,
        [slug, 'needs-recode', 'run-done', 'run session exited without verdict'],
        { cwd: RESEARCH_REPO_DIR, timeout: 15_000 }
      );
      finalized = 'needs-recode';
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error('run-done finalize failed:', message);
  }

  const killer = spawn(
    'sh',
    ['-c', `sleep 2; tmux kill-session -t ${session} 2>/dev/null`],
    { detached: true, stdio: 'ignore' }
  );
  killer.unref();

  // Note: this only finalizes a manual single "Run next" (per-idea lab-run-*
  // session). Autorun no longer chains here — the lab-autorun runner agent owns
  // and drains the whole needs-run queue itself.
  return Response.json(
    { success: true, slug, status: finalized, killed: session },
    { status: 200 }
  );
}
