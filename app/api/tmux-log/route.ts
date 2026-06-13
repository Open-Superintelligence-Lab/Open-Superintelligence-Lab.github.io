import { execFile } from 'child_process';
import { existsSync } from 'fs';
import { mkdir, readFile, readdir, stat, unlink, writeFile } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

const TMUX_BIN = ['/opt/homebrew/bin/tmux', '/usr/local/bin/tmux', '/usr/bin/tmux'].find(
  (p) => existsSync(p)
) ?? 'tmux';

// Where saved tmux logs live. tmux only keeps scrollback in memory and loses it
// when a session is killed, so we mirror the captured pane to disk — that way
// the UI can still show the last-known output of a finished session.
//
// NOTE: these are meant to be short-lived. We currently sweep anything older
// than LOG_TTL_MS (1 hour) on each request; revisit if we ever want longer
// retention or a dedicated cron instead of this opportunistic cleanup.
const LOG_DIR = join(tmpdir(), 'lab-tmux-logs');
const LOG_TTL_MS = 60 * 60 * 1000; // 1 hour
const SCROLLBACK_LINES = 2000;

// tmux session names: letters, digits, dash, underscore, dot.
const NAME_RE = /^[A-Za-z0-9._-]+$/;

async function sweepOldLogs() {
  try {
    const entries = await readdir(LOG_DIR);
    const now = Date.now();
    await Promise.all(
      entries.map(async (f) => {
        const p = join(LOG_DIR, f);
        try {
          const s = await stat(p);
          if (now - s.mtimeMs > LOG_TTL_MS) await unlink(p);
        } catch {
          /* ignore */
        }
      })
    );
  } catch {
    /* dir may not exist yet — nothing to sweep */
  }
}

export async function POST(req: Request) {
  let name = '';
  try {
    ({ name } = await req.json());
  } catch {
    name = '';
  }
  if (!name || !NAME_RE.test(name)) {
    return Response.json({ success: false, error: 'invalid session name' }, { status: 400 });
  }

  await sweepOldLogs();
  await mkdir(LOG_DIR, { recursive: true }).catch(() => {});
  const savedPath = join(LOG_DIR, `${name}.log`);

  // Is the session still alive? If so, capture its current scrollback and
  // persist it. If not, fall back to whatever we last saved.
  let alive = false;
  try {
    await execFileAsync(TMUX_BIN, ['has-session', '-t', name], { timeout: 8_000 });
    alive = true;
  } catch {
    alive = false;
  }

  if (alive) {
    try {
      const { stdout } = await execFileAsync(
        TMUX_BIN,
        ['capture-pane', '-t', name, '-p', '-S', `-${SCROLLBACK_LINES}`],
        { timeout: 10_000, maxBuffer: 8 * 1024 * 1024 }
      );
      // Trim trailing blank lines tmux pads the pane with.
      const text = stdout.replace(/\s+$/, '');
      await writeFile(savedPath, text, 'utf8').catch(() => {});
      return Response.json(
        { success: true, name, alive: true, location: 'local', text, capturedAt: Date.now() },
        { status: 200 }
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return Response.json({ success: false, name, alive: true, error: message }, { status: 200 });
    }
  }

  // Dead session: serve the saved copy if we have one.
  try {
    const text = await readFile(savedPath, 'utf8');
    const s = await stat(savedPath);
    return Response.json(
      { success: true, name, alive: false, location: 'local', text, savedAt: s.mtimeMs },
      { status: 200 }
    );
  } catch {
    return Response.json(
      { success: true, name, alive: false, location: 'local', text: '', note: 'no saved log' },
      { status: 200 }
    );
  }
}
