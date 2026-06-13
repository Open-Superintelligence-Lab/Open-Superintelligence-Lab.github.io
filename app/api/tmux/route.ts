import { execFile } from 'child_process';
import { existsSync } from 'fs';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

const TMUX_BIN = ['/opt/homebrew/bin/tmux', '/usr/local/bin/tmux', '/usr/bin/tmux'].find(
  (p) => existsSync(p)
) ?? 'tmux';

type Session = {
  name: string;
  created: number;
  windows: number;
};

async function listSessions(): Promise<Session[]> {
  try {
    const { stdout } = await execFileAsync(
      TMUX_BIN,
      ['list-sessions', '-F', '#{session_name}|#{session_created}|#{session_windows}'],
      { timeout: 10_000 }
    );
    return stdout
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const [name, created, windows] = line.split('|');
        return {
          name,
          created: Number(created) * 1000,
          windows: Number(windows),
        };
      });
  } catch {
    // No tmux server running -> no sessions.
    return [];
  }
}

// POST-only: this site is built with `output: 'export'`, which rejects
// dynamic GET route handlers. action: "list" (default) returns the sessions;
// action: "kill" + name kills a session and returns the updated list.
export async function POST(req: Request) {
  let body: { action?: string; name?: string } = {};
  try {
    body = await req.json();
  } catch {
    body = {};
  }

  if (body.action === 'kill') {
    if (!body.name) {
      return Response.json({ success: false, error: 'missing session name' }, { status: 400 });
    }
    try {
      await execFileAsync(TMUX_BIN, ['kill-session', '-t', body.name], { timeout: 10_000 });
      return Response.json(
        { success: true, name: body.name, sessions: await listSessions() },
        { status: 200 }
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return Response.json({ success: false, name: body.name, error: message }, { status: 500 });
    }
  }

  return Response.json({ success: true, sessions: await listSessions() }, { status: 200 });
}
