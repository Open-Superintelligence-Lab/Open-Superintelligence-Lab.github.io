import { execFile } from 'child_process';
import { writeFile, chmod, unlink, readFile } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { promisify } from 'util';
import { RESEARCH_REPO_DIR } from '@/lib/codexLauncher';

const execFileAsync = promisify(execFile);
const REMOTE_BOX_PATH = join(RESEARCH_REPO_DIR, 'autoresearch', 'remote-box.json');
const REMOTE_TMUX = 'arq';

// Open a Terminal window running `cmd` by writing a temporary `.command` file
// and `open`-ing it. macOS Terminal executes `.command` files directly, so this
// needs NO "Automation"/Apple-events permission — unlike the old osascript
// approach, which failed with `-1743 Not authorized to send Apple events`.
async function openInTerminal(cmd: string) {
  const file = join(tmpdir(), `lab-attach-${Date.now()}-${Math.random().toString(36).slice(2, 7)}.command`);
  await writeFile(file, `#!/bin/zsh\n${cmd}\n`, 'utf8');
  await chmod(file, 0o755);
  await execFileAsync('/usr/bin/open', [file], { timeout: 15_000 });
  // The shell has already read the script; clean the temp file up shortly after.
  setTimeout(() => void unlink(file).catch(() => {}), 10_000);
}

export async function POST(req: Request) {
  let body: { name?: string; remote?: boolean } = {};
  try {
    body = await req.json();
  } catch {
    body = {};
  }

  // Remote attach: SSH into the GPU box and attach its `arq` tmux directly.
  if (body.remote) {
    try {
      const box = JSON.parse(await readFile(REMOTE_BOX_PATH, 'utf8')) as {
        host?: string;
        port?: number;
        user?: string;
      };
      if (!box.host || !box.port || !box.user) {
        return Response.json({ success: false, error: 'remote-box.json missing host/port/user' }, { status: 400 });
      }
      // The training tmux (`arq`) only exists while a run is live. If it isn't
      // running, don't just attach-and-close (the window flashes shut) — show
      // the latest STATUS and drop into a login shell so the user can look
      // around. Single-quoted for ssh; inner double-quotes are fine.
      const remoteScript =
        `tmux attach -t ${REMOTE_TMUX} 2>/dev/null || ` +
        `{ echo; echo "No live GPU run — tmux \\"${REMOTE_TMUX}\\" is not running."; ` +
        `echo "(it only exists while a run is active; start one with Run next)"; echo; ` +
        `echo "Latest ~/arq/STATUS:"; tail -n 6 ~/arq/STATUS 2>/dev/null || echo "(no STATUS yet)"; ` +
        `echo; exec bash -l; }`;
      const cmd = `ssh -t -p ${box.port} ${box.user}@${box.host} '${remoteScript}'`;
      await openInTerminal(cmd);
      return Response.json({ success: true, remote: true, cmd }, { status: 200 });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error('Failed to attach to remote GPU tmux:', message);
      return Response.json({ success: false, error: message }, { status: 500 });
    }
  }

  const name = body.name ?? '';
  // tmux session names: letters, digits, dash, underscore, dot.
  if (!name || !/^[A-Za-z0-9._-]+$/.test(name)) {
    return Response.json({ success: false, error: 'invalid session name' }, { status: 400 });
  }

  try {
    await openInTerminal(`tmux attach -t ${name}`);
    return Response.json({ success: true, name }, { status: 200 });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error('Failed to attach to tmux session:', message);
    return Response.json({ success: false, name, error: message }, { status: 500 });
  }
}
