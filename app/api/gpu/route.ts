import { execFile } from 'child_process';
import { readFile } from 'fs/promises';
import { join } from 'path';
import { promisify } from 'util';
import { RESEARCH_REPO_DIR } from '@/lib/codexLauncher';

const execFileAsync = promisify(execFile);
const REMOTE_BOX_PATH = join(RESEARCH_REPO_DIR, 'autoresearch', 'remote-box.json');

// The actual GPU training runs inside a detached tmux session named `arq` on
// the remote Vast box (see autoresearch/prompts/run-idea.md). This route SSHes
// in and reports what that box is doing right now — STATUS, the live log tail,
// GPU utilisation — so the run can be followed from the browser without
// attaching a terminal at all.
const REMOTE_TMUX = 'arq';

type RemoteBox = { host?: string; port?: number; user?: string };

// One round-trip: print every section with a marker so we can split locally.
const REMOTE_SCRIPT = [
  'echo "###STATUS"',
  'cat ~/arq/STATUS 2>/dev/null',
  `echo "###TMUX"`,
  `tmux has-session -t ${REMOTE_TMUX} 2>/dev/null && echo alive || echo dead`,
  'echo "###GPU"',
  'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null',
  'echo "###LOG"',
  'L="$(ls -t ~/arq/logs/*.log 2>/dev/null | head -1)"',
  'echo "${L##*/}"',
  'echo "###TAIL"',
  'tail -n 40 "$L" 2>/dev/null',
].join('; ');

function section(out: string, name: string): string {
  const re = new RegExp(`###${name}\\n([\\s\\S]*?)(?=\\n###|$)`);
  const m = out.match(re);
  return m ? m[1].trim() : '';
}

export async function POST() {
  let box: RemoteBox = {};
  try {
    box = JSON.parse(await readFile(REMOTE_BOX_PATH, 'utf8'));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return Response.json({ success: false, error: `cannot read remote-box.json: ${message}` }, { status: 200 });
  }

  const { host, port, user } = box;
  if (!host || !port || !user) {
    return Response.json({ success: false, error: 'remote-box.json missing host/port/user' }, { status: 200 });
  }

  const sshAttach = `ssh -t -p ${port} ${user}@${host} 'tmux attach -t ${REMOTE_TMUX}'`;

  try {
    const { stdout } = await execFileAsync(
      'ssh',
      [
        '-o', 'ConnectTimeout=10',
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'BatchMode=yes',
        '-p', String(port),
        `${user}@${host}`,
        REMOTE_SCRIPT,
      ],
      { timeout: 25_000, maxBuffer: 4 * 1024 * 1024 }
    );

    return Response.json(
      {
        success: true,
        host: `${host}:${port}`,
        status: section(stdout, 'STATUS'),
        tmuxAlive: section(stdout, 'TMUX') === 'alive',
        gpu: section(stdout, 'GPU'),
        logName: section(stdout, 'LOG'),
        logTail: section(stdout, 'TAIL'),
        sshAttach,
      },
      { status: 200 }
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return Response.json({ success: false, host: `${host}:${port}`, sshAttach, error: message }, { status: 200 });
  }
}
