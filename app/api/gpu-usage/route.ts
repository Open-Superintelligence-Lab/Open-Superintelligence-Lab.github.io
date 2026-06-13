import { execFile } from 'child_process';
import { readFile } from 'fs/promises';
import { join } from 'path';
import { promisify } from 'util';
import { RESEARCH_REPO_DIR } from '@/lib/codexLauncher';

const execFileAsync = promisify(execFile);
const REMOTE_BOX_PATH = join(RESEARCH_REPO_DIR, 'autoresearch', 'remote-box.json');

// Deliberately tiny sibling of /api/gpu: this route asks the box for ONLY the
// live GPU usage numbers (util %, VRAM used/total) so the UI can poll it on a
// short interval without the cost of pulling STATUS + log tails every tick.
//
// SSH connection multiplexing (ControlMaster/ControlPersist) keeps one TCP
// connection warm between polls, so consistent polling doesn't pay a fresh SSH
// handshake every time — cheap on both ends.
type RemoteBox = { host?: string; port?: number; user?: string };

// Also report whether the training tmux (`arq`) is alive — it only exists while
// a run is active, so the UI needs this to know if "Attach GPU" has anything to
// attach to. Markers keep the one round-trip cheap.
const QUERY = [
  'echo "###GPU"',
  'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null',
  'echo "###ARQ"',
  'tmux has-session -t arq 2>/dev/null && echo alive || echo dead',
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

  try {
    const { stdout } = await execFileAsync(
      'ssh',
      [
        '-o', 'ConnectTimeout=5',
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'BatchMode=yes',
        // Reuse a warm connection across polls.
        '-o', 'ControlMaster=auto',
        '-o', `ControlPath=/tmp/lab-gpu-ctl-${user}-${host}-${port}`,
        '-o', 'ControlPersist=30',
        '-p', String(port),
        `${user}@${host}`,
        QUERY,
      ],
      { timeout: 8_000, maxBuffer: 64 * 1024 }
    );

    const arqAlive = section(stdout, 'ARQ') === 'alive';

    // "NVIDIA GeForce RTX 3060, 0, 1234, 12288"
    const line = (section(stdout, 'GPU').split('\n')[0] ?? '').trim();
    const parts = line.split(',').map((s) => s.trim());
    if (parts.length < 4) {
      return Response.json({ success: false, arqAlive, error: 'no GPU reading', raw: line }, { status: 200 });
    }
    const [name, util, memUsed, memTotal] = parts;

    return Response.json(
      {
        success: true,
        name,
        utilization: Number(util),
        memUsed: Number(memUsed),
        memTotal: Number(memTotal),
        arqAlive,
      },
      { status: 200 }
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return Response.json({ success: false, error: message }, { status: 200 });
  }
}
