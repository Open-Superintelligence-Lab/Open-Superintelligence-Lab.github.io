import { readFile, readdir } from 'fs/promises';
import { join } from 'path';
import { RESEARCH_REPO_DIR } from '@/lib/codexLauncher';

const RESULTS_DIR = join(RESEARCH_REPO_DIR, 'remote-results');

// Only letters, digits, dot, dash, underscore — keeps path traversal impossible
// and matches the slug shape used everywhere else for idea ids.
const ID_RE = /^[A-Za-z0-9._-]+$/;

type Role = 'control' | 'treatment' | 'control2';

type Run = {
  role: Role;
  label: string;
  steps: number[];
  valLosses: number[];
};

// Pull every "Step N: Val Loss: X" line out of a training log, in file order.
// The lone "Final Val Loss: X" line at the bottom has no step number and is
// intentionally ignored — the UI plots the per-step curve, not the summary.
function parseValLossCurve(content: string): { steps: number[]; valLosses: number[] } {
  const re = /Step\s+(\d+):\s*Val Loss:\s*([0-9.]+)/g;
  const steps: number[] = [];
  const valLosses: number[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(content)) !== null) {
    steps.push(Number(m[1]));
    valLosses.push(Number(m[2]));
  }
  return { steps, valLosses };
}

// Walk remote-results/*/ for a subdir named exactly "arq-<NUM>" or starting
// with "<NUM>-". When the same NUM appears under multiple date dirs (re-runs),
// keep the one under the lexicographically largest date dir — date dirs are
// YYYY-MM-DD-... so string sort == newest first.
async function findExperimentDir(id: string): Promise<string | null> {
  const numMatch = id.match(/^(\d+)/);
  if (!numMatch) return null;
  const num = numMatch[1];
  const arqName = `arq-${num}`;
  const numPrefix = `${num}-`;

  let dateDirs: string[];
  try {
    dateDirs = await readdir(RESULTS_DIR);
  } catch {
    return null;
  }
  dateDirs.sort((a, b) => b.localeCompare(a));

  for (const dateDir of dateDirs) {
    const datePath = join(RESULTS_DIR, dateDir);
    let subs: string[];
    try {
      subs = await readdir(datePath);
    } catch {
      continue;
    }
    const hit = subs.find((s) => s === arqName || s.startsWith(numPrefix));
    if (hit) return join(datePath, hit);
  }
  return null;
}

async function readRun(
  dir: string,
  file: string | undefined,
  role: Role,
  label: string
): Promise<Run | null> {
  if (!file) return null;
  try {
    const content = await readFile(join(dir, file), 'utf8');
    const { steps, valLosses } = parseValLossCurve(content);
    return { role, label, steps, valLosses };
  } catch {
    return null;
  }
}

export async function POST(request: Request) {
  // id comes from the POST body — POST route handlers aren't prerendered under
  // `output: export`, which a GET handler would be (and would then 500). This
  // matches the other dynamic API routes (ideas/gpu/tmux), all POST.
  const body = await request.json().catch(() => ({}));
  const id = typeof body?.id === 'string' ? body.id : '';
  if (!id || !ID_RE.test(id)) {
    return Response.json({ error: 'invalid id' }, { status: 400 });
  }

  const dir = await findExperimentDir(id);
  if (!dir) {
    return Response.json({ id, runs: [] }, { status: 200 });
  }

  let files: string[];
  try {
    files = await readdir(dir);
  } catch {
    return Response.json({ id, runs: [] }, { status: 200 });
  }

  const ctrlFile = files.find((f) => f.startsWith('ctrl_') && f.endsWith('.log'));
  const ctrl2File = files.find((f) => f.startsWith('ctrl2_') && f.endsWith('.log'));
  const treatmentFile = files.find(
    (f) =>
      f.startsWith(`${id}_`) &&
      f.endsWith('.log') &&
      !f.startsWith('ctrl_') &&
      !f.startsWith('ctrl2_')
  );

  const candidates = await Promise.all([
    readRun(dir, ctrlFile, 'control', 'ctrl'),
    readRun(dir, treatmentFile, 'treatment', id),
    readRun(dir, ctrl2File, 'control2', 'ctrl2'),
  ]);
  const runs = candidates.filter((r): r is Run => r !== null);

  return Response.json({ id, runs }, { status: 200 });
}
