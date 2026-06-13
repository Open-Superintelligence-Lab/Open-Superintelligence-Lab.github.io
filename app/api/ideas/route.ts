import { readFile, readdir } from 'fs/promises';
import { join } from 'path';
import { RESEARCH_REPO_DIR } from '@/lib/codexLauncher';

const IDEAS_DIR = join(RESEARCH_REPO_DIR, 'autoresearch', 'ideas');

type Result = {
  verdict: string; // WIN | NULL | DRIFT | FAIL | ...
  controlVal: number | null;
  treatmentVal: number | null;
  ctrl2Val: number | null;
  deltaCtrl: number | null;
  deltaCtrl2: number | null;
};

type Idea = {
  id: string;
  title: string;
  status: string;
  plain: string;
  updated: string;
  path: string;
  evidencePath: string | null;
  result: Result | null;
};

function num(md: string, label: string): number | null {
  // e.g. "- control val: 6.4037" / "- delta vs ctrl: +0.0054"
  const m = md.match(new RegExp(`${label}\\s*:?\\s*([+-]?\\d+(?:\\.\\d+)?)`, 'i'));
  return m ? Number(m[1]) : null;
}

// Pull the baseline-vs-experiment numbers out of an evidence.md so the UI can
// chart them. Returns null if the file has no recognisable val losses.
function parseEvidence(md: string): Result | null {
  const verdictMatch = md.match(/##\s*Verdict:\s*([A-Za-z]+)/i);
  const controlVal = num(md, 'control val');
  const treatmentVal = num(md, 'treatment val');
  const ctrl2Val = num(md, 'ctrl2 val');
  if (controlVal === null && treatmentVal === null) return null;
  return {
    verdict: verdictMatch ? verdictMatch[1].toUpperCase() : '',
    controlVal,
    treatmentVal,
    ctrl2Val,
    deltaCtrl: num(md, 'delta vs ctrl(?!2)') ?? num(md, 'delta vs ctrl'),
    deltaCtrl2: num(md, 'delta vs ctrl2'),
  };
}

function parseFrontmatter(md: string): Record<string, string> {
  const match = md.match(/^---\n([\s\S]*?)\n---/);
  if (!match) return {};
  const fields: Record<string, string> = {};
  for (const line of match[1].split('\n')) {
    const idx = line.indexOf(':');
    if (idx === -1) continue;
    const key = line.slice(0, idx).trim();
    const value = line.slice(idx + 1).trim();
    if (key) fields[key] = value;
  }
  return fields;
}

function parseTitle(md: string, fallback: string): string {
  const match = md.match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : fallback;
}

async function listIdeas(): Promise<Idea[]> {
  let entries: string[];
  try {
    entries = await readdir(IDEAS_DIR);
  } catch {
    return [];
  }

  const ideas: Idea[] = [];
  for (const dir of entries) {
    try {
      const md = await readFile(join(IDEAS_DIR, dir, 'idea.md'), 'utf8');
      const fm = parseFrontmatter(md);
      // Read evidence.md if present, and parse the loss numbers out of it.
      let evidenceMd: string | null = null;
      try {
        evidenceMd = await readFile(join(IDEAS_DIR, dir, 'evidence.md'), 'utf8');
      } catch {
        evidenceMd = null;
      }
      ideas.push({
        id: fm.id || dir,
        title: parseTitle(md, dir),
        status: fm.status || 'unknown',
        plain: fm.plain || '',
        updated: fm.updated || '',
        path: `autoresearch/ideas/${dir}/idea.md`,
        evidencePath: evidenceMd !== null ? `autoresearch/ideas/${dir}/evidence.md` : null,
        result: evidenceMd !== null ? parseEvidence(evidenceMd) : null,
      });
    } catch {
      // No idea.md in this folder — skip.
    }
  }

  // Newest first by updated timestamp, falling back to id.
  ideas.sort((a, b) => (b.updated || b.id).localeCompare(a.updated || a.id));
  return ideas;
}

// POST-only: this site builds with `output: 'export'`, which rejects dynamic
// GET route handlers. Returns the list of ideas on disk.
export async function POST() {
  return Response.json({ success: true, ideas: await listIdeas() }, { status: 200 });
}
