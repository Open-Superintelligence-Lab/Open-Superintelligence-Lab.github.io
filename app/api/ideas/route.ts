import { readFile, readdir } from 'fs/promises';
import { join } from 'path';
import { RESEARCH_REPO_DIR } from '@/lib/codexLauncher';

const IDEAS_DIR = join(RESEARCH_REPO_DIR, 'autoresearch', 'ideas');

type Idea = {
  id: string;
  title: string;
  status: string;
  plain: string;
  updated: string;
  path: string;
};

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
      ideas.push({
        id: fm.id || dir,
        title: parseTitle(md, dir),
        status: fm.status || 'unknown',
        plain: fm.plain || '',
        updated: fm.updated || '',
        path: `autoresearch/ideas/${dir}/idea.md`,
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
