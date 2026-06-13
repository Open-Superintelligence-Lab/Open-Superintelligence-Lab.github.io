import { readFile, writeFile } from 'fs/promises';
import { resolve, sep } from 'path';
import { RESEARCH_REPO_DIR } from '@/lib/codexLauncher';

// All file access is confined to this root. `path` in the request body is
// relative to it. This keeps the generic editor from touching anything else.
const ROOT = RESEARCH_REPO_DIR;

function safeResolve(relPath: string): string | null {
  if (!relPath || typeof relPath !== 'string') return null;
  const full = resolve(ROOT, relPath);
  // Must stay inside ROOT and be a markdown file.
  if (full !== ROOT && !full.startsWith(ROOT + sep)) return null;
  if (!full.endsWith('.md')) return null;
  return full;
}

// POST-only: this site builds with `output: 'export'`, which rejects dynamic
// GET route handlers. action "read" (default) returns content; "save" writes it.
export async function POST(req: Request) {
  let body: { action?: string; path?: string; content?: string } = {};
  try {
    body = await req.json();
  } catch {
    body = {};
  }

  const full = safeResolve(body.path ?? '');
  if (!full) {
    return Response.json(
      { success: false, error: 'invalid path (must be a .md file inside the research repo)' },
      { status: 400 }
    );
  }

  if (body.action === 'save') {
    if (typeof body.content !== 'string') {
      return Response.json({ success: false, error: 'missing content' }, { status: 400 });
    }
    try {
      await writeFile(full, body.content, 'utf8');
      return Response.json({ success: true }, { status: 200 });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return Response.json({ success: false, error: message }, { status: 500 });
    }
  }

  try {
    const content = await readFile(full, 'utf8');
    return Response.json({ success: true, content }, { status: 200 });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return Response.json({ success: false, error: message }, { status: 500 });
  }
}
