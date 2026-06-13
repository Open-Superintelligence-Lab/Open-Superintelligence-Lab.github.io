import { readFile } from 'fs/promises';
import { launchCodexWithText, RESEARCH_REPO_DIR } from '@/lib/codexLauncher';

const TEMPLATE_PATH = `${RESEARCH_REPO_DIR}/autoresearch/prompts/implement-idea.md`;

export async function POST(req: Request) {
  let slug = '';
  let agent: string | undefined;
  let headless = true;
  try {
    const body = await req.json();
    slug = body.slug;
    agent = body.agent;
    if (typeof body.headless === 'boolean') headless = body.headless;
  } catch {
    slug = '';
  }

  // Idea slugs are folder names like "107-exclusive-self-attn".
  if (!slug || !/^[a-z0-9][a-z0-9-]*$/i.test(slug)) {
    return Response.json({ success: false, error: 'invalid idea slug' }, { status: 400 });
  }

  let template: string;
  try {
    template = await readFile(TEMPLATE_PATH, 'utf8');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return Response.json({ success: false, error: message }, { status: 500 });
  }

  // The finalize endpoint the implement session curls when it's done. Built
  // from the request host so the port is always right (dev server picks it).
  const host = req.headers.get('host') ?? 'localhost:3000';
  const doneUrl = `http://${host}/api/implement-done/`;

  const prompt = template
    .replaceAll('{{IDEA_SLUG}}', slug)
    .replaceAll('{{DONE_URL}}', doneUrl);
  // Deterministic, identifiable session name per idea.
  const session = `lab-implement-${slug}`;

  // Headless safety net: run the finalize curl after the agent exits, so the
  // idea lands at needs-run even if the agent forgot to ping. Single-quoted to
  // survive the launcher's double-quoted send-keys line; slug is validated.
  const onExit = `curl -s -X POST '${doneUrl}' -H 'Content-Type: application/json' -d '{"slug":"${slug}"}' >/dev/null 2>&1`;

  const result = await launchCodexWithText(prompt, 'lab-implement', RESEARCH_REPO_DIR, session, agent, {
    headless,
    onExit,
  });

  if (result.success) {
    return Response.json(
      {
        success: true,
        session: result.session,
        message: `Implementing ${slug} in tmux session ${result.session}`,
        stdout: result.stdout,
      },
      { status: 200 }
    );
  }

  console.error('Failed to launch implement-idea:', result.error);
  return Response.json(
    { success: false, session: result.session, error: result.error },
    { status: 500 }
  );
}
