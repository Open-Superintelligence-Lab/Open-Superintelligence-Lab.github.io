import { readFile } from 'fs/promises';
import { launchCodexWithText, RESEARCH_REPO_DIR } from '@/lib/codexLauncher';

const TEMPLATE_PATH = `${RESEARCH_REPO_DIR}/autoresearch/prompts/implement-idea.md`;

export async function POST(req: Request) {
  let slug = '';
  try {
    ({ slug } = await req.json());
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

  const prompt = template.replaceAll('{{IDEA_SLUG}}', slug);
  // Deterministic, identifiable session name per idea.
  const session = `lab-implement-${slug}`;

  const result = await launchCodexWithText(prompt, 'lab-implement', RESEARCH_REPO_DIR, session);

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
