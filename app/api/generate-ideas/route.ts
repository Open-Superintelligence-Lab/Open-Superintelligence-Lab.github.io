import { launchCodexWithPrompt, RESEARCH_REPO_DIR } from '@/lib/codexLauncher';

const PROMPT_PATH = `${RESEARCH_REPO_DIR}/autoresearch/prompts/generate-ideas.md`;

export async function POST(req: Request) {
  let agent: string | undefined;
  let headless = true;
  try {
    const body = await req.json();
    agent = body.agent;
    if (typeof body.headless === 'boolean') headless = body.headless;
  } catch {
    agent = undefined;
  }

  // Generate has no finalize endpoint — headless just makes the session exit
  // (and self-close) when idea filing is done, instead of lingering at a REPL.
  const result = await launchCodexWithPrompt(PROMPT_PATH, 'lab-generate-ideas', undefined, agent, {
    headless,
  });

  if (result.success) {
    return Response.json(
      {
        success: true,
        session: result.session,
        message: `Idea generation launched in tmux session ${result.session}`,
        stdout: result.stdout,
      },
      { status: 200 }
    );
  }

  console.error('Failed to launch idea generation:', result.error);
  return Response.json(
    { success: false, session: result.session, error: result.error },
    { status: 500 }
  );
}
