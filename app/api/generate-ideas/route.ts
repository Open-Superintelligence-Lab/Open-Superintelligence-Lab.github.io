import { launchCodexWithPrompt, RESEARCH_REPO_DIR } from '@/lib/codexLauncher';

const PROMPT_PATH = `${RESEARCH_REPO_DIR}/autoresearch/prompts/generate-ideas.md`;

export async function POST(req: Request) {
  let agent: string | undefined;
  try {
    ({ agent } = await req.json());
  } catch {
    agent = undefined;
  }

  const result = await launchCodexWithPrompt(PROMPT_PATH, 'lab-generate-ideas', undefined, agent);

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
