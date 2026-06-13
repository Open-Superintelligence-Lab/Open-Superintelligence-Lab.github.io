import { readFile } from 'fs/promises';
import { launchCodexWithText, RESEARCH_REPO_DIR } from '@/lib/codexLauncher';

const PROMPT_PATH = `${RESEARCH_REPO_DIR}/autoresearch/prompts/generate-ideas.md`;

// Swap the count in the prompt's "Generate exactly N new ideas this pass." line
// with the number the user picked in the UI. If the line was edited away, append
// an unambiguous instruction so the count is always honoured.
function injectCount(prompt: string, count: number): string {
  const line = `Generate exactly ${count} new idea${count === 1 ? '' : 's'} this pass.`;
  const re = /Generate exactly \d+ new ideas? this pass\./;
  if (re.test(prompt)) return prompt.replace(re, line);
  return `${prompt.trimEnd()}\n\n**${line}**\n`;
}

export async function POST(req: Request) {
  let agent: string | undefined;
  let headless = true;
  let count = 3;
  try {
    const body = await req.json();
    agent = body.agent;
    if (typeof body.headless === 'boolean') headless = body.headless;
    if (Number.isFinite(body.count)) {
      // clamp to a sane batch so a stray value can't ask for hundreds.
      count = Math.max(1, Math.min(20, Math.round(body.count)));
    }
  } catch {
    agent = undefined;
  }

  let prompt: string;
  try {
    prompt = injectCount(await readFile(PROMPT_PATH, 'utf8'), count);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return Response.json({ success: false, error: message }, { status: 500 });
  }

  // Generate has no finalize endpoint — headless just makes the session exit
  // (and self-close) when idea filing is done, instead of lingering at a REPL.
  const result = await launchCodexWithText(prompt, 'lab-generate-ideas', undefined, undefined, agent, {
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
