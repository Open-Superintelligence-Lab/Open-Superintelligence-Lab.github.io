import { execFile } from 'child_process';
import { readFile } from 'fs/promises';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

export const RESEARCH_REPO_DIR = '/Users/vukrosic/my-life/llm-research-kit-scaling';

const LAUNCHER =
  process.env.CODEX_LAUNCHER ??
  '/Users/vukrosic/.agents/skills/launch-codex-tmux/scripts/launch_codex.sh';

export function makeSessionName(prefix: string) {
  const stamp = new Date().toISOString().replace(/[-:.TZ]/g, '');
  const suffix = Math.random().toString(36).slice(2, 7);
  return `${prefix}-${stamp}-${suffix}`;
}

type LaunchResult =
  | { success: true; session: string; stdout: string }
  | { success: false; session: string; error: string };

/**
 * Fire-and-forget: launch Codex in a detached tmux session via launch_codex.sh
 * with the given prompt text. Returns once the session is started — it does NOT
 * wait for Codex to finish. Pass an explicit `session` name to make the tmux
 * session identifiable (e.g. per-idea); otherwise one is generated.
 */
export async function launchCodexWithText(
  promptText: string,
  sessionPrefix: string,
  cwd: string = RESEARCH_REPO_DIR,
  session: string = makeSessionName(sessionPrefix)
): Promise<LaunchResult> {
  try {
    // The Next dev server sets npm_config_prefix, which makes nvm refuse to
    // load in the spawned tmux shell — and `codex` lives under nvm. Strip it
    // so the tmux shell sources nvm normally and finds codex on PATH.
    const env = { ...process.env };
    delete env.npm_config_prefix;
    delete env.NPM_CONFIG_PREFIX;

    const { stdout } = await execFileAsync(LAUNCHER, [session, promptText], {
      cwd,
      env,
      maxBuffer: 10 * 1024 * 1024,
      timeout: 60_000,
    });

    return { success: true, session, stdout: stdout.trim() };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { success: false, session, error: message };
  }
}

/** Read a prompt file, then launch it via {@link launchCodexWithText}. */
export async function launchCodexWithPrompt(
  promptPath: string,
  sessionPrefix: string,
  cwd: string = RESEARCH_REPO_DIR
): Promise<LaunchResult> {
  try {
    const prompt = await readFile(promptPath, 'utf8');
    return launchCodexWithText(prompt, sessionPrefix, cwd);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { success: false, session: makeSessionName(sessionPrefix), error: message };
  }
}
