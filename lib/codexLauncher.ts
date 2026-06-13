import { execFile } from 'child_process';
import { readFile } from 'fs/promises';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

export const RESEARCH_REPO_DIR = '/Users/vukrosic/my-life/llm-research-kit-scaling';

// Generic agent launcher: takes <session> <agent-cmd> <prompt> and types
// `<agent-cmd> "$(cat prompt)"` into a detached tmux session. The runner is no
// longer hardcoded — pick an agent via AGENTS below.
const LAUNCHER =
  process.env.AGENT_LAUNCHER ??
  '/Users/vukrosic/.agents/skills/launch-codex-tmux/scripts/launch_agent.sh';

// ---- Agents -----------------------------------------------------------------
// Each agent is just a command prefix; the prompt is appended as a final quoted
// positional arg (both codex and claude take the prompt that way). To add an
// agent, drop another entry here — nothing else in the launch path is agent
// specific.
export type AgentId = 'minimax' | 'codex';

export type AgentDef = { id: AgentId; label: string; cmd: string };

const CODEX_MODEL = process.env.CODEX_MODEL ?? 'gpt-5.4-mini';

export const AGENTS: Record<AgentId, AgentDef> = {
  // Default. `cmf` in the user's shell — Claude Code routed to MiniMax-M3.
  minimax: { id: 'minimax', label: 'MiniMax (cmf)', cmd: 'claude-minimax-free' },
  codex: {
    id: 'codex',
    label: 'Codex',
    cmd: `codex -m ${CODEX_MODEL} --dangerously-bypass-approvals-and-sandbox`,
  },
};

export const DEFAULT_AGENT: AgentId = 'minimax';

export function resolveAgent(agent?: string): AgentDef {
  if (agent && agent in AGENTS) return AGENTS[agent as AgentId];
  return AGENTS[DEFAULT_AGENT];
}

export function makeSessionName(prefix: string) {
  const stamp = new Date().toISOString().replace(/[-:.TZ]/g, '');
  const suffix = Math.random().toString(36).slice(2, 7);
  return `${prefix}-${stamp}-${suffix}`;
}

type LaunchResult =
  | { success: true; session: string; agent: AgentId; stdout: string }
  | { success: false; session: string; agent: AgentId; error: string };

/**
 * Fire-and-forget: launch the chosen agent in a detached tmux session via
 * launch_agent.sh with the given prompt text. Returns once the session is
 * started — it does NOT wait for the agent to finish. Pass an explicit
 * `session` name to make the tmux session identifiable (e.g. per-idea);
 * otherwise one is generated. `agent` selects the runner (defaults to MiniMax).
 */
export async function launchCodexWithText(
  promptText: string,
  sessionPrefix: string,
  cwd: string = RESEARCH_REPO_DIR,
  session: string = makeSessionName(sessionPrefix),
  agent?: string
): Promise<LaunchResult> {
  const def = resolveAgent(agent);
  try {
    // The Next dev server sets npm_config_prefix, which makes nvm refuse to
    // load in the spawned tmux shell — and codex/claude live under nvm/local
    // bin. Strip it so the tmux shell sources nvm normally and finds the
    // runner on PATH.
    const env = { ...process.env };
    delete env.npm_config_prefix;
    delete env.NPM_CONFIG_PREFIX;

    const { stdout } = await execFileAsync(LAUNCHER, [session, def.cmd, promptText], {
      cwd,
      env,
      maxBuffer: 10 * 1024 * 1024,
      timeout: 60_000,
    });

    return { success: true, session, agent: def.id, stdout: stdout.trim() };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { success: false, session, agent: def.id, error: message };
  }
}

/** Read a prompt file, then launch it via {@link launchCodexWithText}. */
export async function launchCodexWithPrompt(
  promptPath: string,
  sessionPrefix: string,
  cwd: string = RESEARCH_REPO_DIR,
  agent?: string
): Promise<LaunchResult> {
  try {
    const prompt = await readFile(promptPath, 'utf8');
    return launchCodexWithText(prompt, sessionPrefix, cwd, makeSessionName(sessionPrefix), agent);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      success: false,
      session: makeSessionName(sessionPrefix),
      agent: resolveAgent(agent).id,
      error: message,
    };
  }
}
