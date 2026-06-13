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

// `cmd` is the interactive invocation (agent stays in its REPL after the task —
// the user can attach and watch). `headlessCmd` runs the task to completion and
// then *exits the process* (claude `-p` print mode / `codex exec`), so the tmux
// pane closes by itself. Headless is the default; see launchCodexWithText opts.
export type AgentDef = { id: AgentId; label: string; cmd: string; headlessCmd: string };

const CODEX_MODEL = process.env.CODEX_MODEL ?? 'gpt-5.4-mini';

export const AGENTS: Record<AgentId, AgentDef> = {
  // Default. `cmf` in the user's shell — Claude Code routed to MiniMax-M3.
  // `claude-minimax-free` execs `claude --dangerously-skip-permissions "$@"`,
  // so appending `-p` gives non-interactive print mode that exits when done.
  minimax: {
    id: 'minimax',
    label: 'MiniMax (cmf)',
    cmd: 'claude-minimax-free',
    headlessCmd: 'claude-minimax-free -p',
  },
  codex: {
    id: 'codex',
    label: 'Codex',
    cmd: `codex -m ${CODEX_MODEL} --dangerously-bypass-approvals-and-sandbox`,
    headlessCmd: `codex exec -m ${CODEX_MODEL} --dangerously-bypass-approvals-and-sandbox`,
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

export type LaunchOpts = {
  // Default true. Run the agent non-interactively so the process exits on
  // completion (deterministic auto-close) instead of lingering at its REPL.
  headless?: boolean;
  // Optional shell command run *after* the agent process exits (headless only),
  // e.g. a curl to the done/finalize endpoint. Runs before the self-kill.
  onExit?: string;
};

/**
 * Fire-and-forget: launch the chosen agent in a detached tmux session via
 * launch_agent.sh with the given prompt text. Returns once the session is
 * started — it does NOT wait for the agent to finish. Pass an explicit
 * `session` name to make the tmux session identifiable (e.g. per-idea);
 * otherwise one is generated. `agent` selects the runner (defaults to MiniMax).
 *
 * In headless mode (default) we append a post-command to the typed line:
 * `<agent> "<prompt>" ; <onExit?> ; tmux kill-session`. Because the headless
 * agent exits when the task finishes, that curl + self-kill actually run — so
 * finalize no longer depends on the agent voluntarily curling as its last step.
 */
export async function launchCodexWithText(
  promptText: string,
  sessionPrefix: string,
  cwd: string = RESEARCH_REPO_DIR,
  session: string = makeSessionName(sessionPrefix),
  agent?: string,
  opts: LaunchOpts = {}
): Promise<LaunchResult> {
  const def = resolveAgent(agent);
  const headless = opts.headless ?? true;
  const cmd = headless ? def.headlessCmd : def.cmd;

  // Post-command: only meaningful headless (interactive agents hold the shell,
  // so anything appended here would never run). Run onExit, then self-kill.
  let postCmd = '';
  if (headless) {
    const parts: string[] = [];
    if (opts.onExit) parts.push(opts.onExit);
    parts.push(`tmux kill-session -t ${session} 2>/dev/null`);
    postCmd = parts.join(' ; ');
  }

  try {
    // The Next dev server sets npm_config_prefix, which makes nvm refuse to
    // load in the spawned tmux shell — and codex/claude live under nvm/local
    // bin. Strip it so the tmux shell sources nvm normally and finds the
    // runner on PATH.
    const env = { ...process.env };
    delete env.npm_config_prefix;
    delete env.NPM_CONFIG_PREFIX;

    const { stdout } = await execFileAsync(LAUNCHER, [session, cmd, promptText, postCmd], {
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
  agent?: string,
  opts: LaunchOpts = {}
): Promise<LaunchResult> {
  try {
    const prompt = await readFile(promptPath, 'utf8');
    return launchCodexWithText(prompt, sessionPrefix, cwd, makeSessionName(sessionPrefix), agent, opts);
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
