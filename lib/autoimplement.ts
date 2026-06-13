import { readFile, writeFile } from 'fs/promises';
import { join } from 'path';
import { RESEARCH_REPO_DIR } from './codexLauncher';

// Auto-implement state. UNLIKE autorun, this defaults ON: a missing flag file
// means "on" (the user wanted it on by default). The file only ever exists to
// record an explicit "off", or to pin a non-default agent. Stored on disk so the
// completion hook (implement-done) and the UI agree across requests/restarts.
const FLAG_PATH = join(RESEARCH_REPO_DIR, 'autoresearch', 'autoimplement.flag');

const KNOWN_AGENTS = new Set(['minimax', 'codex']);

// At most this many implement agents run in parallel. Caps MiniMax load so a
// fresh batch of 10 proposed ideas doesn't spawn 10 agents at once — the rest
// are picked up as slots free (poll tick + implement-done chain).
export const MAX_PARALLEL_IMPLEMENTS = 2;

// Returns the agent to auto-implement with, or null when auto-implement is off.
// Absent file => default agent (ON). File holding "off" => null. Otherwise the
// stored agent (falling back to the default for an unknown value).
export async function getAutoImplementAgent(): Promise<string | null> {
  try {
    const raw = (await readFile(FLAG_PATH, 'utf8')).trim();
    if (raw === 'off') return null;
    if (!raw) return 'minimax';
    return KNOWN_AGENTS.has(raw) ? raw : 'minimax';
  } catch {
    // No file yet => default ON.
    return 'minimax';
  }
}

// Turn auto-implement on (storing the agent) or off (writing the "off" marker).
export async function setAutoImplement(agent: string | null): Promise<void> {
  if (!agent) {
    await writeFile(FLAG_PATH, 'off', 'utf8');
    return;
  }
  await writeFile(FLAG_PATH, KNOWN_AGENTS.has(agent) ? agent : 'minimax', 'utf8');
}
