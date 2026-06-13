import { readFile, writeFile, unlink } from 'fs/promises';
import { join } from 'path';
import { RESEARCH_REPO_DIR } from './codexLauncher';

// Autorun state lives in one small file. Its presence = autorun is ON; its
// contents = which agent to launch the next queued run with (e.g. "minimax").
// Absent file = OFF. Keeping it on disk (not in memory) means the completion
// hook (run-done) and the UI agree across requests and server restarts.
const FLAG_PATH = join(RESEARCH_REPO_DIR, 'autoresearch', 'autorun.flag');

const KNOWN_AGENTS = new Set(['minimax', 'codex']);

// Returns the agent id to use for auto-launched runs, or null when autorun is
// off. Falls back to "minimax" if the file exists but holds an unknown value.
export async function getAutorunAgent(): Promise<string | null> {
  try {
    const raw = (await readFile(FLAG_PATH, 'utf8')).trim();
    if (!raw) return null;
    return KNOWN_AGENTS.has(raw) ? raw : 'minimax';
  } catch {
    return null;
  }
}

// Turn autorun on (storing the agent) or off (removing the flag file).
export async function setAutorun(agent: string | null): Promise<void> {
  if (agent && KNOWN_AGENTS.has(agent)) {
    await writeFile(FLAG_PATH, agent, 'utf8');
    return;
  }
  if (agent) {
    // unknown agent but truthy -> treat as "on" with the default runner.
    await writeFile(FLAG_PATH, 'minimax', 'utf8');
    return;
  }
  await unlink(FLAG_PATH).catch(() => {
    /* already off */
  });
}
