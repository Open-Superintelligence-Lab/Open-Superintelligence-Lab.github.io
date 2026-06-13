"use client";

import { useCallback, useEffect, useState } from "react";
import { MarkdownPanel } from "@/components/markdown-panel";

type Session = {
  name: string;
  created: number;
  windows: number;
};

type Idea = {
  id: string;
  title: string;
  status: string;
  plain: string;
  updated: string;
  path: string;
};

const IDEAS_PROMPT_PATH = "autoresearch/prompts/generate-ideas.md";

export default function LaunchCodexPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generateMessage, setGenerateMessage] = useState("");
  const [sessions, setSessions] = useState<Session[]>([]);
  const [killing, setKilling] = useState<string | null>(null);
  const [ideas, setIdeas] = useState<Idea[]>([]);
  const [openFile, setOpenFile] = useState<{ path: string; title: string } | null>(
    null
  );
  const [implementing, setImplementing] = useState<string | null>(null);
  const [attaching, setAttaching] = useState<string | null>(null);
  const [ideaActionMsg, setIdeaActionMsg] = useState("");
  const [sessionMsg, setSessionMsg] = useState("");

  const refreshSessions = useCallback(async () => {
    try {
      const response = await fetch("/api/tmux", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "list" }),
      });
      const data = await response.json().catch(() => ({}));
      setSessions(Array.isArray(data.sessions) ? data.sessions : []);
    } catch {
      setSessions([]);
    }
  }, []);

  const refreshIdeas = useCallback(async () => {
    try {
      const response = await fetch("/api/ideas", { method: "POST" });
      const data = await response.json().catch(() => ({}));
      setIdeas(Array.isArray(data.ideas) ? data.ideas : []);
    } catch {
      setIdeas([]);
    }
  }, []);

  useEffect(() => {
    refreshSessions();
    refreshIdeas();
    const interval = setInterval(() => {
      refreshSessions();
      refreshIdeas();
    }, 5000);
    return () => clearInterval(interval);
  }, [refreshSessions, refreshIdeas]);

  const handleLaunch = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setMessage("");

    try {
      const response = await fetch("/api/launch-codex", { method: "POST" });
      const data = await response.json().catch(() => ({}));

      if (response.ok && data.success) {
        setMessage(`✓ Codex launched in tmux session ${data.session}`);
      } else {
        setMessage(`✗ Failed to launch Codex: ${data.error ?? "unknown error"}`);
      }
    } catch (error) {
      setMessage("✗ Error launching Codex");
    } finally {
      setIsLoading(false);
      refreshSessions();
    }
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsGenerating(true);
    setGenerateMessage("");

    try {
      const response = await fetch("/api/generate-ideas", { method: "POST" });
      const data = await response.json().catch(() => ({}));

      if (response.ok && data.success) {
        setGenerateMessage(`✓ Generating ideas in tmux session ${data.session}`);
      } else {
        setGenerateMessage(
          `✗ Failed to generate ideas: ${data.error ?? "unknown error"}`
        );
      }
    } catch (error) {
      setGenerateMessage("✗ Error generating ideas");
    } finally {
      setIsGenerating(false);
      refreshSessions();
    }
  };

  const handleKill = async (name: string) => {
    setKilling(name);
    try {
      const response = await fetch("/api/tmux", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "kill", name }),
      });
      const data = await response.json().catch(() => ({}));
      if (response.ok && data.success && Array.isArray(data.sessions)) {
        setSessions(data.sessions);
      } else {
        await refreshSessions();
      }
    } catch {
      await refreshSessions();
    } finally {
      setKilling(null);
    }
  };

  const handleImplement = async (slug: string) => {
    setImplementing(slug);
    setIdeaActionMsg("");
    try {
      const response = await fetch("/api/implement-idea", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slug }),
      });
      const data = await response.json().catch(() => ({}));
      setIdeaActionMsg(
        response.ok && data.success
          ? `✓ Implementing ${slug} (session ${data.session})`
          : `✗ Failed to implement ${slug}: ${data.error ?? "unknown error"}`
      );
    } catch {
      setIdeaActionMsg("✗ Error launching implementation");
    } finally {
      setImplementing(null);
      refreshSessions();
      refreshIdeas();
    }
  };

  const handleAttach = async (name: string) => {
    setAttaching(name);
    setSessionMsg("");
    try {
      const response = await fetch("/api/attach", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const data = await response.json().catch(() => ({}));
      if (!(response.ok && data.success)) {
        setSessionMsg(`✗ Attach failed: ${data.error ?? "unknown error"}`);
      }
    } catch {
      setSessionMsg("✗ Error attaching");
    } finally {
      setAttaching(null);
    }
  };

  return (
    <main className="min-h-screen bg-[#1f1e1d] pt-28 text-[#faf9f6] md:pt-36">
      <div className="container mx-auto flex min-h-[calc(100vh-12rem)] flex-col items-center px-6 py-24">
        <div className="flex flex-col items-center gap-6 text-center">
          <form onSubmit={handleLaunch}>
            <button
              type="submit"
              disabled={isLoading}
              className="rounded-full border border-cyan-300/30 bg-cyan-300/10 px-8 py-4 text-sm font-semibold uppercase tracking-[0.24em] text-cyan-200 transition hover:border-cyan-300/60 hover:bg-cyan-300/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-cyan-300/40 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? "Launching..." : "Launch Codex"}
            </button>
          </form>
          {message && <p className="text-sm text-cyan-300">{message}</p>}

          <form onSubmit={handleGenerate}>
            <button
              type="submit"
              disabled={isGenerating}
              className="rounded-full border border-amber-300/30 bg-amber-300/10 px-8 py-4 text-sm font-semibold uppercase tracking-[0.24em] text-amber-200 transition hover:border-amber-300/60 hover:bg-amber-300/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-amber-300/40 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isGenerating ? "Generating..." : "Generate Ideas"}
            </button>
          </form>
          {generateMessage && (
            <p className="text-sm text-amber-300">{generateMessage}</p>
          )}

          <button
            type="button"
            onClick={() =>
              setOpenFile({ path: IDEAS_PROMPT_PATH, title: "generate-ideas.md" })
            }
            className="text-xs uppercase tracking-[0.2em] text-amber-300/70 underline-offset-4 transition hover:text-amber-200 hover:underline"
          >
            Edit ideas prompt
          </button>
        </div>

        {/* Ideas list */}
        <div className="mt-16 w-full max-w-2xl">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-xs font-semibold uppercase tracking-[0.24em] text-[#faf9f6]/50">
              ideas
            </h2>
            <button
              type="button"
              onClick={refreshIdeas}
              className="text-xs uppercase tracking-[0.2em] text-amber-300/70 transition hover:text-amber-200"
            >
              Refresh
            </button>
          </div>

          {ideaActionMsg && (
            <p className="mb-2 text-xs text-amber-300">{ideaActionMsg}</p>
          )}

          {ideas.length === 0 ? (
            <p className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-6 text-center text-sm text-[#faf9f6]/40">
              No ideas yet.
            </p>
          ) : (
            <ul className="space-y-2">
              {ideas.map((idea) => (
                <li
                  key={idea.id}
                  className="flex items-start justify-between gap-3 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3"
                >
                  <button
                    type="button"
                    onClick={() =>
                      setOpenFile({ path: idea.path, title: idea.title })
                    }
                    className="min-w-0 flex-1 text-left transition hover:opacity-80 focus:outline-none"
                  >
                    <p className="truncate text-sm font-semibold text-[#faf9f6]">
                      {idea.title}
                    </p>
                    {idea.plain && (
                      <p className="mt-1 text-xs text-[#faf9f6]/55">{idea.plain}</p>
                    )}
                  </button>
                  <div className="flex shrink-0 flex-col items-end gap-2">
                    <span className="rounded-full border border-amber-300/20 bg-amber-300/5 px-2.5 py-0.5 text-[10px] uppercase tracking-[0.15em] text-amber-200/80">
                      {idea.status}
                    </span>
                    <button
                      type="button"
                      onClick={() => handleImplement(idea.id)}
                      disabled={implementing === idea.id}
                      className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] text-emerald-300 transition hover:border-emerald-400/60 hover:bg-emerald-400/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-emerald-400/40 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {implementing === idea.id ? "Launching…" : "Implement"}
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* tmux sessions list */}
        <div className="mt-12 w-full max-w-2xl">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-xs font-semibold uppercase tracking-[0.24em] text-[#faf9f6]/50">
              tmux sessions
            </h2>
            <button
              type="button"
              onClick={refreshSessions}
              className="text-xs uppercase tracking-[0.2em] text-cyan-300/70 transition hover:text-cyan-200"
            >
              Refresh
            </button>
          </div>

          {sessionMsg && (
            <p className="mb-2 text-xs text-red-300">{sessionMsg}</p>
          )}

          {sessions.length === 0 ? (
            <p className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-6 text-center text-sm text-[#faf9f6]/40">
              No active tmux sessions.
            </p>
          ) : (
            <ul className="space-y-2">
              {sessions.map((session) => (
                <li
                  key={session.name}
                  className="flex items-center justify-between gap-4 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3"
                >
                  <div className="min-w-0">
                    <p className="truncate font-mono text-sm text-[#faf9f6]">
                      {session.name}
                    </p>
                    <p className="text-xs text-[#faf9f6]/40">
                      {session.windows} window{session.windows === 1 ? "" : "s"} ·
                      started {new Date(session.created).toLocaleTimeString()}
                    </p>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <button
                      type="button"
                      onClick={() => handleAttach(session.name)}
                      disabled={attaching === session.name}
                      className="rounded-full border border-cyan-300/30 bg-cyan-300/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-cyan-200 transition hover:border-cyan-300/60 hover:bg-cyan-300/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-cyan-300/40 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {attaching === session.name ? "…" : "Attach"}
                    </button>
                    <button
                      type="button"
                      onClick={() => handleKill(session.name)}
                      disabled={killing === session.name}
                      className="rounded-full border border-red-400/30 bg-red-400/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-red-300 transition hover:border-red-400/60 hover:bg-red-400/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-red-400/40 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {killing === session.name ? "Killing…" : "Kill"}
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <MarkdownPanel
        path={openFile?.path ?? null}
        title={openFile?.title ?? ""}
        onClose={() => setOpenFile(null)}
      />
    </main>
  );
}
