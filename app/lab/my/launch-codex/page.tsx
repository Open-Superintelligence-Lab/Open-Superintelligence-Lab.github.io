"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { MarkdownPanel } from "@/components/markdown-panel";

type Session = {
  name: string;
  created: number;
  windows: number;
};

type Result = {
  verdict: string;
  controlVal: number | null;
  treatmentVal: number | null;
  ctrl2Val: number | null;
  deltaCtrl: number | null;
  deltaCtrl2: number | null;
};

type Idea = {
  id: string;
  title: string;
  status: string;
  plain: string;
  updated: string;
  path: string;
  evidencePath: string | null;
  result: Result | null;
};

type GpuInfo = {
  host: string;
  status: string;
  tmuxAlive: boolean;
  gpu: string;
  logName: string;
  logTail: string;
  sshAttach: string;
};

const IDEAS_PROMPT_PATH = "autoresearch/prompts/generate-ideas.md";
const IMPLEMENT_PROMPT_PATH = "autoresearch/prompts/implement-idea.md";
const RUN_PROMPT_PATH = "autoresearch/prompts/run-idea.md";
const REMOTE_BOX_PATH = "autoresearch/remote-box.json";
const IMPLEMENT_SESSION_PREFIX = "lab-implement-";
const RUN_SESSION_PREFIX = "lab-run-";

type GpuUsage = {
  name: string;
  utilization: number;
  memUsed: number;
  memTotal: number;
};

const GENERATE_SESSION_PREFIX = "lab-generate";

// Keep in sync with AGENTS in lib/codexLauncher.ts. minimax is the default.
const AGENT_OPTIONS: { id: string; label: string }[] = [
  { id: "minimax", label: "MiniMax (cmf)" },
  { id: "codex", label: "Codex" },
];

// "3s" / "2m 5s" — compact relative age for freshness labels.
function formatAgo(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${s % 60}s`;
}

export default function LaunchCodexPage() {
  const [agent, setAgent] = useState<string>("minimax");
  // Headless = run the agent non-interactively so it exits (and the tmux pane
  // self-closes) when the task finishes. On by default; uncheck to keep the
  // agent open at its REPL so you can attach and watch/intervene.
  const [headless, setHeadless] = useState<boolean>(true);
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
  const [isRunningNext, setIsRunningNext] = useState(false);
  const [runMessage, setRunMessage] = useState("");
  const [ideaLoadError, setIdeaLoadError] = useState("");
  const [sessionLoadError, setSessionLoadError] = useState("");
  const [gpuInfo, setGpuInfo] = useState<GpuInfo | null>(null);
  const [gpuError, setGpuError] = useState("");
  const [gpuLoading, setGpuLoading] = useState(false);
  const [gpuUsage, setGpuUsage] = useState<GpuUsage | null>(null);
  const [gpuUsageStale, setGpuUsageStale] = useState(false);
  // When the last GPU-usage reading landed, and how long that SSH round-trip
  // took — so the UI can show how far behind the compute/VRAM numbers are.
  const [gpuUsageAt, setGpuUsageAt] = useState<number | null>(null);
  const [gpuUsageLatencyMs, setGpuUsageLatencyMs] = useState<number | null>(null);
  // Whether the remote training tmux (`arq`) is alive right now — only true
  // while a run is active. Drives the "Attach GPU" button.
  const [arqAlive, setArqAlive] = useState(false);
  // Guards against overlapping usage polls when a request is slow / box is down.
  const usageInFlight = useRef(false);
  // A 1s ticker so "updated Ns ago" labels stay live between polls.
  const [now, setNow] = useState(() => Date.now());

  // Per-session expandable logs (tmux capture-pane, mirrored to disk).
  const [expandedLogs, setExpandedLogs] = useState<Set<string>>(new Set());
  const [logData, setLogData] = useState<
    Record<string, { text: string; alive: boolean; at: number }>
  >({});
  const logInFlight = useRef<Set<string>>(new Set());

  const refreshSessions = useCallback(async () => {
    try {
      const response = await fetch("/api/tmux/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "list" }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !Array.isArray(data.sessions)) {
        setSessionLoadError(data.error ?? "Failed to refresh tmux sessions");
        return;
      }
      setSessionLoadError("");
      setSessions(data.sessions);
    } catch {
      setSessionLoadError("Failed to refresh tmux sessions");
    }
  }, []);

  const refreshIdeas = useCallback(async () => {
    try {
      const response = await fetch("/api/ideas/", { method: "POST" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !Array.isArray(data.ideas)) {
        setIdeaLoadError(data.error ?? "Failed to refresh ideas");
        return;
      }
      setIdeaLoadError("");
      setIdeas(data.ideas);
    } catch {
      setIdeaLoadError("Failed to refresh ideas");
    }
  }, []);

  const refreshGpu = useCallback(async () => {
    setGpuLoading(true);
    try {
      const response = await fetch("/api/gpu/", { method: "POST" });
      const data = await response.json().catch(() => ({}));
      if (data.success) {
        setGpuError("");
        setGpuInfo({
          host: data.host ?? "",
          status: data.status ?? "",
          tmuxAlive: Boolean(data.tmuxAlive),
          gpu: data.gpu ?? "",
          logName: data.logName ?? "",
          logTail: data.logTail ?? "",
          sshAttach: data.sshAttach ?? "",
        });
      } else {
        setGpuError(data.error ?? "Failed to reach GPU box");
      }
    } catch {
      setGpuError("Failed to reach GPU box");
    } finally {
      setGpuLoading(false);
    }
  }, []);

  // Lightweight, always-on GPU usage poll (util % + VRAM). Skips when a poll is
  // already in flight so a slow/unreachable box can't stack requests.
  const refreshGpuUsage = useCallback(async () => {
    if (usageInFlight.current) return;
    usageInFlight.current = true;
    const startedAt = Date.now();
    try {
      const response = await fetch("/api/gpu-usage/", { method: "POST" });
      const data = await response.json().catch(() => ({}));
      setGpuUsageLatencyMs(Date.now() - startedAt);
      if (data.success) {
        setGpuUsage({
          name: data.name ?? "",
          utilization: Number(data.utilization) || 0,
          memUsed: Number(data.memUsed) || 0,
          memTotal: Number(data.memTotal) || 0,
        });
        setGpuUsageStale(false);
        setGpuUsageAt(Date.now());
      } else {
        setGpuUsageStale(true);
      }
      // arqAlive is reported even when there's no GPU reading.
      setArqAlive(Boolean(data.arqAlive));
    } catch {
      setGpuUsageStale(true);
    } finally {
      usageInFlight.current = false;
    }
  }, []);

  // Fetch (and persist) one session's tmux log. Mirrors capture-pane to disk so
  // it survives the session ending. Skips if a fetch for this name is in flight.
  const fetchLog = useCallback(async (name: string) => {
    if (logInFlight.current.has(name)) return;
    logInFlight.current.add(name);
    try {
      const response = await fetch("/api/tmux-log/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const data = await response.json().catch(() => ({}));
      if (data.success) {
        setLogData((prev) => ({
          ...prev,
          [name]: { text: data.text ?? "", alive: Boolean(data.alive), at: Date.now() },
        }));
      }
    } catch {
      /* leave the last-known log in place */
    } finally {
      logInFlight.current.delete(name);
    }
  }, []);

  const toggleLog = useCallback(
    (name: string) => {
      setExpandedLogs((prev) => {
        const next = new Set(prev);
        if (next.has(name)) {
          next.delete(name);
        } else {
          next.add(name);
          fetchLog(name); // fetch immediately on expand
        }
        return next;
      });
    },
    [fetchLog]
  );

  useEffect(() => {
    refreshSessions();
    refreshIdeas();
    const interval = setInterval(() => {
      refreshSessions();
      refreshIdeas();
    }, 5000);
    return () => clearInterval(interval);
  }, [refreshSessions, refreshIdeas]);

  // Poll GPU usage every 4s, but pause while the tab is hidden so we don't SSH
  // the box in the background for a page nobody is looking at.
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;
    const start = () => {
      if (interval) return;
      refreshGpuUsage();
      interval = setInterval(refreshGpuUsage, 4000);
    };
    const stop = () => {
      if (interval) clearInterval(interval);
      interval = null;
    };
    const onVisibility = () => (document.hidden ? stop() : start());
    if (!document.hidden) start();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [refreshGpuUsage]);

  // 1s ticker so "updated Ns ago" labels count up between polls. Pauses with
  // the tab hidden. Cheap: just bumps a timestamp.
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;
    const start = () => {
      if (!interval) interval = setInterval(() => setNow(Date.now()), 1000);
    };
    const stop = () => {
      if (interval) clearInterval(interval);
      interval = null;
    };
    const onVisibility = () => (document.hidden ? stop() : start());
    if (!document.hidden) start();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  // Refresh every expanded session log every 3s while the tab is visible.
  useEffect(() => {
    if (expandedLogs.size === 0) return;
    const tick = () => {
      if (document.hidden) return;
      expandedLogs.forEach((name) => fetchLog(name));
    };
    const interval = setInterval(tick, 3000);
    return () => clearInterval(interval);
  }, [expandedLogs, fetchLog]);

  // While an idea is marked `running`, poll the GPU box so the panel stays live.
  // When nothing is running we don't SSH on a timer — refresh on demand instead.
  const hasRunningIdea = ideas.some((idea) => idea.status === "running");
  useEffect(() => {
    if (!hasRunningIdea) return;
    refreshGpu();
    const interval = setInterval(refreshGpu, 10000);
    return () => clearInterval(interval);
  }, [hasRunningIdea, refreshGpu]);

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsGenerating(true);
    setGenerateMessage("");

    try {
      const response = await fetch("/api/generate-ideas/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent, headless }),
      });
      const data = await response.json().catch(() => ({}));

      if (response.ok && data.success) {
        setGenerateMessage(`✓ Generating ideas in tmux session ${data.session}`);
      } else {
        setGenerateMessage(
          `✗ Failed to generate ideas: ${data.error ?? "unknown error"}`
        );
      }
    } catch {
      setGenerateMessage("✗ Error generating ideas");
    } finally {
      setIsGenerating(false);
      refreshSessions();
    }
  };

  const handleKill = async (name: string) => {
    setKilling(name);
    try {
      const response = await fetch("/api/tmux/", {
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
      const response = await fetch("/api/implement-idea/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slug, agent }),
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

  const handleRunNext = async () => {
    setIsRunningNext(true);
    setRunMessage("");
    try {
      const response = await fetch("/api/run-next/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent }),
      });
      const data = await response.json().catch(() => ({}));
      setRunMessage(
        response.ok && data.success
          ? `Running ${data.slug} (session ${data.session})`
          : `Run next blocked: ${data.error ?? "unknown error"}`
      );
    } catch {
      setRunMessage("Error launching GPU run");
    } finally {
      setIsRunningNext(false);
      refreshSessions();
      refreshIdeas();
    }
  };

  const handleReset = async (
    slug: string,
    status = "needs-taste",
    note = "reset stuck idea from UI"
  ) => {
    setImplementing(slug);
    setIdeaActionMsg("");
    try {
      const response = await fetch("/api/flip/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          slug,
          status,
          note,
        }),
      });
      const data = await response.json().catch(() => ({}));
      setIdeaActionMsg(
        response.ok && data.success
          ? `Reset ${slug} -> ${status}`
          : `✗ Failed to reset ${slug}: ${data.error ?? "unknown error"}`
      );
    } catch {
      setIdeaActionMsg("✗ Error resetting idea");
    } finally {
      setImplementing(null);
      refreshIdeas();
    }
  };

  const handleAttach = async (name: string) => {
    setAttaching(name);
    setSessionMsg("");
    try {
      const response = await fetch("/api/attach/", {
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

  // Open a Terminal SSH'd straight into the remote GPU tmux (`arq`).
  const handleAttachGpu = async () => {
    setAttaching("__gpu__");
    setGpuError("");
    try {
      const response = await fetch("/api/attach/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ remote: true }),
      });
      const data = await response.json().catch(() => ({}));
      if (!(response.ok && data.success)) {
        setGpuError(`Attach GPU failed: ${data.error ?? "unknown error"}`);
      }
    } catch {
      setGpuError("Error attaching to GPU tmux");
    } finally {
      setAttaching(null);
    }
  };

  // Join: which ideas have a live implement/run session right now.
  const liveSessions = new Set(sessions.map((s) => s.name));
  const queuedIdeas = ideas
    .filter((idea) => idea.status === "needs-run")
    .sort((a, b) => (a.updated || a.id).localeCompare(b.updated || b.id));
  const runningIdeas = ideas
    .filter((idea) => idea.status === "running")
    .sort((a, b) => (a.updated || a.id).localeCompare(b.updated || b.id));
  const gpuQueue = [...runningIdeas, ...queuedIdeas];
  const gpuBusy = runningIdeas.length > 0;

  // Split the flat tmux list by what each session is for, so idea-generation
  // sessions sit with the Ideas section and GPU-run supervisors with the GPU
  // section. Anything unrecognised falls into "other".
  const ideaSessions = sessions.filter(
    (s) =>
      s.name.startsWith(GENERATE_SESSION_PREFIX) ||
      s.name.startsWith(IMPLEMENT_SESSION_PREFIX)
  );
  const runSessions = sessions.filter((s) => s.name.startsWith(RUN_SESSION_PREFIX));
  const otherSessions = sessions.filter(
    (s) => !ideaSessions.includes(s) && !runSessions.includes(s)
  );

  // location: where these tmux sessions live ("Local · Mac"). Each row can be
  // expanded to follow (and scroll back through) its saved log.
  const renderSessionList = (list: Session[], emptyText: string, location = "Local · Mac") =>
    list.length === 0 ? (
      <p className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-5 text-center text-sm text-[#faf9f6]/40">
        {emptyText}
      </p>
    ) : (
      <ul className="space-y-2">
        {list.map((session) => {
          const open = expandedLogs.has(session.name);
          const log = logData[session.name];
          return (
            <li
              key={session.name}
              className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3"
            >
              <div className="flex items-center justify-between gap-4">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="truncate font-mono text-sm text-[#faf9f6]">
                      {session.name}
                    </p>
                    <span className="shrink-0 rounded-full border border-sky-300/25 bg-sky-300/10 px-2 py-0.5 text-[9px] uppercase tracking-[0.14em] text-sky-200/80">
                      {location}
                    </span>
                  </div>
                  <p className="text-xs text-[#faf9f6]/40">
                    {session.windows} window{session.windows === 1 ? "" : "s"} ·
                    started {new Date(session.created).toLocaleTimeString()}
                  </p>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <button
                    type="button"
                    onClick={() => toggleLog(session.name)}
                    className="rounded-full border border-white/15 bg-white/[0.04] px-3 py-2 text-xs font-semibold uppercase tracking-[0.18em] text-[#faf9f6]/70 transition hover:border-white/30 hover:text-white focus:outline-none focus:ring-2 focus:ring-white/20"
                  >
                    {open ? "Hide log" : "Logs"}
                  </button>
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
              </div>

              {open && (
                <div className="mt-3">
                  <div className="mb-1 flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-[#faf9f6]/35">
                    <span>tmux capture · {location}</span>
                    <span>
                      {log
                        ? `${log.alive ? "live" : "ended"} · updated ${formatAgo(now - log.at)} ago`
                        : "loading…"}
                    </span>
                  </div>
                  <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-white/10 bg-black/40 px-3 py-2 font-mono text-[11px] leading-relaxed text-[#faf9f6]/75">
                    {log
                      ? log.text || "(no output captured yet)"
                      : "Loading log…"}
                  </pre>
                  <p className="mt-1 text-[10px] text-[#faf9f6]/30">
                    Refreshes every 3s · saved to disk, swept after 1h.
                  </p>
                </div>
              )}
            </li>
          );
        })}
      </ul>
    );

  // Baseline-vs-experiment val-loss comparison drawn under a finished idea.
  // Lower loss = better. The axis is zoomed around the three values so the
  // (tiny) differences are actually visible.
  const renderResult = (r: Result) => {
    const rows: { label: string; val: number | null; kind: "ctrl" | "trt" }[] = [
      { label: "Baseline (ctrl)", val: r.controlVal, kind: "ctrl" },
      { label: "Experiment", val: r.treatmentVal, kind: "trt" },
      { label: "Baseline₂ (ctrl2)", val: r.ctrl2Val, kind: "ctrl" },
    ];
    const vals = rows.map((x) => x.val).filter((v): v is number => v != null);
    if (vals.length === 0) return null;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const span = max - min || 1;
    const pad = span * 0.6 + 1e-6;
    const axisMin = min - pad;
    const axisMax = max + pad;
    const pct = (v: number) =>
      Math.max(2, Math.min(100, ((v - axisMin) / (axisMax - axisMin)) * 100));

    const verdict = r.verdict || "—";
    const vColor =
      verdict === "WIN"
        ? "border-emerald-400/40 bg-emerald-400/15 text-emerald-200"
        : verdict === "DRIFT" || verdict === "FAIL"
          ? "border-red-400/40 bg-red-400/15 text-red-200"
          : "border-[#faf9f6]/20 bg-white/5 text-[#faf9f6]/60"; // NULL / unknown
    // For Δ: negative = experiment lower than baseline = better (green).
    const deltaText = (d: number | null) =>
      d == null ? "—" : `${d > 0 ? "+" : ""}${d.toFixed(4)}`;
    const deltaColor = (d: number | null) =>
      d == null
        ? "text-[#faf9f6]/40"
        : d < 0
          ? "text-emerald-300"
          : d > 0
            ? "text-amber-300"
            : "text-[#faf9f6]/60";

    return (
      <div className="mt-3 rounded-lg border border-white/10 bg-black/20 px-3 py-3">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-[10px] uppercase tracking-[0.2em] text-[#faf9f6]/40">
            val loss · baseline vs experiment
          </span>
          <span
            className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em] ${vColor}`}
          >
            {verdict}
          </span>
        </div>
        <div className="space-y-1.5">
          {rows.map((row) =>
            row.val == null ? null : (
              <div key={row.label} className="flex items-center gap-2">
                <span className="w-32 shrink-0 text-[11px] text-[#faf9f6]/55">
                  {row.label}
                </span>
                <div className="relative h-3 flex-1 overflow-hidden rounded-full bg-white/10">
                  <div
                    className={`h-full rounded-full ${
                      row.kind === "trt"
                        ? verdict === "WIN"
                          ? "bg-emerald-400/80"
                          : verdict === "DRIFT" || verdict === "FAIL"
                            ? "bg-red-400/70"
                            : "bg-sky-400/80"
                        : "bg-[#faf9f6]/30"
                    }`}
                    style={{ width: `${pct(row.val)}%` }}
                  />
                </div>
                <span className="w-16 shrink-0 text-right font-mono text-[11px] text-[#faf9f6]/80">
                  {row.val.toFixed(4)}
                </span>
              </div>
            )
          )}
        </div>
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
          <span className="text-[#faf9f6]/40">
            Δ vs ctrl{" "}
            <span className={`font-mono ${deltaColor(r.deltaCtrl)}`}>
              {deltaText(r.deltaCtrl)}
            </span>
          </span>
          <span className="text-[#faf9f6]/40">
            Δ vs ctrl2{" "}
            <span className={`font-mono ${deltaColor(r.deltaCtrl2)}`}>
              {deltaText(r.deltaCtrl2)}
            </span>
          </span>
          <span className="text-[#faf9f6]/30">(− = experiment better)</span>
        </div>
      </div>
    );
  };

  return (
    <main className="min-h-screen bg-[#1f1e1d] pt-28 text-[#faf9f6] md:pt-36">
      <div className="container mx-auto flex min-h-[calc(100vh-12rem)] flex-col items-center px-6 py-24">
        {/* Global control — the agent every launch uses. */}
        <div className="flex flex-col items-center gap-2 text-center">
          <span className="text-[10px] font-semibold uppercase tracking-[0.24em] text-[#faf9f6]/45">
            Agent
          </span>
          <select
            value={agent}
            onChange={(e) => setAgent(e.target.value)}
            className="rounded-full border border-cyan-300/30 bg-[#1f1e1d] px-5 py-2.5 text-sm font-semibold tracking-[0.08em] text-cyan-200 transition hover:border-cyan-300/60 focus:outline-none focus:ring-2 focus:ring-cyan-300/40"
          >
            {AGENT_OPTIONS.map((opt) => (
              <option key={opt.id} value={opt.id} className="bg-[#1f1e1d] text-cyan-100">
                {opt.label}
              </option>
            ))}
          </select>
          <label className="mt-1 flex cursor-pointer items-center gap-2 text-[11px] text-[#faf9f6]/55">
            <input
              type="checkbox"
              checked={headless}
              onChange={(e) => setHeadless(e.target.checked)}
              className="h-3.5 w-3.5 cursor-pointer accent-cyan-400"
            />
            Headless — exit &amp; auto-close tmux when done
            <span className="text-[#faf9f6]/30">
              (uncheck to keep it open to watch)
            </span>
          </label>
        </div>

        {/* ================= SECTION 1 · IDEAS ================= */}
        <section className="mt-14 w-full max-w-2xl">
          <div className="mb-5 flex items-end justify-between gap-3 border-b border-amber-300/20 pb-3">
            <div className="flex items-center gap-3">
              <span className="h-7 w-1 rounded-full bg-amber-300/70" />
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-[0.28em] text-amber-200">
                  Ideas
                </h2>
                <p className="text-[11px] text-[#faf9f6]/40">
                  Brainstorm ideas, then implement them into a runnable A/B.
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={refreshIdeas}
              className="shrink-0 text-xs uppercase tracking-[0.2em] text-amber-300/70 transition hover:text-amber-200"
            >
              Refresh
            </button>
          </div>

          {/* Generate controls + prompt edit links */}
          <div className="mb-6 flex flex-col items-center gap-3 text-center">
            <form onSubmit={handleGenerate}>
              <button
                type="submit"
                disabled={isGenerating}
                className="rounded-full border border-amber-300/30 bg-amber-300/10 px-8 py-3.5 text-sm font-semibold uppercase tracking-[0.24em] text-amber-200 transition hover:border-amber-300/60 hover:bg-amber-300/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-amber-300/40 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isGenerating ? "Generating..." : "Generate Ideas"}
              </button>
            </form>
            {generateMessage && (
              <p className="text-sm text-amber-300">{generateMessage}</p>
            )}
            <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2">
              <button
                type="button"
                onClick={() =>
                  setOpenFile({ path: IDEAS_PROMPT_PATH, title: "generate-ideas.md" })
                }
                className="text-xs uppercase tracking-[0.2em] text-amber-300/70 underline-offset-4 transition hover:text-amber-200 hover:underline"
              >
                Edit ideas prompt
              </button>
              <button
                type="button"
                onClick={() =>
                  setOpenFile({
                    path: IMPLEMENT_PROMPT_PATH,
                    title: "implement-idea.md",
                  })
                }
                className="text-xs uppercase tracking-[0.2em] text-emerald-300/70 underline-offset-4 transition hover:text-emerald-200 hover:underline"
              >
                Edit implement prompt
              </button>
            </div>
          </div>

          {ideaActionMsg && (
            <p className="mb-2 text-xs text-amber-300">{ideaActionMsg}</p>
          )}
          {ideaLoadError && (
            <p className="mb-2 text-xs text-orange-300">{ideaLoadError}</p>
          )}

          {ideas.length === 0 ? (
            <p className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-6 text-center text-sm text-[#faf9f6]/40">
              No ideas yet.
            </p>
          ) : (
            <ul className="space-y-2">
              {ideas.map((idea) => {
                const implementSessionName = IMPLEMENT_SESSION_PREFIX + idea.id;
                const runSessionName = RUN_SESSION_PREFIX + idea.id;
                const liveImplement = liveSessions.has(implementSessionName);
                const liveRun = liveSessions.has(runSessionName);
                const liveSessionName = liveRun ? runSessionName : implementSessionName;
                const isLive = liveImplement || liveRun;
                const isTrackedWip =
                  idea.status === "implementing" || idea.status === "running";
                const isStuck = isTrackedWip && !isLive;
                const busy = implementing === idea.id;
                const canImplement = !["needs-run", "running", "done"].includes(
                  idea.status
                );

                return (
                  <li
                    key={idea.id}
                    className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3"
                  >
                    <div className="flex items-start justify-between gap-3">
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
                        <p className="mt-1 text-xs text-[#faf9f6]/55">
                          {idea.plain}
                        </p>
                      )}
                    </button>
                    <div className="flex shrink-0 flex-col items-end gap-2">
                      <div className="flex items-center gap-2">
                        {isLive && (
                          <span className="flex items-center gap-1 text-[10px] uppercase tracking-[0.15em] text-emerald-300">
                            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
                            {liveRun ? "gpu" : "working"}
                          </span>
                        )}
                        {isStuck && (
                          <span className="text-[10px] uppercase tracking-[0.15em] text-orange-300">
                            stuck
                          </span>
                        )}
                        <span className="rounded-full border border-amber-300/20 bg-amber-300/5 px-2.5 py-0.5 text-[10px] uppercase tracking-[0.15em] text-amber-200/80">
                          {idea.status}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        {idea.evidencePath && (
                          <button
                            type="button"
                            onClick={() =>
                              setOpenFile({
                                path: idea.evidencePath!,
                                title: `${idea.title} — evidence`,
                              })
                            }
                            className="rounded-full border border-fuchsia-300/30 bg-fuchsia-300/10 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] text-fuchsia-200 transition hover:border-fuchsia-300/60 hover:bg-fuchsia-300/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-fuchsia-300/40"
                          >
                            Evidence
                          </button>
                        )}
                        {isStuck && (
                          <button
                            type="button"
                            onClick={() =>
                              handleReset(
                                idea.id,
                                idea.status === "running" ? "needs-run" : "needs-taste",
                                idea.status === "running"
                                  ? "requeued stuck GPU run from UI"
                                  : "reset stuck idea from UI"
                              )
                            }
                            disabled={busy}
                            className="rounded-full border border-orange-400/30 bg-orange-400/10 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] text-orange-300 transition hover:border-orange-400/60 hover:bg-orange-400/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-orange-400/40 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            {idea.status === "running" ? "Requeue" : "Reset"}
                          </button>
                        )}
                        {isLive ? (
                          <button
                            type="button"
                            onClick={() => handleAttach(liveSessionName)}
                            disabled={attaching === liveSessionName}
                            className="rounded-full border border-cyan-300/30 bg-cyan-300/10 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] text-cyan-200 transition hover:border-cyan-300/60 hover:bg-cyan-300/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-cyan-300/40 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            {attaching === liveSessionName ? "..." : "Attach"}
                          </button>
                        ) : canImplement || (isStuck && idea.status !== "running") ? (
                          <button
                            type="button"
                            onClick={() => handleImplement(idea.id)}
                            disabled={busy}
                            className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] text-emerald-300 transition hover:border-emerald-400/60 hover:bg-emerald-400/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-emerald-400/40 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            {busy
                              ? "Launching…"
                              : isStuck
                                ? "Retry"
                                : "Implement"}
                          </button>
                        ) : idea.status === "needs-run" ? (
                          <span className="rounded-full border border-cyan-300/20 bg-cyan-300/5 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] text-cyan-200/70">
                            Queued
                          </span>
                        ) : null}
                      </div>
                    </div>
                    </div>
                    {idea.result && renderResult(idea.result)}
                  </li>
                );
              })}
            </ul>
          )}

          {/* Idea-work tmux sessions (generate + implement) */}
          <div className="mt-6">
            <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.24em] text-amber-200/50">
              Idea-work tmux ({ideaSessions.length})
            </h3>
            {renderSessionList(ideaSessions, "No generate/implement sessions running.")}
          </div>
        </section>

        {/* ================= SECTION 2 · GPU RUNS ================= */}
        <section className="mt-16 w-full max-w-2xl">
          <div className="mb-5 flex items-end justify-between gap-3 border-b border-cyan-300/20 pb-3">
            <div className="flex items-center gap-3">
              <span className="h-7 w-1 rounded-full bg-cyan-300/70" />
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-[0.28em] text-cyan-200">
                  GPU runs
                </h2>
                <p className="text-[11px] text-[#faf9f6]/40">
                  Run the queued A/Bs on the Vast box and watch the GPU live.
                </p>
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-4">
              <button
                type="button"
                onClick={() =>
                  setOpenFile({ path: RUN_PROMPT_PATH, title: "run-idea.md" })
                }
                className="text-xs uppercase tracking-[0.2em] text-cyan-300/70 underline-offset-4 transition hover:text-cyan-200 hover:underline"
              >
                Edit run prompt
              </button>
              <button
                type="button"
                onClick={() =>
                  setOpenFile({ path: REMOTE_BOX_PATH, title: "remote-box.json" })
                }
                className="text-xs uppercase tracking-[0.2em] text-fuchsia-300/70 underline-offset-4 transition hover:text-fuchsia-200 hover:underline"
              >
                Edit GPU box
              </button>
            </div>
          </div>

          {/* GPU queue */}
          <div className="w-full rounded-xl border border-cyan-300/15 bg-cyan-300/[0.04] px-4 py-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-xs font-semibold uppercase tracking-[0.24em] text-cyan-200/70">
                GPU queue
              </h2>
              <p className="mt-1 text-xs text-[#faf9f6]/45">
                {runningIdeas.length} running · {queuedIdeas.length} ready
              </p>
            </div>
            <button
              type="button"
              onClick={handleRunNext}
              disabled={isRunningNext || gpuBusy || queuedIdeas.length === 0}
              className="rounded-full border border-cyan-300/30 bg-cyan-300/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-cyan-200 transition hover:border-cyan-300/60 hover:bg-cyan-300/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-cyan-300/40 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isRunningNext
                ? "Launching..."
                : gpuBusy
                  ? "GPU busy"
                  : queuedIdeas.length === 0
                    ? "Queue empty"
                    : "Run next"}
            </button>
          </div>

          {runMessage && <p className="mt-3 text-xs text-cyan-200">{runMessage}</p>}
          {(ideaLoadError || sessionLoadError) && (
            <p className="mt-3 text-xs text-orange-200">
              {ideaLoadError || sessionLoadError}
            </p>
          )}

          {gpuQueue.length === 0 ? (
            <p className="mt-4 rounded-lg border border-white/10 bg-white/[0.03] px-4 py-5 text-center text-sm text-[#faf9f6]/40">
              No ready GPU work.
            </p>
          ) : (
            <ul className="mt-4 space-y-2">
              {gpuQueue.map((idea, index) => {
                const sessionName = RUN_SESSION_PREFIX + idea.id;
                const isRunLive = liveSessions.has(sessionName);
                const isRunStuck = idea.status === "running" && !isRunLive;

                return (
                  <li
                    key={idea.id}
                    className="flex items-center justify-between gap-3 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2.5"
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
                      <p className="mt-1 font-mono text-[11px] text-[#faf9f6]/35">
                        {idea.id}
                      </p>
                    </button>
                    <div className="flex shrink-0 items-center gap-2">
                      {isRunLive && (
                        <span className="flex items-center gap-1 text-[10px] uppercase tracking-[0.15em] text-emerald-300">
                          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
                          running
                        </span>
                      )}
                      {isRunStuck && (
                        <span className="text-[10px] uppercase tracking-[0.15em] text-orange-300">
                          stuck
                        </span>
                      )}
                      {idea.status === "needs-run" && (
                        <span className="rounded-full border border-cyan-300/20 bg-cyan-300/5 px-2.5 py-0.5 text-[10px] uppercase tracking-[0.15em] text-cyan-200/80">
                          #{index + 1}
                        </span>
                      )}
                      {isRunLive ? (
                        <button
                          type="button"
                          onClick={() => handleAttach(sessionName)}
                          disabled={attaching === sessionName}
                          title="Attach the local supervisor tmux (SSHes the box, polls, writes evidence). Not the GPU itself — use the GPU box panel below for that."
                          className="rounded-full border border-cyan-300/30 bg-cyan-300/10 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] text-cyan-200 transition hover:border-cyan-300/60 hover:bg-cyan-300/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-cyan-300/40 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          {attaching === sessionName ? "..." : "Runner"}
                        </button>
                      ) : isRunStuck ? (
                        <button
                          type="button"
                          onClick={() =>
                            handleReset(
                              idea.id,
                              "needs-run",
                              "requeued stuck GPU run from UI"
                            )
                          }
                          disabled={implementing === idea.id}
                          className="rounded-full border border-orange-400/30 bg-orange-400/10 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] text-orange-300 transition hover:border-orange-400/60 hover:bg-orange-400/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-orange-400/40 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          Requeue
                        </button>
                      ) : null}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        {/* GPU box — the real training, in tmux `arq` on the remote Vast box */}
        <div className="mt-6 w-full max-w-2xl rounded-xl border border-fuchsia-300/15 bg-fuchsia-300/[0.04] px-4 py-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-xs font-semibold uppercase tracking-[0.24em] text-fuchsia-200/70">
                  GPU box
                </h2>
                <span className="rounded-full border border-fuchsia-300/25 bg-fuchsia-300/10 px-2 py-0.5 text-[9px] uppercase tracking-[0.14em] text-fuchsia-200/80">
                  Remote · Vast GPU
                </span>
              </div>
              <p className="mt-1 text-xs text-[#faf9f6]/45">
                {gpuInfo?.host ? `${gpuInfo.host} · ` : ""}
                {arqAlive ? (
                  <span className="text-emerald-300">tmux arq live</span>
                ) : (
                  "tmux arq idle (starts when a run is active)"
                )}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={refreshGpu}
                disabled={gpuLoading}
                className="rounded-full border border-fuchsia-300/30 bg-fuchsia-300/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-fuchsia-200 transition hover:border-fuchsia-300/60 hover:bg-fuchsia-300/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-fuchsia-300/40 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {gpuLoading ? "Checking…" : "Refresh"}
              </button>
              <button
                type="button"
                onClick={handleAttachGpu}
                disabled={attaching === "__gpu__" || !arqAlive}
                title={
                  arqAlive
                    ? "Open a Terminal SSH'd into the live remote GPU tmux (arq)."
                    : "No live GPU run. The arq tmux only exists while a run is active — start one with Run next."
                }
                className="rounded-full border border-fuchsia-300/30 bg-fuchsia-300/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-fuchsia-200 transition hover:border-fuchsia-300/60 hover:bg-fuchsia-300/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-fuchsia-300/40 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {attaching === "__gpu__"
                  ? "…"
                  : arqAlive
                    ? "Attach GPU"
                    : "GPU idle"}
              </button>
            </div>
          </div>

          {/* Live usage — polled independently every 4s (util % + VRAM). */}
          <div className="mt-4 space-y-2.5">
            {(() => {
              const u = gpuUsage;
              const memPct =
                u && u.memTotal > 0 ? Math.round((u.memUsed / u.memTotal) * 100) : 0;
              const util = u ? Math.max(0, Math.min(100, Math.round(u.utilization))) : 0;
              return (
                <>
                  <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.2em] text-[#faf9f6]/40">
                    <span>{u?.name || "GPU usage"}</span>
                    <span className={gpuUsageStale ? "text-orange-300/80" : "text-emerald-300/70"}>
                      {gpuUsageStale ? "stale" : u ? "live" : "—"}
                    </span>
                  </div>
                  <div>
                    <div className="mb-1 flex justify-between text-[11px] text-[#faf9f6]/60">
                      <span>Compute</span>
                      <span className="font-mono">{u ? `${util}%` : "—"}</span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
                      <div
                        className="h-full rounded-full bg-emerald-400/80 transition-[width] duration-500"
                        style={{ width: `${util}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="mb-1 flex justify-between text-[11px] text-[#faf9f6]/60">
                      <span>VRAM</span>
                      <span className="font-mono">
                        {u ? `${u.memUsed} / ${u.memTotal} MiB · ${memPct}%` : "—"}
                      </span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
                      <div
                        className="h-full rounded-full bg-fuchsia-400/80 transition-[width] duration-500"
                        style={{ width: `${memPct}%` }}
                      />
                    </div>
                  </div>
                  {/* How far behind the numbers are: age of the reading + the
                      SSH round-trip it took to fetch, polled every 4s. */}
                  <p className="text-[10px] text-[#faf9f6]/35">
                    {gpuUsageAt
                      ? `Reading ${formatAgo(now - gpuUsageAt)} old · ${
                          gpuUsageLatencyMs != null ? `~${gpuUsageLatencyMs}ms to fetch · ` : ""
                        }polled every 4s`
                      : "Polling GPU every 4s…"}
                  </p>
                </>
              );
            })()}
          </div>

          {gpuError && <p className="mt-3 text-xs text-orange-200">{gpuError}</p>}

          {gpuInfo && (
            <div className="mt-4 space-y-3">
              {gpuInfo.gpu && (
                <p className="font-mono text-[11px] text-fuchsia-200/80">{gpuInfo.gpu}</p>
              )}
              {gpuInfo.status && (
                <div>
                  <p className="mb-1 text-[10px] uppercase tracking-[0.2em] text-[#faf9f6]/40">
                    STATUS
                  </p>
                  <pre className="overflow-x-auto rounded-lg border border-white/10 bg-black/30 px-3 py-2 font-mono text-[11px] leading-relaxed text-[#faf9f6]/75">
                    {gpuInfo.status}
                  </pre>
                </div>
              )}
              {gpuInfo.logTail && (
                <div>
                  <p className="mb-1 text-[10px] uppercase tracking-[0.2em] text-[#faf9f6]/40">
                    {gpuInfo.logName || "log"}
                  </p>
                  <pre className="max-h-64 overflow-auto rounded-lg border border-white/10 bg-black/30 px-3 py-2 font-mono text-[11px] leading-relaxed text-[#faf9f6]/75">
                    {gpuInfo.logTail}
                  </pre>
                </div>
              )}
              {gpuInfo.sshAttach && (
                <button
                  type="button"
                  onClick={() => {
                    navigator.clipboard?.writeText(gpuInfo.sshAttach);
                    setGpuError("Copied ssh attach command to clipboard");
                  }}
                  title="Copy: SSH into the remote GPU tmux"
                  className="block w-full overflow-x-auto rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-left font-mono text-[11px] text-[#faf9f6]/55 transition hover:text-[#faf9f6]/80"
                >
                  $ {gpuInfo.sshAttach}
                </button>
              )}
            </div>
          )}
        </div>

          {/* Run-supervisor tmux sessions (lab-run-*) */}
          <div className="mt-6">
            <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.24em] text-cyan-200/50">
              Run-supervisor tmux ({runSessions.length})
            </h3>
            <p className="mb-2 text-[11px] text-[#faf9f6]/35">
              These local sessions SSH the box, poll STATUS, and write evidence —
              the training itself runs in tmux <span className="font-mono">arq</span> on the box (above).
            </p>
            {renderSessionList(runSessions, "No run supervisors active.")}
          </div>
        </section>

        {/* ================= SECTION 3 · OTHER SESSIONS ================= */}
        <section className="mt-16 w-full max-w-2xl">
          <div className="mb-5 flex items-end justify-between gap-3 border-b border-white/15 pb-3">
            <div className="flex items-center gap-3">
              <span className="h-7 w-1 rounded-full bg-white/40" />
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-[0.28em] text-[#faf9f6]/70">
                  Other tmux
                </h2>
                <p className="text-[11px] text-[#faf9f6]/40">
                  Any sessions not tied to idea generation or GPU runs.
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={refreshSessions}
              className="shrink-0 text-xs uppercase tracking-[0.2em] text-[#faf9f6]/50 transition hover:text-[#faf9f6]/80"
            >
              Refresh
            </button>
          </div>

          {sessionMsg && <p className="mb-2 text-xs text-red-300">{sessionMsg}</p>}
          {sessionLoadError && (
            <p className="mb-2 text-xs text-orange-300">{sessionLoadError}</p>
          )}

          {renderSessionList(otherSessions, "No other tmux sessions.")}
        </section>
      </div>

      <MarkdownPanel
        path={openFile?.path ?? null}
        title={openFile?.title ?? ""}
        onClose={() => setOpenFile(null)}
      />
    </main>
  );
}
