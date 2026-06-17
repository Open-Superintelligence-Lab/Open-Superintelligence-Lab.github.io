"use client";

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
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
const RUNNER_PROMPT_PATH = "autoresearch/prompts/runner.md";
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

// Human-readable labels + colour for each on-disk pipeline status. The raw
// status string (what flip.sh writes) stays the source of truth — this only
// renames them for display, so the jargon ("needs-taste", "needs-recode")
// reads clearly without touching the agents' state machine. Hover shows raw.
const STATUS_META: Record<string, { label: string; cls: string }> = {
  "needs-taste": { label: "Proposed", cls: "border-amber-300/25 bg-amber-300/5 text-amber-200/80" },
  implementing: { label: "Implementing", cls: "border-emerald-300/25 bg-emerald-300/10 text-emerald-200/90" },
  "needs-run": { label: "Queued · GPU", cls: "border-cyan-300/25 bg-cyan-300/10 text-cyan-200/90" },
  running: { label: "Running · GPU", cls: "border-sky-300/40 bg-sky-300/15 text-sky-100" },
  "needs-recode": { label: "Fixing · failed run", cls: "border-orange-300/25 bg-orange-300/10 text-orange-200/90" },
  "needs-review": { label: "Review", cls: "border-violet-300/25 bg-violet-300/10 text-violet-200/90" },
  done: { label: "Done", cls: "border-[#faf9f6]/20 bg-white/5 text-[#faf9f6]/70" },
  rejected: { label: "Rejected", cls: "border-red-300/25 bg-red-300/5 text-red-200/80" },
  win: { label: "Win", cls: "border-emerald-400/40 bg-emerald-400/15 text-emerald-200" },
  null: { label: "Null", cls: "border-[#faf9f6]/20 bg-white/5 text-[#faf9f6]/60" },
  drift: { label: "Drift", cls: "border-red-400/40 bg-red-400/15 text-red-200" },
  fail: { label: "Fail", cls: "border-red-400/40 bg-red-400/15 text-red-200" },
};

function statusMeta(s: string): { label: string; cls: string } {
  return (
    STATUS_META[s] ?? {
      label: s,
      cls: "border-amber-300/20 bg-amber-300/5 text-amber-200/80",
    }
  );
}

// "3s" / "2m 5s" — compact relative age for freshness labels.
function formatAgo(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${s % 60}s`;
}

// Statuses that mean an experiment is finished — they show the full training
// curve and live in the "Finished experiments" section at the bottom.
const FINISHED_STATUSES = new Set([
  "done",
  "win",
  "null",
  "drift",
  "fail",
  "rejected",
]);

// Whether a status gets a live "time in this state" timer. Rule, not a fixed
// list, so any in-flight status (current or future) is covered: every idea is
// timed EXCEPT needs-taste (Proposed — just waiting to be picked up) and the
// finished statuses (those show final results, not an elapsed clock).
const isTimedStatus = (status: string) =>
  status !== "needs-taste" && !FINISHED_STATUSES.has(status);

type CurveRun = {
  role: "control" | "treatment" | "control2";
  label: string;
  steps: number[];
  valLosses: number[];
};
type CurveData = { id: string; runs: CurveRun[] };

// Module-level cache so re-renders / re-mounts don't re-hit the API for the same
// finished experiment. The curve never changes once a run is done.
const curveCache = new Map<string, CurveData>();

const CURVE_COLOR: Record<CurveRun["role"], string> = {
  control: "#9ca3af", // grey — baseline
  treatment: "#34d399", // emerald — the experiment
  control2: "#c084fc", // violet — second baseline
};

// Full per-step validation-loss curve for one finished experiment, drawn as an
// inline SVG (no chart lib). One line per run (ctrl / experiment / ctrl2) over a
// shared, auto-scaled axis, with a small legend. Renders nothing until data
// arrives, and nothing if the experiment has no usable curve.
function TrainingCurve({ id }: { id: string }) {
  const [data, setData] = useState<CurveData | null>(
    () => curveCache.get(id) ?? null
  );

  useEffect(() => {
    if (curveCache.has(id)) {
      setData(curveCache.get(id)!);
      return;
    }
    let cancelled = false;
    fetch("/api/training-curve/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((d: CurveData | null) => {
        if (cancelled || !d || !Array.isArray(d.runs)) return;
        curveCache.set(id, d);
        setData(d);
      })
      .catch(() => {
        /* leave chart hidden on failure */
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  const runs = (data?.runs ?? []).filter((r) => r.steps.length > 1);
  if (runs.length === 0) return null;

  const W = 460;
  const H = 190;
  const padL = 40;
  const padR = 12;
  const padT = 12;
  const padB = 26;

  const allSteps = runs.flatMap((r) => r.steps);
  const allLoss = runs.flatMap((r) => r.valLosses);
  const minX = Math.min(...allSteps);
  const maxX = Math.max(...allSteps);
  const minY = Math.min(...allLoss);
  const maxY = Math.max(...allLoss);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const px = (x: number) => padL + ((x - minX) / spanX) * (W - padL - padR);
  const py = (y: number) =>
    padT + (1 - (y - minY) / spanY) * (H - padT - padB);

  const yTicks = [minY, minY + spanY / 2, maxY];

  return (
    <div className="mt-3 rounded-lg border border-white/10 bg-black/20 px-3 py-3">
      <div className="mb-2 text-[10px] uppercase tracking-[0.2em] text-[#faf9f6]/40">
        val loss · full training curve
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label={`Validation loss curve for ${id}`}
      >
        {/* y gridlines + labels */}
        {yTicks.map((v, i) => (
          <g key={i}>
            <line
              x1={padL}
              y1={py(v)}
              x2={W - padR}
              y2={py(v)}
              stroke="#ffffff"
              strokeOpacity={0.07}
              strokeWidth={1}
            />
            <text
              x={padL - 5}
              y={py(v) + 3}
              textAnchor="end"
              fontSize={9}
              fill="#faf9f6"
              fillOpacity={0.45}
              fontFamily="monospace"
            >
              {v.toFixed(2)}
            </text>
          </g>
        ))}
        {/* x end labels */}
        <text
          x={padL}
          y={H - 8}
          textAnchor="start"
          fontSize={9}
          fill="#faf9f6"
          fillOpacity={0.4}
          fontFamily="monospace"
        >
          step {minX}
        </text>
        <text
          x={W - padR}
          y={H - 8}
          textAnchor="end"
          fontSize={9}
          fill="#faf9f6"
          fillOpacity={0.4}
          fontFamily="monospace"
        >
          {maxX}
        </text>
        {/* one polyline per run */}
        {runs.map((r) => (
          <polyline
            key={r.role}
            fill="none"
            stroke={CURVE_COLOR[r.role]}
            strokeWidth={1.75}
            strokeOpacity={r.role === "treatment" ? 1 : 0.85}
            points={r.steps
              .map((s, idx) => `${px(s)},${py(r.valLosses[idx])}`)
              .join(" ")}
          />
        ))}
      </svg>
      {/* legend */}
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-[#faf9f6]/55">
        {runs.map((r) => (
          <span key={r.role} className="flex items-center gap-1.5">
            <span
              className="inline-block h-2 w-3 rounded-sm"
              style={{ backgroundColor: CURVE_COLOR[r.role] }}
            />
            <span className="font-mono">{r.label}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

export default function LaunchCodexPage() {
  const [agent, setAgent] = useState<string>("minimax");
  // Headless = run the agent non-interactively so it exits (and the tmux pane
  // self-closes) when the task finishes. On by default; uncheck to keep the
  // agent open at its REPL so you can attach and watch/intervene.
  const [headless, setHeadless] = useState<boolean>(true);
  // How many ideas the next "Generate Ideas" press asks the agent for — injected
  // into the generate prompt server-side.
  const [ideaCount, setIdeaCount] = useState<number>(10);
  // Autorun: when on, each finished run auto-launches the next queued idea
  // (server-side, via run-done). Mirrors a persisted flag; reflects real state.
  const [autorunOn, setAutorunOn] = useState<boolean>(false);
  const [autorunBusy, setAutorunBusy] = useState<boolean>(false);
  // Auto-implement: when on, Proposed ideas get implemented automatically (up to
  // a parallel cap). Defaults ON — optimistic so the toggle reads "on" before
  // the first state fetch resolves.
  const [autoImplementOn, setAutoImplementOn] = useState<boolean>(true);
  const [autoImplementBusy, setAutoImplementBusy] = useState<boolean>(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generateMessage, setGenerateMessage] = useState("");
  const [sessions, setSessions] = useState<Session[]>([]);
  const [killing, setKilling] = useState<string | null>(null);
  const [ideas, setIdeas] = useState<Idea[]>([]);
  const [openFile, setOpenFile] = useState<{ path: string; title: string } | null>(
    null
  );
  // Settings popover (uncommon controls: agent, headless, prompt files).
  const [settingsOpen, setSettingsOpen] = useState<boolean>(false);
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

  // Autorun: a bodyless POST reports state AND drives one tick — it re-invokes
  // the single lab-autorun runner agent (which drains the whole needs-run queue
  // on the box) whenever autorun is on, no runner is alive, and there's work.
  // Poll on load + every 5s so the queue keeps draining without a click.
  useEffect(() => {
    const tick = () => {
      fetch("/api/autorun/", { method: "POST" })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => {
          if (d && typeof d.enabled === "boolean") setAutorunOn(d.enabled);
        })
        .catch(() => {
          /* default off */
        });
    };
    tick();
    const interval = setInterval(tick, 5000);
    return () => clearInterval(interval);
  }, []);

  // Auto-implement: a bodyless POST reports state AND drives one tick (launches
  // implements for Proposed ideas up to the cap). Call it on load and every 5s
  // so freshly-generated ideas get picked up without a click. The server-side
  // implement-done chain keeps it going when the tab is closed.
  useEffect(() => {
    const tick = () => {
      fetch("/api/auto-implement/", { method: "POST" })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => {
          if (d && typeof d.enabled === "boolean") setAutoImplementOn(d.enabled);
        })
        .catch(() => {
          /* leave the optimistic state; next tick retries */
        });
    };
    tick();
    const interval = setInterval(tick, 5000);
    return () => clearInterval(interval);
  }, []);

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
        body: JSON.stringify({ agent, headless, count: ideaCount }),
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
        body: JSON.stringify({ slug, agent, headless }),
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

  // Flip autorun on/off. Enabling also kicks the first run server-side (and we
  // pass the selected agent so the whole chain uses it).
  const handleToggleAutorun = async () => {
    const next = !autorunOn;
    setAutorunBusy(true);
    try {
      const response = await fetch("/api/autorun/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: next, agent, headless }),
      });
      const data = await response.json().catch(() => ({}));
      if (response.ok && typeof data.enabled === "boolean") {
        setAutorunOn(data.enabled);
        setRunMessage(
          data.enabled
            ? "Autorun on — finished runs auto-launch the next queued idea."
            : "Autorun off."
        );
      }
    } catch {
      setRunMessage("Failed to toggle autorun");
    } finally {
      setAutorunBusy(false);
      refreshSessions();
      refreshIdeas();
    }
  };

  // Flip auto-implement on/off. The toggle response also carries the running
  // state back; enabling drives an immediate tick server-side.
  const handleToggleAutoImplement = async () => {
    const next = !autoImplementOn;
    setAutoImplementBusy(true);
    setAutoImplementOn(next); // optimistic
    try {
      const response = await fetch("/api/auto-implement/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: next, agent }),
      });
      const data = await response.json().catch(() => ({}));
      if (response.ok && typeof data.enabled === "boolean") {
        setAutoImplementOn(data.enabled);
      }
    } catch {
      setAutoImplementOn(!next); // revert on failure
    } finally {
      setAutoImplementBusy(false);
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
        body: JSON.stringify({ agent, headless }),
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

  // "How long it's been in this state" — measured from the idea's last status
  // flip (idea.updated, written by flip.sh). `now` ticks every 1s, so the label
  // counts up live between the 5s polls. Returns null for an unparseable stamp.
  const timeInState = (iso: string): string | null => {
    const t = Date.parse(iso);
    return Number.isFinite(t) ? formatAgo(now - t) : null;
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

  // Group ideas into clear buckets instead of one scattered list. needs-run /
  // running live in the GPU section below, and finished experiments get their
  // own section at the very bottom — so the Ideas section only shows pre-GPU
  // work, grouped by where it is in the pipeline.
  const byStatus = (...statuses: string[]) =>
    ideas
      .filter((i) => statuses.includes(i.status))
      .sort((a, b) => (a.updated || a.id).localeCompare(b.updated || b.id));
  const ideaGroups: { key: string; label: string; ideas: Idea[] }[] = [
    { key: "proposed", label: "Proposed", ideas: byStatus("needs-taste") },
    { key: "implementing", label: "Implementing", ideas: byStatus("implementing", "needs-recode", "recoding") },
    { key: "review", label: "In review", ideas: byStatus("needs-review") },
  ];
  // Anything not finished, not on the GPU, and not in a named bucket above —
  // so a new/unexpected status never silently disappears.
  const bucketed = new Set([
    "needs-taste",
    "implementing",
    "needs-recode",
    "recoding",
    "needs-review",
    "needs-run",
    "running",
  ]);
  const otherIdeas = ideas.filter(
    (i) => !FINISHED_STATUSES.has(i.status) && !bucketed.has(i.status)
  );
  if (otherIdeas.length > 0) {
    ideaGroups.push({ key: "other", label: "Other", ideas: otherIdeas });
  }
  const activeIdeaCount = ideaGroups.reduce((n, g) => n + g.ideas.length, 0);
  const finishedIdeas = ideas
    .filter((i) => FINISHED_STATUSES.has(i.status))
    .sort((a, b) => (b.updated || b.id).localeCompare(a.updated || a.id));

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
                    started {new Date(session.created).toLocaleTimeString()} ·{" "}
                    <span className="font-mono tabular-nums text-[#faf9f6]/55">
                      ⏱ {formatAgo(now - session.created)}
                    </span>
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

  // One idea row — the title, status badge, action buttons, and (for finished
  // ideas) the final-loss summary bars. Used by every grouped list so the cards
  // stay identical wherever they appear.
  const renderIdeaCard = (idea: Idea, extra?: ReactNode) => {
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
    const canImplement = !["needs-run", "running", "done"].includes(idea.status);

    return (
      <li
        key={idea.id}
        className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3"
      >
        <div className="flex items-start justify-between gap-3">
          <button
            type="button"
            onClick={() => setOpenFile({ path: idea.path, title: idea.title })}
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
              {isTimedStatus(idea.status) && timeInState(idea.updated) && (
                <span
                  title="time in this state"
                  className="rounded-full border border-white/10 bg-white/[0.03] px-2 py-0.5 font-mono text-[10px] tabular-nums text-[#faf9f6]/55"
                >
                  ⏱ {timeInState(idea.updated)}
                </span>
              )}
              <span
                title={idea.status}
                className={`rounded-full border px-2.5 py-0.5 text-[10px] uppercase tracking-[0.15em] ${statusMeta(idea.status).cls}`}
              >
                {statusMeta(idea.status).label}
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
              ) : (canImplement && !autoImplementOn) ||
                (isStuck && idea.status !== "running") ? (
                // When auto-implement is on, the normal "Implement" run-once
                // button is hidden (the tick handles Proposed ideas); the stuck
                // "Retry" recovery button still shows.
                <button
                  type="button"
                  onClick={() => handleImplement(idea.id)}
                  disabled={busy}
                  className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] text-emerald-300 transition hover:border-emerald-400/60 hover:bg-emerald-400/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-emerald-400/40 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {busy ? "Launching…" : isStuck ? "Retry" : "Implement"}
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
        {extra}
      </li>
    );
  };

  return (
    <main className="min-h-screen bg-[#1f1e1d] pt-10 text-[#faf9f6] md:pt-12">
      <div className="container mx-auto flex min-h-[calc(100vh-12rem)] flex-col items-center px-6 py-10">
        {/* Uncommon controls (agent, headless, prompt files) live behind this
            gear so the main view stays focused on ideas + the queue. */}
        <div className="relative w-full max-w-2xl">
          <div className="flex justify-end">
            <button
              type="button"
              onClick={() => setSettingsOpen((v) => !v)}
              aria-expanded={settingsOpen}
              className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] transition focus:outline-none focus:ring-2 focus:ring-white/20 ${
                settingsOpen
                  ? "border-white/30 bg-white/[0.08] text-white"
                  : "border-white/12 bg-white/[0.03] text-[#faf9f6]/55 hover:border-white/30 hover:text-white"
              }`}
            >
              <span aria-hidden>⚙</span>
              Settings · {AGENT_OPTIONS.find((o) => o.id === agent)?.label ?? agent}
            </button>
          </div>

          {settingsOpen && (
            <div className="absolute right-0 z-20 mt-2 w-72 rounded-xl border border-white/12 bg-[#262524] p-4 text-left shadow-xl shadow-black/40">
              <div className="mb-3">
                <span className="text-[10px] font-semibold uppercase tracking-[0.24em] text-[#faf9f6]/45">
                  Agent
                </span>
                <select
                  value={agent}
                  onChange={(e) => setAgent(e.target.value)}
                  className="mt-1.5 w-full rounded-lg border border-cyan-300/30 bg-[#1f1e1d] px-3 py-2 text-sm font-semibold tracking-[0.04em] text-cyan-200 transition hover:border-cyan-300/60 focus:outline-none focus:ring-2 focus:ring-cyan-300/40"
                >
                  {AGENT_OPTIONS.map((opt) => (
                    <option key={opt.id} value={opt.id} className="bg-[#1f1e1d] text-cyan-100">
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>

              <label className="flex cursor-pointer items-start gap-2 text-[11px] text-[#faf9f6]/65">
                <input
                  type="checkbox"
                  checked={headless}
                  onChange={(e) => setHeadless(e.target.checked)}
                  className="mt-0.5 h-3.5 w-3.5 cursor-pointer accent-cyan-400"
                />
                <span>
                  Headless — exit &amp; auto-close tmux when done{" "}
                  <span className="text-[#faf9f6]/30">(uncheck to watch)</span>
                </span>
              </label>

              <div className="mt-4 border-t border-white/10 pt-3">
                <span className="text-[10px] font-semibold uppercase tracking-[0.24em] text-[#faf9f6]/40">
                  Edit prompts
                </span>
                <div className="mt-2 flex flex-col gap-1.5 text-xs">
                  {[
                    { path: IDEAS_PROMPT_PATH, title: "generate-ideas.md", label: "Generate ideas" },
                    { path: IMPLEMENT_PROMPT_PATH, title: "implement-idea.md", label: "Implement" },
                    { path: RUNNER_PROMPT_PATH, title: "runner.md", label: "Autorun runner" },
                    { path: RUN_PROMPT_PATH, title: "run-idea.md", label: "Single run" },
                    { path: REMOTE_BOX_PATH, title: "remote-box.json", label: "GPU box" },
                  ].map((p) => (
                    <button
                      key={p.path}
                      type="button"
                      onClick={() => {
                        setOpenFile({ path: p.path, title: p.title });
                        setSettingsOpen(false);
                      }}
                      className="flex items-center justify-between rounded-md px-2 py-1 text-left text-[#faf9f6]/70 transition hover:bg-white/[0.06] hover:text-white"
                    >
                      <span>{p.label}</span>
                      <span className="text-[10px] text-[#faf9f6]/35">{p.title}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ================= SECTION 1 · IDEAS ================= */}
        <section className="mt-14 w-full max-w-4xl">
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
            <div className="flex shrink-0 items-center gap-3">
              <button
                type="button"
                onClick={handleToggleAutoImplement}
                disabled={autoImplementBusy}
                title={
                  autoImplementOn
                    ? "Auto-implement is ON — Proposed ideas get implemented automatically (max 2 at once). Click to stop."
                    : "Auto-implement is OFF. Click to auto-implement Proposed ideas (max 2 at once)."
                }
                className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] transition focus:outline-none focus:ring-2 disabled:cursor-not-allowed disabled:opacity-50 ${
                  autoImplementOn
                    ? "border-emerald-400/50 bg-emerald-400/15 text-emerald-200 hover:bg-emerald-400/25 focus:ring-emerald-400/40"
                    : "border-white/15 bg-white/[0.04] text-[#faf9f6]/55 hover:border-white/30 hover:text-white focus:ring-white/20"
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    autoImplementOn ? "animate-pulse bg-emerald-400" : "bg-[#faf9f6]/40"
                  }`}
                />
                {autoImplementBusy
                  ? "…"
                  : autoImplementOn
                    ? "Auto-implement on"
                    : "Auto-implement off"}
              </button>
              <button
                type="button"
                onClick={refreshIdeas}
                className="shrink-0 text-xs uppercase tracking-[0.2em] text-amber-300/70 transition hover:text-amber-200"
              >
                Refresh
              </button>
            </div>
          </div>

          {/* Generate controls + prompt edit links */}
          <div className="mb-6 flex flex-col items-center gap-3 text-center">
            <form onSubmit={handleGenerate} className="flex items-center gap-2">
              <label className="flex items-center gap-1.5 rounded-full border border-amber-300/30 bg-amber-300/5 px-3 py-2 text-[11px] uppercase tracking-[0.18em] text-amber-200/80">
                <span>How many</span>
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={ideaCount}
                  onChange={(e) => {
                    const n = Math.round(Number(e.target.value));
                    setIdeaCount(Number.isFinite(n) ? Math.max(1, Math.min(20, n)) : 1);
                  }}
                  className="w-12 rounded-md border border-amber-300/30 bg-[#1f1e1d] px-1.5 py-1 text-center text-sm font-semibold text-amber-100 focus:outline-none focus:ring-2 focus:ring-amber-300/40"
                  aria-label="Number of ideas to generate"
                />
              </label>
              <button
                type="submit"
                disabled={isGenerating}
                className="rounded-full border border-amber-300/30 bg-amber-300/10 px-8 py-3.5 text-sm font-semibold uppercase tracking-[0.24em] text-amber-200 transition hover:border-amber-300/60 hover:bg-amber-300/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-amber-300/40 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isGenerating ? "Generating..." : `Generate ${ideaCount} Idea${ideaCount === 1 ? "" : "s"}`}
              </button>
            </form>
            {generateMessage && (
              <p className="text-sm text-amber-300">{generateMessage}</p>
            )}
          </div>

          {ideaActionMsg && (
            <p className="mb-2 text-xs text-amber-300">{ideaActionMsg}</p>
          )}
          {ideaLoadError && (
            <p className="mb-2 text-xs text-orange-300">{ideaLoadError}</p>
          )}

          {activeIdeaCount === 0 ? (
            <p className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-6 text-center text-sm text-[#faf9f6]/40">
              No open ideas — finished experiments are at the bottom.
            </p>
          ) : (
            <div className="space-y-6">
              {ideaGroups
                .filter((group) => group.ideas.length > 0)
                .map((group) => (
                  <div key={group.key}>
                    <h3 className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.24em] text-amber-200/50">
                      {group.label}
                      <span className="text-[#faf9f6]/30">({group.ideas.length})</span>
                    </h3>
                    <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                      {group.ideas.map((idea) => renderIdeaCard(idea))}
                    </ul>
                  </div>
                ))}
            </div>
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
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleToggleAutorun}
                disabled={autorunBusy}
                title={
                  autorunOn
                    ? "Autorun is ON — each finished run auto-launches the next queued idea. Click to stop."
                    : "Autorun is OFF. Click to march through the queue automatically (one run at a time)."
                }
                className={`flex items-center gap-1.5 rounded-full border px-3 py-2 text-xs font-semibold uppercase tracking-[0.2em] transition focus:outline-none focus:ring-2 disabled:cursor-not-allowed disabled:opacity-50 ${
                  autorunOn
                    ? "border-emerald-400/50 bg-emerald-400/15 text-emerald-200 hover:bg-emerald-400/25 focus:ring-emerald-400/40"
                    : "border-white/15 bg-white/[0.04] text-[#faf9f6]/60 hover:border-white/30 hover:text-white focus:ring-white/20"
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    autorunOn ? "animate-pulse bg-emerald-400" : "bg-[#faf9f6]/40"
                  }`}
                />
                {autorunBusy ? "…" : autorunOn ? "Autorun on" : "Autorun off"}
              </button>
              {/* Manual single-run only makes sense when autorun is off — when
                  it's on, the lab-autorun runner agent drains the queue itself. */}
              {!autorunOn && (
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
              )}
            </div>
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
                      {timeInState(idea.updated) && (
                        <span
                          title={
                            idea.status === "running"
                              ? "time running"
                              : "time waiting in queue"
                          }
                          className="font-mono text-[10px] tabular-nums text-[#faf9f6]/45"
                        >
                          ⏱ {timeInState(idea.updated)}
                        </span>
                      )}
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

        {/* ============ SECTION 4 · FINISHED EXPERIMENTS (bottom) ============ */}
        <section className="mt-16 w-full max-w-2xl">
          <div className="mb-5 flex items-end justify-between gap-3 border-b border-emerald-300/20 pb-3">
            <div className="flex items-center gap-3">
              <span className="h-7 w-1 rounded-full bg-emerald-300/70" />
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-[0.28em] text-emerald-200">
                  Finished experiments
                </h2>
                <p className="text-[11px] text-[#faf9f6]/40">
                  Completed A/Bs with the full validation-loss curve, newest first.
                </p>
              </div>
            </div>
            <span className="shrink-0 text-xs uppercase tracking-[0.2em] text-emerald-300/60">
              {finishedIdeas.length}
            </span>
          </div>

          {finishedIdeas.length === 0 ? (
            <p className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-6 text-center text-sm text-[#faf9f6]/40">
              No finished experiments yet.
            </p>
          ) : (
            <ul className="space-y-2">
              {finishedIdeas.map((idea) =>
                renderIdeaCard(idea, <TrainingCurve id={idea.id} />)
              )}
            </ul>
          )}
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
