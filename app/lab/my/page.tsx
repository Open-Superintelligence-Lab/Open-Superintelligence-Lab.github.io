import Link from "next/link";
import fs from "fs";
import path from "path";

type CountSummary = {
  threads: number;
  runs: number;
  queue_items: number;
  decisions: number;
  ideas: number;
};

type ThreadItem = {
  name: string;
  title: string;
  status: string;
  created: string;
};

type RunMetric = {
  label: string;
  value: string;
} | null;

type RunItem = {
  name: string;
  command_summary: string;
  status: string;
  metric: RunMetric;
  created: string;
};

type DecisionItem = {
  decision: string;
  created: string;
};

type LabSnapshot = {
  generated_at: string;
  summary: CountSummary;
  threads: ThreadItem[];
  runs: RunItem[];
  decisions: DecisionItem[];
};

type QueueItem = {
  id: string;
  title: string;
  summary: string;
  status: string;
  metric: string | null;
  owner: string | null;
};

type QueueSnapshot = {
  generated_at: string;
  items: QueueItem[];
  found: boolean;
};

const EMPTY_LAB_SNAPSHOT: LabSnapshot = {
  generated_at: "",
  summary: {
    threads: 0,
    runs: 0,
    queue_items: 0,
    decisions: 0,
    ideas: 0,
  },
  threads: [],
  runs: [],
  decisions: [],
};

const EMPTY_QUEUE_SNAPSHOT: QueueSnapshot = {
  generated_at: "",
  items: [],
  found: false,
};

const labSnapshotPath = path.join(process.cwd(), "public", "data", "lab-snapshot.json");
const queueSnapshotPath = path.join(process.cwd(), "public", "data", "queue-snapshot.json");

function readJson<T>(filePath: string, fallback: T): T {
  try {
    const raw = fs.readFileSync(filePath, "utf8");
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function loadLabSnapshot(): LabSnapshot {
  const parsed = readJson<Partial<LabSnapshot>>(labSnapshotPath, {});
  return {
    generated_at: typeof parsed.generated_at === "string" ? parsed.generated_at : "",
    summary: {
      threads: parsed.summary?.threads ?? 0,
      runs: parsed.summary?.runs ?? 0,
      queue_items: parsed.summary?.queue_items ?? 0,
      decisions: parsed.summary?.decisions ?? 0,
      ideas: parsed.summary?.ideas ?? 0,
    },
    threads: Array.isArray(parsed.threads) ? parsed.threads : [],
    runs: Array.isArray(parsed.runs) ? parsed.runs : [],
    decisions: Array.isArray(parsed.decisions) ? parsed.decisions : [],
  };
}

function normalizeText(value: unknown, fallback = ""): string {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed || fallback;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  return fallback;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function metricText(value: unknown): string | null {
  if (value == null) return null;
  if (typeof value === "string") {
    const text = value.trim();
    return text || null;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  if (!isRecord(value)) return null;

  const label =
    normalizeText(value.label) ||
    normalizeText(value.name) ||
    normalizeText(value.metric) ||
    normalizeText(value.kind);
  const rawValue =
    normalizeText(value.value) ||
    normalizeText(value.amount) ||
    normalizeText(value.text) ||
    normalizeText(value.result);

  if (label && rawValue) return `${label}: ${rawValue}`;
  if (rawValue) return rawValue;
  if (label) return label;
  return null;
}

function statusClass(status: string) {
  const s = status.toLowerCase();
  if (s.includes("done") || s.includes("complete") || s.includes("shipped")) {
    return "text-emerald-400 border-emerald-400/30 bg-emerald-400/10";
  }
  if (
    s.includes("active") ||
    s.includes("running") ||
    s.includes("queued") ||
    s.includes("claimed") ||
    s.includes("planned") ||
    s.includes("planning")
  ) {
    return "text-cyan-300 border-cyan-300/30 bg-cyan-300/10";
  }
  if (s.includes("pending") || s.includes("review") || s.includes("draft")) {
    return "text-amber-300 border-amber-300/30 bg-amber-300/10";
  }
  if (s.includes("fail") || s.includes("stop") || s.includes("broken") || s.includes("blocked")) {
    return "text-rose-400 border-rose-400/30 bg-rose-400/10";
  }
  return "text-[#faf9f6]/60 border-[#f0eee6]/10 bg-[#f0eee6]/[0.04]";
}

function isNowThread(status: string) {
  const s = status.toLowerCase();
  return s.includes("active") || s.includes("running") || s.includes("planning") || s.includes("queued");
}

function formatDate(value: string) {
  return value ? value.slice(0, 10) : "—";
}

function formatTimestamp(value: string) {
  return value ? value.slice(0, 16).replace("T", " ") : "—";
}

function makeQueueItem(raw: unknown, forcedStatus?: string): QueueItem | null {
  if (!isRecord(raw)) return null;

  const status = normalizeText(raw.status ?? raw.state ?? forcedStatus ?? "", "").toLowerCase();
  const title =
    normalizeText(raw.title) ||
    normalizeText(raw.name) ||
    normalizeText(raw.spec) ||
    normalizeText(raw.summary) ||
    normalizeText(raw.goal) ||
    normalizeText(raw.label) ||
    normalizeText(raw.id) ||
    "Untitled";

  const summary =
    normalizeText(raw.summary) ||
    normalizeText(raw.description) ||
    normalizeText(raw.spec) ||
    normalizeText(raw.input) ||
    normalizeText(raw.prompt) ||
    normalizeText(raw.note) ||
    "";

  const metric = metricText(
    raw.key_metric ?? raw.keyMetric ?? raw.metric ?? raw.result ?? raw.outcome ?? raw.final_metric
  );

  return {
    id: normalizeText(raw.id) || normalizeText(raw.name) || title,
    title,
    summary,
    status: status || "queued",
    metric,
    owner:
      normalizeText(raw.owner) ||
      normalizeText(raw.claimed_by) ||
      normalizeText(raw.assignee) ||
      normalizeText(raw.worker) ||
      null,
  };
}

function loadQueueSnapshot(): QueueSnapshot {
  try {
    if (!fs.existsSync(queueSnapshotPath)) return EMPTY_QUEUE_SNAPSHOT;

    const raw = fs.readFileSync(queueSnapshotPath, "utf8");
    const parsed = JSON.parse(raw) as unknown;
    const items: QueueItem[] = [];

    if (Array.isArray(parsed)) {
      for (const item of parsed) {
        const normalized = makeQueueItem(item);
        if (normalized) items.push(normalized);
      }
    } else if (isRecord(parsed)) {
      const statusBuckets = [
        ["queued", parsed.queued],
        ["claimed", parsed.claimed],
        ["done", parsed.done],
      ] as const;

      let collected = false;
      for (const [bucket, value] of statusBuckets) {
        if (!Array.isArray(value)) continue;
        collected = true;
        for (const item of value) {
          const normalized = makeQueueItem(item, bucket);
          if (normalized) items.push(normalized);
        }
      }

      if (!collected) {
        const arrays = [
          parsed.items,
          parsed.specs,
          parsed.queue,
          parsed.results,
          parsed.jobs,
        ];
        for (const value of arrays) {
          if (!Array.isArray(value)) continue;
          for (const item of value) {
            const normalized = makeQueueItem(item);
            if (normalized) items.push(normalized);
          }
        }
      }
    }

    return {
      generated_at: isRecord(parsed) ? normalizeText(parsed.generated_at, "") : "",
      items,
      found: true,
    };
  } catch {
    return EMPTY_QUEUE_SNAPSHOT;
  }
}

const labSnapshot = loadLabSnapshot();
const queueSnapshot = loadQueueSnapshot();

const nowThreads = labSnapshot.threads.filter((thread) => isNowThread(thread.status));
const nowStrip = nowThreads.length > 0 ? nowThreads : labSnapshot.threads.slice(0, 6);

const queueBuckets = ["queued", "claimed", "done"] as const;
const queueByStatus = new Map<string, QueueItem[]>(
  queueBuckets.map((status) => [
    status,
    queueSnapshot.items.filter((item) => item.status.toLowerCase().includes(status)),
  ])
);

export const dynamic = "force-static";

export default function MyLabPage() {
  return (
    <main className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]">
      <div className="container mx-auto max-w-6xl px-6 py-16">
        <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/70">Lab</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight md:text-5xl">My lab</h1>
        <p className="mt-4 max-w-3xl text-sm leading-relaxed text-[#faf9f6]/70">
          Personal view only. Local exports, not a public account system.
        </p>
        <p className="mt-3 text-sm text-[#faf9f6]/55">
          <Link href="/lab/my/launch-codex" className="text-cyan-300 transition hover:text-cyan-200">
            Open the Codex launcher
          </Link>
        </p>

        <div className="mt-8 rounded-xl border border-amber-300/20 bg-amber-300/[0.04] px-4 py-3 text-sm text-amber-100/90">
          This dashboard reads exported local data at build time. It is a private working view, not a
          public account system.
        </div>

        <section className="mt-10">
          <div className="flex items-baseline justify-between gap-4">
            <h2 className="text-xl font-semibold md:text-2xl">Now</h2>
            <span className="text-xs text-[#faf9f6]/40">{nowStrip.length} shown</span>
          </div>

          {nowStrip.length === 0 ? (
            <p className="mt-4 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5 text-sm text-[#faf9f6]/55">
              No active threads in the lab snapshot.
            </p>
          ) : (
            <div className="mt-4 overflow-x-auto pb-1">
              <div className="flex min-w-full gap-3">
                {nowStrip.map((thread) => (
                  <article
                    key={thread.name}
                    className="min-w-[16rem] flex-1 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-4"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="text-xs uppercase tracking-[0.2em] text-[#faf9f6]/40">Thread</p>
                        <h3 className="mt-2 text-sm font-semibold leading-snug text-[#faf9f6]">{thread.title}</h3>
                      </div>
                      <span
                        className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wider ${statusClass(thread.status)}`}
                      >
                        {thread.status}
                      </span>
                    </div>
                    <p className="mt-3 text-xs text-[#faf9f6]/55">{thread.name}</p>
                    <p className="mt-2 text-xs text-[#faf9f6]/45">Created {formatDate(thread.created)}</p>
                  </article>
                ))}
              </div>
            </div>
          )}
        </section>

        <section className="mt-10">
          <div className="flex items-baseline justify-between gap-4">
            <h2 className="text-xl font-semibold md:text-2xl">My queue</h2>
            <span className="text-xs text-[#faf9f6]/40">
              {queueSnapshot.found ? `${queueSnapshot.items.length} specs` : "snapshot missing"}
            </span>
          </div>

          {!queueSnapshot.found ? (
            <p className="mt-4 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5 text-sm text-[#faf9f6]/55">
              Queue snapshot is not available yet.
            </p>
          ) : queueSnapshot.items.length === 0 ? (
            <p className="mt-4 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5 text-sm text-[#faf9f6]/55">
              No queue specs were exported into the snapshot.
            </p>
          ) : (
            <div className="mt-4 grid gap-4 lg:grid-cols-3">
              {queueBuckets.map((status) => {
                const items = queueByStatus.get(status) ?? [];
                return (
                  <section
                    key={status}
                    className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-4"
                  >
                    <div className="flex items-baseline justify-between gap-3">
                      <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-[#faf9f6]/55">
                        {status}
                      </h3>
                      <span className="text-xs text-[#faf9f6]/40">{items.length}</span>
                    </div>

                    {items.length === 0 ? (
                      <p className="mt-3 text-sm text-[#faf9f6]/45">No {status} specs.</p>
                    ) : (
                      <div className="mt-3 space-y-3">
                        {items.map((item) => (
                          <article
                            key={item.id}
                            className="rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-4"
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div className="min-w-0">
                                <h4 className="text-sm font-semibold leading-snug text-[#faf9f6]">
                                  {item.title}
                                </h4>
                                {item.summary && (
                                  <p className="mt-2 text-xs leading-relaxed text-[#faf9f6]/60">
                                    {item.summary}
                                  </p>
                                )}
                              </div>
                              <span
                                className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wider ${statusClass(item.status)}`}
                              >
                                {item.status}
                              </span>
                            </div>

                            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-[#faf9f6]/50">
                              {item.owner && <span>Owner: {item.owner}</span>}
                              {status === "done" && (
                                <span className="font-mono text-cyan-300/90">
                                  {item.metric ?? "Metric unavailable"}
                                </span>
                              )}
                            </div>
                          </article>
                        ))}
                      </div>
                    )}
                  </section>
                );
              })}
            </div>
          )}
        </section>

        <section className="mt-10">
          <div className="flex items-baseline justify-between gap-4">
            <h2 className="text-xl font-semibold md:text-2xl">Recent runs</h2>
            <span className="text-xs text-[#faf9f6]/40">{labSnapshot.runs.length} shown</span>
          </div>

          {labSnapshot.runs.length === 0 ? (
            <p className="mt-4 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5 text-sm text-[#faf9f6]/55">
              No runs available in the lab snapshot.
            </p>
          ) : (
            <div className="mt-4 overflow-x-auto rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02]">
              <table className="min-w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-[#f0eee6]/10 text-xs uppercase tracking-widest text-[#faf9f6]/50">
                    <th className="px-4 py-3">Run</th>
                    <th className="px-4 py-3">Summary</th>
                    <th className="px-4 py-3">Status</th>
                    <th className="px-4 py-3">Metric</th>
                    <th className="px-4 py-3">Date</th>
                  </tr>
                </thead>
                <tbody>
                  {labSnapshot.runs.map((run) => (
                    <tr key={`${run.name}-${run.created}`} className="border-b border-[#f0eee6]/5 last:border-b-0">
                      <td className="px-4 py-4 font-medium text-[#faf9f6]">{run.name}</td>
                      <td className="px-4 py-4 text-[#faf9f6]/65">{run.command_summary}</td>
                      <td className="px-4 py-4">
                        <span
                          className={`rounded-full border px-3 py-1 text-[11px] uppercase tracking-wider ${statusClass(run.status)}`}
                        >
                          {run.status}
                        </span>
                      </td>
                      <td className="px-4 py-4 font-mono text-[#faf9f6]/80">
                        {run.metric ? `${run.metric.label}: ${run.metric.value}` : "—"}
                      </td>
                      <td className="px-4 py-4 text-[#faf9f6]/50">{formatDate(run.created)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="mt-10">
          <div className="flex items-baseline justify-between gap-4">
            <h2 className="text-xl font-semibold md:text-2xl">Decisions log</h2>
            <span className="text-xs text-[#faf9f6]/40">{labSnapshot.decisions.length} items</span>
          </div>

          {labSnapshot.decisions.length === 0 ? (
            <p className="mt-4 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5 text-sm text-[#faf9f6]/55">
              No decisions were exported into the snapshot.
            </p>
          ) : (
            <div className="mt-4 space-y-3">
              {labSnapshot.decisions.map((decision) => (
                <article
                  key={`${decision.created}-${decision.decision}`}
                  className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-4"
                >
                  <p className="text-xs uppercase tracking-[0.2em] text-[#faf9f6]/40">
                    {formatDate(decision.created)}
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-[#faf9f6]/82">{decision.decision}</p>
                </article>
              ))}
            </div>
          )}
        </section>

        {labSnapshot.generated_at && (
          <p className="mt-8 text-xs text-[#faf9f6]/40">
            updated {formatTimestamp(labSnapshot.generated_at)}
          </p>
        )}
      </div>
    </main>
  );
}
