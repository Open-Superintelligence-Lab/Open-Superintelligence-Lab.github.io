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

const EMPTY_SNAPSHOT: LabSnapshot = {
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

const snapshotPath = path.join(process.cwd(), "public", "data", "lab-snapshot.json");

function loadSnapshot(): LabSnapshot {
  try {
    const raw = fs.readFileSync(snapshotPath, "utf8");
    const parsed = JSON.parse(raw) as Partial<LabSnapshot>;
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
  } catch {
    return EMPTY_SNAPSHOT;
  }
}

const snapshot = loadSnapshot();

const statusClass = (status: string) => {
  const s = status.toLowerCase();
  if (s.includes("done") || s.includes("complete") || s.includes("shipped")) {
    return "text-emerald-400 border-emerald-400/30 bg-emerald-400/10";
  }
  if (s.includes("active") || s.includes("running") || s.includes("planned") || s.includes("queued") || s.includes("proposed")) {
    return "text-cyan-300 border-cyan-300/30 bg-cyan-300/10";
  }
  if (s.includes("pending") || s.includes("writing") || s.includes("review") || s.includes("planning")) {
    return "text-amber-300 border-amber-300/30 bg-amber-300/10";
  }
  if (s.includes("reject") || s.includes("fail") || s.includes("stop") || s.includes("broken") || s.includes("stopped")) {
    return "text-rose-400 border-rose-400/30 bg-rose-400/10";
  }
  return "text-[#faf9f6]/60 border-[#f0eee6]/10 bg-[#f0eee6]/[0.04]";
};

const formatDate = (value: string) => (value ? value.slice(0, 10) : "—");
const formatTimestamp = (value: string) => (value ? value.slice(0, 16).replace("T", " ") : "—");

const summaryCards = [
  { key: "threads", label: "threads", value: snapshot.summary.threads },
  { key: "runs", label: "runs", value: snapshot.summary.runs },
  { key: "queue_items", label: "queue items", value: snapshot.summary.queue_items },
  { key: "decisions", label: "decisions", value: snapshot.summary.decisions },
  { key: "ideas", label: "ideas", value: snapshot.summary.ideas },
];

export const dynamic = "force-static";

export default function LabExperimentsPage() {
  return (
    <main className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]">
      <div className="container mx-auto max-w-6xl px-6 py-16">
        <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/70">Lab</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight md:text-4xl">
          Experiment activity
        </h1>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-[#faf9f6]/70">
          Read-only snapshot of the local experiment registry.
        </p>

        <div className="mt-8 grid grid-cols-2 gap-4 md:grid-cols-5">
          {summaryCards.map((card) => (
            <section key={card.key} className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-4">
              <p className="text-xs uppercase tracking-widest text-[#faf9f6]/40">{card.label}</p>
              <p className="mt-2 text-3xl font-semibold tracking-tight text-cyan-300">{card.value}</p>
            </section>
          ))}
        </div>

        <p className="mt-4 text-xs text-[#faf9f6]/40">
          updated {formatTimestamp(snapshot.generated_at)}
        </p>

        <section className="mt-10">
          <div className="flex items-baseline justify-between gap-4">
            <h2 className="text-xl font-semibold md:text-2xl">Threads</h2>
            <span className="text-xs text-[#faf9f6]/40">{snapshot.threads.length} items</span>
          </div>

          {snapshot.threads.length === 0 ? (
            <p className="mt-4 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6 text-sm text-[#faf9f6]/55">
              No threads available in the snapshot.
            </p>
          ) : (
            <div className="mt-4 space-y-4">
              {snapshot.threads.map((thread) => (
                <article key={thread.name} className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5 md:p-6">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <h3 className="text-lg font-semibold md:text-xl">{thread.title}</h3>
                      <p className="mt-1 text-sm text-[#faf9f6]/55">{thread.name}</p>
                    </div>
                    <span className={`rounded-full border px-3 py-1 text-xs uppercase tracking-wider ${statusClass(thread.status)}`}>
                      {thread.status}
                    </span>
                  </div>
                  <p className="mt-3 text-sm text-[#faf9f6]/50">Created {formatDate(thread.created)}</p>
                </article>
              ))}
            </div>
          )}
        </section>

        <section className="mt-10">
          <div className="flex items-baseline justify-between gap-4">
            <h2 className="text-xl font-semibold md:text-2xl">Recent runs</h2>
            <span className="text-xs text-[#faf9f6]/40">{snapshot.runs.length} shown</span>
          </div>

          {snapshot.runs.length === 0 ? (
            <p className="mt-4 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6 text-sm text-[#faf9f6]/55">
              No runs available in the snapshot.
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
                  {snapshot.runs.map((run) => (
                    <tr key={`${run.name}-${run.created}`} className="border-b border-[#f0eee6]/5 last:border-b-0">
                      <td className="px-4 py-4 font-medium text-[#faf9f6]">{run.name}</td>
                      <td className="px-4 py-4 text-[#faf9f6]/65">{run.command_summary}</td>
                      <td className="px-4 py-4">
                        <span className={`rounded-full border px-3 py-1 text-[11px] uppercase tracking-wider ${statusClass(run.status)}`}>
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
            <h2 className="text-xl font-semibold md:text-2xl">Decisions</h2>
            <span className="text-xs text-[#faf9f6]/40">{snapshot.decisions.length} items</span>
          </div>

          {snapshot.decisions.length === 0 ? (
            <p className="mt-4 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6 text-sm text-[#faf9f6]/55">
              No decisions available in the snapshot.
            </p>
          ) : (
            <div className="mt-4 space-y-4">
              {snapshot.decisions.map((decision, index) => (
                <article key={`${decision.created}-${index}`} className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5 md:p-6">
                  <p className="text-xs uppercase tracking-widest text-[#faf9f6]/40">{formatDate(decision.created)}</p>
                  <p className="mt-3 text-sm leading-relaxed text-[#faf9f6]/80">{decision.decision}</p>
                </article>
              ))}
            </div>
          )}
        </section>

        <p className="mt-10 text-xs text-[#faf9f6]/40">
          Snapshot from the lab's local experiment registry — updated when the site deploys.
        </p>
      </div>
    </main>
  );
}
