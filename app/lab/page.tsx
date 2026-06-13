import fs from "fs";
import path from "path";
import Link from "next/link";
import { getAllProposals } from "@/lib/proposals";

type Experiment = { goal_id: string; label: string; result: string };
type Curves = {
  tier: string;
  steps: number[];
  series: Record<string, number[]>;
  finals: Record<string, number>;
  noise_band: number;
};
type Paper = {
  id: string;
  title: string;
  status: string;
  authors: string[];
  created: string;
  experiments: Experiment[];
  curves: Curves | null;
};
type Summary = { generated_at: string; papers: Paper[]; n_experiments: number };

type PublicGoal = {
  id: string;
  title: string;
  status: string;
  description: string;
  why: string;
  milestones: { label: string; done: boolean }[];
};
type GoalsData = {
  mission: string;
  public_goals: PublicGoal[];
  strategy_ladder?: string[];
  generated_at: string;
};

type Problem = {
  id: string;
  title: string;
  status: string;
  answer: string | null;
  paper: string | null;
};
type ProblemsData = { generated_at: string; problems: Problem[] };

type QueueSpec = {
  id: string;
  title: string;
  status: string;
  gpu_vram_gb: number | null;
  hours: number | null;
};
type QueueResult = {
  spec_id: string;
  worker: string;
  finished: string;
  exit_status: string;
  metrics: Record<string, string | number>;
};
type QueueSnapshot = { generated_at: string; specs: QueueSpec[]; results: QueueResult[] };

type PipelineIdea = { id: string; status: string; round: number; updated: string };
type PipelineSnapshot = {
  generated_at: string;
  gate_summary: Record<string, number>;
  feeding_gpu: number;
  gpu_idle_risk: boolean;
  in_flight: PipelineIdea[];
};

type RunMetric = { label: string; value: string } | null;
type ActivityThread = { name: string; title: string; status: string; created: string };
type ActivityRun = {
  name: string;
  command_summary: string;
  status: string;
  metric: RunMetric;
  created: string;
};
type ActivityDecision = { decision: string; created: string };
type ActivitySnapshot = {
  generated_at: string;
  summary: {
    threads: number;
    runs: number;
    queue_items: number;
    decisions: number;
    ideas: number;
  };
  threads: ActivityThread[];
  runs: ActivityRun[];
  decisions: ActivityDecision[];
};

type LeaderboardRow = {
  rank: number;
  val_loss: number;
  delta: number;
  run: string;
  summary: string;
  date: string;
};
type LeaderboardTier = { name: string; description: string; rows: LeaderboardRow[] };
type LeaderboardData = { generated_at: string; tiers: LeaderboardTier[] };

const EMPTY_SUMMARY: Summary = { generated_at: "", papers: [], n_experiments: 0 };
const EMPTY_GOALS: GoalsData = { mission: "", public_goals: [], strategy_ladder: [], generated_at: "" };
const EMPTY_PROBLEMS: ProblemsData = { generated_at: "", problems: [] };
const EMPTY_QUEUE: QueueSnapshot = { generated_at: "", specs: [], results: [] };
const EMPTY_PIPELINE: PipelineSnapshot = {
  generated_at: "",
  gate_summary: {},
  feeding_gpu: 0,
  gpu_idle_risk: true,
  in_flight: [],
};
const EMPTY_ACTIVITY: ActivitySnapshot = {
  generated_at: "",
  summary: { threads: 0, runs: 0, queue_items: 0, decisions: 0, ideas: 0 },
  threads: [],
  runs: [],
  decisions: [],
};
const EMPTY_LEADERBOARD: LeaderboardData = { generated_at: "", tiers: [] };

const labGroups = [
  {
    title: "Active work",
    links: [
      { href: "/lab/queue", label: "Queue", description: "Specs waiting for GPUs." },
      { href: "/lab/pipeline", label: "Pipeline", description: "Idea gates and GPU feeding state." },
      { href: "/lab/experiments", label: "Experiments", description: "Threads, runs, and decisions." },
    ],
  },
  {
    title: "Results",
    links: [
      { href: "/lab/research", label: "Papers", description: "Living manuscripts and PDFs." },
      { href: "/lab/leaderboard", label: "Leaderboard", description: "Best validation loss per tier." },
      { href: "/lab/problems", label: "Problems", description: "Open questions and answers." },
      { href: "/lab/ideas", label: "Ideas", description: "Idea flow from taste to done." },
    ],
  },
  {
    title: "Direction",
    links: [
      { href: "/lab/goals", label: "Goals", description: "Mission, strategy, and milestones." },
      { href: "/proposals", label: "Proposals", description: "AI plans for human approval." },
      { href: "/contribute", label: "Contribute", description: "How people and machines plug in." },
    ],
  },
  {
    title: "Private",
    links: [{ href: "/lab/my", label: "My lab", description: "Local private working view." }],
  },
];

function loadJson<T>(relativePath: string, fallback: T): T {
  try {
    const raw = fs.readFileSync(path.join(process.cwd(), relativePath), "utf8");
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function formatTimestamp(value: string) {
  return value ? value.slice(0, 16).replace("T", " ") : "not exported";
}

function formatDate(value: string) {
  return value ? value.slice(0, 10) : "unknown";
}

function compactDate(value: string) {
  if (!value) return "unknown";
  if (value.includes("T")) return value.slice(5, 16).replace("T", " ");
  return value.slice(5, 10);
}

function isDone(status: string) {
  const s = status.toLowerCase();
  return s.includes("done") || s.includes("complete") || s.includes("finished") || s.includes("success") || s.includes("approved");
}

function isBad(status: string) {
  const s = status.toLowerCase();
  return s.includes("fail") || s.includes("worse") || s.includes("reject") || s.includes("stopped") || s.includes("broken");
}

function statusTone(status: string) {
  const s = status.toLowerCase();
  if (s.includes("win") || s.includes("done") || s.includes("complete") || s.includes("success") || s.includes("approved")) {
    return "border-emerald-400/30 bg-emerald-400/10 text-emerald-300";
  }
  if (s.includes("running") || s.includes("active") || s.includes("queued") || s.includes("needs-run")) {
    return "border-cyan-300/30 bg-cyan-300/10 text-cyan-300";
  }
  if (s.includes("pending") || s.includes("review") || s.includes("planning") || s.includes("forming") || s.includes("draft")) {
    return "border-amber-300/30 bg-amber-300/10 text-amber-300";
  }
  if (isBad(s) || s.includes("risk")) {
    return "border-rose-400/30 bg-rose-400/10 text-rose-300";
  }
  return "border-[#f0eee6]/10 bg-[#f0eee6]/[0.04] text-[#faf9f6]/65";
}

function verdictColor(label: string) {
  const l = label.toLowerCase();
  if (l.includes("win")) return "text-emerald-400 border-emerald-400/30 bg-emerald-400/10";
  if (l.includes("pending")) return "text-amber-300 border-amber-300/30 bg-amber-300/10";
  if (l.includes("diverged") || l.includes("worse")) return "text-rose-400 border-rose-400/30 bg-rose-400/10";
  if (l.includes("null")) return "text-zinc-400 border-zinc-400/30 bg-zinc-400/10";
  return "text-cyan-300 border-cyan-300/30 bg-cyan-300/10";
}

function verdictWord(label: string) {
  const l = label.toLowerCase();
  if (l.includes("pending")) return "pending";
  if (l.includes("win")) return "win";
  if (l.includes("diverged")) return "diverged";
  if (l.includes("worse")) return "worse";
  if (l.includes("null")) return "null";
  return "info";
}

function formatMetricValue(value: string | number) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/\.?0+$/, "");
  }
  return value;
}

function metricSummary(metrics: Record<string, string | number>) {
  const priority = ["val_loss", "val_accuracy", "tokens_per_sec", "train_loss", "val_perplexity"];
  return priority
    .filter((key) => key in metrics)
    .map((key) => `${key}: ${formatMetricValue(metrics[key])}`)
    .join("  ");
}

function deltaText(delta: number) {
  if (!Number.isFinite(delta)) return "unknown";
  return `${delta > 0 ? "+" : ""}${delta.toFixed(4)}`;
}

function deltaTone(delta: number) {
  if (delta < 0) return "text-emerald-300";
  if (delta > 0) return "text-rose-300";
  return "text-[#faf9f6]/55";
}

function MetricCard({ label, value, detail }: { label: string; value: string | number; detail: string }) {
  return (
    <section className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-4">
      <p className="text-xs uppercase tracking-[0.2em] text-[#faf9f6]/40">{label}</p>
      <p className="mt-2 text-3xl font-semibold tracking-tight text-cyan-300">{value}</p>
      <p className="mt-2 text-xs leading-relaxed text-[#faf9f6]/55">{detail}</p>
    </section>
  );
}

function SectionHeading({
  eyebrow,
  title,
  body,
  href,
  linkLabel,
}: {
  eyebrow: string;
  title: string;
  body?: string;
  href?: string;
  linkLabel?: string;
}) {
  return (
    <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
      <div>
        <p className="text-xs uppercase tracking-[0.28em] text-cyan-300/70">{eyebrow}</p>
        <h2 className="mt-2 text-2xl font-semibold tracking-tight">{title}</h2>
        {body && <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[#faf9f6]/55">{body}</p>}
      </div>
      {href && linkLabel && (
        <Link href={href} className="w-fit text-sm font-medium text-cyan-300 transition hover:text-cyan-200">
          {linkLabel}
        </Link>
      )}
    </div>
  );
}

function ActionCard({
  href,
  eyebrow,
  title,
  body,
  meta,
}: {
  href: string;
  eyebrow: string;
  title: string;
  body: string;
  meta: string;
}) {
  return (
    <Link
      href={href}
      className="group rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5 transition hover:border-cyan-300/35 hover:bg-cyan-300/[0.04]"
    >
      <p className="text-xs uppercase tracking-[0.22em] text-cyan-300/70">{eyebrow}</p>
      <h3 className="mt-3 text-lg font-semibold leading-snug text-[#faf9f6]">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-[#faf9f6]/68">{body}</p>
      <div className="mt-5 flex items-center justify-between gap-3 text-sm">
        <span className="text-[#faf9f6]/45">{meta}</span>
        <span className="text-cyan-300 transition group-hover:text-cyan-200">Open</span>
      </div>
    </Link>
  );
}

function HealthCard({
  title,
  value,
  body,
  tone,
}: {
  title: string;
  value: string;
  body: string;
  tone: "good" | "watch" | "bad";
}) {
  const toneClass =
    tone === "good"
      ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
      : tone === "watch"
        ? "border-amber-300/30 bg-amber-300/10 text-amber-300"
        : "border-rose-400/30 bg-rose-400/10 text-rose-300";

  return (
    <section className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5">
      <div className={`inline-flex rounded-full border px-3 py-1 text-xs uppercase tracking-[0.18em] ${toneClass}`}>
        {value}
      </div>
      <h3 className="mt-4 text-lg font-semibold">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-[#faf9f6]/65">{body}</p>
    </section>
  );
}

function ProgressBar({ value, max }: { value: number; max: number }) {
  return (
    <div className="h-2 overflow-hidden rounded-full bg-[#f0eee6]/10">
      <div
        className="h-full rounded-full bg-cyan-300/70"
        style={{ width: `${value === 0 ? 0 : Math.max(8, (value / Math.max(max, 1)) * 100)}%` }}
      />
    </div>
  );
}

function LossChart({ curves }: { curves: Curves }) {
  const width = 720;
  const height = 300;
  const padding = 44;
  const names = Object.keys(curves.series);
  const allValues = names.flatMap((name) => curves.series[name]).filter((value) => Number.isFinite(value));
  const steps = curves.steps;

  if (names.length === 0 || steps.length < 2 || allValues.length === 0) {
    return <p className="text-sm text-[#faf9f6]/50">No curve points exported.</p>;
  }

  const yMin = Math.min(...allValues);
  const yMax = Math.max(...allValues);
  const ySpread = Math.max(yMax - yMin, 1);
  const xStart = steps[0];
  const xEnd = steps[steps.length - 1];
  const xSpread = Math.max(xEnd - xStart, 1);
  const x = (step: number) => padding + ((step - xStart) / xSpread) * (width - 2 * padding);
  const y = (value: number) => height - padding - ((value - yMin) / ySpread) * (height - 2 * padding);
  const colors = ["#71717a", "#34d399", "#f59e0b", "#f472b6"];

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="Validation loss curves">
      {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
        const value = yMin + tick * (yMax - yMin);
        return (
          <g key={tick}>
            <line x1={padding} x2={width - padding} y1={y(value)} y2={y(value)} stroke="#f0eee6" strokeOpacity="0.08" />
            <text x={padding - 6} y={y(value) + 4} textAnchor="end" fontSize="11" fill="#faf9f6" fillOpacity="0.5">
              {value.toFixed(1)}
            </text>
          </g>
        );
      })}
      {steps.map((step) => (
        <text key={step} x={x(step)} y={height - padding + 16} textAnchor="middle" fontSize="10" fill="#faf9f6" fillOpacity="0.4">
          {step}
        </text>
      ))}
      {names.map((name, index) => (
        <polyline
          key={name}
          fill="none"
          stroke={colors[index % colors.length]}
          strokeWidth="2"
          strokeDasharray={index === 0 ? "" : "6 4"}
          points={curves.series[name].map((value, stepIndex) => `${x(steps[stepIndex])},${y(value)}`).join(" ")}
        />
      ))}
      {names.map((name, index) => (
        <g key={name} transform={`translate(${padding + 8 + index * 230}, ${padding - 18})`}>
          <line x1="0" x2="22" y1="0" y2="0" stroke={colors[index % colors.length]} strokeWidth="2" strokeDasharray={index === 0 ? "" : "6 4"} />
          <text x="28" y="4" fontSize="12" fill="#faf9f6" fillOpacity="0.8">
            {name.replaceAll("_", " ")}
          </text>
        </g>
      ))}
    </svg>
  );
}

const summary = loadJson<Summary>("public/data/lab/summary.json", EMPTY_SUMMARY);
const goals = loadJson<GoalsData>("public/data/lab/goals.json", EMPTY_GOALS);
const problems = loadJson<ProblemsData>("public/data/lab/problems.json", EMPTY_PROBLEMS);
const queue = loadJson<QueueSnapshot>("public/data/queue-snapshot.json", EMPTY_QUEUE);
const pipeline = loadJson<PipelineSnapshot>("public/data/pipeline-snapshot.json", EMPTY_PIPELINE);
const activity = loadJson<ActivitySnapshot>("public/data/lab-snapshot.json", EMPTY_ACTIVITY);
const leaderboard = loadJson<LeaderboardData>("public/data/lab/leaderboard.json", EMPTY_LEADERBOARD);
const proposals = getAllProposals();

const papers = [...summary.papers].sort((a, b) => b.created.localeCompare(a.created));
const highlightedPaper = papers[0] ?? null;
const queuedSpecs = queue.specs.filter((spec) => !isDone(spec.status));
const doneSpecs = queue.specs.filter((spec) => isDone(spec.status));
const nextSpec = queuedSpecs[0] ?? queue.specs[0] ?? null;
const latestResult = [...queue.results].sort((a, b) => b.finished.localeCompare(a.finished))[0] ?? null;
const openProblems = problems.problems.filter((problem) => !problem.status.toLowerCase().includes("answered"));
const activeGoals = goals.public_goals.filter((goal) => !isDone(goal.status));
const totalMilestones = goals.public_goals.reduce((sum, goal) => sum + goal.milestones.length, 0);
const doneMilestones = goals.public_goals.reduce(
  (sum, goal) => sum + goal.milestones.filter((milestone) => milestone.done).length,
  0
);
const activeThreads = activity.threads.filter((thread) => !isDone(thread.status) && !isBad(thread.status)).slice(0, 4);
const recentRuns = [...activity.runs].sort((a, b) => b.created.localeCompare(a.created)).slice(0, 5);
const recentDecisions = [...activity.decisions].sort((a, b) => b.created.localeCompare(a.created)).slice(0, 4);
const pendingProposals = proposals.filter((proposal) => proposal.status === "draft" || proposal.status === "changes-requested");
const approvedProposals = proposals.filter((proposal) => proposal.status === "approved");

const gateEntries = [
  ["taste", "Taste"],
  ["definition", "Definition"],
  ["code", "Code"],
  ["needs_run", "Needs run"],
  ["running", "Running"],
  ["done", "Done"],
  ["rejected", "Rejected"],
].map(([key, label]) => ({ key, label, value: pipeline.gate_summary[key] ?? 0 }));
const maxGate = Math.max(...gateEntries.map((entry) => entry.value), 1);
const totalGateItems = gateEntries.reduce((sum, entry) => sum + entry.value, 0);
const bottleneck = [...gateEntries].sort((a, b) => b.value - a.value)[0];
const inFlight = [...pipeline.in_flight].sort((a, b) => b.updated.localeCompare(a.updated)).slice(0, 8);
const runningIdeas = pipeline.in_flight.filter((idea) => idea.status.toLowerCase() === "running");
const needsRunIdeas = pipeline.in_flight.filter((idea) => idea.status.toLowerCase() === "needs-run");

const topLeaderboard = leaderboard.tiers
  .map((tier) => ({ tier: tier.name, row: tier.rows[0] }))
  .filter((entry): entry is { tier: string; row: LeaderboardRow } => Boolean(entry.row))
  .slice(0, 5);

const gpuState = pipeline.gpu_idle_risk ? "GPU idle risk" : "GPU fed";
const gpuStateClass = pipeline.gpu_idle_risk
  ? "border-rose-400/30 bg-rose-400/10 text-rose-300"
  : "border-emerald-400/30 bg-emerald-400/10 text-emerald-300";
const computeMeta = nextSpec
  ? `${nextSpec.gpu_vram_gb ?? "unknown"}GB VRAM, about ${nextSpec.hours ?? "unknown"}h`
  : "No public spec queued";
const labHealth = [
  {
    title: "Compute feed",
    value: gpuState,
    tone: pipeline.gpu_idle_risk ? ("bad" as const) : ("good" as const),
    body: `${pipeline.feeding_gpu} ideas are feeding GPUs; ${runningIdeas.length} are marked running.`,
  },
  {
    title: "Queue readiness",
    value: queuedSpecs.length > 0 ? `${queuedSpecs.length} ready` : "empty",
    tone: queuedSpecs.length > 0 ? ("good" as const) : ("watch" as const),
    body: queuedSpecs.length > 0 ? `${computeMeta} for the next public spec.` : "No exported queue spec is ready to run.",
  },
  {
    title: "Human approval",
    value: pendingProposals.length > 0 ? `${pendingProposals.length} pending` : `${approvedProposals.length} approved`,
    tone: pendingProposals.length > 0 ? ("watch" as const) : ("good" as const),
    body:
      pendingProposals.length > 0
        ? "A proposal needs approval or requested changes."
        : "No pending proposal was exported.",
  },
  {
    title: "Research bottleneck",
    value: bottleneck ? `${bottleneck.label}: ${bottleneck.value}` : "unknown",
    tone: bottleneck && bottleneck.key === "code" ? ("watch" as const) : ("good" as const),
    body: `${totalGateItems} idea-gate items exported from the pipeline snapshot.`,
  },
];

export const dynamic = "force-static";

export default function Lab() {
  return (
    <main className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]">
      <div className="container mx-auto max-w-7xl px-6 py-16">
        <section className="rounded-2xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.025] p-6 md:p-8">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-4xl">
              <p className="text-xs uppercase tracking-[0.32em] text-cyan-300/75">Lab command center</p>
              <h1 className="mt-3 text-3xl font-semibold tracking-tight md:text-5xl">
                Start real open LLM research here.
              </h1>
              <p className="mt-4 max-w-3xl text-sm leading-relaxed text-[#faf9f6]/70 md:text-base">
                One page for the current research artifact, compute queue, human approvals, pipeline bottlenecks,
                recent decisions, and the fastest path into the lab.
              </p>
            </div>
            <div className={`w-fit rounded-full border px-4 py-2 text-sm ${gpuStateClass}`}>
              {gpuState} - {pipeline.feeding_gpu} feeding
            </div>
          </div>

          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-6">
            <MetricCard label="Papers" value={summary.papers.length} detail={`${summary.n_experiments} linked experiments`} />
            <MetricCard label="Queue" value={queue.specs.length} detail={`${queuedSpecs.length} waiting, ${doneSpecs.length} done`} />
            <MetricCard label="Runs" value={activity.summary.runs || queue.results.length} detail={`${activity.summary.threads} threads exported`} />
            <MetricCard label="Ideas" value={activity.summary.ideas || pipeline.in_flight.length} detail={`${needsRunIdeas.length} need runs`} />
            <MetricCard label="Problems" value={problems.problems.length} detail={`${openProblems.length} open or active`} />
            <MetricCard label="Goals" value={`${doneMilestones}/${totalMilestones}`} detail="milestones done" />
          </div>

          <div className="mt-5 flex flex-wrap gap-x-5 gap-y-2 text-xs text-[#faf9f6]/40">
            <span>research {formatTimestamp(summary.generated_at)}</span>
            <span>queue {formatTimestamp(queue.generated_at)}</span>
            <span>pipeline {formatTimestamp(pipeline.generated_at)}</span>
            <span>activity {formatTimestamp(activity.generated_at)}</span>
          </div>
        </section>

        <section className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {labHealth.map((item) => (
            <HealthCard key={item.title} title={item.title} value={item.value} body={item.body} tone={item.tone} />
          ))}
        </section>

        <section className="mt-10">
          <SectionHeading
            eyebrow="What to do now"
            title="Choose a path"
            body="Each card points to a concrete lab action. The raw dashboards are still available, but the default view is now action first."
          />

          <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <ActionCard
              href="/proposals"
              eyebrow="Review"
              title={pendingProposals[0]?.title ?? "Approve the AI plan"}
              body="Read what the orchestrator wants to build before it becomes work."
              meta={`${activity.summary.decisions} decisions logged`}
            />
            <ActionCard
              href="/lab/queue"
              eyebrow="Run"
              title={nextSpec ? nextSpec.title : "Pick a queue spec"}
              body="Queue specs are the smallest unit of useful lab work for machines."
              meta={computeMeta}
            />
            <ActionCard
              href="/lab/research"
              eyebrow="Read"
              title={highlightedPaper ? highlightedPaper.title : "Read published research"}
              body="Start from the newest manuscript, then inspect the attached experiments."
              meta={highlightedPaper ? formatDate(highlightedPaper.created) : "No papers exported"}
            />
            <ActionCard
              href="/lab/leaderboard"
              eyebrow="Compare"
              title={topLeaderboard[0]?.row.run ?? "Open the leaderboard"}
              body="See which mechanisms are actually winning in each tier."
              meta={topLeaderboard[0] ? `delta ${deltaText(topLeaderboard[0].row.delta)}` : "No rows exported"}
            />
          </div>
        </section>

        <section className="mt-10 grid gap-6 xl:grid-cols-[1.35fr_0.95fr]">
          <article className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6 md:p-7">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <p className="text-xs uppercase tracking-[0.28em] text-cyan-300/70">Latest research artifact</p>
                <h2 className="mt-3 text-2xl font-semibold tracking-tight">
                  {highlightedPaper ? highlightedPaper.title : "No paper exported yet"}
                </h2>
                {highlightedPaper && (
                  <p className="mt-2 text-sm text-[#faf9f6]/50">
                    {highlightedPaper.authors.join(", ")} - {formatDate(highlightedPaper.created)} - {highlightedPaper.experiments.length} experiments
                  </p>
                )}
              </div>
              {highlightedPaper && (
                <span className={`rounded-full border px-3 py-1 text-xs uppercase tracking-[0.18em] ${statusTone(highlightedPaper.status)}`}>
                  {highlightedPaper.status}
                </span>
              )}
            </div>

            {highlightedPaper?.curves && (
              <div className="mt-6 rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-4">
                <p className="mb-3 text-sm text-[#faf9f6]/60">
                  Validation loss - {highlightedPaper.curves.tier} - noise band +/-{highlightedPaper.curves.noise_band}
                </p>
                <LossChart curves={highlightedPaper.curves} />
              </div>
            )}

            {highlightedPaper && (
              <div className="mt-6 grid gap-3 lg:grid-cols-2">
                {highlightedPaper.experiments.slice(0, 6).map((experiment) => (
                  <div key={experiment.goal_id} className="rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-4">
                    <div className="flex items-start gap-3">
                      <span className={`mt-0.5 shrink-0 rounded-full border px-2.5 py-0.5 text-xs ${verdictColor(experiment.label)}`}>
                        {verdictWord(experiment.label)}
                      </span>
                      <p className="text-sm leading-relaxed text-[#faf9f6]/78">{experiment.label}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div className="mt-6 flex flex-wrap gap-3">
              <Link
                href="/lab/research"
                className="inline-flex rounded-full border border-cyan-300/30 bg-cyan-300/10 px-4 py-2 text-sm font-medium text-cyan-200 transition hover:bg-cyan-300/15"
              >
                Open research library
              </Link>
              <Link
                href="/lab/problems"
                className="inline-flex rounded-full border border-[#f0eee6]/15 bg-[#f0eee6]/[0.04] px-4 py-2 text-sm font-medium text-[#faf9f6]/75 transition hover:border-cyan-300/30 hover:text-cyan-200"
              >
                See linked problems
              </Link>
            </div>
          </article>

          <aside className="space-y-6">
            <section className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-cyan-300/70">Pipeline health</p>
                  <h2 className="mt-2 text-xl font-semibold">Idea gates</h2>
                </div>
                <Link href="/lab/pipeline" className="text-sm text-cyan-300 hover:text-cyan-200">
                  Details
                </Link>
              </div>

              <div className="mt-5 space-y-3">
                {gateEntries.map((entry) => (
                  <div key={entry.key}>
                    <div className="mb-1 flex justify-between gap-3 text-xs text-[#faf9f6]/55">
                      <span>{entry.label}</span>
                      <span>{entry.value}</span>
                    </div>
                    <ProgressBar value={entry.value} max={maxGate} />
                  </div>
                ))}
              </div>
            </section>

            <section className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-cyan-300/70">In flight</p>
                  <h2 className="mt-2 text-xl font-semibold">{pipeline.in_flight.length} ideas exported</h2>
                </div>
                <span className="text-xs text-[#faf9f6]/40">{inFlight.length} shown</span>
              </div>

              <div className="mt-4 space-y-3">
                {inFlight.length === 0 ? (
                  <p className="text-sm text-[#faf9f6]/55">No in-flight ideas exported.</p>
                ) : (
                  inFlight.map((idea) => (
                    <div key={`${idea.id}-${idea.updated}`} className="rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-3">
                      <div className="flex items-start justify-between gap-3">
                        <p className="text-sm font-medium leading-snug">{idea.id}</p>
                        <span className={`rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.16em] ${statusTone(idea.status)}`}>
                          {idea.status}
                        </span>
                      </div>
                      <p className="mt-2 text-xs text-[#faf9f6]/45">round {idea.round} - {formatTimestamp(idea.updated)}</p>
                    </div>
                  ))
                )}
              </div>
            </section>
          </aside>
        </section>

        <section className="mt-10 grid gap-6 xl:grid-cols-3">
          <article className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6 xl:col-span-2">
            <SectionHeading
              eyebrow="Queue and results"
              title="Next compute jobs"
              body="The queue preview shows the specs that are ready to run and the latest returned metric."
              href="/lab/queue"
              linkLabel="Open queue"
            />

            <div className="mt-5 grid gap-4 md:grid-cols-2">
              {queuedSpecs.length === 0 ? (
                <p className="rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-4 text-sm text-[#faf9f6]/55">
                  No queued specs exported.
                </p>
              ) : (
                queuedSpecs.slice(0, 4).map((spec) => (
                  <Link
                    key={spec.id}
                    href="/lab/queue"
                    className="rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-4 transition hover:border-cyan-300/30 hover:bg-cyan-300/[0.04]"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <h3 className="text-sm font-semibold leading-snug">{spec.title}</h3>
                      <span className={`shrink-0 rounded-full border px-2.5 py-0.5 text-[10px] uppercase tracking-[0.16em] ${statusTone(spec.status)}`}>
                        {spec.status}
                      </span>
                    </div>
                    <p className="mt-2 text-xs text-[#faf9f6]/45">{spec.id}</p>
                    <p className="mt-3 text-xs text-[#faf9f6]/55">
                      {spec.gpu_vram_gb ?? "unknown"}GB VRAM - about {spec.hours ?? "unknown"}h
                    </p>
                  </Link>
                ))
              )}
            </div>

            {latestResult && (
              <div className="mt-5 rounded-lg border border-emerald-400/20 bg-emerald-400/[0.06] p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-emerald-300/80">Latest returned result</p>
                    <h3 className="mt-2 text-base font-semibold">{latestResult.spec_id}</h3>
                    <p className="mt-2 text-xs leading-relaxed text-[#faf9f6]/68">{metricSummary(latestResult.metrics) || "No metrics exported."}</p>
                  </div>
                  <span className={`rounded-full border px-3 py-1 text-xs uppercase tracking-[0.18em] ${statusTone(latestResult.exit_status)}`}>
                    {latestResult.exit_status}
                  </span>
                </div>
              </div>
            )}
          </article>

          <article className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6">
            <SectionHeading eyebrow="Approvals" title="Proposal state" href="/proposals" linkLabel="Open proposals" />
            <div className="mt-5 space-y-3">
              {proposals.length === 0 ? (
                <p className="text-sm text-[#faf9f6]/55">No proposals exported.</p>
              ) : (
                proposals.slice(0, 3).map((proposal) => (
                  <Link
                    key={proposal.slug}
                    href={`/proposals/${proposal.slug}`}
                    className="block rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-4 transition hover:border-cyan-300/30"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <h3 className="text-sm font-semibold leading-snug">{proposal.title}</h3>
                      <span className={`shrink-0 rounded-full border px-2.5 py-0.5 text-[10px] uppercase tracking-[0.16em] ${statusTone(proposal.status)}`}>
                        {proposal.status.replace("-", " ")}
                      </span>
                    </div>
                    <p className="mt-2 text-xs text-[#faf9f6]/45">{proposal.date || proposal.slug}</p>
                  </Link>
                ))
              )}
            </div>
          </article>
        </section>

        <section className="mt-10 grid gap-6 xl:grid-cols-[1fr_1fr]">
          <article className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6">
            <SectionHeading
              eyebrow="Leaderboard"
              title="Current winners"
              body="Top rows from each exported leaderboard tier."
              href="/lab/leaderboard"
              linkLabel="Full leaderboard"
            />
            <div className="mt-5 space-y-3">
              {topLeaderboard.length === 0 ? (
                <p className="text-sm text-[#faf9f6]/55">No leaderboard rows exported.</p>
              ) : (
                topLeaderboard.map((entry) => (
                  <Link
                    key={`${entry.tier}-${entry.row.run}`}
                    href="/lab/leaderboard"
                    className="block rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-4 transition hover:border-cyan-300/30"
                  >
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <p className="text-xs text-[#faf9f6]/42">{entry.tier}</p>
                        <h3 className="mt-1 text-sm font-semibold leading-snug">{entry.row.run}</h3>
                      </div>
                      <div className="text-right">
                        <p className="font-mono text-sm text-[#faf9f6]">{entry.row.val_loss.toFixed(4)}</p>
                        <p className={`font-mono text-xs ${deltaTone(entry.row.delta)}`}>{deltaText(entry.row.delta)}</p>
                      </div>
                    </div>
                    <p className="mt-3 line-clamp-2 text-xs leading-relaxed text-[#faf9f6]/55">{entry.row.summary}</p>
                  </Link>
                ))
              )}
            </div>
          </article>

          <article className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6">
            <SectionHeading
              eyebrow="Recent activity"
              title="Threads and runs"
              body="A compact readout from the exported activity snapshot."
              href="/lab/experiments"
              linkLabel="Activity log"
            />
            <div className="mt-5 grid gap-4 md:grid-cols-2">
              <div className="space-y-3">
                <p className="text-xs uppercase tracking-[0.2em] text-[#faf9f6]/40">Active threads</p>
                {activeThreads.length === 0 ? (
                  <p className="text-sm text-[#faf9f6]/55">No active threads exported.</p>
                ) : (
                  activeThreads.map((thread) => (
                    <div key={`${thread.name}-${thread.created}`} className="rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-3">
                      <div className="flex items-start justify-between gap-3">
                        <h3 className="text-sm font-medium leading-snug">{thread.title}</h3>
                        <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.16em] ${statusTone(thread.status)}`}>
                          {thread.status}
                        </span>
                      </div>
                      <p className="mt-2 text-xs text-[#faf9f6]/40">{compactDate(thread.created)}</p>
                    </div>
                  ))
                )}
              </div>
              <div className="space-y-3">
                <p className="text-xs uppercase tracking-[0.2em] text-[#faf9f6]/40">Recent runs</p>
                {recentRuns.length === 0 ? (
                  <p className="text-sm text-[#faf9f6]/55">No runs exported.</p>
                ) : (
                  recentRuns.map((run) => (
                    <div key={`${run.name}-${run.created}`} className="rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-3">
                      <div className="flex items-start justify-between gap-3">
                        <h3 className="text-sm font-medium leading-snug">{run.name}</h3>
                        <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.16em] ${statusTone(run.status)}`}>
                          {run.status}
                        </span>
                      </div>
                      <p className="mt-2 text-xs text-[#faf9f6]/48">
                        {run.metric ? `${run.metric.label}: ${run.metric.value}` : compactDate(run.created)}
                      </p>
                    </div>
                  ))
                )}
              </div>
            </div>
          </article>
        </section>

        <section className="mt-10 grid gap-6 xl:grid-cols-[1fr_1fr]">
          <article className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6">
            <SectionHeading
              eyebrow="Open questions"
              title="Problems to attack"
              href="/lab/problems"
              linkLabel="All problems"
            />
            <div className="mt-5 grid gap-3">
              {openProblems.length === 0 ? (
                <p className="text-sm text-[#faf9f6]/55">No open problems exported.</p>
              ) : (
                openProblems.slice(0, 5).map((problem) => (
                  <div key={problem.id} className="rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-4">
                    <div className="flex items-start justify-between gap-3">
                      <h3 className="text-sm font-semibold leading-snug">{problem.title}</h3>
                      <span className={`shrink-0 rounded-full border px-2.5 py-0.5 text-[10px] uppercase tracking-[0.16em] ${statusTone(problem.status)}`}>
                        {problem.status}
                      </span>
                    </div>
                    {problem.answer && <p className="mt-3 text-xs leading-relaxed text-[#faf9f6]/62">{problem.answer}</p>}
                    <p className="mt-2 text-xs text-[#faf9f6]/45">linked paper: {problem.paper ?? "none yet"}</p>
                  </div>
                ))
              )}
            </div>
          </article>

          <article className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6">
            <SectionHeading eyebrow="Direction" title="Active goals" href="/lab/goals" linkLabel="All goals" />
            <div className="mt-5 space-y-4">
              {activeGoals.length === 0 ? (
                <p className="text-sm text-[#faf9f6]/55">No active public goals exported.</p>
              ) : (
                activeGoals.slice(0, 3).map((goal) => {
                  const done = goal.milestones.filter((milestone) => milestone.done).length;
                  return (
                    <div key={goal.id} className="rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-4">
                      <div className="flex items-start justify-between gap-3">
                        <h3 className="text-sm font-semibold leading-snug">{goal.title}</h3>
                        <span className={`shrink-0 rounded-full border px-2.5 py-0.5 text-[10px] uppercase tracking-[0.16em] ${statusTone(goal.status)}`}>
                          {goal.status}
                        </span>
                      </div>
                      <p className="mt-2 text-xs leading-relaxed text-[#faf9f6]/58">{goal.description}</p>
                      <div className="mt-4">
                        <div className="mb-1 flex justify-between text-xs text-[#faf9f6]/45">
                          <span>Milestones</span>
                          <span>{done}/{goal.milestones.length}</span>
                        </div>
                        <ProgressBar value={done} max={goal.milestones.length} />
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </article>
        </section>

        <section className="mt-10 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6">
          <SectionHeading
            eyebrow="Decisions"
            title="Recent human-readable decisions"
            body="The lab should make it clear why work moved forward, stopped, or changed direction."
            href="/lab/experiments"
            linkLabel="Full log"
          />
          <div className="mt-5 grid gap-3 md:grid-cols-2">
            {recentDecisions.length === 0 ? (
              <p className="text-sm text-[#faf9f6]/55">No decisions exported.</p>
            ) : (
              recentDecisions.map((decision) => (
                <div key={`${decision.created}-${decision.decision}`} className="rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-[#faf9f6]/40">{compactDate(decision.created)}</p>
                  <p className="mt-3 text-sm leading-relaxed text-[#faf9f6]/76">{decision.decision}</p>
                </div>
              ))
            )}
          </div>
        </section>

        <details className="group mt-10 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5">
          <summary className="flex cursor-pointer list-none items-center justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-cyan-300/70">Advanced map</p>
              <h2 className="mt-2 text-xl font-semibold">Open the raw lab sections</h2>
            </div>
            <span className="rounded-full border border-[#f0eee6]/10 px-3 py-1 text-sm text-[#faf9f6]/50 transition group-open:rotate-180">
              menu
            </span>
          </summary>

          <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {labGroups.map((group) => (
              <section key={group.title} className="rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-4">
                <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-[#faf9f6]/55">{group.title}</h3>
                <div className="mt-4 space-y-3">
                  {group.links.map((link) => (
                    <Link
                      key={link.href}
                      href={link.href}
                      className="block rounded-lg border border-transparent p-3 transition hover:border-cyan-300/25 hover:bg-cyan-300/[0.04]"
                    >
                      <div className="text-sm font-semibold text-[#faf9f6]">{link.label}</div>
                      <div className="mt-1 text-xs leading-relaxed text-[#faf9f6]/52">{link.description}</div>
                    </Link>
                  ))}
                </div>
              </section>
            ))}
          </div>
        </details>
      </div>
    </main>
  );
}
