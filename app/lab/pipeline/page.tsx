import fs from "fs";
import path from "path";

type GateSummary = {
  taste: number;
  definition: number;
  code: number;
  needs_run: number;
  running: number;
  done: number;
  rejected: number;
};

type PipelineIdea = {
  id: string;
  status: string;
  round: number;
  updated: string;
};

type PipelineSnapshot = {
  generated_at: string;
  counts: Record<string, number>;
  gate_summary: GateSummary;
  feeding_gpu: number;
  gpu_idle_risk: boolean;
  in_flight: PipelineIdea[];
};

const EMPTY_SNAPSHOT: PipelineSnapshot = {
  generated_at: "",
  counts: {},
  gate_summary: {
    taste: 0,
    definition: 0,
    code: 0,
    needs_run: 0,
    running: 0,
    done: 0,
    rejected: 0,
  },
  feeding_gpu: 0,
  gpu_idle_risk: true,
  in_flight: [],
};

const snapshotPath = path.join(process.cwd(), "public", "data", "pipeline-snapshot.json");

function readSnapshot(): PipelineSnapshot {
  try {
    const raw = fs.readFileSync(snapshotPath, "utf8");
    const parsed = JSON.parse(raw) as Partial<PipelineSnapshot>;
    const gateSummary = (parsed.gate_summary ?? {}) as Partial<GateSummary>;
    return {
      generated_at: typeof parsed.generated_at === "string" ? parsed.generated_at : "",
      counts:
        parsed.counts && typeof parsed.counts === "object" && !Array.isArray(parsed.counts)
          ? Object.fromEntries(
              Object.entries(parsed.counts).filter(([, value]) => typeof value === "number" && Number.isFinite(value))
            )
          : {},
      gate_summary: {
        taste: typeof gateSummary.taste === "number" ? gateSummary.taste : 0,
        definition: typeof gateSummary.definition === "number" ? gateSummary.definition : 0,
        code: typeof gateSummary.code === "number" ? gateSummary.code : 0,
        needs_run: typeof gateSummary.needs_run === "number" ? gateSummary.needs_run : 0,
        running: typeof gateSummary.running === "number" ? gateSummary.running : 0,
        done: typeof gateSummary.done === "number" ? gateSummary.done : 0,
        rejected: typeof gateSummary.rejected === "number" ? gateSummary.rejected : 0,
      },
      feeding_gpu: typeof parsed.feeding_gpu === "number" ? parsed.feeding_gpu : 0,
      gpu_idle_risk: typeof parsed.gpu_idle_risk === "boolean" ? parsed.gpu_idle_risk : true,
      in_flight: Array.isArray(parsed.in_flight)
        ? parsed.in_flight
            .map((idea) => ({
              id: typeof idea?.id === "string" ? idea.id : "",
              status: typeof idea?.status === "string" ? idea.status : "",
              round: typeof idea?.round === "number" ? idea.round : 0,
              updated: typeof idea?.updated === "string" ? idea.updated : "",
            }))
            .filter((idea) => idea.id)
        : [],
    };
  } catch {
    return EMPTY_SNAPSHOT;
  }
}

function formatDate(value: string) {
  return value ? value.slice(0, 16).replace("T", " ") : "—";
}

function statusTone(status: string) {
  const s = status.toLowerCase();
  if (s === "running") return "text-emerald-300 border-emerald-300/30 bg-emerald-300/10";
  if (s === "needs-run") return "text-cyan-300 border-cyan-300/30 bg-cyan-300/10";
  if (s.includes("taste") || s.includes("pitch")) return "text-amber-300 border-amber-300/30 bg-amber-300/10";
  if (s.includes("review") || s.includes("revision")) return "text-sky-300 border-sky-300/30 bg-sky-300/10";
  if (s.includes("plan") || s.includes("code")) return "text-violet-300 border-violet-300/30 bg-violet-300/10";
  if (s === "done") return "text-emerald-400 border-emerald-400/30 bg-emerald-400/10";
  if (s === "rejected") return "text-rose-400 border-rose-400/30 bg-rose-400/10";
  return "text-[#faf9f6]/70 border-[#f0eee6]/10 bg-[#f0eee6]/[0.04]";
}

function gateCard(label: string, value: number) {
  return (
    <div className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-4">
      <div className="text-xs uppercase tracking-[0.18em] text-[#faf9f6]/45">{label}</div>
      <div className="mt-2 text-2xl font-semibold tabular-nums text-[#faf9f6]">{value}</div>
    </div>
  );
}

function sortInFlight(a: PipelineIdea, b: PipelineIdea) {
  const priority: Record<string, number> = {
    running: 0,
    "needs-run": 1,
    "needs-codereview": 2,
    codereviewing: 3,
    "needs-recode": 4,
    recoding: 5,
    "needs-plan": 6,
    planning: 7,
    "needs-review": 8,
    reviewing: 9,
    "needs-revision": 10,
    revising: 11,
    "needs-taste": 12,
    tasting: 13,
    "needs-repitch": 14,
    repitching: 15,
  };

  const aRank = priority[a.status.toLowerCase()] ?? 50;
  const bRank = priority[b.status.toLowerCase()] ?? 50;
  if (aRank !== bRank) return aRank - bRank;
  if (a.updated !== b.updated) return b.updated.localeCompare(a.updated);
  return a.id.localeCompare(b.id);
}

const snapshot = readSnapshot();
const inFlight = [...snapshot.in_flight].sort(sortInFlight);

export const dynamic = "force-static";

export default function LabPipelinePage() {
  const gpuFed = !snapshot.gpu_idle_risk;
  const gpuCopy = gpuFed
    ? `GPU fed (${snapshot.feeding_gpu} in flight)`
    : `GPU idle risk — only ${snapshot.feeding_gpu} feeding`;
  const gpuClass = gpuFed
    ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
    : "border-rose-400/30 bg-rose-400/10 text-rose-300";

  return (
    <main className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]">
      <div className="container mx-auto max-w-6xl px-6 py-16">
        <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/70">Lab</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight md:text-4xl">Pipeline</h1>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-[#faf9f6]/70">
          live readout for the autoresearch idea pipeline: how many ideas are feeding the GPU, what gate they are
          stuck behind, and whether the prime directive is at risk
        </p>

        <div className={`mt-6 inline-flex rounded-full border px-4 py-2 text-sm ${gpuClass}`}>{gpuCopy}</div>

        <div className="mt-8 grid gap-3 md:grid-cols-4 xl:grid-cols-7">
          {gateCard("Taste", snapshot.gate_summary.taste)}
          {gateCard("Definition", snapshot.gate_summary.definition)}
          {gateCard("Code", snapshot.gate_summary.code)}
          {gateCard("needs-run", snapshot.gate_summary.needs_run)}
          {gateCard("running", snapshot.gate_summary.running)}
          {gateCard("done", snapshot.gate_summary.done)}
          {gateCard("rejected", snapshot.gate_summary.rejected)}
        </div>

        <section className="mt-10">
          <div className="flex items-baseline justify-between gap-4">
            <h2 className="text-xl font-semibold md:text-2xl">In-Flight</h2>
            <span className="text-xs text-[#faf9f6]/40">{inFlight.length} ideas</span>
          </div>

          {inFlight.length === 0 ? (
            <p className="mt-4 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6 text-sm text-[#faf9f6]/55">
              No in-flight ideas found.
            </p>
          ) : (
            <div className="mt-4 overflow-x-auto rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02]">
              <table className="min-w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-[#f0eee6]/10 text-xs uppercase tracking-widest text-[#faf9f6]/50">
                    <th className="px-4 py-3">Idea</th>
                    <th className="px-4 py-3">Status</th>
                    <th className="px-4 py-3">Round</th>
                    <th className="px-4 py-3">Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {inFlight.map((idea) => (
                    <tr key={idea.id} className="border-b border-[#f0eee6]/5 last:border-b-0">
                      <td className="px-4 py-4">
                        <div className="font-medium text-[#faf9f6]">{idea.id}</div>
                      </td>
                      <td className="px-4 py-4">
                        <span className={`inline-flex rounded-full border px-2.5 py-1 text-xs uppercase tracking-[0.18em] ${statusTone(idea.status)}`}>
                          {idea.status}
                        </span>
                      </td>
                      <td className="px-4 py-4 font-mono text-[#faf9f6]/75 tabular-nums">{idea.round}</td>
                      <td className="px-4 py-4 font-mono text-[#faf9f6]/65">{formatDate(idea.updated)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
