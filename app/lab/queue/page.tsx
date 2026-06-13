import fs from "fs";
import path from "path";
import ReviewControls from "./ReviewControls";

type QueueSpec = {
  id: string;
  title: string;
  idea?: string;
  status: string;
  gpu_vram_gb: number | null;
  hours: number | null;
  plain?: string;
};

type QueueResult = {
  spec_id: string;
  worker: string;
  finished: string;
  exit_status: string;
  metrics: Record<string, string | number>;
};

type QueueSnapshot = {
  generated_at: string;
  specs: QueueSpec[];
  results: QueueResult[];
};

const EMPTY_SNAPSHOT: QueueSnapshot = {
  generated_at: "",
  specs: [],
  results: [],
};

const snapshotPath = path.join(process.cwd(), "public", "data", "queue-snapshot.json");
const queuePath = path.resolve(process.cwd(), "..", "llm-research-kit-scaling", "queue");

function parseField(source: string, field: string) {
  const match = source.match(new RegExp(`^${field}:\\s*(.+)$`, "m"));
  if (!match) return "";
  let value = match[1].trim();
  if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
    value = value.slice(1, -1);
  }
  return value;
}

function loadIdeaMap(): Record<string, string> {
  try {
    if (!fs.existsSync(queuePath)) {
      return {};
    }

    const map: Record<string, string> = {};
    for (const file of fs.readdirSync(queuePath)) {
      if (!file.endsWith(".yaml")) continue;
      const fullPath = path.join(queuePath, file);
      const contents = fs.readFileSync(fullPath, "utf8");
      const id = parseField(contents, "id");
      const idea = parseField(contents, "idea");
      if (id && idea) {
        map[id] = idea;
      }
    }
    return map;
  } catch {
    return {};
  }
}

const ideaMap = loadIdeaMap();

function loadSnapshot(): QueueSnapshot {
  try {
    const raw = fs.readFileSync(snapshotPath, "utf8");
    const parsed = JSON.parse(raw) as Partial<QueueSnapshot>;
    return {
      generated_at: typeof parsed.generated_at === "string" ? parsed.generated_at : "",
      specs: Array.isArray(parsed.specs)
        ? parsed.specs.map((spec) => ({
            id: typeof spec?.id === "string" ? spec.id : "",
            title: typeof spec?.title === "string" ? spec.title : typeof spec?.id === "string" ? spec.id : "",
            idea: typeof spec?.idea === "string" ? spec.idea : typeof spec?.id === "string" ? ideaMap[spec.id] ?? "" : "",
            status: typeof spec?.status === "string" ? spec.status : "queued",
            gpu_vram_gb: typeof spec?.gpu_vram_gb === "number" ? spec.gpu_vram_gb : null,
            hours: typeof spec?.hours === "number" ? spec.hours : null,
            plain: typeof spec?.plain === "string" ? spec.plain : "",
          }))
        : [],
      results: Array.isArray(parsed.results)
        ? parsed.results.map((result) => ({
            spec_id: typeof result?.spec_id === "string" ? result.spec_id : "",
            worker: typeof result?.worker === "string" ? result.worker : "",
            finished: typeof result?.finished === "string" ? result.finished : "",
            exit_status: typeof result?.exit_status === "string" ? result.exit_status : "",
            metrics:
              result && typeof result.metrics === "object" && !Array.isArray(result.metrics)
                ? (result.metrics as Record<string, string | number>)
              : {},
          }))
        : [],
    };
  } catch {
    return EMPTY_SNAPSHOT;
  }
}

const snapshot = loadSnapshot();

const formatDate = (value: string) => (value ? value.slice(0, 16).replace("T", " ") : "—");

const normalizeStatus = (status: string) => {
  const s = status.toLowerCase();
  if (s.includes("done") || s.includes("complete") || s.includes("finished") || s.includes("success") || s.includes("failed")) {
    return "done";
  }
  if (s.includes("claim") || s.includes("running") || s.includes("active") || s.includes("leased") || s.includes("progress")) {
    return "claimed";
  }
  return "queued";
};

const resultStatusClass = (status: string) => {
  const s = status.toLowerCase();
  if (s.includes("success") || s.includes("done") || s.includes("complete") || s.includes("finished")) {
    return "text-emerald-400 border-emerald-400/30 bg-emerald-400/10";
  }
  if (s.includes("failed") || s.includes("error") || s.includes("out-of-time") || s.includes("timeout")) {
    return "text-rose-400 border-rose-400/30 bg-rose-400/10";
  }
  if (s.includes("claim") || s.includes("running") || s.includes("active") || s.includes("leased") || s.includes("progress")) {
    return "text-amber-300 border-amber-300/30 bg-amber-300/10";
  }
  return "text-cyan-300 border-cyan-300/30 bg-cyan-300/10";
};

const formatMetricValue = (value: string | number) =>
  typeof value === "number" ? (Number.isInteger(value) ? value.toString() : value.toFixed(4).replace(/\.?0+$/, "")) : value;

const formatMetrics = (metrics: Record<string, string | number>) => {
  const entries = Object.entries(metrics);
  if (entries.length === 0) return "—";
  return entries
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, value]) => `${key}: ${formatMetricValue(value)}`)
    .join(", ");
};

const sectionOrder = ["queued", "claimed", "done"] as const;

function specSectionTitle(status: (typeof sectionOrder)[number]) {
  if (status === "claimed") return "Claimed";
  if (status === "done") return "Done";
  return "Queued";
}

function splitSpecs(specs: QueueSpec[]) {
  const grouped: Record<(typeof sectionOrder)[number], QueueSpec[]> = {
    queued: [],
    claimed: [],
    done: [],
  };

  for (const spec of specs) {
    grouped[normalizeStatus(spec.status)].push(spec);
  }

  for (const status of sectionOrder) {
    grouped[status].sort((a, b) => a.title.localeCompare(b.title) || a.id.localeCompare(b.id));
  }

  return grouped;
}

export const dynamic = "force-static";

export default function LabQueuePage() {
  const grouped = splitSpecs(snapshot.specs);
  const hasSpecs = snapshot.specs.length > 0;

  return (
    <main className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]">
      <div className="container mx-auto max-w-6xl px-6 py-16">
        <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/70">Lab</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight md:text-4xl">Queue</h1>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-[#faf9f6]/70">
          experiments anyone will be able to run for the lab; today it runs on the lab&apos;s own machines
        </p>

        <details className="group mt-5 max-w-3xl rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] px-4 py-3">
          <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-sm text-[#faf9f6]/70">
            <span className="text-xs uppercase tracking-[0.18em] text-cyan-300/70">What every experiment here shares</span>
            <span className="text-[#faf9f6]/40 transition group-open:rotate-180">⌄</span>
          </summary>
          <p className="mt-3 text-sm leading-relaxed text-[#faf9f6]/65">
            Each experiment trains the same kind of small language model on the same text. The only thing that
            changes from one to the next is the single idea it adds or swaps in. And every one is judged the same
            way: does this change help the model learn the text better — a lower error — than the plain version?
          </p>
        </details>

        {!hasSpecs && (
          <p className="mt-8 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6 text-sm text-[#faf9f6]/60">
            Queue is being seeded — first specs land shortly.
          </p>
        )}

        <div className="mt-8 grid gap-4 md:grid-cols-3">
          {sectionOrder.map((status) => {
            const specs = grouped[status];
            return (
              <section key={status} className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5">
                <div className="flex items-baseline justify-between gap-3">
                  <h2 className="text-xl font-semibold">{specSectionTitle(status)}</h2>
                  <span className="text-xs uppercase tracking-widest text-[#faf9f6]/40">{specs.length}</span>
                </div>
                {specs.length === 0 ? (
                  <p className="mt-4 text-sm leading-relaxed text-[#faf9f6]/55">No specs in this state yet.</p>
                ) : (
                  <div className="mt-4 space-y-3">
                    {specs.map((spec) => (
                      <ReviewControls key={spec.id} spec={spec} idea={spec.idea} />
                    ))}
                  </div>
                )}
              </section>
            );
          })}
        </div>

        <section className="mt-10">
          <div className="flex items-baseline justify-between gap-4">
            <h2 className="text-xl font-semibold md:text-2xl">Results</h2>
            <span className="text-xs text-[#faf9f6]/40">{snapshot.results.length} runs</span>
          </div>

          {snapshot.results.length === 0 ? (
            <p className="mt-4 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6 text-sm text-[#faf9f6]/55">
              No results recorded yet.
            </p>
          ) : (
            <div className="mt-4 overflow-x-auto rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02]">
              <table className="min-w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-[#f0eee6]/10 text-xs uppercase tracking-widest text-[#faf9f6]/50">
                    <th className="px-4 py-3">Spec</th>
                    <th className="px-4 py-3">Worker</th>
                    <th className="px-4 py-3">Metrics</th>
                    <th className="px-4 py-3">Status</th>
                    <th className="px-4 py-3">Date</th>
                  </tr>
                </thead>
                <tbody>
                  {snapshot.results.map((result) => (
                    <tr key={`${result.spec_id}-${result.worker}-${result.finished}`} className="border-b border-[#f0eee6]/5 last:border-b-0">
                      <td className="px-4 py-4">
                        <div className="font-medium text-[#faf9f6]">{result.spec_id}</div>
                      </td>
                      <td className="px-4 py-4 text-[#faf9f6]/65">{result.worker || "—"}</td>
                      <td className="px-4 py-4 font-mono text-xs leading-relaxed text-[#faf9f6]/75">
                        {formatMetrics(result.metrics)}
                      </td>
                      <td className="px-4 py-4">
                        <span className={`rounded-full border px-3 py-1 text-[11px] uppercase tracking-wider ${resultStatusClass(result.exit_status)}`}>
                          {result.exit_status || "—"}
                        </span>
                      </td>
                      <td className="px-4 py-4 text-[#faf9f6]/50">{formatDate(result.finished)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <p className="mt-10 text-xs text-[#faf9f6]/40">
          Snapshot from the queue registry at generated_at {snapshot.generated_at ? formatDate(snapshot.generated_at) : "—"}.
        </p>
      </div>
    </main>
  );
}
