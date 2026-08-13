import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Commercial Algorithm Strategy | Open Discovery",
  description:
    "The markets, buyers, experiment protocols, evidence gates, and commercialization plan for autonomous algorithm improvement.",
};

const markets = [
  {
    rank: "01",
    name: "Databases",
    decision: "Start here",
    body: "Accessible code and benchmarks make credible proof possible. Runtime, memory, and fleet capacity connect directly to customer cost.",
    algorithms: "Joins · aggregation · sorting · scans · planning · indexes · compression · spilling",
  },
  {
    rank: "02",
    name: "Logistics",
    decision: "Design partner",
    body: "Routing and scheduling improvements repeat across vehicles, orders, warehouses, and shifts—but real constraints and replay data are essential.",
    algorithms: "Vehicle routing · dispatch · matching · fleet assignment · picking · packing · scheduling",
  },
  {
    rank: "03",
    name: "AI and cloud systems",
    decision: "Customer-led",
    body: "GPU throughput, placement, autoscaling, memory, and latency are valuable, but the stacks move quickly and realistic workloads are often private.",
    algorithms: "Inference · batching · caching · placement · autoscaling · queueing · kernels",
  },
  {
    rank: "04",
    name: "Compilers and data paths",
    decision: "Specialist lane",
    body: "Compiler, search, indexing, compression, and streaming wins can multiply across fleets, but require deeper integration and specialist buyers.",
    algorithms: "Codegen · vectorization · fusion · indexing · compression · streaming state",
  },
];

const databaseTargets = [
  "Hash and grouped aggregation",
  "Hash, merge, and nested-loop joins",
  "Join ordering and cardinality estimation",
  "Sorting, Top-N, partitioning, and spilling",
  "Scans, filters, and vectorized expressions",
  "Indexes, range lookup, caching, and prefetch",
  "Compression, encoding, and late materialization",
  "Shuffle, skew handling, ingestion, and compaction",
];

const databaseBuyers = [
  "Database and analytical-engine vendors",
  "Warehouses, lakehouses, and query platforms",
  "ETL, observability, and data-infrastructure companies",
  "Teams operating a costly internal data platform",
];

const databaseProtocol = [
  "Pin the engine commit, compiler, flags, machine, and untouched baseline binary.",
  "Freeze exact-result and schema checks, customer-like query shapes, visible cases, and a hidden confirmation suite.",
  "Profile first; give each search lane a distinct mechanism and isolated candidate folder.",
  "Build one candidate at a time. Run semantic and adversarial SQL checks before timing.",
  "Interleave baseline and candidate trials; report medians, tails, variability, and peak resident memory.",
  "Rebuild a provisional winner cleanly and confirm it on untouched cases without tuning.",
];

const logisticsTargets = [
  "Capacitated vehicle routing",
  "Time windows, breaks, skills, and service rules",
  "Dynamic dispatch and same-day insertion",
  "Pickup-and-delivery matching",
  "Fleet assignment and empty-mile reduction",
  "Warehouse picking, slotting, and wave planning",
  "Load planning, packing, and utilization",
  "Workforce, dock, and delivery scheduling",
];

const logisticsBuyers = [
  "Route-optimization and field-service platforms",
  "3PL, freight, delivery, and fleet technology companies",
  "Fulfillment, warehouse, and marketplace operators",
  "Large operators that control dispatch and planning",
];

const logisticsProtocol = [
  "Obtain anonymized historical jobs or a realistic simulator plus the complete hard-constraint validator.",
  "Record the production baseline and the actual weighted objective: distance, time, labor, vehicles, lateness, cancellations, and service level.",
  "Split replay chronologically into development and untouched holdout periods, including seasonal and high-stress days.",
  "Search heuristics, neighborhoods, initialization, decomposition, batching, and policies only on development periods.",
  "Require every hard constraint to pass, then compare full cost and service distributions across seeds and stress scenarios.",
  "Run in shadow mode before any live routing decision. Operators keep approval, rollback, and incident controls.",
];

const universalSteps = [
  {
    number: "01",
    title: "Define the economic objective",
    body: "Name one primary metric and the real capability it represents: latency, throughput, memory, cloud cost, route cost, utilization, or service quality.",
  },
  {
    number: "02",
    title: "Freeze the contract",
    body: "Pin source, toolchain, baseline, evaluator, visible development cases, held-out confirmation cases, constraints, resource limits, and the decision rule before search.",
  },
  {
    number: "03",
    title: "Search isolated candidates",
    body: "Every hypothesis gets a separate durable folder and patch. Cheap static checks run first. Candidates never edit the baseline, evaluator, or one another.",
  },
  {
    number: "04",
    title: "Prove correctness first",
    body: "Compare outputs and semantics, then exercise edge, randomized, adversarial, concurrency, failure, and unsupported-path behavior before timing anything.",
  },
  {
    number: "05",
    title: "Measure under control",
    body: "Use the same hardware and toolchain, warmups, repeated trials, interleaved order, median and tail statistics, peak memory, and low-contention timing windows.",
  },
  {
    number: "06",
    title: "Confirm or reject",
    body: "Rebuild the winner from the pinned baseline and rerun fresh held-out cases. Preserve regressions and negative results. Promote only what the frozen rule supports.",
  },
];

const resourceRules = [
  "One owner per candidate and one durable write location.",
  "Scout mechanisms broadly in parallel; serialize compilers and claim-bearing benchmarks.",
  "Never time a candidate while another build or high-CPU scan competes for the machine.",
  "Record CPU, memory, swap, toolchain, source hash, build parallelism, and deviations.",
  "Run cheap correctness and targeted screens before long confirmation suites.",
  "Keep failed builds, slower candidates, noisy runs, and blocked paths as evidence.",
];

const offers = [
  {
    number: "01",
    title: "Paid diagnostic",
    body: "Reproduce one expensive hot path, freeze its evaluator, and rank mechanisms worth testing.",
  },
  {
    number: "02",
    title: "Optimization sprint",
    body: "Search, implement, test, and independently confirm isolated candidates.",
  },
  {
    number: "03",
    title: "Integration",
    body: "Productionize the winner, add regression coverage, and verify it in the customer environment.",
  },
  {
    number: "04",
    title: "Continuous optimizer",
    body: "After repeat demand exists, run private ongoing search within approved surfaces and budgets.",
  },
];

function Eyebrow({ children, amber = false }: { children: React.ReactNode; amber?: boolean }) {
  return (
    <p
      className={
        "font-mono text-xs uppercase tracking-[0.3em] " +
        (amber ? "text-amber-300/80" : "text-cyan-300/75")
      }
    >
      {children}
    </p>
  );
}

function BulletList({ items }: { items: string[] }) {
  return (
    <ul className="mt-5 space-y-3">
      {items.map((item) => (
        <li key={item} className="flex gap-3 text-sm leading-relaxed text-[#faf9f6]/75">
          <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-300/70" />
          {item}
        </li>
      ))}
    </ul>
  );
}

function NumberedList({ items }: { items: string[] }) {
  return (
    <ol className="mt-5 space-y-4">
      {items.map((item, index) => (
        <li key={item} className="flex gap-4 text-sm leading-relaxed text-[#faf9f6]/75">
          <span className="font-mono text-xs text-cyan-300/60">0{index + 1}</span>
          {item}
        </li>
      ))}
    </ol>
  );
}

export default function StrategyPage() {
  return (
    <main className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]">
      <section className="relative overflow-hidden border-b border-[#f0eee6]/10 pt-28">
        <div className="pointer-events-none absolute left-1/2 top-0 h-[36rem] w-[54rem] -translate-x-1/2 rounded-full bg-cyan-400/[0.08] blur-[150px]" />
        <div className="relative mx-auto grid max-w-6xl gap-12 px-6 py-20 md:py-28 lg:grid-cols-[1.35fr_0.8fr] lg:items-center">
          <div>
            <Eyebrow>Open Discovery · Commercial operating strategy · 2026</Eyebrow>
            <h1 className="mt-6 max-w-4xl text-5xl font-semibold leading-[1.02] tracking-tight md:text-7xl">
              AI that finds better algorithms.{" "}
              <span className="text-cyan-300">Evidence before claims.</span>
            </h1>
            <p className="mt-7 max-w-3xl text-lg leading-relaxed text-[#faf9f6]/70 md:text-xl">
              Open Discovery searches for commercially useful algorithm
              improvements, implements them, and proves whether they work against
              a frozen contract. The customer buys a verified economic gain—not
              an abstract autonomous experiment.
            </p>
            <div className="mt-10 flex flex-col gap-3 sm:flex-row">
              <a
                href="#protocol"
                className="inline-flex items-center justify-center rounded-full bg-cyan-300 px-6 py-3 font-semibold text-[#1f1e1d] hover:bg-cyan-200"
              >
                See the experiment protocol ↓
              </a>
              <Link
                href="/cofounder"
                className="inline-flex items-center justify-center rounded-full border border-[#f0eee6]/15 px-6 py-3 font-medium text-[#faf9f6]/85 hover:border-cyan-300/35 hover:text-cyan-200"
              >
                Read the cofounder brief →
              </Link>
            </div>
          </div>

          <aside className="rounded-2xl border border-[#f0eee6]/12 bg-[#faf9f6]/[0.035] p-6 md:p-8">
            <Eyebrow amber>Current decisions</Eyebrow>
            <ol className="mt-6 divide-y divide-[#f0eee6]/10">
              {[
                ["Primary now", "Commercial algorithm improvement"],
                ["First wedge", "Database and data-system hot paths"],
                ["Second wedge", "Logistics with a real design partner"],
                ["Future layer", "General autonomous experiments"],
              ].map(([label, value], index) => (
                <li key={label} className="grid grid-cols-[2rem_1fr] gap-3 py-4">
                  <span className="font-mono text-xs text-cyan-300/50">0{index + 1}</span>
                  <div>
                    <strong className="block text-sm text-[#faf9f6]/45">{label}</strong>
                    <span className="mt-1 block text-sm text-[#faf9f6]/90">{value}</span>
                  </div>
                </li>
              ))}
            </ol>
            <p className="mt-5 text-sm leading-relaxed text-[#faf9f6]/45">
              This is a strategy to test—not proof of demand, revenue, or product-market fit.
            </p>
          </aside>
        </div>
      </section>

      <section id="markets" className="mx-auto max-w-6xl scroll-mt-28 px-6 py-20 md:py-28">
        <div className="grid gap-8 lg:grid-cols-[0.75fr_1.25fr]">
          <div>
            <Eyebrow>Market sequence</Eyebrow>
            <h2 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
              Broad technology. Narrow go-to-market.
            </h2>
            <p className="mt-5 max-w-xl leading-relaxed text-[#faf9f6]/65">
              Choose domains where correctness is machine-checkable, a
              representative workload is available, and a small recurring gain
              has a clear buyer.
            </p>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            {markets.map((market) => (
              <article
                key={market.rank}
                className="rounded-2xl border border-[#f0eee6]/10 bg-[#faf9f6]/[0.025] p-6"
              >
                <div className="flex items-center justify-between gap-4">
                  <span className="font-mono text-xs text-cyan-300/55">{market.rank}</span>
                  <span className="rounded-full border border-cyan-300/20 bg-cyan-300/[0.06] px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-200/80">
                    {market.decision}
                  </span>
                </div>
                <h3 className="mt-6 text-2xl font-semibold">{market.name}</h3>
                <p className="mt-3 text-sm leading-relaxed text-[#faf9f6]/65">{market.body}</p>
                <p className="mt-5 border-t border-[#f0eee6]/10 pt-4 font-mono text-xs leading-relaxed text-[#faf9f6]/40">
                  {market.algorithms}
                </p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section id="databases" className="scroll-mt-28 border-y border-[#f0eee6]/10 bg-cyan-300/[0.018]">
        <div className="mx-auto max-w-6xl px-6 py-20 md:py-28">
          <Eyebrow>Lane 01 · Databases</Eyebrow>
          <div className="mt-5 grid gap-6 lg:grid-cols-[1fr_0.8fr]">
            <h2 className="text-4xl font-semibold tracking-tight md:text-5xl">
              The fastest path to a credible commercial proof.
            </h2>
            <p className="text-lg leading-relaxed text-[#faf9f6]/65">
              Start with teams that own a costly data-system hot path and can
              deploy a patch. Open code and deterministic semantics let us prove
              a narrow claim before asking a customer to trust the system.
            </p>
          </div>

          <div className="mt-12 grid gap-5 lg:grid-cols-3">
            <article className="rounded-2xl border border-[#f0eee6]/10 bg-[#1f1e1d] p-6">
              <h3 className="text-xl font-semibold">Algorithms to search</h3>
              <BulletList items={databaseTargets} />
            </article>
            <article className="rounded-2xl border border-[#f0eee6]/10 bg-[#1f1e1d] p-6">
              <h3 className="text-xl font-semibold">Who buys</h3>
              <BulletList items={databaseBuyers} />
              <div className="mt-8 rounded-xl border border-amber-300/15 bg-amber-300/[0.04] p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-amber-200/70">Economic proof</p>
                <p className="mt-2 text-sm leading-relaxed text-[#faf9f6]/70">
                  Translate the speedup into lower cloud spend, more workload per
                  machine, lower tail latency, or delayed infrastructure expansion.
                </p>
              </div>
            </article>
            <article className="rounded-2xl border border-cyan-300/20 bg-cyan-300/[0.04] p-6">
              <Eyebrow>Example pilot gate</Eyebrow>
              <p className="mt-3 text-sm leading-relaxed text-[#faf9f6]/55">
                Freeze the actual customer threshold before any candidate is run.
              </p>
              <dl className="mt-5 space-y-4 text-sm">
                <div><dt className="text-[#faf9f6]/40">Correctness</dt><dd className="mt-1 text-[#faf9f6]/85">Zero output, schema, semantic, or exception mismatches</dd></div>
                <div><dt className="text-[#faf9f6]/40">Primary objective</dt><dd className="mt-1 text-[#faf9f6]/85">For example, at least 1.05× geometric-mean speedup</dd></div>
                <div><dt className="text-[#faf9f6]/40">Regression bound</dt><dd className="mt-1 text-[#faf9f6]/85">No critical workload below the predeclared floor</dd></div>
                <div><dt className="text-[#faf9f6]/40">Memory</dt><dd className="mt-1 text-[#faf9f6]/85">Within the agreed peak-memory cap</dd></div>
                <div><dt className="text-[#faf9f6]/40">Confirmation</dt><dd className="mt-1 text-[#faf9f6]/85">Clean build, held-out workload, low-contention rerun</dd></div>
              </dl>
            </article>
          </div>

          <article className="mt-5 rounded-2xl border border-[#f0eee6]/10 bg-[#1f1e1d] p-6 md:p-8">
            <h3 className="text-2xl font-semibold">How to run the database experiment</h3>
            <div className="mt-2 max-w-4xl">
              <NumberedList items={databaseProtocol} />
            </div>
          </article>
        </div>
      </section>

      <section id="logistics" className="scroll-mt-28">
        <div className="mx-auto max-w-6xl px-6 py-20 md:py-28">
          <Eyebrow amber>Lane 02 · Logistics</Eyebrow>
          <div className="mt-5 grid gap-6 lg:grid-cols-[1fr_0.8fr]">
            <h2 className="text-4xl font-semibold tracking-tight md:text-5xl">
              Potentially larger wins. Harder reality.
            </h2>
            <p className="text-lg leading-relaxed text-[#faf9f6]/65">
              Logistics becomes attractive only with a design partner that can
              supply real replay data, the complete constraint system, and the
              business objective used in production.
            </p>
          </div>

          <div className="mt-12 grid gap-5 lg:grid-cols-3">
            <article className="rounded-2xl border border-[#f0eee6]/10 bg-[#faf9f6]/[0.025] p-6">
              <h3 className="text-xl font-semibold">Algorithms to search</h3>
              <BulletList items={logisticsTargets} />
            </article>
            <article className="rounded-2xl border border-[#f0eee6]/10 bg-[#faf9f6]/[0.025] p-6">
              <h3 className="text-xl font-semibold">Who buys</h3>
              <BulletList items={logisticsBuyers} />
              <div className="mt-8 rounded-xl border border-amber-300/15 bg-amber-300/[0.04] p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-amber-200/70">Economic proof</p>
                <p className="mt-2 text-sm leading-relaxed text-[#faf9f6]/70">
                  Measure total route or operating cost while preserving service,
                  feasibility, operator control, and incident thresholds.
                </p>
              </div>
            </article>
            <article className="rounded-2xl border border-amber-300/20 bg-amber-300/[0.035] p-6">
              <Eyebrow amber>Required partner inputs</Eyebrow>
              <dl className="mt-5 space-y-4 text-sm">
                <div><dt className="text-[#faf9f6]/40">Replay</dt><dd className="mt-1 text-[#faf9f6]/85">Orders, locations, resources, travel times, outcomes</dd></div>
                <div><dt className="text-[#faf9f6]/40">Validator</dt><dd className="mt-1 text-[#faf9f6]/85">Capacity, windows, skills, breaks, compatibility, policy</dd></div>
                <div><dt className="text-[#faf9f6]/40">Objective</dt><dd className="mt-1 text-[#faf9f6]/85">The real business cost function, not a distance proxy</dd></div>
                <div><dt className="text-[#faf9f6]/40">Baseline</dt><dd className="mt-1 text-[#faf9f6]/85">The planner used today or a serious production candidate</dd></div>
                <div><dt className="text-[#faf9f6]/40">Safety</dt><dd className="mt-1 text-[#faf9f6]/85">Shadow mode, operator review, rollback, incidents</dd></div>
              </dl>
            </article>
          </div>

          <article className="mt-5 rounded-2xl border border-[#f0eee6]/10 bg-[#faf9f6]/[0.025] p-6 md:p-8">
            <h3 className="text-2xl font-semibold">How to run the logistics experiment</h3>
            <div className="mt-2 max-w-4xl">
              <NumberedList items={logisticsProtocol} />
            </div>
          </article>
        </div>
      </section>

      <section id="protocol" className="scroll-mt-28 border-y border-[#f0eee6]/10 bg-[#faf9f6]/[0.018]">
        <div className="mx-auto max-w-6xl px-6 py-20 md:py-28">
          <Eyebrow>The universal experiment contract</Eyebrow>
          <div className="mt-5 grid gap-6 lg:grid-cols-[1fr_0.8fr]">
            <h2 className="text-4xl font-semibold tracking-tight md:text-5xl">
              Search aggressively. Measure conservatively.
            </h2>
            <p className="text-lg leading-relaxed text-[#faf9f6]/65">
              The domain changes. The evidence rules do not. A candidate earns
              promotion only after it passes the contract that existed before the result.
            </p>
          </div>
          <div className="mt-12 grid gap-px overflow-hidden rounded-2xl border border-[#f0eee6]/10 bg-[#f0eee6]/10 md:grid-cols-2 lg:grid-cols-3">
            {universalSteps.map((step) => (
              <article key={step.number} className="bg-[#1f1e1d] p-6 md:p-7">
                <span className="font-mono text-xs text-cyan-300/55">{step.number}</span>
                <h3 className="mt-5 text-xl font-semibold">{step.title}</h3>
                <p className="mt-3 text-sm leading-relaxed text-[#faf9f6]/65">{step.body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-6xl gap-12 px-6 py-20 md:py-28 lg:grid-cols-[0.8fr_1.2fr]">
        <div>
          <Eyebrow>Resource and agent coordination</Eyebrow>
          <h2 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
            Parallel ideas. Serialized evidence.
          </h2>
          <p className="mt-5 leading-relaxed text-[#faf9f6]/65">
            More agents should increase mechanism diversity without contaminating
            builds, overwriting candidates, or turning machine contention into a fake speedup.
          </p>
        </div>
        <ul className="space-y-4">
          {resourceRules.map((rule) => (
            <li
              key={rule}
              className="flex gap-4 rounded-xl border border-[#f0eee6]/10 bg-[#faf9f6]/[0.025] p-5 leading-relaxed text-[#faf9f6]/75"
            >
              <span className="text-emerald-300">✓</span>
              {rule}
            </li>
          ))}
        </ul>
      </section>

      <section className="border-y border-[#f0eee6]/10 bg-cyan-300/[0.018]">
        <div className="mx-auto max-w-6xl px-6 py-20 md:py-28">
          <Eyebrow>What to sell now</Eyebrow>
          <h2 className="mt-5 max-w-4xl text-4xl font-semibold tracking-tight md:text-5xl">
            Start as a productized algorithm R&amp;D service.
          </h2>
          <p className="mt-5 max-w-2xl text-lg leading-relaxed text-[#faf9f6]/65">
            Revenue and customer evidence come before a fully autonomous platform.
          </p>
          <div className="mt-12 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {offers.map((offer) => (
              <article key={offer.number} className="rounded-2xl border border-[#f0eee6]/10 bg-[#1f1e1d] p-6">
                <span className="font-mono text-xs text-cyan-300/55">{offer.number}</span>
                <h3 className="mt-5 text-xl font-semibold">{offer.title}</h3>
                <p className="mt-3 text-sm leading-relaxed text-[#faf9f6]/65">{offer.body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-20 md:py-28">
        <Eyebrow amber>First 90 days</Eyebrow>
        <h2 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
          One proof. One buyer. Then repeat.
        </h2>
        <div className="mt-12 grid gap-5 md:grid-cols-3">
          {[
            ["0—30", "Prove databases", "Confirm the strongest database candidate, publish a readable case study, interview ten buyers, and define the paid sprint."],
            ["31—60", "Deliver a pilot", "Close one database or data-system engagement and measure value, integration effort, and willingness to continue."],
            ["61—90", "Find repeat demand", "Sell a second engagement in the winning segment. Open logistics only when a partner supplies real replay data and constraints."],
          ].map(([period, title, body]) => (
            <article key={period} className="rounded-2xl border border-[#f0eee6]/10 bg-[#faf9f6]/[0.025] p-6">
              <strong className="font-mono text-sm text-amber-300/75">{period}</strong>
              <h3 className="mt-5 text-2xl font-semibold">{title}</h3>
              <p className="mt-3 leading-relaxed text-[#faf9f6]/65">{body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="border-t border-[#f0eee6]/10">
        <div className="mx-auto max-w-5xl px-6 py-24 text-center">
          <Eyebrow>The claim boundary</Eyebrow>
          <h2 className="mx-auto mt-5 max-w-4xl text-4xl font-semibold tracking-tight md:text-6xl">
            A compiled patch is not an improvement. A benchmark win is not customer value.
          </h2>
          <p className="mx-auto mt-6 max-w-3xl text-lg leading-relaxed text-[#faf9f6]/65">
            Open Discovery claims only what the frozen evaluator,
            representative workload, resource evidence, and independent confirmation support.
          </p>
          <div className="mt-9 flex flex-col justify-center gap-3 sm:flex-row">
            <Link
              href="/cofounder"
              className="rounded-full bg-cyan-300 px-6 py-3 font-semibold text-[#1f1e1d] hover:bg-cyan-200"
            >
              Build this with us
            </Link>
            <Link
              href="/contribute"
              className="rounded-full border border-[#f0eee6]/15 px-6 py-3 font-medium text-[#faf9f6]/85 hover:border-cyan-300/35 hover:text-cyan-200"
            >
              Bring us a bottleneck
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
}
