import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Cofounder | Open Discovery",
  description:
    "Join Open Discovery to build autonomous, evidence-first algorithm improvement for databases, logistics, and other software-intensive industries.",
};

const currentFacts = [
  ["Product wedge", "Verified algorithm improvement"],
  ["First market", "Database and data systems"],
  ["Next market", "Logistics with real replay data"],
  ["Stage", "Pre-revenue, proving capability"],
  ["Near-term target", "Paid pilots and a strong YC application"],
];

const principles = [
  {
    number: "01",
    title: "Build for measurable value",
    body: "Start where correctness is testable and a better algorithm changes compute cost, latency, capacity, route cost, utilization, or service quality.",
  },
  {
    number: "02",
    title: "Evidence before claims",
    body: "Freeze the baseline and evaluator, preserve every candidate, and independently confirm a winner before calling it an improvement.",
  },
  {
    number: "03",
    title: "Partners, not task takers",
    body: "Challenge assumptions, bring your own ideas, share hard decisions, and help choose the company we build together.",
  },
];

const trialWork = [
  "Talk to database and logistics buyers about expensive algorithmic bottlenecks.",
  "Help turn one real workload into a frozen, trustworthy evaluation contract.",
  "Search and test candidate improvements without contaminating the baseline.",
  "Package a verified result into a paid pilot that a customer can adopt.",
];

const workingAgreement = [
  "Equal founder ownership, with normal vesting, if we commit.",
  "No salary until funding, meaningful revenue, or YC.",
  "Part-time is fine at the start with reliable weekly contribution.",
  "Joint decisions on product, customers, research, and company direction.",
];

export default function CofounderPage() {
  return (
    <main className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]">
      <section className="relative overflow-hidden border-b border-[#f0eee6]/10 pt-28">
        <div className="pointer-events-none absolute left-1/2 top-0 h-[34rem] w-[50rem] -translate-x-1/2 rounded-full bg-cyan-400/[0.08] blur-[140px]" />
        <div className="relative mx-auto grid max-w-6xl gap-12 px-6 py-20 md:py-28 lg:grid-cols-[1.35fr_0.8fr] lg:items-center">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.3em] text-cyan-300/80">
              Open Discovery · Cofounder brief · 2026
            </p>
            <h1 className="mt-6 max-w-4xl text-5xl font-semibold leading-[1.02] tracking-tight md:text-7xl">
              Build the lab that makes algorithms{" "}
              <span className="text-cyan-300">better.</span>
            </h1>
            <p className="mt-7 max-w-3xl text-lg leading-relaxed text-[#faf9f6]/70 md:text-xl">
              Open Discovery is building AI that autonomously finds, implements,
              and verifies commercially useful algorithm improvements. Databases
              are the first proving ground. Logistics is the next design-partner
              market. The broader autonomous experiment vision comes later.
            </p>
            <div className="mt-10 flex flex-col gap-3 sm:flex-row">
              <Link
                href="/strategy"
                className="inline-flex items-center justify-center rounded-full bg-cyan-300 px-6 py-3 font-semibold text-[#1f1e1d] transition hover:bg-cyan-200"
              >
                Read the operating strategy →
              </Link>
              <Link
                href="/contribute"
                className="inline-flex items-center justify-center rounded-full border border-[#f0eee6]/15 bg-[#f0eee6]/[0.03] px-6 py-3 font-medium text-[#faf9f6]/85 transition hover:border-cyan-300/35 hover:text-cyan-200"
              >
                Start a conversation
              </Link>
            </div>
          </div>

          <aside className="rounded-2xl border border-[#f0eee6]/12 bg-[#faf9f6]/[0.035] p-6 shadow-2xl shadow-black/20 md:p-8">
            <p className="font-mono text-xs uppercase tracking-[0.3em] text-amber-300/80">
              What is true today
            </p>
            <dl className="mt-6 divide-y divide-[#f0eee6]/10">
              {currentFacts.map(([term, value]) => (
                <div key={term} className="grid gap-1 py-4 sm:grid-cols-[0.8fr_1.2fr] sm:gap-5">
                  <dt className="text-sm text-[#faf9f6]/45">{term}</dt>
                  <dd className="text-sm font-medium text-[#faf9f6]/90">{value}</dd>
                </div>
              ))}
            </dl>
            <p className="mt-5 text-sm leading-relaxed text-[#faf9f6]/50">
              No inflated traction claims. No promise that every search produces
              a winner. The company is being built around honest evidence.
            </p>
          </aside>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-20 md:py-28">
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-cyan-300/75">
          The thesis
        </p>
        <blockquote className="mt-6 max-w-5xl text-3xl font-medium leading-tight tracking-tight text-[#faf9f6]/95 md:text-5xl">
          AI coding agents are becoming capable of sustained search against
          objective evaluators. Turning that capability into trusted, deployable
          algorithmic gains can create value across every software-intensive industry.
        </blockquote>

        <div className="mt-16 grid gap-px overflow-hidden rounded-2xl border border-[#f0eee6]/10 bg-[#f0eee6]/10 md:grid-cols-3">
          {principles.map((principle) => (
            <article key={principle.number} className="bg-[#1f1e1d] p-7 md:p-8">
              <span className="font-mono text-sm text-cyan-300/60">{principle.number}</span>
              <h2 className="mt-5 text-xl font-semibold">{principle.title}</h2>
              <p className="mt-3 leading-relaxed text-[#faf9f6]/65">{principle.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="border-y border-[#f0eee6]/10 bg-[#faf9f6]/[0.018]">
        <div className="mx-auto grid max-w-6xl gap-12 px-6 py-20 md:py-28 lg:grid-cols-2">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.3em] text-cyan-300/75">
              Start by building together
            </p>
            <h2 className="mt-5 text-3xl font-semibold tracking-tight md:text-5xl">
              A focused working trial before a permanent commitment.
            </h2>
            <p className="mt-6 max-w-xl text-lg leading-relaxed text-[#faf9f6]/65">
              A serious candidate can begin part-time. The point is to discover
              whether we make good decisions, handle difficult evidence, and can
              turn technical work into something customers value.
            </p>
          </div>
          <ol className="space-y-4">
            {trialWork.map((item, index) => (
              <li
                key={item}
                className="flex gap-4 rounded-xl border border-[#f0eee6]/10 bg-[#1f1e1d] p-5"
              >
                <span className="font-mono text-sm text-cyan-300/60">0{index + 1}</span>
                <span className="leading-relaxed text-[#faf9f6]/80">{item}</span>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="mx-auto grid max-w-6xl gap-12 px-6 py-20 md:py-28 lg:grid-cols-[0.8fr_1.2fr]">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.3em] text-amber-300/80">
            Working agreement
          </p>
          <h2 className="mt-5 text-3xl font-semibold tracking-tight md:text-4xl">
            Build the company as equals.
          </h2>
        </div>
        <div>
          <ul className="grid gap-4 sm:grid-cols-2">
            {workingAgreement.map((item) => (
              <li
                key={item}
                className="rounded-xl border border-[#f0eee6]/10 bg-[#faf9f6]/[0.025] p-5 leading-relaxed text-[#faf9f6]/75"
              >
                <span className="mr-3 text-emerald-300">✓</span>
                {item}
              </li>
            ))}
          </ul>
          <p className="mt-8 text-lg leading-relaxed text-[#faf9f6]/65">
            This is for a technical founder who cares about algorithms, systems,
            optimization, scientific rigor, and enterprise problem-solving—and
            wants to build more than another AI wrapper.
          </p>
        </div>
      </section>

      <section className="border-t border-[#f0eee6]/10">
        <div className="mx-auto max-w-5xl px-6 py-24 text-center">
          <p className="font-mono text-xs uppercase tracking-[0.3em] text-cyan-300/75">
            The next conversation
          </p>
          <h2 className="mx-auto mt-5 max-w-3xl text-4xl font-semibold tracking-tight md:text-6xl">
            Bring a bottleneck, a hard question, or a better plan.
          </h2>
          <div className="mt-9 flex flex-col justify-center gap-3 sm:flex-row">
            <Link
              href="/strategy"
              className="rounded-full bg-cyan-300 px-6 py-3 font-semibold text-[#1f1e1d] hover:bg-cyan-200"
            >
              Inspect the strategy
            </Link>
            <Link
              href="/contribute"
              className="rounded-full border border-[#f0eee6]/15 px-6 py-3 font-medium text-[#faf9f6]/85 hover:border-cyan-300/35 hover:text-cyan-200"
            >
              Build with us
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
}
