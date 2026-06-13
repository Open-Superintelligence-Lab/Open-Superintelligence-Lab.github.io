"use client";

import { useEffect, useState } from "react";

type Milestone = {
  label: string;
  done: boolean;
};

type GoalsData = {
  mission: string;
  public_goals: {
    id: string;
    title: string;
    status: string;
    description: string;
    why: string;
    milestones: Milestone[];
  }[];
  private_goals: unknown[];
  generated_at: string;
};

export default function ComputePage() {
  const [data, setData] = useState<GoalsData | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    fetch("/data/lab/goals.json")
      .then((r) => {
        if (!r.ok) throw new Error("not found");
        return r.json();
      })
      .then((j) => {
        if (!j || !Array.isArray(j.public_goals)) throw new Error("bad shape");
        setData(j);
      })
      .catch(() => setErr(true));
  }, []);

  const infraGoal = data?.public_goals.find((goal) => goal.id === "training-infrastructure");

  return (
    <main className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]">
      <div className="container mx-auto max-w-6xl px-6 py-16">
        <h1 className="text-2xl font-semibold tracking-tight">Compute</h1>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[#faf9f6]/70">
          Donated GPUs pull screening jobs from a queue and push results back into the lab.
        </p>

        {err && <p className="mt-12 text-rose-400">Could not load compute data.</p>}
        {!data && !err && <p className="mt-12 text-[#faf9f6]/50">Loading…</p>}

        {data && (
          <>
            <section className="mt-10 grid gap-4 md:grid-cols-2">
              <article className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6">
                <p className="text-xs uppercase tracking-widest text-cyan-300/70">How it works</p>
                <p className="mt-3 text-sm leading-relaxed text-[#faf9f6]/75">
                  The infrastructure keeps one shared training target and hands out screening jobs in
                  queue order. A GPU does not need to own the whole stack; it only needs to run the
                  next job cleanly and return the artifact.
                </p>
              </article>

              <article className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6">
                <p className="text-xs uppercase tracking-widest text-cyan-300/70">Coming soon</p>
                <h2 className="mt-3 text-lg font-semibold">Attach your GPU</h2>
                <p className="mt-2 text-sm leading-relaxed text-[#faf9f6]/75">
                  The next step is a simple attach flow: connect a machine, accept queued screening
                  jobs, and let the lab schedule work across donated compute.
                </p>
              </article>
            </section>

            {infraGoal && (
              <section className="mt-10 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6 md:p-8">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-widest text-cyan-300/70">Training infrastructure goal</p>
                    <h2 className="mt-2 text-2xl font-semibold md:text-3xl">{infraGoal.title}</h2>
                    <p className="mt-3 max-w-4xl text-sm leading-relaxed text-[#faf9f6]/70">
                      {infraGoal.description}
                    </p>
                  </div>
                  <span className="rounded-full border border-cyan-300/30 bg-cyan-300/10 px-3 py-1 text-xs uppercase tracking-wider text-cyan-300">
                    {infraGoal.status}
                  </span>
                </div>

                <div className="mt-5 rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-4">
                  <p className="text-xs uppercase tracking-widest text-[#faf9f6]/40">Why</p>
                  <p className="mt-2 text-sm leading-relaxed text-[#faf9f6]/80">{infraGoal.why}</p>
                </div>

                <div className="mt-6">
                  <p className="text-xs uppercase tracking-widest text-[#faf9f6]/40">Milestones</p>
                  <ul className="mt-3 space-y-3">
                    {infraGoal.milestones.map((milestone) => (
                      <li key={milestone.label} className="flex items-start gap-3 text-sm text-[#faf9f6]/80">
                        <span
                          className={`mt-0.5 h-3 w-3 shrink-0 rounded-full border ${
                            milestone.done
                              ? "border-emerald-400 bg-emerald-400"
                              : "border-[#faf9f6]/35 bg-transparent"
                          }`}
                        />
                        <span className={milestone.done ? "text-[#faf9f6]/85" : "text-[#faf9f6]/65"}>
                          {milestone.label}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              </section>
            )}

            <section className="mt-10 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6 md:p-8">
              <p className="text-xs uppercase tracking-widest text-cyan-300/70">State of play</p>
              <p className="mt-3 max-w-4xl text-sm leading-relaxed text-[#faf9f6]/75">
                No fake queue depth, no fake throughput, no fake attach rate. This page only
                describes the model and surfaces the milestones that already exist in the public goal
                data.
              </p>
            </section>

            <p className="mt-8 text-xs text-[#faf9f6]/40">
              updated {data.generated_at.slice(0, 16).replace("T", " ")}
            </p>
          </>
        )}
      </div>
    </main>
  );
}
