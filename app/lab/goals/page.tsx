"use client";

import { Check, Circle } from "lucide-react";
import { useEffect, useState } from "react";

type Milestone = {
  label: string;
  done: boolean;
};

type PublicGoal = {
  id: string;
  title: string;
  status: string;
  description: string;
  why: string;
  milestones: Milestone[];
};

type PrivateGoal = {
  id: string;
  title: string;
  owner: string;
  status: string;
  next: string;
};

type GoalsData = {
  mission: string;
  public_goals: PublicGoal[];
  private_goals: PrivateGoal[];
  strategy_ladder?: string[];
  generated_at: string;
};

const goalStatusColor = (status: string) => {
  const s = status.toLowerCase();
  if (s.includes("active")) return "text-cyan-300 border-cyan-300/30 bg-cyan-300/10";
  if (s.includes("forming")) return "text-amber-300 border-amber-300/30 bg-amber-300/10";
  if (s.includes("done") || s.includes("shipped")) return "text-emerald-400 border-emerald-400/30 bg-emerald-400/10";
  return "text-[#faf9f6]/60 border-[#f0eee6]/10 bg-[#f0eee6]/[0.04]";
};

export default function GoalsPage() {
  const [data, setData] = useState<GoalsData | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    fetch("/data/lab/goals.json")
      .then((r) => {
        if (!r.ok) throw new Error("not found");
        return r.json();
      })
      .then((j) => {
        if (!j || !Array.isArray(j.public_goals) || !Array.isArray(j.private_goals)) {
          throw new Error("bad shape");
        }
        setData(j);
      })
      .catch(() => setErr(true));
  }, []);

  return (
    <main className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]">
      <div className="container mx-auto max-w-6xl px-6 py-16">
        <h1 className="text-2xl font-semibold tracking-tight">Goals</h1>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[#faf9f6]/70">
          Public mission, strategy, and goals.
        </p>

        {err && <p className="mt-12 text-rose-400">Could not load goals data.</p>}
        {!data && !err && <p className="mt-12 text-[#faf9f6]/50">Loading…</p>}

        {data && (
          <>
            <section className="mt-10 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6 md:p-8">
              <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/70">Mission</p>
              <h2 className="mt-3 text-2xl font-semibold md:text-4xl">{data.mission}</h2>
            </section>

            {Array.isArray(data.strategy_ladder) && data.strategy_ladder.length > 0 && (
              <section className="mt-10 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6 md:p-8">
                <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/70">Strategy ladder</p>
                <ul className="mt-4 space-y-3 text-sm leading-relaxed text-[#faf9f6]/75">
                  {data.strategy_ladder.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </section>
            )}

            <section className="mt-10">
              <div className="flex items-baseline justify-between gap-4">
                <h2 className="text-xl font-semibold md:text-2xl">Public goals</h2>
                <span className="text-xs text-[#faf9f6]/40">{data.public_goals.length} goals</span>
              </div>
              <div className="mt-4 space-y-4">
                {data.public_goals.map((goal) => (
                  <article
                    key={goal.id}
                    className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5 md:p-6"
                  >
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <h3 className="text-lg font-semibold md:text-xl">{goal.title}</h3>
                        <p className="mt-2 max-w-4xl text-sm leading-relaxed text-[#faf9f6]/70">
                          {goal.description}
                        </p>
                      </div>
                      <span className={`rounded-full border px-3 py-1 text-xs uppercase tracking-wider ${goalStatusColor(goal.status)}`}>
                        {goal.status}
                      </span>
                    </div>

                    <div className="mt-4 rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-4">
                      <p className="text-xs uppercase tracking-widest text-[#faf9f6]/40">Why</p>
                      <p className="mt-2 text-sm leading-relaxed text-[#faf9f6]/80">{goal.why}</p>
                    </div>

                    <div className="mt-5">
                      <p className="text-xs uppercase tracking-widest text-[#faf9f6]/40">Milestones</p>
                      <ul className="mt-3 space-y-3">
                        {goal.milestones.map((milestone) => (
                          <li key={milestone.label} className="flex items-start gap-3 text-sm text-[#faf9f6]/80">
                            {milestone.done ? (
                              <Check className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
                            ) : (
                              <Circle className="mt-0.5 h-4 w-4 shrink-0 text-[#faf9f6]/30" />
                            )}
                            <span className={milestone.done ? "text-[#faf9f6]/85" : "text-[#faf9f6]/65"}>
                              {milestone.label}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </article>
                ))}
              </div>
            </section>

            <p className="mt-8 text-xs text-[#faf9f6]/40">
              updated {data.generated_at?.slice(0, 16).replace("T", " ") || "—"}
            </p>
          </>
        )}
      </div>
    </main>
  );
}
