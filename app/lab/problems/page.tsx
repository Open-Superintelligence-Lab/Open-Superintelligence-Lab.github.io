"use client";

import { useEffect, useState } from "react";

type Problem = {
  id: string;
  title: string;
  status: string;
  answer: string | null;
  paper: string | null;
};

type ProblemsData = {
  generated_at: string;
  problems: Problem[];
};

const problemStatusColor = (status: string) => {
  const s = status.toLowerCase();
  if (s.includes("answered")) return "text-emerald-400 border-emerald-400/30 bg-emerald-400/10";
  if (s.includes("being-attacked")) return "text-cyan-300 border-cyan-300/30 bg-cyan-300/10";
  if (s.includes("open")) return "text-amber-300 border-amber-300/30 bg-amber-300/10";
  return "text-[#faf9f6]/60 border-[#f0eee6]/10 bg-[#f0eee6]/[0.04]";
};

export default function ProblemsPage() {
  const [data, setData] = useState<ProblemsData | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    fetch("/data/lab/problems.json")
      .then((r) => {
        if (!r.ok) throw new Error("not found");
        return r.json();
      })
      .then((j) => {
        if (!j || !Array.isArray(j.problems)) throw new Error("bad shape");
        setData(j);
      })
      .catch(() => setErr(true));
  }, []);

  return (
    <main className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]">
      <div className="container mx-auto max-w-6xl px-6 py-16">
        <h1 className="text-2xl font-semibold tracking-tight">Problems</h1>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[#faf9f6]/70">
          Open questions tracked as explicit problems.
        </p>

        {err && <p className="mt-12 text-rose-400">Could not load problems data.</p>}
        {!data && !err && <p className="mt-12 text-[#faf9f6]/50">Loading…</p>}

        {data && (
          <div className="mt-10 space-y-4">
            {data.problems.map((problem) => (
              <article
                key={problem.id}
                className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5 md:p-6"
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <h2 className="text-lg font-semibold md:text-xl">{problem.title}</h2>
                  <span className={`rounded-full border px-3 py-1 text-xs uppercase tracking-wider ${problemStatusColor(problem.status)}`}>
                    {problem.status}
                  </span>
                </div>

                {problem.answer && (
                  <div className="mt-4 rounded-lg border border-[#f0eee6]/10 bg-[#1f1e1d] p-4">
                    <p className="text-xs uppercase tracking-widest text-[#faf9f6]/40">Answer</p>
                    <p className="mt-2 text-sm leading-relaxed text-[#faf9f6]/80">{problem.answer}</p>
                  </div>
                )}

                <p className="mt-4 text-sm text-[#faf9f6]/60">
                  <span className="text-[#faf9f6]/40">Linked paper:</span>{" "}
                  {problem.paper ? (
                    <span className="rounded-full border border-[#f0eee6]/10 bg-[#f0eee6]/[0.04] px-2.5 py-0.5 font-mono text-xs text-[#faf9f6]/75">
                      {problem.paper}
                    </span>
                  ) : (
                    <span className="text-[#faf9f6]/35">none</span>
                  )}
                </p>
              </article>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
