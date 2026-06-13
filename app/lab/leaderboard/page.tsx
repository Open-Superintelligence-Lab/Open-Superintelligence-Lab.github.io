"use client";

import { useEffect, useState } from "react";

type Row = {
  rank: number;
  val_loss: number;
  delta: number;
  run: string;
  summary: string;
  date: string;
};
type Tier = { name: string; description: string; rows: Row[] };
type LeaderboardData = { generated_at: string; tiers: Tier[] };

const deltaColor = (d: number) => {
  if (d < 0) return "text-emerald-400";
  if (d > 0) return "text-rose-400";
  return "text-[#faf9f6]/50";
};
const deltaSign = (d: number) => (d > 0 ? "+" : "");

export default function Leaderboard() {
  const [data, setData] = useState<LeaderboardData | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    fetch("/data/lab/leaderboard.json")
      .then((r) => {
        if (!r.ok) throw new Error("not found");
        return r.json();
      })
      .then((j) => {
        if (!j || !Array.isArray(j.tiers)) throw new Error("bad shape");
        setData(j);
      })
      .catch(() => setErr(true));
  }, []);

  return (
    <main className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]">
      <div className="container mx-auto max-w-6xl px-6 py-16">
        <h1 className="text-2xl font-semibold tracking-tight">Leaderboard</h1>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[#faf9f6]/70">
          Best validation loss per tier, measured against the fixed baseline.
        </p>

        {err && (
          <p className="mt-12 rounded-xl border border-rose-400/30 bg-rose-400/10 p-6 text-rose-300">
            data not exported yet
          </p>
        )}
        {!data && !err && <p className="mt-12 text-[#faf9f6]/50">Loading…</p>}

        {data && data.tiers.length === 0 && (
          <p className="mt-12 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6 text-[#faf9f6]/60">
            data not exported yet
          </p>
        )}

        {data &&
          data.tiers.map((tier) => (
            <section
              key={tier.name}
              className="mt-10 rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6 md:p-8"
            >
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h2 className="text-xl font-semibold md:text-2xl">{tier.name}</h2>
                <span className="text-xs text-[#faf9f6]/40">{tier.rows.length} runs</span>
              </div>
              {tier.description && (
                <p className="mt-2 text-sm text-[#faf9f6]/60">{tier.description}</p>
              )}

              {tier.rows.length === 0 ? (
                <p className="mt-6 text-sm italic text-[#faf9f6]/40">no runs yet</p>
              ) : (
                <div className="mt-6 overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="border-b border-[#f0eee6]/10 text-xs uppercase tracking-widest text-[#faf9f6]/50">
                        <th className="py-2 pr-4">#</th>
                        <th className="py-2 pr-4">val loss</th>
                        <th className="py-2 pr-4">Δ</th>
                        <th className="py-2 pr-4">run</th>
                        <th className="py-2 pr-4">summary</th>
                        <th className="py-2">date</th>
                      </tr>
                    </thead>
                    <tbody>
                      {tier.rows.map((r) => {
                        const isFirst = r.rank === 1;
                        return (
                          <tr
                            key={`${tier.name}-${r.rank}-${r.run}`}
                            className={`border-b border-[#f0eee6]/5 ${
                              isFirst ? "bg-cyan-300/[0.04]" : ""
                            }`}
                          >
                            <td className="py-3 pr-4 font-mono text-[#faf9f6]/70">{r.rank}</td>
                            <td className="py-3 pr-4 font-mono">
                              {r.val_loss.toFixed(4)}
                              {isFirst && (
                                <span className="ml-2 rounded-full border border-cyan-300/30 bg-cyan-300/10 px-2 py-0.5 text-[10px] uppercase tracking-wider text-cyan-300">
                                  best
                                </span>
                              )}
                            </td>
                            <td className={`py-3 pr-4 font-mono ${deltaColor(r.delta)}`}>
                              {deltaSign(r.delta)}
                              {r.delta.toFixed(4)}
                            </td>
                            <td className="py-3 pr-4 text-[#faf9f6]/80">{r.run}</td>
                            <td className="py-3 pr-4 text-[#faf9f6]/60">{r.summary}</td>
                            <td className="py-3 text-[#faf9f6]/50">{r.date}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          ))}

        {data && (
          <p className="mt-8 text-xs text-[#faf9f6]/40">
            updated {data.generated_at?.slice(0, 16).replace("T", " ") || "—"}
          </p>
        )}
      </div>
    </main>
  );
}
