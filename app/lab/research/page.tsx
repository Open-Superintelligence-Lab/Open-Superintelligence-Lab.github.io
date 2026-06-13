"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useEffect, useState } from "react";

type Paper = {
  id: string;
  title: string;
  status: string;
  authors: string[];
  created: string;
  experiments: { goal_id: string; label: string; result: string }[];
  manuscript_md: string;
  assets?: { pdf: string | null; tex: string | null; images: string[] } | null;
};

type ResearchData = {
  generated_at: string;
  papers: Paper[];
  n_experiments: number;
};

const paperStatusColor = (status: string) => {
  const s = status.toLowerCase();
  if (s.includes("win") || s.includes("done")) return "text-emerald-400 border-emerald-400/30 bg-emerald-400/10";
  if (s.includes("pending") || s.includes("experiments")) return "text-amber-300 border-amber-300/30 bg-amber-300/10";
  if (s.includes("fail") || s.includes("broken")) return "text-rose-400 border-rose-400/30 bg-rose-400/10";
  return "text-cyan-300 border-cyan-300/30 bg-cyan-300/10";
};

export default function ResearchPage() {
  const [data, setData] = useState<ResearchData | null>(null);
  const [err, setErr] = useState(false);
  const [expanded, setExpanded] = useState<string[]>([]);

  useEffect(() => {
    fetch("/data/lab/summary.json")
      .then((r) => {
        if (!r.ok) throw new Error("not found");
        return r.json();
      })
      .then((j) => {
        if (!j || !Array.isArray(j.papers)) throw new Error("bad shape");
        setData(j);
      })
      .catch(() => setErr(true));
  }, []);

  const toggle = (id: string) => {
    setExpanded((prev) => (prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]));
  };

  return (
    <main className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]">
      <div className="container mx-auto max-w-6xl px-6 py-16">
        <h1 className="text-2xl font-semibold tracking-tight">Research</h1>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[#faf9f6]/70">
          Living manuscripts and their attached figures.
        </p>

        {err && <p className="mt-12 text-rose-400">Could not load research data.</p>}
        {!data && !err && <p className="mt-12 text-[#faf9f6]/50">Loading…</p>}

        {data && (
          <>
            <div className="mt-10 space-y-4">
              {data.papers.map((paper) => {
                const isOpen = expanded.includes(paper.id);
                return (
                  <article
                    key={paper.id}
                    className="rounded-xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5 md:p-6"
                  >
                    <button
                      type="button"
                      onClick={() => toggle(paper.id)}
                      className="flex w-full items-start justify-between gap-4 text-left"
                    >
                      <div>
                        <h2 className="text-lg font-semibold md:text-xl">{paper.title}</h2>
                        <p className="mt-2 text-sm text-[#faf9f6]/55">
                          {paper.authors.join(", ")} · {paper.created} · {paper.experiments.length} experiments
                        </p>
                      </div>
                      <span className={`rounded-full border px-3 py-1 text-xs uppercase tracking-wider ${paperStatusColor(paper.status)}`}>
                        {paper.status}
                      </span>
                    </button>

                    {(paper.assets?.pdf || paper.assets?.tex) && (
                      <div className="mt-4 flex flex-wrap items-center gap-3">
                        {paper.assets?.pdf && (
                          <a
                            href={paper.assets.pdf}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-2 rounded-lg border border-cyan-300/50 bg-cyan-300/15 px-4 py-2 text-sm font-semibold text-cyan-200 transition hover:bg-cyan-300/25"
                          >
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                              <path d="M14 2v6h6" />
                            </svg>
                            Open PDF
                          </a>
                        )}
                        {paper.assets?.tex && (
                          <a
                            href={paper.assets.tex}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="rounded-lg border border-[#f0eee6]/20 px-4 py-2 text-sm text-[#faf9f6]/75 transition hover:border-cyan-300/40 hover:text-cyan-300"
                          >
                            LaTeX source
                          </a>
                        )}
                        <span className="text-xs text-[#faf9f6]/40">rebuilt from the live manuscript</span>
                      </div>
                    )}

                    {isOpen && paper.assets && paper.assets.images.length > 0 && (
                      <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2">
                        {paper.assets.images.map((src) => (
                          <a key={src} href={src} target="_blank" rel="noopener noreferrer" className="rounded-lg border border-[#f0eee6]/10 bg-white/95 p-2">
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img src={src} alt="paper figure" className="w-full rounded" />
                          </a>
                        ))}
                      </div>
                    )}

                    {isOpen && (
                      <div className="mt-6 rounded-xl border border-[#f0eee6]/10 bg-[#1f1e1d] p-5">
                        <div
                          className="prose prose-invert max-w-none text-[#faf9f6]/85
                            [&_a]:text-cyan-300 [&_a]:underline-offset-4 [&_a:hover]:text-cyan-200
                            [&_h1]:mt-0 [&_h1]:text-2xl [&_h1]:font-semibold [&_h1]:text-[#faf9f6]
                            [&_h2]:mt-8 [&_h2]:text-xl [&_h2]:font-semibold [&_h2]:text-[#faf9f6]
                            [&_h3]:mt-6 [&_h3]:text-lg [&_h3]:font-semibold [&_h3]:text-[#faf9f6]
                            [&_p]:leading-7 [&_p]:text-[#faf9f6]/85
                            [&_ul]:my-4 [&_ol]:my-4 [&_ul]:space-y-2 [&_ol]:space-y-2
                            [&_li]:leading-7 [&_strong]:text-[#faf9f6] [&_em]:text-[#faf9f6]/90
                            [&_code]:rounded [&_code]:bg-[#f0eee6]/[0.06] [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:text-sm
                            [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:border [&_pre]:border-[#f0eee6]/10 [&_pre]:bg-[#0f0e0d] [&_pre]:p-4
                            [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_blockquote]:border-l-[#f0eee6]/20 [&_blockquote]:text-[#faf9f6]/70
                            [&_hr]:border-[#f0eee6]/10 [&_table]:block [&_table]:overflow-x-auto [&_table]:border-collapse
                            [&_thead_tr]:border-b [&_thead_tr]:border-[#f0eee6]/10 [&_th]:px-3 [&_th]:py-2 [&_td]:px-3 [&_td]:py-2"
                        >
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{paper.manuscript_md}</ReactMarkdown>
                        </div>
                      </div>
                    )}
                  </article>
                );
              })}
            </div>
          </>
        )}
      </div>
    </main>
  );
}
