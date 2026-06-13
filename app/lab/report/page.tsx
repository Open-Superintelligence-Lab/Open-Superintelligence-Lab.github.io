"use client";

import Link from "next/link";
import { Suspense, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { useSearchParams } from "next/navigation";
import { MarkdownRenderer } from "@/components/markdown-renderer";

type ReportSection = {
  key: string;
  title: string;
  markdown: string;
};

type ReportPayload = {
  slug: string;
  sections: ReportSection[];
};

const reportApiBaseUrl = "http://localhost:4500/api/report";
const reportApiFallbackBaseUrl = "http://127.0.0.1:4500/api/report";

function SectionShell({ title, children }: { title: string; children: ReactNode }) {
  return (
    <details className="group rounded-2xl border border-[#f0eee6]/10 bg-[#121316] px-4 py-3">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 text-sm font-medium text-[#faf9f6]">
        <span>{title}</span>
        <span className="text-[#faf9f6]/35 transition group-open:rotate-180">⌄</span>
      </summary>
      <div className="mt-4 border-t border-[#f0eee6]/10 pt-4">{children}</div>
    </details>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="rounded-2xl border border-[#f0eee6]/10 bg-[#121316] p-5 text-sm leading-relaxed text-[#faf9f6]/70">
      {message}
    </div>
  );
}

function LoadingShell() {
  return (
    <main className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]">
      <div className="mx-auto max-w-5xl px-6 py-10 md:px-8 md:py-14">
        <Link href="/lab/queue" className="inline-flex items-center text-sm font-medium text-cyan-300 transition hover:text-cyan-200">
          ← Back to queue
        </Link>
        <section className="mt-8 rounded-3xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5 shadow-[0_18px_60px_rgba(0,0,0,0.25)] md:p-6">
          <EmptyState message="Loading report…" />
        </section>
      </div>
    </main>
  );
}

function LabReportContent() {
  const searchParams = useSearchParams();
  const id = searchParams.get("id")?.trim() ?? "";
  const [report, setReport] = useState<ReportPayload | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");

  useEffect(() => {
    let cancelled = false;

    if (!id) {
      setReport(null);
      setStatus("idle");
      return () => {
        cancelled = true;
      };
    }

    setStatus("loading");
    setReport(null);

    const fetchReport = async () => {
      const urls = [reportApiBaseUrl, reportApiFallbackBaseUrl];
      let lastError: unknown = null;

      for (const baseUrl of urls) {
        try {
          const response = await fetch(`${baseUrl}/${encodeURIComponent(id)}`, { cache: "no-store" });
          if (!response.ok) {
            throw new Error(`Failed to load report (${response.status})`);
          }
          return (await response.json()) as ReportPayload;
        } catch (error) {
          lastError = error;
          if (baseUrl === urls[urls.length - 1]) {
            throw lastError;
          }
        }
      }

      throw lastError ?? new Error("Failed to load report");
    };

    fetchReport()
      .then((payload) => {
        if (!cancelled) {
          setReport(payload);
          setStatus("ready");
        }
      })
      .catch(() => {
        if (!cancelled) {
          setStatus("error");
          setReport(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [id]);

  const evidence = report?.sections.find((section) => section.key === "evidence") ?? null;
  const secondarySections = report?.sections.filter((section) => section.key !== "evidence") ?? [];
  const hasReport = status === "ready" && report != null;

  return (
    <main className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]">
      <div className="mx-auto max-w-5xl px-6 py-10 md:px-8 md:py-14">
        <Link href="/lab/queue" className="inline-flex items-center text-sm font-medium text-cyan-300 transition hover:text-cyan-200">
          ← Back to queue
        </Link>

        <header className="mt-6">
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/70">Lab</p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight md:text-4xl">Experiment report</h1>
          <p className="mt-3 max-w-3xl text-sm leading-relaxed text-[#faf9f6]/70">
            Read the written verdict for a queue item directly from its idea folder.
          </p>
        </header>

        <section className="mt-8 rounded-3xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-5 shadow-[0_18px_60px_rgba(0,0,0,0.25)] md:p-6">
          <div className="flex flex-wrap items-start justify-between gap-4 border-b border-[#f0eee6]/10 pb-4">
            <div>
              <p className="text-xs uppercase tracking-[0.28em] text-cyan-300/70">Report id</p>
              <h2 className="mt-1 text-lg font-semibold text-[#faf9f6]">{id || "No report selected"}</h2>
            </div>
            {report?.slug ? (
              <span className="rounded-full border border-[#f0eee6]/10 bg-[#121316] px-3 py-1 text-xs uppercase tracking-[0.18em] text-[#faf9f6]/55">
                {report.slug}
              </span>
            ) : null}
          </div>

          {!id ? (
            <div className="mt-5">
              <EmptyState message="Add ?id=<slug> to open a report." />
            </div>
          ) : status === "loading" ? (
            <div className="mt-5">
              <EmptyState message="Loading report…" />
            </div>
          ) : status === "error" ? (
            <div className="mt-5">
              <EmptyState message="report available on localhost when the review server is running" />
            </div>
          ) : hasReport ? (
            <div className="mt-5 space-y-4">
              {evidence ? (
                <section className="rounded-2xl border border-cyan-300/20 bg-[#101214] p-5">
                  <div className="mb-4 flex flex-wrap items-baseline justify-between gap-3">
                    <div>
                      <p className="text-xs uppercase tracking-[0.28em] text-cyan-300/70">Evidence</p>
                      <h3 className="mt-1 text-xl font-semibold text-[#faf9f6]">Headline verdict</h3>
                    </div>
                    <span className="rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-xs uppercase tracking-[0.18em] text-cyan-200">
                      Top line
                    </span>
                  </div>
                  <MarkdownRenderer content={evidence.markdown} />
                </section>
              ) : (
                <EmptyState message="No evidence.md file was found for this idea." />
              )}

              {secondarySections.length > 0 ? (
                <div className="space-y-3">
                  {secondarySections.map((section) => (
                    <SectionShell key={section.key} title={section.title}>
                      <MarkdownRenderer content={section.markdown} />
                    </SectionShell>
                  ))}
                </div>
              ) : null}
            </div>
          ) : (
            <div className="mt-5">
              <EmptyState message="report available on localhost when the review server is running" />
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

export default function LabReportPage() {
  return (
    <Suspense fallback={<LoadingShell />}>
      <LabReportContent />
    </Suspense>
  );
}
