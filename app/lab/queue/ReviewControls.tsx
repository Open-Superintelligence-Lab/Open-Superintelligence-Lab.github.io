"use client";

import type { ChangeEvent } from "react";
import { useEffect, useRef, useState } from "react";

type QueueSpec = {
  id: string;
  title: string;
  idea?: string;
  status: string;
  gpu_vram_gb: number | null;
  hours: number | null;
  plain?: string;
};

type ReviewDecision = "approve" | "disapprove" | "keep";

type LiveQueueSpec = QueueSpec & {
  decision: string;
  note: string;
  updated_at: string;
  plain?: string;
};

type IdeaValidation = {
  slug: string;
  status: string;
  round: number;
  gates: {
    taste: "accept" | "revise" | "reject" | "pending";
    definition: "accept" | "revise" | "reject" | "pending";
    code: "accept" | "revise" | "reject" | "pending";
  };
  ran: boolean;
  evidence_verdict: string | null;
  timeline: Array<{
    ts: string;
    from: string;
    to: string;
    agent: string;
    note: string;
  }>;
};

function PlainExplainer({ text }: { text: string }) {
  if (!text) return null;
  const sentences = text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
  return (
    <div className="mt-3 rounded-lg border-l-2 border-cyan-300/40 bg-[#f0eee6]/[0.03] px-3.5 py-3">
      <p className="text-[11px] uppercase tracking-[0.18em] text-cyan-300/70">In plain words</p>
      <ul className="mt-2 space-y-1.5">
        {sentences.map((sentence, index) => (
          <li key={index} className="flex gap-2 text-[13px] leading-relaxed text-[#faf9f6]/80">
            <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-cyan-300/50" />
            <span>{sentence}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

const reviewApiUrl = "http://localhost:4500/api/specs";
const reviewPostUrl = "http://localhost:4500/api/review";
const ideaApiBaseUrl = "http://localhost:4500/api/idea";
const decisions: ReviewDecision[] = ["approve", "disapprove", "keep"];
const ideaValidationCache = new Map<string, Promise<IdeaValidation | null>>();

let liveSpecsPromise: Promise<LiveQueueSpec[] | null> | null = null;

function loadLiveSpecs() {
  if (!liveSpecsPromise) {
    liveSpecsPromise = fetch(reviewApiUrl, { cache: "no-store" })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load specs (${response.status})`);
        }
        return response.json() as Promise<LiveQueueSpec[]>;
      })
      .catch(() => null);
  }

  return liveSpecsPromise;
}

function normalizeDecision(value: string): ReviewDecision {
  return value === "approve" || value === "disapprove" ? value : "keep";
}

function normalizeStatus(status: string) {
  const lowered = status.toLowerCase();
  if (
    lowered.includes("done") ||
    lowered.includes("complete") ||
    lowered.includes("finished") ||
    lowered.includes("success") ||
    lowered.includes("failed")
  ) {
    return "done";
  }
  if (lowered.includes("claim") || lowered.includes("running") || lowered.includes("active") || lowered.includes("leased") || lowered.includes("progress")) {
    return "claimed";
  }
  return "queued";
}

function statusClass(status: string) {
  const normalized = normalizeStatus(status);
  if (normalized === "done") return "text-emerald-400 border-emerald-400/30 bg-emerald-400/10";
  if (normalized === "claimed") return "text-amber-300 border-amber-300/30 bg-amber-300/10";
  return "text-cyan-300 border-cyan-300/30 bg-cyan-300/10";
}

function ideaBadgeClass(verdict: "accept" | "revise" | "reject" | "pending") {
  if (verdict === "accept") return "border-emerald-400/35 bg-emerald-400/10 text-emerald-100";
  if (verdict === "reject") return "border-rose-400/35 bg-rose-400/10 text-rose-100";
  if (verdict === "revise") return "border-amber-300/35 bg-amber-300/10 text-amber-100";
  return "border-[#f0eee6]/10 bg-[#121316] text-[#faf9f6]/55";
}

function ideaStampSymbol(verdict: "accept" | "revise" | "reject" | "pending") {
  if (verdict === "accept") return "✓";
  if (verdict === "reject") return "✗";
  if (verdict === "revise") return "↻";
  return "—";
}

function formatIdeaTime(value: string) {
  if (!value) return "—";
  return value.replace("T", " ").replace("Z", "");
}

function loadIdeaValidation(slug: string) {
  const existing = ideaValidationCache.get(slug);
  if (existing) {
    return existing;
  }

  const promise = fetch(`${ideaApiBaseUrl}/${encodeURIComponent(slug)}`, { cache: "no-store" })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Failed to load idea validation (${response.status})`);
      }
      return response.json() as Promise<IdeaValidation>;
    })
    .catch(() => null);

  ideaValidationCache.set(slug, promise);
  return promise;
}

function ValidationPanel({ validation }: { validation: IdeaValidation }) {
  return (
    <details className="group mt-4 rounded-xl border border-[#f0eee6]/10 bg-[#121316] p-3">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-sm font-medium text-[#faf9f6]">
        <span>Validation &amp; history</span>
        <span className="text-[#faf9f6]/35 transition group-open:rotate-180">⌄</span>
      </summary>
      <div className="mt-4 space-y-4 text-sm">
        <div className="flex flex-wrap gap-2">
          {[
            ["Taste", validation.gates.taste],
            ["Definition", validation.gates.definition],
            ["Code", validation.gates.code],
          ].map(([label, verdict]) => (
            <span
              key={label}
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs uppercase tracking-[0.18em] ${ideaBadgeClass(verdict as IdeaValidation["gates"]["taste"])}`}
            >
              <span aria-hidden className="text-sm leading-none">
                {ideaStampSymbol(verdict as IdeaValidation["gates"]["taste"])}
              </span>
              <span>{label}</span>
            </span>
          ))}
        </div>

        <div className="flex flex-wrap gap-3 text-xs uppercase tracking-[0.18em] text-[#faf9f6]/55">
          <span>Pipeline: {validation.status}</span>
          <span>Round {validation.round}</span>
          {validation.ran ? <span>Evidence: {validation.evidence_verdict || "—"}</span> : null}
        </div>

        {validation.timeline.length > 0 ? (
          <ul className="space-y-2 border-t border-[#f0eee6]/10 pt-3 text-xs leading-5 text-[#faf9f6]/72">
            {validation.timeline.map((entry, index) => (
              <li key={`${entry.ts}-${index}`} className="rounded-lg bg-[#f0eee6]/[0.03] px-3 py-2">
                <span className="text-[#faf9f6]/45">{formatIdeaTime(entry.ts)}</span>
                <span className="px-1.5 text-[#faf9f6]/30">·</span>
                <span>{entry.from}</span>
                <span className="px-1 text-[#faf9f6]/30">→</span>
                <span>{entry.to}</span>
                <span className="px-1.5 text-[#faf9f6]/30">·</span>
                <span>{entry.agent}</span>
                <span className="px-1.5 text-[#faf9f6]/30">·</span>
                <span>{entry.note}</span>
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </details>
  );
}

function formatTimestamp(value: string) {
  if (!value) return "";
  return new Date(value).toLocaleString();
}

function buttonClass(decision: ReviewDecision, active: ReviewDecision) {
  const base = "rounded-full border px-3 py-2 text-xs font-medium capitalize transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300/60";
  if (decision !== active) {
    return `${base} border-[#f0eee6]/10 bg-[#121316] text-[#faf9f6]/72 hover:border-[#f0eee6]/20 hover:bg-[#f0eee6]/[0.04]`;
  }
  if (decision === "approve") {
    return `${base} border-emerald-400/50 bg-emerald-400/15 text-emerald-200`;
  }
  if (decision === "disapprove") {
    return `${base} border-rose-400/50 bg-rose-400/15 text-rose-200`;
  }
  return `${base} border-amber-300/50 bg-amber-300/15 text-amber-100`;
}

export default function ReviewControls({ spec, idea }: { spec: QueueSpec; idea?: string }) {
  const [liveSpec, setLiveSpec] = useState<LiveQueueSpec | null>(null);
  const [ideaValidation, setIdeaValidation] = useState<IdeaValidation | null>(null);
  const [decision, setDecision] = useState<ReviewDecision>("keep");
  const [note, setNote] = useState("");
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [savedLabel, setSavedLabel] = useState("");
  const saveTimerRef = useRef<number | null>(null);
  const savedStatusTimerRef = useRef<number | null>(null);
  const savingRef = useRef(false);
  const saveRequestedRef = useRef(false);
  const currentRef = useRef({ decision: "keep" as ReviewDecision, note: "" });
  const lastSavedRef = useRef({ decision: "keep" as ReviewDecision, note: "" });

  useEffect(() => {
    currentRef.current = { decision, note };
  }, [decision, note]);

  useEffect(() => {
    let cancelled = false;

    loadLiveSpecs().then((items) => {
      if (cancelled || !items) {
        return;
      }

      const next = items.find((item) => item.id === spec.id);
      if (!next) {
        return;
      }

      const nextDecision = normalizeDecision(next.decision);
      const nextNote = typeof next.note === "string" ? next.note : "";
      setLiveSpec(next);
      setDecision(nextDecision);
      setNote(nextNote);
      currentRef.current = { decision: nextDecision, note: nextNote };
      lastSavedRef.current = { decision: nextDecision, note: nextNote };
      setSavedLabel(next.updated_at ? formatTimestamp(next.updated_at) : "");
      setSaveStatus("idle");
    });

    return () => {
      cancelled = true;
    };
  }, [spec.id]);

  useEffect(() => {
    let cancelled = false;

    if (!idea) {
      setIdeaValidation(null);
      return () => {
        cancelled = true;
      };
    }

    setIdeaValidation(null);
    loadIdeaValidation(idea).then((validation) => {
      if (!cancelled && validation) {
        setIdeaValidation(validation);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [idea]);

  useEffect(() => {
    return () => {
      if (saveTimerRef.current !== null) {
        window.clearTimeout(saveTimerRef.current);
      }
      if (savedStatusTimerRef.current !== null) {
        window.clearTimeout(savedStatusTimerRef.current);
      }
    };
  }, []);

  const queueSave = () => {
    if (!liveSpec) {
      return;
    }

    saveRequestedRef.current = true;
    setSaveStatus("saving");

    if (saveTimerRef.current !== null) {
      window.clearTimeout(saveTimerRef.current);
    }

    saveTimerRef.current = window.setTimeout(() => {
      saveTimerRef.current = null;
      void flushSave();
    }, 350);
  };

  const flushSave = async () => {
    if (!liveSpec || !saveRequestedRef.current || savingRef.current) {
      return;
    }

    const next = currentRef.current;
    const last = lastSavedRef.current;
    if (next.decision === last.decision && next.note === last.note) {
      saveRequestedRef.current = false;
      setSaveStatus("idle");
      return;
    }

    savingRef.current = true;
    let savedSuccessfully = false;

    try {
      const response = await fetch(reviewPostUrl, {
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify({
          id: spec.id,
          decision: next.decision,
          note: next.note,
        }),
      });

      if (!response.ok) {
        throw new Error(`Save failed (${response.status})`);
      }

      const updated = (await response.json()) as { updated_at?: string };
      lastSavedRef.current = next;
      saveRequestedRef.current = false;
      setSaveStatus("saved");
      setSavedLabel(updated.updated_at ? formatTimestamp(updated.updated_at) : "Saved");
      savedSuccessfully = true;

      if (savedStatusTimerRef.current !== null) {
        window.clearTimeout(savedStatusTimerRef.current);
      }
      savedStatusTimerRef.current = window.setTimeout(() => {
        setSaveStatus("idle");
      }, 1200);
    } catch {
      setSaveStatus("error");
      saveRequestedRef.current = false;
    } finally {
      savingRef.current = false;
      if (savedSuccessfully) {
        const latest = currentRef.current;
        const last = lastSavedRef.current;
        if (latest.decision !== last.decision || latest.note !== last.note) {
          saveRequestedRef.current = true;
          if (saveTimerRef.current !== null) {
            window.clearTimeout(saveTimerRef.current);
          }
          saveTimerRef.current = window.setTimeout(() => {
            saveTimerRef.current = null;
            void flushSave();
          }, 350);
        }
      }
    }
  };

  const onDecision = (nextDecision: ReviewDecision) => {
    setDecision(nextDecision);
    queueSave();
  };

  const onNoteChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    setNote(event.target.value);
    queueSave();
  };

  const validation = idea && ideaValidation ? ideaValidation : null;

  if (!liveSpec) {
    return (
      <article className="rounded-lg border border-[#f0eee6]/8 bg-[#1f1e1d]/50 p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold leading-snug">{spec.title}</h3>
            <p className="mt-1 text-xs text-[#faf9f6]/50">{spec.id}</p>
          </div>
          <span className={`rounded-full border px-3 py-1 text-[11px] uppercase tracking-wider ${statusClass(spec.status)}`}>
            {spec.status}
          </span>
        </div>
        <div className="mt-3 flex flex-wrap gap-2 text-xs text-[#faf9f6]/55">
          <span className="rounded-full border border-[#f0eee6]/10 bg-[#f0eee6]/[0.03] px-2.5 py-1">
            {spec.gpu_vram_gb == null ? "VRAM —" : `${spec.gpu_vram_gb} GB VRAM`}
          </span>
          <span className="rounded-full border border-[#f0eee6]/10 bg-[#f0eee6]/[0.03] px-2.5 py-1">
            {spec.hours == null ? "Hours —" : `${spec.hours} hours`}
          </span>
        </div>
        <PlainExplainer text={spec.plain ?? ""} />
        {validation ? <ValidationPanel validation={validation} /> : null}
        {idea ? (
          <div className="mt-3">
            <a
              href={`/lab/report?id=${encodeURIComponent(idea)}`}
              className="inline-flex items-center text-sm font-medium text-cyan-300 transition hover:text-cyan-200"
            >
              Read full report →
            </a>
          </div>
        ) : null}
      </article>
    );
  }

  return (
    <article className="rounded-lg border border-[#f0eee6]/8 bg-[#1f1e1d]/50 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold leading-snug">{spec.title}</h3>
          <p className="mt-1 text-xs text-[#faf9f6]/50">{spec.id}</p>
        </div>
        <span className={`rounded-full border px-3 py-1 text-[11px] uppercase tracking-wider ${statusClass(spec.status)}`}>
          {spec.status}
        </span>
      </div>

      <div className="mt-3 flex flex-wrap gap-2 text-xs text-[#faf9f6]/55">
        <span className="rounded-full border border-[#f0eee6]/10 bg-[#f0eee6]/[0.03] px-2.5 py-1">
          {spec.gpu_vram_gb == null ? "VRAM —" : `${spec.gpu_vram_gb} GB VRAM`}
        </span>
        <span className="rounded-full border border-[#f0eee6]/10 bg-[#f0eee6]/[0.03] px-2.5 py-1">
          {spec.hours == null ? "Hours —" : `${spec.hours} hours`}
        </span>
      </div>

      <PlainExplainer text={liveSpec.plain ?? spec.plain ?? ""} />

      {validation ? <ValidationPanel validation={validation} /> : null}
      {idea ? (
        <div className="mt-3">
          <a
            href={`/lab/report?id=${encodeURIComponent(idea)}`}
            className="inline-flex items-center text-sm font-medium text-cyan-300 transition hover:text-cyan-200"
          >
            Read full report →
          </a>
        </div>
      ) : null}

      <div className="mt-4 rounded-xl border border-[#f0eee6]/10 bg-[#121316] p-3">
        <div className="flex flex-wrap items-center gap-2">
          {decisions.map((item) => (
            <button
              key={item}
              type="button"
              className={buttonClass(item, decision)}
              aria-pressed={item === decision}
              onClick={() => onDecision(item)}
            >
              {item}
            </button>
          ))}
        </div>

        <textarea
          value={note}
          onChange={onNoteChange}
          placeholder="Notes for orchestrator..."
          className="mt-3 min-h-28 w-full rounded-xl border border-[#f0eee6]/10 bg-[#1b1c1f] px-3 py-3 text-sm leading-relaxed text-[#faf9f6] placeholder:text-[#faf9f6]/35 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300/60"
        />

        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-[#faf9f6]/50">
          <span>Autosaves to <code className="text-[#faf9f6]/70">queue/reviews.json</code> through localhost:4500.</span>
          <span className="uppercase tracking-[0.2em] text-[#faf9f6]/55">
            {saveStatus === "saving"
              ? "Saving..."
              : saveStatus === "saved"
                ? "Saved"
                : saveStatus === "error"
                  ? "Save failed"
                  : savedLabel || "Ready"}
          </span>
        </div>
      </div>
    </article>
  );
}
