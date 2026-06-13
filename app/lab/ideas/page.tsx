"use client";

import type { ChangeEvent } from "react";
import { useEffect, useRef, useState } from "react";

type Decision = "approve" | "disapprove" | "request-review" | "request-improvement";

type Suggestion = {
  id: string;
  name: string;
  status: string;
  round: number;
  summary: string;
  decision: Decision | null;
  note: string;
  updated_at: string;
};

type SuggestionsPayload = Suggestion[] | { generated_at?: string; ideas?: Suggestion[] };

const LIVE_API_URL = "http://localhost:4500/api/suggestions";
const REVIEW_API_URL = "http://localhost:4500/api/suggestion-review";
const FALLBACK_URL = "/data/lab/ideas.json";

const DECISIONS: Array<{ value: Decision; label: string }> = [
  { value: "approve", label: "Approve" },
  { value: "disapprove", label: "Disapprove" },
  { value: "request-review", label: "Request review" },
  { value: "request-improvement", label: "Request improvement" },
];

const decisionLabel = (decision: Decision | null) => {
  if (decision === "approve") return "Approve";
  if (decision === "disapprove") return "Disapprove";
  if (decision === "request-review") return "Request review";
  if (decision === "request-improvement") return "Request improvement";
  return "Unreviewed";
};

const formatTimestamp = (value: string) => {
  if (!value) return "";
  return value.replace("T", " ").replace(/([+-]\d{2})(\d{2})$/, "$1:$2");
};

const formatDisplayName = (name: string, id: string) => {
  const raw = (name || id || "").trim();
  const stripped = raw.replace(/^\s*\d+\s*[-—]\s*/, "");
  return stripped || raw || id;
};

const formatStatus = (status: string) => status.replace(/[-_]/g, " ");

const normalizeDecision = (value: unknown): Decision | null => {
  if (
    value === "approve" ||
    value === "disapprove" ||
    value === "request-review" ||
    value === "request-improvement"
  ) {
    return value;
  }
  return null;
};

const normalizeRound = (value: unknown) => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number.parseInt(value, 10);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return 0;
};

const normalizeSuggestion = (record: any): Suggestion | null => {
  const id = typeof record?.id === "string" ? record.id : "";
  if (!id) return null;

  return {
    id,
    name: typeof record?.name === "string" && record.name.trim() ? record.name.trim() : id,
    status: typeof record?.status === "string" && record.status.trim() ? record.status.trim() : "unknown",
    round: normalizeRound(record?.round),
    summary: typeof record?.summary === "string" ? record.summary.trim() : "",
    decision: normalizeDecision(record?.decision),
    note: typeof record?.note === "string" ? record.note : "",
    updated_at: typeof record?.updated_at === "string" ? record.updated_at : "",
  };
};

const normalizeSuggestionsPayload = (payload: SuggestionsPayload) => {
  const items = Array.isArray(payload) ? payload : Array.isArray(payload?.ideas) ? payload.ideas : [];
  return items.map(normalizeSuggestion).filter((item): item is Suggestion => item !== null);
};

const sortIdeas = (ideas: Suggestion[]) => {
  return [...ideas].sort((a, b) => {
    const aUnreviewed = !a.decision;
    const bUnreviewed = !b.decision;
    if (aUnreviewed !== bUnreviewed) {
      return aUnreviewed ? -1 : 1;
    }

    if (a.round !== b.round) {
      return b.round - a.round;
    }

    const statusCompare = a.status.localeCompare(b.status);
    if (statusCompare !== 0) {
      return statusCompare;
    }

    return a.id.localeCompare(b.id, undefined, { numeric: true, sensitivity: "base" });
  });
};

const statusTone = (status: string) => {
  const lowered = status.toLowerCase();
  if (lowered.includes("done") || lowered.includes("complete") || lowered.includes("success") || lowered.includes("closed")) {
    return "border-emerald-400/30 bg-emerald-400/10 text-emerald-100";
  }
  if (
    lowered.includes("fail") ||
    lowered.includes("reject") ||
    lowered.includes("blocked") ||
    lowered.includes("cancel")
  ) {
    return "border-rose-400/30 bg-rose-400/10 text-rose-100";
  }
  if (
    lowered.includes("run") ||
    lowered.includes("review") ||
    lowered.includes("plan") ||
    lowered.includes("taste") ||
    lowered.includes("need") ||
    lowered.includes("draft") ||
    lowered.includes("progress")
  ) {
    return "border-amber-300/30 bg-amber-300/10 text-amber-100";
  }
  return "border-cyan-300/30 bg-cyan-300/10 text-cyan-100";
};

const decisionTone = (decision: Decision | null) => {
  if (decision === "approve") {
    return "border-emerald-400/35 bg-emerald-400/12 text-emerald-100";
  }
  if (decision === "disapprove") {
    return "border-rose-400/35 bg-rose-400/12 text-rose-100";
  }
  if (decision === "request-review") {
    return "border-amber-300/35 bg-amber-300/12 text-amber-100";
  }
  if (decision === "request-improvement") {
    return "border-sky-300/35 bg-sky-300/12 text-sky-100";
  }
  return "border-[#f0eee6]/10 bg-[#121316] text-[#faf9f6]/55";
};

function IdeaCard({
  idea,
  live,
  onSaved,
}: {
  idea: Suggestion;
  live: boolean;
  onSaved: (id: string, review: { decision: Decision; note: string; updated_at: string }) => void;
}) {
  const [decision, setDecision] = useState<Decision | null>(idea.decision);
  const [note, setNote] = useState(idea.note);
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const saveTimerRef = useRef<number | null>(null);
  const savedStatusTimerRef = useRef<number | null>(null);
  const savingRef = useRef(false);
  const saveRequestedRef = useRef(false);
  const currentRef = useRef({ decision: idea.decision, note: idea.note });
  const lastSavedRef = useRef({ decision: idea.decision, note: idea.note });

  useEffect(() => {
    currentRef.current = { decision, note };
  }, [decision, note]);

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
    if (!live) {
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
    if (!live || !saveRequestedRef.current || savingRef.current) {
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
      const response = await fetch(REVIEW_API_URL, {
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify({
          id: idea.id,
          decision: next.decision,
          note: next.note,
        }),
      });

      if (!response.ok) {
        throw new Error(`Save failed (${response.status})`);
      }

      const updated = (await response.json()) as { updated_at?: string };
      const updatedAt = typeof updated.updated_at === "string" ? updated.updated_at : new Date().toISOString();
      const savedReview = {
        decision: next.decision as Decision,
        note: next.note,
        updated_at: updatedAt,
      };

      lastSavedRef.current = { decision: next.decision, note: next.note };
      saveRequestedRef.current = false;
      setSaveStatus("saved");
      onSaved(idea.id, savedReview);
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
        const lastSaved = lastSavedRef.current;
        if (latest.decision !== lastSaved.decision || latest.note !== lastSaved.note) {
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

  const onDecision = (nextDecision: Decision) => {
    setDecision(nextDecision);
    queueSave();
  };

  const onNoteChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    setNote(event.target.value);
    queueSave();
  };

  const title = formatDisplayName(idea.name, idea.id);
  const readyLabel = idea.decision ? "Ready" : "Unreviewed";
  const saveLabel =
    saveStatus === "saving"
      ? "Saving..."
      : saveStatus === "saved"
        ? "Saved"
        : saveStatus === "error"
          ? "Save failed"
          : live
            ? readyLabel
            : "Read only";

  return (
    <article className="rounded-[28px] border border-[#f0eee6]/10 bg-[#171615]/90 p-5 shadow-[0_18px_60px_rgba(0,0,0,0.28)] backdrop-blur-sm sm:p-6">
      <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-[#f0eee6]/10 bg-[#f0eee6]/[0.03] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-[#faf9f6]/60">
              {idea.id}
            </span>
            {idea.round > 0 ? (
              <span className="rounded-full border border-[#f0eee6]/10 bg-[#f0eee6]/[0.03] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-[#faf9f6]/55">
                r{idea.round}
              </span>
            ) : null}
          </div>

          <h2 className="mt-3 text-2xl font-semibold tracking-tight text-[#faf9f6] sm:text-[1.9rem]">
            {title}
          </h2>

          <div className="mt-3 flex flex-wrap gap-2">
            <span className={`rounded-full border px-3 py-1 text-xs uppercase tracking-[0.18em] ${statusTone(idea.status)}`}>
              {formatStatus(idea.status)}
            </span>
            <span className={`rounded-full border px-3 py-1 text-xs uppercase tracking-[0.18em] ${decisionTone(decision)}`}>
              {decisionLabel(decision)}
            </span>
          </div>
        </div>

        <div className="shrink-0 text-right text-xs uppercase tracking-[0.18em] text-[#faf9f6]/42">
          {saveLabel}
        </div>
      </div>

      <p className="mt-4 max-w-4xl text-sm leading-7 text-[#faf9f6]/76">
        {idea.summary || "No summary available."}
      </p>

      <div className="mt-5 flex flex-wrap gap-2">
        {DECISIONS.map((item) => {
          const active = decision === item.value;
          const tone =
            item.value === "approve"
              ? "border-emerald-400/35 bg-emerald-400/12 text-emerald-100"
              : item.value === "disapprove"
                ? "border-rose-400/35 bg-rose-400/12 text-rose-100"
                : item.value === "request-review"
                  ? "border-amber-300/35 bg-amber-300/12 text-amber-100"
                  : "border-sky-300/35 bg-sky-300/12 text-sky-100";
          return (
            <button
              key={item.value}
              type="button"
              disabled={!live}
              aria-pressed={active}
              onClick={() => onDecision(item.value)}
              className={`rounded-full border px-3.5 py-2 text-xs font-medium transition ${
                active
                  ? tone
                  : "border-[#f0eee6]/10 bg-[#121316] text-[#faf9f6]/74 hover:border-[#f0eee6]/20 hover:bg-[#f0eee6]/[0.04]"
              } disabled:cursor-not-allowed disabled:opacity-55`}
            >
              {item.label}
            </button>
          );
        })}
      </div>

      <textarea
        value={note}
        onChange={onNoteChange}
        disabled={!live}
        placeholder="Notes for the orchestrator..."
        className="mt-4 min-h-28 w-full rounded-2xl border border-[#f0eee6]/10 bg-[#121316] px-4 py-3 text-sm leading-relaxed text-[#faf9f6] placeholder:text-[#faf9f6]/34 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300/60 disabled:cursor-not-allowed disabled:opacity-55"
      />

      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-[#faf9f6]/48">
        <span>
          {live
            ? "Autosaves verdicts and notes to autoresearch/idea-reviews.json only."
            : "Static fallback only. Start localhost:4500 to enable autosave."}
        </span>
        <span className="uppercase tracking-[0.2em]">
          {saveLabel}
        </span>
      </div>
    </article>
  );
}

export default function IdeasPage() {
  const [ideas, setIdeas] = useState<Suggestion[]>([]);
  const [source, setSource] = useState<"loading" | "live" | "fallback" | "error">("loading");
  const [generatedAt, setGeneratedAt] = useState("");
  const [error, setError] = useState("");
  const [showReviewed, setShowReviewed] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadSuggestions() {
      try {
        const liveResponse = await fetch(LIVE_API_URL, { cache: "no-store" });
        if (!liveResponse.ok) {
          throw new Error(`Live suggestions unavailable (${liveResponse.status})`);
        }

        const livePayload = (await liveResponse.json()) as SuggestionsPayload;
        const normalized = normalizeSuggestionsPayload(livePayload);

        if (cancelled) {
          return;
        }

        setIdeas(normalized);
        setSource("live");
        setGeneratedAt("");
        setError("");
        return;
      } catch {
        // Fall through to the static snapshot.
      }

      try {
        const fallbackResponse = await fetch(FALLBACK_URL, { cache: "no-store" });
        if (!fallbackResponse.ok) {
          throw new Error(`Fallback data unavailable (${fallbackResponse.status})`);
        }

        const fallbackPayload = (await fallbackResponse.json()) as SuggestionsPayload;
        const normalized = normalizeSuggestionsPayload(fallbackPayload);
        const fallbackGeneratedAt =
          !Array.isArray(fallbackPayload) && typeof fallbackPayload.generated_at === "string"
            ? fallbackPayload.generated_at
            : "";

        if (cancelled) {
          return;
        }

        setIdeas(normalized);
        setSource("fallback");
        setGeneratedAt(fallbackGeneratedAt);
        setError("");
      } catch (fallbackError) {
        if (cancelled) {
          return;
        }

        setIdeas([]);
        setSource("error");
        setGeneratedAt("");
        setError(fallbackError instanceof Error ? fallbackError.message : "Failed to load ideas");
      }
    }

    void loadSuggestions();

    return () => {
      cancelled = true;
    };
  }, []);

  const sortedIdeas = sortIdeas(ideas);
  const actionableIdeas = sortedIdeas.filter((idea) => !idea.decision);
  const visibleIdeas = showReviewed ? sortedIdeas : actionableIdeas;
  const live = source === "live";
  const totalCount = sortedIdeas.length;
  const reviewedCount = totalCount - actionableIdeas.length;

  const updateIdeaReview = (id: string, review: { decision: Decision; note: string; updated_at: string }) => {
    setIdeas((current) =>
      current.map((item) =>
        item.id === id
          ? {
              ...item,
              decision: review.decision,
              note: review.note,
              updated_at: review.updated_at,
            }
          : item
      )
    );
  };

  const sourceLabel =
    source === "live"
      ? "Live API"
      : source === "fallback"
        ? "Static fallback"
        : source === "error"
          ? "Load failed"
          : "Loading";

  const loadedLabel =
    source === "fallback" && generatedAt
      ? `generated ${formatTimestamp(generatedAt)}`
      : source === "live"
        ? "synced from localhost:4500"
        : source === "error"
          ? "check localhost:4500 or the static snapshot"
          : "loading suggestions...";

  return (
    <main className="min-h-screen bg-[#101112] text-[#faf9f6]">
      <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 sm:py-8 lg:px-8 lg:py-10">
        <section
          className="relative overflow-hidden rounded-[28px] border border-[#f0eee6]/10 bg-[#171615] px-5 py-6 shadow-[0_18px_60px_rgba(0,0,0,0.28)] sm:px-7 sm:py-7"
          style={{
            backgroundImage:
              "radial-gradient(circle at top left, rgba(240, 217, 168, 0.12), transparent 30%), radial-gradient(circle at 88% 0%, rgba(140, 207, 151, 0.08), transparent 22%), radial-gradient(circle at 12% 110%, rgba(96, 165, 250, 0.05), transparent 26%)",
          }}
        >
          <div className="relative flex flex-col gap-5">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              <div className="max-w-3xl">
                <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/75">Lab</p>
                <h1 className="mt-2 text-3xl font-semibold tracking-tight text-[#faf9f6] sm:text-4xl">
                  Ideas inbox
                </h1>
                <p className="mt-3 max-w-2xl text-sm leading-7 text-[#faf9f6]/72">
                  Human verdicts stay in the sidecar only. This view never flips pipeline status; it only captures
                  approve, disapprove, request review, or request improvement with a note.
                </p>
              </div>

              <div className="flex flex-wrap items-center justify-start gap-2 lg:justify-end">
                <span className="rounded-full border border-[#f0eee6]/10 bg-[#121316] px-3 py-1 text-xs uppercase tracking-[0.2em] text-[#faf9f6]/68">
                  {sourceLabel}
                </span>
                <span className="rounded-full border border-[#f0eee6]/10 bg-[#121316] px-3 py-1 text-xs uppercase tracking-[0.2em] text-[#faf9f6]/60">
                  {totalCount} total
                </span>
                <span className="rounded-full border border-[#f0eee6]/10 bg-[#121316] px-3 py-1 text-xs uppercase tracking-[0.2em] text-[#faf9f6]/60">
                  {actionableIdeas.length} actionable
                </span>
                <span className="rounded-full border border-[#f0eee6]/10 bg-[#121316] px-3 py-1 text-xs uppercase tracking-[0.2em] text-[#faf9f6]/60">
                  {reviewedCount} reviewed
                </span>
                <button
                  type="button"
                  onClick={() => setShowReviewed((value) => !value)}
                  className="rounded-full border border-[#f0eee6]/12 bg-[#121316] px-3.5 py-1.5 text-xs font-medium uppercase tracking-[0.2em] text-[#faf9f6]/78 transition hover:border-[#f0eee6]/22 hover:bg-[#f0eee6]/[0.04]"
                >
                  {showReviewed ? "Hide reviewed" : "Show reviewed"}
                </button>
              </div>
            </div>

            <div className="flex flex-col gap-2 text-xs text-[#faf9f6]/48 sm:flex-row sm:items-center sm:justify-between">
              <span>{loadedLabel}</span>
              <span>Autosave endpoint: localhost:4500/api/suggestion-review</span>
            </div>
          </div>
        </section>

        {source === "error" ? (
          <div className="mt-5 rounded-2xl border border-rose-400/30 bg-rose-400/10 px-5 py-4 text-sm text-rose-100">
            {error || "Failed to load ideas."}
          </div>
        ) : null}

        {source !== "error" && source === "loading" ? (
          <p className="mt-6 text-sm text-[#faf9f6]/52">Loading suggestions…</p>
        ) : null}

        {source !== "error" && source !== "loading" && visibleIdeas.length === 0 ? (
          <div className="mt-6 rounded-2xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] px-5 py-6 text-sm leading-7 text-[#faf9f6]/68">
            {showReviewed
              ? "No ideas found in the current snapshot."
              : "Every idea already has a human verdict. Toggle Show reviewed to inspect the full set."}
          </div>
        ) : null}

        <div className="mt-6 space-y-4">
          {visibleIdeas.map((idea) => (
            <IdeaCard
              key={idea.id}
              idea={idea}
              live={live}
              onSaved={updateIdeaReview}
            />
          ))}
        </div>
      </div>
    </main>
  );
}
