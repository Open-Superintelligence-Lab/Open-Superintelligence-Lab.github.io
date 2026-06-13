import Link from "next/link";
import { getAllProposals } from "@/lib/proposals";

function statusClasses(status: string) {
  switch (status) {
    case "approved":
      return "bg-emerald-500/15 text-emerald-300 border-emerald-500/30";
    case "rejected":
      return "bg-rose-500/15 text-rose-300 border-rose-500/30";
    case "changes-requested":
      return "bg-amber-500/15 text-amber-300 border-amber-500/30";
    default:
      return "bg-slate-500/15 text-slate-300 border-slate-500/30";
  }
}

export default function ProposalsPage() {
  const proposals = getAllProposals();

  return (
    <div className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]/90 pt-32 pb-24">
      <div className="container mx-auto px-6 max-w-5xl">
        <div className="mb-12">
          <div className="flex items-center gap-4 text-sm text-[#faf9f6]/60 mb-4">
            <span>Open Superintelligence Lab</span>
            <span>•</span>
            <span>Proposals</span>
          </div>

          <h1 className="text-4xl md:text-5xl font-bold text-[#faf9f6] mb-6 leading-tight">
            Proposals
          </h1>

          <p className="max-w-3xl text-lg leading-relaxed text-[#faf9f6]/75">
            This is where the lab&apos;s AI orchestrator publishes what it wants to build, and humans approve or reject before anything is built.
          </p>
        </div>

        <div className="space-y-4">
          {proposals.length === 0 ? (
            <div className="rounded-2xl border border-white/10 bg-white/5 p-8 text-[#faf9f6]/70">
              No proposals found.
            </div>
          ) : (
            proposals.map((proposal) => (
              <Link
                key={proposal.slug}
                href={`/proposals/${proposal.slug}`}
                className="block rounded-2xl border border-white/10 bg-white/5 p-6 transition-all duration-200 hover:border-blue-500/40 hover:bg-white/[0.07]"
              >
                <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                  <div>
                    <h2 className="text-2xl font-semibold text-[#faf9f6]">
                      {proposal.title}
                    </h2>
                    <div className="mt-3 flex items-center gap-3 text-sm text-[#faf9f6]/60">
                      <span>{proposal.date || "No date"}</span>
                      <span>•</span>
                      <span className="font-mono">{proposal.slug}</span>
                    </div>
                  </div>

                  <span
                    className={`inline-flex w-fit items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] ${statusClasses(
                      proposal.status
                    )}`}
                  >
                    {proposal.status.replace(/-/g, " ")}
                  </span>
                </div>
              </Link>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
