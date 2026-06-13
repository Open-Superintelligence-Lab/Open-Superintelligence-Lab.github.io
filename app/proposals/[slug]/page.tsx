import Link from "next/link";
import { notFound } from "next/navigation";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { getAllProposals, getProposalBySlug } from "@/lib/proposals";

interface PageProps {
  params: Promise<{
    slug: string;
  }>;
}

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

export async function generateStaticParams() {
  return getAllProposals().map((proposal) => ({
    slug: proposal.slug,
  }));
}

export default async function ProposalPage({ params }: PageProps) {
  const { slug } = await params;
  const proposal = getProposalBySlug(slug);

  if (!proposal) {
    notFound();
  }

  return (
    <div className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]/90 pt-32 pb-24">
      <div className="container mx-auto px-6 max-w-4xl">
        <div className="mb-8">
          <Link href="/proposals" className="text-sm text-[#faf9f6]/60 hover:text-[#faf9f6] transition-colors">
            ← Back to Proposals
          </Link>
        </div>

        <div className="mb-12">
          <div className="flex flex-wrap items-center gap-3 text-sm text-[#faf9f6]/60 mb-4">
            <span>{proposal.date || "No date"}</span>
            <span>•</span>
            <span className="font-mono">{proposal.slug}</span>
            <span
              className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] ${statusClasses(
                proposal.status
              )}`}
            >
              {proposal.status.replace(/-/g, " ")}
            </span>
          </div>

          <h1 className="text-4xl md:text-5xl font-bold text-[#faf9f6] leading-tight">
            {proposal.title}
          </h1>
        </div>

        <article className="rounded-3xl border border-white/10 bg-white/5 p-6 md:p-8">
          <MarkdownRenderer content={proposal.content} />
        </article>
      </div>
    </div>
  );
}
