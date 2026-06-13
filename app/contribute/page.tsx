import Link from "next/link";

type Path = {
  title: string;
  status: string;
  statusClass: string;
  body: string;
  link: { href: string; label: string };
};

const paths: Path[] = [
  {
    title: "Review and approve the lab's direction",
    status: "LIVE",
    statusClass: "border-emerald-400/30 bg-emerald-400/10 text-emerald-300",
    body: "The AI orchestrator publishes proposals at /proposals. People read them and approve or reject before work starts.",
    link: { href: "/proposals", label: "Open proposals" },
  },
  {
    title: "AI task queue",
    status: "LOCAL PROTOTYPE",
    statusClass: "border-amber-400/30 bg-amber-400/10 text-amber-300",
    body: "Goal: clone, install, and your machine pulls AI tasks for research, review, and coding from a shared queue. It runs locally in the lab today.",
    link: { href: "/lab", label: "See the lab" },
  },
  {
    title: "Contribute compute",
    status: "LOCAL PROTOTYPE",
    statusClass: "border-amber-400/30 bg-amber-400/10 text-amber-300",
    body: "Goal: the same install story, but your GPU runs queued experiments and reports results. The local registry tracks runs now, with a public snapshot at /lab/experiments.",
    link: { href: "/lab/experiments", label: "Open experiment snapshot" },
  },
  {
    title: "Personal research spaces",
    status: "PLANNED",
    statusClass: "border-slate-400/30 bg-slate-400/10 text-slate-300",
    body: "Private project spaces with prompts and tools for automated AI research.",
    link: { href: "/proposals", label: "Follow the roadmap" },
  },
];

export default function ContributePage() {
  return (
    <main className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]">
      <section className="relative overflow-hidden border-b border-[#f0eee6]/10">
        <div className="absolute inset-0 bg-[#1f1e1d]" />
        <div className="absolute inset-0 opacity-15">
          <div className="absolute -top-20 left-1/4 h-72 w-72 rounded-full bg-cyan-500/30 blur-3xl animate-float" />
          <div className="absolute top-24 right-1/4 h-72 w-72 rounded-full bg-amber-400/20 blur-3xl animate-float-delayed" />
        </div>
        <div className="absolute inset-0 bg-[linear-gradient(rgba(240,238,230,.03)_1px,transparent_1px),linear-gradient(90deg,rgba(240,238,230,.03)_1px,transparent_1px)] bg-[size:72px_72px]" />

        <div className="relative container mx-auto px-6 pb-20 pt-32 md:pt-40">
          <div className="max-w-4xl">
            <p className="text-sm uppercase tracking-[0.24em] text-cyan-300/80">Contribute</p>
            <h1 className="mt-4 text-4xl font-semibold tracking-tight md:text-6xl md:leading-[1.02]">
              Help build the #1 fully open-source LLM.
            </h1>
            <p className="mt-6 max-w-3xl text-lg leading-relaxed text-[#faf9f6]/80 md:text-xl">
              Training pipeline, research, and data are all open. These are the ways people will be able to
              plug in.
            </p>
          </div>
        </div>
      </section>

      <section className="container mx-auto max-w-5xl px-6 py-16 md:py-20">
        <div className="grid gap-5">
          {paths.map((path) => (
            <article
              key={path.title}
              className="rounded-2xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.03] p-6 transition-colors hover:border-[#f0eee6]/20 hover:bg-[#f0eee6]/[0.05] md:p-7"
            >
              <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                <div className="max-w-3xl">
                  <h2 className="text-2xl font-semibold text-[#faf9f6]">{path.title}</h2>
                  <p className="mt-3 text-base leading-relaxed text-[#faf9f6]/75 md:text-lg">{path.body}</p>
                </div>

                <span
                  className={`inline-flex w-fit shrink-0 items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] ${path.statusClass}`}
                >
                  {path.status}
                </span>
              </div>

              <div className="mt-5">
                <Link
                  href={path.link.href}
                  className="inline-flex items-center gap-2 rounded-full border border-[#f0eee6]/15 bg-[#f0eee6]/5 px-4 py-2 text-sm font-medium text-[#faf9f6] transition-colors hover:border-cyan-300/40 hover:bg-cyan-300/10 hover:text-cyan-200"
                >
                  {path.link.label}
                </Link>
              </div>
            </article>
          ))}
        </div>

        <p className="mt-10 text-base leading-relaxed text-[#faf9f6]/70">
          Follow progress at <Link href="/proposals" className="text-cyan-300 hover:text-cyan-200">/proposals</Link>.
        </p>
      </section>
    </main>
  );
}
