import Link from "next/link";

type PrimaryLink = {
  href: string;
  title: string;
  description: string;
};

type SecondaryLink = {
  href: string;
  label: string;
};

const primaryLinks: PrimaryLink[] = [
  {
    href: "/proposals",
    title: "What the lab wants to build next",
    description: "Read and judge it.",
  },
  {
    href: "/contribute",
    title: "How you'll be able to plug in",
    description: "See the paths for people, compute, and code.",
  },
  {
    href: "/lab/experiments",
    title: "Live experiment activity",
    description: "Check what the lab is running now.",
  },
];

const secondaryLinks: SecondaryLink[] = [
  { href: "/learn", label: "Learn" },
  { href: "/blog", label: "Blog" },
  { href: "/about", label: "About" },
  { href: "/lab", label: "Lab" },
  { href: "/request-review", label: "Request review" },
];

export default function Home() {
  return (
    <main className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]">
      <section className="relative overflow-hidden border-b border-[#f0eee6]/10">
        <div className="absolute inset-0 bg-[#1f1e1d]" />
        <div className="absolute inset-0 opacity-15">
          <div className="absolute -top-20 left-1/4 h-72 w-72 rounded-full bg-cyan-500/30 blur-3xl animate-float" />
          <div className="absolute top-24 right-1/4 h-72 w-72 rounded-full bg-amber-400/20 blur-3xl animate-float-delayed" />
        </div>
        <div className="absolute inset-0 bg-[linear-gradient(rgba(240,238,230,.03)_1px,transparent_1px),linear-gradient(90deg,rgba(240,238,230,.03)_1px,transparent_1px)] bg-[size:72px_72px]" />

        <div className="relative container mx-auto px-6 pb-20 pt-32 md:pb-24 md:pt-40">
          <div className="max-w-4xl">
            <p className="text-sm uppercase tracking-[0.24em] text-cyan-300/80">
              Open Superintelligence Lab
            </p>
            <h1 className="mt-4 text-4xl font-semibold tracking-tight md:text-6xl md:leading-[1.02]">
              Building the #1 fully open-source LLM.
            </h1>
            <p className="mt-6 max-w-3xl text-lg leading-relaxed text-[#faf9f6]/80 md:text-xl">
              Training pipeline, research, and data are all open. Day to day, an AI orchestrator runs the work,
              and humans approve direction.
            </p>
          </div>
        </div>
      </section>

      <section className="container mx-auto max-w-6xl px-6 py-16 md:py-20">
        <div className="grid gap-5 md:grid-cols-3">
          {primaryLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="group flex h-full flex-col rounded-2xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.03] p-6 transition-colors hover:border-cyan-300/30 hover:bg-[#f0eee6]/[0.05]"
            >
              <p className="text-xs uppercase tracking-[0.24em] text-cyan-300/75">{link.href}</p>
              <h2 className="mt-4 text-2xl font-semibold text-[#faf9f6]">{link.title}</h2>
              <p className="mt-3 max-w-md text-base leading-relaxed text-[#faf9f6]/75">
                {link.description}
              </p>
              <span className="mt-auto pt-6 text-sm font-medium text-cyan-300 transition-colors group-hover:text-cyan-200">
                Open
              </span>
            </Link>
          ))}
        </div>
      </section>

      <section className="container mx-auto max-w-6xl px-6 pb-20">
        <div className="rounded-2xl border border-[#f0eee6]/10 bg-[#f0eee6]/[0.02] p-6 md:p-7">
          <p className="text-xs uppercase tracking-[0.24em] text-[#faf9f6]/50">More</p>
          <div className="mt-4 flex flex-wrap gap-3">
            {secondaryLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="inline-flex items-center rounded-full border border-[#f0eee6]/12 bg-[#f0eee6]/[0.04] px-4 py-2 text-sm font-medium text-[#faf9f6]/85 transition-colors hover:border-cyan-300/30 hover:bg-cyan-300/10 hover:text-cyan-200"
              >
                {link.label}
              </Link>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
