import Link from "next/link";

const labLinks = [
  { href: "/lab/experiments", label: "Experiments" },
  { href: "/lab/goals", label: "Goals" },
  { href: "/lab/ideas", label: "Ideas" },
  { href: "/lab/problems", label: "Problems" },
  { href: "/lab/research", label: "Research" },
  { href: "/lab/compute", label: "Compute" },
  { href: "/lab/leaderboard", label: "Leaderboard" },
];

const readLinks = [
  { href: "/blog", label: "Blog" },
  { href: "/learn", label: "Learn" },
  { href: "/research-plan", label: "Research plan" },
];

const metaLinks = [
  { href: "/about", label: "About" },
  { href: "/proposals", label: "Proposals" },
  { href: "/contribute", label: "Contribute" },
  { href: "/request-review", label: "Request review" },
  { href: "/publish", label: "Publish" },
];

function FooterGroup({
  title,
  links,
}: {
  title: string;
  links: Array<{ href: string; label: string }>;
}) {
  return (
    <section>
      <h2 className="text-xs font-semibold uppercase tracking-[0.35em] text-[#faf9f6]/35">
        {title}
      </h2>
      <ul className="mt-4 space-y-2 text-sm">
        {links.map((link) => (
          <li key={link.href}>
            <Link
              href={link.href}
              className="text-[#faf9f6]/65 transition hover:text-cyan-300"
            >
              {link.label}
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function Footer() {
  return (
    <footer className="border-t border-[#f0eee6]/10 bg-[#1f1e1d]/90 backdrop-blur-xl">
      <div className="container mx-auto max-w-6xl px-6 py-10">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-[#faf9f6]/55">
          <Link href="/" className="transition hover:text-cyan-300">
            Home
          </Link>
          <span className="text-[#f0eee6]/20">•</span>
          <Link href="/lab" className="transition hover:text-cyan-300">
            Lab
          </Link>
        </div>

        <div className="mt-8 grid gap-8 md:grid-cols-3">
          <FooterGroup title="Lab" links={labLinks} />
          <FooterGroup title="Read" links={readLinks} />
          <FooterGroup title="Meta" links={metaLinks} />
        </div>
      </div>
    </footer>
  );
}
