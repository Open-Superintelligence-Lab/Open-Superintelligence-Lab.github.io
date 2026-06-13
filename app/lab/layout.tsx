"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const navGroups = [
  {
    label: "Work",
    links: [
      { href: "/lab", label: "Overview" },
      { href: "/lab/queue", label: "Queue" },
      { href: "/lab/pipeline", label: "Pipeline" },
      { href: "/lab/experiments", label: "Experiments" },
    ],
  },
  {
    label: "Results",
    links: [
      { href: "/lab/research", label: "Papers" },
      { href: "/lab/leaderboard", label: "Leaderboard" },
      { href: "/lab/problems", label: "Problems" },
      { href: "/lab/ideas", label: "Ideas" },
    ],
  },
  {
    label: "Direction",
    links: [
      { href: "/lab/goals", label: "Goals" },
      { href: "/lab/compute", label: "Compute" },
      { href: "/proposals", label: "Proposals" },
      { href: "/contribute", label: "Contribute" },
    ],
  },
  {
    label: "Private",
    links: [{ href: "/lab/my", label: "My research" }],
  },
];

const normalizePath = (value: string) => value.replace(/\/+$/, "") || "/";

const isActive = (pathname: string, href: string) => {
  const current = normalizePath(pathname);
  const target = normalizePath(href);

  if (target === "/lab") {
    return current === "/lab";
  }

  return current === target || current.startsWith(`${target}/`);
};

function SidebarLink({ href, label, pathname }: { href: string; label: string; pathname: string }) {
  const active = isActive(pathname, href);

  return (
    <Link
      href={href}
      className={`shrink-0 border-l-2 px-3 py-2 text-sm transition md:w-full ${
        active
          ? "border-cyan-300 text-cyan-300"
          : "border-transparent text-[#faf9f6]/70 hover:border-[#f0eee6]/20 hover:text-[#faf9f6]"
      }`}
    >
      {label}
    </Link>
  );
}

function SidebarGroup({
  label,
  links,
  pathname,
}: {
  label: string;
  links: { href: string; label: string }[];
  pathname: string;
}) {
  const active = links.some((link) => isActive(pathname, link.href));

  return (
    <details open={active || label === "Work"} className="group shrink-0 md:w-full">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.35em] text-[#faf9f6]/35 transition hover:text-[#faf9f6]/55">
        <span>{label}</span>
        <span className="text-[#faf9f6]/30 transition group-open:rotate-180">v</span>
      </summary>
      <div className="flex gap-1 md:flex-col md:gap-0">
        {links.map((link) => (
          <SidebarLink key={link.href} href={link.href} label={link.label} pathname={pathname} />
        ))}
      </div>
    </details>
  );
}

export default function LabLayout({ children }: { children: ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-[#1f1e1d] text-[#faf9f6] md:flex">
      <aside className="sticky top-0 z-20 w-full border-b border-[#f0eee6]/10 bg-[#1f1e1d] md:h-screen md:w-56 md:border-r md:border-b-0">
        <div className="flex gap-4 overflow-x-auto px-4 py-4 md:flex-col md:overflow-visible md:px-5 md:py-6">
          <Link
            href="/"
            className="shrink-0 border-l-2 border-transparent px-3 py-2 text-sm font-semibold tracking-tight text-[#faf9f6] transition hover:text-cyan-300"
          >
            Open Superintelligence Lab
          </Link>

          {navGroups.map((group) => (
            <SidebarGroup key={group.label} label={group.label} links={group.links} pathname={pathname} />
          ))}
        </div>
      </aside>

      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}
