'use client';

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/", label: "Home" },
  { href: "/lab", label: "Lab" },
  { href: "/proposals", label: "Proposals" },
  { href: "/contribute", label: "Contribute" },
  { href: "/blog", label: "Blog" },
  { href: "/learn", label: "Learn" },
  { href: "/about", label: "About" },
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <header className="fixed top-0 left-0 right-0 z-50">
      {/* Glassmorphism Navigation Bar */}
      <div className="border-b border-[#f0eee6]/10 bg-[#1f1e1d]/80 backdrop-blur-xl">
        <div className="container mx-auto px-6 py-4">
          <nav className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div className="text-sm font-medium tracking-[0.2em] uppercase text-[#f0eee6]/55">
              Open Superintelligence Lab
            </div>
            <div className="flex flex-wrap items-center gap-2 text-sm">
              {navItems.map((item) => {
                const isActive = item.href === "/" ? pathname === "/" : pathname?.startsWith(item.href);

                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`rounded-full border px-3 py-1.5 transition-colors ${
                      isActive
                        ? "border-[#f0eee6]/30 bg-[#f0eee6]/10 text-[#faf9f6]"
                        : "border-transparent text-[#f0eee6]/65 hover:border-[#f0eee6]/20 hover:bg-[#f0eee6]/5 hover:text-[#faf9f6]"
                    }`}
                  >
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </nav>
        </div>
      </div>
    </header>
  );
}
