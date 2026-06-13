'use client';

import type { ReactNode } from "react";
import { usePathname } from "next/navigation";
import { Navigation } from "@/components/navigation";
import { Footer } from "@/components/footer";

// Wraps app/layout.tsx children. On /run (the private dashboard) we render the
// page full-bleed with no global Navigation/Footer, so the dashboard owns the
// whole viewport. Every other route keeps the normal chrome.
export function ConditionalChrome({ children }: { children: ReactNode }) {
  const pathname = usePathname() ?? "";
  const isRun = pathname === "/run" || pathname.startsWith("/run/");

  if (isRun) {
    return <main className="min-h-screen">{children}</main>;
  }

  return (
    <div className="min-h-screen text-white flex flex-col">
      <Navigation />
      <main className="flex-1">{children}</main>
      <Footer />
    </div>
  );
}
