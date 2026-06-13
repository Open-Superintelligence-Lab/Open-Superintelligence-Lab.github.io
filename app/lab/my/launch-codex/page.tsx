"use client";

import { useState } from "react";

export default function LaunchCodexPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState("");

  const handleLaunch = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setMessage("");

    try {
      const response = await fetch("/api/launch-codex", {
        method: "POST",
      });

      if (response.ok) {
        setMessage("✓ Codex launcher started");
      } else {
        setMessage("✗ Failed to launch Codex");
      }
    } catch (error) {
      setMessage("✗ Error launching Codex");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#1f1e1d] pt-28 text-[#faf9f6] md:pt-36">
      <div className="container mx-auto flex min-h-[calc(100vh-12rem)] items-center justify-center px-6 py-32">
        <div className="text-center">
          <form onSubmit={handleLaunch}>
            <button
              type="submit"
              disabled={isLoading}
              className="rounded-full border border-cyan-300/30 bg-cyan-300/10 px-8 py-4 text-sm font-semibold uppercase tracking-[0.24em] text-cyan-200 transition hover:border-cyan-300/60 hover:bg-cyan-300/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-cyan-300/40 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? "Launching..." : "Launch Codex"}
            </button>
          </form>
          {message && (
            <p className="mt-4 text-sm text-cyan-300">{message}</p>
          )}
        </div>
      </div>
    </main>
  );
}
