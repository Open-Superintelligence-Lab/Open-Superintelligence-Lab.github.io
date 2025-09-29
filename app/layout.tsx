import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import "highlight.js/styles/github-dark.css";
import { LanguageProvider } from "@/components/providers/language-provider";
import { Navigation } from "@/components/navigation";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Open Superintelligence Lab",
  description: "开放超级智能实验室 - Advancing AI research and development",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <LanguageProvider>
          <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
            {/* Enhanced background effects */}
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-blue-600/20"></div>
            <div className="absolute inset-0 opacity-30">
              <div className="absolute inset-0 bg-gradient-to-br from-transparent via-blue-500/5 to-transparent"></div>
            </div>
            
            {/* Enhanced animated background particles */}
            <div className="absolute inset-0 overflow-hidden">
              {/* Large floating particles - some made more glowy and dreamy */}
              <div className="absolute top-1/6 left-1/6 w-3 h-3 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full opacity-60 animate-pulse"></div>
              <div className="absolute top-1/4 right-1/5 w-2 h-2 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full opacity-50 animate-pulse delay-300 shadow-lg shadow-purple-400/40"></div>
              <div className="absolute top-1/3 left-1/8 w-4 h-4 bg-gradient-to-r from-emerald-400 to-teal-400 rounded-full opacity-40 animate-pulse delay-700 blur-sm shadow-lg shadow-emerald-400/30"></div>
              <div className="absolute bottom-1/4 right-1/6 w-2.5 h-2.5 bg-gradient-to-r from-cyan-400 to-blue-400 rounded-full opacity-55 animate-pulse delay-1000"></div>
              <div className="absolute bottom-1/3 left-1/4 w-1.5 h-1.5 bg-gradient-to-r from-pink-400 to-purple-400 rounded-full opacity-60 animate-pulse delay-500 blur-sm shadow-lg shadow-pink-400/35"></div>
              <div className="absolute top-2/3 right-1/3 w-3.5 h-3.5 bg-gradient-to-r from-teal-400 to-emerald-400 rounded-full opacity-35 animate-pulse delay-1200 blur-sm shadow-lg shadow-teal-400/25"></div>
              
              {/* Medium particles - enhanced with glow */}
              <div className="absolute top-1/5 left-2/5 w-1 h-1 bg-blue-400/60 rounded-full animate-pulse delay-200 shadow-lg shadow-blue-400/50"></div>
              <div className="absolute top-2/5 right-2/5 w-1.5 h-1.5 bg-purple-400/50 rounded-full animate-pulse delay-800 blur-sm shadow-lg shadow-purple-400/45"></div>
              <div className="absolute bottom-2/5 left-3/5 w-1 h-1 bg-cyan-400/70 rounded-full animate-pulse delay-400"></div>
              <div className="absolute bottom-1/5 right-1/5 w-1.5 h-1.5 bg-pink-400/45 rounded-full animate-pulse delay-900 blur-sm shadow-lg shadow-pink-400/40"></div>
              <div className="absolute top-3/5 left-1/5 w-1 h-1 bg-emerald-400/65 rounded-full animate-pulse delay-600 shadow-lg shadow-emerald-400/55"></div>
              
              {/* Small twinkling particles - some made more dreamy */}
              <div className="absolute top-1/8 left-3/8 w-0.5 h-0.5 bg-white/80 rounded-full animate-pulse delay-150 blur-sm shadow-lg shadow-white/60"></div>
              <div className="absolute top-1/7 right-3/8 w-0.5 h-0.5 bg-blue-300/90 rounded-full animate-pulse delay-750"></div>
              <div className="absolute bottom-1/8 left-2/8 w-0.5 h-0.5 bg-purple-300/85 rounded-full animate-pulse delay-350 blur-sm shadow-lg shadow-purple-300/70"></div>
              <div className="absolute bottom-1/7 right-2/8 w-0.5 h-0.5 bg-cyan-300/80 rounded-full animate-pulse delay-950"></div>
              <div className="absolute top-4/5 left-4/8 w-0.5 h-0.5 bg-pink-300/75 rounded-full animate-pulse delay-550 blur-sm shadow-lg shadow-pink-300/65"></div>
              <div className="absolute top-3/8 right-1/8 w-0.5 h-0.5 bg-emerald-300/85 rounded-full animate-pulse delay-1150"></div>
              
              {/* Floating geometric shapes - some made more ethereal */}
              <div className="absolute top-1/6 right-1/8 w-2 h-2 bg-gradient-to-r from-blue-500/30 to-purple-500/30 rotate-45 animate-pulse delay-250 blur-sm shadow-lg shadow-blue-500/25"></div>
              <div className="absolute bottom-1/6 left-1/8 w-1.5 h-1.5 bg-gradient-to-r from-cyan-500/40 to-pink-500/40 rotate-12 animate-pulse delay-650"></div>
              <div className="absolute top-1/2 right-1/6 w-1 h-3 bg-gradient-to-r from-emerald-500/35 to-teal-500/35 rotate-75 animate-pulse delay-850 blur-sm shadow-lg shadow-emerald-500/20"></div>
              
              {/* Enhanced glowing orbs - made more dreamy */}
              <div className="absolute top-1/4 left-1/2 w-6 h-6 bg-gradient-to-r from-blue-400/20 to-purple-400/20 rounded-full blur-sm animate-pulse delay-450 shadow-lg shadow-blue-400/15"></div>
              <div className="absolute bottom-1/4 right-1/2 w-4 h-4 bg-gradient-to-r from-cyan-400/25 to-pink-400/25 rounded-full blur-sm animate-pulse delay-1050 shadow-lg shadow-cyan-400/20"></div>
              <div className="absolute top-1/2 left-1/3 w-5 h-5 bg-gradient-to-r from-emerald-400/15 to-teal-400/15 rounded-full blur-sm animate-pulse delay-750 shadow-lg shadow-emerald-400/12"></div>
              
              {/* Additional dreamy particles */}
              <div className="absolute top-1/5 right-1/4 w-2 h-2 bg-gradient-to-r from-violet-400/30 to-fuchsia-400/30 rounded-full blur-sm animate-pulse delay-1100 shadow-lg shadow-violet-400/25"></div>
              <div className="absolute bottom-1/5 left-2/5 w-1.5 h-1.5 bg-gradient-to-r from-amber-400/35 to-orange-400/35 rounded-full blur-sm animate-pulse delay-550 shadow-lg shadow-amber-400/30"></div>
              <div className="absolute top-2/5 right-1/5 w-1 h-1 bg-gradient-to-r from-rose-400/40 to-pink-400/40 rounded-full blur-sm animate-pulse delay-850 shadow-lg shadow-rose-400/35"></div>
            </div>
            
            <Navigation />
            <div className="relative z-10">
              {children}
            </div>
          </div>
        </LanguageProvider>
      </body>
    </html>
  );
}
