'use client';

import Link from 'next/link';
import { useEffect, useRef } from 'react';

export function OSLLogo() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = 80; // Wider to fit OSL without overlap
    const height = 36; // Height for the letters
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    // Get colors from CSS variables
    const getCSSColor = (varName: string, fallback: string) => {
      if (typeof window === 'undefined') return fallback;
      const root = getComputedStyle(document.documentElement);
      const value = root.getPropertyValue(varName).trim();
      return value || fallback;
    };

    const drawLogo = () => {
      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // Get gradient colors from palette
      const accent1 = getCSSColor('--gradient-accent-1', '210 100% 83%');
      const accent2 = getCSSColor('--gradient-accent-2', '270 100% 86%');
      const accent3 = getCSSColor('--gradient-accent-3', '180 77% 81%');

      // Parse HSL to RGB for canvas
      const hslToRgb = (hsl: string): [number, number, number] => {
        const parts = hsl.split(/\s+/);
        const h = parseFloat(parts[0]) / 360;
        const s = parseFloat(parts[1]) / 100;
        const l = parseFloat(parts[2]) / 100;

        let r: number, g: number, b: number;
        if (s === 0) {
          r = g = b = l;
        } else {
          const hue2rgb = (p: number, q: number, t: number) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1/6) return p + (q - p) * 6 * t;
            if (t < 1/2) return q;
            if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
          };
          const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
          const p = 2 * l - q;
          r = hue2rgb(p, q, h + 1/3);
          g = hue2rgb(p, q, h);
          b = hue2rgb(p, q, h - 1/3);
        }
        return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
      };

      const [r1, g1, b1] = hslToRgb(accent1);
      const [r2, g2, b2] = hslToRgb(accent2);
      const [r3, g3, b3] = hslToRgb(accent3);

      // Create gradient
      const gradient = ctx.createLinearGradient(0, 0, width, height);
      gradient.addColorStop(0, `rgb(${r1}, ${g1}, ${b1})`);
      gradient.addColorStop(0.5, `rgb(${r2}, ${g2}, ${b2})`);
      gradient.addColorStop(1, `rgb(${r3}, ${g3}, ${b3})`);

      ctx.fillStyle = gradient;
      ctx.strokeStyle = gradient;
      ctx.lineWidth = 1.5;

      // Set font - bold and clear
      ctx.font = 'bold 24px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      const centerY = height / 2;

      // O - Left
      ctx.fillText('O', 16, centerY);

      // S - Center
      ctx.fillText('S', width / 2, centerY);

      // L - Right
      ctx.fillText('L', width - 16, centerY);

      // Add subtle underline accent
      ctx.save();
      ctx.strokeStyle = gradient;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(4, height - 4);
      ctx.lineTo(width - 4, height - 4);
      ctx.stroke();
      ctx.restore();
    };

    // Initial draw
    drawLogo();

    // Redraw on palette changes (check every 2 seconds)
    let lastCheck = Date.now();
    const checkInterval = setInterval(() => {
      const now = Date.now();
      if (now - lastCheck > 2000) {
        drawLogo();
        lastCheck = now;
      }
    }, 2000);

    return () => clearInterval(checkInterval);
  }, []);

  return (
    <Link href="/" className="group flex items-center gap-3 hover:scale-105 transition-all duration-300">
      <canvas
        ref={canvasRef}
        className="block"
        style={{ border: 'none', outline: 'none', background: 'transparent', width: '80px', height: '36px' }}
        aria-label="Open Superintelligence Lab"
      />

      {/* Brand Name */}
      <div className="hidden md:flex flex-col">
        <span className="text-sm font-bold bg-gradient-to-r from-gradient-accent-1 via-gradient-accent-2 to-gradient-accent-3 bg-clip-text text-transparent">
          Open Superintelligence
        </span>
        <span className="text-xs text-muted-foreground">Building the Future</span>
      </div>
    </Link>
  );
}