'use client';

import React, { useRef, useEffect } from "react";

export default function VaporwaveGrid() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    if (!ctx) return;

    let time = 0;
    let rafId = 0;
    let waveStartTime = -Infinity;
    let waveTimerId = 0;
    let resizeObserver: ResizeObserver | null = null;

    // Function to get color from CSS variable
    const getCSSColor = (varName: string, fallback: string) => {
      if (typeof window === 'undefined') return fallback;
      const root = getComputedStyle(document.documentElement);
      const hslValue = root.getPropertyValue(varName).trim();
      if (!hslValue) return fallback;
      
      // Convert HSL "H S% L%" to hex
      const parts = hslValue.split(/\s+/);
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
      
      const toHex = (n: number) => Math.round(n * 255).toString(16).padStart(2, '0');
      return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
    };

    // Get colors from palette (cache them, update every 2 seconds)
    const colorCache = {
      SKY_TOP: "#A9DCFF",
      SKY_MID: "#CFC5FF",
      SKY_BOT: "#C0AAFF",
      GRID_COLOR: "255,255,255",
      SUN_COLOR: "255,196,138"
    };
    
    let lastColorUpdate = 0;
    const updateColors = () => {
      colorCache.SKY_TOP = getCSSColor('--gradient-start', '#A9DCFF');
      colorCache.SKY_MID = getCSSColor('--gradient-mid', '#CFC5FF');
      colorCache.SKY_BOT = getCSSColor('--gradient-end', '#C0AAFF');
      
      // Get foreground color for grid
      const fgHsl = getComputedStyle(document.documentElement).getPropertyValue('--foreground').trim();
      if (fgHsl) {
        const parts = fgHsl.split(/\s+/);
        const h = parseInt(parts[0]) / 360;
        const s = parseInt(parts[1]) / 100;
        const l = parseInt(parts[2]) / 100;
        
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
        colorCache.GRID_COLOR = `${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)}`;
      }
      
      // Get accent color for sun
      const accentHsl = getComputedStyle(document.documentElement).getPropertyValue('--gradient-accent-2').trim();
      if (accentHsl) {
        const parts = accentHsl.split(/\s+/);
        const h = parseInt(parts[0]) / 360;
        const s = parseInt(parts[1]) / 100;
        const l = parseInt(parts[2]) / 100;
        
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
        colorCache.SUN_COLOR = `${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)}`;
      }
    };

    // ----- Layout & palette -----
    const HORIZON_RATIO = 0.55;

    // Grid
    const NUM_H_LINES = 18;
    const NUM_V_LINES = 12;
    const V_MARGIN = 0.18;
    const EASE_POWER = 2.15;
    const BASE_ALPHA_BOTTOM = 0.22;
    const BASE_ALPHA_HORIZON = 0.01;

    // Shimmer
    const WAVE_DURATION = 5;
    const WAVE_INTERVAL_MS = 9000;
    const WAVE_BAND_WIDTH_T = 0.12;
    const WAVE_AMPLITUDE = 10;
    const WAVE_FREQ = 0.018;
    const WAVE_TIME_FREQ = 2.6;

    const SUN_RADIUS = 78;
    const SUN_POS = { xPct: 0.80, yPct: 0.24 };

    // ----- setup -----
    const measure = () => {
      const parent = canvas.parentElement;
      const rect = parent ? parent.getBoundingClientRect() : canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.floor(rect.width));
      const h = Math.max(1, Math.floor(rect.height));
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const triggerWave = () => {
      waveStartTime = time;
      waveTimerId = window.setTimeout(triggerWave, WAVE_INTERVAL_MS);
    };

    function drawSky(w: number, canvasHeight: number) {
      const g = ctx.createLinearGradient(0, 0, 0, canvasHeight);
      g.addColorStop(0, colorCache.SKY_TOP);
      g.addColorStop(0.55, colorCache.SKY_MID);
      g.addColorStop(1, colorCache.SKY_BOT);
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, canvasHeight);
    }

    function drawSunTopToBottom(cx: number, cy: number, r: number) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.clip();
      const lg = ctx.createLinearGradient(cx, cy - r, cx, cy + r);
      lg.addColorStop(0, `rgba(${colorCache.SUN_COLOR},0.95)`);
      lg.addColorStop(1, `rgba(${colorCache.SUN_COLOR},0)`);
      ctx.fillStyle = lg;
      ctx.fillRect(cx - r, cy - r, r * 2, r * 2);
      ctx.restore();

      const halo = ctx.createRadialGradient(cx, cy, r * 0.7, cx, cy, r * 1.6);
      halo.addColorStop(0, `rgba(${colorCache.SUN_COLOR},0.22)`);
      halo.addColorStop(1, `rgba(${colorCache.SUN_COLOR},0)`);
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 1.6, 0, Math.PI * 2);
      ctx.fill();
    }

    function drawSunrise(horizon: number, w: number, h: number) {
      // Main gradient - extended and more gradual
      let g = ctx.createLinearGradient(0, horizon - 180, 0, horizon + 220);
      g.addColorStop(0.00, `rgba(${colorCache.SUN_COLOR}, 0.00)`);
      g.addColorStop(0.20, `rgba(${colorCache.SUN_COLOR}, 0.28)`);
      g.addColorStop(0.35, `rgba(${colorCache.SUN_COLOR}, 0.45)`);
      g.addColorStop(0.50, `rgba(${colorCache.SUN_COLOR}, 0.55)`);
      g.addColorStop(0.65, `rgba(${colorCache.SUN_COLOR}, 0.45)`);
      g.addColorStop(0.80, `rgba(${colorCache.SUN_COLOR}, 0.28)`);
      g.addColorStop(1.00, `rgba(${colorCache.SUN_COLOR}, 0.00)`);
      ctx.fillStyle = g;
      ctx.fillRect(0, horizon - 180, w, 400);

      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      // Secondary gradient - more extended
      g = ctx.createLinearGradient(0, horizon - 120, 0, horizon + 140);
      g.addColorStop(0.00, `rgba(${colorCache.SUN_COLOR}, 0.00)`);
      g.addColorStop(0.35, `rgba(${colorCache.SUN_COLOR}, 0.12)`);
      g.addColorStop(0.50, `rgba(${colorCache.SUN_COLOR}, 0.16)`);
      g.addColorStop(0.65, `rgba(${colorCache.SUN_COLOR}, 0.12)`);
      g.addColorStop(1.00, `rgba(${colorCache.SUN_COLOR}, 0.00)`);
      ctx.fillStyle = g;
      ctx.fillRect(0, horizon - 120, w, 260);
      ctx.restore();

      // Horizon line glow
      g = ctx.createLinearGradient(0, horizon - 8, 0, horizon + 8);
      g.addColorStop(0, `rgba(${colorCache.GRID_COLOR},0.00)`);
      g.addColorStop(0.5, `rgba(${colorCache.GRID_COLOR},0.10)`);
      g.addColorStop(1, `rgba(${colorCache.GRID_COLOR},0.00)`);
      ctx.fillStyle = g;
      ctx.fillRect(0, horizon - 8, w, 16);

      // Top fade - more extended
      g = ctx.createLinearGradient(0, horizon - 240, 0, horizon);
      g.addColorStop(0, `rgba(${colorCache.SUN_COLOR}, 0.00)`);
      g.addColorStop(0.6, `rgba(${colorCache.SUN_COLOR}, 0.06)`);
      g.addColorStop(1, `rgba(${colorCache.SUN_COLOR}, 0.10)`);
      ctx.fillStyle = g;
      ctx.fillRect(0, horizon - 240, w, 240);

      // Bottom fade - more extended
      g = ctx.createLinearGradient(0, horizon, 0, horizon + 100);
      g.addColorStop(0, `rgba(${colorCache.SUN_COLOR}, 0.18)`);
      g.addColorStop(0.5, `rgba(${colorCache.SUN_COLOR}, 0.08)`);
      g.addColorStop(1, `rgba(${colorCache.SUN_COLOR}, 0.00)`);
      ctx.fillStyle = g;
      ctx.fillRect(0, horizon, w, 100);
    }

    const draw = () => {
      try {
        if (!canvas || !ctx || canvas.width === 0 || canvas.height === 0) {
          rafId = requestAnimationFrame(draw);
          return;
        }
        
        // Update colors every 2 seconds
        const now = Date.now();
        if (now - lastColorUpdate > 2000) {
          try {
            updateColors();
            lastColorUpdate = now;
          } catch (err) {
            console.error('Color update failed:', err);
          }
        }

        time += 1 / 60;
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.width / dpr;
        const h = canvas.height / dpr;
        
        if (w === 0 || h === 0) {
          rafId = requestAnimationFrame(draw);
          return;
        }
        
        const horizon = h * HORIZON_RATIO;

        // 1) Sky
        drawSky(w, h);

        // 2) Sun + smooth blob
        drawSunTopToBottom(w * SUN_POS.xPct, h * SUN_POS.yPct, SUN_RADIUS);

        // 3) Warm sunrise (colored)
        drawSunrise(horizon, w, h);

      // 4) Grid with shimmer
      const tSince = time - waveStartTime;
      const active = tSince >= 0 && tSince < WAVE_DURATION;
      const waveFrontT = active ? 1 - tSince / WAVE_DURATION : -1;

      // horizontals
      for (let i = 0; i < NUM_H_LINES; i++) {
        const t = i / (NUM_H_LINES - 1);
        const eased = Math.pow(t, EASE_POWER);
        const baseY = horizon + eased * (h - horizon);
        const alpha = BASE_ALPHA_HORIZON + (BASE_ALPHA_BOTTOM - BASE_ALPHA_HORIZON) * t;

        ctx.strokeStyle = `rgba(${colorCache.GRID_COLOR},${alpha})`;
        ctx.beginPath();
        const step = 5;
        for (let x = 0; x <= w; x += step) {
          let y = baseY;
          if (active) {
            const dist = Math.abs(t - waveFrontT);
            const influence = Math.max(0, 1 - dist / WAVE_BAND_WIDTH_T);
            if (influence > 0) {
              const depthScale = t;
              const local = Math.sin(x * WAVE_FREQ + time * WAVE_TIME_FREQ);
              y += local * WAVE_AMPLITUDE * influence * depthScale;
            }
          }
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      // verticals
      for (let i = 0; i <= NUM_V_LINES; i++) {
        const p = i / NUM_V_LINES;
        const xTop = w * (V_MARGIN + p * (1 - 2 * V_MARGIN));
        const vanishingX = w / 2;
        const xBottom = vanishingX + (xTop - vanishingX) * 3.2;

        const alpha = BASE_ALPHA_HORIZON + (BASE_ALPHA_BOTTOM - BASE_ALPHA_HORIZON) * 0.75;
        ctx.strokeStyle = `rgba(${colorCache.GRID_COLOR},${alpha})`;
        ctx.beginPath();
        ctx.moveTo(xTop, horizon);
        ctx.lineTo(xBottom, h);
        ctx.stroke();
      }

        rafId = requestAnimationFrame(draw);
      } catch (err) {
        console.error('Draw error:', err);
        rafId = requestAnimationFrame(draw); // Keep animating even if there's an error
      }
    };

    // Set up listeners
    const onWindowResize = () => measure();
    window.addEventListener("resize", onWindowResize);

    // Use ResizeObserver to react to container size changes
    if (typeof ResizeObserver !== "undefined") {
      const parent = canvas.parentElement || canvas;
      resizeObserver = new ResizeObserver(() => {
        measure();
      });
      resizeObserver.observe(parent);
    }

    // Initial setup
    updateColors();
    lastColorUpdate = Date.now();
    
    // Wait for browser layout before measuring
    requestAnimationFrame(() => {
      measure();
      
      // Draw initial frame
      if (canvas.width > 0 && canvas.height > 0) {
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.width / dpr;
        const h = canvas.height / dpr;
        drawSky(w, h);
      }
      
      // Start animation loop
      draw();
    });
    
    waveTimerId = window.setTimeout(() => {
      waveStartTime = time;
      triggerWave();
    }, 800);

    return () => {
      window.removeEventListener("resize", onWindowResize);
      if (resizeObserver) resizeObserver.disconnect();
      cancelAnimationFrame(rafId);
      clearTimeout(waveTimerId);
    };
  }, []);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none z-0" style={{ display: 'block' }} />;
}
