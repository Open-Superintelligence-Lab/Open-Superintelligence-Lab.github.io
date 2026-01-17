'use client';

import { generateCssVars, type Palette } from './utils';

/**
 * Get the palette name from environment variable or default to 'default'
 */
export function getPaletteName(): string {
  // For client-side, we can't access process.env directly in browser
  // We'll use a data attribute or script tag approach
  if (typeof window !== 'undefined') {
    // Try to get from a script tag or meta tag
    const paletteScript = document.getElementById('palette-config');
    if (paletteScript && paletteScript.textContent) {
      try {
        const config = JSON.parse(paletteScript.textContent);
        return config.palette || 'default';
      } catch (e) {
        console.warn('Failed to parse palette config', e);
      }
    }
  }
  
  // Server-side or fallback - use NEXT_PUBLIC_ prefix for client access
  if (typeof process !== 'undefined' && process.env) {
    return process.env.NEXT_PUBLIC_PALETTE || process.env.PALETTE || 'default';
  }
  return 'default';
}

/**
 * Load palette from JSON file at runtime
 */
export async function loadPaletteClient(name: string): Promise<Palette> {
  try {
    const response = await fetch(`/palettes/${name}.json`);
    if (!response.ok) {
      throw new Error(`Failed to load palette: ${name}`);
    }
    return await response.json();
  } catch (error) {
    console.warn(`Failed to load palette "${name}", falling back to default`, error);
    // Try to load default
    try {
      const response = await fetch('/palettes/default.json');
      return await response.json();
    } catch (defaultError) {
      console.error('Failed to load default palette', defaultError);
      // Return minimal fallback
      return {
        background: '#FFFFFF',
        foreground: '#000000',
      };
    }
  }
}

/**
 * Inject CSS variables into the document root
 */
export function injectCssVars(cssVars: Record<string, string>) {
  if (typeof document === 'undefined') return;
  
  const root = document.documentElement;
  
  for (const [key, value] of Object.entries(cssVars)) {
    root.style.setProperty(key, value);
  }
}

/**
 * Apply palette by loading it and injecting CSS variables
 */
export async function applyPalette(paletteName?: string) {
  const name = paletteName || getPaletteName();
  const palette = await loadPaletteClient(name);
  const cssVars = generateCssVars(palette);
  injectCssVars(cssVars);
  return { name, palette, cssVars };
}

/**
 * Initialize palette on page load
 */
export function initPalette() {
  if (typeof window === 'undefined') return;
  
  // Apply palette immediately
  applyPalette().catch(console.error);
}
