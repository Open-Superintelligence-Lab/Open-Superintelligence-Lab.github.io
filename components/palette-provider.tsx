'use client';

import { useEffect } from 'react';
import { applyPalette } from '@/lib/palettes/loader';

const STORAGE_KEY = 'palette-preference';

export function PaletteProvider() {
  useEffect(() => {
    // Check for saved preference first, otherwise use default
    const saved = localStorage.getItem(STORAGE_KEY);
    const paletteName = saved || 'default';
    applyPalette(paletteName).catch(console.error);
  }, []);

  return null;
}
