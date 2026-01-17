'use client';

import { useEffect } from 'react';
import { initPalette } from '@/lib/palettes/loader';

export function PaletteProvider() {
  useEffect(() => {
    initPalette();
  }, []);

  return null;
}
