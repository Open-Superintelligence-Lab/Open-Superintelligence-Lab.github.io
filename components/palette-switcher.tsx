'use client';

import { useState, useEffect } from 'react';
import { applyPalette } from '@/lib/palettes/loader';

const AVAILABLE_PALETTES = [
  { name: 'default', label: 'Default (Dark)' },
  { name: 'ink-wash', label: 'Ink Wash (Light)' },
  { name: 'vaporwave', label: 'Vaporwave (Dark)' },
  { name: 'emerald-odyssey', label: 'Emerald Odyssey' },
  { name: 'harvest-moon', label: 'Harvest Moon' },
  { name: 'deep-ocean', label: 'Deep Ocean (Dark)' },
  { name: 'amber-glow', label: 'Amber Glow (Dark)' },
  { name: 'terminal-green', label: 'Terminal Green (Dark)' },
];

const STORAGE_KEY = 'palette-preference';

export function PaletteSwitcher() {
  const [currentPalette, setCurrentPalette] = useState<string>('default');
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    // Load saved preference or use default
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && AVAILABLE_PALETTES.some(p => p.name === saved)) {
      setCurrentPalette(saved);
      applyPalette(saved).catch(console.error);
    }
  }, []);

  const handlePaletteChange = async (paletteName: string) => {
    try {
      await applyPalette(paletteName);
      setCurrentPalette(paletteName);
      localStorage.setItem(STORAGE_KEY, paletteName);
      setIsOpen(false);
    } catch (error) {
      console.error('Failed to apply palette:', error);
    }
  };

  return (
    <div className="fixed bottom-6 right-6 z-50">
      <div className="relative">
        {/* Dropdown Menu */}
        {isOpen && (
          <>
            <div
              className="fixed inset-0 z-40"
              onClick={() => setIsOpen(false)}
            />
            <div 
              className="absolute bottom-full right-0 mb-2 w-60 bg-card border border-border rounded-lg shadow-xl overflow-hidden z-50 max-h-[500px] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-2">
                {AVAILABLE_PALETTES.map((palette) => (
                  <button
                    key={palette.name}
                    onClick={(e) => {
                      e.stopPropagation();
                      handlePaletteChange(palette.name);
                    }}
                    className={`w-full text-left px-3 py-2 text-sm rounded-md transition-colors ${
                      currentPalette === palette.name
                        ? 'bg-accent text-accent-foreground'
                        : 'text-foreground hover:bg-muted'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span>{palette.label}</span>
                      {currentPalette === palette.name && (
                        <svg
                          className="w-4 h-4"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M5 13l4 4L19 7"
                          />
                        </svg>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </>
        )}

        {/* Toggle Button */}
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="group relative w-12 h-12 bg-gradient-to-r from-gradient-accent-1 via-gradient-accent-2 to-gradient-accent-3 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-110 flex items-center justify-center"
          aria-label="Switch color palette"
          title="Switch color palette"
        >
          <svg
            className="w-6 h-6 text-primary-foreground"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01"
            />
          </svg>
        </button>
      </div>
    </div>
  );
}
