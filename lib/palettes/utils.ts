export interface Palette {
  [key: string]: string;
}

/**
 * Convert hex color to HSL format (without hsl() wrapper)
 * Returns format: "H S% L%" (e.g., "240 10% 3.9%")
 */
export function hexToHsl(hex: string): string {
  // Remove # if present
  const cleanHex = hex.replace('#', '');
  
  // Parse hex values
  const r = parseInt(cleanHex.substring(0, 2), 16) / 255;
  const g = parseInt(cleanHex.substring(2, 4), 16) / 255;
  const b = parseInt(cleanHex.substring(4, 6), 16) / 255;

  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h = 0;
  let s = 0;
  const l = (max + min) / 2;

  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    
    switch (max) {
      case r:
        h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
        break;
      case g:
        h = ((b - r) / d + 2) / 6;
        break;
      case b:
        h = ((r - g) / d + 4) / 6;
        break;
    }
  }

  h = Math.round(h * 360);
  s = Math.round(s * 1000) / 10;
  const lightness = Math.round(l * 1000) / 10;

  return `${h} ${s}% ${lightness}%`;
}

/**
 * Generate CSS variable object from palette
 * Converts hex colors to HSL format for CSS variables
 */
export function generateCssVars(palette: Palette): Record<string, string> {
  const cssVars: Record<string, string> = {};
  
  for (const [key, value] of Object.entries(palette)) {
    // Convert CSS variable names (kebab-case to --var-name)
    const cssVarName = `--${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`;
    
    // Handle alpha values and other non-hex values (like radius)
    if (key.includes('alpha') || key === 'radius' || !value.startsWith('#')) {
      cssVars[cssVarName] = value;
    } else {
      cssVars[cssVarName] = hexToHsl(value);
    }
  }
  
  return cssVars;
}

