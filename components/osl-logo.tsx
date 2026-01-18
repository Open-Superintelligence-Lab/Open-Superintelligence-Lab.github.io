'use client';

import Link from 'next/link';
import { useState } from 'react';

export function OSLLogo() {
  const [isHovering, setIsHovering] = useState(false);

  return (
    <Link 
      href="/" 
      className="group relative flex items-center justify-center py-1 px-6 transition-all duration-500"
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
    >
      {/* Dynamic Background Glow */}
      <div className={`absolute inset-0 bg-gradient-to-r from-gradient-accent-1/20 via-gradient-accent-2/20 to-gradient-accent-3/20 blur-2xl transition-opacity duration-1000 rounded-full ${isHovering ? 'opacity-100' : 'opacity-0'}`} />
      
      {/* The OSL Monogram */}
      <div className="relative flex items-center">
        <span 
          className="text-5xl md:text-6xl font-black italic tracking-tighter transition-all duration-700 ease-out transform group-hover:scale-105"
          style={{
            background: 'linear-gradient(135deg, hsl(var(--gradient-accent-1)) 0%, hsl(var(--gradient-accent-2)) 50%, hsl(var(--gradient-accent-3)) 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            filter: isHovering 
              ? 'drop-shadow(0 0 25px hsl(var(--gradient-accent-2) / 0.6)) drop-shadow(0 0 2px hsl(var(--foreground) / 0.2))' 
              : 'drop-shadow(0 0 15px hsl(var(--gradient-accent-2) / 0.3))',
            textShadow: isHovering ? '0 0 1px rgba(255,255,255,0.3)' : 'none'
          }}
        >
          OSL
        </span>
        
        {/* Sleek animated accent line */}
        <div 
          className={`absolute -bottom-2 left-1/2 -translate-x-1/2 h-[3px] bg-gradient-to-r from-transparent via-gradient-accent-2 to-transparent transition-all duration-700 ease-in-out rounded-full ${isHovering ? 'w-full opacity-100' : 'w-0 opacity-0'}`} 
        />
      </div>
    </Link>
  );
}
