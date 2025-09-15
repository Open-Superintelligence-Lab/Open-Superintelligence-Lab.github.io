import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  // GitHub Pages configuration
  basePath: process.env.NODE_ENV === 'production' ? '/Open-Superintelligence-Lab.github.io' : '',
  assetPrefix: process.env.NODE_ENV === 'production' ? '/Open-Superintelligence-Lab.github.io/' : '',
  distDir: 'out',
};

export default nextConfig;
