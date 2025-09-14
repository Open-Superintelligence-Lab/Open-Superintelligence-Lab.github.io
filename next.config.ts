import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  // GitHub Pages configuration
  basePath: process.env.NODE_ENV === 'production' ? '/open-superintelligence-lab-github-io' : '',
  assetPrefix: process.env.NODE_ENV === 'production' ? '/open-superintelligence-lab-github-io/' : '',
  distDir: 'out',
};

export default nextConfig;
