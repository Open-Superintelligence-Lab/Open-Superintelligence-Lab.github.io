import type { NextConfig } from "next";

const isProd = process.env.NODE_ENV === 'production';

const nextConfig: NextConfig = {
  // Static export configuration for production builds only
  ...(isProd && {
    output: 'export',
    trailingSlash: true,
    distDir: 'out',
  }),
  images: {
    unoptimized: true
  },
  // Custom domain configuration (opensuperintelligencelab.com)
  // No basePath needed for custom domain
};

export default nextConfig;
