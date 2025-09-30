import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export configuration for production builds
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  // Custom domain configuration (opensuperintelligencelab.com)
  // No basePath needed for custom domain
  distDir: 'out',
};

export default nextConfig;
