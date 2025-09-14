/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  assetPrefix: process.env.NODE_ENV === 'production' ? '/Open-Superintelligence-Lab.github.io' : '',
  basePath: process.env.NODE_ENV === 'production' ? '/Open-Superintelligence-Lab.github.io' : ''
}

module.exports = nextConfig
