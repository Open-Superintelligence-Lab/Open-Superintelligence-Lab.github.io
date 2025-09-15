# GitHub Pages Deployment Guide

This Next.js app is configured for deployment to GitHub Pages at `https://open-superintelligence-lab.github.io/Open-Superintelligence-Lab.github.io/`.

## ✅ What's Fixed

1. **Base Path Configuration**: Updated `next.config.ts` with correct `basePath` and `assetPrefix`
2. **Tailwind CSS**: Verified content paths include `./app/**/*.{js,ts,jsx,tsx,mdx}`
3. **GitHub Actions**: Created automatic deployment workflow
4. **Build Process**: Fixed Turbopack issues and PostCSS configuration

## 🚀 Deployment Methods

### Method 1: Automatic (Recommended)
The GitHub Actions workflow will automatically deploy when you push to the `main` branch:

1. Push your changes to `main`
2. The workflow will build and deploy automatically
3. Your site will be available at: `https://open-superintelligence-lab.github.io/Open-Superintelligence-Lab.github.io/`

### Method 2: Manual Deployment
```bash
# Build for production
npm run build:prod

# Serve locally to test
npm run serve:out

# The built files are in the `out/` directory
# Push the contents of `out/` to your `gh-pages` branch
```

## 🔧 Local Testing

To test the production build locally:

```bash
# Build with production settings
npm run build:prod

# Serve the built files
npm run serve:out
```

## 📁 Key Files

- `next.config.ts` - Contains basePath and assetPrefix configuration
- `.github/workflows/deploy.yml` - GitHub Actions deployment workflow
- `tailwind.config.js` - Tailwind CSS configuration with correct content paths
- `postcss.config.mjs` - PostCSS configuration for CSS processing

## 🐛 Troubleshooting

If CSS is missing on GitHub Pages:

1. ✅ Check `basePath` matches your repo name exactly
2. ✅ Verify `assetPrefix` has trailing slash
3. ✅ Ensure Tailwind content paths include all your component directories
4. ✅ Make sure you're deploying the `out/` folder, not `.next/`

## 🌐 Your Site URL

Once deployed, your site will be available at:
`https://open-superintelligence-lab.github.io/Open-Superintelligence-Lab.github.io/`
