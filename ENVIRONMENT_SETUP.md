# 🔧 Environment Configuration Guide

## Required Environment Variables

Add these to your `.env.local` file:

```bash
# AI Provider - OpenRouter (for research planning and analysis)
OPENROUTER_API_KEY=your-openrouter-api-key

# GPU Providers  
NOVITA_API_KEY=your-novita-api-key

# Storage (for artifacts)
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_S3_BUCKET=your-s3-bucket
```

## Getting API Keys

### 1. OpenRouter API Key
- Visit: https://openrouter.ai/
- Sign up and get your API key
- Supports multiple AI models including Grok-4, Claude, GPT-4, etc.

### 2. Novita AI API Key  
- Visit: https://novita.ai/
- Sign up for GPU access
- Get your API key for job dispatch

### 3. AWS S3 (Optional)
- For artifact storage
- Create S3 bucket for storing model checkpoints, logs, etc.

## Setup Instructions

1. Copy `.env.example` to `.env.local`
2. Fill in your API keys
3. Restart the development servers:
   ```bash
   npx convex dev
   npm run dev
   ```

## Demo Mode

If you don't have API keys yet, the system will work in demo mode with:
- Mock AI responses for research planning
- Mock GPU job dispatch
- Simulated progress updates
