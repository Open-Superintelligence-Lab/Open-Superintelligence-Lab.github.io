# Real AI Integration Setup

The chatbot now uses the real Grok API via OpenRouter instead of mock responses!

## Setup Instructions

1. **Get an OpenRouter API Key:**
   - Go to [OpenRouter.ai](https://openrouter.ai/)
   - Sign up and get your API key
   - The Grok-4-Fast model is free to use

2. **Set Convex Environment Variables:**
   The API key is now stored securely in Convex! Run these commands:
   ```bash
   npx convex env set OPENROUTER_API_KEY your_openrouter_api_key_here
   npx convex env set SITE_URL http://localhost:3000
   npx convex env set SITE_NAME "Open Superintelligence Lab"
   ```

3. **Test the Integration:**
   - Start the development server: `npm run dev`
   - Go to any project page (e.g., `/projects/test`)
   - Click on the "AI Assistant" tab
   - Start chatting with the real Grok AI!

## What's Real vs Mock

✅ **Real AI:** The chatbot responses are now powered by Grok-4-Fast via OpenRouter
✅ **Real API:** Actual HTTP calls to OpenRouter's API through Convex
✅ **Real Context:** The AI knows about your project and can have natural conversations
✅ **Secure:** API key is stored securely in Convex environment variables

🔄 **Still Mocked:** The tools (run_experiment, analyze_data, etc.) are still simulated for demo purposes

## Features

- **Natural Conversations:** Ask the AI anything about your research project
- **Context Awareness:** The AI knows which project you're working on
- **Tool Suggestions:** The AI can suggest using mock tools based on your requests
- **Error Handling:** Graceful fallback if the API is unavailable
- **Free Usage:** Grok-4-Fast is free on OpenRouter

## Example Prompts to Try

- "Help me design an experiment for machine learning"
- "What should I analyze in my dataset?"
- "How can I improve my model's performance?"
- "Explain the latest trends in AI research"
- "What tools do you have available?"

The AI will respond naturally and may suggest using the available tools when appropriate!
