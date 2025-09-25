import { action } from "./_generated/server";
import { v } from "convex/values";
import OpenAI from "openai";

export const chatWithGrok = action({
  args: {
    message: v.string(),
    context: v.string(),
    projectName: v.string(),
  },
  handler: async (ctx, { message, context, projectName }) => {
    // Get environment variables from Convex
    const apiKey = process.env.OPENROUTER_API_KEY;
    const siteUrl = process.env.SITE_URL || "https://open-superintelligence-lab-github-io.vercel.app";
    const siteName = process.env.SITE_NAME || "Open Superintelligence Lab";

    if (!apiKey) {
      throw new Error("OPENROUTER_API_KEY not configured in Convex environment");
    }

    const client = new OpenAI({
      baseURL: "https://openrouter.ai/api/v1",
      apiKey: apiKey,
    });

    try {
      const completion = await client.chat.completions.create({
        model: "x-ai/grok-4-fast:free",
        messages: [
          {
            role: "system",
            content: `You are an AI research assistant for the "${projectName || 'Open Superintelligence Lab'}" project. You help users run experiments, analyze data, train models, and deploy them. 

You have access to several tools:
- run_experiment: Execute machine learning experiments
- analyze_data: Perform data analysis and visualization  
- train_model: Train machine learning models
- deploy_model: Deploy models to production

When users ask about running experiments, analyzing data, training models, or deploying models, you should suggest using the appropriate tool. Be helpful and provide detailed explanations of what you can do.

Current context: ${context || 'General research assistance'}

Respond naturally and helpfully to the user's request.`
          },
          {
            role: "user",
            content: message as string
          }
        ],
        max_tokens: 1000,
        temperature: 0.7,
      });

      const aiResponse = completion.choices[0].message.content || "I'm sorry, I couldn't generate a response.";

      // Simple tool detection based on AI response content
      const lowerResponse = aiResponse.toLowerCase();
      const tools: string[] = [];
      const toolParams: any = {};

      if (lowerResponse.includes('experiment') || lowerResponse.includes('run experiment')) {
        tools.push('run_experiment');
        toolParams.type = 'classification';
        toolParams.dataset = 'custom';
      }
      
      if (lowerResponse.includes('analyze') || lowerResponse.includes('data analysis')) {
        tools.push('analyze_data');
        toolParams.dataset = 'current';
        toolParams.analysisType = 'comprehensive';
      }
      
      if (lowerResponse.includes('train') || lowerResponse.includes('model training')) {
        tools.push('train_model');
        toolParams.algorithm = 'neural_network';
        toolParams.epochs = 100;
      }
      
      if (lowerResponse.includes('deploy') || lowerResponse.includes('production')) {
        tools.push('deploy_model');
        toolParams.environment = 'production';
        toolParams.scaling = 'auto';
      }

      return {
        response: aiResponse,
        tools,
        toolParams
      };

    } catch (error) {
      console.error('Error calling Grok API:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      throw new Error(`Failed to get AI response: ${errorMessage}`);
    }
  },
});
