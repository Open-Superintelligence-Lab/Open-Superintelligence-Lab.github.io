import { action, mutation, query } from "./_generated/server";
import { v } from "convex/values";
import OpenAI from "openai";

// Create a new conversation
export const createConversation = mutation({
  args: {
    projectId: v.id("projects"),
    title: v.string(),
  },
  handler: async (ctx, { projectId, title }) => {
    const now = Date.now();
    return await ctx.db.insert("conversations", {
      projectId,
      title,
      createdAt: now,
      updatedAt: now,
    });
  },
});

// Get conversations for a project
export const getConversations = query({
  args: {
    projectId: v.id("projects"),
  },
  handler: async (ctx, { projectId }) => {
    return await ctx.db
      .query("conversations")
      .withIndex("by_project", (q) => q.eq("projectId", projectId))
      .order("desc")
      .collect();
  },
});

// Get messages for a conversation
export const getMessages = query({
  args: {
    conversationId: v.id("conversations"),
  },
  handler: async (ctx, { conversationId }) => {
    return await ctx.db
      .query("messages")
      .withIndex("by_conversation", (q) => q.eq("conversationId", conversationId))
      .order("asc")
      .collect();
  },
});

// Delete a conversation and all its messages
export const deleteConversation = mutation({
  args: {
    conversationId: v.id("conversations"),
  },
  handler: async (ctx, { conversationId }) => {
    // Delete all messages in the conversation
    const messages = await ctx.db
      .query("messages")
      .withIndex("by_conversation", (q) => q.eq("conversationId", conversationId))
      .collect();

    for (const message of messages) {
      await ctx.db.delete(message._id);
    }

    // Delete the conversation
    await ctx.db.delete(conversationId);

    return conversationId;
  },
});

// Update conversation title
export const updateConversationTitle = mutation({
  args: {
    conversationId: v.id("conversations"),
    title: v.string(),
  },
  handler: async (ctx, { conversationId, title }) => {
    await ctx.db.patch(conversationId, {
      title,
      updatedAt: Date.now(),
    });

    return conversationId;
  },
});

// Add a message to a conversation
export const addMessage = mutation({
  args: {
    conversationId: v.id("conversations"),
    role: v.union(v.literal("user"), v.literal("assistant"), v.literal("system")),
    content: v.string(),
    tools: v.optional(v.array(v.any())),
  },
  handler: async (ctx, { conversationId, role, content, tools }) => {
    const messageId = await ctx.db.insert("messages", {
      conversationId,
      role,
      content,
      timestamp: Date.now(),
      tools,
    });

    // Update conversation's updatedAt timestamp
    await ctx.db.patch(conversationId, {
      updatedAt: Date.now(),
    });

    return messageId;
  },
});

export const chatWithGrok = action({
  args: {
    message: v.string(),
    context: v.string(),
    projectName: v.string(),
    conversationHistory: v.optional(v.array(v.object({
      role: v.union(v.literal("user"), v.literal("assistant"), v.literal("system")),
      content: v.string()
    }))),
  },
  handler: async (ctx, { message, context, projectName, conversationHistory = [] }) => {
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

      // Build the messages array with system prompt, history, and current message
      const messages = [
        {
          role: "system" as const,
          content: `You are an AI research assistant for the "${projectName || 'Open Superintelligence Lab'}" project. You help users run experiments, analyze data, train models, and deploy them. 

You have access to several tools:
- run_experiment: Execute machine learning experiments
- analyze_data: Perform data analysis and visualization  
- train_model: Train machine learning models
- deploy_model: Deploy models to production

When users ask about running experiments, analyzing data, training models, or deploying models, you should suggest using the appropriate tool. Be helpful and provide detailed explanations of what you can do.

Current context: ${context || 'General research assistance'}

Respond naturally and helpfully to the user's request. Remember previous messages in this conversation to provide context-aware responses.`
        },
        ...conversationHistory,
        {
          role: "user" as const,
          content: message as string
        }
      ];

      const completion = await client.chat.completions.create({
        model: "x-ai/grok-4-fast:free",
        messages,
        max_tokens: 8192,
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
