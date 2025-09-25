import { v } from "convex/values";
import { mutation, query } from "./_generated/server";

// Create a new chat session for a tutorial
export const createTutorialChatSession = mutation({
  args: {
    tutorialId: v.id("tutorials"),
    userId: v.string(),
    title: v.string(),
  },
  handler: async (ctx, args) => {
    const now = Date.now();
    
    const sessionId = await ctx.db.insert("tutorialChatSessions", {
      tutorialId: args.tutorialId,
      userId: args.userId,
      title: args.title,
      createdAt: now,
      updatedAt: now,
    });

    return sessionId;
  },
});

// Get chat sessions for a tutorial
export const getTutorialChatSessions = query({
  args: { 
    tutorialId: v.id("tutorials"),
    userId: v.string(),
  },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("tutorialChatSessions")
      .withIndex("by_tutorial", (q) => q.eq("tutorialId", args.tutorialId))
      .filter((q) => q.eq(q.field("userId"), args.userId))
      .order("desc")
      .collect();
  },
});

// Get chat messages for a session
export const getTutorialChatMessages = query({
  args: { sessionId: v.id("tutorialChatSessions") },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("tutorialChatMessages")
      .withIndex("by_session", (q) => q.eq("sessionId", args.sessionId))
      .order("asc")
      .collect();
  },
});

// Add message to tutorial chat
export const addTutorialChatMessage = mutation({
  args: {
    sessionId: v.id("tutorialChatSessions"),
    role: v.union(
      v.literal("user"),
      v.literal("assistant"),
      v.literal("system")
    ),
    content: v.string(),
    contextSections: v.optional(v.array(v.string())),
  },
  handler: async (ctx, args) => {
    const messageId = await ctx.db.insert("tutorialChatMessages", {
      sessionId: args.sessionId,
      role: args.role,
      content: args.content,
      timestamp: Date.now(),
      contextSections: args.contextSections,
    });

    // Update session timestamp
    await ctx.db.patch(args.sessionId, {
      updatedAt: Date.now(),
    });

    return messageId;
  },
});

// Update chat session title
export const updateTutorialChatSessionTitle = mutation({
  args: {
    sessionId: v.id("tutorialChatSessions"),
    title: v.string(),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.sessionId, {
      title: args.title,
      updatedAt: Date.now(),
    });

    return args.sessionId;
  },
});

// Delete chat session
export const deleteTutorialChatSession = mutation({
  args: { sessionId: v.id("tutorialChatSessions") },
  handler: async (ctx, args) => {
    // Delete all messages in the session
    const messages = await ctx.db
      .query("tutorialChatMessages")
      .withIndex("by_session", (q) => q.eq("sessionId", args.sessionId))
      .collect();

    for (const message of messages) {
      await ctx.db.delete(message._id);
    }

    // Delete the session
    await ctx.db.delete(args.sessionId);

    return args.sessionId;
  },
});

// Get tutorial for chat context
export const getTutorialForChat = query({
  args: { tutorialId: v.id("tutorials") },
  handler: async (ctx, args) => {
    const tutorial = await ctx.db.get(args.tutorialId);
    if (!tutorial) {
      return null;
    }

    // Return only necessary fields for chat context
    return {
      _id: tutorial._id,
      title: tutorial.title,
      description: tutorial.description,
      content: tutorial.content,
      category: tutorial.category,
      difficulty: tutorial.difficulty,
      tags: tutorial.tags,
    };
  },
});
