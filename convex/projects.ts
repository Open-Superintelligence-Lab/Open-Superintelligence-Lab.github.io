import { query, mutation } from "./_generated/server";
import { v } from "convex/values";

// Get all projects
export const list = query({
  args: {},
  handler: async (ctx) => {
    return await ctx.db
      .query("projects")
      .order("desc")
      .collect();
  },
});

// Get a single project by ID
export const get = query({
  args: { id: v.id("projects") },
  handler: async (ctx, args) => {
    return await ctx.db.get(args.id);
  },
});

// Create a new project
export const create = mutation({
  args: {
    name: v.string(),
    description: v.string(),
    status: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const now = Date.now();
    return await ctx.db.insert("projects", {
      name: args.name,
      description: args.description,
      status: args.status || "active",
      createdAt: now,
    });
  },
});

// Delete a project
export const remove = mutation({
  args: { id: v.id("projects") },
  handler: async (ctx, args) => {
    await ctx.db.delete(args.id);
  },
});
