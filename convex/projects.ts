import { query, mutation } from "./_generated/server";
import { v } from "convex/values";

// Get all projects for a user
export const list = query({
  args: {
    ownerId: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const ownerId = args.ownerId || "user-1"; // Default user for now
    return await ctx.db
      .query("projects")
      .withIndex("by_owner", (q) => q.eq("ownerId", ownerId))
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
    budget: v.number(),
    settings: v.optional(v.any()),
  },
  handler: async (ctx, args) => {
    const now = Date.now();
    return await ctx.db.insert("projects", {
      ...args,
      ownerId: "user-1", // Default user for now
      status: "running",
      usedBudget: 0,
      settings: args.settings || {},
      createdAt: now,
      updatedAt: now,
    });
  },
});

// Update project status
export const updateStatus = mutation({
  args: {
    id: v.id("projects"),
    status: v.union(
      v.literal("running"),
      v.literal("completed"),
      v.literal("paused"),
      v.literal("failed")
    ),
  },
  handler: async (ctx, args) => {
    return await ctx.db.patch(args.id, {
      status: args.status,
      updatedAt: Date.now(),
    });
  },
});

// Update project budget
export const updateBudget = mutation({
  args: {
    id: v.id("projects"),
    budget: v.number(),
    usedBudget: v.number(),
  },
  handler: async (ctx, args) => {
    return await ctx.db.patch(args.id, {
      budget: args.budget,
      usedBudget: args.usedBudget,
      updatedAt: Date.now(),
    });
  },
});

// Update project settings
export const updateSettings = mutation({
  args: {
    id: v.id("projects"),
    settings: v.any(),
  },
  handler: async (ctx, args) => {
    return await ctx.db.patch(args.id, {
      settings: args.settings,
      updatedAt: Date.now(),
    });
  },
});

// Delete a project
export const remove = mutation({
  args: { id: v.id("projects") },
  handler: async (ctx, args) => {
    // First delete all related runs
    const runs = await ctx.db
      .query("runs")
      .withIndex("by_project", (q) => q.eq("projectId", args.id))
      .collect();
    
    for (const run of runs) {
      await ctx.db.delete(run._id);
    }
    
    // Then delete the project
    await ctx.db.delete(args.id);
  },
});

// Get project statistics
export const getStats = query({
  args: { id: v.id("projects") },
  handler: async (ctx, args) => {
    const project = await ctx.db.get(args.id);
    if (!project) return null;

    const runs = await ctx.db
      .query("runs")
      .withIndex("by_project", (q) => q.eq("projectId", args.id))
      .collect();

    const totalRuns = runs.length;
    const completedRuns = runs.filter(run => run.status === "completed").length;
    const runningRuns = runs.filter(run => run.status === "running").length;
    const totalCost = runs.reduce((sum, run) => sum + run.cost, 0);

    return {
      project,
      totalRuns,
      completedRuns,
      runningRuns,
      totalCost,
      budgetRemaining: project.budget - totalCost,
    };
  },
});
