import { query, mutation } from "./_generated/server";
import { v } from "convex/values";

// Get all runs for a project
export const listByProject = query({
  args: { projectId: v.id("projects") },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("runs")
      .withIndex("by_project", (q) => q.eq("projectId", args.projectId))
      .order("desc")
      .collect();
  },
});

// Get a single run by ID
export const get = query({
  args: { id: v.id("runs") },
  handler: async (ctx, args) => {
    return await ctx.db.get(args.id);
  },
});

// Get run with all related data
export const getWithDetails = query({
  args: { id: v.id("runs") },
  handler: async (ctx, args) => {
    const run = await ctx.db.get(args.id);
    if (!run) return null;

    const steps = await ctx.db
      .query("runSteps")
      .withIndex("by_run", (q) => q.eq("runId", args.id))
      .order("asc")
      .collect();

    const metrics = await ctx.db
      .query("metrics")
      .withIndex("by_run", (q) => q.eq("runId", args.id))
      .order("desc")
      .collect();

    const artifacts = await ctx.db
      .query("artifacts")
      .withIndex("by_run", (q) => q.eq("runId", args.id))
      .order("desc")
      .collect();

    return {
      run,
      steps,
      metrics,
      artifacts,
    };
  },
});

// Create a new run
export const create = mutation({
  args: {
    projectId: v.id("projects"),
    name: v.string(),
    config: v.any(),
    gpuProvider: v.string(),
  },
  handler: async (ctx, args) => {
    const now = Date.now();
    const runId = await ctx.db.insert("runs", {
      ...args,
      status: "queued",
      progress: 0,
      cost: 0,
      jobRef: "",
      startedAt: now,
      eta: "Calculating...",
    });

    // Create initial steps
    const steps = [
      { name: "Plan Generation", description: "Agent analyzed repository and generated training plan" },
      { name: "Environment Setup", description: "GPU provisioning and dependency installation" },
      { name: "Data Preprocessing", description: "Dataset loading and tokenization" },
      { name: "Model Training", description: "Training epochs" },
      { name: "Evaluation", description: "Model evaluation on validation set" },
      { name: "Analysis & Next Steps", description: "Performance analysis and plan updates" },
    ];

    for (let i = 0; i < steps.length; i++) {
      await ctx.db.insert("runSteps", {
        runId,
        stepName: steps[i].name,
        status: "pending",
        description: steps[i].description,
        stepIndex: i,
      });
    }

    return runId;
  },
});

// Update run status
export const updateStatus = mutation({
  args: {
    id: v.id("runs"),
    status: v.union(
      v.literal("running"),
      v.literal("completed"),
      v.literal("paused"),
      v.literal("failed"),
      v.literal("queued")
    ),
  },
  handler: async (ctx, args) => {
    const updates: any = { status: args.status };
    
    if (args.status === "running") {
      updates.startedAt = Date.now();
    } else if (args.status === "completed" || args.status === "failed") {
      updates.endedAt = Date.now();
    }

    return await ctx.db.patch(args.id, updates);
  },
});

// Update run progress
export const updateProgress = mutation({
  args: {
    id: v.id("runs"),
    progress: v.number(),
    eta: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    return await ctx.db.patch(args.id, {
      progress: args.progress,
      eta: args.eta,
    });
  },
});

// Update run cost
export const updateCost = mutation({
  args: {
    id: v.id("runs"),
    cost: v.number(),
  },
  handler: async (ctx, args) => {
    return await ctx.db.patch(args.id, {
      cost: args.cost,
    });
  },
});

// Update job reference
export const updateJobRef = mutation({
  args: {
    id: v.id("runs"),
    jobRef: v.string(),
  },
  handler: async (ctx, args) => {
    return await ctx.db.patch(args.id, {
      jobRef: args.jobRef,
    });
  },
});

// Add a metric to a run
export const addMetric = mutation({
  args: {
    runId: v.id("runs"),
    name: v.string(),
    value: v.number(),
    stepIndex: v.number(),
  },
  handler: async (ctx, args) => {
    return await ctx.db.insert("metrics", {
      ...args,
      timestamp: Date.now(),
    });
  },
});

// Add an artifact to a run
export const addArtifact = mutation({
  args: {
    runId: v.id("runs"),
    name: v.string(),
    type: v.string(),
    size: v.number(),
    url: v.string(),
    checksum: v.string(),
  },
  handler: async (ctx, args) => {
    return await ctx.db.insert("artifacts", {
      ...args,
      createdAt: Date.now(),
    });
  },
});

// Update step status
export const updateStepStatus = mutation({
  args: {
    runId: v.id("runs"),
    stepIndex: v.number(),
    status: v.union(
      v.literal("pending"),
      v.literal("running"),
      v.literal("completed"),
      v.literal("failed")
    ),
  },
  handler: async (ctx, args) => {
    const steps = await ctx.db
      .query("runSteps")
      .withIndex("by_run_and_index", (q) => 
        q.eq("runId", args.runId).eq("stepIndex", args.stepIndex)
      )
      .collect();

    if (steps.length === 0) return null;

    const step = steps[0];
    const updates: any = { status: args.status };

    if (args.status === "running") {
      updates.startedAt = Date.now();
    } else if (args.status === "completed" || args.status === "failed") {
      updates.endedAt = Date.now();
      if (step.startedAt) {
        updates.duration = Date.now() - step.startedAt;
      }
    }

    return await ctx.db.patch(step._id, updates);
  },
});

// Delete a run
export const remove = mutation({
  args: { id: v.id("runs") },
  handler: async (ctx, args) => {
    // Delete related data
    const steps = await ctx.db
      .query("runSteps")
      .withIndex("by_run", (q) => q.eq("runId", args.id))
      .collect();
    
    for (const step of steps) {
      await ctx.db.delete(step._id);
    }

    const metrics = await ctx.db
      .query("metrics")
      .withIndex("by_run", (q) => q.eq("runId", args.id))
      .collect();
    
    for (const metric of metrics) {
      await ctx.db.delete(metric._id);
    }

    const artifacts = await ctx.db
      .query("artifacts")
      .withIndex("by_run", (q) => q.eq("runId", args.id))
      .collect();
    
    for (const artifact of artifacts) {
      await ctx.db.delete(artifact._id);
    }

    // Delete the run
    await ctx.db.delete(args.id);
  },
});
