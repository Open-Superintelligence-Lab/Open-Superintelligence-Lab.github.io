import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  projects: defineTable({
    name: v.string(),
    description: v.string(),
    ownerId: v.string(),
    status: v.union(
      v.literal("running"),
      v.literal("completed"),
      v.literal("paused"),
      v.literal("failed")
    ),
    budget: v.number(),
    usedBudget: v.number(),
    settings: v.any(),
    createdAt: v.number(),
    updatedAt: v.number(),
  })
    .index("by_owner", ["ownerId"])
    .index("by_status", ["status"]),

  runs: defineTable({
    projectId: v.id("projects"),
    name: v.string(),
    status: v.union(
      v.literal("running"),
      v.literal("completed"),
      v.literal("paused"),
      v.literal("failed"),
      v.literal("queued")
    ),
    progress: v.number(),
    config: v.any(),
    cost: v.number(),
    gpuProvider: v.string(),
    jobRef: v.string(),
    startedAt: v.number(),
    endedAt: v.optional(v.number()),
    eta: v.optional(v.string()),
  })
    .index("by_project", ["projectId"])
    .index("by_status", ["status"]),

  runSteps: defineTable({
    runId: v.id("runs"),
    stepName: v.string(),
    status: v.union(
      v.literal("pending"),
      v.literal("running"),
      v.literal("completed"),
      v.literal("failed")
    ),
    description: v.string(),
    startedAt: v.optional(v.number()),
    endedAt: v.optional(v.number()),
    duration: v.optional(v.number()),
    stepIndex: v.number(),
  })
    .index("by_run", ["runId"])
    .index("by_run_and_index", ["runId", "stepIndex"]),

  metrics: defineTable({
    runId: v.id("runs"),
    name: v.string(),
    value: v.number(),
    timestamp: v.number(),
    stepIndex: v.number(),
  })
    .index("by_run", ["runId"])
    .index("by_run_and_name", ["runId", "name"]),

  artifacts: defineTable({
    runId: v.id("runs"),
    name: v.string(),
    type: v.string(),
    size: v.number(),
    url: v.string(),
    checksum: v.string(),
    createdAt: v.number(),
  })
    .index("by_run", ["runId"])
    .index("by_type", ["type"]),

  credentials: defineTable({
    projectId: v.id("projects"),
    serviceType: v.string(), // "openai", "github", "novita", "s3"
    encryptedData: v.string(),
    createdAt: v.number(),
  })
    .index("by_project", ["projectId"])
    .index("by_service", ["serviceType"]),
});
