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

  conversations: defineTable({
    projectId: v.id("projects"),
    title: v.string(),
    createdAt: v.number(),
    updatedAt: v.number(),
  })
    .index("by_project", ["projectId"])
    .index("by_updated", ["updatedAt"]),

  messages: defineTable({
    conversationId: v.id("conversations"),
    role: v.union(
      v.literal("user"),
      v.literal("assistant"),
      v.literal("system")
    ),
    content: v.string(),
    timestamp: v.number(),
    tools: v.optional(v.array(v.any())),
  })
    .index("by_conversation", ["conversationId"])
    .index("by_conversation_and_timestamp", ["conversationId", "timestamp"]),

  tutorials: defineTable({
    title: v.string(),
    description: v.string(),
    content: v.string(), // Markdown content
    authorId: v.string(),
    status: v.union(
      v.literal("draft"),
      v.literal("published"),
      v.literal("archived")
    ),
    tags: v.array(v.string()),
    category: v.string(),
    difficulty: v.union(
      v.literal("beginner"),
      v.literal("intermediate"),
      v.literal("advanced")
    ),
    estimatedReadTime: v.number(), // in minutes
    views: v.number(),
    likes: v.number(),
    isPublic: v.boolean(),
    aiGenerated: v.boolean(),
    createdAt: v.number(),
    updatedAt: v.number(),
    publishedAt: v.optional(v.number()),
  })
    .index("by_author", ["authorId"])
    .index("by_status", ["status"])
    .index("by_category", ["category"])
    .index("by_difficulty", ["difficulty"])
    .index("by_public", ["isPublic"])
    .index("by_published", ["publishedAt"])
    .index("by_tags", ["tags"]),

  tutorialVersions: defineTable({
    tutorialId: v.id("tutorials"),
    version: v.number(),
    content: v.string(),
    changeDescription: v.string(),
    createdAt: v.number(),
    createdBy: v.string(),
  })
    .index("by_tutorial", ["tutorialId"])
    .index("by_tutorial_and_version", ["tutorialId", "version"]),

  tutorialCollaborations: defineTable({
    tutorialId: v.id("tutorials"),
    userId: v.string(),
    role: v.union(
      v.literal("editor"),
      v.literal("reviewer"),
      v.literal("viewer")
    ),
    permissions: v.array(v.string()),
    invitedAt: v.number(),
    acceptedAt: v.optional(v.number()),
  })
    .index("by_tutorial", ["tutorialId"])
    .index("by_user", ["userId"]),

  tutorialComments: defineTable({
    tutorialId: v.id("tutorials"),
    userId: v.string(),
    content: v.string(),
    parentCommentId: v.optional(v.id("tutorialComments")),
    isResolved: v.boolean(),
    createdAt: v.number(),
    updatedAt: v.number(),
  })
    .index("by_tutorial", ["tutorialId"])
    .index("by_user", ["userId"])
    .index("by_parent", ["parentCommentId"]),

  tutorialChatSessions: defineTable({
    tutorialId: v.id("tutorials"),
    userId: v.string(),
    title: v.string(),
    createdAt: v.number(),
    updatedAt: v.number(),
  })
    .index("by_tutorial", ["tutorialId"])
    .index("by_user", ["userId"]),

  tutorialChatMessages: defineTable({
    sessionId: v.id("tutorialChatSessions"),
    role: v.union(
      v.literal("user"),
      v.literal("assistant"),
      v.literal("system")
    ),
    content: v.string(),
    timestamp: v.number(),
    contextSections: v.optional(v.array(v.string())), // References to tutorial sections
  })
    .index("by_session", ["sessionId"])
    .index("by_session_and_timestamp", ["sessionId", "timestamp"]),
});
