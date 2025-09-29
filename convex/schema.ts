import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  // Simple schema for future expansion
  projects: defineTable({
    name: v.string(),
    description: v.string(),
    status: v.string(),
    createdAt: v.number(),
  })
    .index("by_status", ["status"]),
});
