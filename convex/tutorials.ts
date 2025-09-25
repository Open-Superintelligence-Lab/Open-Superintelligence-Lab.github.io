import { v } from "convex/values";
import { mutation, query } from "./_generated/server";

// Create a new tutorial
export const createTutorial = mutation({
  args: {
    title: v.string(),
    description: v.string(),
    content: v.string(),
    authorId: v.string(),
    tags: v.array(v.string()),
    category: v.string(),
    difficulty: v.union(
      v.literal("beginner"),
      v.literal("intermediate"),
      v.literal("advanced")
    ),
    isPublic: v.boolean(),
    aiGenerated: v.boolean(),
  },
  handler: async (ctx, args) => {
    const now = Date.now();
    
    // Calculate estimated read time (roughly 200 words per minute)
    const wordCount = args.content.split(/\s+/).length;
    const estimatedReadTime = Math.max(1, Math.ceil(wordCount / 200));

    const tutorialId = await ctx.db.insert("tutorials", {
      title: args.title,
      description: args.description,
      content: args.content,
      authorId: args.authorId,
      status: "draft",
      tags: args.tags,
      category: args.category,
      difficulty: args.difficulty,
      estimatedReadTime,
      views: 0,
      likes: 0,
      isPublic: args.isPublic,
      aiGenerated: args.aiGenerated,
      createdAt: now,
      updatedAt: now,
    });

    // Create initial version
    await ctx.db.insert("tutorialVersions", {
      tutorialId,
      version: 1,
      content: args.content,
      changeDescription: "Initial version",
      createdAt: now,
      createdBy: args.authorId,
    });

    return tutorialId;
  },
});

// Update tutorial content
export const updateTutorial = mutation({
  args: {
    id: v.id("tutorials"),
    title: v.optional(v.string()),
    description: v.optional(v.string()),
    content: v.optional(v.string()),
    tags: v.optional(v.array(v.string())),
    category: v.optional(v.string()),
    difficulty: v.optional(v.union(
      v.literal("beginner"),
      v.literal("intermediate"),
      v.literal("advanced")
    )),
    isPublic: v.optional(v.boolean()),
    changeDescription: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const tutorial = await ctx.db.get(args.id);
    if (!tutorial) {
      throw new Error("Tutorial not found");
    }

    const updateData: any = {
      updatedAt: Date.now(),
    };

    if (args.title !== undefined) updateData.title = args.title;
    if (args.description !== undefined) updateData.description = args.description;
    if (args.content !== undefined) {
      updateData.content = args.content;
      // Recalculate read time
      const wordCount = args.content.split(/\s+/).length;
      updateData.estimatedReadTime = Math.max(1, Math.ceil(wordCount / 200));
    }
    if (args.tags !== undefined) updateData.tags = args.tags;
    if (args.category !== undefined) updateData.category = args.category;
    if (args.difficulty !== undefined) updateData.difficulty = args.difficulty;
    if (args.isPublic !== undefined) updateData.isPublic = args.isPublic;

    await ctx.db.patch(args.id, updateData);

    // Create new version if content changed
    if (args.content !== undefined) {
      const versions = await ctx.db
        .query("tutorialVersions")
        .withIndex("by_tutorial", (q) => q.eq("tutorialId", args.id))
        .order("desc")
        .collect();

      const nextVersion = versions.length > 0 ? versions[0].version + 1 : 1;

      await ctx.db.insert("tutorialVersions", {
        tutorialId: args.id,
        version: nextVersion,
        content: args.content,
        changeDescription: args.changeDescription || "Content updated",
        createdAt: Date.now(),
        createdBy: tutorial.authorId,
      });
    }

    return args.id;
  },
});

// Publish tutorial
export const publishTutorial = mutation({
  args: {
    id: v.id("tutorials"),
  },
  handler: async (ctx, args) => {
    const tutorial = await ctx.db.get(args.id);
    if (!tutorial) {
      throw new Error("Tutorial not found");
    }

    await ctx.db.patch(args.id, {
      status: "published",
      publishedAt: Date.now(),
      updatedAt: Date.now(),
    });

    return args.id;
  },
});

// Get tutorial by ID
export const getTutorial = query({
  args: { id: v.id("tutorials") },
  handler: async (ctx, args) => {
    const tutorial = await ctx.db.get(args.id);
    if (!tutorial) {
      return null;
    }

    // Note: We can't patch in a query function, so we'll handle view counting differently
    // Views will be incremented when the tutorial is actually viewed

    return tutorial;
  },
});

// Get tutorials by author
export const getTutorialsByAuthor = query({
  args: { authorId: v.string() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("tutorials")
      .withIndex("by_author", (q) => q.eq("authorId", args.authorId))
      .order("desc")
      .collect();
  },
});

// Get published tutorials
export const getPublishedTutorials = query({
  args: {
    category: v.optional(v.string()),
    difficulty: v.optional(v.union(
      v.literal("beginner"),
      v.literal("intermediate"),
      v.literal("advanced")
    )),
    tags: v.optional(v.array(v.string())),
    limit: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    let query = ctx.db
      .query("tutorials")
      .withIndex("by_status", (q) => q.eq("status", "published"))
      .filter((q) => q.eq(q.field("isPublic"), true));

    if (args.category) {
      query = query.filter((q) => q.eq(q.field("category"), args.category));
    }

    if (args.difficulty) {
      query = query.filter((q) => q.eq(q.field("difficulty"), args.difficulty));
    }

    const tutorials = await query.order("desc").collect();
    
    // Apply tag filtering after query execution
    let filteredTutorials = tutorials;
    if (args.tags && args.tags.length > 0) {
      filteredTutorials = tutorials.filter(tutorial => 
        args.tags!.some(tag => tutorial.tags.includes(tag))
      );
    }
    
    if (args.limit) {
      return filteredTutorials.slice(0, args.limit);
    }
    
    return filteredTutorials;
  },
});

// Search tutorials
export const searchTutorials = query({
  args: {
    query: v.string(),
    category: v.optional(v.string()),
    difficulty: v.optional(v.union(
      v.literal("beginner"),
      v.literal("intermediate"),
      v.literal("advanced")
    )),
    limit: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const searchTerm = args.query.toLowerCase();
    
    let tutorials = await ctx.db
      .query("tutorials")
      .withIndex("by_status", (q) => q.eq("status", "published"))
      .filter((q) => q.eq(q.field("isPublic"), true))
      .collect();

    // Filter by search term
    tutorials = tutorials.filter(tutorial => 
      tutorial.title.toLowerCase().includes(searchTerm) ||
      tutorial.description.toLowerCase().includes(searchTerm) ||
      tutorial.content.toLowerCase().includes(searchTerm) ||
      tutorial.tags.some(tag => tag.toLowerCase().includes(searchTerm))
    );

    // Apply additional filters
    if (args.category) {
      tutorials = tutorials.filter(tutorial => tutorial.category === args.category);
    }

    if (args.difficulty) {
      tutorials = tutorials.filter(tutorial => tutorial.difficulty === args.difficulty);
    }

    // Sort by relevance (views + likes)
    tutorials.sort((a, b) => (b.views + b.likes) - (a.views + a.likes));

    if (args.limit) {
      return tutorials.slice(0, args.limit);
    }

    return tutorials;
  },
});

// Get tutorial versions
export const getTutorialVersions = query({
  args: { tutorialId: v.id("tutorials") },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("tutorialVersions")
      .withIndex("by_tutorial", (q) => q.eq("tutorialId", args.tutorialId))
      .order("desc")
      .collect();
  },
});

// Like tutorial
export const likeTutorial = mutation({
  args: { id: v.id("tutorials") },
  handler: async (ctx, args) => {
    const tutorial = await ctx.db.get(args.id);
    if (!tutorial) {
      throw new Error("Tutorial not found");
    }

    await ctx.db.patch(args.id, {
      likes: tutorial.likes + 1,
      updatedAt: Date.now(),
    });

    return tutorial.likes + 1;
  },
});

// Archive tutorial
export const archiveTutorial = mutation({
  args: { id: v.id("tutorials") },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.id, {
      status: "archived",
      updatedAt: Date.now(),
    });

    return args.id;
  },
});

// Increment view count
export const incrementViewCount = mutation({
  args: { id: v.id("tutorials") },
  handler: async (ctx, args) => {
    const tutorial = await ctx.db.get(args.id);
    if (!tutorial) {
      throw new Error("Tutorial not found");
    }

    await ctx.db.patch(args.id, {
      views: tutorial.views + 1,
      updatedAt: Date.now(),
    });

    return tutorial.views + 1;
  },
});

// Delete tutorial
export const deleteTutorial = mutation({
  args: { id: v.id("tutorials") },
  handler: async (ctx, args) => {
    // Delete related data
    const versions = await ctx.db
      .query("tutorialVersions")
      .withIndex("by_tutorial", (q) => q.eq("tutorialId", args.id))
      .collect();

    for (const version of versions) {
      await ctx.db.delete(version._id);
    }

    const collaborations = await ctx.db
      .query("tutorialCollaborations")
      .withIndex("by_tutorial", (q) => q.eq("tutorialId", args.id))
      .collect();

    for (const collaboration of collaborations) {
      await ctx.db.delete(collaboration._id);
    }

    const comments = await ctx.db
      .query("tutorialComments")
      .withIndex("by_tutorial", (q) => q.eq("tutorialId", args.id))
      .collect();

    for (const comment of comments) {
      await ctx.db.delete(comment._id);
    }

    // Delete the tutorial
    await ctx.db.delete(args.id);

    return args.id;
  },
});
