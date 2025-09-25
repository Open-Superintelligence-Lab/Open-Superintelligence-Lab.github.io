import { mutation } from "./_generated/server";

// Create sample data for development
export const createSampleData = mutation({
  args: {},
  handler: async (ctx) => {
    // Create sample projects
    const project1 = await ctx.db.insert("projects", {
      name: "Language Model Training",
      description: "Training a 7B parameter language model on custom dataset",
      ownerId: "user-1",
      status: "running",
      budget: 1000,
      usedBudget: 345.20,
      settings: {
        model: "Llama-2-7B",
        batchSize: 32,
        learningRate: 0.0001,
        epochs: 10,
      },
      createdAt: Date.now() - 86400000, // 1 day ago
      updatedAt: Date.now(),
    });

    const project2 = await ctx.db.insert("projects", {
      name: "Computer Vision Classification",
      description: "Image classification model for medical imaging",
      ownerId: "user-1",
      status: "completed",
      budget: 500,
      usedBudget: 123.50,
      settings: {
        model: "ResNet-50",
        batchSize: 16,
        learningRate: 0.001,
        epochs: 20,
      },
      createdAt: Date.now() - 172800000, // 2 days ago
      updatedAt: Date.now() - 86400000,
    });

    const project3 = await ctx.db.insert("projects", {
      name: "Reinforcement Learning Agent",
      description: "RL agent for autonomous navigation",
      ownerId: "user-1",
      status: "paused",
      budget: 2000,
      usedBudget: 78.90,
      settings: {
        model: "PPO",
        batchSize: 64,
        learningRate: 0.0003,
        epochs: 100,
      },
      createdAt: Date.now() - 259200000, // 3 days ago
      updatedAt: Date.now() - 172800000,
    });

    // Create sample runs for project1
    const run1 = await ctx.db.insert("runs", {
      projectId: project1,
      name: "Base Model Training",
      status: "running",
      progress: 75,
      config: {
        model: "Llama-2-7B",
        batchSize: 32,
        learningRate: 0.0001,
        epochs: 10,
        currentEpoch: 7,
      },
      cost: 45.20,
      gpuProvider: "novita",
      jobRef: "novita-job-123",
      startedAt: Date.now() - 14400000, // 4 hours ago
      eta: "2h 15m",
    });

    const run2 = await ctx.db.insert("runs", {
      projectId: project1,
      name: "Hyperparameter Sweep",
      status: "completed",
      progress: 100,
      config: {
        model: "Llama-2-7B",
        batchSize: 32,
        learningRate: 0.0001,
        epochs: 10,
      },
      cost: 123.50,
      gpuProvider: "novita",
      jobRef: "novita-job-122",
      startedAt: Date.now() - 86400000, // 1 day ago
      endedAt: Date.now() - 43200000, // 12 hours ago
      eta: "—",
    });

    // Create sample steps for run1
    const steps = [
      { name: "Plan Generation", status: "completed" as const, stepIndex: 0 },
      { name: "Environment Setup", status: "completed" as const, stepIndex: 1 },
      { name: "Data Preprocessing", status: "completed" as const, stepIndex: 2 },
      { name: "Model Training", status: "running" as const, stepIndex: 3 },
      { name: "Evaluation", status: "pending" as const, stepIndex: 4 },
      { name: "Analysis & Next Steps", status: "pending" as const, stepIndex: 5 },
    ];

    for (const step of steps) {
      await ctx.db.insert("runSteps", {
        runId: run1,
        stepName: step.name,
        status: step.status,
        description: `Step: ${step.name}`,
        stepIndex: step.stepIndex,
        startedAt: step.status === "completed" || step.status === "running" ? Date.now() - 14400000 : undefined,
        endedAt: step.status === "completed" ? Date.now() - 7200000 : undefined,
        duration: step.status === "completed" ? 3600000 : undefined,
      });
    }

    // Create sample metrics for run1
    const metrics = [
      { name: "Training Loss", value: 2.323, stepIndex: 3 },
      { name: "Validation Accuracy", value: 73.2, stepIndex: 3 },
      { name: "Throughput", value: 1247, stepIndex: 3 },
      { name: "GPU Utilization", value: 97.8, stepIndex: 3 },
    ];

    for (const metric of metrics) {
      await ctx.db.insert("metrics", {
        runId: run1,
        name: metric.name,
        value: metric.value,
        timestamp: Date.now(),
        stepIndex: metric.stepIndex,
      });
    }

    // Create sample artifacts for run1
    const artifacts = [
      {
        name: "model_checkpoint_epoch_7.pt",
        type: "Model Checkpoint",
        size: 2300000000, // 2.3 GB
        url: "s3://bucket/model_checkpoint_epoch_7.pt",
        checksum: "abc123def456",
      },
      {
        name: "training_logs.json",
        type: "Logs",
        size: 450000, // 450 KB
        url: "s3://bucket/training_logs.json",
        checksum: "def456ghi789",
      },
      {
        name: "evaluation_results.csv",
        type: "Metrics",
        size: 125000, // 125 KB
        url: "s3://bucket/evaluation_results.csv",
        checksum: "ghi789jkl012",
      },
    ];

    for (const artifact of artifacts) {
      await ctx.db.insert("artifacts", {
        runId: run1,
        name: artifact.name,
        type: artifact.type,
        size: artifact.size,
        url: artifact.url,
        checksum: artifact.checksum,
        createdAt: Date.now(),
      });
    }

    return {
      projects: [project1, project2, project3],
      runs: [run1, run2],
    };
  },
});
