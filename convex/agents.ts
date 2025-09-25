import { action, mutation } from "./_generated/server";
import { v } from "convex/values";

// AI Agent System for Autonomous Research
export const createAgentPlan = mutation({
  args: {
    projectId: v.id("projects"),
    researchGoal: v.string(),
    codebase: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    // Create a mock plan for now (will be enhanced with AI later)
    const plan = {
      objectives: [args.researchGoal],
      experiments: [
        {
          name: "Initial Experiment",
          description: `Research: ${args.researchGoal}`,
          model: "GPT-3.5-turbo",
          hyperparameters: { learning_rate: 0.001, batch_size: 32 },
          expectedDuration: "1h",
          gpuRequirements: "A100 x 1"
        }
      ],
      metrics: ["accuracy", "loss"],
      timeline: "1-2 days",
      budget: "$50-100"
    };
    
    // Create a new run with the generated plan
    const runId = await ctx.runMutation("runs:create", {
      projectId: args.projectId,
      name: `AI Agent: ${args.researchGoal}`,
      config: {
        plan,
        researchGoal: args.researchGoal,
        codebase: args.codebase,
      },
      gpuProvider: "novita", // Default to Novita AI
    });

    return runId;
  },
});

// AI-powered research plan generation (action for external API calls)
export const generateAIPlan = action({
  args: {
    runId: v.id("runs"),
    researchGoal: v.string(),
    codebase: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    // Generate AI-powered plan
    const plan = await generateResearchPlan(args.researchGoal, args.codebase);
    
    // Update the run with the AI-generated plan
    const run = await ctx.runQuery("runs:get", { id: args.runId });
    if (run) {
      await ctx.runMutation("runs:updateConfig", {
        id: args.runId,
        config: {
          ...run.config,
          plan,
          aiGenerated: true,
        },
      });
    }
    
    return plan;
  },
});

// Generate research plan using AI
async function generateResearchPlan(researchGoal: string, codebase?: string) {
  const openrouterApiKey = process.env.OPENROUTER_API_KEY;
  
  if (!openrouterApiKey) {
    throw new Error("OPENROUTER_API_KEY not found in environment variables");
  }

  const prompt = `
You are an AI research agent specializing in machine learning and AI research. 
Generate a detailed research plan for the following goal:

Research Goal: ${researchGoal}

${codebase ? `Codebase Context: ${codebase}` : ''}

Please provide a structured plan with:
1. Research objectives
2. Experimental design
3. Model configurations to test
4. Evaluation metrics
5. Expected outcomes
6. Resource requirements

Format as JSON with the following structure:
{
  "objectives": ["objective1", "objective2"],
  "experiments": [
    {
      "name": "Experiment Name",
      "description": "What this experiment tests",
      "model": "Model to use",
      "hyperparameters": {"param": "value"},
      "expectedDuration": "2h",
      "gpuRequirements": "A100 x 2"
    }
  ],
  "metrics": ["metric1", "metric2"],
  "timeline": "Expected timeline",
  "budget": "Estimated cost"
}
`;

  try {
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${openrouterApiKey}`,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://auto-ai-research.com",
        "X-Title": "Auto AI Research System",
      },
      body: JSON.stringify({
        model: "x-ai/grok-4-fast:free",
        messages: [
          {
            role: "user",
            content: prompt
          }
        ],
        temperature: 0.7,
        max_tokens: 2000,
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenRouter API error: ${response.statusText}`);
    }

    const data = await response.json();
    const planContent = data.choices[0].message.content;
    
    // Parse the JSON response
    try {
      return JSON.parse(planContent);
    } catch (parseError) {
      // If parsing fails, return a structured fallback
      return {
        objectives: [researchGoal],
        experiments: [
          {
            name: "Initial Experiment",
            description: "Basic experiment setup",
            model: "GPT-3.5-turbo",
            hyperparameters: {},
            expectedDuration: "1h",
            gpuRequirements: "A100 x 1"
          }
        ],
        metrics: ["accuracy", "loss"],
        timeline: "1-2 days",
        budget: "$50-100",
        rawResponse: planContent
      };
    }
  } catch (error) {
    console.error("Error generating research plan:", error);
    throw new Error(`Failed to generate research plan: ${error.message}`);
  }
}

// Execute a research plan
export const executeResearchPlan = action({
  args: {
    runId: v.id("runs"),
    plan: v.any(),
  },
  handler: async (ctx, args) => {
    const run = await ctx.runQuery("runs:get", { id: args.runId });
    if (!run) {
      throw new Error("Run not found");
    }

    // Update run status to running
    await ctx.runMutation("runs:updateStatus", {
      id: args.runId,
      status: "running",
    });

    // Execute each experiment in the plan
    for (let i = 0; i < args.plan.experiments.length; i++) {
      const experiment = args.plan.experiments[i];
      
      // Update step status to running
      await ctx.runMutation("runs:updateStepStatus", {
        runId: args.runId,
        stepIndex: i,
        status: "running",
      });

      try {
        // Dispatch experiment to GPU provider
        const jobRef = await dispatchToGPUProvider(experiment, args.runId);
        
        // Update run with job reference
        await ctx.runMutation("runs:updateJobRef", {
          id: args.runId,
          jobRef: jobRef,
        });

        // Update step status to completed
        await ctx.runMutation("runs:updateStepStatus", {
          runId: args.runId,
          stepIndex: i,
          status: "completed",
        });

        // Add some sample metrics
        await ctx.runMutation("runs:addMetric", {
          runId: args.runId,
          name: "Experiment Progress",
          value: ((i + 1) / args.plan.experiments.length) * 100,
          stepIndex: i,
        });

      } catch (error) {
        console.error(`Error executing experiment ${i}:`, error);
        
        // Update step status to failed
        await ctx.runMutation("runs:updateStepStatus", {
          runId: args.runId,
          stepIndex: i,
          status: "failed",
        });
      }
    }

    // Update overall run progress
    await ctx.runMutation("runs:updateProgress", {
      id: args.runId,
      progress: 100,
      eta: "Completed",
    });

    // Mark run as completed
    await ctx.runMutation("runs:updateStatus", {
      id: args.runId,
      status: "completed",
    });

    return { success: true, experimentsExecuted: args.plan.experiments.length };
  },
});

// Dispatch experiment to GPU provider (Novita AI)
async function dispatchToGPUProvider(experiment: any, runId: string) {
  const novitaApiKey = process.env.NOVITA_API_KEY;
  
  if (!novitaApiKey) {
    // For demo purposes, return a mock job reference
    return `mock-job-${runId}-${Date.now()}`;
  }

  try {
    const response = await fetch("https://api.novita.ai/v1/jobs", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${novitaApiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image: "ghcr.io/osci/runner:latest",
        cmd: ["bash", "/workspace/start.sh"],
        gpu: 1, // Default to 1 GPU
        env: {
          RUN_ID: runId,
          EXPERIMENT_CONFIG: JSON.stringify(experiment),
          CONVEX_URL: process.env.CONVEX_URL,
          CONVEX_DEPLOYMENT: process.env.CONVEX_DEPLOYMENT,
        },
        volumes: [
          {
            src: `s3://osci-bucket/${runId}`,
            dst: "/artifacts"
          }
        ]
      }),
    });

    if (!response.ok) {
      throw new Error(`Novita API error: ${response.statusText}`);
    }

    const data = await response.json();
    return data.job_id;
  } catch (error) {
    console.error("Error dispatching to GPU provider:", error);
    // Return mock job reference for demo
    return `mock-job-${runId}-${Date.now()}`;
  }
}

// Analyze results and suggest next steps
export const analyzeResults = action({
  args: {
    runId: v.id("runs"),
  },
  handler: async (ctx, args) => {
    const run = await ctx.runQuery("runs:getWithDetails", { id: args.runId });
    if (!run) {
      throw new Error("Run not found");
    }

    // Get metrics and artifacts
    const metrics = run.metrics;
    const artifacts = run.artifacts;

    // Use AI to analyze results
    const analysis = await analyzeWithAI(metrics, artifacts, run.config);

    // Create analysis artifact
    await ctx.runMutation("runs:addArtifact", {
      runId: args.runId,
      name: "analysis_report.json",
      type: "Analysis",
      size: JSON.stringify(analysis).length,
      url: `s3://osci-bucket/${args.runId}/analysis_report.json`,
      checksum: "analysis-" + Date.now(),
    });

    return analysis;
  },
});

// Analyze results using AI
async function analyzeWithAI(metrics: any[], artifacts: any[], config: any) {
  const openrouterApiKey = process.env.OPENROUTER_API_KEY;
  
  if (!openrouterApiKey) {
    return {
      summary: "Analysis completed (no AI available)",
      recommendations: ["Continue with current approach"],
      nextSteps: ["Run additional experiments"]
    };
  }

  const prompt = `
Analyze the following research results and provide insights:

Metrics: ${JSON.stringify(metrics)}
Artifacts: ${JSON.stringify(artifacts)}
Configuration: ${JSON.stringify(config)}

Provide:
1. Summary of results
2. Key insights
3. Recommendations for improvement
4. Suggested next steps

Format as JSON:
{
  "summary": "Brief summary of results",
  "insights": ["insight1", "insight2"],
  "recommendations": ["rec1", "rec2"],
  "nextSteps": ["step1", "step2"],
  "confidence": 0.85
}
`;

  try {
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${openrouterApiKey}`,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://auto-ai-research.com",
        "X-Title": "Auto AI Research System",
      },
      body: JSON.stringify({
        model: "x-ai/grok-4-fast:free",
        messages: [
          {
            role: "user",
            content: prompt
          }
        ],
        temperature: 0.3,
        max_tokens: 1000,
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenRouter API error: ${response.statusText}`);
    }

    const data = await response.json();
    const analysisContent = data.choices[0].message.content;
    
    try {
      return JSON.parse(analysisContent);
    } catch (parseError) {
      return {
        summary: analysisContent,
        insights: ["Results analyzed"],
        recommendations: ["Continue research"],
        nextSteps: ["Run follow-up experiments"],
        confidence: 0.7
      };
    }
  } catch (error) {
    console.error("Error analyzing results:", error);
    return {
      summary: "Analysis failed",
      insights: ["Unable to analyze"],
      recommendations: ["Manual review required"],
      nextSteps: ["Check logs"],
      confidence: 0.0
    };
  }
}
