// Script to create and run an AI agent
// This will be executed in the browser console

async function createAndRunAgent() {
  try {
    console.log("🤖 Creating AI Agent...");
    
    // First, let's create a sample project if none exists
    const projects = await window.convex.query("projects:list", {});
    console.log("Current projects:", projects);
    
    let projectId;
    if (projects && projects.length > 0) {
      projectId = projects[0]._id;
      console.log("Using existing project:", projects[0].name);
    } else {
      console.log("No projects found. Please create a project first.");
      return;
    }
    
    // Create an AI agent
    const runId = await window.convex.mutation("agents:createAgentPlan", {
      projectId: projectId,
      researchGoal: "Train a language model for code generation using transformer architecture",
      codebase: "Python codebase with PyTorch for neural network training"
    });
    
    console.log("✅ Agent created successfully! Run ID:", runId);
    
    // Simulate agent execution
    const result = await window.convex.mutation("agents:simulateAgentExecution", {
      runId: runId
    });
    
    console.log("🚀 Agent execution completed:", result);
    console.log("🎉 Your AI agent is now running! Check the dashboard to see progress.");
    
  } catch (error) {
    console.error("❌ Error creating agent:", error);
  }
}

// Run the agent creation
createAndRunAgent();
