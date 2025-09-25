'use client';

import { useState } from 'react';
import { useQuery, useMutation } from 'convex/react';
import { api } from '../../convex/_generated/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Play, 
  Pause, 
  Square, 
  Brain, 
  Zap, 
  Target, 
  TrendingUp,
  Clock,
  Cpu,
  DollarSign,
  Trash2
} from 'lucide-react';
import { AppLayout } from '@/components/layout/app-layout';

export default function AgentDashboard() {
  const [selectedProject, setSelectedProject] = useState<string>('');
  const [researchGoal, setResearchGoal] = useState('');
  const [codebase, setCodebase] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  // Convex queries and mutations
  const projects = useQuery(api.projects.list, {});
  const runs = useQuery(
    api.runs.listByProject, 
    selectedProject ? { projectId: selectedProject as any } : "skip"
  );
  const createAgentPlan = useMutation(api.agents.createAgentPlan);
  const updateRunStatus = useMutation(api.runs.updateStatus);
  const deleteRun = useMutation(api.runs.remove);

  const handleStartAgent = async () => {
    if (!selectedProject || !researchGoal.trim()) return;
    
    setIsCreating(true);
    try {
      // Create a run first, then generate the plan
      const runId = await createAgentPlan({
        projectId: selectedProject as any,
        researchGoal,
        codebase: codebase || undefined,
      });
      
      console.log("Agent started with run ID:", runId);
      
      // Clear the form
      setResearchGoal('');
      setCodebase('');
      
      // Show success message
      alert(`AI Agent launched successfully! Run ID: ${runId}`);
    } catch (error) {
      console.error("Error starting agent:", error);
      alert("Failed to start agent. Check console for details.");
    } finally {
      setIsCreating(false);
    }
  };

  const handleStopRun = async (runId: string) => {
    try {
      await updateRunStatus({
        id: runId as any,
        status: "paused"
      });
      console.log("Run paused successfully");
    } catch (error) {
      console.error("Error pausing run:", error);
      alert("Failed to pause run. Check console for details.");
    }
  };

  const handleDeleteRun = async (runId: string) => {
    if (!confirm("Are you sure you want to delete this run? This action cannot be undone.")) {
      return;
    }
    
    try {
      await deleteRun({
        id: runId as any
      });
      console.log("Run deleted successfully");
    } catch (error) {
      console.error("Error deleting run:", error);
      alert("Failed to delete run. Check console for details.");
    }
  };

  const selectedProjectData = projects?.find(p => p._id === selectedProject);

  return (
    <AppLayout>
      <div className="container mx-auto px-6 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">AI Agent Dashboard</h1>
          <p className="text-muted-foreground">
            Deploy autonomous AI agents to conduct research experiments
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Agent Control Panel */}
          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5" />
                  Agent Control
                </CardTitle>
                <CardDescription>
                  Configure and launch AI research agents
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="project">Select Project</Label>
                  <select
                    id="project"
                    value={selectedProject}
                    onChange={(e) => setSelectedProject(e.target.value)}
                    className="w-full mt-1 p-2 border rounded-md"
                  >
                    <option value="">Choose a project...</option>
                    {projects?.map((project) => (
                      <option key={project._id} value={project._id}>
                        {project.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <Label htmlFor="goal">Research Goal</Label>
                  <Textarea
                    id="goal"
                    placeholder="Describe what you want the agent to research..."
                    value={researchGoal}
                    onChange={(e) => setResearchGoal(e.target.value)}
                    className="mt-1"
                    rows={3}
                  />
                </div>

                <div>
                  <Label htmlFor="codebase">Codebase Context (Optional)</Label>
                  <Textarea
                    id="codebase"
                    placeholder="Paste relevant code or repository information..."
                    value={codebase}
                    onChange={(e) => setCodebase(e.target.value)}
                    className="mt-1"
                    rows={4}
                  />
                </div>

                <Button 
                  onClick={handleStartAgent}
                  disabled={!selectedProject || !researchGoal.trim() || isCreating}
                  className="w-full"
                >
                  <Play className="w-4 h-4 mr-2" />
                  {isCreating ? "Launching Agent..." : "Launch AI Agent"}
                </Button>
              </CardContent>
            </Card>

            {/* Project Info */}
            {selectedProjectData && (
              <Card className="mt-6">
                <CardHeader>
                  <CardTitle>Project Info</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div>
                    <div className="text-sm text-muted-foreground">Budget</div>
                    <div className="font-semibold">${selectedProjectData.budget}</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Used</div>
                    <div className="font-semibold">${selectedProjectData.usedBudget.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Status</div>
                    <Badge variant="secondary" className="capitalize">
                      {selectedProjectData.status}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Active Runs */}
          <div className="lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5" />
                  Active Agent Runs
                </CardTitle>
                <CardDescription>
                  Monitor autonomous research experiments
                </CardDescription>
              </CardHeader>
              <CardContent>
                {!selectedProject ? (
                  <div className="text-center py-8 text-muted-foreground">
                    Select a project to view agent runs
                  </div>
                ) : !runs ? (
                  <div className="text-center py-8 text-muted-foreground">
                    Loading runs...
                  </div>
                ) : runs.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    No agent runs yet. Launch an agent to get started!
                  </div>
                ) : (
                  <div className="space-y-4">
                    {runs.map((run) => (
                      <div key={run._id} className="border rounded-lg p-4">
                        <div className="flex items-center justify-between mb-3">
                          <div>
                            <h3 className="font-semibold">{run.name}</h3>
                            <p className="text-sm text-muted-foreground">
                              {run.config?.researchGoal || 'Research experiment'}
                            </p>
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge variant="secondary" className="capitalize">
                              {run.status}
                            </Badge>
                            <div className="flex items-center gap-1 text-sm text-muted-foreground">
                              <Cpu className="w-3 h-3" />
                              {run.gpuProvider}
                            </div>
                          </div>
                        </div>

                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span>Progress</span>
                            <span>{run.progress}%</span>
                          </div>
                          <Progress value={run.progress} />
                        </div>

                        <div className="flex items-center justify-between mt-3 text-sm text-muted-foreground">
                          <div className="flex items-center gap-4">
                            <div className="flex items-center gap-1">
                              <DollarSign className="w-3 h-3" />
                              {run.cost}
                            </div>
                            <div className="flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              {run.eta || 'Calculating...'}
                            </div>
                          </div>
                          <div className="flex items-center gap-1">
                            {run.status === 'running' && (
                              <>
                                <Button 
                                  size="sm" 
                                  variant="outline"
                                  onClick={() => handleStopRun(run._id)}
                                  title="Pause run"
                                >
                                  <Pause className="w-3 h-3" />
                                </Button>
                                <Button 
                                  size="sm" 
                                  variant="outline"
                                  onClick={() => handleStopRun(run._id)}
                                  title="Stop run"
                                >
                                  <Square className="w-3 h-3" />
                                </Button>
                              </>
                            )}
                            <Button 
                              size="sm" 
                              variant="outline"
                              onClick={() => handleDeleteRun(run._id)}
                              title="Delete run"
                              className="text-red-600 hover:text-red-700"
                            >
                              <Trash2 className="w-3 h-3" />
                            </Button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Agent Capabilities */}
        <Card className="mt-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="w-5 h-5" />
              Agent Capabilities
            </CardTitle>
            <CardDescription>
              What our AI agents can do autonomously
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                  <Brain className="w-6 h-6 text-blue-600" />
                </div>
                <h3 className="font-semibold mb-2">Research Planning</h3>
                <p className="text-sm text-muted-foreground">
                  AI agents analyze your research goals and generate detailed experimental plans
                </p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                  <Zap className="w-6 h-6 text-green-600" />
                </div>
                <h3 className="font-semibold mb-2">Autonomous Execution</h3>
                <p className="text-sm text-muted-foreground">
                  Agents dispatch experiments to GPU providers and monitor progress in real-time
                </p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                  <TrendingUp className="w-6 h-6 text-purple-600" />
                </div>
                <h3 className="font-semibold mb-2">Intelligent Analysis</h3>
                <p className="text-sm text-muted-foreground">
                  AI analyzes results and suggests next steps for continuous improvement
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </AppLayout>
  );
}
