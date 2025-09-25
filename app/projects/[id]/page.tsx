'use client';

import React, { useState } from 'react';
import { useQuery, useMutation } from 'convex/react';
import { api } from '../../../convex/_generated/api';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { ArrowLeft, Play, Pause, Square, Settings, GitBranch, ExternalLink, Activity, Terminal, Clock } from 'lucide-react';
import Link from 'next/link';
import { AppLayout } from '@/components/layout/app-layout';
import ChatGPTStyleChatbot from '@/components/chatgpt-style-chatbot';
import Canvas from '@/components/canvas';

export default function ProjectDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const [activeTab, setActiveTab] = useState('chatbot');
  const [resolvedParams, setResolvedParams] = useState<{ id: string } | null>(null);

  // Resolve params
  React.useEffect(() => {
    params.then((resolved) => {
      console.log('Resolved params:', resolved);
      setResolvedParams(resolved);
    }).catch(console.error);
  }, [params]);

  // For now, use a mock project to test the interface
  const mockProject = {
    _id: resolvedParams?.id || 'mock-project',
    name: 'Test Project',
    description: 'A test project for the chatbot interface',
    status: 'running',
    budget: 1000,
    usedBudget: 250,
    createdAt: Date.now()
  };

  // Convex queries - skip for now to test interface
  const project = mockProject;
  const runs: any[] = [];
  const updateRunStatus = useMutation(api.runs.updateStatus);
  const deleteRun = useMutation(api.runs.remove);

  const handleStopRun = async (runId: string) => {
    try {
      await updateRunStatus({
        id: runId as any,
        status: "paused"
      });
    } catch (error) {
      console.error("Error pausing run:", error);
    }
  };

  const handleDeleteRun = async (runId: string) => {
    if (!confirm("Are you sure you want to delete this run?")) return;
    try {
      await deleteRun({ id: runId as any });
    } catch (error) {
      console.error("Error deleting run:", error);
    }
  };

  if (!resolvedParams || !project) {
    return (
      <AppLayout>
        <div className="container mx-auto px-6 py-8">
          <div className="text-center">
            <div>Loading project...</div>
            <div className="text-sm text-muted-foreground mt-2">
              Resolved params: {resolvedParams ? JSON.stringify(resolvedParams) : 'null'}
            </div>
          </div>
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div className="min-h-screen bg-background">
        {/* Header */}
        <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="container mx-auto px-6 py-4">
            <div className="flex items-center gap-4">
              <Link href="/projects">
                <Button variant="ghost" size="sm">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back to Projects
                </Button>
              </Link>
              <div className="flex-1">
                <div className="flex items-center gap-3">
                  <h1 className="text-2xl font-bold">{project.name}</h1>
                  <Badge variant="secondary" className="capitalize">
                    {project.status}
                  </Badge>
                </div>
                <p className="text-muted-foreground">{project.description}</p>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="outline" onClick={() => setActiveTab('advanced')}>
                  <Settings className="w-4 h-4 mr-2" />
                  Advanced View
                </Button>
                <Button onClick={() => setActiveTab('chatbot')}>
                  <Play className="w-4 h-4 mr-2" />
                  Start Chat
                </Button>
              </div>
            </div>
          </div>
        </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="chatbot">AI Assistant</TabsTrigger>
            <TabsTrigger value="canvas">Results Canvas</TabsTrigger>
            <TabsTrigger value="advanced">Advanced Dashboard</TabsTrigger>
          </TabsList>

          {/* Chatbot Tab */}
          <TabsContent value="chatbot" className="space-y-6">
            {resolvedParams && (
              <ChatGPTStyleChatbot 
                projectId={resolvedParams.id as any} 
                projectName={project?.name || 'Unknown Project'} 
              />
            )}
          </TabsContent>

          {/* Canvas Tab */}
          <TabsContent value="canvas" className="space-y-6">
            {resolvedParams && (
              <Canvas projectId={resolvedParams.id} />
            )}
          </TabsContent>

          {/* Advanced Dashboard Tab */}
          <TabsContent value="advanced" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Project Info */}
              <Card>
                <CardHeader>
                  <CardTitle>Project Information</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="text-sm text-muted-foreground">Created</div>
                    <div className="font-semibold">
                      {new Date(project.createdAt).toLocaleDateString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Status</div>
                    <Badge variant="secondary" className="capitalize">
                      {project.status}
                    </Badge>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Total Runs</div>
                    <div className="font-semibold">{runs?.length || 0}</div>
                  </div>
                </CardContent>
              </Card>

              {/* Budget */}
              <Card>
                <CardHeader>
                  <CardTitle>Budget</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Used: ${project.usedBudget.toFixed(2)}</span>
                      <span>Limit: ${project.budget}</span>
                    </div>
                    <Progress value={(project.usedBudget / project.budget) * 100} />
                  </div>
                  <div className="text-sm text-muted-foreground">
                    ${(project.budget - project.usedBudget).toFixed(2)} remaining
                  </div>
                </CardContent>
              </Card>

              {/* Recent Activity */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="w-4 h-4" />
                    Recent Activity
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {runs && runs.length > 0 ? (
                    runs.slice(0, 3).map((run) => (
                      <div key={run._id} className="flex items-center justify-between">
                        <div>
                          <div className="font-semibold text-sm">{run.name}</div>
                          <div className="text-xs text-muted-foreground">{run.status}</div>
                        </div>
                        <Badge variant="outline" className="text-xs">
                          {run.progress}%
                        </Badge>
                      </div>
                    ))
                  ) : (
                    <div className="text-sm text-muted-foreground">No recent activity</div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Recent Runs */}
            <Card>
              <CardHeader>
                <CardTitle>Recent Runs</CardTitle>
                <CardDescription>Latest autonomous runs for this project</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {runs && runs.length > 0 ? (
                    runs.slice(0, 3).map((run) => (
                      <div key={run._id} className="flex items-center justify-between p-4 border rounded-lg">
                        <div className="flex items-center gap-4">
                          <div className={`w-3 h-3 rounded-full ${
                            run.status === 'running' ? 'bg-green-500' :
                            run.status === 'completed' ? 'bg-blue-500' :
                            run.status === 'paused' ? 'bg-yellow-500' :
                            'bg-gray-500'
                          }`} />
                          <div>
                            <div className="font-semibold">{run.name}</div>
                            <div className="text-sm text-muted-foreground">
                              {run.gpuProvider} • Started {new Date(run.startedAt).toLocaleString()}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="text-right">
                            <div className="text-sm font-semibold">${run.cost}</div>
                            <div className="text-sm text-muted-foreground">
                              {run.status === 'running' ? `ETA: ${run.eta}` : run.status}
                            </div>
                          </div>
                          {run.status === 'running' && (
                            <Progress value={run.progress} className="w-20" />
                          )}
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      No runs yet. Launch an agent to get started!
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Runs Tab */}
          <TabsContent value="runs" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold">Runs</h2>
                <p className="text-muted-foreground">All autonomous runs for this project</p>
              </div>
              <Link href="/agents">
                <Button>
                  <Play className="w-4 h-4 mr-2" />
                  Launch Agent
                </Button>
              </Link>
            </div>

            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Progress</TableHead>
                      <TableHead>GPU Provider</TableHead>
                      <TableHead>Cost</TableHead>
                      <TableHead>Started</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {runs && runs.length > 0 ? (
                      runs.map((run) => (
                        <TableRow key={run._id}>
                          <TableCell>
                            <Link href={`/projects/${resolvedParams?.id}/runs/${run._id}`} className="font-semibold hover:underline">
                              {run.name}
                            </Link>
                          </TableCell>
                          <TableCell>
                            <Badge variant="secondary" className="capitalize">
                              {run.status}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <Progress value={run.progress} className="w-16" />
                              <span className="text-sm">{run.progress}%</span>
                            </div>
                          </TableCell>
                          <TableCell>{run.gpuProvider}</TableCell>
                          <TableCell>${run.cost}</TableCell>
                          <TableCell>{new Date(run.startedAt).toLocaleString()}</TableCell>
                          <TableCell>
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
                              {run.status === 'queued' && (
                                <Button size="sm" variant="outline">
                                  <Play className="w-3 h-3" />
                                </Button>
                              )}
                            </div>
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                          No runs yet. Launch an agent to get started!
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Agent Activity Tab */}
          <TabsContent value="activity" className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold">Agent Activity</h2>
              <p className="text-muted-foreground">Real-time tracking of agent actions and API calls</p>
            </div>

            <div className="space-y-4">
              {runs && runs.length > 0 ? (
                runs.map((run) => (
                  <Card key={run._id}>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Terminal className="w-4 h-4" />
                        {run.name}
                      </CardTitle>
                      <CardDescription>
                        Status: {run.status} • Progress: {run.progress}%
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {/* Agent Plan */}
                      {run.config?.plan && (
                        <div>
                          <h4 className="font-semibold mb-2">Generated Plan</h4>
                          <div className="bg-muted p-3 rounded-lg text-sm">
                            <div className="font-semibold mb-2">Objectives:</div>
                            <ul className="list-disc list-inside mb-2">
                              {run.config.plan.objectives?.map((obj: string, i: number) => (
                                <li key={i}>{obj}</li>
                              ))}
                            </ul>
                            <div className="font-semibold mb-2">Experiments:</div>
                            <ul className="list-disc list-inside">
                              {run.config.plan.experiments?.map((exp: any, i: number) => (
                                <li key={i}>{exp.name}: {exp.description}</li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      )}

                      {/* API Call Logs */}
                      <div>
                        <h4 className="font-semibold mb-2">API Activity</h4>
                        <div className="space-y-2">
                          <div className="bg-green-50 border border-green-200 p-3 rounded-lg">
                            <div className="flex items-center gap-2 mb-1">
                              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                              <span className="font-semibold text-sm">Plan Generation</span>
                              <span className="text-xs text-muted-foreground">
                                <Clock className="w-3 h-3 inline mr-1" />
                                {new Date(run.startedAt).toLocaleTimeString()}
                              </span>
                            </div>
                            <div className="text-sm text-muted-foreground">
                              Generated research plan with {run.config?.plan?.experiments?.length || 0} experiments
                            </div>
                          </div>
                          
                          {run.status === 'running' && (
                            <div className="bg-blue-50 border border-blue-200 p-3 rounded-lg">
                              <div className="flex items-center gap-2 mb-1">
                                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                                <span className="font-semibold text-sm">GPU Provisioning</span>
                                <span className="text-xs text-muted-foreground">
                                  <Clock className="w-3 h-3 inline mr-1" />
                                  In progress
                                </span>
                              </div>
                              <div className="text-sm text-muted-foreground">
                                Requesting {run.gpuProvider} resources for experiment execution
                              </div>
                            </div>
                          )}

                          {run.status === 'completed' && (
                            <div className="bg-gray-50 border border-gray-200 p-3 rounded-lg">
                              <div className="flex items-center gap-2 mb-1">
                                <div className="w-2 h-2 bg-gray-500 rounded-full"></div>
                                <span className="font-semibold text-sm">Execution Complete</span>
                                <span className="text-xs text-muted-foreground">
                                  <Clock className="w-3 h-3 inline mr-1" />
                                  {run.endedAt ? new Date(run.endedAt).toLocaleTimeString() : 'Completed'}
                                </span>
                              </div>
                              <div className="text-sm text-muted-foreground">
                                All experiments completed successfully. Total cost: ${run.cost}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))
              ) : (
                <Card>
                  <CardContent className="text-center py-8">
                    <Terminal className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                    <h3 className="font-semibold mb-2">No Agent Activity</h3>
                    <p className="text-muted-foreground mb-4">
                      Launch an AI agent to see real-time activity tracking and API call logs.
                    </p>
                    <Link href="/agents">
                      <Button>
                        <Play className="w-4 h-4 mr-2" />
                        Launch Agent
                      </Button>
                    </Link>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          {/* Artifacts Tab */}
          <TabsContent value="artifacts" className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold">Artifacts</h2>
              <p className="text-muted-foreground">Generated files, models, and outputs</p>
            </div>
            <Card>
              <CardContent className="text-center py-8">
                <p className="text-muted-foreground">Artifacts will appear here as agents generate them.</p>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Reports Tab */}
          <TabsContent value="reports" className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold">Reports</h2>
              <p className="text-muted-foreground">Experiment summaries and comparisons</p>
            </div>
            <Card>
              <CardContent className="text-center py-8">
                <p className="text-muted-foreground">Reports will be generated automatically as experiments complete.</p>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Settings Tab */}
          <TabsContent value="settings" className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold">Project Settings</h2>
              <p className="text-muted-foreground">Configure project settings and preferences</p>
            </div>
            <Card>
              <CardContent className="space-y-4">
                <div>
                  <div className="text-sm text-muted-foreground">Total Budget</div>
                  <div className="font-semibold">${project.budget}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Used Budget</div>
                  <div className="font-semibold">${project.usedBudget.toFixed(2)}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Status</div>
                  <Badge variant="secondary" className="capitalize">
                    {project.status}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
      </div>
    </AppLayout>
  );
}
