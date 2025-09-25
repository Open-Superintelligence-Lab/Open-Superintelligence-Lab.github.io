'use client';

import { useState } from 'react';
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

// Mock data for project details
const mockProject = {
  id: '1',
  name: 'Language Model Training',
  description: 'Training a 7B parameter language model on custom dataset with advanced fine-tuning techniques',
  status: 'running',
  created: '2024-01-15',
  owner: 'research-team',
  repo: {
    url: 'https://github.com/research-team/llm-training',
    branch: 'main',
    commit: 'a1b2c3d',
    lastSync: '2 hours ago'
  },
  credentials: {
    openai: 'connected',
    github: 'connected',
    novita: 'connected',
    s3: 'connected'
  },
  budget: {
    total: 1000,
    used: 345.20,
    remaining: 654.80
  }
};

const mockRuns = [
  {
    id: 'run-1',
    name: 'Base Model Training',
    status: 'running',
    progress: 75,
    eta: '2h 15m',
    cost: '$45.20',
    gpu: 'A100 x 4',
    started: '4 hours ago'
  },
  {
    id: 'run-2',
    name: 'Hyperparameter Sweep',
    status: 'completed',
    progress: 100,
    eta: '—',
    cost: '$123.50',
    gpu: 'H100 x 8',
    started: '1 day ago'
  },
  {
    id: 'run-3',
    name: 'Ablation Study',
    status: 'queued',
    progress: 0,
    eta: '—',
    cost: '$0.00',
    gpu: 'Pending',
    started: '—'
  }
];

const mockArtifacts = [
  {
    name: 'model_checkpoint_epoch_10.pt',
    size: '2.3 GB',
    type: 'Model Checkpoint',
    created: '2 hours ago'
  },
  {
    name: 'training_logs.json',
    size: '450 KB',
    type: 'Logs',
    created: '1 hour ago'
  },
  {
    name: 'evaluation_results.csv',
    size: '125 KB',
    type: 'Metrics',
    created: '30 min ago'
  }
];

export default function ProjectDetailPage({ params }: { params: { id: string } }) {
  const [activeTab, setActiveTab] = useState('overview');

  return (
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
                <h1 className="text-2xl font-bold">{mockProject.name}</h1>
                <Badge variant="secondary" className="capitalize">
                  {mockProject.status}
                </Badge>
              </div>
              <p className="text-muted-foreground">{mockProject.description}</p>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline">
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </Button>
              <Button>
                <Play className="w-4 h-4 mr-2" />
                New Run
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="runs">Runs</TabsTrigger>
            <TabsTrigger value="artifacts">Artifacts</TabsTrigger>
            <TabsTrigger value="reports">Reports</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Project Info */}
              <Card>
                <CardHeader>
                  <CardTitle>Project Information</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="text-sm text-muted-foreground">Owner</div>
                    <div className="font-semibold">{mockProject.owner}</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Created</div>
                    <div className="font-semibold">{mockProject.created}</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Status</div>
                    <Badge variant="secondary" className="capitalize">
                      {mockProject.status}
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              {/* Repository */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <GitBranch className="w-4 h-4" />
                    Repository
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="text-sm text-muted-foreground">URL</div>
                    <div className="flex items-center gap-2">
                      <code className="text-sm bg-muted px-2 py-1 rounded">
                        {mockProject.repo.url.split('/').slice(-2).join('/')}
                      </code>
                      <Button size="sm" variant="ghost">
                        <ExternalLink className="w-3 h-3" />
                      </Button>
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Branch</div>
                    <div className="font-semibold">{mockProject.repo.branch}</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Last Sync</div>
                    <div className="font-semibold">{mockProject.repo.lastSync}</div>
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
                      <span>Used: ${mockProject.budget.used}</span>
                      <span>Limit: ${mockProject.budget.total}</span>
                    </div>
                    <Progress value={(mockProject.budget.used / mockProject.budget.total) * 100} />
                  </div>
                  <div className="text-sm text-muted-foreground">
                    ${mockProject.budget.remaining} remaining
                  </div>
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
                  {mockRuns.slice(0, 3).map((run) => (
                    <div key={run.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center gap-4">
                        <div className={`w-3 h-3 rounded-full ${
                          run.status === 'running' ? 'bg-green-500' :
                          run.status === 'completed' ? 'bg-blue-500' :
                          'bg-gray-500'
                        }`} />
                        <div>
                          <div className="font-semibold">{run.name}</div>
                          <div className="text-sm text-muted-foreground">
                            {run.gpu} • Started {run.started}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <div className="text-sm font-semibold">{run.cost}</div>
                          <div className="text-sm text-muted-foreground">
                            {run.status === 'running' ? `ETA: ${run.eta}` : run.status}
                          </div>
                        </div>
                        {run.status === 'running' && (
                          <Progress value={run.progress} className="w-20" />
                        )}
                      </div>
                    </div>
                  ))}
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
              <Button>
                <Play className="w-4 h-4 mr-2" />
                Start New Run
              </Button>
            </div>

            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Progress</TableHead>
                      <TableHead>GPU</TableHead>
                      <TableHead>Cost</TableHead>
                      <TableHead>Started</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {mockRuns.map((run) => (
                      <TableRow key={run.id}>
                        <TableCell>
                          <Link href={`/projects/${params.id}/runs/${run.id}`} className="font-semibold hover:underline">
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
                        <TableCell>{run.gpu}</TableCell>
                        <TableCell>{run.cost}</TableCell>
                        <TableCell>{run.started}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-1">
                            {run.status === 'running' && (
                              <>
                                <Button size="sm" variant="outline">
                                  <Pause className="w-3 h-3" />
                                </Button>
                                <Button size="sm" variant="outline">
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
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Artifacts Tab */}
          <TabsContent value="artifacts" className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold">Artifacts</h2>
              <p className="text-muted-foreground">Generated files, models, and outputs</p>
            </div>

            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Size</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {mockArtifacts.map((artifact, index) => (
                      <TableRow key={index}>
                        <TableCell className="font-semibold">{artifact.name}</TableCell>
                        <TableCell>
                          <Badge variant="outline">{artifact.type}</Badge>
                        </TableCell>
                        <TableCell>{artifact.size}</TableCell>
                        <TableCell>{artifact.created}</TableCell>
                        <TableCell>
                          <Button size="sm" variant="outline">
                            Download
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
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
              <CardHeader>
                <CardTitle>Generate Report</CardTitle>
                <CardDescription>Create automated reports and summaries</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <Button>Generate Experiment Summary</Button>
                  <Button variant="outline">Compare Runs</Button>
                  <Button variant="outline">Export to Paper Draft</Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Settings Tab */}
          <TabsContent value="settings" className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold">Project Settings</h2>
              <p className="text-muted-foreground">Configure credentials, budgets, and preferences</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Credentials</CardTitle>
                  <CardDescription>Connected API keys and services</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {Object.entries(mockProject.credentials).map(([service, status]) => (
                    <div key={service} className="flex items-center justify-between">
                      <div className="capitalize">{service}</div>
                      <Badge variant={status === 'connected' ? 'default' : 'secondary'}>
                        {status}
                      </Badge>
                    </div>
                  ))}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Budget Limits</CardTitle>
                  <CardDescription>Configure spending limits and alerts</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="text-sm text-muted-foreground">Total Budget</div>
                    <div className="font-semibold">${mockProject.budget.total}</div>
                  </div>
                  <Button variant="outline">Update Budget</Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
