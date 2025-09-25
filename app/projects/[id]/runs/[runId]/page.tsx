'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { 
  ArrowLeft, 
  Play, 
  Pause, 
  Square, 
  Clock, 
  Cpu, 
  DollarSign, 
  Activity,
  FileText,
  TrendingUp,
  AlertCircle
} from 'lucide-react';
import Link from 'next/link';

// Mock data for run details
const mockRun = {
  id: 'run-1',
  name: 'Base Model Training',
  status: 'running',
  progress: 75,
  eta: '2h 15m',
  cost: '$45.20',
  gpu: 'A100 x 4',
  started: '4 hours ago',
  projectId: '1',
  config: {
    model: 'Llama-2-7B',
    batchSize: 32,
    learningRate: 0.0001,
    epochs: 10,
    currentEpoch: 7
  }
};

const mockTimeline = [
  {
    step: 'Plan Generation',
    status: 'completed',
    duration: '2m 15s',
    description: 'Agent analyzed repository and generated training plan',
    timestamp: '4 hours ago'
  },
  {
    step: 'Environment Setup',
    status: 'completed', 
    duration: '5m 30s',
    description: 'GPU provisioning and dependency installation',
    timestamp: '3h 55m ago'
  },
  {
    step: 'Data Preprocessing',
    status: 'completed',
    duration: '12m 45s', 
    description: 'Dataset loading and tokenization',
    timestamp: '3h 42m ago'
  },
  {
    step: 'Model Training',
    status: 'running',
    duration: '3h 29m',
    description: 'Training epochs 1-7 of 10',
    timestamp: '3h 29m ago'
  },
  {
    step: 'Evaluation',
    status: 'pending',
    duration: '—',
    description: 'Model evaluation on validation set',
    timestamp: '—'
  },
  {
    step: 'Analysis & Next Steps',
    status: 'pending',
    duration: '—',
    description: 'Performance analysis and plan updates',
    timestamp: '—'
  }
];

const mockLogs = [
  '[15:42:33] INFO: Starting training epoch 7/10',
  '[15:42:33] INFO: Loading checkpoint from epoch 6',
  '[15:42:35] INFO: Checkpoint loaded successfully',
  '[15:42:35] INFO: Resuming training...',
  '[15:43:12] INFO: Batch 1/2847 | Loss: 2.341 | LR: 9.5e-05',
  '[15:43:15] INFO: Batch 2/2847 | Loss: 2.338 | LR: 9.5e-05',
  '[15:43:18] INFO: Batch 3/2847 | Loss: 2.335 | LR: 9.5e-05',
  '[15:43:21] INFO: Batch 4/2847 | Loss: 2.332 | LR: 9.5e-05',
  '[15:43:24] INFO: Batch 5/2847 | Loss: 2.329 | LR: 9.5e-05',
  '[15:43:27] INFO: GPU Memory: 78.2GB / 80GB (97.8%)',
  '[15:43:30] INFO: Batch 6/2847 | Loss: 2.326 | LR: 9.5e-05',
  '[15:43:33] INFO: Batch 7/2847 | Loss: 2.323 | LR: 9.5e-05'
];

const mockMetrics = [
  { name: 'Training Loss', value: '2.323', trend: 'down' },
  { name: 'Validation Accuracy', value: '73.2%', trend: 'up' },
  { name: 'Throughput', value: '1,247 tok/s', trend: 'stable' },
  { name: 'GPU Utilization', value: '97.8%', trend: 'stable' }
];

export default function RunDetailPage({ params }: { params: { id: string; runId: string } }) {
  const [activeTab, setActiveTab] = useState('timeline');

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center gap-4">
            <Link href={`/projects/${params.id}`}>
              <Button variant="ghost" size="sm">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Project
              </Button>
            </Link>
            <div className="flex-1">
              <div className="flex items-center gap-3">
                <h1 className="text-2xl font-bold">{mockRun.name}</h1>
                <Badge variant="secondary" className="capitalize">
                  {mockRun.status}
                </Badge>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Clock className="w-4 h-4" />
                  Started {mockRun.started}
                </div>
              </div>
              <div className="flex items-center gap-4 text-sm text-muted-foreground mt-1">
                <span className="flex items-center gap-1">
                  <Cpu className="w-4 h-4" />
                  {mockRun.gpu}
                </span>
                <span className="flex items-center gap-1">
                  <DollarSign className="w-4 h-4" />
                  {mockRun.cost}
                </span>
                <span>ETA: {mockRun.eta}</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline">
                <Pause className="w-4 h-4 mr-2" />
                Pause
              </Button>
              <Button variant="outline">
                <Square className="w-4 h-4 mr-2" />
                Stop
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Progress Bar */}
      <div className="border-b bg-muted/30">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <div className="flex justify-between text-sm mb-2">
                <span>Overall Progress</span>
                <span>{mockRun.progress}% complete</span>
              </div>
              <Progress value={mockRun.progress} className="h-2" />
            </div>
            <div className="text-sm text-muted-foreground">
              Epoch {mockRun.config.currentEpoch}/{mockRun.config.epochs}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left Sidebar - Quick Stats */}
          <div className="space-y-6">
            {/* Live Metrics */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="w-4 h-4" />
                  Live Metrics
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {mockMetrics.map((metric) => (
                  <div key={metric.name} className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">{metric.name}</span>
                      <div className="flex items-center gap-1">
                        <TrendingUp className={`w-3 h-3 ${
                          metric.trend === 'up' ? 'text-green-500' :
                          metric.trend === 'down' ? 'text-red-500' :
                          'text-gray-500'
                        }`} />
                      </div>
                    </div>
                    <div className="font-semibold">{metric.value}</div>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Configuration */}
            <Card>
              <CardHeader>
                <CardTitle>Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <div className="text-sm text-muted-foreground">Model</div>
                  <div className="font-semibold">{mockRun.config.model}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Batch Size</div>
                  <div className="font-semibold">{mockRun.config.batchSize}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Learning Rate</div>
                  <div className="font-semibold">{mockRun.config.learningRate}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Epochs</div>
                  <div className="font-semibold">{mockRun.config.epochs}</div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Main Content Area */}
          <div className="lg:col-span-3">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="timeline">Timeline</TabsTrigger>
                <TabsTrigger value="logs">Live Logs</TabsTrigger>
                <TabsTrigger value="charts">Charts</TabsTrigger>
                <TabsTrigger value="config">Config</TabsTrigger>
              </TabsList>

              {/* Timeline Tab */}
              <TabsContent value="timeline" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Execution Timeline</CardTitle>
                    <CardDescription>Agent steps and their current status</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-6">
                      {mockTimeline.map((step, index) => (
                        <div key={index} className="flex items-start gap-4">
                          <div className="flex flex-col items-center">
                            <div className={`w-4 h-4 rounded-full border-2 ${
                              step.status === 'completed' ? 'bg-green-500 border-green-500' :
                              step.status === 'running' ? 'bg-blue-500 border-blue-500 animate-pulse' :
                              'bg-gray-200 border-gray-300'
                            }`} />
                            {index < mockTimeline.length - 1 && (
                              <div className={`w-0.5 h-12 ${
                                step.status === 'completed' ? 'bg-green-200' : 'bg-gray-200'
                              }`} />
                            )}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between">
                              <h4 className="font-semibold">{step.step}</h4>
                              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                <Badge variant="outline" className="capitalize">
                                  {step.status}
                                </Badge>
                                <span>{step.duration}</span>
                              </div>
                            </div>
                            <p className="text-sm text-muted-foreground mt-1">{step.description}</p>
                            <p className="text-xs text-muted-foreground mt-1">{step.timestamp}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Logs Tab */}
              <TabsContent value="logs" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <FileText className="w-4 h-4" />
                      Live Logs
                    </CardTitle>
                    <CardDescription>Real-time execution logs</CardDescription>
                  </CardHeader>
                  <CardContent className="p-0">
                    <ScrollArea className="h-96 w-full">
                      <div className="p-4 font-mono text-sm space-y-1">
                        {mockLogs.map((log, index) => (
                          <div key={index} className="flex items-start gap-2">
                            <span className="text-muted-foreground whitespace-nowrap">
                              {index + 1}
                            </span>
                            <span className="break-all">{log}</span>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Charts Tab */}
              <TabsContent value="charts" className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card>
                    <CardHeader>
                      <CardTitle>Training Loss</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="h-64 flex items-center justify-center text-muted-foreground">
                        Training loss chart would be rendered here
                      </div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader>
                      <CardTitle>Validation Accuracy</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="h-64 flex items-center justify-center text-muted-foreground">
                        Validation accuracy chart would be rendered here
                      </div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader>
                      <CardTitle>GPU Utilization</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="h-64 flex items-center justify-center text-muted-foreground">
                        GPU utilization chart would be rendered here
                      </div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader>
                      <CardTitle>Throughput</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="h-64 flex items-center justify-center text-muted-foreground">
                        Throughput chart would be rendered here
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              {/* Config Tab */}
              <TabsContent value="config" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Run Configuration</CardTitle>
                    <CardDescription>View and edit run parameters</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="p-4 bg-muted rounded-lg">
                        <div className="flex items-start gap-2 mb-2">
                          <AlertCircle className="w-4 h-4 text-amber-500 mt-0.5" />
                          <div className="text-sm">
                            <div className="font-semibold">Configuration is read-only during execution</div>
                            <div className="text-muted-foreground">Pause the run to make changes</div>
                          </div>
                        </div>
                      </div>
                      <div className="font-mono text-sm bg-muted p-4 rounded-lg">
                        <pre>{JSON.stringify(mockRun.config, null, 2)}</pre>
                      </div>
                      <Button variant="outline" disabled>
                        Edit Configuration
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </main>
    </div>
  );
}
