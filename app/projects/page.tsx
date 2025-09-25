'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Plus, Search, Filter, Play, Pause, Square, MoreHorizontal } from 'lucide-react';
import Link from 'next/link';

// Mock data for projects
const mockProjects = [
  {
    id: '1',
    name: 'Language Model Training',
    description: 'Training a 7B parameter language model on custom dataset',
    status: 'running',
    runs: 3,
    lastRun: '2 hours ago',
    cost: '$45.20',
    models: ['GPT-3.5', 'Llama-2'],
    tags: ['NLP', 'Training']
  },
  {
    id: '2',
    name: 'Computer Vision Classification',
    description: 'Image classification model for medical imaging',
    status: 'completed',
    runs: 8,
    lastRun: '1 day ago',
    cost: '$123.50',
    models: ['ResNet', 'ViT'],
    tags: ['CV', 'Medical']
  },
  {
    id: '3',
    name: 'Reinforcement Learning Agent',
    description: 'RL agent for autonomous navigation',
    status: 'paused',
    runs: 2,
    lastRun: '3 days ago',
    cost: '$78.90',
    models: ['PPO', 'SAC'],
    tags: ['RL', 'Navigation']
  },
  {
    id: '4',
    name: 'Text Generation Research',
    description: 'Exploring novel text generation techniques',
    status: 'failed',
    runs: 1,
    lastRun: '1 week ago',
    cost: '$12.30',
    models: ['GPT-4', 'Claude'],
    tags: ['NLP', 'Research']
  }
];

const statusColors = {
  running: 'bg-green-500',
  completed: 'bg-blue-500',
  paused: 'bg-yellow-500',
  failed: 'bg-red-500'
};

const statusLabels = {
  running: 'Running',
  completed: 'Completed',
  paused: 'Paused',
  failed: 'Failed'
};

export default function ProjectsPage() {
  const [projects] = useState(mockProjects);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');

  const filteredProjects = projects.filter(project => {
    const matchesSearch = project.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         project.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || project.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">Auto AI Research System</h1>
              <p className="text-muted-foreground">Fully autonomous AI research platform</p>
            </div>
            <div className="flex items-center gap-4">
              <Dialog>
                <DialogTrigger asChild>
                  <Button>
                    <Plus className="w-4 h-4 mr-2" />
                    New Project
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-[425px]">
                  <DialogHeader>
                    <DialogTitle>Create New Project</DialogTitle>
                    <DialogDescription>
                      Create a new AI research project. Configure your research goals and connect your resources.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="grid gap-4 py-4">
                    <div className="grid gap-2">
                      <Label htmlFor="name">Project Name</Label>
                      <Input id="name" placeholder="Enter project name" />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="description">Description</Label>
                      <Textarea id="description" placeholder="Describe your research goals" />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="template">Template</Label>
                      <Select>
                        <SelectTrigger>
                          <SelectValue placeholder="Select a template" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="classification">Classification</SelectItem>
                          <SelectItem value="qa">Question Answering</SelectItem>
                          <SelectItem value="lm">Language Modeling</SelectItem>
                          <SelectItem value="rlhf">RLHF</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button variant="outline">Cancel</Button>
                    <Button>Create Project</Button>
                  </div>
                </DialogContent>
              </Dialog>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        {/* Filters */}
        <div className="flex items-center gap-4 mb-8">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
            <Input
              placeholder="Search projects..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-[180px]">
              <Filter className="w-4 h-4 mr-2" />
              <SelectValue placeholder="Filter by status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="running">Running</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="paused">Paused</SelectItem>
              <SelectItem value="failed">Failed</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Projects Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredProjects.map((project) => (
            <Card key={project.id} className="hover:shadow-lg transition-shadow">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <CardTitle className="text-lg">
                      <Link href={`/projects/${project.id}`} className="hover:underline">
                        {project.name}
                      </Link>
                    </CardTitle>
                    <CardDescription className="line-clamp-2">
                      {project.description}
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${statusColors[project.status as keyof typeof statusColors]}`} />
                    <Badge variant="secondary" className="text-xs">
                      {statusLabels[project.status as keyof typeof statusLabels]}
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Stats */}
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground">Runs</div>
                    <div className="font-semibold">{project.runs}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Cost</div>
                    <div className="font-semibold">{project.cost}</div>
                  </div>
                </div>

                {/* Models */}
                <div>
                  <div className="text-sm text-muted-foreground mb-2">Models</div>
                  <div className="flex flex-wrap gap-1">
                    {project.models.map((model) => (
                      <Badge key={model} variant="outline" className="text-xs">
                        {model}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* Tags */}
                <div>
                  <div className="text-sm text-muted-foreground mb-2">Tags</div>
                  <div className="flex flex-wrap gap-1">
                    {project.tags.map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center justify-between pt-2">
                  <div className="text-xs text-muted-foreground">
                    Last run: {project.lastRun}
                  </div>
                  <div className="flex items-center gap-1">
                    {project.status === 'running' && (
                      <>
                        <Button size="sm" variant="outline">
                          <Pause className="w-3 h-3" />
                        </Button>
                        <Button size="sm" variant="outline">
                          <Square className="w-3 h-3" />
                        </Button>
                      </>
                    )}
                    {project.status === 'paused' && (
                      <Button size="sm" variant="outline">
                        <Play className="w-3 h-3" />
                      </Button>
                    )}
                    <Button size="sm" variant="ghost">
                      <MoreHorizontal className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {filteredProjects.length === 0 && (
          <div className="text-center py-12">
            <div className="text-muted-foreground mb-4">
              {searchTerm || statusFilter !== 'all' 
                ? 'No projects match your filters' 
                : 'No projects yet'
              }
            </div>
            {!searchTerm && statusFilter === 'all' && (
              <Button>
                <Plus className="w-4 h-4 mr-2" />
                Create Your First Project
              </Button>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
