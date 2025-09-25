'use client';

import { useState } from 'react';
import { useQuery, useMutation } from 'convex/react';
import { api } from '../../convex/_generated/api';
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
import { AppLayout } from '@/components/layout/app-layout';

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
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDescription, setNewProjectDescription] = useState('');
  const [newProjectBudget, setNewProjectBudget] = useState(1000);

  // Convex queries and mutations
  const projects = useQuery(api.projects.list, {});
  const createProject = useMutation(api.projects.create);
  const createSampleData = useMutation(api.seed.createSampleData);
  const createAgentPlan = useMutation(api.agents.createAgentPlan);

  const handleCreateProject = async () => {
    if (!newProjectName.trim()) return;
    
    await createProject({
      name: newProjectName,
      description: newProjectDescription,
      budget: newProjectBudget,
    });
    
    setNewProjectName('');
    setNewProjectDescription('');
    setNewProjectBudget(1000);
    setIsCreateDialogOpen(false);
  };

  const handleCreateSampleData = async () => {
    await createSampleData({});
  };

  const handleStartAgent = async (projectId: string, researchGoal: string) => {
    try {
      await createAgentPlan({
        projectId: projectId as any,
        researchGoal,
        codebase: undefined, // Could be enhanced to include actual codebase
      });
    } catch (error) {
      console.error("Error starting agent:", error);
    }
  };

  const filteredProjects = projects?.filter(project => {
    const matchesSearch = project.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         project.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || project.status === statusFilter;
    return matchesSearch && matchesStatus;
  }) || [];

  return (
    <AppLayout>
      <div className="container mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold">Projects</h1>
            <p className="text-muted-foreground">Manage your AI research projects</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex gap-2">
              <Button onClick={handleCreateSampleData} variant="outline">
                Load Sample Data
              </Button>
              <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
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
                      <Input 
                        id="name" 
                        placeholder="Enter project name" 
                        value={newProjectName}
                        onChange={(e) => setNewProjectName(e.target.value)}
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="description">Description</Label>
                      <Textarea 
                        id="description" 
                        placeholder="Describe your research goals"
                        value={newProjectDescription}
                        onChange={(e) => setNewProjectDescription(e.target.value)}
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="budget">Budget ($)</Label>
                      <Input 
                        id="budget" 
                        type="number"
                        placeholder="1000"
                        value={newProjectBudget}
                        onChange={(e) => setNewProjectBudget(Number(e.target.value))}
                      />
                    </div>
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleCreateProject}>
                      Create Project
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>
            </div>
          </div>
        </div>
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
            <Card key={project._id} className="hover:shadow-lg transition-shadow">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <CardTitle className="text-lg">
                      <Link href={`/projects/${project._id}`} className="hover:underline">
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
                    <div className="text-muted-foreground">Budget</div>
                    <div className="font-semibold">${project.budget}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Used</div>
                    <div className="font-semibold">${project.usedBudget.toFixed(2)}</div>
                  </div>
                </div>

                {/* Settings */}
                <div>
                  <div className="text-sm text-muted-foreground mb-2">Configuration</div>
                  <div className="flex flex-wrap gap-1">
                    {project.settings?.model && (
                      <Badge variant="outline" className="text-xs">
                        {project.settings.model}
                      </Badge>
                    )}
                    {project.settings?.batchSize && (
                      <Badge variant="outline" className="text-xs">
                        Batch: {project.settings.batchSize}
                      </Badge>
                    )}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center justify-between pt-2">
                  <div className="text-xs text-muted-foreground">
                    Created: {new Date(project.createdAt).toLocaleDateString()}
                  </div>
                  <div className="flex items-center gap-1">
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => handleStartAgent(project._id, project.description)}
                      title="Start AI Agent Research"
                    >
                      <Play className="w-3 h-3" />
                    </Button>
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
              <div className="flex gap-2 justify-center">
                <Button onClick={handleCreateSampleData} variant="outline">
                  Load Sample Data
                </Button>
                <Button onClick={() => setIsCreateDialogOpen(true)}>
                  <Plus className="w-4 h-4 mr-2" />
                  Create Your First Project
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    </AppLayout>
  );
}
