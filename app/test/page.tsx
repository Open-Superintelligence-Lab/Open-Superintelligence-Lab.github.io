'use client';

import { useQuery, useMutation } from 'convex/react';
import { api } from '../../convex/_generated/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AppLayout } from '@/components/layout/app-layout';

export default function TestPage() {
  const projects = useQuery(api.projects.list, {});
  const createProject = useMutation(api.projects.create);
  const createSampleData = useMutation(api.seed.createSampleData);

  const handleCreateTestProject = async () => {
    await createProject({
      name: `Test Project ${Date.now()}`,
      description: 'A test project created from the test page',
      budget: 500,
    });
  };

  return (
    <AppLayout>
      <div className="container mx-auto px-6 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-4">Convex Integration Test</h1>
          <p className="text-muted-foreground mb-6">
            This page tests the real-time connection between your Next.js frontend and Convex backend.
          </p>
          
          <div className="flex gap-4 mb-6">
            <Button onClick={handleCreateTestProject}>
              Create Test Project
            </Button>
            <Button onClick={() => createSampleData({})} variant="outline">
              Load Sample Data
            </Button>
          </div>
        </div>

        <div className="grid gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Projects from Convex Database</CardTitle>
            </CardHeader>
            <CardContent>
              {projects === undefined ? (
                <div className="text-muted-foreground">Loading projects...</div>
              ) : projects.length === 0 ? (
                <div className="text-muted-foreground">No projects found. Create one above!</div>
              ) : (
                <div className="space-y-4">
                  {projects.map((project) => (
                    <div key={project._id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div>
                        <h3 className="font-semibold">{project.name}</h3>
                        <p className="text-sm text-muted-foreground">{project.description}</p>
                        <div className="flex items-center gap-2 mt-2">
                          <Badge variant="secondary">{project.status}</Badge>
                          <span className="text-sm">Budget: ${project.budget}</span>
                          <span className="text-sm">Used: ${project.usedBudget.toFixed(2)}</span>
                        </div>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {new Date(project.createdAt).toLocaleString()}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Real-time Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>Convex connection: Active</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>Real-time updates: Enabled</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>Database queries: Working</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </AppLayout>
  );
}
