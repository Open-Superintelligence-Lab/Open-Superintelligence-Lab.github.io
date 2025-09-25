'use client';

import React, { useRef, useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Download, RefreshCw, Trash2, BarChart3, TrendingUp, Activity } from 'lucide-react';

interface CanvasResult {
  id: string;
  type: 'chart' | 'metric' | 'text' | 'image';
  title: string;
  content: any;
  timestamp: Date;
  status: 'pending' | 'completed' | 'error';
}

interface CanvasProps {
  projectId: string;
}

export default function Canvas({ projectId }: CanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [results, setResults] = useState<CanvasResult[]>([
    {
      id: '1',
      type: 'text',
      title: 'Welcome to Results Canvas',
      content: 'Your experiment results and visualizations will appear here. Start a conversation with the AI assistant to generate results.',
      timestamp: new Date(),
      status: 'completed'
    }
  ]);
  const [isDrawing, setIsDrawing] = useState(false);

  // Mock function to add new results
  const addResult = (result: Omit<CanvasResult, 'id' | 'timestamp'>) => {
    const newResult: CanvasResult = {
      ...result,
      id: Date.now().toString(),
      timestamp: new Date()
    };
    setResults(prev => [...prev, newResult]);
  };

  // Mock function to simulate drawing on canvas
  const drawOnCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    setIsDrawing(true);
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw a simple chart simulation
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    // Draw a mock line chart
    const points = [
      { x: 50, y: 200 },
      { x: 100, y: 180 },
      { x: 150, y: 160 },
      { x: 200, y: 140 },
      { x: 250, y: 120 },
      { x: 300, y: 100 },
      { x: 350, y: 80 }
    ];
    
    points.forEach((point, index) => {
      if (index === 0) {
        ctx.moveTo(point.x, point.y);
      } else {
        ctx.lineTo(point.x, point.y);
      }
    });
    
    ctx.stroke();
    
    // Add labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px Arial';
    ctx.fillText('Model Performance Over Time', 50, 30);
    ctx.fillText('Epochs', 200, 250);
    
    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.beginPath();
    ctx.moveTo(40, 220);
    ctx.lineTo(360, 220);
    ctx.moveTo(40, 20);
    ctx.lineTo(40, 220);
    ctx.stroke();
    
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const downloadCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const link = document.createElement('a');
    link.download = `canvas-${Date.now()}.png`;
    link.href = canvas.toDataURL();
    link.click();
  };

  const getResultIcon = (type: string) => {
    switch (type) {
      case 'chart': return <BarChart3 className="w-4 h-4" />;
      case 'metric': return <TrendingUp className="w-4 h-4" />;
      case 'text': return <Activity className="w-4 h-4" />;
      case 'image': return <BarChart3 className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  return (
    <div className="h-[600px] flex gap-4">
      {/* Results Panel */}
      <div className="w-1/3">
        <Card className="h-full">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Results
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <ScrollArea className="h-[500px] p-4">
              <div className="space-y-3">
                {results.map((result) => (
                  <div key={result.id} className="p-3 border rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      {getResultIcon(result.type)}
                      <span className="font-medium text-sm">{result.title}</span>
                      <Badge variant="outline" className="text-xs">
                        {result.status}
                      </Badge>
                    </div>
                    <div className="text-xs text-muted-foreground mb-2">
                      {result.timestamp.toLocaleString()}
                    </div>
                    <div className="text-sm">
                      {typeof result.content === 'string' 
                        ? result.content 
                        : JSON.stringify(result.content).substring(0, 100) + '...'
                      }
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      {/* Canvas Area */}
      <div className="flex-1">
        <Card className="h-full">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                Visualization Canvas
              </CardTitle>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={drawOnCanvas} disabled={isDrawing}>
                  <RefreshCw className={`w-4 h-4 ${isDrawing ? 'animate-spin' : ''}`} />
                </Button>
                <Button size="sm" variant="outline" onClick={clearCanvas}>
                  <Trash2 className="w-4 h-4" />
                </Button>
                <Button size="sm" variant="outline" onClick={downloadCanvas}>
                  <Download className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-4">
            <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg h-[500px] flex items-center justify-center">
              <canvas
                ref={canvasRef}
                width={600}
                height={400}
                className="border border-muted rounded bg-background"
              />
            </div>
            <div className="mt-4 text-center text-sm text-muted-foreground">
              <p>Canvas for visualizing experiment results, charts, and data analysis</p>
              <p className="mt-1">Results from AI assistant experiments will be drawn here automatically</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
