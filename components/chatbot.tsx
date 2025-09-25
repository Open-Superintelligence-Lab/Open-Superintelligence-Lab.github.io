'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useAction } from 'convex/react';
import { api } from '../../../convex/_generated/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Send, Bot, User, Loader2, Play, Pause, Square, CheckCircle, AlertCircle } from 'lucide-react';

interface Message {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  isTyping?: boolean;
  tools?: ToolExecution[];
}

interface ToolExecution {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: any;
  error?: string;
}

interface ChatbotProps {
  projectId: string;
  projectName: string;
}

// Mock MCP Tools
const mockTools = {
  'run_experiment': {
    name: 'Run Experiment',
    description: 'Execute a machine learning experiment',
    execute: async (params: any) => {
      await new Promise(resolve => setTimeout(resolve, 2000));
      return {
        experimentId: `exp_${Date.now()}`,
        status: 'running',
        progress: 0,
        estimatedTime: '2-3 hours',
        gpuAllocated: 'A100 x 2'
      };
    }
  },
  'analyze_data': {
    name: 'Analyze Data',
    description: 'Perform data analysis and visualization',
    execute: async (params: any) => {
      await new Promise(resolve => setTimeout(resolve, 1500));
      return {
        analysisId: `analysis_${Date.now()}`,
        insights: ['Data shows normal distribution', 'Outliers detected in 3% of samples'],
        charts: ['distribution_plot.png', 'correlation_matrix.png']
      };
    }
  },
  'train_model': {
    name: 'Train Model',
    description: 'Train a machine learning model',
    execute: async (params: any) => {
      await new Promise(resolve => setTimeout(resolve, 3000));
      return {
        modelId: `model_${Date.now()}`,
        accuracy: 0.94,
        loss: 0.12,
        epochs: 50,
        trainingTime: '45 minutes'
      };
    }
  },
  'deploy_model': {
    name: 'Deploy Model',
    description: 'Deploy model to production',
    execute: async (params: any) => {
      await new Promise(resolve => setTimeout(resolve, 1000));
      return {
        deploymentId: `deploy_${Date.now()}`,
        endpoint: 'https://api.example.com/model/predict',
        status: 'active',
        latency: '45ms'
      };
    }
  }
};

// This will be replaced by the Convex action call

export default function Chatbot({ projectId, projectName }: ChatbotProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      content: `Hello! I'm your AI research assistant for the "${projectName}" project. I can help you run experiments, analyze data, train models, and deploy them using various tools. What would you like to work on today?`,
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  
  // Use Convex action for AI chat
  const chatWithGrok = useAction(api.chat.chatWithGrok);

  const scrollToBottom = () => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await chatWithGrok({
        message: inputMessage,
        context: projectName,
        projectName: projectName
      });
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.response,
        timestamp: new Date(),
        tools: response.tools.map((toolName: string) => ({
          id: `${toolName}_${Date.now()}`,
          name: mockTools[toolName as keyof typeof mockTools].name,
          status: 'pending' as const
        }))
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Execute tools if any
      if (response.tools.length > 0) {
        for (const toolName of response.tools) {
          const tool = mockTools[toolName as keyof typeof mockTools];
          if (tool) {
            // Update tool status to running
            setMessages(prev => prev.map(msg => 
              msg.id === assistantMessage.id 
                ? {
                    ...msg,
                    tools: msg.tools?.map(t => 
                      t.name === tool.name 
                        ? { ...t, status: 'running' as const }
                        : t
                    )
                  }
                : msg
            ));

            try {
              const result = await tool.execute(response.toolParams);
              
              // Update tool status to completed
              setMessages(prev => prev.map(msg => 
                msg.id === assistantMessage.id 
                  ? {
                      ...msg,
                      tools: msg.tools?.map(t => 
                        t.name === tool.name 
                          ? { ...t, status: 'completed' as const, result }
                          : t
                      )
                    }
                  : msg
              ));

              // Add result message
              const resultMessage: Message = {
                id: (Date.now() + 2).toString(),
                type: 'system',
                content: `✅ ${tool.name} completed successfully!`,
                timestamp: new Date()
              };
              setMessages(prev => [...prev, resultMessage]);

            } catch (error) {
              // Update tool status to failed
              setMessages(prev => prev.map(msg => 
                msg.id === assistantMessage.id 
                  ? {
                      ...msg,
                      tools: msg.tools?.map(t => 
                        t.name === tool.name 
                          ? { ...t, status: 'failed' as const, error: error.message }
                          : t
                      )
                    }
                  : msg
              ));
            }
          }
        }
      }

    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: "I'm sorry, I encountered an error. Please try again.",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getToolIcon = (status: string) => {
    switch (status) {
      case 'pending': return <Play className="w-3 h-3" />;
      case 'running': return <Loader2 className="w-3 h-3 animate-spin" />;
      case 'completed': return <CheckCircle className="w-3 h-3 text-green-500" />;
      case 'failed': return <AlertCircle className="w-3 h-3 text-red-500" />;
      default: return <Play className="w-3 h-3" />;
    }
  };

  return (
    <div className="h-[600px] flex flex-col">
      <Card className="flex-1 flex flex-col">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bot className="w-5 h-5" />
            AI Research Assistant
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Chat with your AI assistant to run experiments, analyze data, and manage your research
          </p>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col p-0">
          <ScrollArea className="flex-1 p-4" ref={scrollAreaRef}>
            <div className="space-y-4">
              {messages.map((message) => (
                <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] rounded-lg p-3 ${
                    message.type === 'user' 
                      ? 'bg-primary text-primary-foreground' 
                      : message.type === 'system'
                      ? 'bg-muted text-muted-foreground'
                      : 'bg-muted'
                  }`}>
                    <div className="flex items-center gap-2 mb-2">
                      {message.type === 'user' ? (
                        <User className="w-4 h-4" />
                      ) : message.type === 'assistant' ? (
                        <Bot className="w-4 h-4" />
                      ) : (
                        <CheckCircle className="w-4 h-4" />
                      )}
                      <span className="text-xs opacity-70">
                        {message.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-sm">{message.content}</p>
                    
                    {/* Tool executions */}
                    {message.tools && message.tools.length > 0 && (
                      <div className="mt-3 space-y-2">
                        {message.tools.map((tool) => (
                          <div key={tool.id} className="flex items-center gap-2 p-2 bg-background rounded border">
                            {getToolIcon(tool.status)}
                            <span className="text-xs font-medium">{tool.name}</span>
                            <Badge variant="outline" className="text-xs">
                              {tool.status}
                            </Badge>
                            {tool.result && (
                              <div className="text-xs text-muted-foreground ml-auto">
                                {JSON.stringify(tool.result).substring(0, 50)}...
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-muted rounded-lg p-3">
                    <div className="flex items-center gap-2">
                      <Bot className="w-4 h-4" />
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-sm">AI is thinking...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
          
          <div className="p-4 border-t">
            <div className="flex gap-2">
              <Input
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me to run experiments, analyze data, or train models..."
                disabled={isLoading}
                className="flex-1"
              />
              <Button 
                onClick={handleSendMessage} 
                disabled={!inputMessage.trim() || isLoading}
                size="sm"
              >
                <Send className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
