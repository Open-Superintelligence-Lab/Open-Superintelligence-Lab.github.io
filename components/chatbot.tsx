'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useAction, useMutation, useQuery } from 'convex/react';
import { api } from '../convex/_generated/api';
import { Id } from '../convex/_generated/dataModel';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Send, Bot, User, Loader2, Play, Pause, Square, CheckCircle, AlertCircle, Copy, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';

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
  projectId: Id<"projects">;
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
  },
  'create_colab_notebook': {
    name: 'Create Colab Notebook',
    description: 'Generate and create a Google Colab notebook with code',
    execute: async (params: any) => {
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Generate a Colab notebook with the provided code
      const notebookId = `colab_${Date.now()}`;
      const colabUrl = `https://colab.research.google.com/drive/${notebookId}`;
      
      // Create notebook content (simplified JSON structure for Colab)
      const notebookContent = {
        cells: [
          {
            cell_type: 'markdown',
            source: ['# AI Generated Notebook\n', 'Created by AI Research Assistant\n', `Generated at: ${new Date().toLocaleString()}`]
          },
          {
            cell_type: 'code',
            source: params.code || ['# Add your code here\n', 'print("Hello from AI-generated Colab notebook!")']
          }
        ],
        metadata: {
          accelerator: 'GPU',
          colab: {
            name: params.title || 'AI Generated Notebook',
            version: '0.3.2'
          }
        }
      };
      
      return {
        notebookId,
        colabUrl,
        title: params.title || 'AI Generated Notebook',
        status: 'created',
        message: 'Notebook created successfully! Click the link to open in Google Colab.',
        cells: notebookContent.cells.length,
        hasCode: true
      };
    }
  }
};

// This will be replaced by the Convex action call

export default function Chatbot({ projectId, projectName }: ChatbotProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copiedCodeBlocks, setCopiedCodeBlocks] = useState<Set<string>>(new Set());
  const [copiedMessages, setCopiedMessages] = useState<Set<string>>(new Set());
  const [currentConversationId, setCurrentConversationId] = useState<Id<"conversations"> | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  
  // Convex hooks
  const chatWithGrok = useAction(api.chat.chatWithGrok);
  const createConversation = useMutation(api.chat.createConversation);
  const addMessage = useMutation(api.chat.addMessage);
  const conversations = useQuery(api.chat.getConversations, { projectId });
  const conversationMessages = useQuery(
    api.chat.getMessages, 
    currentConversationId ? { conversationId: currentConversationId } : "skip"
  );

  const scrollToBottom = () => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load conversation messages when conversation changes
  useEffect(() => {
    if (conversationMessages) {
      const formattedMessages: Message[] = conversationMessages.map(msg => ({
        id: msg._id,
        type: msg.role as 'user' | 'assistant' | 'system',
        content: msg.content,
        timestamp: new Date(msg.timestamp),
        tools: msg.tools
      }));
      setMessages(formattedMessages);
    }
  }, [conversationMessages]);

  // Create initial conversation if none exists
  useEffect(() => {
    if (!currentConversationId && conversations && conversations.length === 0) {
      createConversation({ 
        projectId, 
        title: `Chat with ${projectName}` 
      }).then(conversationId => {
        setCurrentConversationId(conversationId);
        // Add welcome message
        addMessage({
          conversationId,
          role: 'assistant',
          content: `Hello! I'm your AI research assistant for the "${projectName}" project. 

I can help you with:
- **General questions** about your research and project
- **Running experiments** (say "run experiment" to use tools)
- **Data analysis** (say "analyze data" to use tools)  
- **Model training** (say "train model" to use tools)
- **Model deployment** (say "deploy model" to use tools)
- **Google Colab notebooks** (say "create colab notebook" to generate notebooks)

I'll only use tools when you explicitly ask me to run experiments or use MCP tools. Otherwise, I'll just chat and provide guidance.

What would you like to work on today?`
        });
      });
    } else if (conversations && conversations.length > 0 && !currentConversationId) {
      // Load the most recent conversation
      setCurrentConversationId(conversations[0]._id);
    }
  }, [conversations, currentConversationId, projectId, projectName, createConversation, addMessage]);

  const copyToClipboard = async (text: string, type: 'code' | 'message', id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      if (type === 'code') {
        setCopiedCodeBlocks(prev => new Set([...prev, id]));
        setTimeout(() => {
          setCopiedCodeBlocks(prev => {
            const newSet = new Set(prev);
            newSet.delete(id);
            return newSet;
          });
        }, 2000);
      } else {
        setCopiedMessages(prev => new Set([...prev, id]));
        setTimeout(() => {
          setCopiedMessages(prev => {
            const newSet = new Set(prev);
            newSet.delete(id);
            return newSet;
          });
        }, 2000);
      }
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading || !currentConversationId) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    
    // Save user message to database
    await addMessage({
      conversationId: currentConversationId,
      role: 'user',
      content: inputMessage
    });

    const messageToSend = inputMessage;
    setInputMessage('');
    setIsLoading(true);

    try {
      // Check if user explicitly wants to run experiments or use tools
      const shouldUseTools = messageToSend.toLowerCase().includes('run experiment') || 
                           messageToSend.toLowerCase().includes('use mcp') ||
                           messageToSend.toLowerCase().includes('run tool') ||
                           messageToSend.toLowerCase().includes('execute') ||
                           messageToSend.toLowerCase().includes('train model') ||
                           messageToSend.toLowerCase().includes('analyze data') ||
                           messageToSend.toLowerCase().includes('deploy model') ||
                           messageToSend.toLowerCase().includes('create colab') ||
                           messageToSend.toLowerCase().includes('colab notebook') ||
                           messageToSend.toLowerCase().includes('generate notebook') ||
                           messageToSend.toLowerCase().includes('open colab');

      // Prepare conversation history for the AI
      const conversationHistory = messages.slice(0, -1).map(msg => ({
        role: msg.type as 'user' | 'assistant' | 'system',
        content: msg.content
      }));

      const response = await chatWithGrok({
        message: messageToSend,
        context: projectName,
        projectName: projectName,
        conversationHistory
      });
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.response,
        timestamp: new Date(),
        tools: shouldUseTools && response.tools ? response.tools.map((toolName: string) => ({
          id: `${toolName}_${Date.now()}`,
          name: mockTools[toolName as keyof typeof mockTools].name,
          status: 'pending' as const
        })) : []
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Save assistant message to database
      await addMessage({
        conversationId: currentConversationId,
        role: 'assistant',
        content: response.response,
        tools: shouldUseTools && response.tools ? response.tools.map((toolName: string) => ({
          id: `${toolName}_${Date.now()}`,
          name: mockTools[toolName as keyof typeof mockTools].name,
          status: 'pending' as const
        })) : []
      });

      // Execute tools if any and user explicitly requested them
      if (shouldUseTools && response.tools && response.tools.length > 0) {
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
              const resultContent = tool.name === 'Create Colab Notebook' && (result as any).colabUrl 
                ? `✅ ${tool.name} completed successfully!\n\n📓 **Notebook Created**: ${(result as any).title}\n🔗 **Colab Link**: [Open in Google Colab](${(result as any).colabUrl})\n\nClick the link above to open your notebook in Google Colab and start running your code!`
                : `✅ ${tool.name} completed successfully!`;
              
              const resultMessage: Message = {
                id: (Date.now() + 2).toString(),
                type: 'system',
                content: resultContent,
                timestamp: new Date()
              };
              setMessages(prev => [...prev, resultMessage]);

              // Save result message to database
              await addMessage({
                conversationId: currentConversationId,
                role: 'system',
                content: resultContent
              });

            } catch (error) {
              // Update tool status to failed
              setMessages(prev => prev.map(msg => 
                msg.id === assistantMessage.id 
                  ? {
                      ...msg,
                      tools: msg.tools?.map(t => 
                        t.name === tool.name 
                          ? { ...t, status: 'failed' as const, error: error instanceof Error ? error.message : String(error) }
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
      const errorContent = "I'm sorry, I encountered an error. Please try again.";
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: errorContent,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);

      // Save error message to database
      if (currentConversationId) {
        await addMessage({
          conversationId: currentConversationId,
          role: 'assistant',
          content: errorContent
        });
      }
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
                  <div className={`max-w-[80%] rounded-lg p-3 relative group ${
                    message.type === 'user' 
                      ? 'bg-primary text-primary-foreground' 
                      : message.type === 'system'
                      ? 'bg-muted text-muted-foreground'
                      : 'bg-muted'
                  }`}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
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
                      <Button
                        size="sm"
                        variant="ghost"
                        className="opacity-0 group-hover:opacity-100 transition-opacity h-6 w-6 p-0"
                        onClick={() => copyToClipboard(message.content, 'message', message.id)}
                      >
                        {copiedMessages.has(message.id) ? (
                          <Check className="w-3 h-3 text-green-600" />
                        ) : (
                          <Copy className="w-3 h-3" />
                        )}
                      </Button>
                    </div>
                    <div className="text-sm chatbot-markdown">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeHighlight]}
                        components={{
                          code: ({ className, children, inline, ...props }: any) => {
                            const match = /language-(\w+)/.exec(className || '');
                            const codeId = `${message.id}_${Date.now()}_${Math.random()}`;
                            
                            if (!inline && match) {
                              const isCopied = copiedCodeBlocks.has(codeId);
                              
                              const handleCopyCode = () => {
                                // Get the text content from the DOM element
                                const codeElement = document.querySelector(`[data-code-id="${codeId}"]`);
                                if (codeElement) {
                                  const textContent = codeElement.textContent || '';
                                  copyToClipboard(textContent, 'code', codeId);
                                }
                              };
                              
                              return (
                                <div className="relative group">
                                  <pre className="bg-gray-100 dark:bg-gray-800 rounded p-4 overflow-x-auto">
                                    <code 
                                      className={className} 
                                      data-code-id={codeId}
                                      {...props}
                                    >
                                      {children}
                                    </code>
                                  </pre>
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-white/80 dark:bg-gray-700/80 hover:bg-white dark:hover:bg-gray-700"
                                    onClick={handleCopyCode}
                                  >
                                    {isCopied ? (
                                      <Check className="w-3 h-3 text-green-600" />
                                    ) : (
                                      <Copy className="w-3 h-3" />
                                    )}
                                  </Button>
                                </div>
                              );
                            }
                            
                            return (
                              <code className="bg-gray-100 dark:bg-gray-800 rounded px-1 py-0.5 text-xs font-mono" {...props}>
                                {children}
                              </code>
                            );
                          },
                          pre: ({ children }) => <>{children}</>,
                          p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                          ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
                          ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
                          li: ({ children }) => <li className="text-sm">{children}</li>,
                          h1: ({ children }) => <h1 className="text-lg font-bold mb-2">{children}</h1>,
                          h2: ({ children }) => <h2 className="text-base font-bold mb-2">{children}</h2>,
                          h3: ({ children }) => <h3 className="text-sm font-bold mb-1">{children}</h3>,
                          blockquote: ({ children }) => <blockquote className="border-l-4 border-gray-300 pl-4 italic mb-2">{children}</blockquote>,
                          table: ({ children }) => <table className="border-collapse border border-gray-300 w-full mb-2">{children}</table>,
                          th: ({ children }) => <th className="border border-gray-300 px-2 py-1 bg-gray-100 dark:bg-gray-700 font-bold text-xs">{children}</th>,
                          td: ({ children }) => <td className="border border-gray-300 px-2 py-1 text-xs">{children}</td>,
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
                    </div>
                    
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
                                {tool.name === 'Create Colab Notebook' && tool.result.colabUrl ? (
                                  <a 
                                    href={tool.result.colabUrl} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className="text-blue-600 hover:text-blue-800 underline"
                                  >
                                    Open Colab Notebook →
                                  </a>
                                ) : (
                                  JSON.stringify(tool.result).substring(0, 50) + '...'
                                )}
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
                placeholder="Ask questions or say 'run experiment' to use tools..."
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
