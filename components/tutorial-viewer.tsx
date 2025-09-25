"use client";

import { useState, useEffect } from "react";
import { useMutation, useQuery } from "convex/react";
import { api } from "@/convex/_generated/api";
import { Id } from "@/convex/_generated/dataModel";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { 
  Heart, 
  Eye, 
  MessageSquare, 
  Send, 
  Bot, 
  User, 
  Plus,
  BookOpen,
  Clock,
  Tag,
  User as UserIcon
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

interface TutorialViewerProps {
  tutorialId: Id<"tutorials">;
}

export function TutorialViewer({ tutorialId }: TutorialViewerProps) {
  const [activeTab, setActiveTab] = useState("content");
  const [chatMessage, setChatMessage] = useState("");
  const [isChatOpen, setIsChatOpen] = useState(false);

  const tutorial = useQuery(api.tutorials.getTutorial, { id: tutorialId });
  const chatSessions = useQuery(api.tutorialChat.getTutorialChatSessions, { 
    tutorialId, 
    userId: "user-123" // TODO: Get from auth
  });
  const currentChatMessages = useQuery(
    api.tutorialChat.getTutorialChatMessages, 
    chatSessions?.[0] ? { sessionId: chatSessions[0]._id } : "skip"
  );

  const likeTutorial = useMutation(api.tutorials.likeTutorial);
  const incrementViewCount = useMutation(api.tutorials.incrementViewCount);
  const createChatSession = useMutation(api.tutorialChat.createTutorialChatSession);
  const addChatMessage = useMutation(api.tutorialChat.addTutorialChatMessage);

  // Increment view count when tutorial loads
  useEffect(() => {
    if (tutorial && tutorial.status === "published") {
      incrementViewCount({ id: tutorialId });
    }
  }, [tutorial, tutorialId, incrementViewCount]);

  const handleLike = async () => {
    try {
      await likeTutorial({ id: tutorialId });
    } catch (error) {
      console.error("Failed to like tutorial:", error);
    }
  };

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatMessage.trim()) return;

    try {
      // Create session if none exists
      let sessionId = chatSessions?.[0]?._id;
      if (!sessionId) {
        sessionId = await createChatSession({
          tutorialId,
          userId: "user-123", // TODO: Get from auth
          title: `Chat about ${tutorial?.title}`,
        });
      }

      // Add user message
      await addChatMessage({
        sessionId,
        role: "user",
        content: chatMessage,
      });

      // TODO: Add AI response
      await addChatMessage({
        sessionId,
        role: "assistant",
        content: "I'm here to help you understand this tutorial better. What specific questions do you have?",
      });

      setChatMessage("");
    } catch (error) {
      console.error("Failed to send message:", error);
    }
  };

  if (!tutorial) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <BookOpen className="w-12 h-12 mx-auto text-gray-400 mb-4" />
          <p className="text-gray-500">Loading tutorial...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6 text-white">
      {/* Tutorial Header */}
      <div className="mb-8">
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <h1 className="text-4xl font-bold mb-2">{tutorial.title}</h1>
            <p className="text-xl text-gray-300 mb-4">{tutorial.description}</p>
            
            <div className="flex items-center gap-4 text-sm text-gray-400 mb-4">
              <div className="flex items-center gap-1">
                <UserIcon className="w-4 h-4" />
                <span>By {tutorial.authorId}</span>
              </div>
              <div className="flex items-center gap-1">
                <Clock className="w-4 h-4" />
                <span>{tutorial.estimatedReadTime} min read</span>
              </div>
              <div className="flex items-center gap-1">
                <Eye className="w-4 h-4" />
                <span>{tutorial.views} views</span>
              </div>
            </div>

            <div className="flex items-center gap-2 mb-4">
              <Badge variant="outline">{tutorial.category}</Badge>
              <Badge variant="outline">{tutorial.difficulty}</Badge>
              {tutorial.tags.map((tag) => (
                <Badge key={tag} variant="secondary" className="flex items-center gap-1">
                  <Tag className="w-3 h-3" />
                  {tag}
                </Badge>
              ))}
            </div>
          </div>

          <div className="flex gap-2">
            <Button onClick={handleLike} variant="outline">
              <Heart className="w-4 h-4 mr-2" />
              {tutorial.likes}
            </Button>
            <Button onClick={() => setIsChatOpen(!isChatOpen)}>
              <MessageSquare className="w-4 h-4 mr-2" />
              Chat
            </Button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Content */}
        <div className="lg:col-span-2">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="content">Content</TabsTrigger>
              <TabsTrigger value="comments">Comments</TabsTrigger>
            </TabsList>

            <TabsContent value="content">
              <Card>
                <CardContent className="p-6">
                  <div className="prose max-w-none">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeHighlight]}
                    >
                      {tutorial.content}
                    </ReactMarkdown>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="comments">
              <Card>
                <CardHeader>
                  <CardTitle>Comments & Discussion</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center py-8 text-gray-500">
                    <MessageSquare className="w-12 h-12 mx-auto mb-4" />
                    <p>Comments feature coming soon!</p>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Chat Panel */}
          {isChatOpen && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bot className="w-5 h-5" />
                  Chat with Tutorial
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-64 mb-4">
                  <div className="space-y-3">
                    {currentChatMessages?.map((message) => (
                      <div
                        key={message._id}
                        className={`flex gap-2 ${
                          message.role === "user" ? "justify-end" : "justify-start"
                        }`}
                      >
                        <div
                          className={`max-w-[80%] p-3 rounded-lg ${
                            message.role === "user"
                              ? "bg-blue-500 text-white"
                              : "bg-gray-100 text-gray-900"
                          }`}
                        >
                          <p className="text-sm">{message.content}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
                
                <form onSubmit={handleChatSubmit} className="flex gap-2">
                  <Input
                    value={chatMessage}
                    onChange={(e) => setChatMessage(e.target.value)}
                    placeholder="Ask a question about this tutorial..."
                    className="flex-1"
                  />
                  <Button type="submit" size="sm">
                    <Send className="w-4 h-4" />
                  </Button>
                </form>
              </CardContent>
            </Card>
          )}

          {/* Tutorial Info */}
          <Card>
            <CardHeader>
              <CardTitle>Tutorial Info</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Status</span>
                <Badge variant={tutorial.status === "published" ? "default" : "secondary"}>
                  {tutorial.status}
                </Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Created</span>
                <span className="text-sm">
                  {new Date(tutorial.createdAt).toLocaleDateString()}
                </span>
              </div>
              {tutorial.publishedAt && (
                <div className="flex justify-between">
                  <span className="text-sm text-gray-500">Published</span>
                  <span className="text-sm">
                    {new Date(tutorial.publishedAt).toLocaleDateString()}
                  </span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Last Updated</span>
                <span className="text-sm">
                  {new Date(tutorial.updatedAt).toLocaleDateString()}
                </span>
              </div>
            </CardContent>
          </Card>

          {/* Related Tutorials */}
          <Card>
            <CardHeader>
              <CardTitle>Related Tutorials</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-4 text-gray-500">
                <p className="text-sm">Related tutorials coming soon!</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
