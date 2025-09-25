"use client";

import { useState, useEffect } from "react";
import { useMutation, useQuery } from "convex/react";
import { api } from "@/convex/_generated/api";
import { Id } from "@/convex/_generated/dataModel";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { 
  Save, 
  Eye, 
  Send, 
  Bot, 
  User, 
  Plus, 
  X, 
  Upload,
  Archive,
  History,
  MessageSquare,
  Sparkles
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

interface TutorialEditorProps {
  tutorialId?: Id<"tutorials">;
  onSave?: (tutorialId: Id<"tutorials">) => void;
  onPublish?: (tutorialId: Id<"tutorials">) => void;
}

export function TutorialEditor({ tutorialId, onSave, onPublish }: TutorialEditorProps) {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [content, setContent] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [newTag, setNewTag] = useState("");
  const [category, setCategory] = useState("");
  const [difficulty, setDifficulty] = useState<"beginner" | "intermediate" | "advanced">("beginner");
  const [isPublic, setIsPublic] = useState(false);
  const [aiPrompt, setAiPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [activeTab, setActiveTab] = useState("edit");

  const createTutorial = useMutation(api.tutorials.createTutorial);
  const updateTutorial = useMutation(api.tutorials.updateTutorial);
  const publishTutorial = useMutation(api.tutorials.publishTutorial);
  const getTutorial = useQuery(api.tutorials.getTutorial, tutorialId ? { id: tutorialId } : "skip");

  // Load existing tutorial data
  useEffect(() => {
    if (getTutorial) {
      setTitle(getTutorial.title);
      setDescription(getTutorial.description);
      setContent(getTutorial.content);
      setTags(getTutorial.tags);
      setCategory(getTutorial.category);
      setDifficulty(getTutorial.difficulty);
      setIsPublic(getTutorial.isPublic);
    }
  }, [getTutorial]);

  const handleSave = async () => {
    try {
      let savedTutorialId = tutorialId;
      
      if (!tutorialId) {
        // Create new tutorial
        savedTutorialId = await createTutorial({
          title,
          description,
          content,
          authorId: "user-123", // TODO: Get from auth
          tags,
          category,
          difficulty,
          isPublic,
          aiGenerated: false,
        });
      } else {
        // Update existing tutorial
        await updateTutorial({
          id: tutorialId,
          title,
          description,
          content,
          tags,
          category,
          difficulty,
          isPublic,
          changeDescription: "Manual edit",
        });
      }

      onSave?.(savedTutorialId);
    } catch (error) {
      console.error("Failed to save tutorial:", error);
    }
  };

  const handlePublish = async () => {
    try {
      await handleSave();
      if (tutorialId) {
        await publishTutorial({ id: tutorialId });
        onPublish?.(tutorialId);
      }
    } catch (error) {
      console.error("Failed to publish tutorial:", error);
    }
  };

  const handleAiGenerate = async () => {
    if (!aiPrompt.trim()) return;

    setIsGenerating(true);
    try {
      // TODO: Integrate with AI service
      const response = await fetch("/api/ai/generate-tutorial", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: aiPrompt,
          existingContent: content,
          category,
          difficulty,
        }),
      });

      const data = await response.json();
      
      if (data.content) {
        setContent(data.content);
        if (data.title) setTitle(data.title);
        if (data.description) setDescription(data.description);
        if (data.tags) setTags(data.tags);
      }
    } catch (error) {
      console.error("Failed to generate content:", error);
    } finally {
      setIsGenerating(false);
    }
  };

  const addTag = () => {
    if (newTag.trim() && !tags.includes(newTag.trim())) {
      setTags([...tags, newTag.trim()]);
      setNewTag("");
    }
  };

  const removeTag = (tagToRemove: string) => {
    setTags(tags.filter(tag => tag !== tagToRemove));
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6 text-white">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">
          {tutorialId ? "Edit Tutorial" : "Create New Tutorial"}
        </h1>
        <div className="flex gap-2">
          <Button onClick={handleSave} variant="outline">
            <Save className="w-4 h-4 mr-2" />
            Save Draft
          </Button>
          <Button onClick={handlePublish}>
            <Upload className="w-4 h-4 mr-2" />
            Publish
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="edit">Edit</TabsTrigger>
          <TabsTrigger value="preview">Preview</TabsTrigger>
          <TabsTrigger value="ai">AI Assistant</TabsTrigger>
        </TabsList>

        <TabsContent value="edit" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Basic Information</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium">Title</label>
                <Input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Enter tutorial title..."
                />
              </div>
              
              <div>
                <label className="text-sm font-medium">Description</label>
                <Textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Brief description of the tutorial..."
                  rows={3}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium">Category</label>
                  <Input
                    value={category}
                    onChange={(e) => setCategory(e.target.value)}
                    placeholder="e.g., Programming, Design, Business"
                  />
                </div>
                
                <div>
                  <label className="text-sm font-medium">Difficulty</label>
                  <Select value={difficulty} onValueChange={(value: any) => setDifficulty(value)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="beginner">Beginner</SelectItem>
                      <SelectItem value="intermediate">Intermediate</SelectItem>
                      <SelectItem value="advanced">Advanced</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div>
                <label className="text-sm font-medium">Tags</label>
                <div className="flex gap-2 mb-2">
                  <Input
                    value={newTag}
                    onChange={(e) => setNewTag(e.target.value)}
                    placeholder="Add a tag..."
                    onKeyPress={(e) => e.key === "Enter" && addTag()}
                  />
                  <Button onClick={addTag} size="sm">
                    <Plus className="w-4 h-4" />
                  </Button>
                </div>
                <div className="flex flex-wrap gap-2">
                  {tags.map((tag) => (
                    <Badge key={tag} variant="secondary" className="flex items-center gap-1">
                      {tag}
                      <X 
                        className="w-3 h-3 cursor-pointer" 
                        onClick={() => removeTag(tag)}
                      />
                    </Badge>
                  ))}
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="isPublic"
                  checked={isPublic}
                  onChange={(e) => setIsPublic(e.target.checked)}
                />
                <label htmlFor="isPublic" className="text-sm font-medium">
                  Make this tutorial public
                </label>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Content</CardTitle>
            </CardHeader>
            <CardContent>
              <Textarea
                value={content}
                onChange={(e) => setContent(e.target.value)}
                placeholder="Write your tutorial content in Markdown..."
                rows={20}
                className="font-mono"
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="preview">
          <Card>
            <CardHeader>
              <CardTitle>Preview</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[600px]">
                <div className="prose max-w-none">
                  <h1>{title || "Untitled Tutorial"}</h1>
                  <p className="text-gray-600">{description}</p>
                  <div className="flex gap-2 mb-4">
                    <Badge variant="outline">{category}</Badge>
                    <Badge variant="outline">{difficulty}</Badge>
                    {tags.map((tag) => (
                      <Badge key={tag} variant="secondary">{tag}</Badge>
                    ))}
                  </div>
                  <Separator className="my-4" />
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeHighlight]}
                  >
                    {content || "No content yet..."}
                  </ReactMarkdown>
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="ai" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="w-5 h-5" />
                AI Content Generator
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium">Describe what you want to create</label>
                <Textarea
                  value={aiPrompt}
                  onChange={(e) => setAiPrompt(e.target.value)}
                  placeholder="e.g., 'Create a tutorial about building a React component with TypeScript'"
                  rows={4}
                />
              </div>
              
              <Button 
                onClick={handleAiGenerate} 
                disabled={isGenerating || !aiPrompt.trim()}
                className="w-full"
              >
                <Bot className="w-4 h-4 mr-2" />
                {isGenerating ? "Generating..." : "Generate Content"}
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>AI Suggestions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <Button variant="outline" className="w-full justify-start">
                  <Sparkles className="w-4 h-4 mr-2" />
                  Improve existing content
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Sparkles className="w-4 h-4 mr-2" />
                  Add examples and code snippets
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Sparkles className="w-4 h-4 mr-2" />
                  Generate exercises and quizzes
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Sparkles className="w-4 h-4 mr-2" />
                  Optimize for SEO
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
