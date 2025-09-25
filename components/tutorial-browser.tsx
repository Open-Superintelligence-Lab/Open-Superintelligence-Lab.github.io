"use client";

import { useState } from "react";
import { useQuery } from "convex/react";
import { api } from "@/convex/_generated/api";
import { Id } from "@/convex/_generated/dataModel";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Search, 
  Plus, 
  Eye, 
  Heart, 
  Clock, 
  Tag,
  Filter,
  BookOpen,
  TrendingUp,
  Star
} from "lucide-react";
import Link from "next/link";

export function TutorialBrowser() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("");
  const [selectedDifficulty, setSelectedDifficulty] = useState("");
  const [activeTab, setActiveTab] = useState("all");

  const publishedTutorials = useQuery(api.tutorials.getPublishedTutorials, {
    category: selectedCategory || undefined,
    difficulty: selectedDifficulty as any || undefined,
  });

  const searchResults = useQuery(
    api.tutorials.searchTutorials,
    searchQuery ? {
      query: searchQuery,
      category: selectedCategory || undefined,
      difficulty: selectedDifficulty as any || undefined,
    } : "skip"
  );

  const tutorials = searchQuery ? searchResults : publishedTutorials;

  const categories = [
    "Programming",
    "Design",
    "Business",
    "Marketing",
    "Data Science",
    "Machine Learning",
    "Web Development",
    "Mobile Development",
    "DevOps",
    "Other"
  ];

  const difficulties = [
    { value: "beginner", label: "Beginner" },
    { value: "intermediate", label: "Intermediate" },
    { value: "advanced", label: "Advanced" }
  ];

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case "beginner": return "bg-green-100 text-green-800";
      case "intermediate": return "bg-yellow-100 text-yellow-800";
      case "advanced": return "bg-red-100 text-red-800";
      default: return "bg-gray-100 text-gray-800";
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6 text-white">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold mb-2">Tutorial Library</h1>
          <p className="text-gray-300">
            Discover and learn from AI-generated tutorials and guides
          </p>
        </div>
        <Link href="/tutorials/create">
          <Button>
            <Plus className="w-4 h-4 mr-2" />
            Create Tutorial
          </Button>
        </Link>
      </div>

      {/* Search and Filters */}
      <Card className="mb-6">
        <CardContent className="p-6">
          <div className="flex gap-4 mb-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <Input
                placeholder="Search tutorials..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            <Select value={selectedCategory} onValueChange={setSelectedCategory}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Category" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">All Categories</SelectItem>
                {categories.map((category) => (
                  <SelectItem key={category} value={category}>
                    {category}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={selectedDifficulty} onValueChange={setSelectedDifficulty}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Difficulty" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">All Levels</SelectItem>
                {difficulties.map((difficulty) => (
                  <SelectItem key={difficulty.value} value={difficulty.value}>
                    {difficulty.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList>
              <TabsTrigger value="all">All Tutorials</TabsTrigger>
              <TabsTrigger value="trending">Trending</TabsTrigger>
              <TabsTrigger value="recent">Recent</TabsTrigger>
              <TabsTrigger value="popular">Most Popular</TabsTrigger>
            </TabsList>
          </Tabs>
        </CardContent>
      </Card>

      {/* Results */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {tutorials?.map((tutorial) => (
          <Card key={tutorial._id} className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="flex items-start justify-between mb-2">
                <Badge variant="outline" className="text-xs">
                  {tutorial.category}
                </Badge>
                <Badge 
                  variant="secondary" 
                  className={`text-xs ${getDifficultyColor(tutorial.difficulty)}`}
                >
                  {tutorial.difficulty}
                </Badge>
              </div>
              <CardTitle className="line-clamp-2 text-lg">
                <Link 
                  href={`/tutorials/${tutorial._id}`}
                  className="hover:text-blue-600 transition-colors"
                >
                  {tutorial.title}
                </Link>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-300 text-sm mb-4 line-clamp-3">
                {tutorial.description}
              </p>
              
              <div className="flex flex-wrap gap-1 mb-4">
                {tutorial.tags.slice(0, 3).map((tag) => (
                  <Badge key={tag} variant="secondary" className="text-xs">
                    <Tag className="w-3 h-3 mr-1" />
                    {tag}
                  </Badge>
                ))}
                {tutorial.tags.length > 3 && (
                  <Badge variant="secondary" className="text-xs">
                    +{tutorial.tags.length - 3} more
                  </Badge>
                )}
              </div>

              <div className="flex items-center justify-between text-sm text-gray-400">
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1">
                    <Eye className="w-4 h-4" />
                    <span>{tutorial.views}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Heart className="w-4 h-4" />
                    <span>{tutorial.likes}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="w-4 h-4" />
                    <span>{tutorial.estimatedReadTime}m</span>
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  {tutorial.aiGenerated && (
                    <Star className="w-4 h-4 text-yellow-500" />
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Empty State */}
      {tutorials?.length === 0 && (
        <div className="text-center py-12">
          <BookOpen className="w-16 h-16 mx-auto text-gray-400 mb-4" />
          <h3 className="text-xl font-semibold mb-2">No tutorials found</h3>
          <p className="text-gray-300 mb-4">
            {searchQuery 
              ? "Try adjusting your search terms or filters"
              : "Be the first to create a tutorial!"
            }
          </p>
          <Link href="/tutorials/create">
            <Button>
              <Plus className="w-4 h-4 mr-2" />
              Create Tutorial
            </Button>
          </Link>
        </div>
      )}

      {/* Stats */}
      {tutorials && tutorials.length > 0 && (
        <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card>
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">
                {tutorials.length}
              </div>
              <div className="text-gray-300">Total Tutorials</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-green-600 mb-2">
                {tutorials.reduce((sum, t) => sum + t.views, 0)}
              </div>
              <div className="text-gray-300">Total Views</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-red-600 mb-2">
                {tutorials.reduce((sum, t) => sum + t.likes, 0)}
              </div>
              <div className="text-gray-300">Total Likes</div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
