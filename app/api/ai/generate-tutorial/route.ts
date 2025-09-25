import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(request: NextRequest) {
  try {
    const { prompt, existingContent, category, difficulty } = await request.json();

    if (!prompt) {
      return NextResponse.json({ error: "Prompt is required" }, { status: 400 });
    }

    const systemPrompt = `You are an expert tutorial writer. Create comprehensive, well-structured tutorials in Markdown format.

Guidelines:
- Write clear, engaging content that's easy to follow
- Use proper Markdown formatting (headers, code blocks, lists, etc.)
- Include practical examples and code snippets when relevant
- Structure content with logical flow and progression
- Make it suitable for ${difficulty} level learners
- Focus on the ${category} domain
- Keep explanations clear and concise
- Include actionable steps and takeaways

${existingContent ? `Build upon this existing content:\n\n${existingContent}\n\n` : ""}

Generate a complete tutorial based on this request: ${prompt}`;

    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: prompt }
      ],
      max_tokens: 8192,
      temperature: 0.7,
    });

    const generatedContent = completion.choices[0]?.message?.content || "";

    // Extract title and description from the generated content
    const lines = generatedContent.split('\n');
    const title = lines.find(line => line.startsWith('# '))?.replace('# ', '') || 'Generated Tutorial';
    const description = lines.find(line => line.startsWith('## ') || line.startsWith('### '))?.replace(/^#+\s*/, '') || 'AI-generated tutorial content';

    // Extract tags from content (simple keyword extraction)
    const tagKeywords = ['tutorial', 'guide', 'how-to', 'step-by-step', 'beginner', 'advanced', 'example', 'code', 'implementation'];
    const contentWords = generatedContent.toLowerCase().split(/\s+/);
    const extractedTags = tagKeywords.filter(keyword => contentWords.includes(keyword)).slice(0, 5);

    return NextResponse.json({
      content: generatedContent,
      title,
      description,
      tags: extractedTags,
    });

  } catch (error) {
    console.error("Error generating tutorial content:", error);
    return NextResponse.json(
      { error: "Failed to generate tutorial content" },
      { status: 500 }
    );
  }
}
