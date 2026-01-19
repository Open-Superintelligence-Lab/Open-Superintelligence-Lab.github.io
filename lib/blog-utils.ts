import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

const BLOG_POSTS_PATH = path.join(process.cwd(), 'blog-posts');

export interface BlogPost {
    slug: string;
    title: string;
    date: string;
    description: string;
    youtubeId?: string;
    content: string;
}

export function getAllPosts(): BlogPost[] {
    if (!fs.existsSync(BLOG_POSTS_PATH)) {
        return [];
    }

    const files = fs.readdirSync(BLOG_POSTS_PATH);

    const posts = files
        .filter((file) => file.endsWith('.md'))
        .map((file) => {
            const filePath = path.join(BLOG_POSTS_PATH, file);
            const fileContent = fs.readFileSync(filePath, 'utf8');
            const { data, content } = matter(fileContent);

            return {
                slug: file.replace('.md', ''),
                title: data.title || 'Untitled',
                date: data.date || '',
                description: data.description || '',
                youtubeId: data.youtubeId,
                content,
            };
        });

    return posts.sort((a, b) => (new Date(b.date).getTime() - new Date(a.date).getTime()));
}

export function getPostBySlug(slug: string): BlogPost | null {
    const filePath = path.join(BLOG_POSTS_PATH, `${slug}.md`);

    if (!fs.existsSync(filePath)) {
        return null;
    }

    const fileContent = fs.readFileSync(filePath, 'utf8');
    const { data, content } = matter(fileContent);

    return {
        slug,
        title: data.title || 'Untitled',
        date: data.date || '',
        description: data.description || '',
        youtubeId: data.youtubeId,
        content,
    };
}
