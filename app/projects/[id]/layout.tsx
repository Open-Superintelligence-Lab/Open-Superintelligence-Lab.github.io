import { api } from '../../../convex/_generated/api';
import { fetchQuery } from 'convex/nextjs';

// Server component for static params generation
export async function generateStaticParams() {
  try {
    // Fetch all project IDs from Convex
    const projects = await fetchQuery(api.projects.list, {});
    
    return projects.map((project) => ({
      id: project._id,
    }));
  } catch (error) {
    console.error('Error fetching projects for static params:', error);
    // Return empty array if there's an error
    return [];
  }
}

export default function ProjectLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
