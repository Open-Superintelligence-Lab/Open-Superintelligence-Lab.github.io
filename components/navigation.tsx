import Link from "next/link";

interface NavigationProps {
  currentPath?: string;
}

export function Navigation({ currentPath }: NavigationProps) {
  return (
    <header className="container mx-auto px-6 py-8">
      <nav className="flex justify-between items-center">
        <Link href="/" className="text-2xl font-bold hover:text-gray-400 transition-colors">
          Open Superintelligence Lab
        </Link>
        <div className="flex gap-6">
          <Link 
            href="/about" 
            className={currentPath === "/about" ? "text-gray-400" : "hover:text-gray-400 transition-colors"}
          >
            About
          </Link>
          <Link 
            href="/blog" 
            className={currentPath === "/blog" ? "text-gray-400" : "hover:text-gray-400 transition-colors"}
          >
            Blog
          </Link>
          <Link 
            href="/tutorials" 
            className={currentPath?.startsWith("/tutorials") ? "text-gray-400" : "hover:text-gray-400 transition-colors"}
          >
            Tutorials
          </Link>
          <Link 
            href="/projects" 
            className={currentPath === "/projects" ? "text-gray-400" : "hover:text-gray-400 transition-colors"}
          >
            Projects
          </Link>
          <a 
            href="https://github.com/Open-Superintelligence-Lab/blueberry-llm" 
            className="hover:text-gray-400 transition-colors" 
            target="_blank" 
            rel="noopener noreferrer"
          >
            Blueberry LLM
          </a>
          <a 
            href="https://github.com/open-superintelligence-lab" 
            className="hover:text-gray-400 transition-colors" 
            target="_blank" 
            rel="noopener noreferrer"
          >
            GitHub
          </a>
        </div>
      </nav>
    </header>
  );
}
