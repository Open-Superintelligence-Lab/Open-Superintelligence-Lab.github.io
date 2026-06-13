import fs from "fs";
import path from "path";

const PROPOSALS_PATH = path.join(process.cwd(), "proposals");

export type ProposalStatus =
  | "draft"
  | "approved"
  | "rejected"
  | "changes-requested";

export interface Proposal {
  slug: string;
  title: string;
  date: string;
  status: ProposalStatus;
  content: string;
}

function normalizeStatus(rawStatus: string | undefined): ProposalStatus {
  const normalized = (rawStatus || "").trim().toLowerCase();

  if (normalized.startsWith("approved")) {
    return "approved";
  }

  if (normalized.startsWith("rejected")) {
    return "rejected";
  }

  if (
    normalized.startsWith("changes-requested") ||
    normalized.startsWith("changes requested")
  ) {
    return "changes-requested";
  }

  return "draft";
}

function extractTitle(content: string, slug: string): string {
  const headingMatch = content.match(/^#\s+Proposal:\s*(.+)$/m) || content.match(/^#\s+(.+)$/m);

  if (headingMatch?.[1]) {
    return headingMatch[1].trim();
  }

  return slug
    .replace(/^\d{4}-\d{2}-\d{2}-/, "")
    .replace(/-/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function extractDate(content: string): string {
  const dateMatch = content.match(/^- \*\*Date:\*\*\s*(.+)$/m);
  return dateMatch?.[1]?.trim() || "";
}

function extractStatus(content: string): ProposalStatus {
  const statusMatch = content.match(/^- \*\*Status:\*\*\s*(.+)$/m);
  return normalizeStatus(statusMatch?.[1]);
}

function parseDateForSort(date: string): number {
  const parsed = Date.parse(date);
  return Number.isNaN(parsed) ? 0 : parsed;
}

export function getAllProposals(): Proposal[] {
  if (!fs.existsSync(PROPOSALS_PATH)) {
    return [];
  }

  const files = fs
    .readdirSync(PROPOSALS_PATH)
    .filter((file) => file.endsWith(".md"));

  const proposals = files.map((file) => {
    const filePath = path.join(PROPOSALS_PATH, file);
    const content = fs.readFileSync(filePath, "utf8").replace(/\r\n/g, "\n");
    const slug = file.replace(/\.md$/, "");

    return {
      slug,
      title: extractTitle(content, slug),
      date: extractDate(content),
      status: extractStatus(content),
      content,
    };
  });

  return proposals.sort((a, b) => parseDateForSort(b.date) - parseDateForSort(a.date));
}

export function getProposalBySlug(slug: string): Proposal | null {
  const filePath = path.join(PROPOSALS_PATH, `${slug}.md`);

  if (!fs.existsSync(filePath)) {
    return null;
  }

  const content = fs.readFileSync(filePath, "utf8").replace(/\r\n/g, "\n");

  return {
    slug,
    title: extractTitle(content, slug),
    date: extractDate(content),
    status: extractStatus(content),
    content,
  };
}
