'use client';

import { useEffect, useState } from 'react';
import { MarkdownRenderer } from './markdown-renderer';

interface MarkdownPanelProps {
  /** Path relative to the research repo, e.g. "autoresearch/prompts/generate-ideas.md". Null = closed. */
  path: string | null;
  title: string;
  onClose: () => void;
}

/**
 * Generic slide-in Markdown viewer/editor. Decoupled from ideas/prompts —
 * give it any `.md` path under the research repo and it reads, previews, and
 * saves via /api/file. Tabs switch between rendered preview and raw editing.
 */
export function MarkdownPanel({ path, title, onClose }: MarkdownPanelProps) {
  const [tab, setTab] = useState<'preview' | 'edit'>('preview');
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    if (!path) return;
    let cancelled = false;
    setLoading(true);
    setMessage('');
    setTab('preview');
    fetch('/api/file/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'read', path }),
    })
      .then((r) => r.json())
      .then((d) => {
        if (cancelled) return;
        if (d.success) setContent(d.content ?? '');
        else setMessage(`✗ Failed to load: ${d.error ?? 'unknown error'}`);
      })
      .catch(() => !cancelled && setMessage('✗ Error loading file'))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [path]);

  const save = async () => {
    if (!path) return;
    setSaving(true);
    setMessage('');
    try {
      const response = await fetch('/api/file/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'save', path, content }),
      });
      const data = await response.json().catch(() => ({}));
      setMessage(data.success ? '✓ Saved' : `✗ Failed to save: ${data.error ?? 'unknown error'}`);
    } catch {
      setMessage('✗ Error saving file');
    } finally {
      setSaving(false);
    }
  };

  // Strip leading YAML frontmatter for the rendered preview (the raw Edit tab
  // keeps it so it stays editable).
  const previewContent = content.replace(/^---\n[\s\S]*?\n---\n?/, "");

  if (!path) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="relative flex h-full w-full max-w-2xl flex-col border-l border-white/10 bg-[#1f1e1d] shadow-2xl">
        {/* Header: title + tab switch + close */}
        <div className="flex items-center justify-between gap-4 border-b border-white/10 px-6 py-4">
          <h2 className="min-w-0 truncate font-mono text-sm text-amber-200/80">{title}</h2>
          <div className="flex items-center gap-2">
            <div className="flex rounded-full border border-white/10 p-0.5">
              {(['preview', 'edit'] as const).map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setTab(t)}
                  className={`rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] transition ${
                    tab === t
                      ? 'bg-amber-300/20 text-amber-100'
                      : 'text-[#faf9f6]/50 hover:text-[#faf9f6]'
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={onClose}
              className="text-xs uppercase tracking-[0.2em] text-[#faf9f6]/50 transition hover:text-[#faf9f6]"
            >
              Close
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-auto p-6">
          {loading ? (
            <p className="text-sm text-[#faf9f6]/40">Loading…</p>
          ) : tab === 'edit' ? (
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              spellCheck={false}
              className="h-full min-h-[60vh] w-full resize-none rounded-xl border border-white/10 bg-black/30 p-4 font-mono text-xs leading-relaxed text-[#faf9f6] focus:border-amber-300/40 focus:outline-none"
            />
          ) : (
            <MarkdownRenderer content={previewContent} />
          )}
        </div>

        {/* Footer: status + save (only meaningful in edit) */}
        <div className="flex items-center justify-between gap-4 border-t border-white/10 px-6 py-4">
          <span className="text-sm text-amber-300">{message}</span>
          <button
            type="button"
            onClick={save}
            disabled={saving || loading}
            className="rounded-full border border-amber-300/30 bg-amber-300/10 px-6 py-2.5 text-xs font-semibold uppercase tracking-[0.2em] text-amber-200 transition hover:border-amber-300/60 hover:bg-amber-300/20 hover:text-white focus:outline-none focus:ring-2 focus:ring-amber-300/40 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}
