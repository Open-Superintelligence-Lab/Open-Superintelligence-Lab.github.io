#!/usr/bin/env node

import http from "node:http";
import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const HOST = process.env.CODEX_LAUNCH_HOST ?? "127.0.0.1";
const PORT = Number(process.env.CODEX_LAUNCH_PORT ?? "4511");
const REPO_DIR = "/Users/vukrosic/my-life/llm-research-kit-scaling";
const PROMPT_PATH = `${REPO_DIR}/autoresearch/prompts/auto-research.md`;
const LAUNCHER =
  process.env.CODEX_LAUNCHER ?? "/Users/vukrosic/.agents/skills/launch-codex-tmux/scripts/launch_codex.sh";
const SESSION_PREFIX = process.env.CODEX_LAUNCH_SESSION_PREFIX ?? "lab-launch-codex";

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function htmlPage(title, body) {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${escapeHtml(title)}</title>
    <style>
      :root {
        color-scheme: dark;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #1f1e1d;
        color: #faf9f6;
      }
      html, body {
        margin: 0;
        min-height: 100%;
        background: #1f1e1d;
        color: #faf9f6;
      }
      body {
        display: grid;
        place-items: center;
        padding: 32px;
      }
      main {
        max-width: 760px;
        width: 100%;
        border: 1px solid rgba(240, 238, 230, 0.12);
        border-radius: 20px;
        background: rgba(240, 238, 230, 0.03);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.25);
        padding: 28px;
      }
      h1 {
        margin: 0 0 12px;
        font-size: 28px;
        letter-spacing: -0.03em;
      }
      p {
        margin: 0 0 12px;
        line-height: 1.6;
        color: rgba(250, 249, 246, 0.82);
      }
      pre {
        margin: 16px 0 0;
        padding: 16px;
        overflow: auto;
        border-radius: 14px;
        border: 1px solid rgba(240, 238, 230, 0.12);
        background: rgba(31, 30, 29, 0.92);
        color: #d0f4ff;
        white-space: pre-wrap;
        word-break: break-word;
      }
      code {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      }
      a {
        color: #6ee7ff;
        text-decoration: none;
      }
      a:hover {
        text-decoration: underline;
      }
      .meta {
        margin-top: 16px;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: rgba(250, 249, 246, 0.45);
      }
    </style>
  </head>
  <body>
    <main>
      ${body}
    </main>
  </body>
</html>`;
}

function send(res, statusCode, contentType, body, extraHeaders = {}) {
  res.writeHead(statusCode, {
    "Content-Type": contentType,
    "Cache-Control": "no-store",
    ...extraHeaders,
  });
  res.end(body);
}

function addCorsHeaders(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "content-type");
}

function makeSessionName() {
  const stamp = new Date().toISOString().replace(/[-:.TZ]/g, "");
  const suffix = Math.random().toString(36).slice(2, 7);
  return `${SESSION_PREFIX}-${stamp}-${suffix}`;
}

async function readAutoResearchPrompt() {
  return readFile(PROMPT_PATH, "utf8");
}

async function launchCodex(prompt) {
  const session = makeSessionName();
  try {
    const { stdout, stderr } = await execFileAsync(LAUNCHER, [session, prompt], {
      cwd: REPO_DIR,
      maxBuffer: 10 * 1024 * 1024,
      timeout: 60_000,
    });

    return {
      ok: true,
      session,
      stdout: stdout.trim(),
      stderr: stderr.trim(),
    };
  } catch (error) {
    const stdout = typeof error?.stdout === "string" ? error.stdout.trim() : "";
    const stderr = typeof error?.stderr === "string" ? error.stderr.trim() : "";
    const message = error instanceof Error ? error.message : String(error);

    return {
      ok: false,
      session,
      stdout,
      stderr: stderr || message,
      message,
    };
  }
}

const server = http.createServer(async (req, res) => {
  addCorsHeaders(res);

  const hostHeader = req.headers.host ?? `${HOST}:${PORT}`;
  const url = new URL(req.url ?? "/", `http://${hostHeader}`);

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  if ((req.method === "GET" || req.method === "HEAD") && url.pathname === "/health") {
    send(
      res,
      200,
      "application/json; charset=utf-8",
      req.method === "HEAD" ? "" : JSON.stringify({ ok: true, port: PORT, host: HOST }, null, 2),
    );
    return;
  }

  if (req.method === "GET" && url.pathname === "/") {
    send(
      res,
      200,
      "text/html; charset=utf-8",
      htmlPage(
        "Codex Launcher",
        `
        <h1>Codex launcher service</h1>
        <p>POST to <code>/launch</code> from the single-button page in the lab site.</p>
        <p><strong>Prompt:</strong> <code>${escapeHtml(PROMPT_PATH)}</code></p>
        <p><strong>Working directory:</strong> <code>${escapeHtml(REPO_DIR)}</code></p>
        <p class="meta">Local host: <code>${escapeHtml(HOST)}:${PORT}</code></p>
      `,
      ),
    );
    return;
  }

  if (req.method === "POST" && url.pathname === "/launch") {
    let prompt;
    try {
      prompt = await readAutoResearchPrompt();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      send(
        res,
        500,
        "text/html; charset=utf-8",
        htmlPage(
          "Codex launch failed",
          `
            <h1>Codex launch failed</h1>
            <p>Could not read the auto-research prompt.</p>
            <p><strong>Prompt:</strong> <code>${escapeHtml(PROMPT_PATH)}</code></p>
            <p><strong>Error:</strong> ${escapeHtml(message)}</p>
          `,
        ),
      );
      return;
    }

    const result = await launchCodex(prompt);

    if (result.ok) {
      send(
        res,
        200,
        "text/html; charset=utf-8",
      htmlPage(
        "Codex launched",
        `
            <h1>Codex launched</h1>
            <p><strong>Session:</strong> <code>${escapeHtml(result.session)}</code></p>
            <p><strong>Status:</strong> tmux session is running and waiting for Codex.</p>
            <p><strong>Attach:</strong> <code>tmux attach -t ${escapeHtml(result.session)}</code></p>
            <p><strong>Prompt:</strong> <code>${escapeHtml(PROMPT_PATH)}</code></p>
            <p><strong>Working directory:</strong> <code>${escapeHtml(REPO_DIR)}</code></p>
            <p><strong>Output:</strong></p>
            <pre>${escapeHtml(result.stdout || "(no stdout)")}</pre>
            ${
              result.stderr
                ? `<p class="meta">stderr</p><pre>${escapeHtml(result.stderr)}</pre>`
                : ""
            }
            <p class="meta">Only the auto-research prompt file was sent to the interactive launch-codex-tmux wrapper.</p>
          `,
        ),
      );
      return;
    }

    send(
      res,
      500,
      "text/html; charset=utf-8",
      htmlPage(
        "Codex launch failed",
        `
          <h1>Codex launch failed</h1>
          <p><strong>Session:</strong> <code>${escapeHtml(result.session)}</code></p>
          <p><strong>Error:</strong> ${escapeHtml(result.message ?? "unknown error")}</p>
          <p><strong>Details:</strong></p>
          <pre>${escapeHtml(result.stderr || result.stdout || "no details")}</pre>
        `,
      ),
    );
    return;
  }

  send(
    res,
    404,
    "text/html; charset=utf-8",
    htmlPage("Not found", `<h1>Not found</h1><p>The requested path does not exist.</p>`),
  );
});

server.listen(PORT, HOST, () => {
  console.log(`Codex launcher listening on http://${HOST}:${PORT}`);
});
