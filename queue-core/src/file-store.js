/* eslint-disable @typescript-eslint/no-require-imports */
"use strict";

const fs = require("fs");
const path = require("path");

const {
  CLAIM_STATUSES,
  JOB_KINDS,
  JOB_STATUSES,
  REVIEW_DECISIONS,
  TARGET_TYPES,
} = require("./constants");

const DEFAULT_ROOT = path.resolve(__dirname, "..", "data");
const DEFAULT_LEASE_MINUTES = 120;

function nowIso() {
  return new Date().toISOString();
}

function addMinutes(date, minutes) {
  return new Date(date.getTime() + minutes * 60 * 1000);
}

function makeId(prefix) {
  const compact = nowIso().replace(/[-:.TZ]/g, "").slice(0, 14);
  const random = Math.random().toString(36).slice(2, 8);
  return `${prefix}_${compact}_${random}`;
}

function assertPlainId(value, label) {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${label} is required`);
  }
  if (value.includes("/") || value.includes("\\") || value.includes("..")) {
    throw new Error(`${label} must be a plain id, got ${value}`);
  }
  return value.trim();
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function readJson(filePath, fallback = null) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch (error) {
    if (error && error.code === "ENOENT") return fallback;
    throw error;
  }
}

function writeJson(filePath, value) {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function appendJsonl(filePath, value) {
  ensureDir(path.dirname(filePath));
  fs.appendFileSync(filePath, `${JSON.stringify(value)}\n`, "utf8");
}

function listJsonFiles(dirPath) {
  try {
    return fs
      .readdirSync(dirPath)
      .filter((file) => file.endsWith(".json"))
      .sort()
      .map((file) => path.join(dirPath, file));
  } catch (error) {
    if (error && error.code === "ENOENT") return [];
    throw error;
  }
}

class FileQueueStore {
  constructor(options = {}) {
    this.root = path.resolve(options.root || DEFAULT_ROOT);
    this.jobsDir = path.join(this.root, "jobs");
    this.claimsDir = path.join(this.root, "claims");
    this.resultsDir = path.join(this.root, "results");
    this.eventsPath = path.join(this.root, "events.jsonl");
    this.reviewsPath = path.join(this.root, "reviews.jsonl");
  }

  init() {
    ensureDir(this.jobsDir);
    ensureDir(this.claimsDir);
    ensureDir(this.resultsDir);
    if (!fs.existsSync(this.eventsPath)) fs.writeFileSync(this.eventsPath, "", "utf8");
    if (!fs.existsSync(this.reviewsPath)) fs.writeFileSync(this.reviewsPath, "", "utf8");
    return { root: this.root };
  }

  jobPath(id) {
    return path.join(this.jobsDir, `${assertPlainId(id, "job id")}.json`);
  }

  claimPath(id) {
    return path.join(this.claimsDir, `${assertPlainId(id, "lease id")}.json`);
  }

  resultPath(jobId, runId) {
    return path.join(this.resultsDir, assertPlainId(jobId, "job id"), assertPlainId(runId, "run id"), "result.json");
  }

  resultRevisionsDir(jobId, runId) {
    return path.join(this.resultsDir, assertPlainId(jobId, "job id"), assertPlainId(runId, "run id"), "revisions");
  }

  emit(type, payload = {}) {
    const event = {
      id: makeId("evt"),
      type,
      created_at: nowIso(),
      ...payload,
    };
    appendJsonl(this.eventsPath, event);
    return event;
  }

  createJob(payload) {
    this.init();
    const current = nowIso();
    const job = {
      ...payload,
      id: assertPlainId(payload.id, "job id"),
      kind: payload.kind || "experiment",
      status: payload.status || "draft",
      priority: Number.isInteger(payload.priority) ? payload.priority : 3,
      owner: payload.owner || "unknown",
      created_at: payload.created_at || current,
      updated_at: payload.updated_at || current,
      input: payload.input && typeof payload.input === "object" ? payload.input : { brief: "" },
      expected_output:
        payload.expected_output && typeof payload.expected_output === "object" ? payload.expected_output : { summary: "" },
    };
    validateJob(job);
    if (fs.existsSync(this.jobPath(job.id))) {
      throw new Error(`job already exists: ${job.id}`);
    }
    writeJson(this.jobPath(job.id), job);
    this.emit("job.created", { job_id: job.id, payload: { status: job.status } });
    return job;
  }

  getJob(id) {
    const job = readJson(this.jobPath(id));
    if (!job) throw new Error(`job not found: ${id}`);
    return job;
  }

  listJobs(filter = {}) {
    this.init();
    this.sweepExpiredClaims();
    const jobs = listJsonFiles(this.jobsDir).map((file) => readJson(file));
    return jobs
      .filter((job) => !filter.status || job.status === filter.status)
      .filter((job) => !filter.kind || job.kind === filter.kind)
      .sort((a, b) => (b.priority - a.priority) || a.id.localeCompare(b.id));
  }

  updateJob(id, patch) {
    const job = this.getJob(id);
    const next = {
      ...job,
      ...patch,
      id: job.id,
      updated_at: nowIso(),
    };
    validateJob(next);
    writeJson(this.jobPath(id), next);
    this.emit("job.updated", { job_id: id, payload: { patch } });
    return next;
  }

  claimJob(id, worker, options = {}) {
    this.init();
    this.sweepExpiredClaims();
    const workerId = assertPlainId(worker.worker_id || worker.id || worker, "worker id");
    const job = this.getJob(id);
    if (job.status !== "ready") {
      throw new Error(`job ${id} is not ready; current status is ${job.status}`);
    }
    const claimedAt = new Date();
    const ttl = Number(options.ttl_minutes || options.ttlMinutes || DEFAULT_LEASE_MINUTES);
    const claim = {
      job_id: job.id,
      lease_id: makeId("lease"),
      worker_id: workerId,
      claimed_at: claimedAt.toISOString(),
      expires_at: addMinutes(claimedAt, ttl).toISOString(),
      heartbeat_at: claimedAt.toISOString(),
      status: "active",
    };
    validateClaim(claim);
    writeJson(this.claimPath(claim.lease_id), claim);
    this.updateJob(job.id, { status: "claimed" });
    this.emit("job.claimed", { job_id: job.id, lease_id: claim.lease_id, payload: { worker_id: workerId } });
    return claim;
  }

  getClaim(leaseId) {
    const claim = readJson(this.claimPath(leaseId));
    if (!claim) throw new Error(`claim not found: ${leaseId}`);
    return claim;
  }

  heartbeatClaim(leaseId, options = {}) {
    const claim = this.getClaim(leaseId);
    if (claim.status !== "active") {
      throw new Error(`claim ${leaseId} is not active; current status is ${claim.status}`);
    }
    const ttl = Number(options.ttl_minutes || options.ttlMinutes || DEFAULT_LEASE_MINUTES);
    const heartbeatAt = new Date();
    const next = {
      ...claim,
      heartbeat_at: heartbeatAt.toISOString(),
      expires_at: addMinutes(heartbeatAt, ttl).toISOString(),
    };
    validateClaim(next);
    writeJson(this.claimPath(leaseId), next);
    this.updateJob(claim.job_id, { status: "running" });
    this.emit("claim.heartbeat", { job_id: claim.job_id, lease_id: leaseId });
    return next;
  }

  releaseClaim(leaseId, reason = "") {
    const claim = this.getClaim(leaseId);
    if (claim.status !== "active") {
      throw new Error(`claim ${leaseId} is not active; current status is ${claim.status}`);
    }
    const next = {
      ...claim,
      status: "released",
      released_at: nowIso(),
      reason,
    };
    validateClaim(next);
    writeJson(this.claimPath(leaseId), next);
    this.updateJob(claim.job_id, { status: "ready" });
    return this.emit("claim.released", { job_id: claim.job_id, lease_id: leaseId, payload: { reason } });
  }

  submitResult(leaseId, payload) {
    this.init();
    const claim = this.getClaim(leaseId);
    if (claim.status !== "active") {
      throw new Error(`claim ${leaseId} is not active; current status is ${claim.status}`);
    }
    const job = this.getJob(claim.job_id);
    const finishedAt = payload.finished_at || nowIso();
    const runId = payload.run_id || makeRunId(claim.worker_id, finishedAt);
    const result = {
      ...payload,
      job_id: job.id,
      lease_id: claim.lease_id,
      worker_id: payload.worker_id || claim.worker_id,
      finished_at: finishedAt,
      exit_status: payload.exit_status || "unknown",
      summary: payload.summary || "",
      metrics: payload.metrics && typeof payload.metrics === "object" ? payload.metrics : {},
      artifacts: Array.isArray(payload.artifacts) ? payload.artifacts : [],
    };
    validateResult(result);
    this.writeResultVersion(job.id, runId, result, { preserveHistory: true });
    const nextClaim = {
      ...claim,
      status: "completed",
      completed_at: nowIso(),
    };
    writeJson(this.claimPath(claim.lease_id), nextClaim);
    this.updateJob(job.id, { status: "submitted" });
    this.emit("result.submitted", { job_id: job.id, lease_id: claim.lease_id, payload: { run_id: runId } });
    return { run_id: runId, result };
  }

  reviseResult(jobId, runId, nextResult) {
    this.init();
    const current = this.getResult(jobId, runId);
    const merged = {
      ...current,
      ...nextResult,
      job_id: jobId,
      run_id: runId,
      updated_at: nowIso(),
    };
    validateResult(merged);
    this.writeResultVersion(jobId, runId, merged, { preserveHistory: true });
    this.emit("result.revised", {
      job_id: jobId,
      payload: { run_id: runId },
    });
    return merged;
  }

  writeResultVersion(jobId, runId, result, options = {}) {
    const currentPath = this.resultPath(jobId, runId);
    if (options.preserveHistory && fs.existsSync(currentPath)) {
      const revisionDir = this.resultRevisionsDir(jobId, runId);
      ensureDir(revisionDir);
      const revisionId = makeId("rev");
      const revisionPath = path.join(revisionDir, `${revisionId}.json`);
      fs.copyFileSync(currentPath, revisionPath);
    }
    writeJson(currentPath, result);
  }

  listResults() {
    this.init();
    const results = [];
    for (const jobDir of listDirs(this.resultsDir)) {
      for (const runDir of listDirs(jobDir)) {
        const result = readJson(path.join(runDir, "result.json"));
        if (result) {
          results.push({ run_id: path.basename(runDir), ...result });
        }
      }
    }
    return results.sort((a, b) => String(b.finished_at || "").localeCompare(String(a.finished_at || "")));
  }

  reviewTarget(targetType, targetId, payload) {
    this.init();
    if (!TARGET_TYPES.has(targetType)) {
      throw new Error(`invalid target type: ${targetType}`);
    }
    let jobId = targetId;
    if (targetType === "job") {
      this.getJob(targetId);
    } else {
      const parts = String(targetId).split("/");
      if (parts.length !== 2) {
        throw new Error("result target id must be <job_id>/<run_id>");
      }
      jobId = parts[0];
      this.getResult(parts[0], parts[1]);
    }
    const review = {
      target_type: targetType,
      target_id: targetId,
      reviewer_id: payload.reviewer_id || payload.reviewer || "unknown",
      decision: payload.decision || "keep",
      created_at: payload.created_at || nowIso(),
      note: payload.note || "",
    };
    validateReview(review);
    appendJsonl(this.reviewsPath, review);
    if (jobId) {
      if (review.decision === "approve") this.updateJob(jobId, { status: "accepted" });
      if (review.decision === "request_changes") this.updateJob(jobId, { status: "ready" });
      if (review.decision === "reject") this.updateJob(jobId, { status: "rejected" });
    }
    this.emit("review.created", {
      job_id: jobId,
      target_type: targetType,
      target_id: targetId,
      payload: { decision: review.decision },
    });
    return review;
  }

  getResult(jobId, runId) {
    const result = readJson(this.resultPath(jobId, runId));
    if (!result) throw new Error(`result not found: ${jobId}/${runId}`);
    return result;
  }

  exportSnapshot(filter = {}) {
    const jobs = this.listJobs(filter).map(publicJob);
    const results = this.listResults().map(publicResult);
    return {
      generated_at: nowIso(),
      jobs,
      results,
    };
  }

  writeSnapshot(filePath, filter = {}) {
    const snapshot = this.exportSnapshot(filter);
    writeJson(path.resolve(filePath), snapshot);
    return snapshot;
  }

  migrate(options) {
    this.init();
    const sourceQueue = options.source_queue || options.sourceQueue;
    const sourceResults = options.source_results || options.sourceResults;
    const imported = { jobs: 0, results: 0, skipped_jobs: 0, skipped_results: 0 };
    if (sourceQueue) {
      for (const file of listFilesRecursive(path.resolve(sourceQueue)).filter((name) => name.endsWith(".yaml") || name.endsWith(".yml"))) {
        const parsed = parseSimpleYaml(fs.readFileSync(file, "utf8"));
        const id = parsed.id || path.basename(file).replace(/\.(queued|done|claimed|ready)?\.ya?ml$/, "");
        const status = statusFromLegacy(parsed.status || parsed.state || path.basename(file));
        const job = {
          id,
          kind: "experiment",
          title: parsed.title || id,
          status,
          priority: Number.isInteger(parsed.priority) ? parsed.priority : 3,
          owner: parsed.owner || "vukrosic",
          created_at: parsed.created_at || nowIso(),
          updated_at: parsed.updated_at || nowIso(),
          input: {
            brief: portableText(parsed.plain || parsed.idea || parsed.command || parsed.title || id),
            idea: portableText(parsed.idea),
            repo: portableText(parsed.repo),
            commit: portableText(parsed.commit),
            command: portableText(parsed.command),
          },
          expected_output: {
            summary: "Structured result with metrics and artifacts.",
            artifacts: ["result.json", "metrics.json"],
          },
          constraints: {
            gpu_vram_gb: nestedNumber(parsed, ["requires", "gpu_vram_gb"]),
            max_hours: nestedNumber(parsed, ["requires", "hours"]),
          },
          public_summary: portableText(parsed.plain || ""),
          legacy: {
            source_file: file,
          },
        };
        if (fs.existsSync(this.jobPath(job.id))) {
          imported.skipped_jobs += 1;
          continue;
        }
        this.createJob(job);
        imported.jobs += 1;
      }
    }
    if (sourceResults) {
      for (const file of listFilesRecursive(path.resolve(sourceResults)).filter((name) => path.basename(name) === "result.json")) {
        const raw = readJson(file);
        if (!raw) continue;
        const jobId = raw.job_id || raw.spec_id || path.basename(path.dirname(path.dirname(file)));
        const runId = path.basename(path.dirname(file));
        const outputPath = this.resultPath(jobId, runId);
        if (fs.existsSync(outputPath)) {
          imported.skipped_results += 1;
          continue;
        }
        const result = {
          ...raw,
          job_id: jobId,
          lease_id: raw.lease_id || null,
          worker_id: raw.worker_id || raw.worker || "unknown",
          finished_at: raw.finished_at || raw.finished || nowIso(),
          exit_status: raw.exit_status || "unknown",
          summary: raw.summary || "",
          metrics: raw.metrics && typeof raw.metrics === "object" ? raw.metrics : {},
          artifacts: Array.isArray(raw.artifacts) ? raw.artifacts : [],
          legacy: {
            source_file: file,
          },
        };
        validateResult(result);
        writeJson(outputPath, result);
        imported.results += 1;
      }
    }
    this.emit("queue.migrated", { payload: imported });
    return imported;
  }

  sweepExpiredClaims() {
    const now = Date.now();
    for (const file of listJsonFiles(this.claimsDir)) {
      const claim = readJson(file);
      if (!claim || claim.status !== "active") continue;
      if (Date.parse(claim.expires_at) > now) continue;
      const next = { ...claim, status: "expired", reason: "lease expired" };
      writeJson(file, next);
      try {
        const job = this.getJob(claim.job_id);
        if (job.status === "claimed" || job.status === "running") {
          this.updateJob(job.id, { status: "ready" });
        }
      } catch {
        // Keep sweeping even if a stale claim points at a missing job.
      }
      this.emit("claim.expired", { job_id: claim.job_id, lease_id: claim.lease_id });
    }
  }
}

function validateJob(job) {
  if (!JOB_KINDS.has(job.kind)) throw new Error(`invalid job kind: ${job.kind}`);
  if (!JOB_STATUSES.has(job.status)) throw new Error(`invalid job status: ${job.status}`);
  assertPlainId(job.id, "job id");
  if (!job.title) throw new Error("job title is required");
  if (!job.owner) throw new Error("job owner is required");
  if (!job.input || typeof job.input !== "object" || typeof job.input.brief !== "string") {
    throw new Error("job input.brief is required");
  }
  if (!job.expected_output || typeof job.expected_output !== "object" || typeof job.expected_output.summary !== "string") {
    throw new Error("job expected_output.summary is required");
  }
}

function validateClaim(claim) {
  assertPlainId(claim.job_id, "job id");
  assertPlainId(claim.lease_id, "lease id");
  assertPlainId(claim.worker_id, "worker id");
  if (!CLAIM_STATUSES.has(claim.status)) throw new Error(`invalid claim status: ${claim.status}`);
}

function validateResult(result) {
  assertPlainId(result.job_id, "job id");
  if (result.lease_id !== null && result.lease_id !== undefined) assertPlainId(result.lease_id, "lease id");
  assertPlainId(result.worker_id, "worker id");
  if (!result.finished_at) throw new Error("result finished_at is required");
  if (!result.exit_status) throw new Error("result exit_status is required");
  if (!result.metrics || typeof result.metrics !== "object" || Array.isArray(result.metrics)) {
    throw new Error("result metrics object is required");
  }
  if (!Array.isArray(result.artifacts)) throw new Error("result artifacts array is required");
}

function validateReview(review) {
  if (!TARGET_TYPES.has(review.target_type)) throw new Error(`invalid review target type: ${review.target_type}`);
  if (!REVIEW_DECISIONS.has(review.decision)) throw new Error(`invalid review decision: ${review.decision}`);
  if (!review.target_id) throw new Error("review target_id is required");
  assertPlainId(review.reviewer_id, "reviewer id");
}

function makeRunId(workerId, finishedAt) {
  const stamp = String(finishedAt).replace(/[-:.TZ]/g, "").slice(0, 14);
  return `${assertPlainId(workerId, "worker id")}-${stamp}`;
}

function publicStatus(status) {
  if (status === "ready" || status === "draft") return "queued";
  if (status === "claimed" || status === "running" || status === "blocked") return "claimed";
  return "done";
}

function publicJob(job) {
  return {
    id: job.id,
    kind: job.kind,
    title: portableText(job.title),
    status: publicStatus(job.status),
    priority: job.priority,
    summary: portableText(job.public_summary || (job.input && job.input.brief) || ""),
    constraints: {
      gpu_vram_gb: job.constraints ? job.constraints.gpu_vram_gb ?? null : null,
      max_hours: job.constraints ? job.constraints.max_hours ?? null : null,
    },
  };
}

function publicResult(result) {
  return {
    job_id: result.job_id,
    run_id: result.run_id,
    finished_at: result.finished_at,
    exit_status: result.exit_status,
    summary: portableText(result.summary || ""),
    metrics: result.metrics || {},
  };
}

function portableText(value) {
  return String(value || "")
    .replace(/\u2013|\u2014/g, "-")
    .replace(/\u2018|\u2019/g, "'")
    .replace(/\u201c|\u201d/g, '"');
}

function listDirs(dirPath) {
  try {
    return fs
      .readdirSync(dirPath, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => path.join(dirPath, entry.name))
      .sort();
  } catch (error) {
    if (error && error.code === "ENOENT") return [];
    throw error;
  }
}

function listFilesRecursive(root) {
  try {
    const output = [];
    for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
      const fullPath = path.join(root, entry.name);
      if (entry.isDirectory()) output.push(...listFilesRecursive(fullPath));
      if (entry.isFile()) output.push(fullPath);
    }
    return output;
  } catch (error) {
    if (error && error.code === "ENOENT") return [];
    throw error;
  }
}

function parseSimpleYaml(raw) {
  const lines = raw.split(/\r?\n/);
  const root = {};
  const stack = [{ indent: -1, value: root }];
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (!line.trim() || line.trimStart().startsWith("#")) continue;
    const indent = line.match(/^\s*/)[0].length;
    const trimmed = line.trim();
    while (stack.length > 1 && indent <= stack[stack.length - 1].indent) stack.pop();
    const parent = stack[stack.length - 1].value;
    if (trimmed.startsWith("- ")) {
      if (Array.isArray(parent)) parent.push(parseScalar(trimmed.slice(2).trim()));
      continue;
    }
    const match = trimmed.match(/^([^:]+):\s*(.*)$/);
    if (!match) continue;
    const key = match[1].trim();
    const value = match[2].trim();
    if (value === "") {
      const child = nextMeaningfulLineIsList(lines, index, indent) ? [] : {};
      parent[key] = child;
      stack.push({ indent, value: child });
    } else {
      parent[key] = parseScalar(value);
    }
  }
  return root;
}

function nextMeaningfulLineIsList(lines, index, parentIndent) {
  for (let nextIndex = index + 1; nextIndex < lines.length; nextIndex += 1) {
    const line = lines[nextIndex];
    if (!line.trim() || line.trimStart().startsWith("#")) continue;
    const indent = line.match(/^\s*/)[0].length;
    return indent > parentIndent && line.trim().startsWith("- ");
  }
  return false;
}

function parseScalar(value) {
  if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
    return value.slice(1, -1);
  }
  if (value === "true") return true;
  if (value === "false") return false;
  if (value === "null") return null;
  if (/^-?\d+(\.\d+)?$/.test(value)) return Number(value);
  if (value.startsWith("[") && value.endsWith("]")) {
    return value
      .slice(1, -1)
      .split(",")
      .map((item) => parseScalar(item.trim()))
      .filter((item) => item !== "");
  }
  return value;
}

function nestedNumber(source, pathParts) {
  let value = source;
  for (const key of pathParts) {
    if (!value || typeof value !== "object") return null;
    value = value[key];
  }
  return typeof value === "number" ? value : null;
}

function statusFromLegacy(value) {
  const status = String(value || "").toLowerCase();
  if (status.includes("done") || status.includes("success") || status.includes("accepted")) return "accepted";
  if (status.includes("claim") || status.includes("running")) return "claimed";
  if (status.includes("reject")) return "rejected";
  if (status.includes("draft")) return "draft";
  return "ready";
}

module.exports = {
  DEFAULT_ROOT,
  FileQueueStore,
  nowIso,
};
