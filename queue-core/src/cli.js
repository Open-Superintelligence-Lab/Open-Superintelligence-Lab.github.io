#!/usr/bin/env node
/* eslint-disable @typescript-eslint/no-require-imports */
"use strict";

const fs = require("fs");
const path = require("path");
const { FileQueueStore } = require("./file-store");

function main(argv) {
  const parsed = parseArgs(argv);
  const command = parsed.positionals[0];
  const store = new FileQueueStore({ root: parsed.flags.root });

  try {
    if (!command || parsed.flags.help || command === "help") {
      printHelp();
      return 0;
    }
    if (command === "init") {
      print(store.init(), parsed.flags);
      return 0;
    }
    if (command === "create") {
      const file = requiredFlag(parsed.flags, "file");
      print(store.createJob(readJsonFile(file)), parsed.flags);
      return 0;
    }
    if (command === "list") {
      print(store.listJobs({ status: parsed.flags.status, kind: parsed.flags.kind }), parsed.flags);
      return 0;
    }
    if (command === "get") {
      print(store.getJob(requiredArg(parsed.positionals, 1, "job id")), parsed.flags);
      return 0;
    }
    if (command === "claim") {
      const jobId = requiredArg(parsed.positionals, 1, "job id");
      const worker = requiredFlag(parsed.flags, "worker");
      print(store.claimJob(jobId, worker, { ttl_minutes: parsed.flags["ttl-minutes"] }), parsed.flags);
      return 0;
    }
    if (command === "heartbeat") {
      const leaseId = requiredArg(parsed.positionals, 1, "lease id");
      print(store.heartbeatClaim(leaseId, { ttl_minutes: parsed.flags["ttl-minutes"] }), parsed.flags);
      return 0;
    }
    if (command === "release") {
      const leaseId = requiredArg(parsed.positionals, 1, "lease id");
      print(store.releaseClaim(leaseId, parsed.flags.reason || ""), parsed.flags);
      return 0;
    }
    if (command === "submit") {
      const leaseId = requiredArg(parsed.positionals, 1, "lease id");
      const file = requiredFlag(parsed.flags, "file");
      print(store.submitResult(leaseId, readJsonFile(file)), parsed.flags);
      return 0;
    }
    if (command === "revise-result") {
      const jobId = requiredArg(parsed.positionals, 1, "job id");
      const runId = requiredArg(parsed.positionals, 2, "run id");
      const file = requiredFlag(parsed.flags, "file");
      print(store.reviseResult(jobId, runId, readJsonFile(file)), parsed.flags);
      return 0;
    }
    if (command === "review") {
      const targetType = requiredArg(parsed.positionals, 1, "target type");
      const targetId = requiredArg(parsed.positionals, 2, "target id");
      const review = store.reviewTarget(targetType, targetId, {
        reviewer_id: requiredFlag(parsed.flags, "reviewer"),
        decision: requiredFlag(parsed.flags, "decision"),
        note: parsed.flags.note || "",
      });
      print(review, parsed.flags);
      return 0;
    }
    if (command === "export") {
      const publicPath = requiredFlag(parsed.flags, "public");
      const snapshot = store.writeSnapshot(publicPath, { status: parsed.flags.status, kind: parsed.flags.kind });
      print(snapshot, parsed.flags);
      return 0;
    }
    if (command === "migrate") {
      print(
        store.migrate({
          source_queue: parsed.flags["source-queue"],
          source_results: parsed.flags["source-results"],
        }),
        parsed.flags,
      );
      return 0;
    }
    throw new Error(`unknown command: ${command}`);
  } catch (error) {
    console.error(`queue-core: ${error.message}`);
    return 1;
  }
}

function parseArgs(argv) {
  const flags = {};
  const positionals = [];
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg.startsWith("--")) {
      const key = arg.slice(2);
      const next = argv[index + 1];
      if (!next || next.startsWith("--")) {
        flags[key] = true;
      } else {
        flags[key] = next;
        index += 1;
      }
    } else {
      positionals.push(arg);
    }
  }
  return { flags, positionals };
}

function requiredArg(positionals, index, label) {
  const value = positionals[index];
  if (!value) throw new Error(`${label} is required`);
  return value;
}

function requiredFlag(flags, key) {
  if (!flags[key]) throw new Error(`--${key} is required`);
  return flags[key];
}

function readJsonFile(filePath) {
  return JSON.parse(fs.readFileSync(path.resolve(filePath), "utf8"));
}

function print(value, flags) {
  if (flags.json) {
    process.stdout.write(`${JSON.stringify(value)}\n`);
    return;
  }
  process.stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

function printHelp() {
  process.stdout.write(`Queue Core

Usage:
  node queue-core/src/cli.js init [--root path]
  node queue-core/src/cli.js create --file job.json [--root path]
  node queue-core/src/cli.js list [--status ready] [--kind experiment] [--root path]
  node queue-core/src/cli.js get <job_id> [--root path]
  node queue-core/src/cli.js claim <job_id> --worker <worker_id> [--ttl-minutes 120] [--root path]
  node queue-core/src/cli.js heartbeat <lease_id> [--ttl-minutes 120] [--root path]
  node queue-core/src/cli.js release <lease_id> [--reason text] [--root path]
  node queue-core/src/cli.js submit <lease_id> --file result.json [--root path]
  node queue-core/src/cli.js revise-result <job_id> <run_id> --file result.json [--root path]
  node queue-core/src/cli.js review <job|result> <target_id> --reviewer <id> --decision <approve|request_changes|reject|keep> [--note text] [--root path]
  node queue-core/src/cli.js export --public public/data/queue-snapshot.json [--root path]
  node queue-core/src/cli.js migrate --source-queue ../llm-research-kit-scaling/queue --source-results ../llm-research-kit-scaling/results [--root path]
`);
}

if (require.main === module) {
  process.exitCode = main(process.argv.slice(2));
}

module.exports = { main, parseArgs };
