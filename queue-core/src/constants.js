"use strict";

const JOB_KINDS = new Set(["experiment", "review", "code", "write", "ops"]);
const JOB_STATUSES = new Set(["draft", "ready", "claimed", "running", "blocked", "submitted", "accepted", "rejected", "archived"]);
const CLAIM_STATUSES = new Set(["active", "released", "expired", "completed"]);
const REVIEW_DECISIONS = new Set(["approve", "request_changes", "reject", "keep"]);
const TARGET_TYPES = new Set(["job", "result"]);

module.exports = {
  JOB_KINDS,
  JOB_STATUSES,
  CLAIM_STATUSES,
  REVIEW_DECISIONS,
  TARGET_TYPES,
};
