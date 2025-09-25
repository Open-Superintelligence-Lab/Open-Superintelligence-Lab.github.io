/* eslint-disable */
/**
 * Generated `api` utility.
 *
 * THIS CODE IS AUTOMATICALLY GENERATED.
 *
 * To regenerate, run `npx convex dev`.
 * @module
 */

import type {
  ApiFromModules,
  FilterApi,
  FunctionReference,
} from "convex/server";
import type * as agents from "../agents.js";
import type * as chat from "../chat.js";
import type * as projects from "../projects.js";
import type * as runs from "../runs.js";
import type * as seed from "../seed.js";
import type * as tutorialChat from "../tutorialChat.js";
import type * as tutorials from "../tutorials.js";

/**
 * A utility for referencing Convex functions in your app's API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = api.myModule.myFunction;
 * ```
 */
declare const fullApi: ApiFromModules<{
  agents: typeof agents;
  chat: typeof chat;
  projects: typeof projects;
  runs: typeof runs;
  seed: typeof seed;
  tutorialChat: typeof tutorialChat;
  tutorials: typeof tutorials;
}>;
export declare const api: FilterApi<
  typeof fullApi,
  FunctionReference<any, "public">
>;
export declare const internal: FilterApi<
  typeof fullApi,
  FunctionReference<any, "internal">
>;
