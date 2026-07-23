// Minimal scope tracking for tidy(). A stack of scopes; each newly created
// Tensor registers with the innermost scope so tidy() can auto-dispose the
// intermediates a function did not return.

import type { Tensor } from "./tensor.js";

const scopeStack: Set<Tensor>[] = [];

/** Registers a tensor with the innermost active tidy scope, if any. */
export function track(tensor: Tensor): void {
  scopeStack[scopeStack.length - 1]?.add(tensor);
}

/** Removes a tensor from the innermost scope (e.g. on explicit dispose). */
export function untrack(tensor: Tensor): void {
  scopeStack[scopeStack.length - 1]?.delete(tensor);
}

function collectKept(result: unknown, kept: Set<Tensor>): void {
  if (result == null) return;
  // Tensors expose a boolean `isDisposed`; treat any such object as a tensor
  // to keep. Avoids importing the class value (circular) at module scope.
  if (typeof (result as { isDisposed?: unknown }).isDisposed === "boolean") {
    kept.add(result as Tensor);
  } else if (Array.isArray(result)) {
    for (const item of result) collectKept(item, kept);
  } else if (typeof result === "object") {
    for (const value of Object.values(result as Record<string, unknown>)) {
      collectKept(value, kept);
    }
  }
}

/**
 * Runs `fn`, disposing every tensor created within it except those reachable
 * from its return value.
 */
export function tidy<T>(fn: () => T): T {
  const scope = new Set<Tensor>();
  scopeStack.push(scope);
  let result: T;
  try {
    result = fn();
  } finally {
    scopeStack.pop();
  }
  const kept = new Set<Tensor>();
  collectKept(result, kept);
  for (const tensor of scope) {
    if (!kept.has(tensor)) {
      tensor.dispose();
    }
  }
  return result;
}
