// Records eager op executions so they can be replayed into a graph.
//
// The C API has no eager gradient tape, so gradients are obtained by rebuilding
// the recorded computation as a TF_Graph and calling TF_AddGradients on it.
// Recording is a no-op (one null check) unless a tape is active.

import type { AttrMap } from "../backend/native.js";
import type { Tensor } from "../tensor/tensor.js";

export interface TapeEntry {
  op: string;
  attrs: AttrMap;
  inputs: Tensor[];
  outputs: Tensor[];
}

interface Tape {
  entries: TapeEntry[];
  /** Helper tensors that must outlive the traced function (see disposeTemporary). */
  temporaries: Tensor[];
}

let active: Tape | null = null;

/** True while a tape is recording. */
export function isRecording(): boolean {
  return active !== null;
}

/** Records one op execution. */
export function record(entry: TapeEntry): void {
  if (active !== null) active.entries.push(entry);
}

/**
 * Disposes a short-lived tensor an op created for itself (a reduction's axis
 * tensor, say). While a tape is recording such a tensor is still referenced by
 * the recording, so it is handed to the tape to free after the replay instead.
 */
export function disposeTemporary(tensor: Tensor): void {
  if (active === null) {
    tensor.dispose();
    return;
  }
  active.temporaries.push(tensor);
}

export interface TapeResult<T> {
  result: T;
  entries: TapeEntry[];
  temporaries: Tensor[];
}

/** Runs `fn` with a fresh tape and returns what it recorded. */
export function withTape<T>(fn: () => T): TapeResult<T> {
  const previous = active;
  const tape: Tape = { entries: [], temporaries: [] };
  active = tape;
  try {
    const result = fn();
    return { result, entries: tape.entries, temporaries: tape.temporaries };
  } finally {
    active = previous;
  }
}
