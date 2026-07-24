// The single entry point that loads the compiled N-API addon.
//
// node-gyp-build resolves a prebuilt binary from `prebuilds/` when available,
// and falls back to a locally built `build/` addon otherwise. Nothing outside
// this module should require the addon directly (see docs/DESIGN.md §5).

import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);

// Resolve the package root by walking up to the directory that holds
// binding.gyp. This is robust across layouts: dev runs from src/backend/
// (2 levels deep) while the bundle runs from dist/ (1 level deep), and
// binding.gyp is shipped in the npm tarball so it exists at runtime too.
function findPackageRoot(start: string): string {
  let dir = start;
  for (;;) {
    if (existsSync(join(dir, "binding.gyp"))) return dir;
    const parent = dirname(dir);
    if (parent === dir) {
      throw new Error("tfjs-native: could not locate package root (binding.gyp not found)");
    }
    dir = parent;
  }
}

const packageRoot = findPackageRoot(dirname(fileURLToPath(import.meta.url)));

/**
 * Windows has no rpath: the addon resolves its `tensorflow.dll` dependency
 * through PATH at load time. Linux/macOS embed an rpath at build time instead
 * (see binding.gyp), so this is a no-op there.
 */
function ensureLibraryPath(root: string): void {
  if (process.platform !== "win32") return;
  const override = process.env.LIBTENSORFLOW_ROOT;
  const libDir = override ? join(override, "lib") : join(root, "deps", "libtensorflow", "lib");
  if (!existsSync(libDir)) return;
  const current = process.env.PATH ?? "";
  if (!current.split(";").includes(libDir)) {
    process.env.PATH = `${libDir};${current}`;
  }
}

ensureLibraryPath(packageRoot);

/** Opaque native handle types. Never inspected from JS. */
export type Ctx = { readonly __brand: "Ctx" };
export type Handle = { readonly __brand: "Handle" };
export type Session = { readonly __brand: "Session" };

/**
 * A single op attribute, tagged by its TF attr kind. `type`/`typeList` carry
 * TF_DataType numeric codes. `tensor`/`func` are not supported yet.
 */
export type Attr =
  | { type: "type"; value: number }
  | { type: "bool"; value: boolean }
  | { type: "int"; value: number }
  | { type: "float"; value: number }
  | { type: "string"; value: string }
  | { type: "shape"; value: number[] }
  | { type: "typeList"; value: number[] }
  | { type: "intList"; value: number[] }
  | { type: "floatList"; value: number[] }
  | { type: "boolList"; value: boolean[] }
  | { type: "stringList"; value: string[] };

/** Attribute name -> value, passed to `execute`. */
export type AttrMap = Record<string, Attr>;

export interface NativeBinding {
  version(): string;
  getAllOps(): Uint8Array;

  // --- tensor <-> handle (M1) ---
  // A TFE_Context is not required to create/resolve handles; execute() uses a
  // process-wide singleton context internally.
  /** Wraps flat TypedArray data into a native TFE_TensorHandle. */
  createHandle(data: ArrayBufferView, shape: readonly number[], dtype: number): Handle;
  /** Dimensions of the handle's tensor. */
  handleShape(h: Handle): number[];
  /** TF_DataType numeric code of the handle. */
  handleDtype(h: Handle): number;
  /** Raw little-endian bytes of the handle's tensor (a copy). */
  handleData(h: Handle): Uint8Array;
  /** Eagerly frees the handle. Safe to call once; a no-op after GC finalize. */
  deleteHandle(h: Handle): void;

  // --- eager execution (M2) ---
  /** Runs one op and returns its output handles. `numOutputs` sizes the buffer. */
  execute(opName: string, inputs: Handle[], attrs: AttrMap, numOutputs: number): Handle[];

  // --- sessions: SavedModel (M3) and graphs built for gradients (M4) ---
  /** Loads a SavedModel, returning its session handle and serialized MetaGraphDef. */
  loadSavedModel(dir: string, tags: string[]): { handle: Session; metaGraphDef: Uint8Array };
  /** Runs a session by graph tensor name; feeds and fetches are parallel arrays. */
  sessionRun(
    session: Session,
    inputOps: string[],
    inputIndices: number[],
    inputs: Handle[],
    outputOps: string[],
    outputIndices: number[],
  ): Handle[];
  /** Closes the session and frees its graph. Idempotent. */
  sessionDelete(session: Session): void;

  // The rest of the surface (gradient / ...) is declared in docs/DESIGN.md §5
  // and added as milestones land.
}

// biome-ignore lint/suspicious/noExplicitAny: node-gyp-build has no bundled types.
const load = require("node-gyp-build") as (root: string) => any;

export const binding: NativeBinding = load(packageRoot);
