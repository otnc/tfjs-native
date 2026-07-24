// Op layer. Generated wrappers cover the full op registry; hand-written ops
// (which the generator skips by name) provide friendlier signatures on top.

export * from "./generated/index.js";
export * from "./manual/array-ops.js";
export * from "./manual/elementwise.js";
export * from "./manual/linalg.js";
export * from "./manual/reduction.js";
// Low-level escape hatch: call any op by name (e.g. unsupported attr kinds).
export { runOp, runOp1 } from "./run.js";
