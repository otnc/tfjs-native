// tfjs-native public API.
//
// Goal: expose the full libtensorflow C-API op registry to typed TypeScript,
// plus SavedModel execution and a thin TS-side training layer.
// See docs/DESIGN.md for the design and roadmap.

import { binding } from "./backend/native.js";

export type { DTypeName, DTypeValue } from "./backend/dtype.js";
export { DEFAULT_DTYPE, DType } from "./backend/dtype.js";
export type { Attr, AttrMap } from "./backend/native.js";
export type { SignatureDef } from "./model/proto.js";
export type { LoadSavedModelOptions } from "./model/saved-model.js";
export { DEFAULT_SIGNATURE, loadSavedModel, SavedModel } from "./model/saved-model.js";
export * from "./ops/index.js";
export { tidy } from "./tensor/engine.js";
export { ones, scalar, tensor, zeros } from "./tensor/factory.js";
export { Tensor } from "./tensor/tensor.js";
export type { NestedList, TensorScalar, TensorValues } from "./tensor/types.js";
export type { ValueAndGrads } from "./training/gradients.js";
export { grads, valueAndGrads } from "./training/gradients.js";
export type { AdamOptions, RmsPropOptions, SgdOptions } from "./training/optimizer.js";
export { adam, Optimizer, rmsprop, sgd } from "./training/optimizer.js";
export { Variable, variable } from "./training/variable.js";

/** Returns the linked libtensorflow version (M0 smoke test). */
export function version(): string {
  return binding.version();
}
