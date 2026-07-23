// The one place ops go through: build attrs, call the native eager execute, and
// wrap the resulting handles back into typed Tensors.

import { DType, type DTypeName, dtypeName } from "../backend/dtype.js";
import type { Attr, AttrMap } from "../backend/native.js";
import { binding } from "../backend/native.js";
import { Tensor, tensorHandle } from "../tensor/tensor.js";

/** Runs an op and returns all of its outputs as Tensors. */
export function runOp(name: string, inputs: Tensor[], attrs: AttrMap, numOutputs = 1): Tensor[] {
  const handles = binding.execute(name, inputs.map(tensorHandle), attrs, numOutputs);
  return handles.map(
    (h) => new Tensor(h, binding.handleShape(h), dtypeName(binding.handleDtype(h))),
  );
}

/** Runs a single-output op and returns its one Tensor. */
export function runOp1(name: string, inputs: Tensor[], attrs: AttrMap): Tensor {
  const out = runOp(name, inputs, attrs, 1)[0];
  if (out === undefined) {
    throw new Error(`tfjs-native: op '${name}' produced no output`);
  }
  return out;
}

/** The `type`-kind attr carrying a tensor's dtype (the common `T` attribute). */
export function typeAttr(t: Tensor): Attr {
  return { type: "type", value: DType[t.dtype] };
}

/** The `type` attr for a named dtype. */
export function dtypeAttr(dtype: DTypeName): Attr {
  return { type: "type", value: DType[dtype] };
}
