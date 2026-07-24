// Reductions. TF takes the reduction axes as an int32 tensor input (not an attr),
// so we build a short-lived index tensor and dispose it after the op runs.

import { tensor } from "../../tensor/factory.js";
import type { Tensor } from "../../tensor/tensor.js";
import { disposeTemporary } from "../../training/tape.js";
import { dtypeAttr, runOp1, typeAttr } from "../run.js";

function allAxes(rank: number): number[] {
  return Array.from({ length: rank }, (_, i) => i);
}

function reduce(
  op: string,
  x: Tensor,
  axis: number | number[] | undefined,
  keepDims: boolean,
): Tensor {
  const axes = axis === undefined ? allAxes(x.rank) : Array.isArray(axis) ? axis : [axis];
  const axesTensor = tensor(axes, [axes.length], "int32");
  try {
    return runOp1(op, [x, axesTensor], {
      T: typeAttr(x),
      Tidx: dtypeAttr("int32"),
      keep_dims: { type: "bool", value: keepDims },
    });
  } finally {
    disposeTemporary(axesTensor);
  }
}

/** Sum over `axis` (all axes when omitted). */
export function sum(x: Tensor, axis?: number | number[], keepDims = false): Tensor {
  return reduce("Sum", x, axis, keepDims);
}

/** Mean over `axis` (all axes when omitted). */
export function mean(x: Tensor, axis?: number | number[], keepDims = false): Tensor {
  return reduce("Mean", x, axis, keepDims);
}

/** Max over `axis` (all axes when omitted). */
export function max(x: Tensor, axis?: number | number[], keepDims = false): Tensor {
  return reduce("Max", x, axis, keepDims);
}

/** Min over `axis` (all axes when omitted). */
export function min(x: Tensor, axis?: number | number[], keepDims = false): Tensor {
  return reduce("Min", x, axis, keepDims);
}
