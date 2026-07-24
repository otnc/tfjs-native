// Shape / dtype ops. Reshape and transpose take their shape/permutation as an
// int32 tensor input, built here and disposed after the op runs.

import { DType, type DTypeName } from "../../backend/dtype.js";
import { tensor } from "../../tensor/factory.js";
import type { Tensor } from "../../tensor/tensor.js";
import { disposeTemporary } from "../../training/tape.js";
import { dtypeAttr, runOp1, typeAttr } from "../run.js";

/** Reshapes `x` to `shape` (same element count). */
export function reshape(x: Tensor, shape: number[]): Tensor {
  const shapeTensor = tensor(shape, [shape.length], "int32");
  try {
    return runOp1("Reshape", [x, shapeTensor], {
      T: typeAttr(x),
      Tshape: dtypeAttr("int32"),
    });
  } finally {
    disposeTemporary(shapeTensor);
  }
}

/** Permutes the dimensions of `x` (reverses them when `perm` is omitted). */
export function transpose(x: Tensor, perm?: number[]): Tensor {
  const p = perm ?? Array.from({ length: x.rank }, (_, i) => x.rank - 1 - i);
  const permTensor = tensor(p, [p.length], "int32");
  try {
    return runOp1("Transpose", [x, permTensor], {
      T: typeAttr(x),
      Tperm: dtypeAttr("int32"),
    });
  } finally {
    disposeTemporary(permTensor);
  }
}

/** Casts `x` to another dtype. */
export function cast(x: Tensor, dtype: DTypeName): Tensor {
  return runOp1("Cast", [x], {
    SrcT: typeAttr(x),
    DstT: { type: "type", value: DType[dtype] },
  });
}
