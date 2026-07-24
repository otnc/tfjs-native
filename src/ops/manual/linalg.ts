import type { Tensor } from "../../tensor/tensor.js";
import { runOp1, typeAttr } from "../run.js";

export interface MatMulOptions {
  transposeA?: boolean;
  transposeB?: boolean;
}

/** Matrix product of two rank-2 tensors. */
export function matMul(a: Tensor, b: Tensor, options: MatMulOptions = {}): Tensor {
  return runOp1("MatMul", [a, b], {
    T: typeAttr(a),
    transpose_a: { type: "bool", value: options.transposeA ?? false },
    transpose_b: { type: "bool", value: options.transposeB ?? false },
  });
}
