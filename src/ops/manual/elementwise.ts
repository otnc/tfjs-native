// Element-wise ops. Binary ops assume both inputs share a dtype (TF requires it);
// the `T` attr is taken from the first input.

import type { Tensor } from "../../tensor/tensor.js";
import { runOp1, typeAttr } from "../run.js";

function binary(op: string, a: Tensor, b: Tensor): Tensor {
  return runOp1(op, [a, b], { T: typeAttr(a) });
}

function unary(op: string, x: Tensor): Tensor {
  return runOp1(op, [x], { T: typeAttr(x) });
}

export const add = (a: Tensor, b: Tensor): Tensor => binary("AddV2", a, b);
export const sub = (a: Tensor, b: Tensor): Tensor => binary("Sub", a, b);
export const mul = (a: Tensor, b: Tensor): Tensor => binary("Mul", a, b);
/** Floating-point division (`RealDiv`). */
export const div = (a: Tensor, b: Tensor): Tensor => binary("RealDiv", a, b);
export const maximum = (a: Tensor, b: Tensor): Tensor => binary("Maximum", a, b);
export const minimum = (a: Tensor, b: Tensor): Tensor => binary("Minimum", a, b);
export const pow = (a: Tensor, b: Tensor): Tensor => binary("Pow", a, b);

export const neg = (x: Tensor): Tensor => unary("Neg", x);
export const abs = (x: Tensor): Tensor => unary("Abs", x);
export const exp = (x: Tensor): Tensor => unary("Exp", x);
export const log = (x: Tensor): Tensor => unary("Log", x);
export const sqrt = (x: Tensor): Tensor => unary("Sqrt", x);
export const square = (x: Tensor): Tensor => unary("Square", x);
export const relu = (x: Tensor): Tensor => unary("Relu", x);
export const sigmoid = (x: Tensor): Tensor => unary("Sigmoid", x);
export const tanh = (x: Tensor): Tensor => unary("Tanh", x);
