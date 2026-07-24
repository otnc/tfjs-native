// Tensor factories. These process input on the pure-TS side (infer shape,
// validate, flatten to a TypedArray) and then hand a single buffer to the addon.

import { DEFAULT_DTYPE, DType, type DTypeName } from "../backend/dtype.js";
import { binding } from "../backend/native.js";
import { bytesPerElement, toTypedArray } from "./convert.js";
import {
  assertRectangular,
  assertValidShape,
  flattenInto,
  inferShape,
  sizeFromShape,
} from "./shape.js";
import { Tensor } from "./tensor.js";
import type { TensorScalar, TensorValues } from "./types.js";

function fromView(view: ArrayBufferView, shape: readonly number[], dtype: DTypeName): Tensor {
  assertValidShape(shape);
  const expected = sizeFromShape(shape);
  const length = view.byteLength / bytesPerElement(dtype);
  if (length !== expected) {
    throw new Error(
      `tfjs-native: data length ${length} does not match shape ${JSON.stringify(shape)} (size ${expected})`,
    );
  }
  return new Tensor(binding.createHandle(view, shape, DType[dtype]), shape, dtype);
}

/**
 * Creates a tensor from a scalar, a (rectangular) nested array, or a flat
 * TypedArray. Shape is inferred when omitted.
 */
export function tensor(
  values: TensorValues,
  shape?: readonly number[],
  dtype: DTypeName = DEFAULT_DTYPE,
): Tensor {
  if (ArrayBuffer.isView(values)) {
    const resolved = shape ?? [values.byteLength / bytesPerElement(dtype)];
    return fromView(values, resolved, dtype);
  }
  const resolvedShape = shape ?? inferShape(values);
  assertValidShape(resolvedShape);
  assertRectangular(values, resolvedShape);
  const flat: TensorScalar[] = [];
  flattenInto(values, flat);
  if (flat.length !== sizeFromShape(resolvedShape)) {
    throw new Error(
      `tfjs-native: ${flat.length} values do not fill shape ${JSON.stringify(resolvedShape)}`,
    );
  }
  return fromView(toTypedArray(flat, dtype), resolvedShape, dtype);
}

/** Creates a rank-0 tensor. */
export function scalar(value: TensorScalar, dtype: DTypeName = DEFAULT_DTYPE): Tensor {
  return tensor(value, [], dtype);
}

function filled(shape: readonly number[], dtype: DTypeName, value: number): Tensor {
  assertValidShape(shape);
  const size = sizeFromShape(shape);
  const flat: TensorScalar[] = new Array(size).fill(dtype === "int64" ? BigInt(value) : value);
  return fromView(toTypedArray(flat, dtype), shape, dtype);
}

/** Creates a tensor of zeros. */
export function zeros(shape: readonly number[], dtype: DTypeName = DEFAULT_DTYPE): Tensor {
  return filled(shape, dtype, 0);
}

/** Creates a tensor of ones. */
export function ones(shape: readonly number[], dtype: DTypeName = DEFAULT_DTYPE): Tensor {
  return filled(shape, dtype, 1);
}
