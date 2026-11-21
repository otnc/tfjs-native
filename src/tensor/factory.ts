// Tensor factories. These process input on the pure-TS side (infer shape,
// validate, flatten to a buffer) and then hand it to the addon in one call.

import { DEFAULT_DTYPE, DType, type DTypeName } from "../backend/dtype.js";
import { binding } from "../backend/native.js";
import { bytesPerElement, encodeStrings, toTypedArray } from "./convert.js";
import {
  assertRectangular,
  assertValidShape,
  flattenInto,
  inferShape,
  sizeFromShape,
} from "./shape.js";
import { Tensor } from "./tensor.js";
import type { TensorInputScalar, TensorValues } from "./types.js";

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

function fromStrings(values: readonly TensorInputScalar[], shape: readonly number[]): Tensor {
  return new Tensor(binding.createStringHandle(encodeStrings(values), shape), shape, "string");
}

function isStringLeaf(value: TensorInputScalar | undefined): boolean {
  return typeof value === "string" || value instanceof Uint8Array;
}

/**
 * Creates a tensor from a scalar, a (rectangular) nested array, or a flat
 * TypedArray. Shape is inferred when omitted. Without a dtype, string (or
 * `Uint8Array`) leaves make a `string` tensor and anything else `float32`.
 * complex64 accepts only an interleaved [re, im, ...] Float32Array.
 */
export function tensor(values: TensorValues, shape?: readonly number[], dtype?: DTypeName): Tensor {
  if (ArrayBuffer.isView(values)) {
    const resolvedDtype = dtype ?? DEFAULT_DTYPE;
    const resolved = shape ?? [values.byteLength / bytesPerElement(resolvedDtype)];
    return fromView(values, resolved, resolvedDtype);
  }
  const resolvedShape = shape ?? inferShape(values);
  assertValidShape(resolvedShape);
  assertRectangular(values, resolvedShape);
  const flat: TensorInputScalar[] = [];
  flattenInto(values, flat);
  if (flat.length !== sizeFromShape(resolvedShape)) {
    throw new Error(
      `tfjs-native: ${flat.length} values do not fill shape ${JSON.stringify(resolvedShape)}`,
    );
  }
  const resolvedDtype = dtype ?? (isStringLeaf(flat[0]) ? "string" : DEFAULT_DTYPE);
  if (resolvedDtype === "string") {
    return fromStrings(flat, resolvedShape);
  }
  return fromView(toTypedArray(flat, resolvedDtype), resolvedShape, resolvedDtype);
}

/** Creates a rank-0 tensor. */
export function scalar(value: TensorInputScalar, dtype?: DTypeName): Tensor {
  return tensor(value, [], dtype);
}

function filled(shape: readonly number[], dtype: DTypeName, value: 0 | 1): Tensor {
  assertValidShape(shape);
  const size = sizeFromShape(shape);
  if (dtype === "string") {
    if (value !== 0) {
      throw new Error(
        "tfjs-native: ones() has no meaning for 'string'; zeros() gives empty strings",
      );
    }
    return fromStrings(new Array<string>(size).fill(""), shape);
  }
  if (dtype === "complex64") {
    const interleaved = new Float32Array(2 * size);
    for (let i = 0; i < size; i++) interleaved[2 * i] = value;
    return fromView(interleaved, shape, dtype);
  }
  const flat: TensorInputScalar[] = new Array(size).fill(dtype === "int64" ? BigInt(value) : value);
  return fromView(toTypedArray(flat, dtype), shape, dtype);
}

/** Creates a tensor of zeros (empty strings for `string`). */
export function zeros(shape: readonly number[], dtype: DTypeName = DEFAULT_DTYPE): Tensor {
  return filled(shape, dtype, 0);
}

/** Creates a tensor of ones (1 + 0i for `complex64`). Not defined for `string`. */
export function ones(shape: readonly number[], dtype: DTypeName = DEFAULT_DTYPE): Tensor {
  return filled(shape, dtype, 1);
}
