// Pure conversion between flat JS values and dtype-specific buffers, and back.
// No native dependency.
//
// Numeric dtypes travel as one TypedArray. `complex64` is a Float32Array of
// interleaved [real, imag] pairs, so it holds two floats per element. `string`
// has no fixed width: each element travels as its own UTF-8 byte array.

import { type DTypeName, TypedArrayFor } from "../backend/dtype.js";
import type { TensorInputScalar, TensorScalar } from "./types.js";

type FixedWidthDType = Exclude<DTypeName, "string">;

const utf8Encoder = new TextEncoder();
const utf8Decoder = new TextDecoder();

function assertFixedWidth(dtype: DTypeName): asserts dtype is FixedWidthDType {
  if (dtype === "string") {
    throw new Error("tfjs-native: 'string' tensors have no fixed-width buffer");
  }
}

/** Byte width of one element of `dtype` (8 for complex64). Throws for `string`. */
export function bytesPerElement(dtype: DTypeName): number {
  assertFixedWidth(dtype);
  const width = TypedArrayFor[dtype].BYTES_PER_ELEMENT;
  return dtype === "complex64" ? 2 * width : width;
}

/** Number of TypedArray slots `size` elements of `dtype` occupy. */
function slotsFor(dtype: FixedWidthDType, size: number): number {
  return dtype === "complex64" ? 2 * size : size;
}

function toBigInt(value: TensorScalar): bigint {
  if (typeof value === "bigint") return value;
  if (typeof value === "boolean") return value ? 1n : 0n;
  if (Number.isInteger(value)) return BigInt(value);
  throw new Error(`tfjs-native: cannot store non-integer ${value} as int64`);
}

function assertNumericLeaf(value: TensorInputScalar, dtype: DTypeName): TensorScalar {
  if (typeof value === "string" || value instanceof Uint8Array) {
    throw new Error(`tfjs-native: cannot store a string or byte value in a '${dtype}' tensor`);
  }
  return value;
}

/** Packs flat values into the TypedArray backing a fixed-width `dtype`. */
export function toTypedArray(
  values: readonly TensorInputScalar[],
  dtype: DTypeName,
): ArrayBufferView {
  assertFixedWidth(dtype);
  if (dtype === "complex64") {
    throw new Error(
      "tfjs-native: build complex64 tensors with complex(real, imag) or from an interleaved Float32Array",
    );
  }
  const leaves = values.map((v) => assertNumericLeaf(v, dtype));
  if (dtype === "int64") {
    const out = new BigInt64Array(leaves.length);
    for (let i = 0; i < leaves.length; i++) {
      out[i] = toBigInt(leaves[i] as TensorScalar);
    }
    return out;
  }
  if (dtype === "bool") {
    const out = new Uint8Array(leaves.length);
    for (let i = 0; i < leaves.length; i++) {
      out[i] = leaves[i] ? 1 : 0;
    }
    return out;
  }
  const Ctor = TypedArrayFor[dtype] as
    | typeof Float32Array
    | typeof Float64Array
    | typeof Int32Array
    | typeof Uint8Array;
  const out = new Ctor(leaves.length);
  for (let i = 0; i < leaves.length; i++) {
    out[i] = Number(leaves[i]);
  }
  return out;
}

/** Encodes `string` tensor elements: strings as UTF-8, byte arrays as-is. */
export function encodeStrings(values: readonly TensorInputScalar[]): Uint8Array[] {
  return values.map((value) => {
    if (typeof value === "string") return utf8Encoder.encode(value);
    if (value instanceof Uint8Array) return value;
    throw new Error(
      `tfjs-native: a 'string' tensor needs string or Uint8Array values, got ${typeof value}`,
    );
  });
}

/** Decodes `string` tensor elements as UTF-8 (invalid bytes become U+FFFD). */
export function decodeStrings(values: readonly Uint8Array[]): string[] {
  return values.map((bytes) => utf8Decoder.decode(bytes));
}

/** Reinterprets a raw byte buffer as the TypedArray view for a fixed-width `dtype`. */
export function viewBytesAs(bytes: Uint8Array, dtype: DTypeName, size: number): ArrayBufferView {
  assertFixedWidth(dtype);
  const Ctor = TypedArrayFor[dtype];
  return new Ctor(bytes.buffer as ArrayBuffer, bytes.byteOffset, slotsFor(dtype, size));
}

/**
 * Converts a dtype's TypedArray view into plain JS scalars for `array()`.
 * complex64 stays interleaved: [re0, im0, re1, im1, ...].
 */
export function readValues(view: ArrayBufferView, dtype: DTypeName): TensorScalar[] {
  if (dtype === "bool") {
    const arr = view as Uint8Array;
    return Array.from(arr, (v) => v !== 0);
  }
  if (dtype === "int64") {
    const arr = view as BigInt64Array;
    return Array.from(arr);
  }
  const arr = view as Float32Array | Float64Array | Int32Array | Uint8Array;
  return Array.from(arr);
}
