// Pure conversion between flat JS values and dtype-specific TypedArrays, and
// back. No native dependency. String / complex64 are deferred past M1.

import { type DTypeName, TypedArrayFor } from "../backend/dtype.js";
import type { TensorScalar } from "./types.js";

function unsupported(dtype: DTypeName): never {
  throw new Error(`tfjs-native: dtype '${dtype}' is not supported yet (M1 is numeric-only)`);
}

/** Byte width of one element of `dtype`. Throws for the deferred dtypes. */
export function bytesPerElement(dtype: DTypeName): number {
  if (dtype === "string" || dtype === "complex64") {
    unsupported(dtype);
  }
  return TypedArrayFor[dtype].BYTES_PER_ELEMENT;
}

function toBigInt(value: TensorScalar): bigint {
  if (typeof value === "bigint") return value;
  if (typeof value === "boolean") return value ? 1n : 0n;
  if (Number.isInteger(value)) return BigInt(value);
  throw new Error(`tfjs-native: cannot store non-integer ${value} as int64`);
}

/** Packs flat values into the TypedArray backing `dtype`. */
export function toTypedArray(values: readonly TensorScalar[], dtype: DTypeName): ArrayBufferView {
  if (dtype === "string" || dtype === "complex64") {
    unsupported(dtype);
  }
  if (dtype === "int64") {
    const out = new BigInt64Array(values.length);
    for (let i = 0; i < values.length; i++) {
      out[i] = toBigInt(values[i] as TensorScalar);
    }
    return out;
  }
  if (dtype === "bool") {
    const out = new Uint8Array(values.length);
    for (let i = 0; i < values.length; i++) {
      out[i] = values[i] ? 1 : 0;
    }
    return out;
  }
  const Ctor = TypedArrayFor[dtype] as
    | typeof Float32Array
    | typeof Float64Array
    | typeof Int32Array
    | typeof Uint8Array;
  const out = new Ctor(values.length);
  for (let i = 0; i < values.length; i++) {
    out[i] = Number(values[i]);
  }
  return out;
}

/** Reinterprets a raw byte buffer as the TypedArray view for `dtype`. */
export function viewBytesAs(bytes: Uint8Array, dtype: DTypeName, size: number): ArrayBufferView {
  if (dtype === "string" || dtype === "complex64") {
    unsupported(dtype);
  }
  const Ctor = TypedArrayFor[dtype];
  return new Ctor(bytes.buffer as ArrayBuffer, bytes.byteOffset, size);
}

/** Converts a dtype's TypedArray view into plain JS scalars for `array()`. */
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
