// Pure shape helpers. No dependency on the native addon, so these are fully
// unit-testable on their own.

import type { NestedInput, TensorInputScalar } from "./types.js";

/** Number of elements described by a shape. Empty shape (scalar) is 1. */
export function sizeFromShape(shape: readonly number[]): number {
  let size = 1;
  for (const dim of shape) {
    size *= dim;
  }
  return size;
}

/** Throws when a shape has a non-integer or negative dimension. */
export function assertValidShape(shape: readonly number[]): void {
  for (const dim of shape) {
    if (!Number.isInteger(dim) || dim < 0) {
      throw new Error(`tfjs-native: invalid shape dimension: ${dim}`);
    }
  }
}

/** Infers the shape of a nested list by walking its first elements. */
export function inferShape(value: NestedInput): number[] {
  const shape: number[] = [];
  let cursor: NestedInput = value;
  while (Array.isArray(cursor)) {
    shape.push(cursor.length);
    cursor = cursor[0] as NestedInput;
  }
  return shape;
}

/** Throws when a nested list is ragged (does not match `shape` rectangularly). */
export function assertRectangular(value: NestedInput, shape: readonly number[], dim = 0): void {
  if (dim === shape.length) {
    if (Array.isArray(value)) {
      throw new Error("tfjs-native: ragged array — nested deeper than its inferred shape");
    }
    return;
  }
  if (!Array.isArray(value) || value.length !== shape[dim]) {
    throw new Error(`tfjs-native: ragged array — expected length ${shape[dim]} at depth ${dim}`);
  }
  for (const element of value) {
    assertRectangular(element as NestedInput, shape, dim + 1);
  }
}

/** Flattens a nested list into `out` in row-major order. */
export function flattenInto(value: NestedInput, out: TensorInputScalar[]): void {
  if (Array.isArray(value)) {
    for (const element of value) {
      flattenInto(element as NestedInput, out);
    }
  } else {
    out.push(value as TensorInputScalar);
  }
}

/** Nests a flat, row-major array back into `shape`. Empty shape returns flat[0]. */
export function toNested<T>(flat: readonly T[], shape: readonly number[]): T | unknown[] {
  if (shape.length === 0) {
    return flat[0] as T;
  }
  const [head, ...rest] = shape;
  const step = sizeFromShape(rest);
  const out: unknown[] = [];
  for (let i = 0; i < (head as number); i++) {
    out.push(toNested(flat.slice(i * step, (i + 1) * step), rest));
  }
  return out;
}
