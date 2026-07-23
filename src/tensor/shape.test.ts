import { describe, expect, it } from "bun:test";
import {
  assertRectangular,
  assertValidShape,
  flattenInto,
  inferShape,
  sizeFromShape,
  toNested,
} from "./shape.js";
import type { TensorScalar } from "./types.js";

describe("sizeFromShape", () => {
  it("returns 1 for a scalar (empty shape)", () => {
    expect(sizeFromShape([])).toBe(1);
  });
  it("multiplies dimensions", () => {
    expect(sizeFromShape([2, 3, 4])).toBe(24);
  });
  it("returns 0 when any dimension is 0", () => {
    expect(sizeFromShape([2, 0, 4])).toBe(0);
  });
});

describe("assertValidShape", () => {
  it("accepts non-negative integer dims", () => {
    expect(() => assertValidShape([0, 1, 2])).not.toThrow();
  });
  it("rejects negative dims", () => {
    expect(() => assertValidShape([2, -1])).toThrow();
  });
  it("rejects non-integer dims", () => {
    expect(() => assertValidShape([2.5])).toThrow();
  });
});

describe("inferShape", () => {
  it("infers scalar as empty shape", () => {
    expect(inferShape(5)).toEqual([]);
  });
  it("infers 1D", () => {
    expect(inferShape([1, 2, 3])).toEqual([3]);
  });
  it("infers 2D", () => {
    expect(
      inferShape([
        [1, 2],
        [3, 4],
        [5, 6],
      ]),
    ).toEqual([3, 2]);
  });
});

describe("assertRectangular", () => {
  it("accepts a rectangular 2D array", () => {
    const v = [
      [1, 2],
      [3, 4],
    ];
    expect(() => assertRectangular(v, inferShape(v))).not.toThrow();
  });
  it("rejects a ragged array", () => {
    const v = [[1, 2], [3]];
    expect(() => assertRectangular(v, [2, 2])).toThrow();
  });
  it("rejects over-deep nesting past the shape", () => {
    expect(() => assertRectangular([[1]], [1])).toThrow();
  });
});

describe("flattenInto", () => {
  it("flattens nested arrays row-major", () => {
    const out: TensorScalar[] = [];
    flattenInto(
      [
        [1, 2],
        [3, 4],
      ],
      out,
    );
    expect(out).toEqual([1, 2, 3, 4]);
  });
  it("wraps a scalar", () => {
    const out: TensorScalar[] = [];
    flattenInto(7, out);
    expect(out).toEqual([7]);
  });
});

describe("toNested", () => {
  it("returns the scalar for an empty shape", () => {
    expect(toNested([42], [])).toBe(42);
  });
  it("round-trips with flattenInto", () => {
    const nested = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const flat: TensorScalar[] = [];
    flattenInto(nested, flat);
    expect(toNested(flat, [2, 3])).toEqual(nested);
  });
});
