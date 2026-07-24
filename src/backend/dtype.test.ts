import { describe, expect, it } from "bun:test";
import { DEFAULT_DTYPE, DType, TypedArrayFor } from "./dtype.js";

// Pure-TS tests that do not require the native addon to be built.
describe("dtype mapping", () => {
  it("matches TF_DataType numeric codes", () => {
    expect(DType.float32).toBe(1);
    expect(DType.int32).toBe(3);
    expect(DType.int64).toBe(9);
    expect(DType.bool).toBe(10);
  });

  it("defaults to float32", () => {
    expect(DEFAULT_DTYPE).toBe("float32");
  });

  it("reads int64 through BigInt64Array to avoid precision loss", () => {
    expect(TypedArrayFor.int64).toBe(BigInt64Array);
  });

  it("has a TypedArray for every named dtype except string", () => {
    for (const name of Object.keys(DType)) {
      if (name === "string") continue;
      expect(TypedArrayFor).toHaveProperty(name);
    }
  });
});
