import { describe, expect, it } from "bun:test";
import { readValues, toTypedArray, viewBytesAs } from "./convert.js";

describe("toTypedArray", () => {
  it("packs float32 by default-ish path", () => {
    const view = toTypedArray([1, 2, 3], "float32");
    expect(view).toBeInstanceOf(Float32Array);
    expect(Array.from(view as Float32Array)).toEqual([1, 2, 3]);
  });

  it("packs int32", () => {
    const view = toTypedArray([1, 2, 3], "int32");
    expect(view).toBeInstanceOf(Int32Array);
  });

  it("packs int64 through BigInt64Array", () => {
    const view = toTypedArray([1, 2, 3], "int64");
    expect(view).toBeInstanceOf(BigInt64Array);
    expect(Array.from(view as BigInt64Array)).toEqual([1n, 2n, 3n]);
  });

  it("packs bool as 0/1 bytes", () => {
    const view = toTypedArray([true, false, true], "bool");
    expect(Array.from(view as Uint8Array)).toEqual([1, 0, 1]);
  });

  it("rejects non-integer int64", () => {
    expect(() => toTypedArray([1.5], "int64")).toThrow();
  });

  it("rejects string dtype", () => {
    expect(() => toTypedArray([1], "string")).toThrow();
  });
});

describe("viewBytesAs + readValues round-trip", () => {
  it("round-trips float32", () => {
    const packed = toTypedArray([1.5, -2.5, 3], "float32") as Float32Array;
    const bytes = new Uint8Array(packed.buffer);
    const view = viewBytesAs(bytes, "float32", 3);
    expect(readValues(view, "float32")).toEqual([1.5, -2.5, 3]);
  });

  it("round-trips int64", () => {
    const packed = toTypedArray([10, 20], "int64") as BigInt64Array;
    const bytes = new Uint8Array(packed.buffer);
    const view = viewBytesAs(bytes, "int64", 2);
    expect(readValues(view, "int64")).toEqual([10n, 20n]);
  });

  it("reads bool bytes back as booleans", () => {
    const packed = toTypedArray([true, false], "bool") as Uint8Array;
    const bytes = new Uint8Array(packed.buffer);
    const view = viewBytesAs(bytes, "bool", 2);
    expect(readValues(view, "bool")).toEqual([true, false]);
  });
});
