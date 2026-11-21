import { describe, expect, it } from "bun:test";
import {
  bytesPerElement,
  decodeStrings,
  encodeStrings,
  readValues,
  toTypedArray,
  viewBytesAs,
} from "./convert.js";

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

  it("rejects string leaves in a numeric dtype instead of storing NaN", () => {
    expect(() => toTypedArray(["1"], "float32")).toThrow(/string or byte/);
    expect(() => toTypedArray([new Uint8Array([1])], "int32")).toThrow(/string or byte/);
  });

  it("refuses to pack complex64 from a flat list", () => {
    expect(() => toTypedArray([1, 2], "complex64")).toThrow(/complex\(real, imag\)/);
  });
});

describe("complex64 layout", () => {
  it("counts one element as two float32 slots", () => {
    expect(bytesPerElement("complex64")).toBe(8);
  });

  it("views interleaved bytes as 2 * size floats", () => {
    const packed = new Float32Array([1, 2, 3, 4]);
    const view = viewBytesAs(new Uint8Array(packed.buffer), "complex64", 2);
    expect(Array.from(view as Float32Array)).toEqual([1, 2, 3, 4]);
  });
});

describe("string encoding", () => {
  it("encodes strings as UTF-8 and passes bytes through", () => {
    const raw = new Uint8Array([0, 255]);
    const [a, b] = encodeStrings(["é", raw]);
    expect(Array.from(a as Uint8Array)).toEqual([0xc3, 0xa9]);
    expect(b).toBe(raw);
  });

  it("rejects non-string leaves", () => {
    expect(() => encodeStrings([1])).toThrow(/string or Uint8Array/);
  });

  it("decodes UTF-8 back to strings", () => {
    expect(decodeStrings(encodeStrings(["héllo", ""]))).toEqual(["héllo", ""]);
  });

  it("has no fixed element width", () => {
    expect(() => bytesPerElement("string")).toThrow();
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
