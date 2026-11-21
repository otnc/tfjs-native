// Round-trips for the variable-width `string` dtype and the interleaved
// `complex64` dtype through the native addon. Skipped when it is not built.

import { describe, expect, it } from "bun:test";

let nativeAvailable = false;
try {
  await import("../backend/native.js");
  nativeAvailable = true;
} catch {
  nativeAvailable = false;
}

describe.skipIf(!nativeAvailable)("string tensors (native)", () => {
  it("infers 'string' and reads nested strings back", async () => {
    const { tensor } = await import("./factory.js");
    const t = tensor([
      ["a", "bc"],
      ["", "héllo 世界"],
    ]);
    expect(t.dtype).toBe("string");
    expect(t.shape).toEqual([2, 2]);
    expect(await t.array()).toEqual([
      ["a", "bc"],
      ["", "héllo 世界"],
    ]);
    t.dispose();
  });

  it("keeps strings longer than the inline small-string storage", async () => {
    const { tensor } = await import("./factory.js");
    const long = "x".repeat(1000);
    const t = tensor([long, "short"]);
    expect(t.arraySync()).toEqual([long, "short"]);
    t.dispose();
  });

  it("round-trips arbitrary bytes, including NUL and invalid UTF-8", async () => {
    const { tensor } = await import("./factory.js");
    const raw = new Uint8Array([0, 1, 255, 0, 128]);
    const t = tensor([raw]);
    expect(t.dtype).toBe("string");
    const [back] = await t.bytes();
    expect(Array.from(back as Uint8Array)).toEqual(Array.from(raw));
    t.dispose();
  });

  it("survives many create/dispose cycles of heap-allocated strings", async () => {
    const { tensor } = await import("./factory.js");
    const values = Array.from({ length: 64 }, (_, i) => `${i}-${"y".repeat(100)}`);
    for (let i = 0; i < 500; i++) {
      tensor(values).dispose();
    }
    const t = tensor(values);
    expect(t.arraySync()).toEqual(values);
    t.dispose();
  });

  it("builds a scalar and empty-string zeros", async () => {
    const { scalar, zeros } = await import("./factory.js");
    const s = scalar("hi");
    expect(s.shape).toEqual([]);
    expect(s.arraySync()).toBe("hi");
    const z = zeros([2], "string");
    expect(z.arraySync()).toEqual(["", ""]);
    s.dispose();
    z.dispose();
  });

  it("rejects ones() and dataSync() for strings", async () => {
    const { ones, tensor } = await import("./factory.js");
    expect(() => ones([1], "string")).toThrow();
    const t = tensor(["a"]);
    expect(() => t.dataSync()).toThrow(/bytesSync/);
    t.dispose();
  });

  it("feeds string ops and reads their output", async () => {
    const { tensor } = await import("./factory.js");
    const ops = await import("../ops/index.js");
    const a = tensor(["foo", "bar"]);
    const b = tensor(["1", "2"]);
    const joined = ops.stringJoin([a, b], { separator: "-" });
    expect(joined.arraySync()).toEqual(["foo-1", "bar-2"]);
    const upper = ops.stringUpper(joined);
    expect(upper.arraySync()).toEqual(["FOO-1", "BAR-2"]);
    const lengths = ops.stringLength(upper);
    expect(lengths.arraySync()).toEqual([5, 5]);
    const n = ops.stringToNumber(b);
    expect(n.arraySync()).toEqual([1, 2]);
    for (const t of [a, b, joined, upper, lengths, n]) t.dispose();
  });

  it("formats numbers into a string tensor", async () => {
    const { tensor } = await import("./factory.js");
    const { asString } = await import("../ops/index.js");
    const x = tensor([1, 2], [2], "int32");
    const s = asString(x);
    expect(s.dtype).toBe("string");
    expect(s.arraySync()).toEqual(["1", "2"]);
    x.dispose();
    s.dispose();
  });
});

describe.skipIf(!nativeAvailable)("complex64 tensors (native)", () => {
  it("builds from real and imaginary parts", async () => {
    const { tensor } = await import("./factory.js");
    const { complex, imag, real } = await import("../ops/index.js");
    const re = tensor([1, 2]);
    const im = tensor([3, 4]);
    const c = complex(re, im);
    expect(c.dtype).toBe("complex64");
    expect(c.shape).toEqual([2]);
    expect(Array.from(c.dataSync() as Float32Array)).toEqual([1, 3, 2, 4]);
    expect(c.arraySync()).toEqual([1, 3, 2, 4]);
    const r = real(c);
    const i = imag(c);
    expect(r.arraySync()).toEqual([1, 2]);
    expect(i.arraySync()).toEqual([3, 4]);
    for (const t of [re, im, c, r, i]) t.dispose();
  });

  it("accepts an interleaved Float32Array and nests pairs in the innermost dim", async () => {
    const { tensor } = await import("./factory.js");
    const c = tensor(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2], "complex64");
    expect(c.size).toBe(4);
    expect(c.arraySync()).toEqual([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]);
    c.dispose();
  });

  it("infers the element count from an interleaved buffer", async () => {
    const { tensor } = await import("./factory.js");
    const c = tensor(new Float32Array([1, 2, 3, 4]), undefined, "complex64");
    expect(c.shape).toEqual([2]);
    c.dispose();
  });

  it("does complex arithmetic", async () => {
    const { tensor } = await import("./factory.js");
    const { mul } = await import("../ops/index.js");
    // (1 + 2i)(3 + 4i) = -5 + 10i
    const a = tensor(new Float32Array([1, 2]), [], "complex64");
    const b = tensor(new Float32Array([3, 4]), [], "complex64");
    const p = mul(a, b);
    expect(p.arraySync()).toEqual([-5, 10]);
    for (const t of [a, b, p]) t.dispose();
  });

  it("fills zeros and ones as real values", async () => {
    const { ones, zeros } = await import("./factory.js");
    const o = ones([2], "complex64");
    const z = zeros([1], "complex64");
    expect(o.arraySync()).toEqual([1, 0, 1, 0]);
    expect(z.arraySync()).toEqual([0, 0]);
    o.dispose();
    z.dispose();
  });

  it("rejects a nested number list", async () => {
    const { tensor } = await import("./factory.js");
    expect(() => tensor([1, 2], [2], "complex64")).toThrow(/complex\(real, imag\)/);
  });
});
