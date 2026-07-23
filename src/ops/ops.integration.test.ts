// Integration tests for the eager op layer (M2a). Skipped unless the native
// addon is built (needs libtensorflow + a C++ toolchain), like the tensor tests.

import { describe, expect, it } from "bun:test";

let nativeAvailable = false;
try {
  await import("../backend/native.js");
  nativeAvailable = true;
} catch {
  nativeAvailable = false;
}

describe.skipIf(!nativeAvailable)("eager ops (native)", () => {
  it("element-wise binary ops", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { add, sub, mul, div, maximum } = await import("./manual/elementwise.js");
    const a = tensor([1, 2, 3]);
    const b = tensor([4, 5, 6]);
    expect(await add(a, b).array()).toEqual([5, 7, 9]);
    expect(await sub(b, a).array()).toEqual([3, 3, 3]);
    expect(await mul(a, b).array()).toEqual([4, 10, 18]);
    expect(await div(tensor([6, 9]), tensor([2, 3])).array()).toEqual([3, 3]);
    expect(await maximum(a, tensor([2, 2, 2])).array()).toEqual([2, 2, 3]);
  });

  it("element-wise unary ops", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { neg, relu, square, exp } = await import("./manual/elementwise.js");
    expect(await neg(tensor([1, -2, 3])).array()).toEqual([-1, 2, -3]);
    expect(await relu(tensor([-1, 0, 2])).array()).toEqual([0, 0, 2]);
    expect(await square(tensor([2, 3])).array()).toEqual([4, 9]);
    const e = (await exp(tensor([0, 1])).array()) as number[];
    expect(e[0]).toBeCloseTo(1);
    expect(e[1]).toBeCloseTo(Math.E);
  });

  it("matMul", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { matMul } = await import("./manual/linalg.js");
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = tensor([
      [5, 6],
      [7, 8],
    ]);
    expect(await matMul(a, b).array()).toEqual([
      [19, 22],
      [43, 50],
    ]);
  });

  it("reductions", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { sum, mean } = await import("./manual/reduction.js");
    const x = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(await sum(x).array()).toBe(10); // all axes -> scalar
    expect(await sum(x, 0).array()).toEqual([4, 6]);
    expect(await sum(x, 1).array()).toEqual([3, 7]);
    expect(await mean(tensor([1, 2, 3, 4])).array()).toBe(2.5);
  });

  it("reshape / transpose / cast", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { reshape, transpose, cast } = await import("./manual/array-ops.js");
    expect(await reshape(tensor([1, 2, 3, 4, 5, 6]), [2, 3]).array()).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const t = transpose(
      tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]),
    );
    expect(t.shape).toEqual([3, 2]);
    expect(await t.array()).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
    const c = cast(tensor([1.7, 2.9]), "int32");
    expect(c.dtype).toBe("int32");
    expect(await c.array()).toEqual([1, 2]);
  });

  it("output shape/dtype come from the result handle", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { matMul } = await import("./manual/linalg.js");
    const out = matMul(tensor([[1, 2, 3]]), tensor([[1], [1], [1]]));
    expect(out.shape).toEqual([1, 1]);
    expect(out.dtype).toBe("float32");
    expect(await out.array()).toEqual([[6]]);
  });

  it("runOp escape hatch calls arbitrary ops by name", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { runOp1 } = await import("./run.js");
    const { DType } = await import("../backend/dtype.js");
    const out = runOp1("Sqrt", [tensor([4, 9, 16])], { T: { type: "type", value: DType.float32 } });
    expect(await out.array()).toEqual([2, 3, 4]);
  });

  it("generated: unary ops", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { floor, sign, rsqrt } = await import("./generated/index.js");
    expect(await floor(tensor([1.7, 2.2, -0.5])).array()).toEqual([1, 2, -1]);
    expect(await sign(tensor([-3, 0, 5])).array()).toEqual([-1, 0, 1]);
    const r = (await rsqrt(tensor([4, 16])).array()) as number[];
    expect(r[0]).toBeCloseTo(0.5);
    expect(r[1]).toBeCloseTo(0.25);
  });

  it("generated: addN takes a variadic Tensor[] input", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { addN } = await import("./generated/index.js");
    const out = addN([tensor([1, 2]), tensor([3, 4]), tensor([5, 6])]);
    expect(await out.array()).toEqual([9, 12]);
  });

  it("generated: softmax", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { softmax } = await import("./generated/index.js");
    const s = (await softmax(tensor([[0, 0]])).array()) as number[][];
    expect(s[0]?.[0]).toBeCloseTo(0.5);
    expect(s[0]?.[1]).toBeCloseTo(0.5);
  });

  it("generated: op with an options attr (leakyRelu alpha)", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { leakyRelu } = await import("./generated/index.js");
    const out = (await leakyRelu(tensor([-10, 5]), { alpha: 0.1 }).array()) as number[];
    expect(out[0]).toBeCloseTo(-1);
    expect(out[1]).toBeCloseTo(5);
  });

  it("generated: scatterNd (an op tfjs lacks)", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { scatterNd } = await import("./generated/index.js");
    const indices = tensor([[0], [2]], [2, 1], "int32");
    const updates = tensor([10, 20]);
    const shape = tensor([4], [1], "int32");
    expect(await scatterNd(indices, updates, shape).array()).toEqual([10, 0, 20, 0]);
  });

  it("tidy disposes op intermediates", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { tidy } = await import("../tensor/engine.js");
    const { add, mul } = await import("./manual/elementwise.js");
    const a = tensor([1, 2]);
    const b = tensor([3, 4]);
    let mid: import("../tensor/tensor.js").Tensor | undefined;
    const out = tidy(() => {
      mid = add(a, b);
      return mul(mid, mid);
    });
    expect(mid?.isDisposed).toBe(true);
    expect(out.isDisposed).toBe(false);
    expect(await out.array()).toEqual([16, 36]);
  });
});
