// Gradient tests (M4c). Each expectation is checked against the analytic
// derivative. Skipped unless the native addon is built.

import { describe, expect, it } from "bun:test";

let nativeAvailable = false;
try {
  await import("../backend/native.js");
  nativeAvailable = true;
} catch {
  nativeAvailable = false;
}

describe.skipIf(!nativeAvailable)("gradients (native)", () => {
  it("d/dx sum(x^2) = 2x", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { square } = await import("../ops/manual/elementwise.js");
    const { sum } = await import("../ops/manual/reduction.js");
    const { grads } = await import("./gradients.js");

    const x = tensor([1, 2, 3]);
    const [dx] = grads(() => sum(square(x)), [x]);
    expect(await dx?.array()).toEqual([2, 4, 6]);
  });

  it("d/dx sum(x * y) = y, and d/dy = x", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { mul } = await import("../ops/manual/elementwise.js");
    const { sum } = await import("../ops/manual/reduction.js");
    const { grads } = await import("./gradients.js");

    const x = tensor([1, 2, 3]);
    const y = tensor([10, 20, 30]);
    const [dx, dy] = grads(() => sum(mul(x, y)), [x, y]);
    expect(await dx?.array()).toEqual([10, 20, 30]);
    expect(await dy?.array()).toEqual([1, 2, 3]);
  });

  it("valueAndGrads returns the traced value too", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { square } = await import("../ops/manual/elementwise.js");
    const { sum } = await import("../ops/manual/reduction.js");
    const { valueAndGrads } = await import("./gradients.js");

    const x = tensor([3, 4]);
    const { value, grads: g } = valueAndGrads(() => sum(square(x)), [x]);
    expect(await value.array()).toBe(25); // 9 + 16
    expect(await g[0]?.array()).toEqual([6, 8]);
  });

  it("differentiates a matMul + bias graph", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { add } = await import("../ops/manual/elementwise.js");
    const { matMul } = await import("../ops/manual/linalg.js");
    const { sum } = await import("../ops/manual/reduction.js");
    const { grads } = await import("./gradients.js");

    const x = tensor([[1, 2]]); // 1x2
    const w = tensor([[3], [4]]); // 2x1
    const b = tensor([5]);

    // loss = sum(x @ w + b); d/dw = x^T (summed), d/db = 1
    const [dw, db] = grads(() => sum(add(matMul(x, w), b)), [w, b]);
    expect(await dw?.array()).toEqual([[1], [2]]);
    expect(await db?.array()).toEqual([1]);
  });

  it("gradients flow through relu", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { relu } = await import("../ops/manual/elementwise.js");
    const { sum } = await import("../ops/manual/reduction.js");
    const { grads } = await import("./gradients.js");

    const x = tensor([-2, 0.5, 3]);
    const [dx] = grads(() => sum(relu(x)), [x]);
    // relu' is 0 below zero and 1 above it.
    const g = (await dx?.array()) as number[];
    expect(g[0]).toBe(0);
    expect(g[2]).toBe(1);
  });

  it("reports an input that does not affect the result", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { square } = await import("../ops/manual/elementwise.js");
    const { sum } = await import("../ops/manual/reduction.js");
    const { grads } = await import("./gradients.js");

    const x = tensor([1, 2]);
    const unrelated = tensor([1, 2]);
    // TensorFlow raises "unreachable" for an input the loss does not depend on.
    expect(() => grads(() => sum(square(x)), [unrelated])).toThrow(/cannot compute a gradient/);
  });

  it("reports a non-differentiable op in the path", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { sum } = await import("../ops/manual/reduction.js");
    const { floor } = await import("../ops/generated/index.js");
    const { grads } = await import("./gradients.js");

    const x = tensor([1.5, 2.5]);
    // floor is REGISTER_NO_GRADIENT_OP -> the native call returns null, not a throw.
    expect(() => grads(() => sum(floor(x)), [x])).toThrow(/no gradient for input 0/);
  });

  it("does not record when no tape is active", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { square } = await import("../ops/manual/elementwise.js");
    const { isRecording } = await import("./tape.js");

    expect(isRecording()).toBe(false);
    const out = square(tensor([2]));
    expect(await out.array()).toEqual([4]);
    expect(isRecording()).toBe(false);
  });
});
