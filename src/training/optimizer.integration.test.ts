// Optimizer tests (M4d). Update rules are checked against a hand-computed step;
// an end-to-end linear regression checks that training actually converges.
// Skipped unless the native addon is built.

import { describe, expect, it } from "bun:test";

let nativeAvailable = false;
try {
  await import("../backend/native.js");
  nativeAvailable = true;
} catch {
  nativeAvailable = false;
}

describe.skipIf(!nativeAvailable)("optimizers (native)", () => {
  it("SGD steps w := w - lr * dLoss/dw", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { square } = await import("../ops/manual/elementwise.js");
    const { sum } = await import("../ops/manual/reduction.js");
    const { variable } = await import("./variable.js");
    const { sgd } = await import("./optimizer.js");

    // loss = sum(w^2), dLoss/dw = 2w. w0 = [1, 2], lr = 0.1 -> w1 = w0 - 0.1*2w0
    const w = variable(tensor([1, 2]));
    const opt = sgd({ learningRate: 0.1 });
    const loss = opt.minimize(() => sum(square(w.value)), [w]);

    expect(await loss.array()).toBe(5); // 1 + 4
    const updated = (await w.value.array()) as number[];
    expect(updated[0]).toBeCloseTo(0.8, 6);
    expect(updated[1]).toBeCloseTo(1.6, 6);
    opt.dispose();
    w.dispose();
  });

  it("SGD with momentum accumulates velocity", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { square } = await import("../ops/manual/elementwise.js");
    const { sum } = await import("../ops/manual/reduction.js");
    const { variable } = await import("./variable.js");
    const { sgd } = await import("./optimizer.js");

    const w = variable(tensor([1]));
    const opt = sgd({ learningRate: 0.1, momentum: 0.9 });
    const fn = () => sum(square(w.value));

    // step 1: g=2, v=2,      w = 1 - 0.1*2   = 0.8
    opt.minimize(fn, [w]).dispose();
    expect(((await w.value.array()) as number[])[0]).toBeCloseTo(0.8, 6);
    // step 2: g=1.6, v=0.9*2+1.6=3.4, w = 0.8 - 0.1*3.4 = 0.46
    opt.minimize(fn, [w]).dispose();
    const w2 = ((await w.value.array()) as number[])[0] as number;
    expect(w2).toBeCloseTo(0.46, 6);
    opt.dispose();
    w.dispose();
  });

  it("Adam takes a first step of about -lr in the gradient direction", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { square } = await import("../ops/manual/elementwise.js");
    const { sum } = await import("../ops/manual/reduction.js");
    const { variable } = await import("./variable.js");
    const { adam } = await import("./optimizer.js");

    // On the first step Adam's bias-corrected step is lr * sign(g), so w moves
    // by ~-lr toward the minimum regardless of the gradient magnitude.
    const w = variable(tensor([5]));
    const opt = adam({ learningRate: 0.1 });
    opt.minimize(() => sum(square(w.value)), [w]).dispose();
    const moved = 5 - (((await w.value.array()) as number[])[0] as number);
    expect(moved).toBeCloseTo(0.1, 4);
    opt.dispose();
    w.dispose();
  });

  it("trains a linear regression to convergence (E2E)", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { add, sub, mul } = await import("../ops/manual/elementwise.js");
    const { mean } = await import("../ops/manual/reduction.js");
    const { variable } = await import("./variable.js");
    const { sgd } = await import("./optimizer.js");

    // Fit y = 3x + 2 from data, starting far from the solution.
    const xs = tensor([0, 1, 2, 3, 4]);
    const ys = tensor([2, 5, 8, 11, 14]);
    const w = variable(tensor([0]));
    const b = variable(tensor([0]));
    const opt = sgd({ learningRate: 0.05 });

    const lossFn = () => {
      const pred = add(mul(w.value, xs), b.value); // w*x + b (scalars broadcast)
      const err = sub(pred, ys);
      return mean(square(err));
    };
    const { square } = await import("../ops/manual/elementwise.js");

    let first = 0;
    let last = 0;
    for (let i = 0; i < 400; i++) {
      const loss = opt.minimize(lossFn, [w, b]);
      const value = (await loss.array()) as number;
      if (i === 0) first = value;
      last = value;
      loss.dispose();
    }

    expect(last).toBeLessThan(first); // loss decreased
    expect(last).toBeLessThan(0.01); // and converged
    expect(((await w.value.array()) as number[])[0]).toBeCloseTo(3, 1);
    expect(((await b.value.array()) as number[])[0]).toBeCloseTo(2, 1);

    opt.dispose();
    w.dispose();
    b.dispose();
  }, 30000);

  it("reuses one compiled graph across steps (fast loop)", async () => {
    const { tensor } = await import("../tensor/factory.js");
    const { square } = await import("../ops/manual/elementwise.js");
    const { sum } = await import("../ops/manual/reduction.js");
    const { variable } = await import("./variable.js");
    const { sgd } = await import("./optimizer.js");

    const w = variable(tensor([1, 2, 3, 4]));
    const opt = sgd({ learningRate: 0.01 });
    const loss = () => sum(square(w.value));

    // Many steps must stay well under a graph-rebuild-per-step budget.
    const start = performance.now();
    for (let i = 0; i < 500; i++) opt.minimize(loss, [w]).dispose();
    const elapsed = performance.now() - start;

    expect(((await w.value.array()) as number[])[0]).toBeLessThan(1); // it trained
    expect(elapsed).toBeLessThan(2000); // 500 steps without recompiling each time
    opt.dispose();
    w.dispose();
  }, 30000);
});
