// Exercises the graph-construction primitives the training path is built on
// (M4a): placeholders, baked constants, ops with attrs, and session execution.
// Skipped unless the native addon is built.

import { describe, expect, it } from "bun:test";

let nativeAvailable = false;
try {
  await import("../backend/native.js");
  nativeAvailable = true;
} catch {
  nativeAvailable = false;
}

describe.skipIf(!nativeAvailable)("graph construction (native)", () => {
  it("builds placeholder * const and runs it in a session", async () => {
    const { binding } = await import("../backend/native.js");
    const { DType } = await import("../backend/dtype.js");
    const { tensor, tensorHandle } = await import("./test-helpers.js");

    const graph = binding.graphCreate();
    const x = binding.graphPlaceholder(graph, "x", DType.float32, [2]);

    const two = tensor([2, 2]);
    const constPort = binding.graphConst(graph, "two", tensorHandle(two));

    const [product] = binding.graphAddOp(
      graph,
      "Mul",
      "product",
      [x, constPort],
      { T: { type: "type", value: DType.float32 } },
      1,
    );
    expect(product).toEqual({ op: "product", index: 0 });

    const session = binding.graphNewSession(graph);
    try {
      const feed = tensor([3, 4]);
      const [out] = binding.sessionRun(
        session,
        [x.op],
        [x.index],
        [tensorHandle(feed)],
        [product?.op as string],
        [product?.index as number],
      );
      const { Tensor } = await import("../tensor/tensor.js");
      const result = new Tensor(out as never, binding.handleShape(out as never), "float32");
      expect(await result.array()).toEqual([6, 8]);
    } finally {
      binding.sessionDelete(session);
    }
  });

  it("reports a helpful error for an unknown operation name", async () => {
    const { binding } = await import("../backend/native.js");
    const { DType } = await import("../backend/dtype.js");
    const graph = binding.graphCreate();
    binding.graphPlaceholder(graph, "x", DType.float32, [1]);
    expect(() =>
      binding.graphAddOp(graph, "Mul", "bad", [{ op: "missing", index: 0 }], {}, 1),
    ).toThrow(/no operation named 'missing'/);
  });
});
