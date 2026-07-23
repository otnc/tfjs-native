// Integration tests for SavedModel loading and inference (M3). Skipped unless
// both the native addon is built and the test fixture exists. Regenerate the
// fixture with scripts/fixtures/make_saved_model.py (see CONTRIBUTING).

import { describe, expect, it } from "bun:test";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const FIXTURE = fileURLToPath(new URL("../../test/fixtures/times_two", import.meta.url));

let nativeAvailable = false;
try {
  await import("../backend/native.js");
  nativeAvailable = true;
} catch {
  nativeAvailable = false;
}
const ready = nativeAvailable && existsSync(FIXTURE);

describe.skipIf(!ready)("SavedModel (native)", () => {
  it("exposes the serving signature and its keys", async () => {
    const { loadSavedModel } = await import("./saved-model.js");
    const model = loadSavedModel(FIXTURE);
    try {
      expect(model.signatureNames).toContain("serving_default");
      const sig = model.signatures.serving_default;
      expect(Object.keys(sig?.inputs ?? {})).toContain("x");
      expect(Object.keys(sig?.outputs ?? {})).toContain("y");
    } finally {
      model.dispose();
    }
  });

  it("runs inference through predict()", async () => {
    const { loadSavedModel } = await import("./saved-model.js");
    const { tensor } = await import("../tensor/factory.js");
    const model = loadSavedModel(FIXTURE);
    try {
      const out = model.predict({ x: tensor([1, 2, 3]) });
      expect(await out.y?.array()).toEqual([2, 4, 6]);
      expect(out.y?.dtype).toBe("float32");
      expect(out.y?.shape).toEqual([3]);
    } finally {
      model.dispose();
    }
  });

  it("runs a named signature and reports unknown ones", async () => {
    const { loadSavedModel } = await import("./saved-model.js");
    const { tensor } = await import("../tensor/factory.js");
    const model = loadSavedModel(FIXTURE);
    try {
      const out = model.run("serving_default", { x: tensor([0, -5]) });
      expect(await out.y?.array()).toEqual([0, -10]);
      expect(() => model.run("nope", { x: tensor([1]) })).toThrow(/no signature 'nope'/);
      expect(() => model.predict({ bogus: tensor([1]) })).toThrow(/no input 'bogus'/);
    } finally {
      model.dispose();
    }
  });

  it("dispose() is idempotent and blocks further runs", async () => {
    const { loadSavedModel } = await import("./saved-model.js");
    const { tensor } = await import("../tensor/factory.js");
    const model = loadSavedModel(FIXTURE);
    model.dispose();
    model.dispose();
    expect(model.isDisposed).toBe(true);
    expect(() => model.predict({ x: tensor([1]) })).toThrow(/already disposed/);
  });
});
