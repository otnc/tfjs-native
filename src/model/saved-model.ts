// SavedModel loading and inference (M3).
//
// The addon hands back a session plus the serialized MetaGraphDef; signatures are
// resolved here so callers work with signature keys ("x", "y") instead of raw
// graph tensor names ("serving_default_x:0").

import { dtypeName } from "../backend/dtype.js";
import { binding, type Session } from "../backend/native.js";
import { Tensor, tensorHandle } from "../tensor/tensor.js";
import { parseSignatures, parseTensorName, type SignatureDef } from "./proto.js";

/** The signature TensorFlow's `saved_model` CLI shows by default. */
export const DEFAULT_SIGNATURE = "serving_default";

export interface LoadSavedModelOptions {
  /** MetaGraph tags to select. Defaults to `["serve"]`. */
  tags?: string[];
}

const MODELS = new WeakMap<SavedModel, Session>();

export class SavedModel {
  /** Signature name -> its feed/fetch keys and graph tensor names. */
  readonly signatures: Readonly<Record<string, SignatureDef>>;

  constructor(model: Session, signatures: Record<string, SignatureDef>) {
    MODELS.set(this, model);
    this.signatures = signatures;
  }

  /** Names of every signature in the model (e.g. `["serving_default"]`). */
  get signatureNames(): string[] {
    return Object.keys(this.signatures);
  }

  /** True once the session has been closed. */
  get isDisposed(): boolean {
    return !MODELS.has(this);
  }

  /**
   * Runs one signature. `feeds` is keyed by the signature's input names; the
   * result is keyed by its output names.
   */
  run(signature: string, feeds: Record<string, Tensor>): Record<string, Tensor> {
    const model = MODELS.get(this);
    if (model === undefined) {
      throw new Error("tfjs-native: SavedModel is already disposed");
    }
    const sig = this.signatures[signature];
    if (sig === undefined) {
      throw new Error(
        `tfjs-native: no signature '${signature}' (have: ${this.signatureNames.join(", ")})`,
      );
    }

    const inputOps: string[] = [];
    const inputIndices: number[] = [];
    const inputHandles = [];
    for (const [key, tensor] of Object.entries(feeds)) {
      const name = sig.inputs[key];
      if (name === undefined) {
        throw new Error(
          `tfjs-native: signature '${signature}' has no input '${key}' (have: ${Object.keys(sig.inputs).join(", ")})`,
        );
      }
      const { op, index } = parseTensorName(name);
      inputOps.push(op);
      inputIndices.push(index);
      inputHandles.push(tensorHandle(tensor));
    }

    const outputKeys = Object.keys(sig.outputs);
    const outputOps: string[] = [];
    const outputIndices: number[] = [];
    for (const key of outputKeys) {
      const { op, index } = parseTensorName(sig.outputs[key] as string);
      outputOps.push(op);
      outputIndices.push(index);
    }

    const handles = binding.sessionRun(
      model,
      inputOps,
      inputIndices,
      inputHandles,
      outputOps,
      outputIndices,
    );

    const result: Record<string, Tensor> = {};
    handles.forEach((handle, i) => {
      result[outputKeys[i] as string] = new Tensor(
        handle,
        binding.handleShape(handle),
        dtypeName(binding.handleDtype(handle)),
      );
    });
    return result;
  }

  /** Runs the default serving signature. */
  predict(feeds: Record<string, Tensor>): Record<string, Tensor> {
    return this.run(DEFAULT_SIGNATURE, feeds);
  }

  /** Closes the session and frees the graph. Idempotent. */
  dispose(): void {
    const model = MODELS.get(this);
    if (model === undefined) return;
    binding.sessionDelete(model);
    MODELS.delete(this);
  }
}

/** Loads a SavedModel directory (the one containing `saved_model.pb`). */
export function loadSavedModel(dir: string, options: LoadSavedModelOptions = {}): SavedModel {
  const { handle, metaGraphDef } = binding.loadSavedModel(dir, options.tags ?? ["serve"]);
  return new SavedModel(handle, parseSignatures(metaGraphDef));
}
