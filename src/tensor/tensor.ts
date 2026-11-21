// The Tensor class: a thin, typed wrapper around a native TFE_TensorHandle.
// It owns exactly one handle and reads data back on demand via the addon.

import type { DTypeName } from "../backend/dtype.js";
import { binding, type Handle } from "../backend/native.js";
import { decodeStrings, readValues, viewBytesAs } from "./convert.js";
import { track, untrack } from "./engine.js";
import { sizeFromShape, toNested } from "./shape.js";
import type { NestedList, TensorScalar } from "./types.js";

// The native handle is kept out of the public shape entirely: it lives in a
// module-private WeakMap, readable within the package via `tensorHandle()`.
const HANDLES = new WeakMap<Tensor, Handle>();

export class Tensor {
  readonly shape: readonly number[];
  readonly dtype: DTypeName;
  readonly size: number;

  /** Wraps an existing native handle. Prefer the factories (tensor/zeros/...). */
  constructor(handle: Handle, shape: readonly number[], dtype: DTypeName) {
    HANDLES.set(this, handle);
    this.shape = shape;
    this.dtype = dtype;
    this.size = sizeFromShape(shape);
    track(this);
  }

  /** Number of dimensions. */
  get rank(): number {
    return this.shape.length;
  }

  /** True once the underlying handle has been freed. */
  get isDisposed(): boolean {
    return !HANDLES.has(this);
  }

  /**
   * Reads the tensor's data synchronously as its dtype's TypedArray view.
   * complex64 comes back as interleaved [re, im] pairs (length `2 * size`).
   * `string` tensors have no TypedArray form: use {@link bytesSync} or
   * {@link arraySync} instead.
   */
  dataSync(): ArrayBufferView {
    if (this.dtype === "string") {
      throw new Error(
        "tfjs-native: dataSync() is not defined for 'string'; use bytesSync() or arraySync()",
      );
    }
    const bytes = binding.handleData(tensorHandle(this));
    return viewBytesAs(bytes, this.dtype, this.size);
  }

  /** Reads the tensor's data as its dtype's TypedArray view. */
  async data(): Promise<ArrayBufferView> {
    // Synchronous under the hood for now; a threaded path can replace this
    // without changing callers.
    return this.dataSync();
  }

  /** Reads a `string` tensor's elements as raw bytes, flat in row-major order. */
  bytesSync(): Uint8Array[] {
    if (this.dtype !== "string") {
      throw new Error(`tfjs-native: bytesSync() needs a 'string' tensor, got '${this.dtype}'`);
    }
    return binding.handleStrings(tensorHandle(this));
  }

  /** Async variant of {@link bytesSync}. */
  async bytes(): Promise<Uint8Array[]> {
    return this.bytesSync();
  }

  /**
   * Reads the data back as a nested JS array shaped like `this.shape`.
   * `string` elements are decoded as UTF-8. complex64 follows tfjs: the
   * innermost dimension holds interleaved [re, im] pairs, so it is twice as long.
   */
  arraySync(): NestedList {
    if (this.dtype === "string") {
      return toNested(decodeStrings(this.bytesSync()), this.shape) as NestedList;
    }
    const values: TensorScalar[] = readValues(this.dataSync(), this.dtype);
    if (this.dtype === "complex64") {
      return toNested(values, interleavedShape(this.shape)) as NestedList;
    }
    return toNested(values, this.shape) as NestedList;
  }

  /** Async variant of {@link arraySync}. */
  async array(): Promise<NestedList> {
    return this.arraySync();
  }

  /** Eagerly frees the native handle. Idempotent. */
  dispose(): void {
    const handle = HANDLES.get(this);
    if (handle === undefined) return;
    binding.deleteHandle(handle);
    HANDLES.delete(this);
    untrack(this);
  }
}

/**
 * Package-internal accessor for a tensor's native handle. Throws if disposed.
 * Not part of the public API — used by the ops layer to pass inputs to the addon.
 */
export function tensorHandle(tensor: Tensor): Handle {
  const handle = HANDLES.get(tensor);
  if (handle === undefined) {
    throw new Error("tfjs-native: tensor is already disposed");
  }
  return handle;
}

/** The shape of a complex64 tensor's interleaved data: innermost dim doubled. */
function interleavedShape(shape: readonly number[]): number[] {
  if (shape.length === 0) return [2];
  const out = [...shape];
  out[out.length - 1] = 2 * (out[out.length - 1] as number);
  return out;
}
