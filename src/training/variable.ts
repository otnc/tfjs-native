// A trainable variable: a mutable holder around an immutable Tensor.
//
// Tensors own a native handle and never change, so training needs a box whose
// value can be replaced in place. `grads`/`minimize` differentiate with respect
// to a Variable's current value.

import type { Tensor } from "../tensor/tensor.js";

export class Variable {
  #value: Tensor;
  readonly name: string;

  constructor(initial: Tensor, name = "variable") {
    this.#value = initial;
    this.name = name;
  }

  /** The current value. Do not dispose it directly — use `assign` or `dispose`. */
  get value(): Tensor {
    return this.#value;
  }

  get shape(): readonly number[] {
    return this.#value.shape;
  }

  /** Replaces the value, disposing the previous one. */
  assign(next: Tensor): void {
    if (next === this.#value) return;
    const previous = this.#value;
    this.#value = next;
    previous.dispose();
  }

  /** Frees the underlying tensor. */
  dispose(): void {
    this.#value.dispose();
  }
}

/** Creates a trainable variable from an initial tensor. */
export function variable(initial: Tensor, name?: string): Variable {
  return new Variable(initial, name);
}
