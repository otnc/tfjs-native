// Optimizers (M4d).
//
// Gradients come from the record-and-replay tape; the update rules themselves are
// plain eager ops, run outside any tape (so they are not recorded). Per-variable
// state (momentum, Adam moments) is kept as Tensors and freed on dispose().

import { add, div, mul, sqrt, square, sub } from "../ops/manual/elementwise.js";
import { tidy } from "../tensor/engine.js";
import { scalar, zeros } from "../tensor/factory.js";
import type { Tensor } from "../tensor/tensor.js";
import { valueAndGrads } from "./gradients.js";
import type { Variable } from "./variable.js";

export abstract class Optimizer {
  protected step = 0;

  /** Differentiates `fn` w.r.t. `vars`, applies the update, and returns the loss. */
  minimize(fn: () => Tensor, vars: Variable[]): Tensor {
    const { value, grads } = valueAndGrads(
      fn,
      vars.map((v) => v.value),
    );
    this.applyGradients(vars, grads);
    for (const g of grads) g.dispose();
    return value;
  }

  /** Applies precomputed gradients (one per variable, same order). */
  applyGradients(vars: Variable[], grads: Tensor[]): void {
    this.step += 1;
    vars.forEach((v, i) => {
      this.update(v, grads[i] as Tensor);
    });
  }

  protected abstract update(variable: Variable, grad: Tensor): void;

  /** Frees any per-variable optimizer state. */
  dispose(): void {}
}

class Sgd extends Optimizer {
  readonly #velocities = new Map<Variable, Tensor>();

  constructor(
    private readonly learningRate: number,
    private readonly momentum: number,
  ) {
    super();
  }

  protected update(variable: Variable, grad: Tensor): void {
    if (this.momentum === 0) {
      variable.assign(tidy(() => sub(variable.value, mul(grad, scalar(this.learningRate)))));
      return;
    }
    const previous = this.#velocities.get(variable);
    const { next, velocity } = tidy(() => {
      const prior = previous ?? zeros([...grad.shape], grad.dtype);
      const updated = add(mul(prior, scalar(this.momentum)), grad);
      return {
        velocity: updated,
        next: sub(variable.value, mul(updated, scalar(this.learningRate))),
      };
    });
    previous?.dispose();
    this.#velocities.set(variable, velocity);
    variable.assign(next);
  }

  override dispose(): void {
    for (const v of this.#velocities.values()) v.dispose();
    this.#velocities.clear();
  }
}

class Adam extends Optimizer {
  readonly #m = new Map<Variable, Tensor>();
  readonly #v = new Map<Variable, Tensor>();

  constructor(
    private readonly learningRate: number,
    private readonly beta1: number,
    private readonly beta2: number,
    private readonly epsilon: number,
  ) {
    super();
  }

  protected update(variable: Variable, grad: Tensor): void {
    const priorM = this.#m.get(variable);
    const priorV = this.#v.get(variable);
    const biasM = 1 - this.beta1 ** this.step;
    const biasV = 1 - this.beta2 ** this.step;

    const { next, m, v } = tidy(() => {
      const m0 = priorM ?? zeros([...grad.shape], grad.dtype);
      const v0 = priorV ?? zeros([...grad.shape], grad.dtype);
      // m = b1*m + (1-b1)*g ; v = b2*v + (1-b2)*g^2
      const mNext = add(mul(m0, scalar(this.beta1)), mul(grad, scalar(1 - this.beta1)));
      const vNext = add(mul(v0, scalar(this.beta2)), mul(square(grad), scalar(1 - this.beta2)));
      // w -= lr * (m/biasM) / (sqrt(v/biasV) + eps)
      const mHat = div(mNext, scalar(biasM));
      const vHat = div(vNext, scalar(biasV));
      const stepDir = div(mHat, add(sqrt(vHat), scalar(this.epsilon)));
      return {
        m: mNext,
        v: vNext,
        next: sub(variable.value, mul(stepDir, scalar(this.learningRate))),
      };
    });

    priorM?.dispose();
    priorV?.dispose();
    this.#m.set(variable, m);
    this.#v.set(variable, v);
    variable.assign(next);
  }

  override dispose(): void {
    for (const t of this.#m.values()) t.dispose();
    for (const t of this.#v.values()) t.dispose();
    this.#m.clear();
    this.#v.clear();
  }
}

class RmsProp extends Optimizer {
  readonly #ms = new Map<Variable, Tensor>();

  constructor(
    private readonly learningRate: number,
    private readonly decay: number,
    private readonly epsilon: number,
  ) {
    super();
  }

  protected update(variable: Variable, grad: Tensor): void {
    const prior = this.#ms.get(variable);
    const { next, meanSquare } = tidy(() => {
      const ms0 = prior ?? zeros([...grad.shape], grad.dtype);
      // ms = decay*ms + (1-decay)*g^2 ; w -= lr * g / (sqrt(ms) + eps)
      const ms = add(mul(ms0, scalar(this.decay)), mul(square(grad), scalar(1 - this.decay)));
      const stepDir = div(grad, add(sqrt(ms), scalar(this.epsilon)));
      return { meanSquare: ms, next: sub(variable.value, mul(stepDir, scalar(this.learningRate))) };
    });
    prior?.dispose();
    this.#ms.set(variable, meanSquare);
    variable.assign(next);
  }

  override dispose(): void {
    for (const t of this.#ms.values()) t.dispose();
    this.#ms.clear();
  }
}

export interface SgdOptions {
  learningRate: number;
  momentum?: number;
}

export interface AdamOptions {
  learningRate?: number;
  beta1?: number;
  beta2?: number;
  epsilon?: number;
}

export interface RmsPropOptions {
  learningRate?: number;
  decay?: number;
  epsilon?: number;
}

/** Stochastic gradient descent, optionally with momentum. */
export function sgd(options: SgdOptions): Optimizer {
  return new Sgd(options.learningRate, options.momentum ?? 0);
}

/** Adam. */
export function adam(options: AdamOptions = {}): Optimizer {
  return new Adam(
    options.learningRate ?? 0.001,
    options.beta1 ?? 0.9,
    options.beta2 ?? 0.999,
    options.epsilon ?? 1e-7,
  );
}

/** RMSProp. */
export function rmsprop(options: RmsPropOptions = {}): Optimizer {
  return new RmsProp(options.learningRate ?? 0.001, options.decay ?? 0.9, options.epsilon ?? 1e-7);
}
