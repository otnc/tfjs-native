// Gradients via record-and-replay (M4c).
//
// `fn` runs eagerly while a tape records every op. The recording is then rebuilt
// as a TF_Graph — inputs we differentiate become placeholders, other leaves are
// baked in as constants — and TF_AddGradients differentiates it. This reuses
// TensorFlow's own gradient definitions instead of hand-writing them.

import { DType, dtypeName } from "../backend/dtype.js";
import { binding, type Port } from "../backend/native.js";
import { Tensor, tensorHandle } from "../tensor/tensor.js";
import { type TapeEntry, withTape } from "./tape.js";

export interface ValueAndGrads {
  /** The value `fn` returned. */
  value: Tensor;
  /** One gradient per entry of `xs`, in the same order. */
  grads: Tensor[];
}

function toTensor(handle: ReturnType<typeof binding.sessionRun>[number]): Tensor {
  return new Tensor(handle, binding.handleShape(handle), dtypeName(binding.handleDtype(handle)));
}

function computeGrads(loss: Tensor, xs: Tensor[], entries: TapeEntry[]): Tensor[] {
  const graph = binding.graphCreate();
  const ports = new WeakMap<Tensor, Port>();
  let counter = 0;
  const nextName = (prefix: string) => `${prefix}_${counter++}`;

  // Inputs we differentiate become placeholders so their values are fed in.
  const feeds: { port: Port; tensor: Tensor }[] = [];
  for (const x of xs) {
    const port = binding.graphPlaceholder(graph, nextName("x"), DType[x.dtype], [...x.shape]);
    ports.set(x, port);
    feeds.push({ port, tensor: x });
  }

  // Any other leaf is a constant of the traced computation; bake its value in.
  const portFor = (tensor: Tensor): Port => {
    const known = ports.get(tensor);
    if (known !== undefined) return known;
    const port = binding.graphConst(graph, nextName("const"), tensorHandle(tensor));
    ports.set(tensor, port);
    return port;
  };

  for (const entry of entries) {
    const inputs = entry.inputs.map(portFor);
    const outputs = binding.graphAddOp(
      graph,
      entry.op,
      nextName(entry.op.toLowerCase()),
      inputs,
      entry.attrs,
      entry.outputs.length,
    );
    entry.outputs.forEach((tensor, i) => {
      ports.set(tensor, outputs[i] as Port);
    });
  }

  const lossPort = ports.get(loss);
  if (lossPort === undefined) {
    throw new Error(
      "tfjs-native: the value to differentiate was not produced by an op inside the function",
    );
  }

  const xPorts = xs.map((x) => ports.get(x) as Port);

  // TF_AddGradients has three outcomes for a requested input:
  //   - a gradient node (normal),
  //   - null, when the op is marked non-differentiable (REGISTER_NO_GRADIENT_OP),
  //   - a thrown Status when the input is unreachable or an op has no gradient.
  // All of "no usable gradient" should surface clearly, not as silent zeros.
  let gradPorts: (Port | null)[];
  try {
    gradPorts = binding.graphAddGradients(graph, [lossPort], xPorts);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `tfjs-native: cannot compute a gradient — ${message}. Every differentiated input must affect the result through differentiable ops.`,
    );
  }
  gradPorts.forEach((port, i) => {
    if (port === null) {
      throw new Error(
        `tfjs-native: no gradient for input ${i}: the path to the result crosses a non-differentiable op`,
      );
    }
  });

  const session = binding.graphNewSession(graph);
  try {
    const handles = binding.sessionRun(
      session,
      feeds.map((f) => f.port.op),
      feeds.map((f) => f.port.index),
      feeds.map((f) => tensorHandle(f.tensor)),
      gradPorts.map((p) => (p as Port).op),
      gradPorts.map((p) => (p as Port).index),
    );
    return handles.map(toTensor);
  } finally {
    binding.sessionDelete(session);
  }
}

/** Runs `fn`, returning both its value and the gradients w.r.t. `xs`. */
export function valueAndGrads(fn: () => Tensor, xs: Tensor[]): ValueAndGrads {
  const { result, entries, temporaries } = withTape(fn);
  try {
    return { value: result, grads: computeGrads(result, xs, entries) };
  } finally {
    for (const temporary of temporaries) temporary.dispose();
  }
}

/** Gradients of `fn`'s result with respect to `xs`. */
export function grads(fn: () => Tensor, xs: Tensor[]): Tensor[] {
  return valueAndGrads(fn, xs).grads;
}
