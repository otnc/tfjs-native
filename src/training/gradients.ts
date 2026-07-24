// Gradients via record-and-replay (M4c) with a reusable compiled graph (M4e).
//
// `fn` runs eagerly while a tape records every op. The recording is rebuilt as a
// TF_Graph — inputs we differentiate become placeholders, other leaves are baked
// in as constants — and TF_AddGradients differentiates it, reusing TensorFlow's
// own gradient definitions. The compiled graph can be run again with new input
// values without re-tracing, which is what makes a training loop cheap.

import { DType, dtypeName } from "../backend/dtype.js";
import { binding, type Port, type Session } from "../backend/native.js";
import { Tensor, tensorHandle } from "../tensor/tensor.js";
import { withTape } from "./tape.js";

export interface ValueAndGrads {
  /** The value `fn` returned. */
  value: Tensor;
  /** One gradient per entry of `xs`, in the same order. */
  grads: Tensor[];
}

/** A traced loss graph plus a live session, runnable with fresh input values. */
export interface CompiledGrads {
  session: Session;
  /** Placeholders for the differentiated inputs, in `xs` order. */
  xPorts: Port[];
  lossPort: Port;
  /** Gradient outputs, in `xs` order. */
  gradPorts: Port[];
}

function toTensor(handle: ReturnType<typeof binding.sessionRun>[number]): Tensor {
  return new Tensor(handle, binding.handleShape(handle), dtypeName(binding.handleDtype(handle)));
}

/** A cache key over the shapes/dtypes of the differentiated inputs. */
export function gradSignature(xs: Tensor[]): string {
  return xs.map((x) => `${x.dtype}[${x.shape.join(",")}]`).join(";");
}

/** Traces `fn`, rebuilds it as a graph, and returns a runnable compiled graph. */
export function compileGrads(fn: () => Tensor, xs: Tensor[]): CompiledGrads {
  const { result, entries, temporaries } = withTape(fn);
  const graph = binding.graphCreate();
  const ports = new WeakMap<Tensor, Port>();
  let counter = 0;
  const nextName = (prefix: string) => `${prefix}_${counter++}`;

  // Differentiated inputs become placeholders so values are fed in each run.
  const xPorts: Port[] = [];
  for (const x of xs) {
    const port = binding.graphPlaceholder(graph, nextName("x"), DType[x.dtype], [...x.shape]);
    ports.set(x, port);
    xPorts.push(port);
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

  const lossPort = ports.get(result);
  if (lossPort === undefined) {
    throw new Error(
      "tfjs-native: the value to differentiate was not produced by an op inside the function",
    );
  }

  // TF_AddGradients has three outcomes for a requested input: a gradient node; a
  // null (op marked non-differentiable via REGISTER_NO_GRADIENT_OP); or a thrown
  // Status (unreachable input, or an op with no gradient). Surface all failures.
  let gradPortsRaw: (Port | null)[];
  try {
    gradPortsRaw = binding.graphAddGradients(graph, [lossPort], xPorts);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `tfjs-native: cannot compute a gradient — ${message}. Every differentiated input must affect the result through differentiable ops.`,
    );
  }
  const gradPorts = gradPortsRaw.map((port, i) => {
    if (port === null) {
      throw new Error(
        `tfjs-native: no gradient for input ${i}: the path to the result crosses a non-differentiable op`,
      );
    }
    return port;
  });

  const session = binding.graphNewSession(graph);

  // The eager tensors produced while tracing were only needed to record the
  // structure and to bake constant values; free them now. Leaves (xs, external
  // inputs) are left alone — they are owned by the caller.
  for (const temporary of temporaries) temporary.dispose();
  for (const entry of entries) {
    for (const output of entry.outputs) output.dispose();
  }

  return { session, xPorts, lossPort, gradPorts };
}

/** Runs a compiled graph with fresh input values, returning value and gradients. */
export function runGrads(compiled: CompiledGrads, xs: Tensor[]): ValueAndGrads {
  const fetch = [compiled.lossPort, ...compiled.gradPorts];
  const handles = binding.sessionRun(
    compiled.session,
    compiled.xPorts.map((p) => p.op),
    compiled.xPorts.map((p) => p.index),
    xs.map(tensorHandle),
    fetch.map((p) => p.op),
    fetch.map((p) => p.index),
  );
  const tensors = handles.map(toTensor);
  return { value: tensors[0] as Tensor, grads: tensors.slice(1) };
}

/** Frees a compiled graph's session. */
export function disposeCompiled(compiled: CompiledGrads): void {
  binding.sessionDelete(compiled.session);
}

/** Runs `fn`, returning both its value and the gradients w.r.t. `xs`. */
export function valueAndGrads(fn: () => Tensor, xs: Tensor[]): ValueAndGrads {
  const compiled = compileGrads(fn, xs);
  try {
    return runGrads(compiled, xs);
  } finally {
    disposeCompiled(compiled);
  }
}

/** Gradients of `fn`'s result with respect to `xs`. */
export function grads(fn: () => Tensor, xs: Tensor[]): Tensor[] {
  return valueAndGrads(fn, xs).grads;
}
