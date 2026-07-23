// A tiny protobuf wire-format reader, just enough to pull signature definitions
// out of a serialized MetaGraphDef. Hand-rolled so the runtime keeps zero extra
// dependencies (protobufjs is dev-only, used by codegen).
//
// Fields we read (numbers must match tensorflow/core/protobuf/meta_graph.proto):
//   MetaGraphDef.signature_def = 5   map<string, SignatureDef>
//   SignatureDef.inputs        = 1   map<string, TensorInfo>
//   SignatureDef.outputs       = 2   map<string, TensorInfo>
//   TensorInfo.name            = 1   string
// A protobuf map entry is a message with key = 1 and value = 2.

const WIRE_VARINT = 0;
const WIRE_64BIT = 1;
const WIRE_LENGTH = 2;
const WIRE_32BIT = 5;

const decoder = new TextDecoder();

class Reader {
  #buf: Uint8Array;
  #pos = 0;

  constructor(buf: Uint8Array) {
    this.#buf = buf;
  }

  get eof(): boolean {
    return this.#pos >= this.#buf.length;
  }

  varint(): number {
    let result = 0;
    let scale = 1;
    for (;;) {
      const byte = this.#buf[this.#pos++];
      if (byte === undefined) {
        throw new Error("tfjs-native: truncated protobuf varint");
      }
      result += (byte & 0x7f) * scale;
      if ((byte & 0x80) === 0) return result;
      scale *= 128;
    }
  }

  /** Reads a field tag, returning its field number and wire type. */
  tag(): { field: number; wire: number } {
    const tag = this.varint();
    return { field: tag >>> 3, wire: tag & 7 };
  }

  bytes(): Uint8Array {
    const length = this.varint();
    const start = this.#pos;
    this.#pos += length;
    if (this.#pos > this.#buf.length) {
      throw new Error("tfjs-native: truncated protobuf length-delimited field");
    }
    return this.#buf.subarray(start, this.#pos);
  }

  string(): string {
    return decoder.decode(this.bytes());
  }

  /** Skips a field of the given wire type. */
  skip(wire: number): void {
    switch (wire) {
      case WIRE_VARINT:
        this.varint();
        return;
      case WIRE_64BIT:
        this.#pos += 8;
        return;
      case WIRE_LENGTH:
        this.bytes();
        return;
      case WIRE_32BIT:
        this.#pos += 4;
        return;
      default:
        throw new Error(`tfjs-native: unsupported protobuf wire type ${wire}`);
    }
  }
}

/** One signature: feed/fetch key -> graph tensor name ("op:index"). */
export interface SignatureDef {
  inputs: Record<string, string>;
  outputs: Record<string, string>;
}

function parseMapEntry(buf: Uint8Array): { key: string; value: Uint8Array } {
  const reader = new Reader(buf);
  let key = "";
  let value: Uint8Array = new Uint8Array(0);
  while (!reader.eof) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === WIRE_LENGTH) {
      key = reader.string();
    } else if (field === 2 && wire === WIRE_LENGTH) {
      value = reader.bytes();
    } else {
      reader.skip(wire);
    }
  }
  return { key, value };
}

function parseTensorInfoName(buf: Uint8Array): string {
  const reader = new Reader(buf);
  while (!reader.eof) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === WIRE_LENGTH) return reader.string();
    reader.skip(wire);
  }
  return "";
}

function parseSignatureDef(buf: Uint8Array): SignatureDef {
  const reader = new Reader(buf);
  const inputs: Record<string, string> = {};
  const outputs: Record<string, string> = {};
  while (!reader.eof) {
    const { field, wire } = reader.tag();
    if ((field === 1 || field === 2) && wire === WIRE_LENGTH) {
      const { key, value } = parseMapEntry(reader.bytes());
      const name = parseTensorInfoName(value);
      if (field === 1) {
        inputs[key] = name;
      } else {
        outputs[key] = name;
      }
    } else {
      reader.skip(wire);
    }
  }
  return { inputs, outputs };
}

/** Extracts every signature from a serialized MetaGraphDef. */
export function parseSignatures(metaGraphDef: Uint8Array): Record<string, SignatureDef> {
  const reader = new Reader(metaGraphDef);
  const signatures: Record<string, SignatureDef> = {};
  while (!reader.eof) {
    const { field, wire } = reader.tag();
    if (field === 5 && wire === WIRE_LENGTH) {
      const { key, value } = parseMapEntry(reader.bytes());
      signatures[key] = parseSignatureDef(value);
    } else {
      reader.skip(wire);
    }
  }
  return signatures;
}

/** Splits a graph tensor name ("op:index") into its operation and output index. */
export function parseTensorName(name: string): { op: string; index: number } {
  const colon = name.lastIndexOf(":");
  if (colon === -1) return { op: name, index: 0 };
  const index = Number(name.slice(colon + 1));
  if (!Number.isInteger(index)) return { op: name, index: 0 };
  return { op: name.slice(0, colon), index };
}
