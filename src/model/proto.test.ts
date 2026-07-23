import { describe, expect, it } from "bun:test";
import { parseSignatures, parseTensorName } from "./proto.js";

// Minimal protobuf encoders, mirroring the wire format the reader consumes.
const varint = (n: number): number[] => {
  const out: number[] = [];
  let v = n;
  while (v > 127) {
    out.push((v & 0x7f) | 0x80);
    v = Math.floor(v / 128);
  }
  out.push(v);
  return out;
};
const tag = (field: number, wire: number): number[] => varint((field << 3) | wire);
const lenDelim = (field: number, payload: number[]): number[] => [
  ...tag(field, 2),
  ...varint(payload.length),
  ...payload,
];
const str = (s: string): number[] => [...new TextEncoder().encode(s)];

const tensorInfo = (name: string): number[] => lenDelim(1, str(name));
const mapEntry = (key: string, value: number[]): number[] => [
  ...lenDelim(1, str(key)),
  ...lenDelim(2, value),
];

describe("parseTensorName", () => {
  it("splits op and output index", () => {
    expect(parseTensorName("serving_default_x:0")).toEqual({ op: "serving_default_x", index: 0 });
    expect(parseTensorName("PartitionedCall:2")).toEqual({ op: "PartitionedCall", index: 2 });
  });

  it("defaults the index to 0 when absent", () => {
    expect(parseTensorName("someOp")).toEqual({ op: "someOp", index: 0 });
  });
});

describe("parseSignatures", () => {
  it("reads inputs and outputs of each signature", () => {
    const signatureDef = [
      ...lenDelim(1, mapEntry("x", tensorInfo("in:0"))),
      ...lenDelim(2, mapEntry("y", tensorInfo("out:0"))),
    ];
    const metaGraphDef = lenDelim(5, mapEntry("serving_default", signatureDef));

    expect(parseSignatures(new Uint8Array(metaGraphDef))).toEqual({
      serving_default: { inputs: { x: "in:0" }, outputs: { y: "out:0" } },
    });
  });

  it("skips unrelated fields and wire types", () => {
    const signatureDef = [
      ...tag(3, 0), // method_name-ish varint we do not care about
      ...varint(7),
      ...lenDelim(1, mapEntry("a", tensorInfo("i:1"))),
      ...lenDelim(2, mapEntry("b", tensorInfo("o:0"))),
    ];
    const metaGraphDef = [
      ...tag(1, 0), // an unrelated top-level varint field
      ...varint(42),
      ...lenDelim(2, str("ignored graph_def bytes")),
      ...lenDelim(5, mapEntry("sig", signatureDef)),
    ];

    expect(parseSignatures(new Uint8Array(metaGraphDef))).toEqual({
      sig: { inputs: { a: "i:1" }, outputs: { b: "o:0" } },
    });
  });

  it("returns an empty map when there are no signatures", () => {
    expect(parseSignatures(new Uint8Array([]))).toEqual({});
  });
});
