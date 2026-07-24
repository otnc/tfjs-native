// Codegen: turn the TF op registry into typed TS wrappers (M2b, MVP).
//
// Pipeline (see docs/DESIGN.md §6):
//   1. binding.getAllOps() -> serialized OpList proto
//   2. decode with protobufjs + the minimal op_def.proto
//   3. render one wrapper per op into src/ops/generated/index.ts
//
// MVP shape: inputs are typed (Tensor / Tensor[]) and the output count is typed;
// type/size attrs are auto-derived from inputs; the rest are passed via a generic
// options object (omitted attrs fall back to the op's TF default).
//
// Do not hand-edit the generated file — change this script and run `bun run codegen`.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import protobuf from "protobufjs";

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..");

// camelCase op names already covered by hand-written ops (manual wins).
const MANUAL = new Set([
  "add",
  "sub",
  "mul",
  "div",
  "maximum",
  "minimum",
  "pow",
  "neg",
  "abs",
  "exp",
  "log",
  "sqrt",
  "square",
  "relu",
  "sigmoid",
  "tanh",
  "sum",
  "mean",
  "max",
  "min",
  "matMul",
  "reshape",
  "transpose",
  "cast",
]);

const RESERVED = new Set([
  "break",
  "case",
  "catch",
  "class",
  "const",
  "continue",
  "debugger",
  "default",
  "delete",
  "do",
  "else",
  "enum",
  "export",
  "extends",
  "false",
  "finally",
  "for",
  "function",
  "if",
  "import",
  "in",
  "instanceof",
  "new",
  "null",
  "return",
  "super",
  "switch",
  "this",
  "throw",
  "true",
  "try",
  "typeof",
  "var",
  "void",
  "while",
  "with",
  "let",
  "static",
  "yield",
  "await",
  "implements",
  "package",
  "interface",
  "private",
  "protected",
  "public",
]);

const USER_ATTR = {
  int: { ts: "number", kind: "int", val: (o) => o },
  float: { ts: "number", kind: "float", val: (o) => o },
  bool: { ts: "boolean", kind: "bool", val: (o) => o },
  string: { ts: "string", kind: "string", val: (o) => o },
  type: { ts: "DTypeName", kind: "type", val: (o) => `DType[${o}]` },
  shape: { ts: "number[]", kind: "shape", val: (o) => o },
  "list(int)": { ts: "number[]", kind: "intList", val: (o) => o },
  "list(float)": { ts: "number[]", kind: "floatList", val: (o) => o },
  "list(bool)": { ts: "boolean[]", kind: "boolList", val: (o) => o },
  "list(string)": { ts: "string[]", kind: "stringList", val: (o) => o },
  "list(type)": { ts: "DTypeName[]", kind: "typeList", val: (o) => `${o}.map((d) => DType[d])` },
};

const isIdent = (s) => /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(s);
const key = (s) => (isIdent(s) && !RESERVED.has(s) ? s : JSON.stringify(s));
const access = (obj, s) =>
  isIdent(s) && !RESERVED.has(s) ? `${obj}.${s}` : `${obj}[${JSON.stringify(s)}]`;
const param = (s) => (isIdent(s) && !RESERVED.has(s) ? s : `${s.replace(/[^A-Za-z0-9_$]/g, "_")}_`);

function camel(name) {
  // PascalCase op name -> camelCase (lowercase the leading char only).
  return name.charAt(0).toLowerCase() + name.slice(1);
}

function renderOp(op) {
  const inputs = op.input_arg ?? [];
  const outputs = op.output_arg ?? [];
  const attrs = op.attr ?? [];

  const autoType = new Set();
  const autoNumber = new Set();
  for (const a of inputs) {
    if (a.type_attr) autoType.add(a.type_attr);
    if (a.type_list_attr) autoType.add(a.type_list_attr);
    if (a.number_attr) autoNumber.add(a.number_attr);
  }

  const userAttrs = attrs.filter((a) => !autoType.has(a.name) && !autoNumber.has(a.name));
  for (const a of userAttrs) {
    if (!USER_ATTR[a.type]) return null; // unsupported attr kind (tensor/func/list(shape)/...)
  }

  // input params + the runOp inputs array (list inputs are spread)
  const paramNames = inputs.map((a) => param(a.name));
  const inputParams = inputs.map((a, i) => {
    const isList = Boolean(a.number_attr || a.type_list_attr);
    return `${paramNames[i]}: Tensor${isList ? "[]" : ""}`;
  });
  const inputArrayParts = inputs.map((a, i) =>
    a.number_attr || a.type_list_attr ? `...${paramNames[i]}` : paramNames[i],
  );

  // attrs object entries
  const attrEntries = [];
  for (const a of attrs) {
    const name = a.name;
    if (autoType.has(name)) {
      const inp = inputs.find((x) => x.type_attr === name);
      if (inp) {
        // A number_attr input is a Tensor[]; take the element dtype.
        const isList = Boolean(inp.number_attr || inp.type_list_attr);
        const dtypeExpr = isList ? `${param(inp.name)}[0]!.dtype` : `${param(inp.name)}.dtype`;
        attrEntries.push(`${key(name)}: { type: "type", value: DType[${dtypeExpr}] }`);
      } else {
        const list = inputs.find((x) => x.type_list_attr === name);
        attrEntries.push(
          `${key(name)}: { type: "typeList", value: ${param(list.name)}.map((t) => DType[t.dtype]) }`,
        );
      }
    } else if (autoNumber.has(name)) {
      const inp = inputs.find((x) => x.number_attr === name);
      attrEntries.push(`${key(name)}: { type: "int", value: ${param(inp.name)}.length }`);
    }
  }
  const optionFields = [];
  for (const a of userAttrs) {
    const spec = USER_ATTR[a.type];
    optionFields.push(`${key(a.name)}?: ${spec.ts}`);
    const acc = access("opts", a.name);
    attrEntries.push(
      `...(${acc} !== undefined ? { ${key(a.name)}: { type: "${spec.kind}", value: ${spec.val(acc)} } } : {})`,
    );
  }

  // output count
  let staticCount = 0;
  const dynamicParts = [];
  for (const o of outputs) {
    if (o.number_attr) {
      const inp = inputs.find((x) => x.number_attr === o.number_attr);
      dynamicParts.push(
        inp ? `${param(inp.name)}.length` : `(${access("opts", o.number_attr)} ?? 0)`,
      );
    } else if (o.type_list_attr) {
      const inp = inputs.find((x) => x.type_list_attr === o.type_list_attr);
      dynamicParts.push(
        inp ? `${param(inp.name)}.length` : `(${access("opts", o.type_list_attr)}?.length ?? 0)`,
      );
    } else {
      staticCount += 1;
    }
  }
  const single = dynamicParts.length === 0 && staticCount === 1;
  const numOutputs =
    [String(staticCount), ...dynamicParts]
      .filter((p) => p !== "0" || dynamicParts.length === 0)
      .join(" + ") || "0";

  const params = [...inputParams];
  if (optionFields.length > 0) params.push(`opts: { ${optionFields.join("; ")} } = {}`);
  const attrObj = attrEntries.length ? `{ ${attrEntries.join(", ")} }` : "{}";
  const inputArr = `[${inputArrayParts.join(", ")}]`;

  const body = single
    ? `  return runOp1(${JSON.stringify(op.name)}, ${inputArr}, ${attrObj});`
    : `  return runOp(${JSON.stringify(op.name)}, ${inputArr}, ${attrObj}, ${numOutputs});`;
  const ret = single ? "Tensor" : "Tensor[]";
  return `export function ${camel(op.name)}(${params.join(", ")}): ${ret} {\n${body}\n}`;
}

function main() {
  let binding;
  try {
    binding = require("node-gyp-build")(root);
  } catch (err) {
    console.error("[codegen] native addon not built. Run `bun run build:native` first.");
    console.error(String(err instanceof Error ? err.message : err));
    process.exit(1);
  }

  const bytes = binding.getAllOps();
  // keepCase keeps the .proto's snake_case field names on decoded objects.
  const { root: proto } = protobuf.parse(readFileSync(join(here, "op_def.proto"), "utf8"), {
    keepCase: true,
  });
  const OpList = proto.lookupType("tensorflow.OpList");
  const { op: ops } = OpList.decode(bytes);

  const seen = new Set(MANUAL);
  const rendered = [];
  let skipped = 0;
  for (const op of [...ops].sort((a, b) => a.name.localeCompare(b.name))) {
    if (op.name.startsWith("_")) {
      skipped++;
      continue;
    }
    const fn = camel(op.name);
    if (!isIdent(fn) || RESERVED.has(fn) || seen.has(fn)) {
      skipped++;
      continue;
    }
    const code = renderOp(op);
    if (code === null) {
      skipped++;
      continue;
    }
    seen.add(fn);
    rendered.push(code);
  }

  const header = `// AUTO-GENERATED by scripts/codegen — DO NOT EDIT.
// Typed wrappers for the libtensorflow op registry (${rendered.length} ops).
// Regenerate with \`bun run codegen\`. See docs/DESIGN.md §6.

import { DType, type DTypeName } from "../../backend/dtype.js";
import type { Tensor } from "../../tensor/tensor.js";
import { runOp, runOp1 } from "../run.js";
`;
  const outDir = join(root, "src", "ops", "generated");
  mkdirSync(outDir, { recursive: true });
  const out = join(outDir, "index.ts");
  writeFileSync(out, `${header}\n${rendered.join("\n\n")}\n`);
  console.log(`[codegen] generated ${rendered.length} ops (${skipped} skipped) -> ${out}`);
}

main();
