// postinstall: fetch and extract the libtensorflow C library for this platform.
//
// libtensorflow is NOT shipped in the npm tarball (size / license / arch). This
// downloads the official C-API archive (the same source tfjs-node uses), verifies
// its checksum when known, and extracts it to deps/libtensorflow so the addon can
// dynamically link it. See .claude/CLAUDE.md §10.
//
// Skipped when:
//   - LIBTENSORFLOW_ROOT points at an existing install, or
//   - TFJS_NATIVE_SKIP_INSTALL=1 (e.g. CI stages that build separately), or
//   - deps/libtensorflow already exists.
//
// Overrides:
//   - TFJS_NATIVE_CDN_STORAGE: mirror base URL (default: Google Cloud Storage).
//   - TFJS_NATIVE_LIBTENSORFLOW_VERSION: pin a different libtensorflow version.

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

// Default pinned to 2.15.0, the newest release with official C-library archives
// (none are published for 2.16+). Its Windows zip omits a few public headers
// (tf_buffer.h and part of the tsl/* tree); patchMissingHeaders() restores them
// from the matching source tag (see docs/DESIGN.md §10).
const VERSION = process.env.TFJS_NATIVE_LIBTENSORFLOW_VERSION ?? "2.15.0";
const DEFAULT_CDN = "https://storage.googleapis.com/tensorflow/libtensorflow";
const CDN = (process.env.TFJS_NATIVE_CDN_STORAGE ?? DEFAULT_CDN).replace(/\/+$/, "");

// SHA-256 of each known artifact. Verified when present; a missing entry only warns.
// Verified from actual downloads: download once, copy the sha256 the installer
// prints. See CONTRIBUTING.md.
const CHECKSUMS = {
  "libtensorflow-cpu-windows-x86_64-2.15.0.zip":
    "b0ba631308922083701d8d6b845b6647a264ad2a4a728b026932a93d52532941",
  "libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz":
    "f0fd1eb5db9e4e0603f10aec289574d1decb54f73b675f0ce476fea1f05838c8",
  "libtensorflow-cpu-darwin-x86_64-2.15.0.tar.gz":
    "6bc4cd060633bd08cde7934e7792ec8c2082049c706753e564ec0381a4a0b2da",
};

// SHA-256 of headers patchMissingHeaders() fetches for the pinned version,
// keyed by "<version>/<include path>". Each is byte-identical to the copy the
// same release's Linux/macOS archives ship.
const HEADER_CHECKSUMS = {
  "2.15.0/tensorflow/c/tf_buffer.h":
    "007f280677fc16fe49a153b1e41c6cd9e347ea45a59a2f88b85946bbd9bd7d56",
  "2.15.0/tsl/c/tsl_status.h": "f4d1b05f31a5e62368d3e19b69b93d5897d2367e15bdade87aaf758480f008f7",
  "2.15.0/tsl/platform/ctstring.h":
    "82d701f7649a8652dfaa8e6d0d4194a2ff8b073050df7528d66c1e856b0facc9",
  "2.15.0/tsl/platform/ctstring_internal.h":
    "a3932165371cd128b101fb2b8274fdb0f36b8e7461c16d18fb0a8d891e3b04e0",
};

/** Resolves the archive file name for the current platform/arch, or null. */
function artifactName() {
  const key = `${process.platform}:${process.arch}`;
  switch (key) {
    case "win32:x64":
      return `libtensorflow-cpu-windows-x86_64-${VERSION}.zip`;
    case "linux:x64":
      return `libtensorflow-cpu-linux-x86_64-${VERSION}.tar.gz`;
    case "linux:arm64":
      return `libtensorflow-cpu-linux-arm64-${VERSION}.tar.gz`;
    case "darwin:x64":
      return `libtensorflow-cpu-darwin-x86_64-${VERSION}.tar.gz`;
    default:
      return null;
  }
}

function guidance(reason) {
  return (
    `[tfjs-native] ${reason}\n` +
    "  This platform has no official libtensorflow C-library download.\n" +
    "  Install it another way (e.g. macOS arm64: `brew install libtensorflow`)\n" +
    "  and point LIBTENSORFLOW_ROOT at the directory containing include/ and lib/."
  );
}

async function download(url, file) {
  const res = await fetch(url);
  if (!res.ok || res.body === null) {
    throw new Error(`[tfjs-native] download failed: ${res.status} ${url}`);
  }
  await pipeline(Readable.fromWeb(res.body), createWriteStream(file));
}

async function verify(file, name) {
  const expected = CHECKSUMS[name];
  const digest = createHash("sha256")
    .update(await readFile(file))
    .digest("hex");
  if (!expected) {
    console.log(`[tfjs-native] sha256(${name}) = ${digest} (not pinned; skipping verify)`);
    return;
  }
  if (digest !== expected) {
    throw new Error(`[tfjs-native] checksum mismatch for ${name}: got ${digest}`);
  }
}

function tarBin() {
  // On Windows use the bundled bsdtar (reads .zip and drive-letter paths),
  // not a GNU tar that may come earlier on PATH (e.g. from Git for Windows).
  if (process.platform === "win32") {
    return join(process.env.SystemRoot ?? "C:\\Windows", "System32", "tar.exe");
  }
  return "tar";
}

function extract(file, dest) {
  mkdirSync(dest, { recursive: true });
  // `-xf` autodetects gzip; bsdtar also autodetects .zip.
  const r = spawnSync(tarBin(), ["-xf", file, "-C", dest], { stdio: "inherit" });
  if (r.status !== 0) {
    throw new Error(`[tfjs-native] extraction failed (tar exit ${r.status ?? r.signal})`);
  }
}

const TF_RAW = "https://raw.githubusercontent.com/tensorflow/tensorflow";

/**
 * Where an include path lives in the TensorFlow source tree, most likely first.
 * tsl/ moved from tensorflow/tsl/ (2.12-2.13) to third_party/xla/third_party/tsl/.
 */
function sourceCandidates(rel) {
  if (rel.startsWith("tsl/")) {
    return [`third_party/xla/third_party/tsl/${rel}`, `tensorflow/${rel}`];
  }
  return [rel];
}

async function fetchHeader(rel) {
  for (const path of sourceCandidates(rel)) {
    const res = await fetch(`${TF_RAW}/v${VERSION}/${path}`);
    if (res.ok) return Buffer.from(await res.arrayBuffer());
  }
  return null;
}

function verifyHeader(rel, bytes) {
  const expected = HEADER_CHECKSUMS[`${VERSION}/${rel}`];
  if (!expected) return;
  const digest = createHash("sha256").update(bytes).digest("hex");
  if (digest !== expected) {
    throw new Error(`[tfjs-native] checksum mismatch for header ${rel}: got ${digest}`);
  }
}

// The public headers the addon includes (src/native/*). Only headers reachable
// from these are patched: others in the archive (e.g. tensor_interface.h) pull
// in the C++ runtime tree, which the addon never compiles.
const HEADER_ROOTS = [
  "tensorflow/c/c_api.h",
  "tensorflow/c/eager/c_api.h",
  "tensorflow/c/tf_tstring.h",
];

// Some official archives (notably the Windows zips) omit a public header that
// c_api.h references (e.g. tf_buffer.h and tsl/*, split out in TF 2.12+). Walk
// the includes reachable from HEADER_ROOTS and fetch any missing tensorflow/*.h
// or tsl/*.h from the matching source tag so the addon can compile. A no-op on
// complete archives (Linux/macOS).
async function patchMissingHeaders(dest) {
  const includeRoot = join(dest, "include");
  if (!existsSync(includeRoot)) return;
  const seen = new Set();
  const queue = [...HEADER_ROOTS];
  while (queue.length > 0) {
    const rel = queue.shift();
    if (seen.has(rel)) continue;
    seen.add(rel);
    const target = join(includeRoot, rel);
    if (!existsSync(target)) {
      const bytes = await fetchHeader(rel);
      if (bytes === null) {
        // Non-fatal: the reference may sit behind a preprocessor branch this
        // build never takes. If the addon actually needs it, the compiler says so.
        console.warn(`[tfjs-native] could not fetch ${rel} for v${VERSION}; skipping`);
        continue;
      }
      verifyHeader(rel, bytes);
      mkdirSync(dirname(target), { recursive: true });
      writeFileSync(target, bytes);
      console.log(`[tfjs-native] patched missing header: ${rel}`);
    }
    const text = readFileSync(target, "utf8");
    for (const m of text.matchAll(/#include\s+"((?:tensorflow|tsl)\/[^"]+\.h)"/g)) {
      const ref = m[1];
      // Never fetchable, and never compiled in an open-source build:
      //   *.pb.h            generated protobuf headers (not in the source tree)
      //   platform/google/  Google-internal variants, gated behind PLATFORM_GOOGLE
      if (ref.endsWith(".pb.h") || ref.includes("/platform/google/")) continue;
      queue.push(ref);
    }
  }
}

async function main() {
  if (process.env.TFJS_NATIVE_SKIP_INSTALL === "1") {
    console.log("[tfjs-native] TFJS_NATIVE_SKIP_INSTALL=1 — skipping libtensorflow fetch.");
    return;
  }
  const override = process.env.LIBTENSORFLOW_ROOT;
  if (override && existsSync(override)) {
    console.log(`[tfjs-native] Using LIBTENSORFLOW_ROOT=${override}`);
    return;
  }
  const dest = resolve("deps/libtensorflow");
  if (existsSync(join(dest, "include")) && existsSync(join(dest, "lib"))) {
    await patchMissingHeaders(dest);
    console.log("[tfjs-native] libtensorflow already present.");
    return;
  }

  const name = artifactName();
  if (name === null) {
    console.log(guidance(`no artifact for ${process.platform}/${process.arch}`));
    return;
  }

  const url = `${CDN}/${name}`;
  const tmp = join(tmpdir(), name);
  console.log(`[tfjs-native] downloading ${url}`);
  await download(url, tmp);
  await verify(tmp, name);
  console.log(`[tfjs-native] extracting to ${dest}`);
  extract(tmp, dest);
  rmSync(tmp, { force: true });
  await patchMissingHeaders(dest);
  console.log("[tfjs-native] libtensorflow ready.");
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
