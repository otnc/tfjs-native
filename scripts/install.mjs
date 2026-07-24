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
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

// Default pinned to 2.10.0: its official archives ship a self-contained C-API
// header set on every platform. Newer archives (2.12+) omit split-out headers
// (tf_buffer.h and the tsl/* tree) on Windows, which breaks the compile; bumping
// the default is tracked as future work (see docs/DESIGN.md §10).
const VERSION = process.env.TFJS_NATIVE_LIBTENSORFLOW_VERSION ?? "2.10.0";
const DEFAULT_CDN = "https://storage.googleapis.com/tensorflow/libtensorflow";
const CDN = (process.env.TFJS_NATIVE_CDN_STORAGE ?? DEFAULT_CDN).replace(/\/+$/, "");

// SHA-256 of each known artifact. Verified when present; a missing entry only warns.
const CHECKSUMS = {
  // Verified from actual downloads. Add linux/darwin entries the same way
  // (download once, copy the sha256 the installer prints). See CONTRIBUTING.md.
  "libtensorflow-cpu-windows-x86_64-2.10.0.zip":
    "4c5e6f9a7683583220716fecadea53ace233f31f59062e83585c4821f9968438",
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

// Some official archives (notably the Windows zips) omit a public header that
// c_api.h references (e.g. tf_buffer.h, split out in TF 2.12). Fetch any
// referenced-but-missing tensorflow/*.h from the matching source tag so the
// addon can compile. A no-op on complete archives (Linux/macOS).
async function patchMissingHeaders(dest) {
  const includeRoot = join(dest, "include");
  if (!existsSync(includeRoot)) return;
  const skip = new Set(); // .pb.h (generated) or headers not fetchable from source
  for (let pass = 0; pass < 5; pass++) {
    const headers = readdirSync(includeRoot, { recursive: true })
      .map((f) => f.toString())
      .filter((f) => f.endsWith(".h"));
    const missing = new Set();
    for (const rel of headers) {
      const text = readFileSync(join(includeRoot, rel), "utf8");
      for (const m of text.matchAll(/#include\s+"(tensorflow\/[^"]+\.h)"/g)) {
        const ref = m[1];
        // Never fetchable, and never compiled in an open-source build:
        //   *.pb.h            generated protobuf headers (not in the source tree)
        //   platform/google/  Google-internal variants, gated behind PLATFORM_GOOGLE
        if (ref.endsWith(".pb.h") || ref.includes("/platform/google/") || skip.has(ref)) {
          continue;
        }
        if (!existsSync(join(includeRoot, ref))) missing.add(ref);
      }
    }
    if (missing.size === 0) return;
    for (const rel of missing) {
      const url = `${TF_RAW}/v${VERSION}/${rel}`;
      const res = await fetch(url);
      if (!res.ok) {
        // Non-fatal: a header referenced by an unused shipped header may not be
        // fetchable. If the addon actually needs it, the compiler will say so.
        console.warn(`[tfjs-native] could not fetch ${rel} (${res.status}); skipping`);
        skip.add(rel);
        continue;
      }
      const target = join(includeRoot, rel);
      mkdirSync(dirname(target), { recursive: true });
      writeFileSync(target, Buffer.from(await res.arrayBuffer()));
      console.log(`[tfjs-native] patched missing header: ${rel}`);
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
