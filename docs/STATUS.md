# Status

**English** | [日本語](./ja/STATUS.md)

tfjs-native is in active development. This page tracks what works today; see the [roadmap](./DESIGN.md#12-roadmap) for the plan and [Functions](./FUNCTIONS.md) for the API.

**All planned milestones (M0–M5) are complete.** The native addon builds, prebuilds per OS, and the full test suite passes on Linux, Windows, and macOS in CI. Ongoing work is now incremental (op coverage, API polish) rather than milestone-driven.

| Milestone | Status | Delivers |
|---|---|---|
| M0 Skeleton | ✅ done | addon loads libtensorflow; `version()` |
| M1 Tensor | ✅ done | tensors, dtypes (incl. `int64` as BigInt), `data`/`array`, `tidy` |
| M2 Eager ops | ✅ done | the full op registry (~1,295 generated + hand-written), `runOp` |
| M3 SavedModel | ✅ done | `loadSavedModel`, signature resolution, `predict`/`run` |
| M4 Training | ✅ done | `grads`/`valueAndGrads`, `Variable`, `sgd`/`adam`/`rmsprop`, `minimize` |
| M5 Distribution | ✅ done | per-OS prebuilds, automatic libtensorflow fetch, and a trusted-publish release workflow |
