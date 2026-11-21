:::kiritan{locale=en}
# Status
:::

:::kiritan{locale=ja}
# ステータス
:::

::kiritan{switcher}

:::kiritan{locale=en}
tfjs-native is in active development. This page tracks what works today; see the [roadmap](./DESIGN.md#12-roadmap) for the plan and [Functions](./FUNCTIONS.md) for the API.

**All planned milestones (M0–M5) are complete.** The native addon builds, prebuilds per OS, and the full test suite passes on Linux, Windows, and macOS in CI. Ongoing work is now incremental (op coverage, API polish) rather than milestone-driven.

| Milestone | Status | Delivers |
|---|---|---|
| M0 Skeleton | ✅ done | addon loads libtensorflow; `version()` |
| M1 Tensor | ✅ done | tensors, dtypes (incl. `int64` as BigInt), `data`/`array`, `tidy` |
| M2 Eager ops | ✅ done | the full op registry (~1,334 generated + hand-written), `runOp` |
| M3 SavedModel | ✅ done | `loadSavedModel`, signature resolution, `predict`/`run` |
| M4 Training | ✅ done | `grads`/`valueAndGrads`, `Variable`, `sgd`/`adam`/`rmsprop`, `minimize` |
| M5 Distribution | ✅ done | per-OS prebuilds, automatic libtensorflow fetch, and a trusted-publish release workflow |
:::

:::kiritan{locale=ja}
tfjs-native は活発に開発中です。このページは現時点で動くものを追跡します。計画は [ロードマップ](./DESIGN.md#12-ロードマップ)、API は [機能一覧](./FUNCTIONS.md) を参照してください。

**計画していたマイルストーン（M0〜M5）はすべて完了しました。** ネイティブ addon がビルドでき、OS 別に prebuild でき、CI で Linux・Windows・macOS すべてで全テストが通ります。今後の作業はマイルストーン駆動ではなく、op カバレッジや API の細かな改善が中心になります。

| マイルストーン | 状態 | 提供内容 |
|---|---|---|
| M0 スケルトン | ✅ 完了 | addon が libtensorflow をロード。`version()` |
| M1 Tensor | ✅ 完了 | テンソル、dtype（`int64` は BigInt）、`data`/`array`、`tidy` |
| M2 Eager ops | ✅ 完了 | 全 op registry(生成約 1,334 + 手書き)、`runOp` |
| M3 SavedModel | ✅ 完了 | `loadSavedModel`、signature 解決、`predict`/`run` |
| M4 学習 | ✅ 完了 | `grads`/`valueAndGrads`、`Variable`、`sgd`/`adam`/`rmsprop`、`minimize` |
| M5 配布 | ✅ 完了 | OS 別 prebuild、libtensorflow 自動取得、trusted-publish リリースワークフロー |
:::
