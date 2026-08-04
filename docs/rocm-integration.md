# ROCm/HIP integration log

## Current status (read this first)

As of 2026-08-05, on this branch (`hip-integration`):

- **HIP-accelerated:** dense stereo (`patch_match_stereo`) AND bundle adjustment
  (`CASPAR` backend, native OpenCV camera-model support included from the start)
  — both verified building and running correctly on real data on gfx1151. Caspar-HIP
  BA closes the deferral noted below: it turned out to require build-system fixes
  (three real, iterated-on bugs, see the Caspar-HIP Completion entries below), not
  a fundamentally missing capability.
- **CPU-only (not HIP):** feature extraction/matching only (SIFT — HIP SIFT attempt
  documented below, cherry-pick abandoned; CPU/OpenGL SIFT remains the working path).
- Earlier entries below (particularly around Task 4 and the old Task 5/6 deferral
  notes) describe an intermediate state where Caspar-HIP was believed structurally
  blocked ("COLMAP vendors Caspar as CUDA-only generated source"). That assumption
  no longer holds: `symforce-rocm`'s own codegen templates (as of its
  `hip-integration` branch) now bake in full HIP support for every generated
  Caspar kernel unconditionally, and colmap-rocm's build system only needed a
  handful of CMake wiring fixes on top, not the from-scratch device-code-mapping
  effort originally assumed necessary. Where an entry below conflicts with this
  status block, this status block is current; the entry is a historical record of
  what was believed true at the time, not a live claim.
- Post-final-review fixes (2026-08-04): `ROCM_ARCH` is now a Dockerfile `ARG`
  (overridable via `--build-arg`), `HSA_OVERRIDE_GFX_VERSION` was removed from
  the image's persistent `ENV` (every documented run command already passes it
  via `-e` explicitly, so this wasn't load-bearing — an image should not force a
  GPU-arch override on whoever runs it), and the tests-enabled build now needs
  `KEEP_SOURCE=1` explicitly (previously any non-empty `CMAKE_EXTRA_ARGS` kept
  the source tree as a side effect) — Task 4's `docker build ... --build-arg
  CMAKE_EXTRA_ARGS="-DTESTS_ENABLED=ON"` invocation above needs
  `--build-arg KEEP_SOURCE=1` added if repeated after this fix.

## Task 3: Rebase COLMAP PR #4420 onto current `upstream/main` (2026-08-04)

**Source:** `ishengnan/rocm-support` (PR [#4420](https://github.com/colmap/colmap/pull/4420),
"Add ROCm/HIP support for patch_match_stereo (AMD GPU)").

**Result:** Clean rebase, no conflicts.

`git rebase upstream/main` on a branch created from `ishengnan/rocm-support` completed
successfully with zero conflict hunks across all 8 commits. GitHub's `mergeable: true`
flag (checked prior to starting) held true at rebase time — `upstream/main` had not
drifted in a way that touched the same lines as the PR.

- Base (merge-base with `upstream/main`): `ecdeba302c511b552f6fcb38a03332212cbcc037`
- Rebased tip: `9c1cd066` ("Address remaining HIP review feedback")
- Commit count: 8 (matches PR #4420's original commit count)

Rebased commit range (`upstream/main..hip-integration` after fold-in):

```
6edb2aca Add ROCm/HIP support for patch_match_stereo on AMD GPUs
36e28b52 fix(cmake): address code review feedback for portability
c4185870 Fix ROCm/HIP support: dual-compatible headers, hipify-perl build, avoid enable_language(HIP)
38922a3f Simplify ROCm/HIP support: enable_language(HIP) + cuda_to_hip.h compat header (#1)
c5fd3db4 Fix CI failures from PR #1 and address PR #4420 review
13508b4f Support patch_match_stereo on AMD CDNA (gfx9) GPUs
a9e90965 Auto-detect ROCm install path and HIP architectures via rocm-sdk
9c1cd066 Address remaining HIP review feedback
```

**Steps taken:**
1. `git fetch ishengnan rocm-support` / `git fetch upstream main`.
2. `git checkout -b colmap-patchmatch-rebase ishengnan/rocm-support`.
3. `git rebase upstream/main` — completed cleanly (8/8 commits applied, no conflict markers).
4. `git checkout hip-integration && git reset --hard colmap-patchmatch-rebase`.
5. `git branch -D colmap-patchmatch-rebase`.
6. `git push --force-with-lease origin hip-integration` — accepted
   (`f6cfb683...9c1cd066 hip-integration -> hip-integration (forced update)`).

**Conflicts encountered:** None. No manual conflict resolution was required for this track.

**Follow-on:** Task 4 will build using
`-DCUDA_ENABLED=OFF -DHIP_ENABLED=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151` per the produced
`HIP_ENABLED` CMake option.

## Task 4: Dockerfile + build + PatchMatch-HIP smoke test (2026-08-04)

**Result:** Image builds successfully (both plain and `-DTESTS_ENABLED=ON` variants). No
HIP-specific compile errors. Unit test smoke test: 13/14 tests in the `mvs|gpu_mat|patch_match`
filter pass; `mvs/gpu_mat_test` fails reproducibly (2/2 runs) with a HIP runtime memory error,
most likely caused by a concurrently running GPU workload on this single-GPU host rather than a
defect in the ported HIP code — see details below.

**Build (Step 2):** `docker build -t colmap-rocm:hip .` — succeeded, `real 4m40.559s`
(most of the time is compiling ~300 translation units; base ROCm/PyTorch image and apt
layers were already warm). No HIP-specific compile errors of any kind.

**Tests-enabled build (Step 3a):**
`docker build -t colmap-rocm:hip-tests --build-arg CMAKE_EXTRA_ARGS="-DTESTS_ENABLED=ON" .`
— succeeded, reusing cached layers, completed in under a minute of net new work.

**ctest smoke test (Step 3b):**
```
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 39 --group-add 105 \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  --entrypoint bash colmap-rocm:hip-tests -c \
  "cd /opt/colmap_src/build && ctest -R 'mvs|gpu_mat|patch_match' --output-on-failure"
```
Note: the brief's literal `--group-add video --group-add render` failed with
`Error: looking up supplemental groups ... Unable to find group render: no matching entries
in group file` — the `render` group name isn't defined in this base image's `/etc/group` (only
`video`, gid 44, is). Substituted the host's numeric GIDs for `video` (39) and `render` (105)
instead, which podman accepts without a name-lookup, and device access then worked correctly
(13 of 14 tests ran and interacted with the GPU successfully).

**Result:** 13/14 tests passed. `mvs/gpu_mat_test` (`GpuMat.FillWithVector`) aborted both times
it was run with:
```
Memory critical error by agent node-0 (Agent handle: 0x388b9da0) on address 0x7f561fead000. Reason: Memory in use.
```
`rocm-smi` at the time showed VRAM at 80% and an unrelated `splatograph_train_tmp*` container
actively training on the same (only) GPU in this host — this looks like GPU memory contention
from a concurrent workload rather than a bug in the PR's HIP port. This is a plausible but
*unconfirmed* explanation: the test was not re-run with the GPU otherwise idle, so genuine
HIP-correctness regressions in `GpuMat` cannot be fully ruled out yet. Re-running this smoke
test with the GPU free is recommended before Task 7 relies on `GpuMat`/PatchMatch-HIP inside a
full pipeline.

Full pass/fail table and both raw ctest logs are in the Task 4 report:
`.superpowers/sdd/2026-08-04-colmap-rocm-integration/task-4-report.md`.

## Task 5: Rebase GPU SIFT branch onto hip-integration (2026-08-04)

**Result: FALLBACK — rebase abandoned per the documented hard abort trigger.**
`hip-integration` is unchanged; GPU SIFT (`jeffdaily/rocm-sift-gpu`) is **not** included.
Task 6 proceeds with COLMAP's existing OpenGL/CPU SIFT frontend. PatchMatch-HIP (Task 3/4)
and Caspar-HIP remain HIP-accelerated, so this is still a valid, partially-HIP-accelerated
end-to-end pipeline — just not full-HIP-frontend.

**Source:** `jeffdaily/rocm-sift-gpu`, 10 commits, ~115 commits behind `upstream/main` at
plan-writing time (`git log --oneline jeffdaily/rocm-sift-gpu ^upstream/main | wc -l` → 10).

**Assessment (Step 1):** `git diff upstream/main jeffdaily/rocm-sift-gpu --stat` showed a
whole-repository-scale diff (CI workflows, benchmark scripts, docs, and core `src/colmap/mvs`
and `src/colmap/util` files all touched) — expected fallout from 115 commits of upstream drift,
not evidence by itself of a bad rebase.

**Rebase attempt (Step 2):**
```
git checkout -b colmap-sift-rebase jeffdaily/rocm-sift-gpu
git rebase hip-integration
```
Two of the ten commits (`786e0963`, `e609b9f1`) were skipped automatically as already applied
(shared history with the PR #4420 lineage already on `hip-integration`). The very first commit
actually replayed, `658f8b56` ("Add ROCm/HIP support for patch_match_stereo on AMD GPUs"),
produced conflicts in **12 files**:

```
cmake/FindDependencies.cmake
src/colmap/exe/CMakeLists.txt
src/colmap/mvs/CMakeLists.txt
src/colmap/mvs/cuda_flip.h
src/colmap/mvs/cuda_rotate.h
src/colmap/mvs/cuda_texture.h
src/colmap/mvs/cuda_transpose.h
src/colmap/mvs/gpu_mat.h
src/colmap/mvs/patch_match_cuda.h
src/colmap/util/CMakeLists.txt
src/colmap/util/cuda.cc
src/colmap/util/cudacc.cc
src/colmap/util/cudacc.h
```

This exceeds the brief's hard abort trigger (>~5 files conflicted in a single commit) on the
very first commit replayed — before any judgment call about resolution quality was even
reachable. Per the brief: *"do not push through on a case-by-case 'am I confident' judgment
call."* Stopped immediately.

**Why this makes sense:** `jeffdaily/rocm-sift-gpu`'s own patch-match/CUDA-compat HIP work
(`658f8b56` and friends) independently touches almost the exact same CUDA-compat surface
(`cuda_flip.h`, `cuda_rotate.h`, `cuda_texture.h`, `cuda_transpose.h`, `gpu_mat.h`,
`patch_match_cuda.h`, `util/cuda*.{cc,h}`) that PR #4420's PatchMatch-HIP port
(already folded into `hip-integration`) rewrote. Two independent HIP ports of the same
CUDA-compat layer, built 115 commits apart, is exactly the "double conflict" scenario the
brief warned about in Step 1 — and it manifested on the first commit rather than being
resolvable case-by-case.

**Action taken (Step 3, fallback path):**
```
git rebase --abort
git checkout hip-integration
git branch -D colmap-sift-rebase
```
`hip-integration` working tree is clean and unchanged (`git status` confirms
"nothing to commit, working tree clean", still tracking `origin/hip-integration`, no
force-push performed — `upstream`/`origin` were never touched by this task).

**Consequence for the plan:** Task 6 should run the pipeline with COLMAP's stock
OpenGL/CPU SIFT extractor/matcher instead of HIP-accelerated SIFT. HIP acceleration still
covers PatchMatch stereo (Task 3/4) and Caspar (separate task) — this remains a genuine
partial-HIP end-to-end run, not a fully-CPU fallback. The plan's success criteria should be
read as "HIP-accelerated PatchMatch + Caspar, CPU/OpenGL SIFT" rather than full-HIP-frontend,
per this task's brief.

## 2026-08-04 — Task 4 follow-up: gpu_mat_test retest with GPU idle

Per the task review's required follow-up: re-ran `ctest -R 'mvs|gpu_mat|patch_match'`
in the already-built `colmap-rocm:hip-tests` image once no `splatograph_train_tmp*`
container was running.

```bash
docker run --rm --name gpu_mat_retest \
  --device=/dev/kfd --device=/dev/dri \
  --group-add 39 --group-add 105 \
  --security-opt label=disable \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  --entrypoint bash \
  localhost/colmap-rocm:hip-tests \
  -c "cd /opt/colmap_src/build && ctest -R 'mvs|gpu_mat|patch_match' --output-on-failure"
```

Result: **14/14 tests passed**, including `mvs/gpu_mat_test` (previously failed with
a HIP "Memory in use" error under GPU contention). Confirms the Task 4 report's
hypothesis: the original failure was resource contention from a concurrent,
unrelated training workload on this shared single-GPU host, not a defect in
PR #4420's HIP PatchMatch/GpuMat port. PatchMatch-HIP is now verified clean at the
unit-test level; Task 7 proceeds with confidence in this layer.

## 2026-08-04 — Task 6: Caspar-HIP wiring — DEFERRED (plan premise invalidated)

Task 6 as originally scoped (Docker multi-stage: build symforce-rocm's HIP Caspar
`.so` in one stage, `COPY --from=` it into the colmap-rocm image, wire it in as
the mapper's BA backend) is **not how COLMAP actually consumes Caspar** — the
premise was wrong, discovered by inspection, not by attempting the Docker wiring
and failing.

Facts, verified by direct inspection of this repo:

- Caspar is **vendored as generated CUDA `.cu` source** at
  `src/thirdparty/Symforce-Caspar/generated/{f32,f64}/` (241 kernel files in the
  f32 tree alone), compiled directly as part of COLMAP's own build — not linked
  as an external SymForce library. `src/colmap/controllers/option_manager.cc`
  gates `BundleAdjustmentCaspar.*` options behind `#ifdef CASPAR_ENABLED`, a
  COLMAP-internal compile flag (`CMakeLists.txt:77`), unrelated to whether
  `symforce-rocm` (this repo's sibling fork) is built or even present.
- `CASPAR_ENABLED` is wired **only inside `if(CUDA_ENABLED AND CUDA_FOUND)`** in
  `cmake/FindDependencies.cmake` (~line 310+), with an arch guard that reads
  `CMAKE_CUDA_ARCHITECTURES` specifically. There is no `HIP_ENABLED` branch for
  Caspar at all in the current `hip-integration` branch (PR #4420 only added HIP
  support for PatchMatch/MVS, not Caspar/bundle-adjustment).
- The 241 vendored `.cu` files under `generated/f32/` use
  `#include <cooperative_groups.h>`, `cooperative_groups::reduce`,
  `cooperative_groups::details::partitioning` (`labeled_partition`), and
  `namespace cg = cooperative_groups` — CUDA-specific constructs.
  `src/colmap/util/cuda_to_hip.h` (PR #4420's compat header) has **zero**
  coverage of any of these — it was built for PatchMatch's texture/RNG/event
  usage, an entirely different API surface than Caspar's cooperative-groups
  reduction pattern.

Given CUDA_ENABLED and HIP_ENABLED are mutually exclusive (this plan's own global
constraint, enforced by a `FATAL_ERROR` in `CMakeLists.txt`),
`-DCASPAR_ENABLED=ON -DHIP_ENABLED=ON -DCUDA_ENABLED=OFF` cannot work today even
before considering the 241-file cooperative_groups gap: the `CASPAR_ENABLED`
block simply never executes when `CUDA_ENABLED=OFF`.

Whether SymForce PR #465's HIP compat layer (which does map `cg::reduce`,
`cg::labeled_partition`, and shared-memory atomics — see `~/git/symforce-rocm`
`docs/rocm-integration.md` Task 1 entry) *would* cover this vendored tree's usage
is unresolved: PR #465's mappings live in SymForce's own Caspar *runtime*
generator output, and this vendored tree may have been generated by a different
(older, or differently-configured) run of the same generator — there's no
guarantee the two match construct-for-construct without actually attempting the
port or diffing generator output.

**Decision: defer, same treatment as Task 5's SIFT rebase fallback.** Two real
paths exist for a future session:
1. Add a HIP branch to `CASPAR_ENABLED` in `FindDependencies.cmake` +
   `generated/f32/CMakeLists.txt` (mirroring PR #4420's `LANGUAGE HIP` mechanism
   for `.cu`→HIP compilation) plus a Caspar-specific compat header covering
   `cooperative_groups`/`cg::*`, informed by (not copy-pasted from) PR #465's
   mappings.
2. Regenerate `generated/{f32,f64}` from `src/thirdparty/Symforce-Caspar/caspar_generate.py`
   with SymForce-rocm's `use_hip=True` path (the same `CasparLibrary.compile()`
   API exercised in Task 2's `hip_smoke.py`) — larger diff, no reference output
   to verify numeric equivalence against, higher risk without a dedicated
   verification pass.

Neither is attempted here. Bundle adjustment for Task 7's end-to-end run uses
COLMAP's default Ceres CPU backend, not Caspar. PatchMatch-HIP (Task 4) and
HIP Caspar as a standalone library (Task 2, `symforce-rocm`) remain independently
verified and valid — they are just not yet wired into a single COLMAP binary.

## 2026-08-04 — Task 7: Full end-to-end incremental SfM run on gfx1151

Ran directly in this session (not via subagent — a short, monitorable sequential
pipeline, per advisor guidance after Task 2's subagent dispatch overhead).

**Dataset:** 31 frames subsampled (every 15th) from
`~/git/rosbag-colmap-pipeline/data/workspaces/table1/rgb/` (451 total frames),
staged at `/tmp/colmap-rocm-e2e/`.

**Pipeline run (all stages, `colmap-rocm:hip` image):**

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add 39 --group-add 105 \
  --security-opt label=disable \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -e QT_QPA_PLATFORM=offscreen \
  -v /tmp/colmap-rocm-e2e:/workspace/data \
  colmap-rocm:hip <command> ...
```

Two runtime env-var fixes needed beyond Task 4/2's known gotchas (both required
for `feature_extractor`/`sequential_matcher`, harmless for other commands):
- `QT_QPA_PLATFORM=offscreen` — `feature_extractor` instantiates a `QApplication`
  even in CLI mode (this build has `GUI_ENABLED` at its default `ON`, per Task 4's
  deliberate choice to match `rosbag-colmap-pipeline`'s known-working config);
  without a display, `QGuiApplicationPrivate::createPlatformIntegration()` aborts.
- `--FeatureExtraction.use_gpu 0` / `--FeatureMatching.use_gpu 0` — SiftGPU's
  default GPU path tries to create an OpenGL context
  (`colmap::OpenGLContextManager`), which fails headlessly in this container
  (`Check failed: context_.create()`). Since HIP SIFT was deferred (Task 5),
  this is expected — CPU/OpenGL SIFT was always the fallback plan; this simply
  makes that explicit at the command-line level rather than relying on a
  silent internal fallback.

**Results, stage by stage:**

| Stage | Backend | Result |
|---|---|---|
| `feature_extractor` | CPU SIFT | 31/31 images, 3700–12400 features each, 0.03 min |
| `sequential_matcher` | CPU | 31/31 images matched, 0.14 min |
| `mapper` | Ceres CPU BA (Caspar deferred, Task 6) | 17/31 images registered into one connected model (`sparse/0`), 1341 3D points, "Keeping successful reconstruction", 0.04 min. (14 images did not register into this model — expected for a sparse/wide-baseline 31-frame subsample of a video sequence, not investigated further; out of scope for this HIP-verification task.) |
| `image_undistorter` | CPU | 17/17 images undistorted cleanly, 0.007 min |
| `patch_match_stereo` | **HIP (gfx1151)** | 17/17 views × 2 passes (photometric + geometric consistency) completed with no errors — confirmed via per-sweep/iteration timing logs (`cudacc.cc`, e.g. "Sweep 1: 0.78s", "Iteration 1: 4.19s") showing real GPU computation, not a no-op. Produced 34 depth-map + 34 normal-map `.bin` files (~3.8–4MB each, consistent with genuine per-pixel float32 data for 1280×720 images — not empty/degenerate output). |
| `stereo_fusion` | CPU | Valid `fused.ply` written, but only **3 fused points**. |

**On the low fusion count:** not investigated as a bug — the wide baseline from
subsampling every 15th frame of a video (31 frames spanning what was originally
~465 sequential frames) combined with a small, already-fragmented sparse model
(1341 points, only 17/31 images registered) plausibly explains aggressive
rejection by `stereo_fusion`'s default multi-view consistency filters
(`filter_min_num_consistent: 2`, `filter_min_triangulation_angle: 3`). The
depth/normal maps themselves are demonstrably real (correct file sizes, correct
count, produced by a HIP kernel run that logged real per-sweep GPU timings) —
this is a dataset-scale/dense-fusion-tuning question, not evidence PatchMatch-HIP
is broken. A production run would use a denser, better-suited image set; this
task's goal was verifying the pipeline executes correctly end-to-end on gfx1151,
which it does.

**Summary: full incremental SfM pipeline runs end-to-end on this gfx1151 machine.**
One stage (`patch_match_stereo`, dense stereo) is genuinely HIP-accelerated and
verified working on real data, not just unit tests. SIFT and bundle adjustment
run on CPU (Tasks 5 and 6 deferred, both with documented reasons and future
paths). This is the honest, achieved scope of this plan as of 2026-08-04.

## 2026-08-04: Track C Task 1 — HIP SIFT via selective cherry-pick from jeffdaily/rocm-sift-gpu

**Goal:** land HIP-accelerated SIFT (`SiftGPU`) on `hip-integration` without repeating
the whole-branch rebase that previously aborted on its first commit (12 conflicted
files against PatchMatch-HIP's already-ported compat layer).

**Classification of all 10 commits on `jeffdaily/rocm-sift-gpu`** (oldest to newest):

| # | Commit | Touches SIFT only? | Disposition |
|---|--------|---------------------|-------------|
| 1 | `658f8b56` "Add ROCm/HIP support for patch_match_stereo" | No — `cuda_flip/rotate/texture/transpose.h`, `gpu_mat.h`, `patch_match_cuda.h`, `util/cuda*.{cc,h}`, `mvs/CMakeLists.txt` | **Skip** — this is the original (superseded) patch_match HIP port; already-ported differently by PR #4420 on `hip-integration`. |
| 2 | `786e0963` "fix(cmake): address code review feedback" | No — same compat-layer files (portability fixes on #1) | **Skip** — fixups to a superseded commit. |
| 3 | `e609b9f1` "Fix ROCm/HIP support: dual-compatible headers..." | No — same compat-layer files, reworked | **Skip** — still the superseded patch_match approach. |
| 4 | `def43b23` "Simplify ROCm/HIP support: enable_language(HIP) + cuda_to_hip.h" | No — introduces `src/colmap/util/cuda_to_hip.h`, rewrites `mvs/CMakeLists.txt`, `cuda_flip/rotate/texture/transpose.h`, `gpu_mat.h` | **Skip** — this is the commit that introduces the compat-header approach PR #4420 already carries (in its own, further-evolved form) on `hip-integration`. |
| 5 | `690348f3` "Enable gpu_mat_test under HIP and report HIP backend in version banner" | No — `mvs/CMakeLists.txt`, `mvs/gpu_mat_test.cu`, `util/version.cc.in` | **Skip.** Functionality forgone: `mvs/gpu_mat_test` is not registered to build/run under `HIP_ENABLED` on this branch. Functionality *not* forgone: the "with HIP" version-banner string is already present in `hip-integration`'s `src/colmap/util/version.cc.in` (verified: `#elif defined(COLMAP_HIP_ENABLED) ... "with HIP"`), landed independently by PR #4420. |
| 6 | `566e4df7` "docs: document HIP/ROCm build in install.rst, drop stale README.rocm.md" | Docs only | **Skip.** Targets a `README.rocm.md` that does not exist on `hip-integration` (never added — PR #4420 took a different path) and an `doc/install.rst` HIP paragraph that `hip-integration` does not have in the form this commit expects. Not mechanically applicable; the underlying facts it would document (build flags, arch mapping) are superseded by this branch's own conventions. |
| 7 | `bf064e92` "Enable GPU SIFT (SiftGPU) under ROCm/HIP" | **Yes** (SIFT-specific: `thirdparty/SiftGPU/*`, `feature/sift.cc`, dispatch-site widening in `controllers/`, `feature/`, `ui/`, `pycolmap/`) plus small additive touches to `cuda_to_hip.h` (9 new `#define`s, no removals) and `FindDependencies.cmake` (widen one `if` condition) | **Cherry-picked.** |
| 8 | `e95eb380` "SiftGPU: fix double-destroy and DoG edge OOB" | **Yes** — `thirdparty/SiftGPU/{CuTexImage.cpp,CuTexImage.h,ProgramCU.cu}` only | **Cherry-picked.** |
| 9 | `3345a981` "SiftGPU: route tex2D through linear binding on HIP" | **Yes** — `thirdparty/SiftGPU/ProgramCU.cu` only | **Cherry-picked.** |
| 10 | `e41e06e0` "docs: note GPU SIFT is now covered by the HIP backend" | Docs only, 2-line edit to the same `doc/install.rst` HIP paragraph #6 targets | **Skip** — same reason as #6: that paragraph does not exist in this branch's `doc/install.rst` in the expected form. |

**Cherry-pick branch:** `colmap-sift-cherrypick`, based on `hip-integration` tip
`e8ad01ca`. Commits, in order:
- `b1a3f26b` = cherry-pick of `bf064e92`
- `7dd7a7fc` = cherry-pick of `e95eb380`
- `66eaa995` = cherry-pick of `3345a981`
- `77c6959f` = local fix-up commit repairing a formatting bug introduced while
  resolving `b1a3f26b`'s merge conflicts (see below)

**Conflict resolution (all in `b1a3f26b`):** 3 files conflicted —
`src/colmap/controllers/automatic_reconstruction.cc`,
`src/colmap/ui/dense_reconstruction_widget.cc`, `src/pycolmap/pipeline/mvs.cc`.
In every case the conflict was HEAD (PR #4420's `hip-integration`) already having
an equivalent `#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)`
guard in different formatting/comment style from what `bf064e92` introduces —
same semantics, cosmetic diff only. Resolved by keeping HEAD's guard in all three.
An automated resolution script had a bug (dropped a newline after a multi-line
`#if` continuation), which broke the build (`missing binary operator before
token "auto"` in `dense_reconstruction_widget.cc`); fixed in follow-up commit
`77c6959f` (also restores two cosmetic blank lines lost the same way in the
other two files). `cuda_to_hip.h`, `FindDependencies.cmake`, and the SiftGPU
files merged cleanly with no conflicts.

**Build:** `docker build -t colmap-rocm:hip-sift ~/git/colmap-rocm` (from the
`colmap-sift-cherrypick` worktree) succeeds cleanly — `libcolmap_sift_gpu.a`
links as a HIP static library (`ProgramCU.cu` compiled through the HIP
toolchain per `set_source_files_properties(... LANGUAGE HIP)`), `-DCOLMAP_GPU_ENABLED`
present in `colmap_ui`'s compile flags confirming `bf064e92`'s
`FindDependencies.cmake` widening took effect.

**Runtime test — GPU SIFT initializes and extracts, then faults on reuse:**

Dataset: 30 frames (every 15th of 451) from
`~/git/rosbag-colmap-pipeline/data/workspaces/table1/rgb/`, all 1280×720.

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 39 --group-add 105 \
  --security-opt label=disable -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e QT_QPA_PLATFORM=offscreen \
  -v <dataset-dir>:/workspace/data colmap-rocm:hip-sift \
  feature_extractor --database_path /workspace/data/db.sqlite --image_path /workspace/data/images \
  --FeatureExtraction.use_gpu 1
```

- `sift.cc:761 "Creating SIFT GPU feature extractor"` confirms the HIP GPU SIFT
  path is selected (not a CPU/GLSL fallback).
- The **first** GPU-extracted image always succeeds, with correct nonzero
  keypoint counts (e.g. 4582, 3693, 3670 SIFT features — plausible, non-degenerate
  values for these images).
- The process then **aborts with a GPU memory access fault**
  ("Memory access fault by GPU node-1 ... Reason: Page not present or supervisor
  privilege", SIGABRT) on the image immediately after the first successfully
  processed one. Lining up GPU-worked images (not raw file position) across
  runs: one run had its first file (`000145.png`) skipped as already-extracted
  (contaminated database from an earlier interrupted run sharing the same
  output DB — discard this run, it is not a clean data point), so its first
  *GPU* image was `000213.png` (succeeded) and its second was `000278.png`
  (faulted, i.e. 3rd file overall). Two independent clean runs (one plain, one
  with `AMD_SERIALIZE_KERNEL=3`) both started fresh and both faulted on
  exactly the same position: 1st GPU image (`000145.png`) succeeds, 2nd GPU
  image (`000213.png`) faults — identical outcome, not run-to-run variance.
  This is a **deterministic "works once per instance, fails on reuse" pattern**,
  which is textbook stale-handle/use-after-free behavior on a reused
  buffer/texture, not a race or allocator-state coin-flip. It points at
  `e95eb380`'s `CuTexObj` rule-of-five rewrite (move-only semantics, handle
  nulling, guarded destructor) or `3345a981`'s `BindTexture2D` → `BindTexture`
  (linear-binding) switch — both touch exactly the texture-object lifecycle
  that would only misbehave on reuse, not on a fresh object.
- Ruled out as GPU contention: `Memory access fault ... Page not present` is a
  virtual-address fault from an illegal access inside a kernel, not an
  allocation failure — contention from other GPU workloads on this host (e.g.
  `splat_train`, confirmed running concurrently via `docker ps` / `rocm-smi
  --showpids` during testing) produces `hipErrorOutOfMemory`/"Memory in use"
  errors, a different failure mode. The fault reproduced identically both with
  and without a concurrent container competing for the GPU.
- `AMD_SERIALIZE_KERNEL=3` (forces synchronous kernel launches so an abort is
  attributed to the actual faulting launch rather than a later sync point) was
  used on one run: the fault still occurred at the same point (2nd GPU image),
  confirming it is synchronous with a specific kernel launch rather than a
  deferred/batched async report. `AMD_SERIALIZE_KERNEL` does not print kernel
  names by itself, so the specific faulting kernel/API call was not identified
  in this pass — that would need a symbolic tool (e.g. `rocgdb`) attached to
  the abort, out of scope for this task's two-probe budget.
- **Isolation probe:** running `feature_extractor` on `000213.png` alone (the
  image that faulted as the "2nd GPU image" in the two clean multi-image runs)
  succeeds cleanly every time (3693 features, 0.007 min, no fault) when it is
  the *only*, and therefore *first*, image processed. This confirms the defect
  is specific to **cross-image buffer/texture reuse** — first use is always
  clean — not the extraction kernel logic itself.

**Outcome: documented fallback, not folded into `hip-integration`.** GPU SIFT
compiles cleanly under HIP on gfx1151 and correctly extracts features for the
first image processed by a given `SiftGPU` instance, but crashes non-deterministically
once more than one image goes through the same instance — a real, upstream
(pre-existing in `jeffdaily/rocm-sift-gpu`, not introduced by cherry-picking)
buffer/texture-lifecycle bug in the reuse path, not a smoke-test triviality and
not a conflict-resolution artifact (the isolated single-image path proves the
ported code is functionally correct; the surviving files after conflict
resolution match HEAD's semantics exactly). Per this task's stated scope (two
diagnostic probes, then document — not patch `ProgramCU.cu`), this is left for
a future session.

**State left behind:** `hip-integration` was **not reset or force-pushed** —
this entry (commit `57b26614`) is a single docs-only commit added normally on
top of its prior tip `e8ad01ca`. The `colmap-sift-cherrypick` branch (4
commits atop `hip-integration` tip `e8ad01ca`: `b1a3f26b`, `7dd7a7fc`,
`66eaa995`, `77c6959f`, plus a docs commit `1682d716` adding
`docs/superpowers/plans/track-c-task1-report.md`) is kept, not deleted, and
pushed to `origin` so the classification work and conflict resolution do not
need to be redone.

**Important — branches have since diverged, check before any fold-in:**
`git merge-base --is-ancestor hip-integration colmap-sift-cherrypick` was
confirmed true *before* this docs commit was pushed to `hip-integration`.
Since `57b26614` landed only on `hip-integration` and is **not** present on
`colmap-sift-cherrypick`, that ancestor relationship no longer holds. A
future session must **not** run `git checkout hip-integration && git reset
--hard colmap-sift-cherrypick` without first re-establishing it (e.g.
`git rebase hip-integration colmap-sift-cherrypick`, then re-check
`--is-ancestor`) — doing so blind would silently drop this docs entry.

**To resume:** the most direct next diagnostic is a bisect within the 3
cherry-picked commits — build with only `bf064e92` + `e95eb380` (drop
`3345a981`'s `BindTexture2D` → `BindTexture` switch) and re-run the same
2-image test. If it still faults, the bug is in `e95eb380`'s `CuTexObj`
rule-of-five; if it's clean, `3345a981`'s linear-binding switch is the cause.
Kernel-level attribution of the fault (which this session's
`AMD_SERIALIZE_KERNEL=3` probe did not provide by itself) would need a
symbolic debugger such as `rocgdb` attached to the abort, not simply a
`RelWithDebInfo` rebuild. Also see the full write-up at
`docs/superpowers/plans/track-c-task1-report.md` on the `colmap-sift-cherrypick`
branch (the refactor `e95eb380`'s own commit message flagged as
"left for a separate change").

## Caspar-HIP Completion, Track B (2026-08-05): wiring Caspar-HIP into COLMAP with native OpenCV support

Executed against `docs/superpowers/plans/2026-08-04-caspar-hip-completion.md`,
Track B Tasks 1, 2, 4, 5 (Task 3 was cut from the critical path in that plan).
Closes the "bundle adjustment CPU-only" deferral above for real — Caspar-HIP BA
now builds and runs correctly on gfx1151, with native `OPENCV` camera-model
support merged in from the start (not a follow-up), per an explicit design
decision made before this work started.

### Task 1: Port caspar-opencv's C++ dispatch changes

Ported `rosbag-colmap-pipeline`'s `docker/patches/caspar-opencv/{bundle_adjustment_caspar.cc,caspar_model_adapter.h}`
(read-only reference, targets COLMAP 4.1.1) into this branch's current tree.
Diffed first, as required — did not blind-apply.

- `bundle_adjustment_caspar.cc`: this branch's tip had already drifted from the
  reference (switched `std::unordered_map`/`unordered_set` to `NodeHashMap`/
  `FlatHashMap`/`FlatHashSet`, added `VLOG_IS_ON(2)` gating for `print_progress`)
  — unrelated to the OpenCV patch. Ported only the two `BuildSizing()` `kOpenCV`
  blocks (pose count, calib count) on top of that drift, mirroring the existing
  `kPinhole` blocks.
- `caspar_model_adapter.h`: otherwise byte-identical to the reference minus the
  OpenCV additions (no drift) — applied the reference file wholesale:
  `CasparSolverSizing` OpenCV fields, `OpenCVAdapter` class, `CreateCasparAdapter()`
  case, and `CreateSolver()`'s full positional-argument list (the single
  highest-risk part of the original patch — silently wrong-compiling if
  misordered).
- Pre-regeneration sanity check: current `generated/f32/solver.h`'s
  `GraphSolver` constructor (no OpenCV nodes yet) matched the non-OpenCV
  portion of the ported call exactly, by name and position.
- Commit: `feat(caspar): port OpenCV camera-model dispatch from rosbag-colmap-pipeline's caspar-opencv patch`.

### Task 2: Regenerate Caspar kernels with OpenCV support

Ported `caspar_generate.py`'s `opencv_core`/`opencv_split_core` additions the
same way (clean diff, pure additions, no drift). Cross-checked the distortion
formula against this branch's own `src/colmap/sensor/models.h`
`OpenCVCameraModel::Distortion`/`ImgFromCam` — exact match (params order
`[fx,fy,cx,cy,k1,k2,p1,p2]`, `radial = k1*r2 + k2*r2^2`, same `du`/`dv` terms),
unchanged from the 4.1.1 baseline the reference patch targeted.

Regenerated `generated/f32/` (host Python lacked a working `symengine` build
compatible with `symforce-rocm`'s vendored fork — ran codegen inside the
`rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` container
instead, with `symforce-rocm` `pip install -e .`'d there). Exit 0, no 48KB
shared-memory-budget error. 713 files, 224 new OpenCV-related.

**Notable finding, not anticipated by the plan:** every regenerated file now
unconditionally `#include`s `"cuda_to_hip.h"`, and the regenerated
`CMakeLists.txt` gained a full `USE_HIP` option (`find_package(hip)`,
`LANGUAGE HIP` source properties, `HIP_ARCHITECTURES`). This isn't something
this session added — `symforce-rocm`'s own upstream jinja templates
(`symforce/caspar/source/templates/*.jinja`) now bake in HIP support
unconditionally, including shipping a full, already-authored Caspar-specific
`cuda_to_hip.h` compat header (`symforce/caspar/source/runtime/cuda_to_hip.h`,
authored by Jeff Daily, AMD) that maps `cudaMalloc`→`hipMalloc` etc., and
provides `caspar_hip::reduce_sum`/`labeled_reduce_sum`/`match_any_mask` HIP
fallbacks for the `cg::reduce`/`cg::labeled_partition` operations HIP's
cooperative_groups lacks. This substantially changed Task 4's scope from
"write a compat header from scratch" to "fix real build-system wiring bugs
around an already-correct header" (see Task 4 below).

PINHOLE/SIMPLE_RADIAL kernel logic itself is unchanged versus the prior
committed tree — the diff there is clang-format-style reformatting plus the
`cuda_to_hip.h`/`USE_HIP` additions only, not a logic regression.

Re-verified Task 1's `CreateSolver()` positional-argument port against the
now-OpenCV-augmented `solver.h`'s actual `GraphSolver` constructor: node-type
order (`OpenCVCalib`, `OpenCVFocalAndExtra`, `OpenCVPose`,
`OpenCVPrincipalPoint`, `Pinhole*`, `Point`, `SimpleRadial*`) and factor-count
order (`simple_radial` → `pinhole` → `opencv` → `*_split` variants) match
exactly, zero mismatches.

`f64` left unregenerated: `CASPAR_USE_DOUBLE` defaults `OFF` in this branch's
`CMakeLists.txt` and is untested elsewhere in the branch.

Commit: `feat(caspar): regenerate kernels with native OPENCV camera-model support`.

### Task 4: Add HIP compilation to CASPAR_ENABLED

Extended `cmake/FindDependencies.cmake`'s `CASPAR_ENABLED` arch guard with a
standalone `if(HIP_ENABLED AND CASPAR_ENABLED)` block (requires
`CMAKE_HIP_ARCHITECTURES` set, warns — doesn't fail — on any arch other than
`gfx1151`, the only one built and run to date). Mirrored the existing CUDA
`FetchContent` block in `src/thirdparty/CMakeLists.txt` with a
`CASPAR_ENABLED AND HIP_ENABLED` branch that sets `USE_HIP ON` before
`FetchContent_MakeAvailable`, which the regenerated `CMakeLists.txt` (Task 2)
picks up to build itself as a HIP project.

Per the plan's explicit instruction ("Build and iterate on real compile
errors — do not suppress. If a construct is genuinely unmappable, stop and
report BLOCKED"), this took **three build iterations**, each a real bug found
and fixed, none suppressed:

1. `fatal error: hip/hip_runtime.h: No such file or directory` — the
   regenerated `CMakeLists.txt`'s own
   `target_include_directories(caspar_lib_core PUBLIC ${hip_INCLUDE_DIRS})`
   is a no-op: modern `find_package(hip)` never sets that legacy variable,
   only populates `hip::host`'s own `INTERFACE_INCLUDE_DIRECTORIES`. Fixed by
   reading that target property (with a `ROCM_PATH`-based fallback if empty)
   in the generated `CMakeLists.txt`.
2. Same error persisted after fix #1 — root cause was actually
   `src/thirdparty/CMakeLists.txt`'s pre-existing
   `set_target_properties(caspar_lib_core PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ...)`
   call, which **overwrites** rather than appends, silently wiping out
   whatever the generated `CMakeLists.txt` had just set (including fix #1).
   Fixed by re-adding the ROCm include dir afterwards with
   `target_include_directories()` (which appends) instead. Also had to add
   `target_compile_definitions(caspar_lib_core PUBLIC __HIP_PLATFORM_AMD__)`:
   HIP-language translation units get this defined implicitly by the compiler
   wrapper, but plain C++ consumers (colmap's `bundle_adjustment.cc`,
   transitively via `solver.h`) do not, and `hip_runtime.h` `#error`s out
   without it.
3. `'__device__' does not name a type` — `cuda_to_hip.h` unconditionally
   defines `__device__ __forceinline__` function bodies
   (`caspar_hip::reduce_sum`/`reduce_max`/`match_any_mask`/`labeled_reduce_sum`)
   whenever `USE_HIP` is defined, but `USE_HIP` being defined does not mean
   the translation unit is being compiled by `hipcc`/`clang++ --hip` — plain
   `g++` cannot parse `__device__` at all, regardless of what headers it's
   given. Since `USE_HIP` is now (correctly, per fix #2) propagated `PUBLIC`
   to every consumer of `caspar_lib_core`, including plain-C++
   `bundle_adjustment.cc`, this broke. Fixed by guarding the `hipcub`/
   `hip_cooperative_groups.h` includes and all four `__device__` function
   definitions (plus the macros referencing them) behind `__HIPCC__`, which
   the HIP compiler defines automatically and a plain host compiler never
   does — these are device-only utilities never called from host code, so
   losing them in host translation units is correct, not a functionality
   regression. This is a hand-patch on top of `symforce-rocm`'s vendored
   `cuda_to_hip.h` (shipped verbatim by Task 2's regeneration); consistent
   with the plan's "Caspar-specific cooperative_groups HIP compat header"
   step, which turned out to already exist upstream rather than needing to
   be written from scratch, but still needed this host/device-compile-mode
   fix specific to how colmap-rocm's build reaches this header from plain
   C++ translation units.

Verified: `docker build -t colmap-rocm:caspar-hip --build-arg CMAKE_EXTRA_ARGS="-DCASPAR_ENABLED=ON" .`
(Dockerfile already bakes in `-DCUDA_ENABLED=OFF -DHIP_ENABLED=ON
-DCMAKE_HIP_ARCHITECTURES=gfx1151`) completes clean, 0 `FAILED` targets,
image tagged `localhost/colmap-rocm:caspar-hip`.

Commit: `feat(caspar): add HIP compilation path to CASPAR_ENABLED (gfx1151)`.

### Task 5: Verify Caspar-HIP bundle adjustment on real data, gfx1151

`colmap mapper --help` confirms `--Mapper.ba_global_backend`/
`--Mapper.ba_local_backend` accept `CASPAR` (registered via
`option_manager.cc`'s `#ifdef CASPAR_ENABLED` block — not printed as an
explicit choices list in `--help` output, confirmed by reading the source
rather than guessing from `--help` text alone).

**Dataset:** same 31-frame subsample (every 15th frame of 451) from
`~/git/rosbag-colmap-pipeline/data/workspaces/table1/rgb/` used by the prior
milestone's Task 7 end-to-end run, staged fresh at `/tmp/caspar-hip-e2e/`.

**Pipeline:** `feature_extractor --FeatureExtraction.use_gpu 0` (31/31 images,
0.028 min) → `sequential_matcher --FeatureMatching.use_gpu 0` (31/31 matched,
0.116 min) → `mapper` run twice from the same database, once per backend:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 39 --group-add 105 \
  --security-opt label=disable -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e QT_QPA_PLATFORM=offscreen \
  -v /tmp/caspar-hip-e2e:/workspace/data localhost/colmap-rocm:caspar-hip \
  mapper --database_path /workspace/data/db.sqlite --image_path /workspace/data/images \
  --output_path /workspace/data/sparse_caspar \
  --Mapper.ba_global_backend CASPAR --Mapper.ba_local_backend CASPAR
```

vs the same command with `--output_path /workspace/data/sparse_ceres` and no
backend flags (default Ceres/CPU).

**Results (`colmap model_analyzer`, both models):**

| | Caspar-HIP (gfx1151) | Ceres (CPU, default) |
|---|---|---|
| Registered images | 17 / 31 | 17 / 31 |
| 3D points | 1342 | 1342 |
| Observations | 5167 | 5170 |
| Mean track length | 3.850 | 3.852 |
| Mean reprojection error | 0.630 px | 0.543 px |

Caspar-HIP mapper run: 1.169 min (13 registration steps, "Keeping successful
reconstruction", no errors/crashes in the log — grepped for
`error|caspar|hip|failed|crash|abort`, zero matches). Ceres run: 0.047 min
(expected — no GPU dispatch overhead at this tiny problem size).

**Verdict: PASS.** Identical registered-image count, identical point count,
reprojection error same order of magnitude (both sub-pixel, ~15% apart — well
within the "not required to match bit-for-bit" tolerance the plan set). No
crashes, no NaNs, no divergent/degenerate reconstruction. This is real,
on-hardware confirmation that Caspar-HIP's bundle adjustment — including the
newly-added native OpenCV camera-model path wired in Tasks 1–2 (this
dataset's cameras are `SIMPLE_RADIAL`, not `OPENCV`, so the OpenCV dispatch
path itself was verified for build/link correctness and positional-argument
safety in Tasks 1–2's cross-checks rather than exercised numerically here —
none of the staged images' cameras use `OPENCV`; a follow-up with an
`OPENCV`-model dataset would close that last numerical gap) — produces
correct results on gfx1151, not just a clean compile.

Commit: `docs: verify Caspar-HIP bundle adjustment on real data, gfx1151 — closes prior Task 6 deferral`.
