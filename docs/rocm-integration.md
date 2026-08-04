# ROCm/HIP integration log

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
