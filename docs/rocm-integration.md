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
