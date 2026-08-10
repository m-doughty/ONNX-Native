# Releasing ONNX::Native

Two release cadences run independently:

1. **Binary release** — bumps `BINARY_TAG`, triggers
   `.github/workflows/build-binaries.yml` to publish per-platform
   tarballs to GitHub Releases. Drives the prebuilt-download path
   in `Build.rakumod`.
2. **Raku dist release** — bumps the version in `META6.json` /
   `dist.ini`, runs `mi6 release` (which updates `Changes`, tags,
   publishes to Zef). Pure-Raku changes that don't touch the shim
   only need this one.

The two are decoupled because most Raku-side changes don't need
a fresh shim build, but every shim change requires both.

## Binary release

1. **Bump `BINARY_TAG`** to a new revision: edit both
   `BINARY_TAG` (root) and `resources/BINARY_TAG`. Format:
   `binaries-onnxruntime-<ver>-r<n>`. Increment `r<n>` for any
   shim source change; bump `<ver>` only when moving to a new
   ORT version (also update `ORT_VERSION` in
   `.github/workflows/build-binaries.yml` and `$MIN-GLIBC` in
   `Build.rakumod` if the new ORT raises the floor).
2. **Manual GPU smoke** (see below) on at least one Linux NVIDIA
   box and one Windows NVIDIA box. CI builds the GPU bundles but
   doesn't test them — there are no free GPU runners.
3. **Trigger the build** via Actions → `build-binaries` →
   "Run workflow" (or `gh workflow run build-binaries.yml`). The
   workflow's release step reads `BINARY_TAG` from the repo and
   creates / updates the matching GitHub Release with all
   per-platform artefacts attached. Tag-push (`binaries-*`) also
   works as a trigger but isn't required — manual dispatch is
   the established path here.
4. **Paste the new checksums**: download `checksums.txt` from the
   Release, replace `resources/checksums.txt`'s body (keep the
   header comments intact). Commit.
5. **Bump Raku dist version** (`META6.json` / `dist.ini`) so the
   next `zef install ONNX::Native` pulls the right tag.

## Raku-only release

1. Make changes; run `prove6 t/`.
2. Bump version in `META6.json` + `dist.ini`.
3. `mi6 release` (handles `Changes` + tag + publish).

## Manual GPU smoke checklist

CI verifies the GPU bundles compile + link, but never runs them.
Before any binary release that touches the shim or the build
path, do this on real hardware:

### Linux NVIDIA

```sh
ONNX_NATIVE_WITH_CUDA=1 zef install . --force-install
ONNX_NATIVE_TEST_GPU=1 prove6 -Ilib xt/05-gpu-smoke.rakutest
```

Expected: all subtests pass. The CUDA-provider register call
should produce no warnings about missing libs (would indicate a
broken `libcudnn` / `libcublas` resolve at @rpath).

If you have a multi-GPU box (your A100 cluster), edit the
`:cuda-device-id` test to point at a non-zero device for
additional coverage.

### Windows NVIDIA

```powershell
$env:ONNX_NATIVE_WITH_CUDA="1"
zef install . --force-install
$env:ONNX_NATIVE_TEST_GPU="1"
prove6 -Ilib xt/05-gpu-smoke.rakutest
```

Expected: same. Common failure mode is `cudnn64_9.dll` not
being found — Windows DLL search rules can hijack a different
`cudnn` version installed elsewhere on PATH. Verify the staged
dir's `cudnn64_9.dll` is the one being loaded
(`Process Hacker` / `dumpbin /dependents` of the running
`raku.exe`).

### Negative coverage

After the GPU smoke passes, do one CPU-only run with
`ONNX_NATIVE_WITH_CUDA` UNSET to confirm the existing CPU path
still works on the same machine:

```sh
unset ONNX_NATIVE_WITH_CUDA
zef install . --force-install
prove6 -Ilib t/
```

This catches the case where a GPU-staging change accidentally
broke CPU staging. The two are siblings under
`$XDG_DATA_HOME/ONNX-Native/`; reinstalling without the env var
should pick up the CPU bundle and stage it cleanly alongside any
existing GPU bundle (or sweep it via `cleanup-old-stages` if
`ONNX_NATIVE_KEEP_OLD_STAGES` isn't set).
