[![Actions Status](https://github.com/m-doughty/ONNX-Native/actions/workflows/test.yml/badge.svg)](https://github.com/m-doughty/ONNX-Native/actions)

NAME
====

ONNX::Native - Minimal Raku bindings for ONNX Runtime (inference only)

SYNOPSIS
========

```raku
use ONNX::Native;
use ONNX::Native::Types;

# Load a model, introspect its I/O surface.
my $session = ONNX::Native::Session.new(
	:path<model.onnx>,
	:providers<coreml cpu>,   # macOS: CoreML first, fall back to CPU
);

say $session.input-names;       # <input_ids attention_mask>
say $session.output-names;      # (logits)
say $session.input-info('input_ids').shape;   # (-1, -1)  (dynamic)

# Run inference.
my $ids  = ONNX::Native::Tensor.from-ints(
	@token-ids, :shape([1, @token-ids.elems]), :dtype(INT64));
my $mask = ONNX::Native::Tensor.from-ints(
	@attention,  :shape([1, @attention.elems]),  :dtype(INT64));

my %out = $session.run(
	inputs  => { input_ids => $ids, attention_mask => $mask },
	outputs => ['logits',],
);

my @logits = %out<logits>.to-num-array;
my @shape  = %out<logits>.shape;
```

STATUS
======

This is v0.1 — the minimum surface needed to run classification, embedding, and small-model inference on top of HuggingFace-exported ONNX files. Scoped deliberately tight so the full surface can land on CI without stretching review.

What works
----------

  * Load models from disk path or in-memory Blob.

  * Introspect input/output counts, names, tensor shapes and dtypes.

  * Create input tensors from Raku lists (Int / Num) or from existing Blobs (zero-copy).

  * Run inference with CPU or — on macOS — CoreML.

  * Read output tensors as Raku lists (Num / Int) or raw Blobs.

  * Explicit `.dispose` and automatic `DESTROY` cleanup on sessions and tensors.

What's deferred to later versions
---------------------------------

  * GPU providers beyond the "CPU + platform accelerator" default (CUDA on Linux, DirectML on Windows, ROCm / TensorRT / CoreML on non-macOS, QNN, OpenVINO).

  * Training API, IO Binding, sparse / sequence / map tensor types.

  * String / fp16 / bf16 / int8 / int4 tensor I/O.

  * Fine-grained threading configuration (intra/inter-op pools).

  * Async `RunAsync`.

  * Model metadata API and profiling.

If you need something from the deferred list, file an issue or PR — none of them are deep changes, they're just more of the same shim-and-wrap pattern and I'd rather ship a tight v0.1 first.

INSTALLATION
============

macOS / Linux — prebuilt
------------------------

```shell
zef install ONNX::Native
```

This downloads Microsoft's official ONNX Runtime prebuilt for your platform from their GitHub Releases (verified via SHA256 against bundled checksums), stages it under `$XDG_DATA_HOME/ONNX-Native/`, and compiles a tiny C shim against it. Total install size: ~25 MB for the ORT dylib plus a ~35 KB shim.

No system-wide changes are made. The staged libs live under your user's data dir; `zef uninstall ONNX::Native` leaves them in place (harmless, safe to `rm -rf`).

macOS / Linux — system ORT
--------------------------

If your system already has libonnxruntime and its headers (e.g. `brew install onnxruntime`, or an apt package providing `libonnxruntime-dev`), set:

```shell
ONNX_NATIVE_PREFER_SYSTEM=1 zef install ONNX::Native
```

Build.rakumod probes Homebrew (`/opt/homebrew` / `/usr/local`) and common Linux lib paths for the runtime and header set. It compiles the shim against what it finds and skips staging.

Linux — CUDA GPU
----------------

```shell
ONNX_NATIVE_WITH_CUDA=1 zef install ONNX::Native
```

Picks the GPU variant of Microsoft's Linux prebuilt (ORT + CUDA EP + TensorRT EP) and compiles the shim with the CUDA provider registration path enabled. Requires CUDA + cuDNN installed on the system at runtime. Not yet exercised in CI — treat as experimental.

Windows
-------

Windows is not supported in v0.1 (DirectML provider is deferred). Planned for v0.2.

Environment variables
---------------------

<table class="pod-table">
<tbody>
<tr> <td>ONNX_NATIVE_PREFER_SYSTEM</td> <td>Skip prebuilt download, compile shim against system libonnxruntime</td> </tr> <tr> <td>ONNX_NATIVE_BINARY_ONLY</td> <td>Refuse system fallback; fail if prebuilt unavailable</td> </tr> <tr> <td>ONNX_NATIVE_BINARY_URL</td> <td>Alternate GitHub release base URL (default: Microsoft upstream)</td> </tr> <tr> <td>ONNX_NATIVE_CACHE_DIR</td> <td>Override download cache dir (default: $XDG_CACHE_HOME)</td> </tr> <tr> <td>ONNX_NATIVE_DATA_DIR</td> <td>Override staged-libs base dir (default: $XDG_DATA_HOME)</td> </tr> <tr> <td>ONNX_NATIVE_LIB_DIR</td> <td>Runtime: load shim from this dir instead of the staged dir</td> </tr> <tr> <td>ONNX_NATIVE_WITH_CUDA</td> <td>Opt in to the GPU prebuilt variant on Linux (install-time)</td> </tr>
</tbody>
</table>

API
===

ONNX::Native::Session
---------------------

### new(:$path!, :@providers = [CPU], :$log-id = 'onnx-native')

Load a model from disk. `:@providers` can be a list of `Provider` enum values or lowercase strings; they're tried in order at inference time (per-node), falling back to CPU for any op the chosen provider can't handle.

Throws `X::ONNX::Native::Error` if the model can't be opened. Throws `X::ONNX::Native::ProviderUnavailable` if a requested provider isn't compiled into this build of the shim.

### new(:$bytes!, :@providers, :$log-id)

Load a model from an in-memory Blob.

### input-count(), output-count() → Int

Number of inputs / outputs in the model graph.

### input-names(), output-names() → List[Str]

Input / output names in index order. Cached after first call.

### input-info($name), output-info($name) → TensorInfo

Tensor spec (element type + shape) for the named I/O. Dimensions of `-1` denote symbolic / dynamic shapes.

### run(:%inputs!, :@outputs!) → Hash[Str, Tensor]

Run inference. `%inputs` maps each input name to a Tensor. `@outputs` lists output names to fetch. Returns a hash of output tensors.

### dispose() / DESTROY

Release the underlying native handles. Idempotent; safe to call manually or rely on GC.

ONNX::Native::Tensor
--------------------

### from-blob($data, :@shape!, :$dtype = FLOAT32) → Tensor

Construct a tensor that borrows the given Blob's bytes. The Blob is kept alive for the tensor's lifetime — ORT holds a pointer into its backing storage and doesn't copy on the way in.

### from-nums(@values, :@shape!, :$dtype = FLOAT32) → Tensor

FLOAT32 or DOUBLE tensor from a flat Raku list of numbers.

### from-ints(@values, :@shape!, :$dtype = INT64) → Tensor

INT32 or INT64 tensor from a flat Raku list of integers. Default dtype is INT64 because HuggingFace tokenizers emit int64 token IDs by convention.

### shape() → List[Int]

Concrete dimension list.

### dtype() → DType

Tensor element type.

### byte-length() → Int

Total bytes in the tensor's data (`elem-count × dtype-byte-size`).

### to-num-array() → List[Num]

Decode the tensor's data as a flat list of Num. Requires FLOAT32 or DOUBLE dtype.

### to-int-array() → List[Int]

Decode as a flat list of Int. Accepts any integer dtype (INT8 .. INT64, UINT8 .. UINT32, BOOL).

### to-blob() → Blob

Copy the tensor's data into a fresh Raku Blob. Independent of the tensor; survives disposal.

### dispose() / DESTROY

Release the underlying OrtValue. Input tensors drop their Blob anchor; output tensors free ORT-owned storage.

ONNX::Native::Types
-------------------

### DType enum

`FLOAT32 INT32 INT64 UINT8 INT8 BOOL DOUBLE UINT16 INT16 UINT32 UINT64 FLOAT16 BFLOAT16 STRING`. Only the first three support Raku ↔ ORT round-trips in v0.1. The rest decode for introspection but can't yet be constructed from Raku data.

### Provider enum

`CPU COREML CUDA DML`. CPU is always available. COREML is available on macOS. CUDA / DML require opt-in prebuilt variants.

### TensorInfo

Static description of a tensor's shape + dtype. Returned by `Session.input-info` / `Session.output-info`.

### Exceptions

  * `X::ONNX::Native::Error` — base class; `$.code` is an OrtErrorCode value, `$.reason` is the raw message.

  * `X::ONNX::Native::Unsupported` — a feature isn't implemented in this version of ONNX::Native.

  * `X::ONNX::Native::ProviderUnavailable` — requested execution provider isn't compiled into this shim build.

MEMORY MANAGEMENT
=================

Session and Tensor both manage native resources via `DESTROY`. Normal Raku GC releases them eventually, but if you're running many inferences in a tight loop, explicit `.dispose` on intermediate tensors reclaims memory faster:

```raku
for @batches -> $batch {
	my $input = ONNX::Native::Tensor.from-ints($batch, :shape([1, $batch.elems]));
	my %out = $session.run(:inputs({input_ids => $input}), :outputs(['logits',]));
	process(%out<logits>.to-num-array);
	$input.dispose;
	.dispose for %out.values;
}
```

Input tensors built with `from-blob` keep the backing Blob alive via an internal anchor — don't worry about the Blob being GC'd out from under ORT mid-inference.

BINARY_TAG
==========

The `BINARY_TAG` file at the dist root pins the ONNX Runtime prebuilt version. Format: `binaries-onnxruntime-E<lt>verE<gt>-rE<lt>revE<gt>`, e.g. `binaries-onnxruntime-1.20.0-r1`. The Raku dist version (e.g. 0.1.3) bumps independently for Raku-only bugfix releases that reuse the same ORT binary tag.

At install time Build.rakumod reads BINARY_TAG, fetches the matching tarball from `microsoft/onnxruntime`'s GitHub Releases, verifies SHA256 against `resources/checksums.txt`, and stages the libs under `$XDG_DATA_HOME/ONNX-Native/E<lt>tagE<gt>/`.

PLATFORM NOTES
==============

macOS
-----

CoreML EP is baked into Microsoft's macOS prebuilts since ORT 1.16 — pass `:providers<coreml cpu>` to `Session.new` to use it. On Apple Silicon the Neural Engine handles most common ops; falls back to CPU for the rest.

Gatekeeper may quarantine the downloaded dylib the first time it's loaded. If you hit a signing error, `xattr -dr com.apple.quarantine ~/.local/share/ONNX-Native/` clears it.

Linux
-----

Microsoft's prebuilts target glibc 2.31+ (Ubuntu 20.04). Older systems (CentOS 7, Ubuntu 18.04) trigger the system-libonnxruntime fallback — install `libonnxruntime-dev` from your package manager or build from source.

Apple Silicon note
------------------

If you hit a `dyld[...]: Library not loaded: @rpath/libonnxruntime.*` error at runtime, the shim didn't pick up the `@loader_path` rpath — most likely because you moved the shim out of its staged directory. Either move it back or rebuild with `ONNX_NATIVE_LIB_DIR=/path/to/dir`.

LIMITATIONS
===========

See the "What's deferred" list under STATUS. Major limitations worth calling out explicitly:

  * No Windows support in v0.1.

  * CUDA path on Linux is untested in CI.

  * Input / output tensors are CPU-backed only. GPU tensors (CUDA-device or CoreML-device) require IO Bindings which are deferred.

  * Model shape introspection returns up to 16 dims; higher-rank tensors error. 16 is plenty for anything outside niche research use.

AUTHOR
======

  * Matt Doughty

COPYRIGHT AND LICENSE
=====================

Copyright 2026 Matt Doughty

This library is free software; you can redistribute it and/or modify it under the Artistic License 2.0.

