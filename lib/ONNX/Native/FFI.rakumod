unit module ONNX::Native::FFI;

use NativeCall;

# === Library resolution ===
#
# Resolve the onnx_native_shim library in this order:
#
#   1. $ONNX_NATIVE_LIB_DIR env var — explicit override. Full path
#      to a directory containing libonnx_native_shim. Escape hatch
#      for custom ORT builds / dev iteration; you take
#      responsibility for ABI compatibility.
#   2. XDG-staged dir (Build.rakumod's prebuilt-download path) —
#      $XDG_DATA_HOME/ONNX-Native/<binary-tag>/lib/. Filenames
#      preserved so libonnxruntime's own @loader_path / $ORIGIN
#      refs resolve correctly when the shim is loaded.
#   3. Bare library name — let the OS dynamic loader find the shim
#      via LD_LIBRARY_PATH / DYLD_FALLBACK_LIBRARY_PATH. Useful
#      for ad-hoc dev (symlink the shim into /usr/local/lib).
#
# The shim is the only native lib we bind to from Raku. libonnxruntime
# is a transitive dep of the shim and gets loaded automatically by the
# OS dynamic loader via the shim's LC_LOAD_DYLIB / DT_NEEDED entry.

constant $os = $*KERNEL.name.lc;
constant $ext = $os ~~ /darwin/ ?? 'dylib'
             !! $*DISTRO.is-win ?? 'dll'
             !! 'so';

sub _staged-lib-dir(--> IO::Path) {
	# %?RESOURCES<BINARY_TAG> can misbehave during zef's dep-
	# resolution compile pass — it sometimes returns the resources
	# directory itself when the resources dict isn't fully
	# populated. Insist on .f (regular file) and try {} the slurp
	# so a bad value falls through cleanly. Also cope with the
	# "no staging at all" case (module used standalone without
	# Build.rakumod having populated resources yet) by returning
	# an undefined IO::Path rather than Nil — Nil can't bind to
	# IO::Path-typed params downstream.
	my $res = %?RESOURCES<BINARY_TAG>;
	my Str $tag = '';
	if $res.defined && $res.IO.f {
		$tag = (try $res.IO.slurp.trim) // '';
	}
	return IO::Path unless $tag.chars;
	my Str $base = %*ENV<ONNX_NATIVE_DATA_DIR>
		// %*ENV<XDG_DATA_HOME>
		// ($*DISTRO.is-win
				?? (%*ENV<LOCALAPPDATA>
						// "{%*ENV<USERPROFILE> // '.'}\\AppData\\Local")
				!! "{%*ENV<HOME> // '.'}/.local/share");
	"$base/ONNX-Native/$tag/lib".IO;
}

#| Find a library by basename (no extension) in $dir. Accepts
#| versioned variants (libonnxruntime.1.20.0.dylib, libfoo.so.42)
#| for completeness, though the shim itself is always plain
#| libonnx_native_shim.<ext>.
sub _find-in(IO::Path $dir, Str $name --> Str) {
	return Str unless $dir.defined && $dir.d;
	my $exact = $dir.add("$name.$ext");
	return $exact.Str if $exact.e;

	for $dir.dir -> $entry {
		next unless $entry.e;
		my $bn = $entry.basename;
		return $entry.Str if $bn.starts-with("$name.") && $bn.contains(".$ext");
		return $entry.Str if $bn.starts-with("$name-") && $bn.ends-with(".$ext");
	}
	Str;
}

#| Resolve the shim's library path. Returns either an absolute
#| path (env override or XDG-staged) or the bare short name
#| ('onnx_native_shim') as a loader fallback.
sub _resolve-shim(--> Str) {
	if (my $override = %*ENV<ONNX_NATIVE_LIB_DIR>) && $override.IO.d {
		with _find-in($override.IO, 'libonnx_native_shim') { return $_ }
	}
	with _find-in(_staged-lib-dir(), 'libonnx_native_shim') { return $_ }
	'onnx_native_shim';
}

constant $shim-lib is export = _resolve-shim();

# === Opaque handle types ===
#
# All four are plain CPointer subclasses; the shim treats them
# as opaque `struct Ort*`.

class OrtEnvHandle            is repr('CPointer') is export { }
class OrtSessionHandle        is repr('CPointer') is export { }
class OrtSessionOptionsHandle is repr('CPointer') is export { }
class OrtValueHandle          is repr('CPointer') is export { }

# === ONNXTensorElementDataType constants ===
#
# Exposed for use by Tensor.from-blob callers passing raw element
# type integers. Mirrors the enum in onnxruntime_c_api.h.

constant ONNX_TENSOR_TYPE_UNDEFINED  is export =  0;
constant ONNX_TENSOR_TYPE_FLOAT      is export =  1;
constant ONNX_TENSOR_TYPE_UINT8      is export =  2;
constant ONNX_TENSOR_TYPE_INT8       is export =  3;
constant ONNX_TENSOR_TYPE_UINT16     is export =  4;
constant ONNX_TENSOR_TYPE_INT16      is export =  5;
constant ONNX_TENSOR_TYPE_INT32      is export =  6;
constant ONNX_TENSOR_TYPE_INT64      is export =  7;
constant ONNX_TENSOR_TYPE_STRING     is export =  8;
constant ONNX_TENSOR_TYPE_BOOL       is export =  9;
constant ONNX_TENSOR_TYPE_FLOAT16    is export = 10;
constant ONNX_TENSOR_TYPE_DOUBLE     is export = 11;
constant ONNX_TENSOR_TYPE_UINT32     is export = 12;
constant ONNX_TENSOR_TYPE_UINT64     is export = 13;
constant ONNX_TENSOR_TYPE_BFLOAT16   is export = 16;

# === OrtErrorCode constants ===

constant ORT_OK                is export =  0;
constant ORT_FAIL              is export =  1;
constant ORT_INVALID_ARGUMENT  is export =  2;
constant ORT_NO_SUCHFILE       is export =  3;
constant ORT_NO_MODEL          is export =  4;
constant ORT_ENGINE_ERROR      is export =  5;
constant ORT_RUNTIME_EXCEPTION is export =  6;
constant ORT_INVALID_PROTOBUF  is export =  7;
constant ORT_MODEL_LOADED      is export =  8;
constant ORT_NOT_IMPLEMENTED   is export =  9;
constant ORT_INVALID_GRAPH     is export = 10;
constant ORT_EP_FAIL           is export = 11;

# === Error slot convention ===
#
# The shim populates char** out_error on failure by malloc()ing a
# UTF-8 string. We model this as CArray[Pointer[uint8]] so we can
# hold the raw C pointer — binding as CArray[Str] would have
# NativeCall decode on read and re-encode into a fresh buffer on
# write, and passing that fresh buffer to the shim's free() would
# corrupt the heap.
#
# Allocate an error slot with err-slot() and extract contents with
# ffi-extract-error() — both defined at the bottom of this module.

# === Shim bindings ===

sub onnx_shim_api_version(--> int32)
	is native($shim-lib) is export { * };

# --- Env ---

sub onnx_shim_init(
	Str,                       # log_id
	CArray[OrtEnvHandle],      # *OrtEnv** out_env
	CArray[Pointer[uint8]],    # out_error
	--> int32
) is native($shim-lib) is export { * };

sub onnx_shim_release_env(OrtEnvHandle)
	is native($shim-lib) is export { * };

# --- Session options ---

sub onnx_shim_create_session_options(
	CArray[OrtSessionOptionsHandle],
	CArray[Pointer[uint8]],
	--> int32
) is native($shim-lib) is export { * };

sub onnx_shim_release_session_options(OrtSessionOptionsHandle)
	is native($shim-lib) is export { * };

sub onnx_shim_enable_provider(
	OrtSessionOptionsHandle,
	Str,                       # provider_name
	int64,                     # flags
	CArray[Pointer[uint8]],
	--> int32
) is native($shim-lib) is export { * };

# --- Session ---

sub onnx_shim_create_session_from_path(
	OrtEnvHandle,
	Str,                       # model_path
	OrtSessionOptionsHandle,
	CArray[OrtSessionHandle],
	CArray[Pointer[uint8]],
	--> int32
) is native($shim-lib) is export { * };

sub onnx_shim_create_session_from_buffer(
	OrtEnvHandle,
	Pointer[uint8],            # buf
	size_t,                    # len
	OrtSessionOptionsHandle,
	CArray[OrtSessionHandle],
	CArray[Pointer[uint8]],
	--> int32
) is native($shim-lib) is export { * };

sub onnx_shim_release_session(OrtSessionHandle)
	is native($shim-lib) is export { * };

# --- Introspection ---

sub onnx_shim_session_input_count(
	OrtSessionHandle,
	CArray[size_t],
	CArray[Pointer[uint8]],
	--> int32
) is native($shim-lib) is export { * };

sub onnx_shim_session_output_count(
	OrtSessionHandle,
	CArray[size_t],
	CArray[Pointer[uint8]],
	--> int32
) is native($shim-lib) is export { * };

sub onnx_shim_session_input_name(
	OrtSessionHandle,
	size_t,
	CArray[Pointer[uint8]],    # out_name (malloc'd)
	CArray[Pointer[uint8]],    # out_error
	--> int32
) is native($shim-lib) is export { * };

sub onnx_shim_session_output_name(
	OrtSessionHandle,
	size_t,
	CArray[Pointer[uint8]],
	CArray[Pointer[uint8]],
	--> int32
) is native($shim-lib) is export { * };

sub onnx_shim_free_name(Pointer[uint8])
	is native($shim-lib) is export { * };

sub onnx_shim_session_input_type_info(
	OrtSessionHandle,
	size_t,
	CArray[int32],             # out_elem_type
	CArray[size_t],            # out_rank
	CArray[int64],             # out_shape
	size_t,                    # shape_cap
	CArray[Pointer[uint8]],
	--> int32
) is native($shim-lib) is export { * };

sub onnx_shim_session_output_type_info(
	OrtSessionHandle,
	size_t,
	CArray[int32],
	CArray[size_t],
	CArray[int64],
	size_t,
	CArray[Pointer[uint8]],
	--> int32
) is native($shim-lib) is export { * };

# --- Tensors ---

sub onnx_shim_create_tensor(
	Pointer,                   # data
	size_t,                    # byte_len
	CArray[int64],             # shape
	size_t,                    # rank
	int32,                     # elem_type
	CArray[OrtValueHandle],
	CArray[Pointer[uint8]],
	--> int32
) is native($shim-lib) is export { * };

sub onnx_shim_release_value(OrtValueHandle)
	is native($shim-lib) is export { * };

sub onnx_shim_tensor_shape(
	OrtValueHandle,
	CArray[int32],
	CArray[size_t],
	CArray[int64],
	size_t,
	CArray[Pointer[uint8]],
	--> int32
) is native($shim-lib) is export { * };

sub onnx_shim_tensor_data(
	OrtValueHandle,
	CArray[Pointer],           # *out_data
	CArray[size_t],            # out_byte_len
	CArray[Pointer[uint8]],
	--> int32
) is native($shim-lib) is export { * };

# --- Run ---

sub onnx_shim_run(
	OrtSessionHandle,
	CArray[Str],               # input_names (char* array — Str marshal OK IN)
	CArray[OrtValueHandle],    # inputs
	size_t,                    # num_inputs
	CArray[Str],               # output_names
	CArray[OrtValueHandle],    # outputs (shim writes)
	size_t,                    # num_outputs
	CArray[Pointer[uint8]],
	--> int32
) is native($shim-lib) is export { * };

# --- Error helper ---

sub onnx_shim_free_error(Pointer[uint8])
	is native($shim-lib) is export { * };

# === Helpers for callers ===

#| Allocate a fresh out_error slot. Initialised with a NULL
#| Pointer[uint8] so the shim sees the expected zero state and
#| the caller can check for a set error afterwards.
sub err-slot(--> CArray[Pointer[uint8]]) is export {
	my $slot = CArray[Pointer[uint8]].new;
	$slot[0] = Pointer[uint8];
	$slot;
}

#| Extract the error string from an out_error slot populated by a
#| shim call. Copies the C string into a fresh Raku Str, frees the
#| original C buffer via onnx_shim_free_error, and clears the slot.
#| Returns Str (type object) if no error was set. Must not be
#| called twice for the same slot.
sub ffi-extract-error(CArray[Pointer[uint8]] $slot --> Str) is export {
	return Str unless $slot.defined;
	my $ptr = $slot[0];
	return Str unless $ptr.defined;
	# nativecast(Str, $ptr) decodes the C string the pointer
	# addresses into a Raku Str. We then copy into a fresh Str
	# (defensive — the cast return is tied to the underlying
	# memory lifetime) and free via the raw pointer.
	my $decoded = nativecast(Str, $ptr);
	my Str $copy = $decoded.defined ?? ~$decoded !! '';
	onnx_shim_free_error($ptr);
	$slot[0] = Pointer[uint8];
	$copy;
}
