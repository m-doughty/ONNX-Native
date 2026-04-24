unit module ONNX::Native::Types;

use ONNX::Native::FFI;

# === Public enums ===

#| ONNX tensor element data types the Raku side can create /
#| inspect. Only FLOAT32, INT32, and INT64 support Raku ↔ ORT
#| round-trips in v0.1; the others decode for shape-info queries
#| but cannot yet be constructed from Raku data.
enum DType is export (
	FLOAT32  => ONNX_TENSOR_TYPE_FLOAT,
	INT32    => ONNX_TENSOR_TYPE_INT32,
	INT64    => ONNX_TENSOR_TYPE_INT64,
	UINT8    => ONNX_TENSOR_TYPE_UINT8,
	INT8     => ONNX_TENSOR_TYPE_INT8,
	BOOL     => ONNX_TENSOR_TYPE_BOOL,
	DOUBLE   => ONNX_TENSOR_TYPE_DOUBLE,
	UINT16   => ONNX_TENSOR_TYPE_UINT16,
	INT16    => ONNX_TENSOR_TYPE_INT16,
	UINT32   => ONNX_TENSOR_TYPE_UINT32,
	UINT64   => ONNX_TENSOR_TYPE_UINT64,
	FLOAT16  => ONNX_TENSOR_TYPE_FLOAT16,
	BFLOAT16 => ONNX_TENSOR_TYPE_BFLOAT16,
	STRING   => ONNX_TENSOR_TYPE_STRING,
);

#| Execution providers that can be requested at Session creation.
#| CPU is always available. COREML is available on macOS builds
#| of ONNX Runtime. CUDA / DML require opt-in prebuilt variants
#| and are compile-gated in the shim.
enum Provider is export <CPU COREML CUDA DML>;

#| Lowercase provider name as the shim expects it.
sub provider-name(Provider $p --> Str) is export {
	given $p {
		when CPU    { 'cpu' }
		when COREML { 'coreml' }
		when CUDA   { 'cuda' }
		when DML    { 'dml' }
	}
}

#| Human-readable name for a DType — used in error messages and
#| Pod docs. Mirrors the ONNXTensorElementDataType enum names with
#| the ONNX_TENSOR_ELEMENT_DATA_TYPE_ prefix stripped.
sub dtype-name(DType $t --> Str) is export {
	given $t {
		when FLOAT32  { 'FLOAT32' }
		when INT32    { 'INT32' }
		when INT64    { 'INT64' }
		when UINT8    { 'UINT8' }
		when INT8     { 'INT8' }
		when BOOL     { 'BOOL' }
		when DOUBLE   { 'DOUBLE' }
		when UINT16   { 'UINT16' }
		when INT16    { 'INT16' }
		when UINT32   { 'UINT32' }
		when UINT64   { 'UINT64' }
		when FLOAT16  { 'FLOAT16' }
		when BFLOAT16 { 'BFLOAT16' }
		when STRING   { 'STRING' }
	}
}

#| Byte size for a DType, or Nil for variable-width types (STRING)
#| and types the shim refuses to byte-size.
sub dtype-byte-size(DType $t --> Int) is export {
	given $t {
		when FLOAT32  { 4 }
		when INT32    { 4 }
		when INT64    { 8 }
		when UINT8    { 1 }
		when INT8     { 1 }
		when BOOL     { 1 }
		when DOUBLE   { 8 }
		when UINT16   { 2 }
		when INT16    { 2 }
		when UINT32   { 4 }
		when UINT64   { 8 }
		when FLOAT16  { 2 }
		when BFLOAT16 { 2 }
		when STRING   { Int }
		default       { Int }
	}
}

# === TensorInfo ===

#| Static description of a tensor shape + dtype. Used for Session
#| introspection and as a spec for Tensor construction.
#|
#| Dimension values of -1 denote symbolic / dynamic dimensions
#| (e.g. BERT's batch and sequence dims are typically -1 until the
#| actual inputs pin them down).
class TensorInfo is export {
	has DType $.elem-type is required;
	has Int @.shape is required;

	method gist(--> Str) {
		my $dims = @!shape.map({ $_ == -1 ?? '?' !! $_.Str }).join(', ');
		"TensorInfo({ dtype-name($!elem-type) }[{ $dims }])";
	}

	method Str(--> Str) { self.gist }

	#| Number of elements implied by the shape, or Nil if the
	#| shape contains symbolic (-1) dimensions.
	method element-count(--> Int) {
		return Int if @!shape.first(* < 0).defined;
		[*] @!shape;
	}
}

# === Exceptions ===
#
# Exception's built-in `method message` is the one .message
# returns. We store the raw error text in $!reason so our method
# override can format it with context (error code, provider name,
# etc.) without clashing with the attribute/method namespace.

#| Base class for all ONNX::Native errors. $.code is an
#| OrtErrorCode value (ORT_OK = 0, ORT_FAIL = 1, etc. — see
#| ONNX::Native::FFI).
class X::ONNX::Native::Error is Exception is export {
	has Int $.code is required;
	has Str $.reason is required;

	method message(--> Str) {
		"[ONNX::Native] { code-name($!code) }: $!reason";
	}
}

#| Thrown when a feature is syntactically valid but not supported
#| in this version of ONNX::Native (fp16 input tensors, sequence /
#| map types in model I/O, etc.). Uses the same code slot
#| (ORT_NOT_IMPLEMENTED) ORT would emit.
class X::ONNX::Native::Unsupported is X::ONNX::Native::Error is export {
	method new(Str :$reason!) {
		self.bless(:code(ORT_NOT_IMPLEMENTED), :$reason);
	}

	method message(--> Str) {
		"[ONNX::Native] Unsupported: $.reason";
	}
}

#| Thrown when the caller asks for an execution provider that
#| isn't available in this build of the shim (e.g. CUDA on a
#| CPU-only install, CoreML on Linux).
class X::ONNX::Native::ProviderUnavailable is X::ONNX::Native::Error is export {
	has Str $.provider is required;

	method new(Str :$provider!, Str :$reason!) {
		self.bless(:code(ORT_NOT_IMPLEMENTED), :$provider, :$reason);
	}

	method message(--> Str) {
		"[ONNX::Native] ProviderUnavailable ($!provider): $.reason";
	}
}

#| Map OrtErrorCode int → short name. Used in error messages so
#| humans can tell ORT_INVALID_ARGUMENT (bug on our side) from
#| ORT_NO_SUCHFILE (missing model) at a glance.
sub code-name(Int $code --> Str) is export {
	given $code {
		when ORT_OK                { 'OK' }
		when ORT_FAIL              { 'FAIL' }
		when ORT_INVALID_ARGUMENT  { 'INVALID_ARGUMENT' }
		when ORT_NO_SUCHFILE       { 'NO_SUCHFILE' }
		when ORT_NO_MODEL          { 'NO_MODEL' }
		when ORT_ENGINE_ERROR      { 'ENGINE_ERROR' }
		when ORT_RUNTIME_EXCEPTION { 'RUNTIME_EXCEPTION' }
		when ORT_INVALID_PROTOBUF  { 'INVALID_PROTOBUF' }
		when ORT_MODEL_LOADED      { 'MODEL_LOADED' }
		when ORT_NOT_IMPLEMENTED   { 'NOT_IMPLEMENTED' }
		when ORT_INVALID_GRAPH     { 'INVALID_GRAPH' }
		when ORT_EP_FAIL           { 'EP_FAIL' }
		default                    { "UNKNOWN($code)" }
	}
}

#| Map int ONNXTensorElementDataType → DType enum (or Nil if
#| the type isn't a known member). DType.enums yields key→int
#| pairs; .invert gives int→key (Str). We want int→DType, so we
#| rebuild via DType::{key}.
sub dtype-from-int(Int $t --> DType) is export {
	state %map = DType.enums.map({ .value => DType::{.key} });
	%map{$t} // DType;
}
