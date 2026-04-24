use NativeCall;
use ONNX::Native::FFI;
use ONNX::Native::Types;

unit module ONNX::Native;

# Forward declaration: Session.run returns Tensors, so the name
# has to be known before Session is compiled. Full definition
# lives further down in this file.
class Tensor { ... }

# === Internal helpers ===

#| Throw the right exception class for an error code + message
#| coming back from the shim.
sub throw-ffi-error(Int $code, Str $reason) {
	X::ONNX::Native::Error.new(:$code, :$reason).throw;
}

# err-slot is imported from ONNX::Native::FFI — every shim call
# that can fail takes the same CArray[Pointer[uint8]] out_error
# shape, and allocating it via the FFI helper keeps the type in
# one place.

# === Session ===

class Session is export {
	has OrtEnvHandle            $!env;
	has OrtSessionOptionsHandle $!options;
	has OrtSessionHandle        $!handle;
	has Str                     $!log-id;
	has Bool                    $!disposed;

	# Cached introspection results — populated lazily on first
	# .input-names / .output-names access. Models don't change
	# their IO surface after load, so one fetch is enough.
	has @!input-names;
	has @!output-names;
	has %!input-info;
	has %!output-info;

	submethod BUILD(
		:$!env!, :$!options!, :$!handle!,
		Str :$!log-id = 'onnx-native',
	) {
		$!disposed = False;
	}

	#| Load a model from disk. `:@providers` is tried in the order
	#| given; the first registered provider handles any given op
	#| (falls back to CPU for unsupported ops).
	multi method new(
		Str:D :$path!,
		:@providers = (CPU,),
		Str :$log-id = 'onnx-native',
	) {
		self!create-common(
			:@providers, :$log-id,
			:create-session(-> $env, $opts, $err {
				my $out = CArray[OrtSessionHandle].new;
				$out[0] = OrtSessionHandle;
				my $rc = onnx_shim_create_session_from_path(
					$env, $path, $opts, $out, $err);
				($rc, $out[0]);
			}),
		);
	}

	#| Load a model from an in-memory Blob (e.g. downloaded or
	#| mmap'd).
	multi method new(
		Blob:D :$bytes!,
		:@providers = (CPU,),
		Str :$log-id = 'onnx-native',
	) {
		my $ptr = nativecast(Pointer[uint8], $bytes);
		self!create-common(
			:@providers, :$log-id,
			:create-session(-> $env, $opts, $err {
				my $out = CArray[OrtSessionHandle].new;
				$out[0] = OrtSessionHandle;
				my $rc = onnx_shim_create_session_from_buffer(
					$env, $ptr, $bytes.bytes, $opts, $out, $err);
				($rc, $out[0]);
			}),
		);
	}

	method !create-common(
		:@providers!,
		Str :$log-id!,
		:&create-session!,
		--> Session:D
	) {
		# 1. Env
		my $env-slot = CArray[OrtEnvHandle].new;
		$env-slot[0] = OrtEnvHandle;
		my $err = err-slot();
		my $rc = onnx_shim_init($log-id, $env-slot, $err);
		unless $rc == ORT_OK {
			throw-ffi-error($rc, ffi-extract-error($err) // 'Env init failed');
		}
		my $env = $env-slot[0];

		# 2. Session options
		my $opts-slot = CArray[OrtSessionOptionsHandle].new;
		$opts-slot[0] = OrtSessionOptionsHandle;
		$err = err-slot();
		$rc = onnx_shim_create_session_options($opts-slot, $err);
		unless $rc == ORT_OK {
			onnx_shim_release_env($env);
			throw-ffi-error($rc,
				ffi-extract-error($err) // 'SessionOptions init failed');
		}
		my $options = $opts-slot[0];

		# 3. Providers (except CPU, which is always present)
		for @providers -> $p {
			my Provider $prov = $p ~~ Provider ?? $p !! Provider::{$p.Str.uc};
			without $prov {
				onnx_shim_release_session_options($options);
				onnx_shim_release_env($env);
				X::ONNX::Native::ProviderUnavailable.new(
					provider => "$p",
					reason   => "Unknown provider; expected one of: "
						~ Provider.enums.keys.sort.join(', '),
				).throw;
			}
			next if $prov == CPU;
			$err = err-slot();
			$rc = onnx_shim_enable_provider(
				$options, provider-name($prov), 0, $err);
			unless $rc == ORT_OK {
				my $msg = ffi-extract-error($err)
					// 'enable_provider failed';
				onnx_shim_release_session_options($options);
				onnx_shim_release_env($env);
				if $rc == ORT_NOT_IMPLEMENTED {
					X::ONNX::Native::ProviderUnavailable.new(
						provider => provider-name($prov),
						reason   => $msg,
					).throw;
				}
				throw-ffi-error($rc, $msg);
			}
		}

		# 4. Session (via the closure passed in by the caller)
		$err = err-slot();
		my ($rc2, $handle) = create-session($env, $options, $err);
		unless $rc2 == ORT_OK {
			my $msg = ffi-extract-error($err) // 'CreateSession failed';
			onnx_shim_release_session_options($options);
			onnx_shim_release_env($env);
			throw-ffi-error($rc2, $msg);
		}

		self.bless(:$env, :$options, :handle($handle), :$log-id);
	}

	#| Direct accessor for the native session handle. Exposed for
	#| users who want to call additional shim functions not yet
	#| wrapped at the class level.
	method handle(--> OrtSessionHandle) { $!handle }

	#| Number of model inputs.
	method input-count(--> Int) {
		self!ensure-live;
		my $out = CArray[size_t].new; $out[0] = 0;
		my $err = err-slot();
		my $rc = onnx_shim_session_input_count($!handle, $out, $err);
		self!check-rc($rc, $err, 'SessionGetInputCount');
		$out[0];
	}

	#| Number of model outputs.
	method output-count(--> Int) {
		self!ensure-live;
		my $out = CArray[size_t].new; $out[0] = 0;
		my $err = err-slot();
		my $rc = onnx_shim_session_output_count($!handle, $out, $err);
		self!check-rc($rc, $err, 'SessionGetOutputCount');
		$out[0];
	}

	#| List of model input names, in index order.
	method input-names(--> List) {
		self!ensure-live;
		@!input-names ||= self!fetch-names(self.input-count, :input);
		@!input-names.List;
	}

	#| List of model output names, in index order.
	method output-names(--> List) {
		self!ensure-live;
		@!output-names ||= self!fetch-names(self.output-count, :!input);
		@!output-names.List;
	}

	method !fetch-names(Int $n, Bool :$input!) {
		my @names;
		for ^$n -> $i {
			my $out = CArray[Pointer[uint8]].new;
			$out[0] = Pointer[uint8];
			my $err = err-slot();
			my $rc = $input
				?? onnx_shim_session_input_name($!handle, $i, $out, $err)
				!! onnx_shim_session_output_name($!handle, $i, $out, $err);
			self!check-rc($rc, $err,
				"Session{$input ?? 'Input' !! 'Output'}Name[$i]");
			my $name-ptr = $out[0];
			# Read C string, then free.
			my $s = nativecast(Str, $name-ptr);
			my $copy = ~$s;
			onnx_shim_free_name($name-ptr);
			@names.push: $copy;
		}
		@names;
	}

	#| TensorInfo for the input with the given name. Throws
	#| X::ONNX::Native::Error if the name isn't known.
	method input-info(Str:D $name --> TensorInfo) {
		self!ensure-live;
		%!input-info{$name} //= self!fetch-type-info($name, :input);
	}

	#| TensorInfo for the output with the given name.
	method output-info(Str:D $name --> TensorInfo) {
		self!ensure-live;
		%!output-info{$name} //= self!fetch-type-info($name, :!input);
	}

	method !fetch-type-info(Str $name, Bool :$input!) {
		my @names = $input ?? self.input-names !! self.output-names;
		my $idx = @names.first(* eq $name, :k);
		without $idx {
			X::ONNX::Native::Error.new(
				code => ORT_INVALID_ARGUMENT,
				reason => "No { $input ?? 'input' !! 'output' } named "
					~ "'$name'. Available: " ~ @names.join(', '),
			).throw;
		}

		# Up to 16 dims is plenty for any realistic tensor
		my $elem-out  = CArray[int32].new;  $elem-out[0]  = 0;
		my $rank-out  = CArray[size_t].new; $rank-out[0]  = 0;
		my $shape-out = CArray[int64].new;
		$shape-out[$_] = 0 for ^16;
		my $err = err-slot();
		my $rc = $input
			?? onnx_shim_session_input_type_info(
					$!handle, $idx, $elem-out, $rank-out,
					$shape-out, 16, $err)
			!! onnx_shim_session_output_type_info(
					$!handle, $idx, $elem-out, $rank-out,
					$shape-out, 16, $err);
		self!check-rc($rc, $err,
			"Session{$input ?? 'Input' !! 'Output'}TypeInfo[$name]");

		my $rank = $rank-out[0];
		my @shape = (^$rank).map({ $shape-out[$_].Int });
		my $elem = dtype-from-int($elem-out[0].Int);
		without $elem {
			X::ONNX::Native::Unsupported.new(
				reason => "Unknown tensor element type { $elem-out[0] } "
					~ "in model I/O. Upgrade ONNX::Native or extend "
					~ "the DType enum.",
			).throw;
		}
		TensorInfo.new(:elem-type($elem), :@shape);
	}

	#| Run inference. :%inputs maps each input name to a Tensor;
	#| :@outputs is the names of outputs to fetch. Returns a
	#| Hash[Str, Tensor].
	method run(:%inputs!, :@outputs! --> Hash) {
		self!ensure-live;

		my @inames = %inputs.keys;
		my @ivals = @inames.map({ %inputs{$_}.handle });

		# Allocate name arrays in CArray[Str] — NativeCall encodes
		# Raku Str → C char* on assignment, keeps the buffer alive
		# for the CArray's lifetime.
		my $in-names  = CArray[Str].new;
		$in-names[$_] = @inames[$_] for ^@inames.elems;
		my $out-names = CArray[Str].new;
		$out-names[$_] = @outputs[$_] for ^@outputs.elems;

		my $in-vals   = CArray[OrtValueHandle].new;
		$in-vals[$_] = @ivals[$_] for ^@ivals.elems;

		my $out-vals = CArray[OrtValueHandle].new;
		$out-vals[$_] = OrtValueHandle for ^@outputs.elems;

		my $err = err-slot();
		my $rc = onnx_shim_run(
			$!handle,
			$in-names, $in-vals, @inames.elems,
			$out-names, $out-vals, @outputs.elems,
			$err,
		);
		self!check-rc($rc, $err, 'Session::Run');

		my %result;
		for ^@outputs.elems -> $i {
			%result{@outputs[$i]} = Tensor.wrap-handle($out-vals[$i]);
		}
		%result;
	}

	#| Explicit cleanup. Idempotent. Called by DESTROY too.
	method dispose(--> Nil) {
		return if $!disposed;
		$!disposed = True;
		onnx_shim_release_session($!handle)            if $!handle.defined;
		onnx_shim_release_session_options($!options)   if $!options.defined;
		onnx_shim_release_env($!env)                   if $!env.defined;
		$!handle  = OrtSessionHandle;
		$!options = OrtSessionOptionsHandle;
		$!env     = OrtEnvHandle;
	}

	submethod DESTROY() {
		self.dispose;
	}

	method !ensure-live() {
		if $!disposed || !$!handle.defined {
			die "ONNX::Native::Session has been disposed";
		}
	}

	method !check-rc(Int $rc, CArray[Pointer[uint8]] $err, Str $context) {
		return if $rc == ORT_OK;
		my $msg = ffi-extract-error($err) // 'unknown error';
		throw-ffi-error($rc, "$context: $msg");
	}
}

# === Tensor ===

class Tensor is export {
	has OrtValueHandle $!handle;
	has Bool           $!disposed;

	# Anchor to keep the Raku-side Blob alive for the lifetime of
	# the tensor — ORT borrows (doesn't copy) the data pointer
	# when we call CreateTensorWithDataAsOrtValue, so the Blob
	# must not be GCed until the OrtValue is released. Only set
	# for from-blob / from-nums / from-ints inputs; output tensors
	# returned from Session.run own their storage inside ORT.
	has $!anchor;

	# Cached shape + dtype — populated lazily via .shape / .dtype
	# on first access. Tensor shape doesn't change after creation.
	has DType @!cached-dtype;   # 1-slot, [0] holds the DType
	has Int   @!cached-shape;
	has Int   $!cached-elem-count;

	submethod BUILD(:$!handle!, :$!anchor) {
		$!disposed = False;
	}

	#| Construct a tensor that borrows the given Blob's bytes.
	#| Zero-copy on the way in — ORT holds a pointer into the
	#| Blob's backing storage, so the Blob is kept alive until the
	#| tensor is disposed.
	method from-blob(
		Blob:D $data,
		:@shape!,
		DType :$dtype = FLOAT32,
		--> Tensor:D
	) {
		my $elem-size = dtype-byte-size($dtype);
		without $elem-size {
			X::ONNX::Native::Unsupported.new(
				reason => "Cannot create tensor from Blob with dtype "
					~ "{ dtype-name($dtype) } (variable-width / "
					~ "unsupported in v0.1).",
			).throw;
		}
		my $expected-bytes = ([*] @shape) * $elem-size;
		unless $data.bytes == $expected-bytes {
			X::ONNX::Native::Error.new(
				code => ORT_INVALID_ARGUMENT,
				reason => "Blob size { $data.bytes } bytes doesn't match "
					~ "shape [{ @shape.join(',') }] × "
					~ "{ dtype-name($dtype) } ({ $elem-size }B) = "
					~ "$expected-bytes bytes",
			).throw;
		}
		my $ptr = nativecast(Pointer, $data);
		my $shape-c = CArray[int64].new;
		$shape-c[$_] = @shape[$_].Int for ^@shape.elems;
		my $out = CArray[OrtValueHandle].new;
		$out[0] = OrtValueHandle;
		my $err = err-slot();
		my $rc = onnx_shim_create_tensor(
			$ptr, $data.bytes, $shape-c, @shape.elems,
			$dtype.Int, $out, $err);
		unless $rc == ORT_OK {
			throw-ffi-error($rc, ffi-extract-error($err) // 'CreateTensor failed');
		}
		self.bless(:handle($out[0]), :anchor($data));
	}

	#| Construct a FLOAT32 / DOUBLE tensor from a flat list of
	#| numbers. Dtype defaults to FLOAT32 since that's the common
	#| inference-input case.
	method from-nums(
		@values,
		:@shape!,
		DType :$dtype = FLOAT32,
		--> Tensor:D
	) {
		my $expected = [*] @shape;
		unless @values.elems == $expected {
			X::ONNX::Native::Error.new(
				code => ORT_INVALID_ARGUMENT,
				reason => "{ @values.elems } values don't fit shape "
					~ "[{ @shape.join(',') }] ({ $expected } elements)",
			).throw;
		}
		my $blob = do given $dtype {
			when FLOAT32 { blob32-from-nums(@values) }
			when DOUBLE  { blob64-from-nums(@values) }
			default      {
				X::ONNX::Native::Unsupported.new(
					reason => "from-nums only supports FLOAT32 / DOUBLE "
						~ "in v0.1; got { dtype-name($dtype) }.",
				).throw;
			}
		};
		self.from-blob($blob, :@shape, :$dtype);
	}

	#| Construct an INT32 / INT64 tensor from a flat list of
	#| integers. Dtype defaults to INT64 because HuggingFace
	#| tokenizers emit int64 token IDs by convention.
	method from-ints(
		@values,
		:@shape!,
		DType :$dtype = INT64,
		--> Tensor:D
	) {
		my $expected = [*] @shape;
		unless @values.elems == $expected {
			X::ONNX::Native::Error.new(
				code => ORT_INVALID_ARGUMENT,
				reason => "{ @values.elems } values don't fit shape "
					~ "[{ @shape.join(',') }] ({ $expected } elements)",
			).throw;
		}
		my $blob = do given $dtype {
			when INT64 { blob-int64-from-ints(@values) }
			when INT32 { blob-int32-from-ints(@values) }
			default    {
				X::ONNX::Native::Unsupported.new(
					reason => "from-ints only supports INT32 / INT64 "
						~ "in v0.1; got { dtype-name($dtype) }.",
				).throw;
			}
		};
		self.from-blob($blob, :@shape, :$dtype);
	}

	#| Wrap an OrtValueHandle returned from the shim (e.g. from
	#| Session.run). The output OrtValue owns its data inside
	#| ORT, so no Raku-side anchor is needed. Exposed publicly
	#| so Session can construct Tensors, but considered an
	#| internal seam — normal users should prefer from-blob /
	#| from-nums / from-ints for input tensors and let
	#| Session.run produce output tensors.
	method wrap-handle(OrtValueHandle:D $handle --> Tensor:D) {
		self.bless(:$handle, :anchor(Nil));
	}

	#| Exposed for Session.run to pull the native handle.
	method handle(--> OrtValueHandle) { $!handle }

	#| Shape of the tensor. Dims of -1 would indicate symbolic
	#| but OrtValues always have concrete shapes at this stage.
	method shape(--> List) {
		self!ensure-live;
		self!populate-shape unless @!cached-shape;
		@!cached-shape.List;
	}

	#| DType of the tensor.
	method dtype(--> DType) {
		self!ensure-live;
		self!populate-shape unless @!cached-dtype;
		@!cached-dtype[0];
	}

	method !populate-shape() {
		my $elem-out  = CArray[int32].new;  $elem-out[0]  = 0;
		my $rank-out  = CArray[size_t].new; $rank-out[0]  = 0;
		my $shape-out = CArray[int64].new;
		$shape-out[$_] = 0 for ^16;
		my $err = err-slot();
		my $rc = onnx_shim_tensor_shape(
			$!handle, $elem-out, $rank-out, $shape-out, 16, $err);
		unless $rc == ORT_OK {
			throw-ffi-error($rc,
				ffi-extract-error($err) // 'tensor_shape failed');
		}
		my $rank = $rank-out[0];
		@!cached-shape = (^$rank).map({ $shape-out[$_].Int }).list;
		my $dt = dtype-from-int($elem-out[0].Int);
		@!cached-dtype = ($dt,);
		$!cached-elem-count = [*] @!cached-shape;
	}

	#| Flat byte-length of the tensor's data.
	method byte-length(--> Int) {
		self!ensure-live;
		self!populate-shape unless @!cached-shape;
		my $elem-size = dtype-byte-size(@!cached-dtype[0]);
		without $elem-size {
			X::ONNX::Native::Unsupported.new(
				reason => "Can't compute byte-length for dtype "
					~ "{ dtype-name(@!cached-dtype[0]) }.",
			).throw;
		}
		$!cached-elem-count * $elem-size;
	}

	#| Copy the tensor's data into a fresh Raku Blob. The returned
	#| Blob is independent of the tensor and survives disposal.
	method to-blob(--> Blob:D) {
		self!ensure-live;
		my ($ptr, $bytes) = self!raw-data;
		my $buf = buf8.allocate($bytes);
		my $src = nativecast(Pointer[uint8], $ptr);
		$buf[$_] = $src[$_] for ^$bytes;
		$buf;
	}

	#| Decode the tensor's data as a flat list of Num. Requires
	#| FLOAT32 or DOUBLE dtype.
	method to-num-array(--> List) {
		self!ensure-live;
		self!populate-shape unless @!cached-shape;
		my $dt = @!cached-dtype[0];
		given $dt {
			when FLOAT32 {
				my $ptr = nativecast(
					CArray[num32], self!raw-ptr);
				(^$!cached-elem-count).map({ $ptr[$_].Num }).list;
			}
			when DOUBLE {
				my $ptr = nativecast(
					CArray[num64], self!raw-ptr);
				(^$!cached-elem-count).map({ $ptr[$_].Num }).list;
			}
			default {
				X::ONNX::Native::Unsupported.new(
					reason => "to-num-array requires FLOAT32/DOUBLE; "
						~ "got { dtype-name($dt) }. Use to-int-array "
						~ "or to-blob for other types.",
				).throw;
			}
		}
	}

	#| Decode the tensor's data as a flat list of Int. Requires an
	#| integer dtype (INT32 / INT64 / smaller ints / BOOL).
	method to-int-array(--> List) {
		self!ensure-live;
		self!populate-shape unless @!cached-shape;
		my $dt = @!cached-dtype[0];
		given $dt {
			when INT64 {
				my $ptr = nativecast(CArray[int64], self!raw-ptr);
				(^$!cached-elem-count).map({ $ptr[$_].Int }).list;
			}
			when INT32 {
				my $ptr = nativecast(CArray[int32], self!raw-ptr);
				(^$!cached-elem-count).map({ $ptr[$_].Int }).list;
			}
			when UINT32 {
				my $ptr = nativecast(CArray[uint32], self!raw-ptr);
				(^$!cached-elem-count).map({ $ptr[$_].Int }).list;
			}
			when INT16 {
				my $ptr = nativecast(CArray[int16], self!raw-ptr);
				(^$!cached-elem-count).map({ $ptr[$_].Int }).list;
			}
			when UINT16 {
				my $ptr = nativecast(CArray[uint16], self!raw-ptr);
				(^$!cached-elem-count).map({ $ptr[$_].Int }).list;
			}
			when INT8 {
				my $ptr = nativecast(CArray[int8], self!raw-ptr);
				(^$!cached-elem-count).map({ $ptr[$_].Int }).list;
			}
			when UINT8 {
				my $ptr = nativecast(CArray[uint8], self!raw-ptr);
				(^$!cached-elem-count).map({ $ptr[$_].Int }).list;
			}
			when BOOL {
				my $ptr = nativecast(CArray[uint8], self!raw-ptr);
				(^$!cached-elem-count).map({ $ptr[$_].Int }).list;
			}
			default {
				X::ONNX::Native::Unsupported.new(
					reason => "to-int-array requires an integer dtype; "
						~ "got { dtype-name($dt) }.",
				).throw;
			}
		}
	}

	method !raw-ptr(--> Pointer) {
		my ($ptr, $) = self!raw-data;
		$ptr;
	}

	method !raw-data(--> List) {
		my $data-out = CArray[Pointer].new; $data-out[0] = Pointer;
		my $len-out  = CArray[size_t].new;  $len-out[0]  = 0;
		my $err = err-slot();
		my $rc = onnx_shim_tensor_data(
			$!handle, $data-out, $len-out, $err);
		unless $rc == ORT_OK {
			throw-ffi-error($rc,
				ffi-extract-error($err) // 'tensor_data failed');
		}
		($data-out[0], $len-out[0]);
	}

	method dispose(--> Nil) {
		return if $!disposed;
		$!disposed = True;
		onnx_shim_release_value($!handle) if $!handle.defined;
		$!handle = OrtValueHandle;
		$!anchor = Nil;
	}

	submethod DESTROY() {
		self.dispose;
	}

	method !ensure-live() {
		if $!disposed || !$!handle.defined {
			die "ONNX::Native::Tensor has been disposed";
		}
	}
}

# === Small Blob marshaling helpers ===
#
# Raku doesn't ship a one-liner for "list of Raku Nums → Buf of
# packed num32 bytes", so we roll the handful we need. Performance
# isn't critical — tokenizer output is usually O(100) ints, model
# input data for real models is constructed once per inference.

sub blob32-from-nums(@values --> Blob) {
	my $n = @values.elems;
	my $buf = buf8.allocate($n * 4);
	my $ptr = nativecast(CArray[num32], $buf);
	$ptr[$_] = @values[$_].Num for ^$n;
	$buf;
}

sub blob64-from-nums(@values --> Blob) {
	my $n = @values.elems;
	my $buf = buf8.allocate($n * 8);
	my $ptr = nativecast(CArray[num64], $buf);
	$ptr[$_] = @values[$_].Num for ^$n;
	$buf;
}

sub blob-int64-from-ints(@values --> Blob) {
	my $n = @values.elems;
	my $buf = buf8.allocate($n * 8);
	my $ptr = nativecast(CArray[int64], $buf);
	$ptr[$_] = @values[$_].Int for ^$n;
	$buf;
}

sub blob-int32-from-ints(@values --> Blob) {
	my $n = @values.elems;
	my $buf = buf8.allocate($n * 4);
	my $ptr = nativecast(CArray[int32], $buf);
	$ptr[$_] = @values[$_].Int for ^$n;
	$buf;
}
