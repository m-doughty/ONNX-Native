#!/usr/bin/env raku
#
# Generates t/fixtures/high-rank.onnx — a minimal ONNX model whose
# I/O has rank 17 (one more than the historical Raku-side
# shape-buffer cap of 16, exercised by t/04-shape-edges.rakutest).
#
# The model is a single Identity node: input X → output Y, both
# FLOAT[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] (17 ones, total 1
# element). Identity is in opset 1 and supported by every ORT
# release; rank 17 is beyond what real models need but lets us
# verify the introspection path doesn't silently lose dims.
#
# Run with:
#
#     raku t/fixtures/generate-high-rank-onnx.raku
#
# Output: t/fixtures/high-rank.onnx (relative to this script).

# === protobuf wire-format helpers (mirror generate-add-onnx.raku) ===

sub cat(*@blobs --> Blob) {
	my $buf = buf8.new;
	for @blobs -> $b {
		$buf.append: $b.list if $b.defined && $b.elems;
	}
	$buf;
}

sub varint(Int $n --> Blob) {
	my @bytes;
	my $v = $n;
	repeat {
		my $byte = $v +& 0x7f;
		$v = $v +> 7;
		@bytes.push: $byte +| ($v > 0 ?? 0x80 !! 0);
	} while $v > 0;
	blob8.new(@bytes);
}

sub tag(Int $field, Int $wire --> Blob) {
	varint(($field +< 3) +| $wire);
}

sub field-varint(Int $field, Int $value --> Blob) {
	cat(tag($field, 0), varint($value));
}

sub field-bytes(Int $field, Blob $value --> Blob) {
	cat(tag($field, 2), varint($value.bytes), $value);
}

sub field-string(Int $field, Str $value --> Blob) {
	field-bytes($field, $value.encode('utf8'));
}

sub field-msg(Int $field, Blob $msg --> Blob) {
	field-bytes($field, $msg);
}

sub dim-value(Int $v --> Blob) {
	field-varint(1, $v);
}

sub tensor-shape(*@dims --> Blob) {
	cat(|@dims.map({ field-msg(1, dim-value($_)) }));
}

sub tensor-type(Int $elem-type, Blob $shape --> Blob) {
	cat(field-varint(1, $elem-type), field-msg(2, $shape));
}

sub type-proto-tensor(Int $elem-type, Blob $shape --> Blob) {
	field-msg(1, tensor-type($elem-type, $shape));
}

sub value-info(Str $name, Blob $type --> Blob) {
	cat(field-string(1, $name), field-msg(2, $type));
}

sub node-proto(:@inputs!, :@outputs!, Str :$name = '',
               Str :$op-type!, Str :$domain = '' --> Blob) {
	my @parts;
	@parts.push: field-string(1, $_) for @inputs;
	@parts.push: field-string(2, $_) for @outputs;
	@parts.push: field-string(3, $name) if $name;
	@parts.push: field-string(4, $op-type);
	@parts.push: field-string(7, $domain) if $domain;
	cat(|@parts);
}

sub graph-proto(:@nodes!, Str :$name!, :@inputs!, :@outputs! --> Blob) {
	my @parts;
	@parts.push: field-msg(1, $_) for @nodes;
	@parts.push: field-string(2, $name);
	@parts.push: field-msg(11, $_) for @inputs;
	@parts.push: field-msg(12, $_) for @outputs;
	cat(|@parts);
}

sub opset-id(Str :$domain = '', Int :$version! --> Blob) {
	my @parts;
	@parts.push: field-string(1, $domain) if $domain;
	@parts.push: field-varint(2, $version);
	cat(|@parts);
}

sub model-proto(Int :$ir-version!, :@opset-imports!,
                Str :$producer-name = '', Blob :$graph! --> Blob) {
	my @parts;
	@parts.push: field-varint(1, $ir-version);
	@parts.push: field-string(2, $producer-name) if $producer-name;
	@parts.push: field-msg(7, $graph);
	@parts.push: field-msg(8, $_) for @opset-imports;
	cat(|@parts);
}

# === Build the rank-17 Identity model ===

constant FLOAT-TYPE = 1;
constant RANK = 17;

my @dims = 1 xx RANK;
my $shape-vec = tensor-shape(|@dims);
my $tensor-type-proto = type-proto-tensor(FLOAT-TYPE, $shape-vec);

my $input-x  = value-info('X', $tensor-type-proto);
my $output-y = value-info('Y', $tensor-type-proto);

my $identity-node = node-proto(
	:inputs(['X',]),
	:outputs(['Y',]),
	:name<id_0>,
	:op-type<Identity>,
);

my $graph = graph-proto(
	:name<high_rank_graph>,
	:nodes([$identity-node,]),
	:inputs([$input-x,]),
	:outputs([$output-y,]),
);

my $model = model-proto(
	:ir-version(8),
	:opset-imports([opset-id(:version(17)),]),
	:producer-name('ONNX-Native fixture generator'),
	:$graph,
);

my $out = $*PROGRAM.parent.add('high-rank.onnx');
$out.spurt($model, :bin);
say "Wrote { $model.bytes } bytes to $out (rank { RANK } Identity model)";
