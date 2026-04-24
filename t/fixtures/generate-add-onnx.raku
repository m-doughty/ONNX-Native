#!/usr/bin/env raku
#
# Generates t/fixtures/add.onnx — a minimal ONNX model with two
# FLOAT[3] inputs A and B, one FLOAT[3] output C, and a single
# Add node connecting them. Used as the happy-path fixture for
# ONNX::Native's inference tests.
#
# The ONNX wire format is protobuf. We build it here by hand so
# that regeneration doesn't require installing the Python `onnx`
# package. Run this script with:
#
#     raku t/fixtures/generate-add-onnx.raku
#
# Output file: t/fixtures/add.onnx (relative to this script).
# If you bump ir_version / opset, also update the fixture README.

# === protobuf wire-format helpers ===

#| Concat Blobs by appending bytes to a buf8. Raku's infix:<~> is
#| str-concat, not byte-concat, so we can't just do $a ~ $b.
sub cat(*@blobs --> Blob) {
	my $buf = buf8.new;
	for @blobs -> $b {
		$buf.append: $b.list if $b.defined && $b.elems;
	}
	$buf;
}

#| Emit a varint: 7 bits of value per byte, MSB set on all but the last.
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

#| Tag = (field_number << 3) | wire_type, as a varint.
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

# === ONNX message builders ===
#
# Mirrors onnx.proto field numbers. Only the subset we need to
# describe a single-Add model is implemented. Proto3 defaults
# (empty strings, zeros) are omitted from the wire — matches
# what protoc would emit.

#| TensorShapeProto.Dimension — dim_value (int64) at field 1.
sub dim-value(Int $v --> Blob) {
	field-varint(1, $v);
}

#| TensorShapeProto — repeated Dimension at field 1.
sub tensor-shape(*@dims --> Blob) {
	cat(|@dims.map({ field-msg(1, dim-value($_)) }));
}

#| TypeProto.Tensor — elem_type (int32) at 1, shape (TensorShapeProto) at 2.
sub tensor-type(Int $elem-type, Blob $shape --> Blob) {
	cat(field-varint(1, $elem-type), field-msg(2, $shape));
}

#| TypeProto — tensor_type (Tensor) at field 1 (member of oneof value).
sub type-proto-tensor(Int $elem-type, Blob $shape --> Blob) {
	field-msg(1, tensor-type($elem-type, $shape));
}

#| ValueInfoProto — name (1), type (2).
sub value-info(Str $name, Blob $type --> Blob) {
	cat(field-string(1, $name), field-msg(2, $type));
}

#| NodeProto — input (1, repeated), output (2, repeated),
#| name (3), op_type (4), domain (7).
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

#| GraphProto — node (1, repeated), name (2), input (11, repeated),
#| output (12, repeated).
sub graph-proto(:@nodes!, Str :$name!, :@inputs!, :@outputs! --> Blob) {
	my @parts;
	@parts.push: field-msg(1, $_) for @nodes;
	@parts.push: field-string(2, $name);
	@parts.push: field-msg(11, $_) for @inputs;
	@parts.push: field-msg(12, $_) for @outputs;
	cat(|@parts);
}

#| OperatorSetIdProto — domain (1), version (2).
sub opset-id(Str :$domain = '', Int :$version! --> Blob) {
	my @parts;
	@parts.push: field-string(1, $domain) if $domain;
	@parts.push: field-varint(2, $version);
	cat(|@parts);
}

#| ModelProto — ir_version (1), producer_name (2), graph (7),
#| opset_import (8, repeated).
sub model-proto(Int :$ir-version!, :@opset-imports!,
                Str :$producer-name = '', Blob :$graph! --> Blob) {
	my @parts;
	@parts.push: field-varint(1, $ir-version);
	@parts.push: field-string(2, $producer-name) if $producer-name;
	@parts.push: field-msg(7, $graph);
	@parts.push: field-msg(8, $_) for @opset-imports;
	cat(|@parts);
}

# === Build the Add model ===
#
# FLOAT = 1 in ONNXTensorElementDataType. Shape is [3] (one dim
# of size 3). IR version 8 pairs with opset 17, which every
# recent ONNX Runtime release supports.

constant FLOAT-TYPE = 1;

my $shape-vec   = tensor-shape(3);
my $tensor-f3 = type-proto-tensor(FLOAT-TYPE, $shape-vec);

my $input-a = value-info('A', $tensor-f3);
my $input-b = value-info('B', $tensor-f3);
my $output-c = value-info('C', $tensor-f3);

my $add-node = node-proto(
	:inputs<A B>,
	:outputs(['C',]),
	:name<add_0>,
	:op-type<Add>,
);

my $graph = graph-proto(
	:name<add_graph>,
	:nodes([$add-node,]),
	:inputs([$input-a, $input-b]),
	:outputs([$output-c,]),
);

my $model = model-proto(
	:ir-version(8),
	:opset-imports([opset-id(:version(17)),]),
	:producer-name('ONNX-Native fixture generator'),
	:$graph,
);

my $out = $*PROGRAM.parent.add('add.onnx');
$out.spurt($model, :bin);
say "Wrote { $model.bytes } bytes to $out";
