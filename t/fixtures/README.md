# Test fixtures for ONNX::Native

## add.onnx

A minimal ONNX model that computes `C = A + B` where `A`, `B`, and
`C` are all `FLOAT[3]` tensors. Used by `t/03-inference.rakutest`
to verify an end-to-end round-trip through the shim.

Pinned to `ir_version=8` / `opset=17` — every ONNX Runtime release
since 1.14 supports this combination.

### Regenerating

The fixture was produced by a pure-Raku protobuf emitter (no Python
`onnx` package required):

```
raku t/fixtures/generate-add-onnx.raku
```

The script writes `t/fixtures/add.onnx` and prints the byte count.
If you change the model, re-run the generator and commit both
the script and the new binary.

### Why hand-rolled protobuf?

`onnx.helper.make_model` from the Python `onnx` package is the
standard route. Going through Raku directly avoids adding a
Python build-time dep and keeps the fixture reproducible from
inside the distribution's own toolchain.

The generator covers only the subset of `onnx.proto` needed for
simple op-graph fixtures (`ModelProto`, `GraphProto`, `NodeProto`,
`ValueInfoProto`, `TypeProto.Tensor`, `TensorShapeProto.Dimension`,
`OperatorSetIdProto`). Extending it to emit initializer tensors,
attributes, or string tensors would mean implementing more of the
protobuf schema — probably worth switching to the Python path at
that point.
