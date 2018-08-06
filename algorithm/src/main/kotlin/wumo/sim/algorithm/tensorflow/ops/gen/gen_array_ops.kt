/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.algorithm.tensorflow.ops.gen

import org.bytedeco.javacpp.tensorflow.DT_INT32
import wumo.sim.algorithm.tensorflow.*
import wumo.sim.algorithm.tensorflow.ops.Output
import wumo.sim.util.Dimension

fun TF.batchToSpace(input: Output, crops: Output, block_size: Long, name: String = "BatchToSpace") = run {
  buildOpTensor("BatchToSpace", name) {
    addInput(input, false)
    addInput(crops, false)
    attr("block_size", block_size)
  }
}

fun TF.batchToSpaceND(input: Output, block_shape: Output, crops: Output, name: String = "BatchToSpaceND") = run {
  buildOpTensor("BatchToSpaceND", name) {
    addInput(input, false)
    addInput(block_shape, false)
    addInput(crops, false)
  }
}

fun TF.bitcast(input: Output, _type: Int, name: String = "Bitcast") = run {
  buildOpTensor("Bitcast", name) {
    addInput(input, false)
    attrType("type", _type)
  }
}

fun TF.broadcastArgs(s0: Output, s1: Output, name: String = "BroadcastArgs") = run {
  buildOpTensor("BroadcastArgs", name) {
    addInput(s0, false)
    addInput(s1, false)
  }
}

fun TF.broadcastTo(input: Output, shape: Output, name: String = "BroadcastTo") = run {
  buildOpTensor("BroadcastTo", name) {
    addInput(input, false)
    addInput(shape, false)
  }
}

fun TF.checkNumerics(tensor: Output, message: String, name: String = "CheckNumerics") = run {
  buildOpTensor("CheckNumerics", name) {
    addInput(tensor, false)
    attr("message", message)
  }
}

fun TF.concatV2(values: Array<Output>, axis: Output, name: String = "ConcatV2") = run {
  buildOpTensor("ConcatV2", name) {
    addInput(values, false)
    addInput(axis, false)
  }
}

fun TF.conjugateTranspose(x: Output, perm: Output, name: String = "ConjugateTranspose") = run {
  buildOpTensor("ConjugateTranspose", name) {
    addInput(x, false)
    addInput(perm, false)
  }
}

fun TF.debugGradientIdentity(input: Output, name: String = "DebugGradientIdentity") = run {
  buildOpTensor("DebugGradientIdentity", name) {
    addInput(input, false)
  }
}

fun TF.debugGradientRefIdentity(input: Output, name: String = "DebugGradientRefIdentity") = run {
  buildOpTensor("DebugGradientRefIdentity", name) {
    addInput(input, true)
  }
}

fun TF.deepCopy(x: Output, name: String = "DeepCopy") = run {
  buildOpTensor("DeepCopy", name) {
    addInput(x, false)
  }
}

fun TF.depthToSpace(input: Output, block_size: Long, data_format: String = "NHWC", name: String = "DepthToSpace") = run {
  buildOpTensor("DepthToSpace", name) {
    addInput(input, false)
    attr("block_size", block_size)
    attr("data_format", data_format)
  }
}

fun TF.dequantize(input: Output, min_range: Output, max_range: Output, mode: String = "MIN_COMBINED", name: String = "Dequantize") = run {
  buildOpTensor("Dequantize", name) {
    addInput(input, false)
    addInput(min_range, false)
    addInput(max_range, false)
    attr("mode", mode)
  }
}

fun TF.diag(diagonal: Output, name: String = "Diag") = run {
  buildOpTensor("Diag", name) {
    addInput(diagonal, false)
  }
}

fun TF.diagPart(input: Output, name: String = "DiagPart") = run {
  buildOpTensor("DiagPart", name) {
    addInput(input, false)
  }
}

fun TF.editDistance(hypothesis_indices: Output, hypothesis_values: Output, hypothesis_shape: Output, truth_indices: Output, truth_values: Output, truth_shape: Output, normalize: Boolean = true, name: String = "EditDistance") = run {
  buildOpTensor("EditDistance", name) {
    addInput(hypothesis_indices, false)
    addInput(hypothesis_values, false)
    addInput(hypothesis_shape, false)
    addInput(truth_indices, false)
    addInput(truth_values, false)
    addInput(truth_shape, false)
    attr("normalize", normalize)
  }
}

fun TF.empty(shape: Output, dtype: Int, init: Boolean = false, name: String = "Empty") = run {
  buildOpTensor("Empty", name) {
    addInput(shape, false)
    attrType("dtype", dtype)
    attr("init", init)
  }
}

fun TF.expandDims(input: Output, dim: Output, name: String = "ExpandDims") = run {
  buildOpTensor("ExpandDims", name) {
    addInput(input, false)
    addInput(dim, false)
  }
}

fun TF.extractImagePatches(images: Output, ksizes: Array<Long>, strides: Array<Long>, rates: Array<Long>, padding: String, name: String = "ExtractImagePatches") = run {
  buildOpTensor("ExtractImagePatches", name) {
    addInput(images, false)
    attr("ksizes", ksizes)
    attr("strides", strides)
    attr("rates", rates)
    attr("padding", padding)
  }
}

fun TF.fakeQuantWithMinMaxArgs(inputs: Output, min: Float = -6.0f, max: Float = 6.0f, num_bits: Long = 8L, narrow_range: Boolean = false, name: String = "FakeQuantWithMinMaxArgs") = run {
  buildOpTensor("FakeQuantWithMinMaxArgs", name) {
    addInput(inputs, false)
    attr("min", min)
    attr("max", max)
    attr("num_bits", num_bits)
    attr("narrow_range", narrow_range)
  }
}

fun TF.fakeQuantWithMinMaxArgsGradient(gradients: Output, inputs: Output, min: Float = -6.0f, max: Float = 6.0f, num_bits: Long = 8L, narrow_range: Boolean = false, name: String = "FakeQuantWithMinMaxArgsGradient") = run {
  buildOpTensor("FakeQuantWithMinMaxArgsGradient", name) {
    addInput(gradients, false)
    addInput(inputs, false)
    attr("min", min)
    attr("max", max)
    attr("num_bits", num_bits)
    attr("narrow_range", narrow_range)
  }
}

fun TF.fakeQuantWithMinMaxVars(inputs: Output, min: Output, max: Output, num_bits: Long = 8L, narrow_range: Boolean = false, name: String = "FakeQuantWithMinMaxVars") = run {
  buildOpTensor("FakeQuantWithMinMaxVars", name) {
    addInput(inputs, false)
    addInput(min, false)
    addInput(max, false)
    attr("num_bits", num_bits)
    attr("narrow_range", narrow_range)
  }
}

fun TF.fakeQuantWithMinMaxVarsGradient(gradients: Output, inputs: Output, min: Output, max: Output, num_bits: Long = 8L, narrow_range: Boolean = false, name: String = "FakeQuantWithMinMaxVarsGradient") = run {
  buildOpTensors("FakeQuantWithMinMaxVarsGradient", name) {
    addInput(gradients, false)
    addInput(inputs, false)
    addInput(min, false)
    addInput(max, false)
    attr("num_bits", num_bits)
    attr("narrow_range", narrow_range)
  }
}

fun TF.fakeQuantWithMinMaxVarsPerChannel(inputs: Output, min: Output, max: Output, num_bits: Long = 8L, narrow_range: Boolean = false, name: String = "FakeQuantWithMinMaxVarsPerChannel") = run {
  buildOpTensor("FakeQuantWithMinMaxVarsPerChannel", name) {
    addInput(inputs, false)
    addInput(min, false)
    addInput(max, false)
    attr("num_bits", num_bits)
    attr("narrow_range", narrow_range)
  }
}

fun TF.fakeQuantWithMinMaxVarsPerChannelGradient(gradients: Output, inputs: Output, min: Output, max: Output, num_bits: Long = 8L, narrow_range: Boolean = false, name: String = "FakeQuantWithMinMaxVarsPerChannelGradient") = run {
  buildOpTensors("FakeQuantWithMinMaxVarsPerChannelGradient", name) {
    addInput(gradients, false)
    addInput(inputs, false)
    addInput(min, false)
    addInput(max, false)
    attr("num_bits", num_bits)
    attr("narrow_range", narrow_range)
  }
}

fun TF.fill(dims: Output, value: Output, name: String = "Fill") = run {
  buildOpTensor("Fill", name) {
    addInput(dims, false)
    addInput(value, false)
  }
}

fun TF.gather(params: Output, indices: Output, validate_indices: Boolean = true, name: String = "Gather") = run {
  buildOpTensor("Gather", name) {
    addInput(params, false)
    addInput(indices, false)
    attr("validate_indices", validate_indices)
  }
}

fun TF.gatherNd(params: Output, indices: Output, name: String = "GatherNd") = run {
  buildOpTensor("GatherNd", name) {
    addInput(params, false)
    addInput(indices, false)
  }
}

fun TF.gatherV2(params: Output, indices: Output, axis: Output, name: String = "GatherV2") = run {
  buildOpTensor("GatherV2", name) {
    addInput(params, false)
    addInput(indices, false)
    addInput(axis, false)
  }
}

fun TF.guaranteeConst(input: Output, name: String = "GuaranteeConst") = run {
  buildOpTensor("GuaranteeConst", name) {
    addInput(input, false)
  }
}

fun TF.identity(input: Output, name: String = "Identity") = run {
  buildOpTensor("Identity", name) {
    addInput(input, false)
  }
}

fun TF.identityN(input: Output, name: String = "IdentityN") = run {
  buildOpTensors("IdentityN", name) {
    addInput(input, false)
  }
}

fun TF.immutableConst(dtype: Int, shape: Dimension, memory_region_name: String, name: String = "ImmutableConst") = run {
  buildOpTensor("ImmutableConst", name) {
    attrType("dtype", dtype)
    attr("shape", shape)
    attr("memory_region_name", memory_region_name)
  }
}

fun TF.inplaceAdd(x: Output, i: Output, v: Output, name: String = "InplaceAdd") = run {
  buildOpTensor("InplaceAdd", name) {
    addInput(x, false)
    addInput(i, false)
    addInput(v, false)
  }
}

fun TF.inplaceSub(x: Output, i: Output, v: Output, name: String = "InplaceSub") = run {
  buildOpTensor("InplaceSub", name) {
    addInput(x, false)
    addInput(i, false)
    addInput(v, false)
  }
}

fun TF.inplaceUpdate(x: Output, i: Output, v: Output, name: String = "InplaceUpdate") = run {
  buildOpTensor("InplaceUpdate", name) {
    addInput(x, false)
    addInput(i, false)
    addInput(v, false)
  }
}

fun TF.invertPermutation(x: Output, name: String = "InvertPermutation") = run {
  buildOpTensor("InvertPermutation", name) {
    addInput(x, false)
  }
}

fun TF.listDiff(x: Output, y: Output, out_idx: Int = DT_INT32, name: String = "ListDiff") = run {
  buildOpTensors("ListDiff", name) {
    addInput(x, false)
    addInput(y, false)
    attrType("out_idx", out_idx)
  }
}

fun TF.matrixBandPart(input: Output, num_lower: Output, num_upper: Output, name: String = "MatrixBandPart") = run {
  buildOpTensor("MatrixBandPart", name) {
    addInput(input, false)
    addInput(num_lower, false)
    addInput(num_upper, false)
  }
}

fun TF.matrixDiag(diagonal: Output, name: String = "MatrixDiag") = run {
  buildOpTensor("MatrixDiag", name) {
    addInput(diagonal, false)
  }
}

fun TF.matrixDiagPart(input: Output, name: String = "MatrixDiagPart") = run {
  buildOpTensor("MatrixDiagPart", name) {
    addInput(input, false)
  }
}

fun TF.matrixSetDiag(input: Output, diagonal: Output, name: String = "MatrixSetDiag") = run {
  buildOpTensor("MatrixSetDiag", name) {
    addInput(input, false)
    addInput(diagonal, false)
  }
}

fun TF.mirrorPad(input: Output, paddings: Output, mode: String, name: String = "MirrorPad") = run {
  buildOpTensor("MirrorPad", name) {
    addInput(input, false)
    addInput(paddings, false)
    attr("mode", mode)
  }
}

fun TF.oneHot(indices: Output, depth: Output, on_value: Output, off_value: Output, axis: Long = -1L, name: String = "OneHot") = run {
  buildOpTensor("OneHot", name) {
    addInput(indices, false)
    addInput(depth, false)
    addInput(on_value, false)
    addInput(off_value, false)
    attr("axis", axis)
  }
}

fun TF.onesLike(x: Output, name: String = "OnesLike") = run {
  buildOpTensor("OnesLike", name) {
    addInput(x, false)
  }
}

fun TF.pack(values: Array<Output>, axis: Long = 0L, name: String = "Pack") = run {
  buildOpTensor("Pack", name) {
    addInput(values, false)
    attr("axis", axis)
  }
}

fun TF.pad(input: Output, paddings: Output, name: String = "Pad") = run {
  buildOpTensor("Pad", name) {
    addInput(input, false)
    addInput(paddings, false)
  }
}

fun TF.padV2(input: Output, paddings: Output, constant_values: Output, name: String = "PadV2") = run {
  buildOpTensor("PadV2", name) {
    addInput(input, false)
    addInput(paddings, false)
    addInput(constant_values, false)
  }
}

fun TF.parallelConcat(values: Array<Output>, shape: Dimension, name: String = "ParallelConcat") = run {
  buildOpTensor("ParallelConcat", name) {
    addInput(values, false)
    attr("shape", shape)
  }
}

fun TF.placeholder(dtype: Int, shape: Dimension = Dimension(unknow_rank = true), name: String = "Placeholder") = run {
  buildOpTensor("Placeholder", name) {
    attrType("dtype", dtype)
    attr("shape", shape)
  }
}

fun TF.placeholderWithDefault(input: Output, shape: Dimension, name: String = "PlaceholderWithDefault") = run {
  buildOpTensor("PlaceholderWithDefault", name) {
    addInput(input, false)
    attr("shape", shape)
  }
}

fun TF.preventGradient(input: Output, message: String = "", name: String = "PreventGradient") = run {
  buildOpTensor("PreventGradient", name) {
    addInput(input, false)
    attr("message", message)
  }
}

fun TF.quantizeAndDequantizeV2(input: Output, input_min: Output, input_max: Output, signed_input: Boolean = true, num_bits: Long = 8L, range_given: Boolean = false, name: String = "QuantizeAndDequantizeV2") = run {
  buildOpTensor("QuantizeAndDequantizeV2", name) {
    addInput(input, false)
    addInput(input_min, false)
    addInput(input_max, false)
    attr("signed_input", signed_input)
    attr("num_bits", num_bits)
    attr("range_given", range_given)
  }
}

fun TF.quantizeAndDequantizeV3(input: Output, input_min: Output, input_max: Output, num_bits: Output, signed_input: Boolean = true, range_given: Boolean = true, name: String = "QuantizeAndDequantizeV3") = run {
  buildOpTensor("QuantizeAndDequantizeV3", name) {
    addInput(input, false)
    addInput(input_min, false)
    addInput(input_max, false)
    addInput(num_bits, false)
    attr("signed_input", signed_input)
    attr("range_given", range_given)
  }
}

fun TF.quantizeV2(input: Output, min_range: Output, max_range: Output, t: Int, mode: String = "MIN_COMBINED", round_mode: String = "HALF_AWAY_FROM_ZERO", name: String = "QuantizeV2") = run {
  buildOpTensors("QuantizeV2", name) {
    addInput(input, false)
    addInput(min_range, false)
    addInput(max_range, false)
    attrType("T", t)
    attr("mode", mode)
    attr("round_mode", round_mode)
  }
}

fun TF.quantizedConcat(concat_dim: Output, values: Array<Output>, input_mins: Array<Output>, input_maxes: Array<Output>, name: String = "QuantizedConcat") = run {
  buildOpTensors("QuantizedConcat", name) {
    addInput(concat_dim, false)
    addInput(values, false)
    addInput(input_mins, false)
    addInput(input_maxes, false)
  }
}

fun TF.quantizedInstanceNorm(x: Output, x_min: Output, x_max: Output, output_range_given: Boolean = false, given_y_min: Float = 0.0f, given_y_max: Float = 0.0f, variance_epsilon: Float = 1.0E-5f, min_separation: Float = 0.001f, name: String = "QuantizedInstanceNorm") = run {
  buildOpTensors("QuantizedInstanceNorm", name) {
    addInput(x, false)
    addInput(x_min, false)
    addInput(x_max, false)
    attr("output_range_given", output_range_given)
    attr("given_y_min", given_y_min)
    attr("given_y_max", given_y_max)
    attr("variance_epsilon", variance_epsilon)
    attr("min_separation", min_separation)
  }
}

fun TF.quantizedReshape(tensor: Output, shape: Output, input_min: Output, input_max: Output, name: String = "QuantizedReshape") = run {
  buildOpTensors("QuantizedReshape", name) {
    addInput(tensor, false)
    addInput(shape, false)
    addInput(input_min, false)
    addInput(input_max, false)
  }
}

fun TF.rank(input: Output, name: String = "Rank") = run {
  buildOpTensor("Rank", name) {
    addInput(input, false)
  }
}

fun TF.reshape(tensor: Output, shape: Output, name: String = "Reshape") = run {
  buildOpTensor("Reshape", name) {
    addInput(tensor, false)
    addInput(shape, false)
  }
}

fun TF.resourceStridedSliceAssign(_ref: Output, begin: Output, end: Output, strides: Output, value: Output, begin_mask: Long = 0L, end_mask: Long = 0L, ellipsis_mask: Long = 0L, new_axis_mask: Long = 0L, shrink_axis_mask: Long = 0L, name: String = "ResourceStridedSliceAssign") = run {
  buildOp("ResourceStridedSliceAssign", name) {
    addInput(_ref, false)
    addInput(begin, false)
    addInput(end, false)
    addInput(strides, false)
    addInput(value, false)
    attr("begin_mask", begin_mask)
    attr("end_mask", end_mask)
    attr("ellipsis_mask", ellipsis_mask)
    attr("new_axis_mask", new_axis_mask)
    attr("shrink_axis_mask", shrink_axis_mask)
  }
}

fun TF.reverseSequence(input: Output, seq_lengths: Output, seq_dim: Long, batch_dim: Long = 0L, name: String = "ReverseSequence") = run {
  buildOpTensor("ReverseSequence", name) {
    addInput(input, false)
    addInput(seq_lengths, false)
    attr("seq_dim", seq_dim)
    attr("batch_dim", batch_dim)
  }
}

fun TF.reverseV2(tensor: Output, axis: Output, name: String = "ReverseV2") = run {
  buildOpTensor("ReverseV2", name) {
    addInput(tensor, false)
    addInput(axis, false)
  }
}

fun TF.scatterNd(indices: Output, updates: Output, shape: Output, name: String = "ScatterNd") = run {
  buildOpTensor("ScatterNd", name) {
    addInput(indices, false)
    addInput(updates, false)
    addInput(shape, false)
  }
}

fun TF.scatterNdNonAliasingAdd(input: Output, indices: Output, updates: Output, name: String = "ScatterNdNonAliasingAdd") = run {
  buildOpTensor("ScatterNdNonAliasingAdd", name) {
    addInput(input, false)
    addInput(indices, false)
    addInput(updates, false)
  }
}

fun TF.shape(input: Output, out_type: Int = DT_INT32, name: String = "Shape") = run {
  buildOpTensor("Shape", name) {
    addInput(input, false)
    attrType("out_type", out_type)
  }
}

fun TF.shapeN(input: Array<Output>, out_type: Int = DT_INT32, name: String = "ShapeN") = run {
  buildOpTensors("ShapeN", name) {
    addInput(input, false)
    attrType("out_type", out_type)
  }
}

fun TF.size(input: Output, out_type: Int = DT_INT32, name: String = "Size") = run {
  buildOpTensor("Size", name) {
    addInput(input, false)
    attrType("out_type", out_type)
  }
}

fun TF.slice(input: Output, begin: Output, size: Output, name: String = "Slice") = run {
  buildOpTensor("Slice", name) {
    addInput(input, false)
    addInput(begin, false)
    addInput(size, false)
  }
}

fun TF.snapshot(input: Output, name: String = "Snapshot") = run {
  buildOpTensor("Snapshot", name) {
    addInput(input, false)
  }
}

fun TF.spaceToBatch(input: Output, paddings: Output, block_size: Long, name: String = "SpaceToBatch") = run {
  buildOpTensor("SpaceToBatch", name) {
    addInput(input, false)
    addInput(paddings, false)
    attr("block_size", block_size)
  }
}

fun TF.spaceToBatchND(input: Output, block_shape: Output, paddings: Output, name: String = "SpaceToBatchND") = run {
  buildOpTensor("SpaceToBatchND", name) {
    addInput(input, false)
    addInput(block_shape, false)
    addInput(paddings, false)
  }
}

fun TF.spaceToDepth(input: Output, block_size: Long, data_format: String = "NHWC", name: String = "SpaceToDepth") = run {
  buildOpTensor("SpaceToDepth", name) {
    addInput(input, false)
    attr("block_size", block_size)
    attr("data_format", data_format)
  }
}

fun TF.split(split_dim: Output, value: Output, num_split: Long, name: String = "Split") = run {
  buildOpTensors("Split", name) {
    addInput(split_dim, false)
    addInput(value, false)
    attr("num_split", num_split)
  }
}

fun TF.splitV(value: Output, size_splits: Output, split_dim: Output, num_split: Long, name: String = "SplitV") = run {
  buildOpTensors("SplitV", name) {
    addInput(value, false)
    addInput(size_splits, false)
    addInput(split_dim, false)
    attr("num_split", num_split)
  }
}

fun TF.squeeze(input: Output, squeeze_dims: Array<Long> = arrayOf(), name: String = "Squeeze") = run {
  buildOpTensor("Squeeze", name) {
    addInput(input, false)
    attr("squeeze_dims", squeeze_dims)
  }
}

fun TF.stopGradient(input: Output, name: String = "StopGradient") = run {
  buildOpTensor("StopGradient", name) {
    addInput(input, false)
  }
}

fun TF.stridedSlice(input: Output, begin: Output, end: Output, strides: Output, begin_mask: Long = 0L, end_mask: Long = 0L, ellipsis_mask: Long = 0L, new_axis_mask: Long = 0L, shrink_axis_mask: Long = 0L, name: String = "StridedSlice") = run {
  buildOpTensor("StridedSlice", name) {
    addInput(input, false)
    addInput(begin, false)
    addInput(end, false)
    addInput(strides, false)
    attr("begin_mask", begin_mask)
    attr("end_mask", end_mask)
    attr("ellipsis_mask", ellipsis_mask)
    attr("new_axis_mask", new_axis_mask)
    attr("shrink_axis_mask", shrink_axis_mask)
  }
}

fun TF.stridedSliceAssign(_ref: Output, begin: Output, end: Output, strides: Output, value: Output, begin_mask: Long = 0L, end_mask: Long = 0L, ellipsis_mask: Long = 0L, new_axis_mask: Long = 0L, shrink_axis_mask: Long = 0L, name: String = "StridedSliceAssign") = run {
  buildOpTensor("StridedSliceAssign", name) {
    addInput(_ref, true)
    addInput(begin, false)
    addInput(end, false)
    addInput(strides, false)
    addInput(value, false)
    attr("begin_mask", begin_mask)
    attr("end_mask", end_mask)
    attr("ellipsis_mask", ellipsis_mask)
    attr("new_axis_mask", new_axis_mask)
    attr("shrink_axis_mask", shrink_axis_mask)
  }
}

fun TF.stridedSliceGrad(shape: Output, begin: Output, end: Output, strides: Output, dy: Output, begin_mask: Long = 0L, end_mask: Long = 0L, ellipsis_mask: Long = 0L, new_axis_mask: Long = 0L, shrink_axis_mask: Long = 0L, name: String = "StridedSliceGrad") = run {
  buildOpTensor("StridedSliceGrad", name) {
    addInput(shape, false)
    addInput(begin, false)
    addInput(end, false)
    addInput(strides, false)
    addInput(dy, false)
    attr("begin_mask", begin_mask)
    attr("end_mask", end_mask)
    attr("ellipsis_mask", ellipsis_mask)
    attr("new_axis_mask", new_axis_mask)
    attr("shrink_axis_mask", shrink_axis_mask)
  }
}

fun TF.tile(input: Output, multiples: Output, name: String = "Tile") = run {
  buildOpTensor("Tile", name) {
    addInput(input, false)
    addInput(multiples, false)
  }
}

fun TF.transpose(x: Output, perm: Output, name: String = "Transpose") = run {
  buildOpTensor("Transpose", name) {
    addInput(x, false)
    addInput(perm, false)
  }
}

fun TF.unique(x: Output, out_idx: Int = DT_INT32, name: String = "Unique") = run {
  buildOpTensors("Unique", name) {
    addInput(x, false)
    attrType("out_idx", out_idx)
  }
}

fun TF.uniqueV2(x: Output, axis: Output, out_idx: Int = DT_INT32, name: String = "UniqueV2") = run {
  buildOpTensors("UniqueV2", name) {
    addInput(x, false)
    addInput(axis, false)
    attrType("out_idx", out_idx)
  }
}

fun TF.uniqueWithCounts(x: Output, out_idx: Int = DT_INT32, name: String = "UniqueWithCounts") = run {
  buildOpTensors("UniqueWithCounts", name) {
    addInput(x, false)
    attrType("out_idx", out_idx)
  }
}

fun TF.uniqueWithCountsV2(x: Output, axis: Output, out_idx: Int = DT_INT32, name: String = "UniqueWithCountsV2") = run {
  buildOpTensors("UniqueWithCountsV2", name) {
    addInput(x, false)
    addInput(axis, false)
    attrType("out_idx", out_idx)
  }
}

fun TF.unpack(value: Output, num: Long, axis: Long = 0L, name: String = "Unpack") = run {
  buildOpTensors("Unpack", name) {
    addInput(value, false)
    attr("num", num)
    attr("axis", axis)
  }
}

fun TF.unravelIndex(indices: Output, dims: Output, name: String = "UnravelIndex") = run {
  buildOpTensor("UnravelIndex", name) {
    addInput(indices, false)
    addInput(dims, false)
  }
}

fun TF._where(input: Output, name: String = "Where") = run {
  buildOpTensor("Where", name) {
    addInput(input, false)
  }
}

fun TF.zerosLike(x: Output, name: String = "ZerosLike") = run {
  buildOpTensor("ZerosLike", name) {
    addInput(x, false)
  }
}

fun TF.broadcastGradientArgs(s0: Output, s1: Output, name: String = "BroadcastGradientArgs") = run {
  buildOpTensors("BroadcastGradientArgs", name) {
    addInput(s0, false)
    addInput(s1, false)
  }
}

fun TF.mirrorPadGrad(input: Output, paddings: Output, mode: String, name: String = "MirrorPadGrad") = run {
  buildOpTensor("MirrorPadGrad", name) {
    addInput(input, false)
    addInput(paddings, false)
    attr("mode", mode)
  }
}

fun TF.refIdentity(input: Output, name: String = "RefIdentity") = run {
  buildOpTensor("RefIdentity", name) {
    addInput(input, true)
  }
}
