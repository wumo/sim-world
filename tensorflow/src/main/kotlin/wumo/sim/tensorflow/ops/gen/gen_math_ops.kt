/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.tensorflow.TF
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.util.Shape

fun TF.abs(x: Output, name: String = "Abs") = run {
  buildOpTensor("Abs", name) {
    addInput(x, false)
  }
}

fun TF.accumulateNV2(inputs: Array<Output>, shape: Shape, name: String = "AccumulateNV2") = run {
  buildOpTensor("AccumulateNV2", name) {
    addInput(inputs, false)
    attr("shape", shape)
  }
}

fun TF.acos(x: Output, name: String = "Acos") = run {
  buildOpTensor("Acos", name) {
    addInput(x, false)
  }
}

fun TF.acosh(x: Output, name: String = "Acosh") = run {
  buildOpTensor("Acosh", name) {
    addInput(x, false)
  }
}

fun TF.add(x: Output, y: Output, name: String = "Add") = run {
  buildOpTensor("Add", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.addN(inputs: Array<Output>, name: String = "AddN") = run {
  buildOpTensor("AddN", name) {
    addInput(inputs, false)
  }
}

fun TF.addV2(x: Output, y: Output, name: String = "AddV2") = run {
  buildOpTensor("AddV2", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.all(input: Output, reduction_indices: Output, keep_dims: Boolean = false, name: String = "All") = run {
  buildOpTensor("All", name) {
    addInput(input, false)
    addInput(reduction_indices, false)
    attr("keep_dims", keep_dims)
  }
}

fun TF.angle(input: Output, tout: Int = DT_FLOAT, name: String = "Angle") = run {
  buildOpTensor("Angle", name) {
    addInput(input, false)
    attrType("Tout", tout)
  }
}

fun TF.any(input: Output, reduction_indices: Output, keep_dims: Boolean = false, name: String = "Any") = run {
  buildOpTensor("Any", name) {
    addInput(input, false)
    addInput(reduction_indices, false)
    attr("keep_dims", keep_dims)
  }
}

fun TF.approximateEqual(x: Output, y: Output, tolerance: Float = 1.0E-5f, name: String = "ApproximateEqual") = run {
  buildOpTensor("ApproximateEqual", name) {
    addInput(x, false)
    addInput(y, false)
    attr("tolerance", tolerance)
  }
}

fun TF.argMax(input: Output, dimension: Output, output_type: Int = DT_INT64, name: String = "ArgMax") = run {
  buildOpTensor("ArgMax", name) {
    addInput(input, false)
    addInput(dimension, false)
    attrType("output_type", output_type)
  }
}

fun TF.argMin(input: Output, dimension: Output, output_type: Int = DT_INT64, name: String = "ArgMin") = run {
  buildOpTensor("ArgMin", name) {
    addInput(input, false)
    addInput(dimension, false)
    attrType("output_type", output_type)
  }
}

fun TF.asin(x: Output, name: String = "Asin") = run {
  buildOpTensor("Asin", name) {
    addInput(x, false)
  }
}

fun TF.asinh(x: Output, name: String = "Asinh") = run {
  buildOpTensor("Asinh", name) {
    addInput(x, false)
  }
}

fun TF.atan(x: Output, name: String = "Atan") = run {
  buildOpTensor("Atan", name) {
    addInput(x, false)
  }
}

fun TF.atan2(y: Output, x: Output, name: String = "Atan2") = run {
  buildOpTensor("Atan2", name) {
    addInput(y, false)
    addInput(x, false)
  }
}

fun TF.atanh(x: Output, name: String = "Atanh") = run {
  buildOpTensor("Atanh", name) {
    addInput(x, false)
  }
}

fun TF.batchMatMul(x: Output, y: Output, adj_x: Boolean = false, adj_y: Boolean = false, name: String = "BatchMatMul") = run {
  buildOpTensor("BatchMatMul", name) {
    addInput(x, false)
    addInput(y, false)
    attr("adj_x", adj_x)
    attr("adj_y", adj_y)
  }
}

fun TF.besselI0e(x: Output, name: String = "BesselI0e") = run {
  buildOpTensor("BesselI0e", name) {
    addInput(x, false)
  }
}

fun TF.besselI1e(x: Output, name: String = "BesselI1e") = run {
  buildOpTensor("BesselI1e", name) {
    addInput(x, false)
  }
}

fun TF.betainc(a: Output, b: Output, x: Output, name: String = "Betainc") = run {
  buildOpTensor("Betainc", name) {
    addInput(a, false)
    addInput(b, false)
    addInput(x, false)
  }
}

fun TF.bincount(arr: Output, size: Output, weights: Output, name: String = "Bincount") = run {
  buildOpTensor("Bincount", name) {
    addInput(arr, false)
    addInput(size, false)
    addInput(weights, false)
  }
}

fun TF.bucketize(input: Output, boundaries: Array<Float>, name: String = "Bucketize") = run {
  buildOpTensor("Bucketize", name) {
    addInput(input, false)
    attr("boundaries", boundaries)
  }
}

fun TF.cast(x: Output, dstT: Int, name: String = "Cast") = run {
  buildOpTensor("Cast", name) {
    addInput(x, false)
    attrType("DstT", dstT)
  }
}

fun TF.ceil(x: Output, name: String = "Ceil") = run {
  buildOpTensor("Ceil", name) {
    addInput(x, false)
  }
}

fun TF.clipByValue(t: Output, clip_value_min: Output, clip_value_max: Output, name: String = "ClipByValue") = run {
  buildOpTensor("ClipByValue", name) {
    addInput(t, false)
    addInput(clip_value_min, false)
    addInput(clip_value_max, false)
  }
}

fun TF.compareAndBitpack(input: Output, threshold: Output, name: String = "CompareAndBitpack") = run {
  buildOpTensor("CompareAndBitpack", name) {
    addInput(input, false)
    addInput(threshold, false)
  }
}

fun TF.complex(real: Output, imag: Output, tout: Int = DT_COMPLEX64, name: String = "Complex") = run {
  buildOpTensor("Complex", name) {
    addInput(real, false)
    addInput(imag, false)
    attrType("Tout", tout)
  }
}

fun TF.complexAbs(x: Output, tout: Int = DT_FLOAT, name: String = "ComplexAbs") = run {
  buildOpTensor("ComplexAbs", name) {
    addInput(x, false)
    attrType("Tout", tout)
  }
}

fun TF.conj(input: Output, name: String = "Conj") = run {
  buildOpTensor("Conj", name) {
    addInput(input, false)
  }
}

fun TF.cos(x: Output, name: String = "Cos") = run {
  buildOpTensor("Cos", name) {
    addInput(x, false)
  }
}

fun TF.cosh(x: Output, name: String = "Cosh") = run {
  buildOpTensor("Cosh", name) {
    addInput(x, false)
  }
}

fun TF.cross(a: Output, b: Output, name: String = "Cross") = run {
  buildOpTensor("Cross", name) {
    addInput(a, false)
    addInput(b, false)
  }
}

fun TF.cumprod(x: Output, axis: Output, exclusive: Boolean = false, reverse: Boolean = false, name: String = "Cumprod") = run {
  buildOpTensor("Cumprod", name) {
    addInput(x, false)
    addInput(axis, false)
    attr("exclusive", exclusive)
    attr("reverse", reverse)
  }
}

fun TF.cumsum(x: Output, axis: Output, exclusive: Boolean = false, reverse: Boolean = false, name: String = "Cumsum") = run {
  buildOpTensor("Cumsum", name) {
    addInput(x, false)
    addInput(axis, false)
    attr("exclusive", exclusive)
    attr("reverse", reverse)
  }
}

fun TF.digamma(x: Output, name: String = "Digamma") = run {
  buildOpTensor("Digamma", name) {
    addInput(x, false)
  }
}

fun TF.div(x: Output, y: Output, name: String = "Div") = run {
  buildOpTensor("Div", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.equal(x: Output, y: Output, name: String = "Equal") = run {
  buildOpTensor("Equal", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.erf(x: Output, name: String = "Erf") = run {
  buildOpTensor("Erf", name) {
    addInput(x, false)
  }
}

fun TF.erfc(x: Output, name: String = "Erfc") = run {
  buildOpTensor("Erfc", name) {
    addInput(x, false)
  }
}

fun TF.exp(x: Output, name: String = "Exp") = run {
  buildOpTensor("Exp", name) {
    addInput(x, false)
  }
}

fun TF.expm1(x: Output, name: String = "Expm1") = run {
  buildOpTensor("Expm1", name) {
    addInput(x, false)
  }
}

fun TF.floor(x: Output, name: String = "Floor") = run {
  buildOpTensor("Floor", name) {
    addInput(x, false)
  }
}

fun TF.floorDiv(x: Output, y: Output, name: String = "FloorDiv") = run {
  buildOpTensor("FloorDiv", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.floorMod(x: Output, y: Output, name: String = "FloorMod") = run {
  buildOpTensor("FloorMod", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.greater(x: Output, y: Output, name: String = "Greater") = run {
  buildOpTensor("Greater", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.greaterEqual(x: Output, y: Output, name: String = "GreaterEqual") = run {
  buildOpTensor("GreaterEqual", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.histogramFixedWidth(values: Output, value_range: Output, nbins: Output, dtype: Int = DT_INT32, name: String = "HistogramFixedWidth") = run {
  buildOpTensor("HistogramFixedWidth", name) {
    addInput(values, false)
    addInput(value_range, false)
    addInput(nbins, false)
    attrType("dtype", dtype)
  }
}

fun TF.igamma(a: Output, x: Output, name: String = "Igamma") = run {
  buildOpTensor("Igamma", name) {
    addInput(a, false)
    addInput(x, false)
  }
}

fun TF.igammac(a: Output, x: Output, name: String = "Igammac") = run {
  buildOpTensor("Igammac", name) {
    addInput(a, false)
    addInput(x, false)
  }
}

fun TF.imag(input: Output, tout: Int = DT_FLOAT, name: String = "Imag") = run {
  buildOpTensor("Imag", name) {
    addInput(input, false)
    attrType("Tout", tout)
  }
}

fun TF.inv(x: Output, name: String = "Inv") = run {
  buildOpTensor("Inv", name) {
    addInput(x, false)
  }
}

fun TF.isFinite(x: Output, name: String = "IsFinite") = run {
  buildOpTensor("IsFinite", name) {
    addInput(x, false)
  }
}

fun TF.isInf(x: Output, name: String = "IsInf") = run {
  buildOpTensor("IsInf", name) {
    addInput(x, false)
  }
}

fun TF.isNan(x: Output, name: String = "IsNan") = run {
  buildOpTensor("IsNan", name) {
    addInput(x, false)
  }
}

fun TF.less(x: Output, y: Output, name: String = "Less") = run {
  buildOpTensor("Less", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.lessEqual(x: Output, y: Output, name: String = "LessEqual") = run {
  buildOpTensor("LessEqual", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.lgamma(x: Output, name: String = "Lgamma") = run {
  buildOpTensor("Lgamma", name) {
    addInput(x, false)
  }
}

fun TF.linSpace(start: Output, stop: Output, num: Output, name: String = "LinSpace") = run {
  buildOpTensor("LinSpace", name) {
    addInput(start, false)
    addInput(stop, false)
    addInput(num, false)
  }
}

fun TF.log(x: Output, name: String = "Log") = run {
  buildOpTensor("Log", name) {
    addInput(x, false)
  }
}

fun TF.log1p(x: Output, name: String = "Log1p") = run {
  buildOpTensor("Log1p", name) {
    addInput(x, false)
  }
}

fun TF.logicalAnd(x: Output, y: Output, name: String = "LogicalAnd") = run {
  buildOpTensor("LogicalAnd", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.logicalNot(x: Output, name: String = "LogicalNot") = run {
  buildOpTensor("LogicalNot", name) {
    addInput(x, false)
  }
}

fun TF.logicalOr(x: Output, y: Output, name: String = "LogicalOr") = run {
  buildOpTensor("LogicalOr", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.matMul(a: Output, b: Output, transpose_a: Boolean = false, transpose_b: Boolean = false, name: String = "MatMul") = run {
  buildOpTensor("MatMul", name) {
    addInput(a, false)
    addInput(b, false)
    attr("transpose_a", transpose_a)
    attr("transpose_b", transpose_b)
  }
}

fun TF.max(input: Output, reduction_indices: Output, keep_dims: Boolean = false, name: String = "Max") = run {
  buildOpTensor("Max", name) {
    addInput(input, false)
    addInput(reduction_indices, false)
    attr("keep_dims", keep_dims)
  }
}

fun TF.maximum(x: Output, y: Output, name: String = "Maximum") = run {
  buildOpTensor("Maximum", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.mean(input: Output, reduction_indices: Output, keep_dims: Boolean = false, name: String = "Mean") = run {
  buildOpTensor("Mean", name) {
    addInput(input, false)
    addInput(reduction_indices, false)
    attr("keep_dims", keep_dims)
  }
}

fun TF.min(input: Output, reduction_indices: Output, keep_dims: Boolean = false, name: String = "Min") = run {
  buildOpTensor("Min", name) {
    addInput(input, false)
    addInput(reduction_indices, false)
    attr("keep_dims", keep_dims)
  }
}

fun TF.minimum(x: Output, y: Output, name: String = "Minimum") = run {
  buildOpTensor("Minimum", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.mod(x: Output, y: Output, name: String = "Mod") = run {
  buildOpTensor("Mod", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.mul(x: Output, y: Output, name: String = "Mul") = run {
  buildOpTensor("Mul", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.neg(x: Output, name: String = "Neg") = run {
  buildOpTensor("Neg", name) {
    addInput(x, false)
  }
}

fun TF.notEqual(x: Output, y: Output, name: String = "NotEqual") = run {
  buildOpTensor("NotEqual", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.polygamma(a: Output, x: Output, name: String = "Polygamma") = run {
  buildOpTensor("Polygamma", name) {
    addInput(a, false)
    addInput(x, false)
  }
}

fun TF.pow(x: Output, y: Output, name: String = "Pow") = run {
  buildOpTensor("Pow", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.prod(input: Output, reduction_indices: Output, keep_dims: Boolean = false, name: String = "Prod") = run {
  buildOpTensor("Prod", name) {
    addInput(input, false)
    addInput(reduction_indices, false)
    attr("keep_dims", keep_dims)
  }
}

fun TF.quantizeDownAndShrinkRange(input: Output, input_min: Output, input_max: Output, out_type: Int, name: String = "QuantizeDownAndShrinkRange") = run {
  buildOpTensors("QuantizeDownAndShrinkRange", name) {
    addInput(input, false)
    addInput(input_min, false)
    addInput(input_max, false)
    attrType("out_type", out_type)
  }
}

fun TF.quantizedAdd(x: Output, y: Output, min_x: Output, max_x: Output, min_y: Output, max_y: Output, toutput: Int = DT_QINT32, name: String = "QuantizedAdd") = run {
  buildOpTensors("QuantizedAdd", name) {
    addInput(x, false)
    addInput(y, false)
    addInput(min_x, false)
    addInput(max_x, false)
    addInput(min_y, false)
    addInput(max_y, false)
    attrType("Toutput", toutput)
  }
}

fun TF.quantizedMatMul(a: Output, b: Output, min_a: Output, max_a: Output, min_b: Output, max_b: Output, toutput: Int = DT_QINT32, transpose_a: Boolean = false, transpose_b: Boolean = false, tactivation: Int = DT_QUINT8, name: String = "QuantizedMatMul") = run {
  buildOpTensors("QuantizedMatMul", name) {
    addInput(a, false)
    addInput(b, false)
    addInput(min_a, false)
    addInput(max_a, false)
    addInput(min_b, false)
    addInput(max_b, false)
    attrType("Toutput", toutput)
    attr("transpose_a", transpose_a)
    attr("transpose_b", transpose_b)
    attrType("Tactivation", tactivation)
  }
}

fun TF.quantizedMul(x: Output, y: Output, min_x: Output, max_x: Output, min_y: Output, max_y: Output, toutput: Int = DT_QINT32, name: String = "QuantizedMul") = run {
  buildOpTensors("QuantizedMul", name) {
    addInput(x, false)
    addInput(y, false)
    addInput(min_x, false)
    addInput(max_x, false)
    addInput(min_y, false)
    addInput(max_y, false)
    attrType("Toutput", toutput)
  }
}

fun TF.range(start: Output, limit: Output, delta: Output, name: String = "Range") = run {
  buildOpTensor("Range", name) {
    addInput(start, false)
    addInput(limit, false)
    addInput(delta, false)
  }
}

fun TF.real(input: Output, tout: Int = DT_FLOAT, name: String = "Real") = run {
  buildOpTensor("Real", name) {
    addInput(input, false)
    attrType("Tout", tout)
  }
}

fun TF.realDiv(x: Output, y: Output, name: String = "RealDiv") = run {
  buildOpTensor("RealDiv", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.reciprocal(x: Output, name: String = "Reciprocal") = run {
  buildOpTensor("Reciprocal", name) {
    addInput(x, false)
  }
}

fun TF.requantizationRange(input: Output, input_min: Output, input_max: Output, name: String = "RequantizationRange") = run {
  buildOpTensors("RequantizationRange", name) {
    addInput(input, false)
    addInput(input_min, false)
    addInput(input_max, false)
  }
}

fun TF.requantize(input: Output, input_min: Output, input_max: Output, requested_output_min: Output, requested_output_max: Output, out_type: Int, name: String = "Requantize") = run {
  buildOpTensors("Requantize", name) {
    addInput(input, false)
    addInput(input_min, false)
    addInput(input_max, false)
    addInput(requested_output_min, false)
    addInput(requested_output_max, false)
    attrType("out_type", out_type)
  }
}

fun TF.rint(x: Output, name: String = "Rint") = run {
  buildOpTensor("Rint", name) {
    addInput(x, false)
  }
}

fun TF.round(x: Output, name: String = "Round") = run {
  buildOpTensor("Round", name) {
    addInput(x, false)
  }
}

fun TF.rsqrt(x: Output, name: String = "Rsqrt") = run {
  buildOpTensor("Rsqrt", name) {
    addInput(x, false)
  }
}

fun TF.segmentMax(data: Output, segment_ids: Output, name: String = "SegmentMax") = run {
  buildOpTensor("SegmentMax", name) {
    addInput(data, false)
    addInput(segment_ids, false)
  }
}

fun TF.segmentMean(data: Output, segment_ids: Output, name: String = "SegmentMean") = run {
  buildOpTensor("SegmentMean", name) {
    addInput(data, false)
    addInput(segment_ids, false)
  }
}

fun TF.segmentMin(data: Output, segment_ids: Output, name: String = "SegmentMin") = run {
  buildOpTensor("SegmentMin", name) {
    addInput(data, false)
    addInput(segment_ids, false)
  }
}

fun TF.segmentProd(data: Output, segment_ids: Output, name: String = "SegmentProd") = run {
  buildOpTensor("SegmentProd", name) {
    addInput(data, false)
    addInput(segment_ids, false)
  }
}

fun TF.segmentSum(data: Output, segment_ids: Output, name: String = "SegmentSum") = run {
  buildOpTensor("SegmentSum", name) {
    addInput(data, false)
    addInput(segment_ids, false)
  }
}

fun TF.select(condition: Output, t: Output, e: Output, name: String = "Select") = run {
  buildOpTensor("Select", name) {
    addInput(condition, false)
    addInput(t, false)
    addInput(e, false)
  }
}

fun TF.sigmoid(x: Output, name: String = "Sigmoid") = run {
  buildOpTensor("Sigmoid", name) {
    addInput(x, false)
  }
}

fun TF.sign(x: Output, name: String = "Sign") = run {
  buildOpTensor("Sign", name) {
    addInput(x, false)
  }
}

fun TF.sin(x: Output, name: String = "Sin") = run {
  buildOpTensor("Sin", name) {
    addInput(x, false)
  }
}

fun TF.sinh(x: Output, name: String = "Sinh") = run {
  buildOpTensor("Sinh", name) {
    addInput(x, false)
  }
}

fun TF.sparseMatMul(a: Output, b: Output, transpose_a: Boolean = false, transpose_b: Boolean = false, a_is_sparse: Boolean = false, b_is_sparse: Boolean = false, name: String = "SparseMatMul") = run {
  buildOpTensor("SparseMatMul", name) {
    addInput(a, false)
    addInput(b, false)
    attr("transpose_a", transpose_a)
    attr("transpose_b", transpose_b)
    attr("a_is_sparse", a_is_sparse)
    attr("b_is_sparse", b_is_sparse)
  }
}

fun TF.sparseSegmentMean(data: Output, indices: Output, segment_ids: Output, name: String = "SparseSegmentMean") = run {
  buildOpTensor("SparseSegmentMean", name) {
    addInput(data, false)
    addInput(indices, false)
    addInput(segment_ids, false)
  }
}

fun TF.sparseSegmentMeanGrad(grad: Output, indices: Output, segment_ids: Output, output_dim0: Output, name: String = "SparseSegmentMeanGrad") = run {
  buildOpTensor("SparseSegmentMeanGrad", name) {
    addInput(grad, false)
    addInput(indices, false)
    addInput(segment_ids, false)
    addInput(output_dim0, false)
  }
}

fun TF.sparseSegmentMeanWithNumSegments(data: Output, indices: Output, segment_ids: Output, num_segments: Output, name: String = "SparseSegmentMeanWithNumSegments") = run {
  buildOpTensor("SparseSegmentMeanWithNumSegments", name) {
    addInput(data, false)
    addInput(indices, false)
    addInput(segment_ids, false)
    addInput(num_segments, false)
  }
}

fun TF.sparseSegmentSqrtN(data: Output, indices: Output, segment_ids: Output, name: String = "SparseSegmentSqrtN") = run {
  buildOpTensor("SparseSegmentSqrtN", name) {
    addInput(data, false)
    addInput(indices, false)
    addInput(segment_ids, false)
  }
}

fun TF.sparseSegmentSqrtNGrad(grad: Output, indices: Output, segment_ids: Output, output_dim0: Output, name: String = "SparseSegmentSqrtNGrad") = run {
  buildOpTensor("SparseSegmentSqrtNGrad", name) {
    addInput(grad, false)
    addInput(indices, false)
    addInput(segment_ids, false)
    addInput(output_dim0, false)
  }
}

fun TF.sparseSegmentSqrtNWithNumSegments(data: Output, indices: Output, segment_ids: Output, num_segments: Output, name: String = "SparseSegmentSqrtNWithNumSegments") = run {
  buildOpTensor("SparseSegmentSqrtNWithNumSegments", name) {
    addInput(data, false)
    addInput(indices, false)
    addInput(segment_ids, false)
    addInput(num_segments, false)
  }
}

fun TF.sparseSegmentSum(data: Output, indices: Output, segment_ids: Output, name: String = "SparseSegmentSum") = run {
  buildOpTensor("SparseSegmentSum", name) {
    addInput(data, false)
    addInput(indices, false)
    addInput(segment_ids, false)
  }
}

fun TF.sparseSegmentSumWithNumSegments(data: Output, indices: Output, segment_ids: Output, num_segments: Output, name: String = "SparseSegmentSumWithNumSegments") = run {
  buildOpTensor("SparseSegmentSumWithNumSegments", name) {
    addInput(data, false)
    addInput(indices, false)
    addInput(segment_ids, false)
    addInput(num_segments, false)
  }
}

fun TF.sqrt(x: Output, name: String = "Sqrt") = run {
  buildOpTensor("Sqrt", name) {
    addInput(x, false)
  }
}

fun TF.square(x: Output, name: String = "Square") = run {
  buildOpTensor("Square", name) {
    addInput(x, false)
  }
}

fun TF.squaredDifference(x: Output, y: Output, name: String = "SquaredDifference") = run {
  buildOpTensor("SquaredDifference", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.sub(x: Output, y: Output, name: String = "Sub") = run {
  buildOpTensor("Sub", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.sum(input: Output, reduction_indices: Output, keep_dims: Boolean = false, name: String = "Sum") = run {
  buildOpTensor("Sum", name) {
    addInput(input, false)
    addInput(reduction_indices, false)
    attr("keep_dims", keep_dims)
  }
}

fun TF.tan(x: Output, name: String = "Tan") = run {
  buildOpTensor("Tan", name) {
    addInput(x, false)
  }
}

fun TF.tanh(x: Output, name: String = "Tanh") = run {
  buildOpTensor("Tanh", name) {
    addInput(x, false)
  }
}

fun TF.truncateDiv(x: Output, y: Output, name: String = "TruncateDiv") = run {
  buildOpTensor("TruncateDiv", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.truncateMod(x: Output, y: Output, name: String = "TruncateMod") = run {
  buildOpTensor("TruncateMod", name) {
    addInput(x, false)
    addInput(y, false)
  }
}

fun TF.unsortedSegmentMax(data: Output, segment_ids: Output, num_segments: Output, name: String = "UnsortedSegmentMax") = run {
  buildOpTensor("UnsortedSegmentMax", name) {
    addInput(data, false)
    addInput(segment_ids, false)
    addInput(num_segments, false)
  }
}

fun TF.unsortedSegmentMin(data: Output, segment_ids: Output, num_segments: Output, name: String = "UnsortedSegmentMin") = run {
  buildOpTensor("UnsortedSegmentMin", name) {
    addInput(data, false)
    addInput(segment_ids, false)
    addInput(num_segments, false)
  }
}

fun TF.unsortedSegmentProd(data: Output, segment_ids: Output, num_segments: Output, name: String = "UnsortedSegmentProd") = run {
  buildOpTensor("UnsortedSegmentProd", name) {
    addInput(data, false)
    addInput(segment_ids, false)
    addInput(num_segments, false)
  }
}

fun TF.unsortedSegmentSum(data: Output, segment_ids: Output, num_segments: Output, name: String = "UnsortedSegmentSum") = run {
  buildOpTensor("UnsortedSegmentSum", name) {
    addInput(data, false)
    addInput(segment_ids, false)
    addInput(num_segments, false)
  }
}

fun TF.zeta(x: Output, q: Output, name: String = "Zeta") = run {
  buildOpTensor("Zeta", name) {
    addInput(x, false)
    addInput(q, false)
  }
}

fun TF.igammaGradA(a: Output, x: Output, name: String = "IgammaGradA") = run {
  buildOpTensor("IgammaGradA", name) {
    addInput(a, false)
    addInput(x, false)
  }
}

fun TF.invGrad(y: Output, dy: Output, name: String = "InvGrad") = run {
  buildOpTensor("InvGrad", name) {
    addInput(y, false)
    addInput(dy, false)
  }
}

fun TF.reciprocalGrad(y: Output, dy: Output, name: String = "ReciprocalGrad") = run {
  buildOpTensor("ReciprocalGrad", name) {
    addInput(y, false)
    addInput(dy, false)
  }
}

fun TF.rsqrtGrad(y: Output, dy: Output, name: String = "RsqrtGrad") = run {
  buildOpTensor("RsqrtGrad", name) {
    addInput(y, false)
    addInput(dy, false)
  }
}

fun TF.sigmoidGrad(y: Output, dy: Output, name: String = "SigmoidGrad") = run {
  buildOpTensor("SigmoidGrad", name) {
    addInput(y, false)
    addInput(dy, false)
  }
}

fun TF.sqrtGrad(y: Output, dy: Output, name: String = "SqrtGrad") = run {
  buildOpTensor("SqrtGrad", name) {
    addInput(y, false)
    addInput(dy, false)
  }
}

fun TF.tanhGrad(y: Output, dy: Output, name: String = "TanhGrad") = run {
  buildOpTensor("TanhGrad", name) {
    addInput(y, false)
    addInput(dy, false)
  }
}