package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.*
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.util.Dimension
import wumo.sim.util.arange
import wumo.sim.util.i

fun TF.abs(x: Tensor, name: String = "Abs") = unaryOp("Abs", x, name)
fun TF.accumulateNV2(vararg input: Tensor, shape: Dimension, name: String = "AccumulateNV2"): Tensor {
  val op = g.nodeBuilder("AccumulateNV2", ctxNs.getUniqueFullName(name))
      .addInputList(input as Array<Tensor>)
      .setAttr("shape", shape)
      .build()
  return Tensor(op, 0)
}

fun TF.acos(x: Tensor, name: String = "Acos") = unaryOp("Acos", x, name)
fun TF.acosh(x: Tensor, name: String = "Acosh") = unaryOp("Acosh", x, name)

operator fun Tensor.plus(b: Tensor) = tf.add(this, b)
fun TF.add(a: Tensor, b: Tensor, name: String = "Add") =
    binaryOp("Add", a, b, name)

fun TF.addV2(a: Tensor, b: Tensor, name: String = "AddV2") =
    binaryOp("AddV2", a, b, name)

fun TF.addN(vararg a: Tensor, name: String = "AddN") =
    name_scope(name) {
      val op = g.nodeBuilder("AddN", ctxNs.fullName)
          .addInputList(a as Array<Tensor>)
          .build()
      Tensor(op, 0)
    }

fun TF.all(input: Tensor, axis: Tensor, keep_dims: Boolean = false, name: String = "All") =
    naryOp("All", input, axis, name = name) {
      setAttr("keep_dims", keep_dims)
    }

fun TF.angle(input: Tensor, Tout: Int = DT_FLOAT, name: String = "Angle") =
    naryOp("Angle", input, name = name) {
      setAttrType("Tout", Tout)
    }

fun TF.any(input: Tensor, axis: Tensor, keep_dims: Boolean = false, name: String = "Any") =
    naryOp("Any", input, axis, name = name) {
      setAttr("keep_dims", keep_dims)
    }

fun TF.approximateEqual(x: Tensor, y: Tensor, tolerance: Float = 1e-5f, name: String = "ApproximateEqual") =
    naryOp("ApproximateEqual", x, y, name = name) {
      setAttr("tolerance", tolerance)
    }

fun TF.argmax(a: Tensor, axis: Int, output_type: Int = DT_INT32, name: String = "ArgMax") =
    name_scope(name) {
      val op = g.nodeBuilder("ArgMax", ctxNs.fullName)
          .addInput(a)
          .addInput(const(axis, "dimension"))
          .setAttrType("output_type", output_type)
          .build()
      Tensor(op, 0)
    }

fun TF.argmax(a: Tensor, axis: Tensor, name: String = "ArgMax") =
    name_scope(name) {
      val op = g.nodeBuilder("ArgMax", ctxNs.fullName)
          .addInput(a)
          .addInput(axis)
          .setAttrType("output_type", DT_INT32)
          .build()
      Tensor(op, 0)
    }

fun TF.argmin(a: Tensor, axis: Tensor, name: String = "ArgMin") =
    binaryOp("ArgMin", a, axis, name)

fun TF.argmin(a: Tensor, axis: Int, name: String = "ArgMin") =
    name_scope(name) {
      val op = g.nodeBuilder("ArgMin", ctxNs.fullName)
          .addInput(a)
          .addInput(const(axis, "dimension"))
          .build()
      Tensor(op, 0)
    }

fun TF.asin(x: Tensor, name: String = "Asin") = unaryOp("Asin", x, name)
fun TF.asinh(x: Tensor, name: String = "Asinh") = unaryOp("Asinh", x, name)
fun TF.atan(x: Tensor, name: String = "Atan") = unaryOp("Atan", x, name)
fun TF.atan2(x: Tensor, y: Tensor, name: String = "Atan2") = binaryOp("Atan2", x, y, name)
fun TF.atanh(x: Tensor, name: String = "Atanh") = unaryOp("Atanh", x, name)

fun TF.batchMatMul(a: Tensor, b: Tensor, adj_x: Boolean = false, adj_y: Boolean = false, name: String = "BatchMatMul") =
    naryOp("BatchMatMul", a, b, name = name) {
      setAttr("adj_x", adj_x).setAttr("adj_y", adj_y)
    }

fun TF.besselI0e(x: Tensor, name: String = "BesselI0e") = unaryOp("BesselI0e", x, name)

fun TF.besselI1e(x: Tensor, name: String = "BesselI1e") = unaryOp("BesselI1e", x, name)

fun TF.betainc(a: Tensor, b: Tensor, x: Tensor, name: String = "Betainc") =
    ternaryOp("Betainc", a, b, x, name = name)

fun TF.bincount(arr: Tensor, size: Tensor, weights: Tensor, name: String = "Bincount") =
    ternaryOp("Bincount", arr, size, weights, name = name)

fun TF.Bucketize(input: Tensor, boundaries: FloatArray, name: String = "Bucketize") =
    naryOp("Bucketize", input, name = name) {
      setAttr("boundaries", boundaries)
    }

fun TF.cast(x: Tensor, dstT: Int, name: String = "Cast"): Tensor {
  return if (x.dtype == dstT) x
  else {
    val op = g.nodeBuilder("Cast", ctxNs.getUniqueFullName(name))
        .addInput(x)
        .setAttrType("DstT", dstT)
        .build()
    Tensor(op, 0)
  }
}

fun TF.ceil(x: Tensor, name: String = "Ceil") = unaryOp("Ceil", x, name)

fun TF.clipByValue(t: Tensor, clip_value_min: Tensor, clip_value_max: Tensor, name: String = "ClipByValue") =
    ternaryOp("ClipByValue", t, clip_value_min, clip_value_max, name = name)

fun TF.compareAndBitpack(input: Tensor, threshold: Tensor, name: String = "CompareAndBitpack") =
    binaryOp("CompareAndBitpack", input, threshold, name)

fun TF.complex(input: Tensor, threshold: Tensor, Tout: Int = DT_COMPLEX64, name: String = "Complex") =
    naryOp("Complex", input, threshold, name = name) {
      setAttr("Tout", Tout)
    }

fun TF.ComplexAbs(x: Tensor, Tout: Int = DT_FLOAT, name: String = "ComplexAbs") =
    naryOp("ComplexAbs", x, name = name) {
      setAttr("Tout", Tout)
    }

fun TF.conj(x: Tensor, name: String = "Conj") = unaryOp("Conj", x, name)

fun TF.cos(x: Tensor, name: String = "Cos") = unaryOp("Cos", x, name)
fun TF.cosh(x: Tensor, name: String = "Cosh") = unaryOp("Cosh", x, name)

fun TF.cross(a: Tensor, b: Tensor, name: String = "Cross") =
    binaryOp("Cross", a, b, name)

fun TF.cumprod(x: Tensor, axis: Tensor, exclusive: Boolean = false, reverse: Boolean = false, name: String = "Cumprod") =
    naryOp("Cumprod", x, axis, name = name) {
      setAttr("exclusive", exclusive).setAttr("reverse", reverse)
    }

fun TF.cumsum(x: Tensor, axis: Tensor, exclusive: Boolean = false, reverse: Boolean = false, name: String = "Cumsum") =
    naryOp("Cumsum", x, axis, name = name) {
      setAttr("exclusive", exclusive).setAttr("reverse", reverse)
    }

fun TF.digamma(x: Tensor, name: String = "Digamma") = unaryOp("Digamma", x, name)

operator fun Tensor.div(b: Tensor) = tf.div(this, b)
fun TF.div(a: Tensor, b: Tensor, name: String = "Div") =
    binaryOp("Div", a, b, name)

fun TF.equal(a: Tensor, b: Tensor, name: String = "Equal") =
    binaryOp("Equal", a, b, name)

fun TF.erf(x: Tensor, name: String = "Erf") = unaryOp("Erf", x, name)
fun TF.erfc(x: Tensor, name: String = "Erfc") = unaryOp("Erfc", x, name)

fun TF.exp(x: Tensor, name: String = "Exp") = unaryOp("Exp", x, name)
fun TF.expm1(x: Tensor, name: String = "Expm1") = unaryOp("Expm1", x, name)

fun TF.floor(x: Tensor, name: String = "Floor") = unaryOp("Floor", x, name)
fun TF.floorDiv(a: Tensor, b: Tensor, name: String = "FloorDiv") =
    binaryOp("FloorDiv", a, b, name)

fun TF.floorMod(a: Tensor, b: Tensor, name: String = "FloorMod") =
    binaryOp("FloorMod", a, b, name)

fun TF.greater(a: Tensor, b: Tensor, name: String = "Greater") =
    binaryOp("Greater", a, b, name)

fun TF.greaterEqual(a: Tensor, b: Tensor, name: String = "GreaterEqual") =
    binaryOp("GreaterEqual", a, b, name)

fun TF.HistogramFixedWidth(values: Tensor, value_range: Tensor, nbins: Tensor, dtype: Int = DT_INT32, name: String = "HistogramFixedWidth") =
    naryOp("HistogramFixedWidth", values, value_range, nbins, name = name) {
      setAttrType("dtype", dtype)
    }

fun TF.igamma(a: Tensor, b: Tensor, name: String = "Igamma") =
    binaryOp("Igamma", a, b, name)

fun TF.igammac(a: Tensor, b: Tensor, name: String = "Igammac") =
    binaryOp("Igammac", a, b, name)

fun TF.imag(x: Tensor, Tout: Int = DT_FLOAT, name: String = "Imag") =
    naryOp("Imag", x, name = name) {
      setAttrType("Tout", Tout)
    }

fun TF.inv(x: Tensor, name: String = "Inv") = unaryOp("Inv", x, name)

fun TF.isFinite(x: Tensor, name: String = "IsFinite") = unaryOp("IsFinite", x, name)
fun TF.isInf(x: Tensor, name: String = "IsInf") = unaryOp("IsInf", x, name)
fun TF.isNan(x: Tensor, name: String = "IsNan") = unaryOp("IsNan", x, name)

fun TF.less(a: Tensor, b: Tensor, name: String = "Less") =
    binaryOp("Less", a, b, name)

fun TF.lessEqual(a: Tensor, b: Tensor, name: String = "LessEqual") =
    binaryOp("LessEqual", a, b, name)

fun TF.lgamma(x: Tensor, name: String = "Lgamma") = unaryOp("Lgamma", x, name)

fun TF.LinSpace(start: Tensor, stop: Tensor, num: Tensor, name: String = "LinSpace") =
    ternaryOp("LinSpace", start, stop, num, name)

fun TF.log(a: Tensor, name: String = "Log") =
    unaryOp("Log", a, name)

fun TF.log1p(a: Tensor, name: String = "Log1p") =
    unaryOp("Log1p", a, name)

fun TF.logicalAnd(a: Tensor, b: Tensor, name: String = "LogicalAnd") =
    binaryOp("LogicalAnd", a, b, name)

fun TF.logicalNot(a: Tensor, name: String = "LogicalNot") =
    unaryOp("LogicalNot", a, name)

fun TF.logicalOr(a: Tensor, b: Tensor, name: String = "LogicalOr") =
    binaryOp("LogicalOr", a, b, name)

fun TF.matmul(a: Tensor, b: Tensor, transpose_a: Boolean = false, transpose_b: Boolean = false, name: String = "MatMul") =
    naryOp("MatMul", a, b, name = name) {
      setAttr("transpose_a", transpose_a)
      setAttr("transpose_b", transpose_b)
    }

fun TF.max(input: Tensor, axis: Tensor, keep_dims: Boolean = false, name: String = "Max") =
    naryOp("Max", input, axis, name = name) {
      setAttr("keep_dims", keep_dims)
    }

fun TF.maximum(x: Tensor, y: Tensor, name: String = "Maximum") =
    binaryOp("Maximum", x, y, name = name)

fun TF.reductionDims(x: Tensor, axis: Tensor?): Tensor {
  if (axis != null) return axis
  //Fast path: avoid creating Rank and Range ops if ndims is known.
  return const(arange(x.shape().rank()))
  //TODO SparseTensor
  // Otherwise, we rely on Range and Rank to do the right thing at run-time.
  return range(const(0), rank(x))
}

fun TF.mean(input: Tensor, axis: Tensor? = null, keep_dims: Boolean = false, name: String = "Mean") =
    naryOp("Mean", input, reductionDims(input, axis), name = name) {
      setAttr("keep_dims", keep_dims)
    }

fun TF.min(input: Tensor, axis: Tensor, keep_dims: Boolean = false, name: String = "Min") =
    naryOp("Min", input, axis, name = name) {
      setAttr("keep_dims", keep_dims)
    }

fun TF.minimum(x: Tensor, y: Tensor, name: String = "Minimum") =
    binaryOp("Minimum", x, y, name = name)

fun TF.mod(x: Tensor, y: Tensor, name: String = "Mod") =
    binaryOp("Mod", x, y, name = name)

operator fun Tensor.times(b: Tensor) = tf.mul(this, b)
fun TF.mul(a: Tensor, b: Tensor, name: String = "Mul") =
    binaryOp("Mul", a, b, name)

operator fun Tensor.unaryMinus() = tf.neg(this)
fun TF.neg(a: Tensor, name: String = "Neg") =
    unaryOp("Neg", a, name)

fun TF.notEqual(a: Tensor, b: Tensor, name: String = "NotEqual") =
    binaryOp("NotEqual", a, b, name)

fun TF.polygamma(a: Tensor, b: Tensor, name: String = "Polygamma") =
    binaryOp("Polygamma", a, b, name)

fun TF.pow(a: Tensor, b: Tensor, name: String = "Pow") =
    binaryOp("Pow", a, b, name)

fun TF.prod(input: Tensor, axis: Tensor, keep_dims: Boolean = false, name: String = "Prod") =
    naryOp("Prod", input, axis, name = name) {
      setAttr("keep_dims", keep_dims)
    }

fun TF.quantizeDownAndShrinkRange(input: Tensor, input_min: Tensor, input_max: Tensor, out_type: Int,
                                  name: String = "QuantizeDownAndShrinkRange") =
    naryOp("QuantizeDownAndShrinkRange", input, input_min, input_max, name = name, outputs = 3) {
      setAttrType("out_type", out_type)
    }

fun TF.quantizedAdd(x: Tensor, y: Tensor, min_x: Tensor, max_x: Tensor, min_y: Tensor, max_y: Tensor, Toutput: Int = DT_QINT32,
                    name: String = "QuantizedAdd") =
    naryOp("QuantizedAdd", x, y, min_x, max_x, min_y, max_y, name = name, outputs = 3) {
      setAttrType("Toutput", Toutput)
    }

fun TF.quantizedMatMul(a: Tensor, b: Tensor, min_a: Tensor, max_a: Tensor, min_b: Tensor, max_b: Tensor,
                       Toutput: Int = DT_QINT32,
                       transpose_a: Boolean = false,
                       transpose_b: Boolean = false,
                       Tactivation: Int = DT_QUINT8,
                       name: String = "QuantizedMatMul") =
    naryOp("QuantizedMatMul", a, b, min_a, max_a, min_b, max_b, name = name, outputs = 3) {
      setAttrType("Toutput", Toutput)
      setAttr("transpose_a", transpose_a)
      setAttr("transpose_b", transpose_b)
      setAttrType("Tactivation", Tactivation)
    }

fun TF.quantizedMul(a: Tensor, b: Tensor, min_a: Tensor, max_a: Tensor, min_b: Tensor, max_b: Tensor,
                    Toutput: Int = DT_QINT32,
                    name: String = "QuantizedMul") =
    naryOp("QuantizedMul", a, b, min_a, max_a, min_b, max_b, name = name, outputs = 3) {
      setAttrType("Toutput", Toutput)
    }


private val dtype_hierarchy = mapOf(DT_INT32 to 0,
                                    DT_INT64 to 1,
                                    DT_FLOAT to 2,
                                    DT_DOUBLE to 3)

/**
Creates a sequence of numbers.

Creates a sequence of numbers that begins at `start` and extends by
increments of `delta` up to but not including `limit`.

The dtype of the resulting tensor is inferred from the inputs unless
it is provided explicitly.

Like the Python builtin `range`, `start` defaults to 0, so that
`range(n) = range(0, n)`.

For example:

```python
start = 3
limit = 18
delta = 3
tf.range(start, limit, delta)  # [3, 6, 9, 12, 15]

start = 3
limit = 1
delta = -0.5
tf.range(start, limit, delta)  # [3, 2.5, 2, 1.5]

limit = 5
tf.range(limit)  # [0, 1, 2, 3, 4]
```
 * @param start: A 0-D `Tensor` (scalar). Acts as first entry in the range if
 * `limit` is not None; otherwise, acts as range limit and first entry
 * defaults to 0.
 * @param limit: A 0-D `Tensor` (scalar). Upper limit of sequence,
 * exclusive. If None, defaults to the value of `start` while the first
 * entry of the range defaults to 0.
 * @param delta: A 0-D `Tensor` (scalar). Number that increments
 * `start`. Defaults to 1.
 * @param name: A name for the operation. Defaults to "range".
 * @return An 1-D `Tensor` of type `dtype`.
 */
fun TF.range(start: Tensor, limit: Tensor, delta: Tensor = const(1), name: String = "Range"): Tensor {
  val dtypes = i(start.dtype, limit.dtype, delta.dtype)
  val inferred_dtype = dtypes.maxBy { dtype_hierarchy[it]!! }!!
  val start = cast(start, inferred_dtype)
  val limit = cast(limit, inferred_dtype)
  val delta = cast(delta, inferred_dtype)
  val op = g.nodeBuilder("Range", ctxNs.getUniqueFullName(name))
      .addInput(start)
      .addInput(limit)
      .addInput(delta)
      .build()
  return Tensor(op, 0)
}

fun TF.range(start: Tensor, limit: Tensor, delta: Tensor, dtype: Int, name: String = "Range"): Tensor {
  val op = g.nodeBuilder("Range", ctxNs.getUniqueFullName(name))
      .addInput(start)
      .addInput(limit)
      .addInput(delta)
      .build()
  return Tensor(op, 0)
}

fun TF.real(input: Tensor, Tout: Int = DT_FLOAT, name: String = "Real") =
    naryOp("Real", input, name = name) {
      setAttrType("Tout", Tout)
    }

fun TF.realDiv(a: Tensor, b: Tensor, name: String = "RealDiv") =
    binaryOp("RealDiv", a, b, name)

fun TF.reciprocal(x: Tensor, name: String = "Reciprocal") =
    unaryOp("Reciprocal", x, name)

fun TF.requantizationRange(input: Tensor, input_min: Tensor, input_max: Tensor, name: String = "RequantizationRange") =
    naryOp("RequantizationRange", input, input_min, input_max, name = name, outputs = 2)

fun TF.requantize(input: Tensor, input_min: Tensor, input_max: Tensor, requested_output_min: Tensor, requested_output_max: Tensor, out_type: Int,
                  name: String = "Requantize") =
    naryOp("Requantize", input, input_min, input_max, requested_output_min, requested_output_max, name = name, outputs = 3) {
      setAttrType("out_type", out_type)
    }

fun TF.rint(x: Tensor, name: String = "Rint") = unaryOp("Rint", x, name)

fun TF.round(x: Tensor, name: String = "Round") = unaryOp("Round", x, name)

fun TF.rsqrt(x: Tensor, name: String = "Rsqrt") = unaryOp("Rsqrt", x, name)

fun TF.segmentMax(data: Tensor, segmen_ids: Tensor, name: String = "SegmentMax") =
    binaryOp("SegmentMax", data, segmen_ids, name)

fun TF.segmentMean(data: Tensor, segment_ids: Tensor, name: String = "SegmentMean") =
    binaryOp("SegmentMean", data, segment_ids, name)

fun TF.segmentMin(data: Tensor, segment_ids: Tensor, name: String = "SegmentMin") =
    binaryOp("SegmentMin", data, segment_ids, name)

fun TF.segmentProd(data: Tensor, segment_ids: Tensor, name: String = "SegmentProd") =
    binaryOp("SegmentProd", data, segment_ids, name)

fun TF.segmentSum(data: Tensor, segment_ids: Tensor, name: String = "SegmentSum") =
    binaryOp("SegmentSum", data, segment_ids, name)

fun TF.sigmoid(x: Tensor, name: String = "Sigmoid") =
    unaryOp("Sigmoid", x, name)

fun TF.sign(x: Tensor, name: String = "Sign") =
    unaryOp("Sign", x, name)

fun TF.sin(x: Tensor, name: String = "Sin") = unaryOp("Sin", x, name)
fun TF.sinh(x: Tensor, name: String = "Sinh") = unaryOp("Sinh", x, name)

fun TF.sparseMatMul(a: Tensor, b: Tensor,
                    transpose_a: Boolean = false,
                    transpose_b: Boolean = false,
                    a_is_sparse: Boolean = false,
                    b_is_sparse: Boolean = false,
                    name: String = "SparseMatMul") =
    naryOp("SparseMatMul", a, b, name = name) {
      setAttr("transpose_a", transpose_a)
      setAttr("transpose_b", transpose_b)
      setAttr("a_is_sparse", a_is_sparse)
      setAttr("b_is_sparse", b_is_sparse)
    }

fun TF.sparseSegmentMean(data: Tensor, indices: Tensor, segment_ids: Tensor, name: String = "SparseSegmentMean") =
    ternaryOp("SparseSegmentMean", data, indices, segment_ids, name)

fun TF.sparseSegmentMeanGrad(grad: Tensor, indices: Tensor, segment_ids: Tensor, output_dim0: Tensor, name: String = "SparseSegmentMeanGrad") =
    naryOp("SparseSegmentMeanGrad", grad, indices, segment_ids, output_dim0, name = name)

fun TF.sparseSegmentMeanWithNumSegments(data: Tensor, indices: Tensor, segment_ids: Tensor, num_segments: Tensor, name: String = "SparseSegmentMeanWithNumSegments") =
    naryOp("SparseSegmentMeanWithNumSegments", data, indices, segment_ids, num_segments, name = name)

fun TF.sparseSegmentSqrtN(data: Tensor, indices: Tensor, segment_ids: Tensor, name: String = "SparseSegmentSqrtN") =
    ternaryOp("SparseSegmentSqrtN", data, indices, segment_ids, name)

fun TF.sparseSegmentSqrtNGrad(grad: Tensor, indices: Tensor, segment_ids: Tensor, output_dim0: Tensor, name: String = "SparseSegmentSqrtNGrad") =
    naryOp("SparseSegmentSqrtNGrad", grad, indices, segment_ids, output_dim0, name = name)

fun TF.sparseSegmentSqrtNWithNumSegments(data: Tensor, indices: Tensor, segment_ids: Tensor, num_segments: Tensor, name: String = "SparseSegmentSqrtNWithNumSegments") =
    naryOp("SparseSegmentSqrtNWithNumSegments", data, indices, segment_ids, num_segments, name = name)

fun TF.sparseSegmentSum(data: Tensor, indices: Tensor, segment_ids: Tensor, name: String = "SparseSegmentSum") =
    ternaryOp("SparseSegmentSum", data, indices, segment_ids, name)

fun TF.sparseSegmentSumWithNumSegments(data: Tensor, indices: Tensor, segment_ids: Tensor, num_segments: Tensor, name: String = "SparseSegmentSumWithNumSegments") =
    naryOp("SparseSegmentSumWithNumSegments", data, indices, segment_ids, num_segments, name = name)

fun TF.sqrt(x: Tensor, name: String = "Sqrt") = unaryOp("Sqrt", x, name)

fun TF.square(a: Tensor, name: String = "Square") =
    unaryOp("Square", a, name)

fun TF.squaredDifference(a: Tensor, b: Tensor, name: String = "SquaredDifference") =
    binaryOp("SquaredDifference", a, b, name)

operator fun Tensor.minus(b: Tensor) = tf.sub(this, b)
fun TF.sub(a: Tensor, b: Tensor, name: String = "Sub") =
    binaryOp("Sub", a, b, name)

fun TF.sum(input: Tensor, axis: Tensor, keep_dims: Boolean = false, name: String = "Sum") =
    naryOp("Sum", input, axis, name = name) {
      setAttr("keep_dims", keep_dims)
    }

fun TF.tan(x: Tensor, name: String = "Tan") = unaryOp("Tan", x, name)
fun TF.tanh(x: Tensor, name: String = "Tanh") = unaryOp("Tanh", x, name)

fun TF.truncateDiv(a: Tensor, b: Tensor, name: String = "TruncateDiv") =
    binaryOp("TruncateDiv", a, b, name)

fun TF.truncateMod(a: Tensor, b: Tensor, name: String = "TruncateMod") =
    binaryOp("TruncateMod", a, b, name)

fun TF.unsortedSegmentMax(data: Tensor, segment_ids: Tensor, num_segments: Tensor, name: String = "UnsortedSegmentMax") =
    ternaryOp("UnsortedSegmentMax", data, segment_ids, num_segments, name)

fun TF.unsortedSegmentMin(data: Tensor, segment_ids: Tensor, num_segments: Tensor, name: String = "UnsortedSegmentMin") =
    ternaryOp("UnsortedSegmentMin", data, segment_ids, num_segments, name)

fun TF.unsortedSegmentProd(data: Tensor, segment_ids: Tensor, num_segments: Tensor, name: String = "UnsortedSegmentProd") =
    ternaryOp("UnsortedSegmentProd", data, segment_ids, num_segments, name)

fun TF.unsortedSegmentSum(data: Tensor, segment_ids: Tensor, num_segments: Tensor, name: String = "UnsortedSegmentSum") =
    ternaryOp("UnsortedSegmentSum", data, segment_ids, num_segments, name)

fun TF.zeta(a: Tensor, b: Tensor, name: String = "Zeta") =
    binaryOp("Zeta", a, b, name)

fun TF.tensordot(input: Tensor, kernel: Tensor, const: Tensor): Tensor {
  TODO()
}