package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.*
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.Variable
import wumo.sim.algorithm.tensorflow.ops.gen.gen_math_ops
import wumo.sim.util.Dimension
import wumo.sim.util.arange
import wumo.sim.util.i

fun TF.abs(x: Tensor, name: String = "Abs") = gen_math_ops.abs(x.value(), name)
fun TF.accumulateNV2(vararg input: Tensor, shape: Dimension, name: String = "AccumulateNV2") =
    naryOp("AccumulateNV2", name = name) {
      addInput(input.map { it.value() })
      attr("shape", shape)
    }

fun TF.acos(x: Tensor, name: String = "Acos") = unaryOp("Acos", x.value(), name)
fun TF.acosh(x: Tensor, name: String = "Acosh") = unaryOp("Acosh", x.value(), name)
operator fun Tensor.plus(b: Any) =
    tf.name_scope("add") {
      val y = tf.const(this.dtype.base_dtype, b, name = "y")
      tf.add(this, y, name = tf.ctxNs.scopeName)
    }

operator fun Tensor.plus(b: Tensor) = tf.add(this, b)
fun TF.add(a: Tensor, b: Tensor, name: String = "add") =
    binaryOp("Add", a.value(), b.value(), name)

fun TF.addV2(a: Tensor, b: Tensor, name: String = "AddV2") =
    binaryOp("AddV2", a.value(), b.value(), name)

fun TF.addN(vararg a: Tensor, name: String = "AddN") =
    naryOp("AddN", name = name) {
      addInput(a.map { it.value() })
    }

fun TF.all(input: Tensor, axis: Tensor, keep_dims: Boolean = false, name: String = "All") =
    naryOp("All", input.value(), axis.value(), name = name) {
      attr("keep_dims", keep_dims)
    }

fun TF.angle(input: Tensor, Tout: Int = DT_FLOAT, name: String = "Angle") =
    naryOp("Angle", input.value(), name = name) {
      attrType("Tout", Tout)
    }

fun TF.any(input: Tensor, axis: Tensor, keep_dims: Boolean = false, name: String = "Any") =
    naryOp("Any", input.value(), axis.value(), name = name) {
      attr("keep_dims", keep_dims)
    }

fun TF.approximateEqual(x: Tensor, y: Tensor, tolerance: Float = 1e-5f, name: String = "ApproximateEqual") =
    naryOp("ApproximateEqual", x.value(), y.value(), name = name) {
      attr("tolerance", tolerance)
    }

fun TF.argmax(a: Tensor, axis: Int, output_type: Int = DT_INT32, name: String = "ArgMax") =
    name_scope(name) {
      val dimension = const(axis, "dimension")
      argmax(a, dimension, output_type, ctxNs.scopeName)
    }

fun TF.argmax(a: Tensor, axis: Tensor, output_type: Int = DT_INT32, name: String = "ArgMax") =
    naryOp("ArgMax", a.value(), axis.value(), name = name) {
      attrType("output_type", output_type)
    }

fun TF.argmin(a: Tensor, axis: Tensor, name: String = "ArgMin") =
    binaryOp("ArgMin", a.value(), axis.value(), name)

fun TF.argmin(a: Tensor, axis: Int, name: String = "ArgMin") =
    name_scope(name) {
      val dimension = const(axis, "dimension")
      argmin(a, dimension, ctxNs.scopeName)
    }

fun TF.asin(x: Tensor, name: String = "Asin") = unaryOp("Asin", x.value(), name)
fun TF.asinh(x: Tensor, name: String = "Asinh") = unaryOp("Asinh", x.value(), name)
fun TF.atan(x: Tensor, name: String = "Atan") = unaryOp("Atan", x.value(), name)
fun TF.atan2(x: Tensor, y: Tensor, name: String = "Atan2") = binaryOp("Atan2", x.value(), y.value(), name)
fun TF.atanh(x: Tensor, name: String = "Atanh") = unaryOp("Atanh", x.value(), name)
fun TF.batchMatMul(a: Tensor, b: Tensor, adj_x: Boolean = false, adj_y: Boolean = false, name: String = "BatchMatMul") =
    naryOp("BatchMatMul", a.value(), b.value(), name = name) {
      attr("adj_x", adj_x)
      attr("adj_y", adj_y)
    }

fun TF.besselI0e(x: Tensor, name: String = "BesselI0e") = unaryOp("BesselI0e", x.value(), name)
fun TF.besselI1e(x: Tensor, name: String = "BesselI1e") = unaryOp("BesselI1e", x.value(), name)
fun TF.betainc(a: Tensor, b: Tensor, x: Tensor, name: String = "Betainc") =
    ternaryOp("Betainc", a.value(), b.value(), x.value(), name = name)

fun TF.bincount(arr: Tensor, size: Tensor, weights: Tensor, name: String = "Bincount") =
    ternaryOp("Bincount", arr.value(), size.value(), weights.value(), name = name)

fun TF.Bucketize(input: Tensor, boundaries: FloatArray, name: String = "Bucketize") =
    naryOp("Bucketize", input.value(), name = name) {
      attr("boundaries", boundaries)
    }

fun TF.cast(x: Tensor, dstT: Int, name: String = "Cast"): Tensor {
  val x = (x as? Variable)?.value() ?: x
  return if (x.dtype == dstT) x
  else _cast(x, dstT, name)
}

fun TF._cast(x: Tensor, dstT: Int, name: String = "Cast") =
    naryOp("Cast", x.value(), name = name) {
      attrType("DstT", dstT)
    }

fun TF.ceil(x: Tensor, name: String = "Ceil") = unaryOp("Ceil", x.value(), name)
fun TF.clipByValue(t: Tensor, clip_value_min: Tensor, clip_value_max: Tensor, name: String = "ClipByValue") =
    ternaryOp("ClipByValue", t.value(), clip_value_min.value(), clip_value_max.value(), name = name)

fun TF.compareAndBitpack(input: Tensor, threshold: Tensor, name: String = "CompareAndBitpack") =
    binaryOp("CompareAndBitpack", input.value(), threshold.value(), name)

fun TF.complex(input: Tensor, threshold: Tensor, Tout: Int = DT_COMPLEX64, name: String = "Complex") =
    naryOp("Complex", input.value(), threshold.value(), name = name) {
      attr("Tout", Tout)
    }

fun TF.ComplexAbs(x: Tensor, Tout: Int = DT_FLOAT, name: String = "ComplexAbs") =
    naryOp("ComplexAbs", x.value(), name = name) {
      attr("Tout", Tout)
    }

fun TF.conj(x: Tensor, name: String = "Conj") =
    if (x.dtype == DT_COMPLEX64 || x.dtype == DT_COMPLEX128)
      _conj(x, name)
    else
      x

fun TF._conj(x: Tensor, name: String = "Conj") = unaryOp("Conj", x.value(), name)
fun TF.cos(x: Tensor, name: String = "Cos") = unaryOp("Cos", x.value(), name)
fun TF.cosh(x: Tensor, name: String = "Cosh") = unaryOp("Cosh", x.value(), name)
fun TF.cross(a: Tensor, b: Tensor, name: String = "Cross") =
    binaryOp("Cross", a.value(), b.value(), name)

fun TF.cumprod(x: Tensor, axis: Tensor, exclusive: Boolean = false, reverse: Boolean = false, name: String = "Cumprod") =
    naryOp("Cumprod", x.value(), axis.value(), name = name) {
      attr("exclusive", exclusive)
      attr("reverse", reverse)
    }

fun TF.cumsum(x: Tensor, axis: Tensor, exclusive: Boolean = false, reverse: Boolean = false, name: String = "Cumsum") =
    naryOp("Cumsum", x.value(), axis.value(), name = name) {
      attr("exclusive", exclusive)
      attr("reverse", reverse)
    }

fun TF.digamma(x: Tensor, name: String = "Digamma") = unaryOp("Digamma", x.value(), name)

operator fun Tensor.div(b: Any) =
    tf.name_scope("div") {
      val y = tf.const(this.dtype.base_dtype, b, name = "y")
      tf.div(this, y, name = tf.ctxNs.scopeName)
    }

operator fun Tensor.div(b: Tensor) = tf.div(this, b)
fun TF.div(a: Tensor, b: Tensor, name: String = "Div") =
    binaryOp("Div", a.value(), b.value(), name)

fun TF.equal(a: Tensor, b: Tensor, name: String = "Equal") =
    binaryOp("Equal", a.value(), b.value(), name)

fun TF.erf(x: Tensor, name: String = "Erf") = unaryOp("Erf", x.value(), name)
fun TF.erfc(x: Tensor, name: String = "Erfc") = unaryOp("Erfc", x.value(), name)
fun TF.exp(x: Tensor, name: String = "Exp") = unaryOp("Exp", x.value(), name)
fun TF.expm1(x: Tensor, name: String = "Expm1") = unaryOp("Expm1", x.value(), name)
fun TF.floor(x: Tensor, name: String = "Floor") = unaryOp("Floor", x.value(), name)
fun TF.floorDiv(a: Tensor, b: Tensor, name: String = "FloorDiv") =
    binaryOp("FloorDiv", a.value(), b.value(), name)

fun TF.floorMod(a: Tensor, b: Tensor, name: String = "FloorMod") =
    binaryOp("FloorMod", a.value(), b.value(), name)

fun TF.greater(a: Tensor, b: Tensor, name: String = "Greater") =
    binaryOp("Greater", a.value(), b.value(), name)

fun TF.greaterEqual(a: Tensor, b: Any, name: String = "GreaterEqual") =
    name_scope(name) {
      val y = tf.const(a.dtype.base_dtype, b, name = "y")
      binaryOp("GreaterEqual", a.value(), y, ctxNs.scopeName)
    }

fun TF.greaterEqual(a: Tensor, b: Tensor, name: String = "GreaterEqual") =
    binaryOp("GreaterEqual", a.value(), b.value(), name)

fun TF.HistogramFixedWidth(values: Tensor, value_range: Tensor, nbins: Tensor, dtype: Int = DT_INT32, name: String = "HistogramFixedWidth") =
    naryOp("HistogramFixedWidth", values.value(), value_range.value(), nbins.value(), name = name) {
      attrType("dtype", dtype)
    }

fun TF.igamma(a: Tensor, b: Tensor, name: String = "Igamma") =
    binaryOp("Igamma", a.value(), b.value(), name)

fun TF.igammac(a: Tensor, b: Tensor, name: String = "Igammac") =
    binaryOp("Igammac", a.value(), b.value(), name)

fun TF.imag(x: Tensor, Tout: Int = DT_FLOAT, name: String = "Imag") =
    naryOp("Imag", x.value(), name = name) {
      attrType("Tout", Tout)
    }

fun TF.inv(x: Tensor, name: String = "Inv") = unaryOp("Inv", x.value(), name)
fun TF.isFinite(x: Tensor, name: String = "IsFinite") = unaryOp("IsFinite", x.value(), name)
fun TF.isInf(x: Tensor, name: String = "IsInf") = unaryOp("IsInf", x.value(), name)
fun TF.isNan(x: Tensor, name: String = "IsNan") = unaryOp("IsNan", x.value(), name)
fun TF.less(a: Tensor, b: Tensor, name: String = "Less") =
    binaryOp("Less", a.value(), b.value(), name)

fun TF.lessEqual(a: Tensor, b: Tensor, name: String = "LessEqual") =
    binaryOp("LessEqual", a.value(), b.value(), name)

fun TF.lgamma(x: Tensor, name: String = "Lgamma") = unaryOp("Lgamma", x.value(), name)
fun TF.LinSpace(start: Tensor, stop: Tensor, num: Tensor, name: String = "LinSpace") =
    ternaryOp("LinSpace", start.value(), stop.value(), num.value(), name)

fun TF.log(a: Tensor, name: String = "Log") =
    unaryOp("Log", a.value(), name)

fun TF.log1p(a: Tensor, name: String = "Log1p") =
    unaryOp("Log1p", a.value(), name)

fun TF.logicalAnd(a: Tensor, b: Tensor, name: String = "LogicalAnd") =
    binaryOp("LogicalAnd", a.value(), b.value(), name)

fun TF.logicalNot(a: Tensor, name: String = "LogicalNot") =
    unaryOp("LogicalNot", a.value(), name)

fun TF.logicalOr(a: Tensor, b: Tensor, name: String = "LogicalOr") =
    binaryOp("LogicalOr", a.value(), b.value(), name)

fun TF.matmul(a: Tensor, b: Tensor, transpose_a: Boolean = false, transpose_b: Boolean = false, name: String = "MatMul") =
    naryOp("MatMul", a.value(), b.value(), name = name) {
      attr("transpose_a", transpose_a)
      attr("transpose_b", transpose_b)
    }

fun TF.max(input: Tensor, axis: Tensor, keep_dims: Boolean = false, name: String = "Max") =
    naryOp("Max", input.value(), axis.value(), name = name) {
      attr("keep_dims", keep_dims)
    }

fun TF.maximum(x: Tensor, y: Tensor, name: String = "Maximum") =
    binaryOp("Maximum", x.value(), y.value(), name = name)

fun TF.reductionDims(x: Tensor, axis: Tensor?): Tensor {
  if (axis != null) return axis
  //Fast path: avoid creating Rank and Range ops if ndims is known.
  return const(arange(x.shape.rank()), name = "reduction_indices")
  //TODO SparseTensor
  // Otherwise, we rely on Range and Rank to do the right thing at run-time.
  return range(const(0), rank(x))
}

fun TF.mean(input: Tensor, axis: IntArray? = null, keep_dims: Boolean = false, name: String = "mean") =
    name_scope(name) {
      val reduction_indices =
          reductionDims(input,
                        if (axis != null) tf.const(axis, "reduction_indices")
                        else null)
      naryOp("Mean", input.value(),
             reduction_indices,
             name = ctxNs.scopeName) {
        attr("keep_dims", keep_dims)
      }
    }

//fun TF.mean(input: Tensor, axis: Tensor? = null, keep_dims: Boolean = false, name: String = "mean") =
//    naryOp("Mean", input, reductionDims(input, axis), name = name) {
//      attr("keep_dims", keep_dims)
//    }
fun TF.min(input: Tensor, axis: Tensor, keep_dims: Boolean = false, name: String = "Min") =
    naryOp("Min", input.value(), axis.value(), name = name) {
      attr("keep_dims", keep_dims)
    }

fun TF.minimum(x: Tensor, y: Tensor, name: String = "Minimum") =
    binaryOp("Minimum", x.value(), y.value(), name = name)

fun TF.mod(x: Tensor, y: Tensor, name: String = "Mod") =
    binaryOp("Mod", x.value(), y.value(), name = name)

operator fun Tensor.times(b: Any) =
    tf.name_scope("mul") {
      val y = tf.const(this.dtype.base_dtype, b, name = "y")
      tf.mul(this, y, name = tf.ctxNs.scopeName)
    }

operator fun Tensor.times(b: Tensor) = tf.mul(this, b)
fun TF.mul(a: Tensor, b: Tensor, name: String = "mul") =
    binaryOp("Mul", a.value(), b.value(), name)

operator fun Tensor.unaryMinus() = tf.neg(this)
fun TF.neg(a: Tensor, name: String = "Neg") =
    unaryOp("Neg", a.value(), name)

fun TF.notEqual(a: Tensor, b: Tensor, name: String = "NotEqual") =
    binaryOp("NotEqual", a.value(), b.value(), name)

fun TF.polygamma(a: Tensor, b: Tensor, name: String = "Polygamma") =
    binaryOp("Polygamma", a.value(), b.value(), name)

fun TF.pow(a: Tensor, b: Tensor, name: String = "Pow") =
    binaryOp("Pow", a.value(), b.value(), name)

fun TF.prod(input: Tensor, axis: Tensor, keep_dims: Boolean = false, name: String = "Prod") =
    naryOp("Prod", input.value(), axis.value(), name = name) {
      attr("keep_dims", keep_dims)
    }

fun TF.quantizeDownAndShrinkRange(input: Tensor, input_min: Tensor, input_max: Tensor, out_type: Int,
                                  name: String = "QuantizeDownAndShrinkRange") =
    naryOps("QuantizeDownAndShrinkRange", input.value(), input_min.value(), input_max.value(), name = name) {
      attrType("out_type", out_type)
    }

fun TF.quantizedAdd(x: Tensor, y: Tensor, min_x: Tensor, max_x: Tensor, min_y: Tensor, max_y: Tensor, Toutput: Int = DT_QINT32,
                    name: String = "QuantizedAdd") =
    naryOps("QuantizedAdd", x.value(), y.value(), min_x.value(), max_x.value(), min_y.value(), max_y.value(), name = name) {
      attrType("Toutput", Toutput)
    }

fun TF.quantizedMatMul(a: Tensor, b: Tensor, min_a: Tensor, max_a: Tensor, min_b: Tensor, max_b: Tensor,
                       Toutput: Int = DT_QINT32,
                       transpose_a: Boolean = false,
                       transpose_b: Boolean = false,
                       Tactivation: Int = DT_QUINT8,
                       name: String = "QuantizedMatMul") =
    naryOps("QuantizedMatMul", a.value(), b.value(), min_a.value(), max_a.value(), min_b.value(), max_b.value(), name = name) {
      attrType("Toutput", Toutput)
      attr("transpose_a", transpose_a)
      attr("transpose_b", transpose_b)
      attrType("Tactivation", Tactivation)
    }

fun TF.quantizedMul(a: Tensor, b: Tensor, min_a: Tensor, max_a: Tensor, min_b: Tensor, max_b: Tensor,
                    Toutput: Int = DT_QINT32,
                    name: String = "QuantizedMul") =
    naryOps("QuantizedMul", a.value(), b.value(), min_a.value(), max_a.value(), min_b.value(), max_b.value(), name = name) {
      attrType("Toutput", Toutput)
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
  return naryOp("Range", start, limit, delta, name = name)
}

fun TF.range(start: Tensor, limit: Tensor, delta: Tensor, dtype: Int, name: String = "Range") =
    naryOp("Range", start.value(), limit.value(), delta.value(), name = name)

fun TF.real(input: Tensor, Tout: Int = DT_FLOAT, name: String = "Real") =
    naryOp("Real", input.value(), name = name) {
      attrType("Tout", Tout)
    }

fun TF.realDiv(a: Tensor, b: Any, name: String = "RealDiv") =
    name_scope("truediv") {
      val y = const(a.dtype.base_dtype, b, name = "y")
      realDiv(a, y, name = ctxNs.scopeName)
    }

fun TF.realDiv(a: Tensor, b: Tensor, name: String = "RealDiv") =
    binaryOp("RealDiv", a.value(), b.value(), name)

fun TF.reciprocal(x: Tensor, name: String = "Reciprocal") =
    unaryOp("Reciprocal", x.value(), name)

fun TF.requantizationRange(input: Tensor, input_min: Tensor, input_max: Tensor, name: String = "RequantizationRange") =
    naryOps("RequantizationRange", input.value(), input_min.value(), input_max.value(), name = name)

fun TF.requantize(input: Tensor, input_min: Tensor, input_max: Tensor, requested_output_min: Tensor, requested_output_max: Tensor, out_type: Int,
                  name: String = "Requantize") =
    naryOps("Requantize", input.value(), input_min.value(), input_max.value(),
            requested_output_min.value(), requested_output_max.value(), name = name) {
      attrType("out_type", out_type)
    }

fun TF.rint(x: Tensor, name: String = "Rint") = unaryOp("Rint", x.value(), name)
fun TF.round(x: Tensor, name: String = "Round") = unaryOp("Round", x.value(), name)
fun TF.rsqrt(x: Tensor, name: String = "Rsqrt") = unaryOp("Rsqrt", x.value(), name)
fun TF.segmentMax(data: Tensor, segmen_ids: Tensor, name: String = "SegmentMax") =
    binaryOp("SegmentMax", data.value(), segmen_ids.value(), name)

fun TF.segmentMean(data: Tensor, segment_ids: Tensor, name: String = "SegmentMean") =
    binaryOp("SegmentMean", data.value(), segment_ids.value(), name)

fun TF.segmentMin(data: Tensor, segment_ids: Tensor, name: String = "SegmentMin") =
    binaryOp("SegmentMin", data.value(), segment_ids.value(), name)

fun TF.segmentProd(data: Tensor, segment_ids: Tensor, name: String = "SegmentProd") =
    binaryOp("SegmentProd", data.value(), segment_ids.value(), name)

fun TF.segmentSum(data: Tensor, segment_ids: Tensor, name: String = "SegmentSum") =
    binaryOp("SegmentSum", data.value(), segment_ids.value(), name)

fun TF.sigmoid(x: Tensor, name: String = "Sigmoid") =
    unaryOp("Sigmoid", x.value(), name)

fun TF.sign(x: Tensor, name: String = "Sign") =
    unaryOp("Sign", x.value(), name)

fun TF.sin(x: Tensor, name: String = "Sin") = unaryOp("Sin", x.value(), name)
fun TF.sinh(x: Tensor, name: String = "Sinh") = unaryOp("Sinh", x.value(), name)
fun TF.sparseMatMul(a: Tensor, b: Tensor,
                    transpose_a: Boolean = false,
                    transpose_b: Boolean = false,
                    a_is_sparse: Boolean = false,
                    b_is_sparse: Boolean = false,
                    name: String = "SparseMatMul") =
    naryOp("SparseMatMul", a.value(), b.value(), name = name) {
      attr("transpose_a", transpose_a)
      attr("transpose_b", transpose_b)
      attr("a_is_sparse", a_is_sparse)
      attr("b_is_sparse", b_is_sparse)
    }

fun TF.sparseSegmentMean(data: Tensor, indices: Tensor, segment_ids: Tensor, name: String = "SparseSegmentMean") =
    ternaryOp("SparseSegmentMean", data.value(), indices.value(), segment_ids.value(), name)

fun TF.sparseSegmentMeanGrad(grad: Tensor, indices: Tensor, segment_ids: Tensor, output_dim0: Tensor, name: String = "SparseSegmentMeanGrad") =
    naryOp("SparseSegmentMeanGrad", grad.value(), indices.value(), segment_ids.value(), output_dim0.value(), name = name)

fun TF.sparseSegmentMeanWithNumSegments(data: Tensor, indices: Tensor, segment_ids: Tensor, num_segments: Tensor, name: String = "SparseSegmentMeanWithNumSegments") =
    naryOp("SparseSegmentMeanWithNumSegments", data.value(), indices.value(), segment_ids.value(), num_segments.value(), name = name)

fun TF.sparseSegmentSqrtN(data: Tensor, indices: Tensor, segment_ids: Tensor, name: String = "SparseSegmentSqrtN") =
    ternaryOp("SparseSegmentSqrtN", data.value(), indices.value(), segment_ids.value(), name)

fun TF.sparseSegmentSqrtNGrad(grad: Tensor, indices: Tensor, segment_ids: Tensor, output_dim0: Tensor, name: String = "SparseSegmentSqrtNGrad") =
    naryOp("SparseSegmentSqrtNGrad", grad.value(), indices.value(), segment_ids.value(), output_dim0.value(), name = name)

fun TF.sparseSegmentSqrtNWithNumSegments(data: Tensor, indices: Tensor, segment_ids: Tensor, num_segments: Tensor, name: String = "SparseSegmentSqrtNWithNumSegments") =
    naryOp("SparseSegmentSqrtNWithNumSegments", data.value(), indices.value(), segment_ids.value(), num_segments.value(), name = name)

fun TF.sparseSegmentSum(data: Tensor, indices: Tensor, segment_ids: Tensor, name: String = "SparseSegmentSum") =
    ternaryOp("SparseSegmentSum", data.value(), indices.value(), segment_ids.value(), name)

fun TF.sparseSegmentSumWithNumSegments(data: Tensor, indices: Tensor, segment_ids: Tensor, num_segments: Tensor, name: String = "SparseSegmentSumWithNumSegments") =
    naryOp("SparseSegmentSumWithNumSegments", data.value(), indices.value(), segment_ids.value(), num_segments.value(), name = name)

fun TF.sqrt(x: Tensor, name: String = "Sqrt") = unaryOp("Sqrt", x.value(), name)
fun TF.square(a: Tensor, name: String = "Square") =
    unaryOp("Square", a.value(), name)

fun TF.squaredDifference(a: Tensor, b: Tensor, name: String = "SquaredDifference") =
    binaryOp("SquaredDifference", a.value(), b.value(), name)

operator fun Tensor.minus(b: Tensor) = tf.sub(this, b)
fun TF.sub(a: Tensor, b: Tensor, name: String = "sub") =
    binaryOp("Sub", a.value(), b.value(), name)

fun TF.sum(input: Tensor, axis: Int? = null, keep_dims: Boolean = false, name: String = "sum") =
    name_scope(name) {
      val reduction_indices =
          reductionDims(input,
                        if (axis != null) tf.const(axis, "reduction_indices")
                        else null)
      _sum(input, reduction_indices, keep_dims, ctxNs.scopeName)
    }

fun TF.sum(input: Tensor, axis: Tensor? = null, keep_dims: Boolean = false, name: String = "sum") =
    _sum(input, reductionDims(input, axis), keep_dims, name)

fun TF._sum(input: Tensor, axis: Tensor, keep_dims: Boolean = false, name: String = "Sum") =
    naryOp("Sum", input.value(), axis.value(), name = name) {
      attr("keep_dims", keep_dims)
    }

fun TF.tan(x: Tensor, name: String = "Tan") = unaryOp("Tan", x.value(), name)
fun TF.tanh(x: Tensor, name: String = "Tanh") = unaryOp("Tanh", x.value(), name)
fun TF.truncateDiv(a: Tensor, b: Tensor, name: String = "TruncateDiv") =
    binaryOp("TruncateDiv", a.value(), b.value(), name)

fun TF.truncateMod(a: Tensor, b: Tensor, name: String = "TruncateMod") =
    binaryOp("TruncateMod", a.value(), b.value(), name)

fun TF.unsortedSegmentMax(data: Tensor, segment_ids: Tensor, num_segments: Tensor, name: String = "UnsortedSegmentMax") =
    ternaryOp("UnsortedSegmentMax", data.value(), segment_ids.value(), num_segments.value(), name)

fun TF.unsortedSegmentMin(data: Tensor, segment_ids: Tensor, num_segments: Tensor, name: String = "UnsortedSegmentMin") =
    ternaryOp("UnsortedSegmentMin", data.value(), segment_ids.value(), num_segments.value(), name)

fun TF.unsortedSegmentProd(data: Tensor, segment_ids: Tensor, num_segments: Tensor, name: String = "UnsortedSegmentProd") =
    ternaryOp("UnsortedSegmentProd", data.value(), segment_ids.value(), num_segments.value(), name)

fun TF.unsortedSegmentSum(data: Tensor, segment_ids: Tensor, num_segments: Tensor, name: String = "UnsortedSegmentSum") =
    ternaryOp("UnsortedSegmentSum", data.value(), segment_ids.value(), num_segments.value(), name)

fun TF.zeta(a: Tensor, b: Tensor, name: String = "Zeta") =
    binaryOp("Zeta", a.value(), b.value(), name)

fun TF.tensordot(input: Tensor, kernel: Tensor, const: Tensor): Tensor {
  TODO()
}