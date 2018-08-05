package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.*
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.Variable
import wumo.sim.algorithm.tensorflow.ops.gen.argMax
import wumo.sim.algorithm.tensorflow.ops.gen.argMin
import wumo.sim.algorithm.tensorflow.ops.gen.realDiv
import wumo.sim.algorithm.tensorflow.ops.gen.sub
import wumo.sim.util.arange
import wumo.sim.util.i

fun TF.argmax(a: Tensor, axis: Int = 0, output_type: Int = DT_INT64, name: String = "ArgMax") =
    name_scope(name) {
      val dimension = tf.const(axis, "dimension")
      tf.argMax(a, dimension, output_type, tf.ctxNs.scopeName)
    }

fun TF.argmin(a: Tensor, axis: Int = 0, output_type: Int = DT_INT64, name: String = "ArgMin") =
    tf.name_scope(name) {
      val dimension = tf.const(axis, "dimension")
      tf.argMin(a, dimension, output_type, tf.ctxNs.scopeName)
    }

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

fun TF.cast(x: Tensor, dstT: Int, name: String = "Cast"): Tensor {
  val x = (x as? Variable)?.value() ?: x
  return if (x.dtype == dstT) x
  else _cast(x, dstT, name)
}

fun TF._cast(x: Tensor, dstT: Int, name: String = "Cast") =
    naryOp("Cast", x.value(), name = name) {
      attrType("DstT", dstT)
    }

fun TF.conj(x: Tensor, name: String = "Conj") =
    if (x.dtype == DT_COMPLEX64 || x.dtype == DT_COMPLEX128)
      _conj(x, name)
    else
      x

fun TF._conj(x: Tensor, name: String = "Conj") = unaryOp("Conj", x.value(), name)

operator fun Tensor.div(b: Any) =
    tf.name_scope("div") {
      val y = tf.const(this.dtype.base_dtype, b, name = "y")
      tf.div(this, y, name = tf.ctxNs.scopeName)
    }

operator fun Tensor.div(b: Tensor) = tf.div(this, b)
fun TF.div(a: Tensor, b: Tensor, name: String = "Div") =
    binaryOp("Div", a.value(), b.value(), name)

fun TF.greaterEqual(a: Tensor, b: Any, name: String = "GreaterEqual") =
    name_scope(name) {
      val y = tf.const(a.dtype.base_dtype, b, name = "y")
      binaryOp("GreaterEqual", a.value(), y, ctxNs.scopeName)
    }

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

operator fun Tensor.minus(b: Any) =
    tf.name_scope("add") {
      val y = tf.const(this.dtype.base_dtype, b, name = "y")
      tf.sub(this, y, name = tf.ctxNs.scopeName)
    }

operator fun Tensor.minus(b: Tensor) = tf.sub(this, b)

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

fun TF.realDiv(a: Tensor, b: Any, name: String = "RealDiv"): Tensor =
    name_scope("truediv") {
      val y = const(a.dtype.base_dtype, b, name = "y")
      realDiv(a, y, name = ctxNs.scopeName)
    }

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

fun TF.tensordot(input: Tensor, kernel: Tensor, const: Tensor): Tensor {
  TODO()
}