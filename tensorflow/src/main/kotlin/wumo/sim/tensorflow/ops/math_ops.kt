package wumo.sim.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.tensorflow.TF
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.base_dtype
import wumo.sim.tensorflow.ops.gen.*
import wumo.sim.tensorflow.tf
import wumo.sim.util.arange
import wumo.sim.util.i
import wumo.sim.tensorflow.ops.gen.cast as _cast
import wumo.sim.tensorflow.ops.gen.conj as _conj
import wumo.sim.tensorflow.ops.gen.greaterEqual as _greaterEqual
import wumo.sim.tensorflow.ops.gen.range as _range
import wumo.sim.tensorflow.ops.gen.realDiv as _realDiv
import wumo.sim.tensorflow.ops.gen.sum as _sum

fun TF.argmax(a: Output, axis: Int = 0, output_type: Int = DT_INT64, name: String = "ArgMax") =
    name_scope(name) {
      val dimension = const(axis, "dimension")
      argMax(a, dimension, output_type, tf.ctxNs.scopeName)
    }

fun TF.argmin(a: Output, axis: Int = 0, output_type: Int = DT_INT64, name: String = "ArgMin") =
    name_scope(name) {
      val dimension = const(axis, "dimension")
      argMin(a, dimension, output_type, tf.ctxNs.scopeName)
    }

operator fun Output.plus(b: Any) =
    tf.name_scope("add") {
      val y = tf.const(this.dtype.base_dtype, b, name = "y")
      tf.add(this, y, name = tf.ctxNs.scopeName)
    }

operator fun Output.plus(b: Output) = tf.add(this, b)

fun TF.cast(x: Output, dstT: Int, name: String = "Cast") = run {
  val x = (x as? Variable)?.value ?: x
  if (x.dtype == dstT) x
  else _cast(x, dstT, name)
}

//fun TF._cast(x: Output, dstT: Int, name: String = "Cast") =
//    naryOp("Cast", x.value(), name = name) {
//      attrType("DstT", dstT)
//    }

fun TF.conj(x: Output, name: String = "Conj") =
    if (x.dtype == DT_COMPLEX64 || x.dtype == DT_COMPLEX128)
      _conj(x, name)
    else
      x

operator fun Output.div(b: Any) =
    tf.name_scope("div") {
      val y = tf.const(this.dtype.base_dtype, b, name = "y")
      tf.div(this, y, name = tf.ctxNs.scopeName)
    }

operator fun Output.div(b: Output) = tf.div(this, b)

fun TF.greaterEqual(a: Output, b: Any, name: String = "GreaterEqual") =
    name_scope(name) {
      val y = tf.const(a.dtype.base_dtype, b, name = "y")
      _greaterEqual(a, y, ctxNs.scopeName)
    }

fun TF.reductionDims(x: Output, axis: Output?): Output {
  if (axis != null) return axis
  //Fast path: avoid creating Rank and Range ops if ndims is known.
  return const(arange(x.shape.rank), name = "reduction_indices")
  //TODO SparseOutput
  // Otherwise, we rely on Range and Rank to do the right thing at run-time.
  return range(const(0), rank(x))
}

fun TF.mean(input: Output, axis: LongArray? = null, keep_dims: Boolean = false, name: String = "mean") =
    name_scope(name) {
      val reduction_indices =
          reductionDims(input,
                        if (axis != null) const(axis, "reduction_indices")
                        else null)
      mean(input, reduction_indices, keep_dims, name)
    }

operator fun Output.minus(b: Any) =
    tf.name_scope("add") {
      val y = tf.const(this.dtype.base_dtype, b, name = "y")
      tf.sub(this, y, name = tf.ctxNs.scopeName)
    }

operator fun Output.minus(b: Output) = tf.sub(this, b)

operator fun Output.times(b: Any) =
    tf.name_scope("mul") {
      val y = tf.const(this.dtype.base_dtype, b, name = "y")
      tf.mul(this, y, name = tf.ctxNs.scopeName)
    }

operator fun Output.times(b: Output) = tf.mul(this, b)

operator fun Output.unaryMinus() = tf.neg(this)

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
 * @param start: A 0-D `Output` (scalar). Acts as first entry in the range if
 * `limit` is not None; otherwise, acts as range limit and first entry
 * defaults to 0.
 * @param limit: A 0-D `Output` (scalar). Upper limit of sequence,
 * exclusive. If None, defaults to the value of `start` while the first
 * entry of the range defaults to 0.
 * @param delta: A 0-D `Output` (scalar). Number that increments
 * `start`. Defaults to 1.
 * @param name: A name for the operation. Defaults to "range".
 * @return An 1-D `Output` of type `dtype`.
 */
fun TF.range(start: Output, limit: Output, delta: Output = const(1), name: String = "Range") = run {
  val dtypes = i(start.dtype, limit.dtype, delta.dtype)
  val inferred_dtype = dtypes.maxBy { dtype_hierarchy[it]!! }!!
  val start = cast(start, inferred_dtype)
  val limit = cast(limit, inferred_dtype)
  val delta = cast(delta, inferred_dtype)
  _range(start, limit, delta, name)
}

fun TF.realDiv(a: Output, b: Any, name: String = "RealDiv") =
    name_scope("truediv") {
      val y = const(a.dtype.base_dtype, b, name = "y")
      _realDiv(a, y, name = ctxNs.scopeName)
    }

fun TF.sum(input: Output, axis: Int? = null, keep_dims: Boolean = false, name: String = "sum") =
    name_scope(name) {
      val reduction_indices =
          reductionDims(input,
                        if (axis != null) const(axis, "reduction_indices")
                        else null)
      _sum(input, reduction_indices, keep_dims, ctxNs.scopeName)
    }

fun TF.sum(input: Output, axis: Output? = null, keep_dims: Boolean = false, name: String = "sum") =
    _sum(input, reductionDims(input, axis), keep_dims, name)

fun TF.tensordot(input: Output, kernel: Output, const: Output): Output {
  TODO()
}