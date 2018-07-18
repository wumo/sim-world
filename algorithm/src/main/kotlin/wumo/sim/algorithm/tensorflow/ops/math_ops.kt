package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.binaryOp
import wumo.sim.algorithm.tensorflow.unaryOp
import wumo.sim.util.Dimension
import wumo.sim.util.arange
import wumo.sim.util.i

fun TF.abs(x: Tensor, name: String = "Abs") = unaryOp("Abs", x, name)
fun TF.accumulateNV2(vararg input: Tensor, shape: Dimension, name: String = "AccumulateNV2"): Tensor {
  val op = g.nodeBuilder("AddN", ctx.getUniqueFullName(name))
      .addInputList(input as Array<Tensor>)
      .setAttr("shape", shape)
      .build()
  return Tensor(op, 0)
}

fun TF.acos(x: Tensor, name: String = "Acos") = unaryOp("Acos", x, name)
fun TF.acosh(x: Tensor, name: String = "Acosh") = unaryOp("Acosh", x, name)
fun TF.asin(x: Tensor, name: String = "Asin") = unaryOp("Asin", x, name)
fun TF.asinh(x: Tensor, name: String = "Asinh") = unaryOp("Asinh", x, name)
fun TF.atan(x: Tensor, name: String = "Atan") = unaryOp("Atan", x, name)
fun TF.atan2(x: Tensor, y: Tensor, name: String = "Atan2") = binaryOp("Atan2", x, y, name)
fun TF.atanh(x: Tensor, name: String = "Atanh") = unaryOp("Atanh", x, name)

fun TF.ceil(x: Tensor, name: String = "Ceil") = unaryOp("Ceil", x, name)
fun TF.cos(x: Tensor, name: String = "Cos") = unaryOp("Cos", x, name)
fun TF.cosh(x: Tensor, name: String = "Cosh") = unaryOp("Cosh", x, name)

fun TF.exp(x: Tensor, name: String = "Exp") = unaryOp("Exp", x, name)
fun TF.floor(x: Tensor, name: String = "Floor") = unaryOp("Floor", x, name)

fun TF.pow(a: Tensor, b: Tensor, name: String = "Pow") =
    binaryOp("Pow", a, b, name)

fun TF.sin(x: Tensor, name: String = "Sin") = unaryOp("Sin", x, name)
fun TF.sinh(x: Tensor, name: String = "Sinh") = unaryOp("Sinh", x, name)

fun TF.sqrt(x: Tensor, name: String = "Sqrt") = unaryOp("Sqrt", x, name)

fun TF.tan(x: Tensor, name: String = "Tan") = unaryOp("Tan", x, name)
fun TF.tanh(x: Tensor, name: String = "Tanh") = unaryOp("Tanh", x, name)

operator fun Tensor.plus(b: Tensor) = tf.add(this, b)
fun TF.add(a: Tensor, b: Tensor, name: String = "Add") =
    binaryOp("Add", a, b, name)

fun TF.addV2(a: Tensor, b: Tensor, name: String = "AddV2") =
    binaryOp("AddV2", a, b, name)

operator fun Tensor.minus(b: Tensor) = tf.sub(this, b)
fun TF.sub(a: Tensor, b: Tensor, name: String = "Sub") =
    binaryOp("Sub", a, b, name)

operator fun Tensor.times(b: Tensor) = tf.mul(this, b)
fun TF.mul(a: Tensor, b: Tensor, name: String = "Mul") =
    binaryOp("Mul", a, b, name)

operator fun Tensor.div(b: Tensor) = tf.div(this, b)
fun TF.div(a: Tensor, b: Tensor, name: String = "Div") =
    binaryOp("Div", a, b, name)

fun TF.less(a: Tensor, b: Tensor, name: String = "Less") =
    binaryOp("Less", a, b, name)

fun TF.lessEqual(a: Tensor, b: Tensor, name: String = "LessEqual") =
    binaryOp("LessEqual", a, b, name)

fun TF.greater(a: Tensor, b: Tensor, name: String = "Greater") =
    binaryOp("Greater", a, b, name)

fun TF.greaterEqual(a: Tensor, b: Tensor, name: String = "GreaterEqual") =
    binaryOp("GreaterEqual", a, b, name)

fun TF.sum(input: Tensor, axis: Tensor, name: String = "Sum") =
    binaryOp("Sum", input, axis, name)

fun TF.max(input: Tensor, axis: Tensor, keep_dims: Boolean = false, name: String = "Max"): Tensor {
  val op = g.nodeBuilder("Max", ctx.getUniqueFullName(name))
      .addInput(input)
      .addInput(axis)
      .setAttr("keep_dims", keep_dims)
      .build()
  return Tensor(op, 0)
}

fun TF.min(input: Tensor, axis: Tensor, keep_dims: Boolean = false, name: String = "Min"): Tensor {
  val op = g.nodeBuilder("Min", ctx.getUniqueFullName(name))
      .addInput(input)
      .addInput(axis)
      .setAttr("keep_dims", keep_dims)
      .build()
  return Tensor(op, 0)
}

fun TF.reductionDims(x: Tensor, axis: Tensor?): Tensor {
  if (axis != null) return axis
  //Fast path: avoid creating Rank and Range ops if ndims is known.
  return const(arange(x.shape.rank()))
  //TODO SparseTensor
  // Otherwise, we rely on Range and Rank to do the right thing at run-time.
  return range(const(0), rank(x))
}

fun TF.mean(input: Tensor, axis: Tensor? = null, keep_dims: Boolean = false, name: String = "Mean"): Tensor {
  val op = g.nodeBuilder("Mean", ctx.getUniqueFullName(name))
      .addInput(input)
      .addInput(reductionDims(input, axis))
      .setAttr("keep_dims", keep_dims)
      .build()
  return Tensor(op, 0)
}

fun TF.matmul(a: Tensor, b: Tensor, name: String = "MatMul") =
    binaryOp("MatMul", a, b, name)

fun TF.argmax(a: Tensor, axis: Tensor, name: String = "ArgMax") =
    subscope(name) {
      val op = g.nodeBuilder("ArgMax", parentName)
          .addInput(a)
          .addInput(axis)
          .setAttrType("output_type", DT_INT32)
          .build()
      Tensor(op, 0)
    }

fun TF.argmax(a: Tensor, axis: Int, output_type: Int = DT_INT32, name: String = "ArgMax") =
    subscope(name) {
      val op = g.nodeBuilder("ArgMax", parentName)
          .addInput(a)
          .addInput(const(axis, "dimension"))
          .setAttrType("output_type", output_type)
          .build()
      Tensor(op, 0)
    }

fun TF.argmin(a: Tensor, axis: Tensor, name: String = "ArgMin") =
    binaryOp("ArgMin", a, axis, name)

fun TF.argmin(a: Tensor, axis: Int, name: String = "ArgMin") =
    subscope(name) {
      val op = g.nodeBuilder("ArgMin", parentName)
          .addInput(a)
          .addInput(const(axis, "dimension"))
          .build()
      Tensor(op, 0)
    }

fun TF.sigmoid(x: Tensor, name: String = "Sigmoid") =
    unaryOp("Sigmoid", x, name)

fun TF.square(a: Tensor, name: String = "Square") =
    unaryOp("Square", a, name)

fun TF.log(a: Tensor, name: String = "Log") =
    unaryOp("Log", a, name)

operator fun Tensor.unaryMinus() = tf.neg(this)
fun TF.neg(a: Tensor, name: String = "Neg") =
    unaryOp("Neg", a, name)

fun TF.addN(vararg a: Tensor, name: String = "AddN") =
    subscope(name) {
      val op = g.nodeBuilder("AddN", parentName)
          .addInputList(a as Array<Tensor>)
          .build()
      Tensor(op, 0)
    }

fun TF.cast(x: Tensor, dstT: Int, name: String = "Cast"): Tensor {
  return if (x.dtype == dstT) x
  else {
    val op = g.nodeBuilder("Cast", ctx.getUniqueFullName(name))
        .addInput(x)
        .setAttrType("DstT", dstT)
        .build()
    Tensor(op, 0)
  }
}

fun TF.tensordot(input: Tensor, kernel: Tensor, const: Tensor): Tensor {
  TODO()
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
  val op = g.nodeBuilder("Range", ctx.getUniqueFullName(name))
      .addInput(start)
      .addInput(limit)
      .addInput(delta)
      .build()
  return Tensor(op, 0)
}

fun TF.range(start: Tensor, limit: Tensor, delta: Tensor, dtype: Int, name: String = "Range"): Tensor {
  val op = g.nodeBuilder("Range", ctx.getUniqueFullName(name))
      .addInput(start)
      .addInput(limit)
      .addInput(delta)
      .build()
  return Tensor(op, 0)
}