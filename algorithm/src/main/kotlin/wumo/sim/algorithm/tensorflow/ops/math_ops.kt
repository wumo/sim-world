package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.binaryOp
import wumo.sim.algorithm.tensorflow.unaryOp
import wumo.sim.algorithm.util.helpers.arange
import wumo.sim.algorithm.util.helpers.i

operator fun Tensor.plus(b: Tensor) = tf.add(this, b)
fun TF.add(a: Tensor, b: Tensor, name: String = "Add") =
    binaryOp("Add", a, b, name)

operator fun Tensor.minus(b: Tensor) = tf.sub(this, b)
fun TF.sub(a: Tensor, b: Tensor, name: String = "Sub") =
    binaryOp("Sub", a, b, name)

operator fun Tensor.times(b: Tensor) = tf.mul(this, b)
fun TF.mul(a: Tensor, b: Tensor, name: String = "Mul") =
    binaryOp("Mul", a, b, name)

operator fun Tensor.div(b: Tensor) = tf.div(this, b)
fun TF.div(a: Tensor, b: Tensor, name: String = "Div") =
    binaryOp("Div", a, b, name)

fun TF.sum(input: Tensor, reduction_indices: Tensor, name: String = "Sum") =
    binaryOp("Sum", input, reduction_indices, name)


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

fun TF.argmax(a: Tensor, b: Tensor, name: String = "ArgMax") =
    binaryOp("ArgMax", a, b, name)

fun TF.argmax(a: Tensor, dim: Int, name: String = "ArgMax") =
    subscope(name) {
      val op = g.nodeBuilder("ArgMax", parentName)
          .addInput(a)
          .addInput(const(dim, "dimension"))
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