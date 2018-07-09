package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.binaryOp
import wumo.sim.algorithm.tensorflow.unaryOp

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

fun TF.sum(a: Tensor, b: Tensor, name: String = "Sum") =
    binaryOp("Sum", a, b, name)

fun TF.matmul(a: Tensor, b: Tensor, name: String = "MatMul") =
    binaryOp("MatMul", a, b, name)

fun TF.argmax(a: Tensor, b: Tensor, name: String = "ArgMax") =
    binaryOp("ArgMax", a, b, name)

fun TF.argmax(a: Tensor, dim: Int, name: String = "ArgMax") =
    subscope(name) {
      val op = g.nodeBuilder("ArgMax", ctx.name)
          .addInput(a)
          .addInput(const(dim, "dimension"))
          .build()
      Tensor(op, 0, a.dtype)
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
      val op = g.nodeBuilder("AddN", ctx.name)
          .addInputList(a as Array<Tensor>)
          .build()
      Tensor(op, 0, a[0].dtype)
    }

fun TF.cast(x: Tensor, dstT: Int, name: String = "Cast"): Tensor {
  val op = g.nodeBuilder("Cast", ctx.getUniqueFullName(name))
      .addInput(x)
      .setAttrType("DstT", dstT)
      .build()
  return Tensor(op, 0, dstT)
}

fun TF.tensordot(input: Tensor, kernel: Tensor, const: Tensor): Tensor {
  TODO()
}