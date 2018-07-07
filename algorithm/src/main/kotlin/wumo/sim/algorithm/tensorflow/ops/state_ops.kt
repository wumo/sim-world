package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.Operation
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.dim
import wumo.sim.algorithm.util.scalarDimension


fun TF.variable(initial_value: Float, name: String = "Variable", trainable: Boolean = true) = variable(scalarDimension, initial_value, name, trainable)
fun TF.variable(initial_value: Double, name: String = "Variable", trainable: Boolean = true) = variable(scalarDimension, initial_value, name, trainable)
fun TF.variable(initial_value: Boolean, name: String = "Variable", trainable: Boolean = true) = variable(scalarDimension, initial_value, name, trainable)
fun TF.variable(initial_value: Byte, name: String = "Variable", trainable: Boolean = true) = variable(scalarDimension, initial_value, name, trainable)
fun TF.variable(initial_value: Short, name: String = "Variable", trainable: Boolean = true) = variable(scalarDimension, initial_value, name, trainable)
fun TF.variable(initial_value: Int, name: String = "Variable", trainable: Boolean = true) = variable(scalarDimension, initial_value, name, trainable)
fun TF.variable(initial_value: Long, name: String = "Variable", trainable: Boolean = true) = variable(scalarDimension, initial_value, name, trainable)
fun TF.variable(initial_value: String, name: String = "Variable", trainable: Boolean = true) = variable(scalarDimension, initial_value, name, trainable)

fun TF.variable(initial_value: FloatArray, name: String = "Variable", trainable: Boolean = true) = variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.variable(initial_value: DoubleArray, name: String = "Variable", trainable: Boolean = true) = variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.variable(initial_value: BooleanArray, name: String = "Variable", trainable: Boolean = true) = variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.variable(initial_value: ByteArray, name: String = "Variable", trainable: Boolean = true) = variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.variable(initial_value: ShortArray, name: String = "Variable", trainable: Boolean = true) = variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.variable(initial_value: IntArray, name: String = "Variable", trainable: Boolean = true) = variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.variable(initial_value: LongArray, name: String = "Variable", trainable: Boolean = true) = variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.variable(initial_value: Array<String>, name: String = "Variable", trainable: Boolean = true) = variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)

fun TF.variable(shape: Dimension, initial_value: FloatArray, name: String = "Variable", trainable: Boolean = true) = variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: DoubleArray, name: String = "Variable", trainable: Boolean = true) = variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: BooleanArray, name: String = "Variable", trainable: Boolean = true) = variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: ByteArray, name: String = "Variable", trainable: Boolean = true) = variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: ShortArray, name: String = "Variable", trainable: Boolean = true) = variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: IntArray, name: String = "Variable", trainable: Boolean = true) = variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: LongArray, name: String = "Variable", trainable: Boolean = true) = variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: Array<String>, name: String = "Variable", trainable: Boolean = true) = variable({ const(shape, initial_value, it) }, name, trainable)

fun TF.variable(shape: Dimension, initial_value: Float, name: String = "Variable", trainable: Boolean = true) = variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: Double, name: String = "Variable", trainable: Boolean = true) = variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: Boolean, name: String = "Variable", trainable: Boolean = true) = variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: Byte, name: String = "Variable", trainable: Boolean = true) = variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: Short, name: String = "Variable", trainable: Boolean = true) = variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: Int, name: String = "Variable", trainable: Boolean = true) = variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: Long, name: String = "Variable", trainable: Boolean = true) = variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: String, name: String = "Variable", trainable: Boolean = true) = variable({ const(shape, initial_value, it) }, name, trainable)

private inline fun TF.variable(initializer: (String) -> Tensor, name: String, trainable: Boolean): Tensor {
  subscope(name) {
    val initial_value = initializer("initial_value")
    val v = g.nodeBuilder("VariableV2", ctx.name)
        .setAttrType("dtype", initial_value.dtype)
        .setAttr("shape", initial_value.shape)
        .build()
    val t = Tensor(v, 0, initial_value.dtype)
    if (trainable) trainables += t
    init_ops += assign(t, initial_value)
    return t
  }
}

fun TF.variable(initial_value: Tensor, name: String = ""): Tensor {
  subscope(name) {
    val v = g.nodeBuilder("VariableV2", ctx.name)
        .setAttrType("dtype", initial_value.dtype)
        .setAttr("shape", initial_value.shape)
        .build()
    val t = Tensor(v, 0, initial_value.dtype)
    init_ops += assign(t, initial_value)
    return t
  }
}

fun TF.assign(ref: Tensor, value: Tensor, name: String = "Assign"): Operation {
  return g.nodeBuilder("Assign", ctx.getUniqueFullName(name))
      .addInput(ref)
      .addInput(value)
      .build()
}