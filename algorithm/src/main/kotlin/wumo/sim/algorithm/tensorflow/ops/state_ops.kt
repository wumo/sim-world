@file:Suppress("UNCHECKED_CAST")

package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.*
import wumo.sim.util.*

private val variable_switch = SwitchType3<String, Boolean, Variable>().apply {
  case<Float> { tf.variable(_1, _2, _3) }
  case<Double> { tf.variable(_1, _2, _3) }
  case<Boolean> { tf.variable(_1, _2, _3) }
  case<Byte> { tf.variable(_1, _2, _3) }
  case<Int> { tf.variable(_1, _2, _3) }
  case<Long> { tf.variable(_1, _2, _3) }
  case<String> { tf.variable(_1, _2, _3) }
  case<FloatArray> { tf.variable(_1, _2, _3) }
  case<DoubleArray> { tf.variable(_1, _2, _3) }
  case<BooleanArray> { tf.variable(_1, _2, _3) }
  case<ByteArray> { tf.variable(_1, _2, _3) }
  case<IntArray> { tf.variable(_1, _2, _3) }
  case<LongArray> { tf.variable(_1, _2, _3) }
  case<Array<*>> {
    if (_1::class.java.componentType == String::class.java)
      tf.variable(_1 as Array<String>, _2, _3)
    else
      throw IllegalArgumentException("unsupported ${_1::class}")
  }
}

fun TF.variable(initial_value: Any, name: String = "Variable", trainable: Boolean = true) =
    variable_switch(initial_value, name, trainable)

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

private inline fun TF.variable(initializer: (String) -> Tensor, name: String, trainable: Boolean = true): Variable {
  subscope(name) {
    val initial_value = initializer("initial_value")
    val v = g.nodeBuilder("VariableV2", parentName)
        .setAttrType("dtype", initial_value.dtype.base_dtype)
        .setAttr("shape", initial_value.shape)
        .build()
    
    val t = Variable(v, 0)
    t.initial_value = initial_value
    t.initializer_op = assign(t, t.try_guard_against_uninitialized_dependencies(initial_value))
    //TODO: Change this class to not take caching_device, b
    //ut to take the op to colocate the snapshot with, so we can use
    //colocation rather than devices.
    colocate_with(t.op) {
      t.snapshot = identity(t, name = "read")
    }
    if (trainable) trainables += t
    global_variables += t
    return t
  }
}

fun TF.variable(shape: Dimension, initializer: Initializer, name: String, trainable: Boolean = true, validate_shape: Boolean = true) =
    variable({ initializer(shape, name = "initial_value") }, name, trainable)

fun TF.variable(shape: Dimension, dtype: Int, initializer: Initializer, name: String, trainable: Boolean = true, validate_shape: Boolean = true) =
    variable({ initializer(shape, dtype, "initial_value") }, name, trainable)

fun TF.variable(initial_value: Tensor, name: String = "Variable", trainable: Boolean = true) =
    variable({ initial_value }, name, trainable)

fun TF.assign(ref: Tensor, value: Tensor, name: String = "Assign"): Tensor {
  val op = g.nodeBuilder("Assign", ctx.getUniqueFullName(name))
      .addInput(ref.asRef())
      .addInput(value)
      .build()
  return Tensor(op, 0)
}

fun TF.is_variable_initialized(ref: Tensor, name: String = "IsVariableInitialized"): Tensor {
  if (ref.dtype.is_ref_dytpe) {
    val v = g.nodeBuilder("IsVariableInitialized", ctx.getUniqueFullName(name))
        .addInput(ref)
        .build()
    return Tensor(v, 0)
  }
  TODO("handle resource")
}
