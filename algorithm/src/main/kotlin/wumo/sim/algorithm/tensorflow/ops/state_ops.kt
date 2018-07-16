package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.*
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.dim
import wumo.sim.algorithm.util.scalarDimension

fun TF.variable(initial_value: Any, name: String = "Variable", trainable: Boolean = true) =
    when (initial_value) {
      is Float -> variable(initial_value, name, trainable)
      is Double -> variable(initial_value, name, trainable)
      is Boolean -> variable(initial_value, name, trainable)
      is Byte -> variable(initial_value, name, trainable)
      is Int -> variable(initial_value, name, trainable)
      is Long -> variable(initial_value, name, trainable)
      is String -> variable(initial_value, name, trainable)
      is FloatArray -> variable(initial_value, name, trainable)
      is DoubleArray -> variable(initial_value, name, trainable)
      is BooleanArray -> variable(initial_value, name, trainable)
      is ByteArray -> variable(initial_value, name, trainable)
      is IntArray -> variable(initial_value, name, trainable)
      is LongArray -> variable(initial_value, name, trainable)
      is Array<*> -> {
        if (initial_value::class.java.componentType == String::class.java)
          variable(initial_value as Array<String>, name, trainable)
        else
          throw IllegalArgumentException("unsupported ${initial_value::class}")
      }
      else -> throw IllegalArgumentException("unsupported ${initial_value::class}")
    }

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
  init_subsope(name) {
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

fun TF.variable(shape: Dimension, dtype: Int, initializer: Initializer, name: String, trainable: Boolean = true, validate_shape: Boolean = true) =
    variable({ initializer(shape, dtype, "initial_value") }, name, trainable)

fun TF.variable(initial_value: Tensor, name: String = "Variable", trainable: Boolean = true) =
    variable({ initial_value }, name, trainable)

fun TF.assign(ref: Tensor, value: Tensor, name: String = "Assign"): Operation {
  return g.nodeBuilder("Assign", ctx.getUniqueFullName(name))
      .addInput(ref.asRef())
      .addInput(value)
      .build()
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
