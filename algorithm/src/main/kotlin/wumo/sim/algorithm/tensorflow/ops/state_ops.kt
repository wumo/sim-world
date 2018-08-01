@file:Suppress("UNCHECKED_CAST")

package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow
import wumo.sim.algorithm.tensorflow.*
import wumo.sim.algorithm.tensorflow.scope.NameScope
import wumo.sim.util.Dimension
import wumo.sim.util.SwitchType3
import wumo.sim.util.dim
import wumo.sim.util.scalarDimension

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
fun TF.variable(initial_value: FloatArray, name: String = "Variable", trainable: Boolean = true) = _variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.variable(initial_value: DoubleArray, name: String = "Variable", trainable: Boolean = true) = _variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.variable(initial_value: BooleanArray, name: String = "Variable", trainable: Boolean = true) = _variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.variable(initial_value: ByteArray, name: String = "Variable", trainable: Boolean = true) = _variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.variable(initial_value: ShortArray, name: String = "Variable", trainable: Boolean = true) = _variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.variable(initial_value: IntArray, name: String = "Variable", trainable: Boolean = true) = _variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.variable(initial_value: LongArray, name: String = "Variable", trainable: Boolean = true) = _variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.variable(initial_value: Array<String>, name: String = "Variable", trainable: Boolean = true) = _variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: FloatArray, name: String = "Variable", trainable: Boolean = true) = _variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: DoubleArray, name: String = "Variable", trainable: Boolean = true) = _variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: BooleanArray, name: String = "Variable", trainable: Boolean = true) = _variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: ByteArray, name: String = "Variable", trainable: Boolean = true) = _variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: ShortArray, name: String = "Variable", trainable: Boolean = true) = _variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: IntArray, name: String = "Variable", trainable: Boolean = true) = _variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: LongArray, name: String = "Variable", trainable: Boolean = true) = _variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: Array<String>, name: String = "Variable", trainable: Boolean = true) = _variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: Float, name: String = "Variable", trainable: Boolean = true) = _variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: Double, name: String = "Variable", trainable: Boolean = true) = _variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: Boolean, name: String = "Variable", trainable: Boolean = true) = _variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: Byte, name: String = "Variable", trainable: Boolean = true) = _variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: Short, name: String = "Variable", trainable: Boolean = true) = _variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: Int, name: String = "Variable", trainable: Boolean = true) = _variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: Long, name: String = "Variable", trainable: Boolean = true) = _variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.variable(shape: Dimension, initial_value: String, name: String = "Variable", trainable: Boolean = true) = _variable({ const(shape, initial_value, it) }, name, trainable)
private fun TF._variable(initializer: (String) -> Tensor, name: String, trainable: Boolean = true): Variable {
  name_scope(name) {
    init_scope {
      //Use attr_scope and device(None) to simulate the behavior of
      //colocate_with when the _variable we want to colocate with doesn't
      //yet exist.
      val attr = tensorflow.AttrValue()
      attr.mutable_list().apply {
        add_s("loc:@${ctxNs.scopeName.fullName}")
      }
      val t = attr_scope("_class" to attr) {
        val initial_value = initializer("Initializer")
        val v = naryOp("VariableV2", name = ctxNs.scopeName) {
          attrType("dtype", initial_value.dtype.base_dtype)
          attr("shape", initial_value.shape)
        }
        Variable(v.op!!, 0).apply {
          this.initial_value = initial_value
        }
      }
      t.initializer_op = assign(t, t.try_guard_against_uninitialized_dependencies(t.initial_value))
      //TODO: Change this class to not take caching_device, b
      //ut to take the op to colocate the snapshot with, so we can use
      //colocation rather than devices.
      colocate_with(t.op!!) {
        t.snapshot = identity(t, name = "read")
      }
      if (trainable) trainables += t
      global_variables += t
      return t
    }
  }
}

fun TF.variable(shape: Dimension, initializer: Initializer, name: String, trainable: Boolean = true, validate_shape: Boolean = true) =
    _variable({ initializer(shape, name = "initial_value") }, name, trainable)

fun TF.variable(shape: Dimension, dtype: Int, initializer: Initializer, name: String, trainable: Boolean = true, validate_shape: Boolean = true) =
    _variable({ initializer(shape, dtype.base_dtype, "initial_value") }, name, trainable)

fun TF.variable(initial_value: Tensor, name: String = "Variable", trainable: Boolean = true) =
    _variable({ initial_value }, name, trainable)

fun TF.assign(ref: Tensor, value: Tensor, name: String = "Assign") =
//TODO NOTE(mrry): We add an explicit colocation constraint between
//the newly created op and any of its reference-typed inputs.
    colocate_with(ref) {
      naryOp("Assign", ref.asRef(), value.value(), name = name)
    }

fun TF.is_variable_initialized(ref: Tensor, name: String = "IsVariableInitialized"): Tensor {
  if (ref.dtype.is_ref_dytpe) {
    naryOp("IsVariableInitialized", ref, name = name)
  }
  TODO("handle resource")
}

fun TF.get_variable(initial_value: Float, name: String = "Variable", trainable: Boolean = true) = get_variable(scalarDimension, initial_value, name, trainable)
fun TF.get_variable(initial_value: Double, name: String = "Variable", trainable: Boolean = true) = get_variable(scalarDimension, initial_value, name, trainable)
fun TF.get_variable(initial_value: Boolean, name: String = "Variable", trainable: Boolean = true) = get_variable(scalarDimension, initial_value, name, trainable)
fun TF.get_variable(initial_value: Byte, name: String = "Variable", trainable: Boolean = true) = get_variable(scalarDimension, initial_value, name, trainable)
fun TF.get_variable(initial_value: Short, name: String = "Variable", trainable: Boolean = true) = get_variable(scalarDimension, initial_value, name, trainable)
fun TF.get_variable(initial_value: Int, name: String = "Variable", trainable: Boolean = true) = get_variable(scalarDimension, initial_value, name, trainable)
fun TF.get_variable(initial_value: Long, name: String = "Variable", trainable: Boolean = true) = get_variable(scalarDimension, initial_value, name, trainable)
fun TF.get_variable(initial_value: String, name: String = "Variable", trainable: Boolean = true) = get_variable(scalarDimension, initial_value, name, trainable)
fun TF.get_variable(initial_value: FloatArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.get_variable(initial_value: DoubleArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.get_variable(initial_value: BooleanArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.get_variable(initial_value: ByteArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.get_variable(initial_value: ShortArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.get_variable(initial_value: IntArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.get_variable(initial_value: LongArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.get_variable(initial_value: Array<String>, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(dim(initial_value.size), initial_value, it) }, name, trainable)
fun TF.get_variable(shape: Dimension, initial_value: FloatArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.get_variable(shape: Dimension, initial_value: DoubleArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.get_variable(shape: Dimension, initial_value: BooleanArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.get_variable(shape: Dimension, initial_value: ByteArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.get_variable(shape: Dimension, initial_value: ShortArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.get_variable(shape: Dimension, initial_value: IntArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.get_variable(shape: Dimension, initial_value: LongArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.get_variable(shape: Dimension, initial_value: Array<String>, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.get_variable(shape: Dimension, initial_value: Float, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.get_variable(shape: Dimension, initial_value: Double, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.get_variable(shape: Dimension, initial_value: Boolean, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.get_variable(shape: Dimension, initial_value: Byte, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.get_variable(shape: Dimension, initial_value: Short, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.get_variable(shape: Dimension, initial_value: Int, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.get_variable(shape: Dimension, initial_value: Long, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.get_variable(shape: Dimension, initial_value: String, name: String = "Variable", trainable: Boolean = true) = get_variable({ const(shape, initial_value, it) }, name, trainable)
fun TF.get_variable(name: String): Variable {
  assert(ctxVs.reuse) { "require variable_scope resut=true, but actual=${ctxVs.reuse}" }
  return ctxVs.variables[name]!!
}

fun TF.get_variable(shape: Dimension, initializer: Initializer, name: String, trainable: Boolean = true, validate_shape: Boolean = true) =
    get_variable({ initializer(shape, name = "Initializer") }, name, trainable)

fun TF.get_variable(shape: Dimension, dtype: Int, initializer: Initializer, name: String, trainable: Boolean = true, validate_shape: Boolean = true) =
    get_variable({ initializer(shape, dtype.base_dtype, "Initializer") }, name, trainable)

fun TF.get_variable(initial_value: Tensor, name: String = "Variable", trainable: Boolean = true) =
    get_variable({ initial_value }, name, trainable)

/**
 * [get_variable]的命名只会使用[TF.ctxVs]绑定的[NameScope]
 */
private fun TF.get_variable(initializer: (String) -> Tensor, name: String, trainable: Boolean = true): Variable {
  return if (ctxVs.reuse)
    ctxVs.variables[name]!!
  else {
    if (name in ctxVs.variables)
      throw IllegalArgumentException("_variable $name already exists, disallowed. " +
                                         "Did you mean to set reuse=true in VariableScope?")
    with(ctxVs.namescope) {
      val t = _variable(initializer, name, trainable)
      ctxVs.variables[name] = t
      t
    }
  }
}
