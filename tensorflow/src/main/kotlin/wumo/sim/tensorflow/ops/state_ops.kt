@file:Suppress("UNCHECKED_CAST")

package wumo.sim.tensorflow.ops

import org.bytedeco.javacpp.tensorflow
import wumo.sim.tensorflow.*
import wumo.sim.tensorflow.ops.variables.Initializer
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.scope.NameScope
import wumo.sim.util.Shape
import wumo.sim.util.SwitchType3
import wumo.sim.util.scalarDimension

interface state_ops {
  fun assign(ref: Output, value: Output, name: String = "Assign") =
  //TODO NOTE(mrry): We add an explicit colocation constraint between
  //the newly created op and any of its reference-typed inputs.
      ops.colocate_with(ref) {
        tf._assign(ref, value, name = name)
      }
}

private val variable_switch = SwitchType3<String, Boolean, Variable>().apply {
  case<Float> { variable(_1, _2, _3) }
  case<Double> { variable(_1, _2, _3) }
  case<Boolean> { variable(_1, _2, _3) }
  case<Byte> { variable(_1, _2, _3) }
  case<Int> { variable(_1, _2, _3) }
  case<Long> { variable(_1, _2, _3) }
  case<String> { variable(_1, _2, _3) }
  case<FloatArray> { variable(_1, _2, _3) }
  case<DoubleArray> { variable(_1, _2, _3) }
  case<BooleanArray> { variable(_1, _2, _3) }
  case<ByteArray> { variable(_1, _2, _3) }
  case<IntArray> { variable(_1, _2, _3) }
  case<LongArray> { variable(_1, _2, _3) }
  case<Array<*>> {
    if (_1::class.java.componentType == String::class.java)
      variable(_1 as Array<String>, _2, _3)
    else
      throw IllegalArgumentException("unsupported ${_1::class}")
  }
}

fun variable(initial_value: Any, name: String = "Variable", trainable: Boolean = true) =
    variable_switch(initial_value, name, trainable)

fun variable(initial_value: Float, name: String = "Variable", trainable: Boolean = true) = variable(scalarDimension, initial_value, name, trainable)
fun variable(initial_value: Double, name: String = "Variable", trainable: Boolean = true) = variable(scalarDimension, initial_value, name, trainable)
fun variable(initial_value: Boolean, name: String = "Variable", trainable: Boolean = true) = variable(scalarDimension, initial_value, name, trainable)
fun variable(initial_value: Byte, name: String = "Variable", trainable: Boolean = true) = variable(scalarDimension, initial_value, name, trainable)
fun variable(initial_value: Short, name: String = "Variable", trainable: Boolean = true) = variable(scalarDimension, initial_value, name, trainable)
fun variable(initial_value: Int, name: String = "Variable", trainable: Boolean = true) = variable(scalarDimension, initial_value, name, trainable)
fun variable(initial_value: Long, name: String = "Variable", trainable: Boolean = true) = variable(scalarDimension, initial_value, name, trainable)
fun variable(initial_value: String, name: String = "Variable", trainable: Boolean = true) = variable(scalarDimension, initial_value, name, trainable)
fun variable(initial_value: FloatArray, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(Shape(initial_value.size), initial_value, it) }, name, trainable)
fun variable(initial_value: DoubleArray, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(Shape(initial_value.size), initial_value, it) }, name, trainable)
fun variable(initial_value: BooleanArray, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(Shape(initial_value.size), initial_value, it) }, name, trainable)
fun variable(initial_value: ByteArray, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(Shape(initial_value.size), initial_value, it) }, name, trainable)
fun variable(initial_value: ShortArray, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(Shape(initial_value.size), initial_value, it) }, name, trainable)
fun variable(initial_value: IntArray, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(Shape(initial_value.size), initial_value, it) }, name, trainable)
fun variable(initial_value: LongArray, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(Shape(initial_value.size), initial_value, it) }, name, trainable)
fun variable(initial_value: Array<String>, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(Shape(initial_value.size), initial_value, it) }, name, trainable)
fun variable(shape: Shape, initial_value: FloatArray, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun variable(shape: Shape, initial_value: DoubleArray, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun variable(shape: Shape, initial_value: BooleanArray, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun variable(shape: Shape, initial_value: ByteArray, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun variable(shape: Shape, initial_value: ShortArray, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun variable(shape: Shape, initial_value: IntArray, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun variable(shape: Shape, initial_value: LongArray, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun variable(shape: Shape, initial_value: Array<String>, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun variable(shape: Shape, initial_value: Float, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun variable(shape: Shape, initial_value: Double, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun variable(shape: Shape, initial_value: Boolean, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun variable(shape: Shape, initial_value: Byte, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun variable(shape: Shape, initial_value: Short, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun variable(shape: Shape, initial_value: Int, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun variable(shape: Shape, initial_value: Long, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun variable(shape: Shape, initial_value: String, name: String = "Variable", trainable: Boolean = true) = _variable({ tf.const(shape, initial_value, it) }, name, trainable)
private fun _variable(initializer: (String) -> Output, name: String, trainable: Boolean = true): Variable {
  ops.name_scope(name) {
    ops.init_scope {
      //Use attr_scope and device(None) to simulate the behavior of
      //colocate_with when the _variable we want to colocate with doesn't
      //yet exist.
      val attr = tensorflow.AttrValue()
      attr.mutable_list().apply {
        add_s("loc:@${ctxNs.scopeName.fullName}")
      }
      val t = attr_scope("_class" to attr) {
        val initial_value = initializer("Initializer")
        val v = _variableV2(initial_value.shape, initial_value.dtype.base_dtype, name = ctxNs.scopeName)
        Variable(v.op!!, 0).apply {
          this.initial_value = initial_value
        }
      }
      t.initializer_op = assign(t, t.try_guard_against_uninitialized_dependencies(t.initial_value))
      //TODO: Change this class to not take caching_device, b
      //ut to take the op to colocate the snapshot with, so we can use
      //colocation rather than devices.
      colocate_with(t.op!!) {
        t.snapshot = _identity(t, name = "read")
      }
      if (trainable) trainables += t
      global_variables += t
      return t
    }
  }
}

fun variable(shape: Shape, initializer: Initializer, name: String, trainable: Boolean = true, validate_shape: Boolean = true) =
    _variable({ initializer(shape, name = "initial_value") }, name, trainable)

fun variable(shape: Shape, dtype: Int, initializer: Initializer, name: String, trainable: Boolean = true, validate_shape: Boolean = true) =
    _variable({ initializer(shape, dtype.base_dtype, "initial_value") }, name, trainable)

fun variable(initial_value: Output, name: String = "Variable", trainable: Boolean = true) =
    _variable({ initial_value }, name, trainable)

fun assign(ref: Output, value: Output, name: String = "Assign") =
//TODO NOTE(mrry): We add an explicit colocation constraint between
//the newly created op and any of its reference-typed inputs.
    ops.colocate_with(ref) {
      tf._assign(ref, value, name = name)
    }

fun is_variable_initialized(ref: Output, name: String = "IsVariableInitialized"): Output {
  if (ref.dtype.is_ref_dytpe) {
    tf._isVariableInitialized(ref, name)
  }
  TODO("handle resource")
}

fun get_variable(initial_value: Float, name: String = "Variable", trainable: Boolean = true) = get_variable(scalarDimension, initial_value, name, trainable)
fun get_variable(initial_value: Double, name: String = "Variable", trainable: Boolean = true) = get_variable(scalarDimension, initial_value, name, trainable)
fun get_variable(initial_value: Boolean, name: String = "Variable", trainable: Boolean = true) = get_variable(scalarDimension, initial_value, name, trainable)
fun get_variable(initial_value: Byte, name: String = "Variable", trainable: Boolean = true) = get_variable(scalarDimension, initial_value, name, trainable)
fun get_variable(initial_value: Short, name: String = "Variable", trainable: Boolean = true) = get_variable(scalarDimension, initial_value, name, trainable)
fun get_variable(initial_value: Int, name: String = "Variable", trainable: Boolean = true) = get_variable(scalarDimension, initial_value, name, trainable)
fun get_variable(initial_value: Long, name: String = "Variable", trainable: Boolean = true) = get_variable(scalarDimension, initial_value, name, trainable)
fun get_variable(initial_value: String, name: String = "Variable", trainable: Boolean = true) = get_variable(scalarDimension, initial_value, name, trainable)
fun get_variable(initial_value: FloatArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(Shape(initial_value.size), initial_value, it) }, name, trainable)
fun get_variable(initial_value: DoubleArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(Shape(initial_value.size), initial_value, it) }, name, trainable)
fun get_variable(initial_value: BooleanArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(Shape(initial_value.size), initial_value, it) }, name, trainable)
fun get_variable(initial_value: ByteArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(Shape(initial_value.size), initial_value, it) }, name, trainable)
fun get_variable(initial_value: ShortArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(Shape(initial_value.size), initial_value, it) }, name, trainable)
fun get_variable(initial_value: IntArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(Shape(initial_value.size), initial_value, it) }, name, trainable)
fun get_variable(initial_value: LongArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(Shape(initial_value.size), initial_value, it) }, name, trainable)
fun get_variable(initial_value: Array<String>, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(Shape(initial_value.size), initial_value, it) }, name, trainable)
fun get_variable(shape: Shape, initial_value: FloatArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun get_variable(shape: Shape, initial_value: DoubleArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun get_variable(shape: Shape, initial_value: BooleanArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun get_variable(shape: Shape, initial_value: ByteArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun get_variable(shape: Shape, initial_value: ShortArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun get_variable(shape: Shape, initial_value: IntArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun get_variable(shape: Shape, initial_value: LongArray, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun get_variable(shape: Shape, initial_value: Array<String>, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun get_variable(shape: Shape, initial_value: Float, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun get_variable(shape: Shape, initial_value: Double, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun get_variable(shape: Shape, initial_value: Boolean, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun get_variable(shape: Shape, initial_value: Byte, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun get_variable(shape: Shape, initial_value: Short, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun get_variable(shape: Shape, initial_value: Int, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun get_variable(shape: Shape, initial_value: Long, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun get_variable(shape: Shape, initial_value: String, name: String = "Variable", trainable: Boolean = true) = get_variable({ tf.const(shape, initial_value, it) }, name, trainable)
fun get_variable(name: String): Variable {
  assert(ctxVs.reuse) { "require variable_scope resut=true, but actual=${ctxVs.reuse}" }
  return ctxVs.variables[name]!!
}

fun get_variable(shape: Shape, initializer: Initializer, name: String, trainable: Boolean = true, validate_shape: Boolean = true) =
    get_variable({ initializer(shape, name = "Initializer") }, name, trainable)

fun get_variable(shape: Shape, dtype: Int, initializer: Initializer, name: String, trainable: Boolean = true, validate_shape: Boolean = true) =
    get_variable({ initializer(shape, dtype.base_dtype, "Initializer") }, name, trainable)

fun get_variable(initial_value: Output, name: String = "Variable", trainable: Boolean = true) =
    get_variable({ initial_value }, name, trainable)

/**
 * [get_variable]的命名只会使用[ctxVs]绑定的[NameScope]
 */
private fun get_variable(initializer: (String) -> Output, name: String, trainable: Boolean = true): Variable {
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
