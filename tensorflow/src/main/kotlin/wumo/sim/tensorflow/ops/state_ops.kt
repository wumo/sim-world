@file:Suppress("UNCHECKED_CAST")

package wumo.sim.tensorflow.ops

import wumo.sim.tensorflow.ops.gen.gen_state_ops
import wumo.sim.tensorflow.ops.variables.DynamicInitializer
import wumo.sim.tensorflow.ops.variables.Initializer
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.tf
import wumo.sim.util.Shape
import wumo.sim.util.SwitchType3
import wumo.sim.util.scalarDimension

//
//import org.bytedeco.javacpp.tensorflow
//import wumo.sim.tensorflow.*
//import wumo.sim.tensorflow.ops.variables.Initializer
//import wumo.sim.tensorflow.ops.variables.Variable
//import wumo.sim.tensorflow.scope.NameScope
//import wumo.sim.util.Shape
//import wumo.sim.util.SwitchType3
//import wumo.sim.util.scalarDimension
//
object state_ops {
  
  interface API : gen_state_ops {

//    fun assign(ref: Output, value: Output, name: String = "Assign") =
//    //TODO NOTE(mrry): We add an explicit colocation constraint between
//    //the newly created op and any of its reference-typed inputs.
//        tf.colocateWith(ref) {
//          tf._assign(ref, value, name = name)
//        }
    
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
    fun variable(initial_value: OutputLike, name: String = "Variable", trainable: Boolean = true) = _variable({ initial_value.toOutput() }, name, trainable)
    fun variable(initial_value: Variable, name: String = "Variable", trainable: Boolean = true) = _variable({ initial_value.initializedValue.toOutput() }, name, trainable)
    fun variable(initial_value: Any, name: String = "Variable", trainable: Boolean = true) =
        variable_switch(initial_value, name, trainable)
    
    fun variable(shape: Shape, initializer: Initializer, name: String = "Variable", trainable: Boolean = true) =
        _variable({ initializer(shape) }, name, trainable)
    
  }
}

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

private fun _variable(initializer: (String) -> Output, name: String, trainable: Boolean = true): Variable =
    tf.variable(name, initializer = DynamicInitializer(initializer("${tf.currentNameScope}$name/initial_value")), trainable = trainable)
