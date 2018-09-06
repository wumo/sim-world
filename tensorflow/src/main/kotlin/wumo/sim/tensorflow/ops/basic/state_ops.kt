package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.OutputLike
import wumo.sim.tensorflow.ops.gen.gen_state_ops
import wumo.sim.tensorflow.ops.variables.DynamicInitializer
import wumo.sim.tensorflow.ops.variables.Initializer
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape
import wumo.sim.util.SwitchType3
import wumo.sim.util.scalarDimension

object state_ops {
  interface API {
    fun assign(_ref: Output, value: Output, validateShape: Boolean = true, useLocking: Boolean = true, name: String = "Assign"): Output {
      return gen_state_ops.assign(_ref, value, validateShape, useLocking, name)
    }
    
    fun assignAdd(_ref: Output, value: Output, useLocking: Boolean = false, name: String = "AssignAdd"): Output {
      return gen_state_ops.assignAdd(_ref, value, useLocking, name)
    }
    
    fun assignSub(_ref: Output, value: Output, useLocking: Boolean = false, name: String = "AssignSub"): Output {
      return gen_state_ops.assignSub(_ref, value, useLocking, name)
    }
    
    fun countUpTo(_ref: Output, limit: Long, name: String = "CountUpTo"): Output {
      return gen_state_ops.countUpTo(_ref, limit, name)
    }
    
    fun destroyTemporaryVariable(_ref: Output, varName: String, name: String = "DestroyTemporaryVariable"): Output {
      return gen_state_ops.destroyTemporaryVariable(_ref, varName, name)
    }
    
    fun isVariableInitialized(_ref: Output, name: String = "IsVariableInitialized"): Output {
      return gen_state_ops.isVariableInitialized(_ref, name)
    }
    
    fun resourceCountUpTo(resource: Output, limit: Long, t: DataType<*>, name: String = "ResourceCountUpTo"): Output {
      return gen_state_ops.resourceCountUpTo(resource, limit, t, name)
    }
    
    fun resourceScatterNdAdd(_ref: Output, indices: Output, updates: Output, useLocking: Boolean = true, name: String = "ResourceScatterNdAdd"): Op {
      return gen_state_ops.resourceScatterNdAdd(_ref, indices, updates, useLocking, name)
    }
    
    fun resourceScatterNdUpdate(_ref: Output, indices: Output, updates: Output, useLocking: Boolean = true, name: String = "ResourceScatterNdUpdate"): Op {
      return gen_state_ops.resourceScatterNdUpdate(_ref, indices, updates, useLocking, name)
    }
    
    fun scatterAdd(_ref: Output, indices: Output, updates: Output, useLocking: Boolean = false, name: String = "ScatterAdd"): Output {
      return gen_state_ops.scatterAdd(_ref, indices, updates, useLocking, name)
    }
    
    fun scatterDiv(_ref: Output, indices: Output, updates: Output, useLocking: Boolean = false, name: String = "ScatterDiv"): Output {
      return gen_state_ops.scatterDiv(_ref, indices, updates, useLocking, name)
    }
    
    fun scatterMax(_ref: Output, indices: Output, updates: Output, useLocking: Boolean = false, name: String = "ScatterMax"): Output {
      return gen_state_ops.scatterMax(_ref, indices, updates, useLocking, name)
    }
    
    fun scatterMin(_ref: Output, indices: Output, updates: Output, useLocking: Boolean = false, name: String = "ScatterMin"): Output {
      return gen_state_ops.scatterMin(_ref, indices, updates, useLocking, name)
    }
    
    fun scatterMul(_ref: Output, indices: Output, updates: Output, useLocking: Boolean = false, name: String = "ScatterMul"): Output {
      return gen_state_ops.scatterMul(_ref, indices, updates, useLocking, name)
    }
    
    fun scatterNdAdd(_ref: Output, indices: Output, updates: Output, useLocking: Boolean = false, name: String = "ScatterNdAdd"): Output {
      return gen_state_ops.scatterNdAdd(_ref, indices, updates, useLocking, name)
    }
    
    fun scatterNdSub(_ref: Output, indices: Output, updates: Output, useLocking: Boolean = false, name: String = "ScatterNdSub"): Output {
      return gen_state_ops.scatterNdSub(_ref, indices, updates, useLocking, name)
    }
    
    fun scatterNdUpdate(_ref: Output, indices: Output, updates: Output, useLocking: Boolean = true, name: String = "ScatterNdUpdate"): Output {
      return gen_state_ops.scatterNdUpdate(_ref, indices, updates, useLocking, name)
    }
    
    fun scatterSub(_ref: Output, indices: Output, updates: Output, useLocking: Boolean = false, name: String = "ScatterSub"): Output {
      return gen_state_ops.scatterSub(_ref, indices, updates, useLocking, name)
    }
    
    fun scatterUpdate(_ref: Output, indices: Output, updates: Output, useLocking: Boolean = true, name: String = "ScatterUpdate"): Output {
      return gen_state_ops.scatterUpdate(_ref, indices, updates, useLocking, name)
    }
    
    fun temporaryVariable(shape: Shape, dtype: DataType<*>, varName: String = "", name: String = "TemporaryVariable"): Output {
      return gen_state_ops.temporaryVariable(shape, dtype, varName, name)
    }
    
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
  
}