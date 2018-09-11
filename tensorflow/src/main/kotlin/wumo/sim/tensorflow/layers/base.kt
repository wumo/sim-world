package wumo.sim.tensorflow.layers

import wumo.sim.tensorflow.core.TensorFunction
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.variables.Initializer
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.ops.variables.VariableScope
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape

open class Layer(var name: String,
                 val trainable: Boolean = true,
                 var dataType: DataType<*>? = null,
                 val activity_reqularizer: Any? = null,
                 val _scope: VariableScope? = null) {
  
  val use_resource_variables = false
  var statefule = false
  var built = false
  val losses = mutableListOf<Output>()
  
  open fun build(input_shape: Shape) {
    built = true
  }
  
  open fun call(input: Output) = input
  
  //
  open operator fun invoke(inputs: Output): Output {
    val block = {
      val graph = inputs.graph
    
      if (!built) {
        dataType = dataType ?: inputs.dataType.baseDataType
        val input_shape = inputs.shape
        build(input_shape)
        built = true
      }
      //if not in_deferred_mode:
      val outputs = call(inputs)
      if (activity_reqularizer != null) {
        TODO()
      }
      outputs
    }
    return if (_scope != null)
      tf.updatedScope(_scope, isPure = true, block = block)
    else
      tf.variableScope(name, isDefaultName = true, block = block)
  }
  
  fun addWeight(name: String,
                shape: Shape,
                dataType: DataType<*>? = null,
                initializer: Initializer,
                regularizer: TensorFunction? = null,
                trainable: Boolean = true,
                constraint: Any? = null): Variable {
    val variable = tf.variable(shape, dataType, initializer, regularizer, trainable && this.trainable, name = name)
    if (regularizer != null)
      tf.colocateWith(variable.op) {
        val regularization = tf.nameScope(name + "/Regularizer") {
          regularizer(variable.toOutput())
        }
        if (regularization != null)
          losses += regularization
      }
    return variable
  }
  
//
//  protected fun add_variable(shape: Shape, dataType: DataType<*>? = null,
//                             initializer: Initializer,
//                             regularizer: TensorFunction? = null,
//                             trainable: Boolean = true,
//                             name: String = ""): Output {
//    val v = tf.variable(name, shape, dataType, initializer, trainable = trainable && this.trainable)
//    if (regularizer != null)
//      losses += regularizer(v)
//    return v
//  }
}