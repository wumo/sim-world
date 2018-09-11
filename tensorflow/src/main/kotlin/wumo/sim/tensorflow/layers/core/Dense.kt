package wumo.sim.tensorflow.layers.core

import wumo.sim.tensorflow.core.TensorFunction
import wumo.sim.tensorflow.layers.Layer
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_math_ops
import wumo.sim.tensorflow.ops.variables.Initializer
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.ops.variables.VariableScope
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape
import wumo.sim.util.i

class Dense(val units: Int,
            val activation: TensorFunction? = null,
            val use_bias: Boolean = true,
            val kernel_initializer: Initializer = tf.glorotUniformInitializer(),
            val bias_initializer: Initializer? = tf.zerosInitializer(),
            val kernel_regularizer: TensorFunction? = null,
            val bias_regularizer: TensorFunction? = null,
            activity_regularizer: Any? = null,
            val kernel_constraint: Any? = null,
            val bias_constraint: Any? = null,
            trainable: Boolean = true,
            dataType: DataType<*>,
            name: String,
            _scope: VariableScope) : Layer(trainable = trainable,
                                  name = name,
                                  dataType = dataType,
                                  activity_reqularizer = activity_regularizer,
                                  _scope = _scope) {
  
  lateinit var input_spec: Any
  lateinit var kernel: Variable
  var bias: Variable? = null
  override fun build(input_shape: Shape) {
    
    if (input_shape[-1] == -1)
      throw IllegalArgumentException("The last dimension of the inputs to `Dense`" +
                                         "should be defined. Found `None`.")
    kernel = addWeight(name = "weights",
                       shape = Shape(input_shape[-1], units),
                       initializer = kernel_initializer,
                       regularizer = kernel_regularizer,
                       constraint = kernel_constraint,
                       dataType = dataType,
                       trainable = true)
    
    if (use_bias)
      bias = addWeight(name = "biases",
                       shape = Shape(units),
                       initializer = bias_initializer!!,
                       regularizer = bias_regularizer,
                       constraint = bias_constraint,
                       dataType = dataType,
                       trainable = true)
    built = true
  }
  
  override fun call(input: Output): Output {
    val shape = input.shape
    var output = if (shape.rank > 2) {
      val outputs = tf.tensordot(input, kernel.toOutput(),
                                 tf.const(Shape(2, 1), i(shape.rank - 1, 0)))
      outputs.setShape(shape.slice(0, -1) + units)
      outputs
    } else
      gen_math_ops.matMul(input, kernel.toOutput())
    if (use_bias)
      output = tf.biasAdd(output, bias!!.toOutput())
    activation?.let {
      output = it(output)!!
    }
    return output
  }
}