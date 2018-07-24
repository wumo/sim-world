package wumo.sim.algorithm.tensorflow.layers

import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.ops.*
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.Dimension
import wumo.sim.util.i
import wumo.sim.util.dim
import wumo.sim.util.x

class Dense(val units: Int,
            val activation: TensorFunction? = null,
            val use_bias: Boolean = true,
            val kernel_initializer: Initializer,
            val bias_initializer: Initializer? = tf.zeros_initializer(),
            val kernel_regularizer: TensorFunction? = null,
            val bias_regularizer: TensorFunction? = null,
            activity_regularizer: Any? = null,
            val kernel_constraint: Any? = null,
            val bias_constraint: Any? = null,
            trainable: Boolean = true,
            dtype: Int = 0,
            name: String = "") : Layer(trainable = trainable,
                                       activity_reqularizer = activity_regularizer,
                                       dtype = dtype) {
  lateinit var input_spec: Any
  lateinit var kernel: Tensor
  var bias: Tensor? = null
  
  override fun build(input_shape: Dimension) {
    if (input_shape[-1] == -1)
      throw IllegalArgumentException("The last dimension of the inputs to `Dense`" +
                                     "should be defined. Found `None`.")
    kernel = add_variable(name = "kernel",
                          shape = input_shape[-1] x units, dtype = dtype,
                          initializer = kernel_initializer,
                          regularizer = kernel_regularizer,
                          trainable = true)
    
    if (use_bias)
      bias = add_variable(name = "bias",
                          shape = dim(units), dtype = dtype,
                          initializer = bias_initializer!!,
                          regularizer = bias_regularizer,
                          trainable = true)
    built = true
  }
  
  override fun call(input: Tensor): Tensor {
    val shape = input.shape()
    var output = if (shape.rank() > 2) {
      tf.tensordot(input, kernel, tf.const(2 x 1, i(shape.rank() - 1, 0)))
    } else
      tf.matmul(input, kernel)
    if (use_bias)
      output = tf.biasAdd(output, bias!!)
    if (activation != null)
      output = activation!!(output)
    return output
  }
  
}